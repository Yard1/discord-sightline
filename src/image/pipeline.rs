#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::struct_field_names,
    reason = "Image processing does deliberate pixel math and geometry naming; clippy reports if these suppressions stop being needed."
)]

use crate::{
    configuration::app::{DownloadConfig, MatchConfig, PreviewDownloadMode},
    image::types::{
        CandidateKind, ImageAnchor, ImageCandidate, ImageFingerprint, ImageVisualSignature,
        LocalImageHash,
    },
};
use anyhow::{Context, Result, anyhow, bail};
use bytes::BytesMut;
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use image::{
    DynamicImage, GenericImageView, GrayImage, ImageFormat, ImageReader, Limits, RgbImage,
    buffer::ConvertBuffer, codecs::jpeg::JpegEncoder, imageops,
};
use image_hasher::{HashAlg, Hasher, HasherConfig};
use reqwest::{
    Client, StatusCode,
    header::{CONTENT_TYPE, HeaderMap, RETRY_AFTER},
};
use std::{
    cell::RefCell,
    io::Cursor,
    net::IpAddr,
    sync::{Arc, LazyLock},
    time::{Duration, Instant},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};
use tracing::{info, warn};
use url::Url;
use xxhash_rust::xxh3::xxh3_128;

const WHOLE_HASH_MAX_DIMENSION: u32 = 512;
const OCR_CROP_MAX_BYTES: usize = 1_000_000;
const OCR_CROP_MAX_DIMENSION: u32 = 1600;
const PREVIEW_MIN_PIXEL_REDUCTION_PERCENT: u64 = 35;

thread_local! {
    static IMAGE_RESIZER: RefCell<fast_image_resize::Resizer> =
        RefCell::new(fast_image_resize::Resizer::new());
}

static PHASHER: LazyLock<Hasher<[u8; 8]>> = LazyLock::new(|| {
    HasherConfig::with_bytes_type::<[u8; 8]>()
        .hash_size(8, 8)
        .hash_alg(HashAlg::Median)
        .preproc_dct()
        .to_hasher()
});
static DHASHER: LazyLock<Hasher<[u8; 8]>> = LazyLock::new(|| {
    HasherConfig::with_bytes_type::<[u8; 8]>()
        .hash_size(8, 8)
        .hash_alg(HashAlg::Gradient)
        .to_hasher()
});

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum HashMode {
    Candidate { local_hashes: bool },
    Specimen,
    FullDiagnostics,
}

impl HashMode {
    pub const fn candidate() -> Self {
        Self::Candidate { local_hashes: true }
    }

    pub const fn candidate_without_local_hashes() -> Self {
        Self::Candidate {
            local_hashes: false,
        }
    }

    const fn needs_normalized_luma(self) -> bool {
        match self {
            Self::Candidate { local_hashes } => local_hashes,
            Self::Specimen | Self::FullDiagnostics => true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct PipelineTimings {
    pub total_us: u128,
    pub xxh128_us: u128,
    pub decode_us: u128,
    pub normalize_luma_us: u128,
    pub orientation_us: u128,
    pub base_tile_scorer_us: u128,
    pub local_anchors_us: u128,
    pub local_hashes_us: u128,
    pub whole_thumbnail_us: u128,
    pub visual_signature_us: u128,
    pub text_grid_us: u128,
    pub perceptual_hashes_us: u128,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub enum CdnCacheStatus {
    #[default]
    Unknown,
    Hit,
    Miss,
    Dynamic,
    Expired,
    Bypass,
    Revalidated,
    Updating,
    Stale,
    Other,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct DownloadTimings {
    pub total_us: u128,
    pub gate_wait_us: u128,
    pub request_us: u128,
    pub body_us: u128,
    pub xxh128_us: u128,
    pub bytes: usize,
    pub cdn_cache_status: CdnCacheStatus,
    pub cdn_age_seconds: Option<u32>,
}

#[derive(Clone)]
pub struct DownloadedImage {
    pub bytes: bytes::Bytes,
    pub byte_xxh128: String,
    pub mime: Option<String>,
    pub timings: DownloadTimings,
}

pub struct StagedImageFingerprint {
    pub fingerprint: ImageFingerprint,
    pub timings: PipelineTimings,
    image: DynamicImage,
}

pub struct CpuGate {
    high_priority_tx: mpsc::UnboundedSender<CpuPermitRequest>,
    low_priority_tx: mpsc::UnboundedSender<CpuPermitRequest>,
}

impl CpuGate {
    pub fn new(permits: usize) -> Self {
        let (high_priority_tx, high_priority_rx) = mpsc::unbounded_channel();
        let (low_priority_tx, low_priority_rx) = mpsc::unbounded_channel();
        tokio::spawn(cpu_gate_arbiter(
            Arc::new(Semaphore::new(permits.max(1))),
            high_priority_rx,
            low_priority_rx,
        ));
        Self {
            high_priority_tx,
            low_priority_tx,
        }
    }

    pub async fn acquire_high_priority(&self) -> Result<OwnedSemaphorePermit> {
        acquire_cpu_permit(&self.high_priority_tx).await
    }

    pub async fn acquire_low_priority(&self) -> Result<OwnedSemaphorePermit> {
        acquire_cpu_permit(&self.low_priority_tx).await
    }
}

type CpuPermitRequest = oneshot::Sender<OwnedSemaphorePermit>;

async fn acquire_cpu_permit(
    tx: &mpsc::UnboundedSender<CpuPermitRequest>,
) -> Result<OwnedSemaphorePermit> {
    let (respond_to, response) = oneshot::channel();
    tx.send(respond_to)
        .map_err(|_| anyhow!("CPU gate closed"))?;
    response.await.context("CPU gate stopped")
}

async fn cpu_gate_arbiter(
    permits: Arc<Semaphore>,
    mut high_priority_rx: mpsc::UnboundedReceiver<CpuPermitRequest>,
    mut low_priority_rx: mpsc::UnboundedReceiver<CpuPermitRequest>,
) {
    let mut high_closed = false;
    let mut low_closed = false;

    while !high_closed || !low_closed {
        let Ok(permit) = permits.clone().acquire_owned().await else {
            break;
        };
        tokio::select! {
            biased;
            request = high_priority_rx.recv(), if !high_closed => {
                if let Some(request) = request {
                    let _ = request.send(permit);
                } else {
                    drop(permit);
                    high_closed = true;
                }
            }
            request = low_priority_rx.recv(), if !low_closed => {
                if let Some(request) = request {
                    let _ = request.send(permit);
                } else {
                    drop(permit);
                    low_closed = true;
                }
            }
        }
    }
}

struct DownloadAttemptError {
    source: anyhow::Error,
    retryable: bool,
    retry_after: Option<Duration>,
    request_us: u128,
    body_us: u128,
}

impl DownloadAttemptError {
    fn new(source: anyhow::Error, retryable: bool, request_us: u128, body_us: u128) -> Self {
        Self {
            source,
            retryable,
            retry_after: None,
            request_us,
            body_us,
        }
    }

    fn with_retry_after(mut self, retry_after: Option<Duration>) -> Self {
        self.retry_after = retry_after;
        self
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DiscordPreviewRequest {
    pub url: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PreparedOcrCrop {
    pub label: String,
    pub width: u32,
    pub height: u32,
    pub mime: String,
    #[serde(skip_serializing)]
    pub bytes: Vec<u8>,
}

pub async fn download_image(
    http: &Client,
    url: &str,
    mime_hint: Option<&str>,
    download_config: &DownloadConfig,
    download_gate: &Semaphore,
) -> Result<DownloadedImage> {
    let total_started = Instant::now();
    validate_url(url)?;

    let mut total_request_us = 0_u128;
    let mut total_body_us = 0_u128;
    let mut total_gate_wait_us = 0_u128;
    let max_attempts = download_config.max_retries.saturating_add(1);
    for attempt in 0..max_attempts {
        let gate_started = Instant::now();
        let download_permit = download_gate.acquire().await?;
        total_gate_wait_us = total_gate_wait_us.saturating_add(gate_started.elapsed().as_micros());

        match download_image_attempt(http, url, mime_hint, download_config).await {
            Ok(mut downloaded) => {
                downloaded.timings.total_us = total_started.elapsed().as_micros();
                downloaded.timings.gate_wait_us = total_gate_wait_us;
                downloaded.timings.request_us =
                    total_request_us.saturating_add(downloaded.timings.request_us);
                downloaded.timings.body_us =
                    total_body_us.saturating_add(downloaded.timings.body_us);
                return Ok(downloaded);
            }
            Err(error) => {
                drop(download_permit);
                total_request_us = total_request_us.saturating_add(error.request_us);
                total_body_us = total_body_us.saturating_add(error.body_us);
                let attempt_number = attempt.saturating_add(1);
                if error.retryable && attempt_number < max_attempts {
                    let delay = error
                        .retry_after
                        .unwrap_or_else(|| download_retry_delay(download_config, attempt));
                    warn!(
                        event = "image.download_retry",
                        image_url = %url_log_label(url),
                        attempt = attempt_number,
                        max_attempts,
                        retry_after_ms = delay.as_millis(),
                        source = %error.source,
                        "retrying image download after transient failure"
                    );
                    tokio::time::sleep(delay).await;
                    continue;
                }

                return Err(error.source.context(format!(
                    "downloading image failed after {attempt_number}/{max_attempts} attempts"
                )));
            }
        }
    }

    Err(anyhow!("download retry loop ended without an attempt"))
}

async fn download_image_attempt(
    http: &Client,
    url: &str,
    mime_hint: Option<&str>,
    download_config: &DownloadConfig,
) -> std::result::Result<DownloadedImage, DownloadAttemptError> {
    let request_started = Instant::now();
    let response = http
        .get(url)
        .timeout(Duration::from_secs(download_config.timeout_seconds))
        .send()
        .await
        .map_err(|error| {
            let retryable = reqwest_error_is_retryable(&error);
            DownloadAttemptError::new(
                anyhow::Error::new(error).context("downloading image"),
                retryable,
                request_started.elapsed().as_micros(),
                0,
            )
        })?;
    let request_us = request_started.elapsed().as_micros();

    let cdn_cache_status = cdn_cache_status(response.headers());
    let cdn_age_seconds = cdn_age_seconds(response.headers());
    let (header_mime, content_length) =
        validate_download_response_metadata(&response, url, download_config, request_us)?;
    let (bytes, body_us) = read_limited_response_body(
        response,
        content_length,
        download_config.max_bytes,
        request_us,
    )
    .await?;

    let hash_started = Instant::now();
    let byte_xxh128 = xxh128_hex(&bytes);
    let xxh128_us = hash_started.elapsed().as_micros();
    let mime = header_mime.or_else(|| mime_hint.map(str::to_owned));
    let timings = DownloadTimings {
        total_us: 0,
        gate_wait_us: 0,
        request_us,
        body_us,
        xxh128_us,
        bytes: bytes.len(),
        cdn_cache_status,
        cdn_age_seconds,
    };
    Ok(DownloadedImage {
        bytes,
        byte_xxh128,
        mime,
        timings,
    })
}

fn validate_download_response_metadata(
    response: &reqwest::Response,
    url: &str,
    download_config: &DownloadConfig,
    request_us: u128,
) -> std::result::Result<(Option<String>, Option<u64>), DownloadAttemptError> {
    if response.url().as_str() != url {
        return Err(DownloadAttemptError::new(
            anyhow!("image request followed a redirect; downloader requires a no-redirect client"),
            false,
            request_us,
            0,
        ));
    }

    if !response.status().is_success() {
        let status = response.status();
        let retry_after = response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(parse_retry_after);
        return Err(DownloadAttemptError::new(
            anyhow!("download returned {status}"),
            status_is_retryable(status),
            request_us,
            0,
        )
        .with_retry_after(retry_after));
    }

    let content_length = response.content_length();
    if let Some(content_length) = content_length
        && content_length > download_config.max_bytes as u64
    {
        return Err(DownloadAttemptError::new(
            anyhow!("image content-length exceeds limit"),
            false,
            request_us,
            0,
        ));
    }

    let header_mime = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.split(';').next().unwrap_or(value).trim().to_owned())
        .filter(|mime| !mime.eq_ignore_ascii_case("application/octet-stream"));
    if header_mime
        .as_deref()
        .is_some_and(|mime| !mime_is_image(mime))
    {
        return Err(DownloadAttemptError::new(
            anyhow!("response content-type is not an image"),
            false,
            request_us,
            0,
        ));
    }

    Ok((header_mime, content_length))
}

async fn read_limited_response_body(
    response: reqwest::Response,
    content_length: Option<u64>,
    max_bytes: usize,
    request_us: u128,
) -> std::result::Result<(bytes::Bytes, u128), DownloadAttemptError> {
    let initial_capacity = content_length.map_or(0, |length| (length as usize).min(max_bytes));
    let mut body = BytesMut::with_capacity(initial_capacity);
    let mut stream = response.bytes_stream();

    let body_started = Instant::now();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| {
            let retryable = reqwest_error_is_retryable(&error);
            DownloadAttemptError::new(
                anyhow::Error::new(error).context(format!(
                    "reading image body after {} ms; bytes_read={}; content_length={}; max_bytes={}",
                    body_started.elapsed().as_millis(),
                    body.len(),
                    content_length
                        .map_or_else(|| "unknown".to_owned(), |length| length.to_string()),
                    max_bytes
                )),
                retryable,
                request_us,
                body_started.elapsed().as_micros(),
            )
        })?;
        if body.len().saturating_add(chunk.len()) > max_bytes {
            return Err(DownloadAttemptError::new(
                anyhow!(
                    "image body exceeds limit; bytes_read={}; next_chunk={}; max_bytes={}",
                    body.len(),
                    chunk.len(),
                    max_bytes
                ),
                false,
                request_us,
                body_started.elapsed().as_micros(),
            ));
        }
        body.extend_from_slice(&chunk);
    }
    let body_us = body_started.elapsed().as_micros();

    Ok((body.freeze(), body_us))
}

fn reqwest_error_is_retryable(error: &reqwest::Error) -> bool {
    error.is_timeout() || error.is_connect() || error.is_body()
}

fn status_is_retryable(status: StatusCode) -> bool {
    status == StatusCode::REQUEST_TIMEOUT
        || status == StatusCode::TOO_MANY_REQUESTS
        || status.is_server_error()
}

fn cdn_cache_status(headers: &HeaderMap) -> CdnCacheStatus {
    let Some(status) = headers
        .get("cf-cache-status")
        .and_then(|value| value.to_str().ok())
    else {
        return CdnCacheStatus::Unknown;
    };

    match status.to_ascii_lowercase().as_str() {
        "hit" => CdnCacheStatus::Hit,
        "miss" => CdnCacheStatus::Miss,
        "dynamic" => CdnCacheStatus::Dynamic,
        "expired" => CdnCacheStatus::Expired,
        "bypass" => CdnCacheStatus::Bypass,
        "revalidated" => CdnCacheStatus::Revalidated,
        "updating" => CdnCacheStatus::Updating,
        "stale" => CdnCacheStatus::Stale,
        _ => CdnCacheStatus::Other,
    }
}

fn cdn_age_seconds(headers: &HeaderMap) -> Option<u32> {
    headers
        .get("age")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok())
}

fn download_retry_delay(config: &DownloadConfig, attempt: usize) -> Duration {
    let shift = u32::try_from(attempt.min(10)).unwrap_or(10);
    let multiplier = 1_u64.checked_shl(shift).unwrap_or(u64::MAX);
    Duration::from_millis(config.retry_base_delay_ms.saturating_mul(multiplier))
}

fn parse_retry_after(value: &str) -> Option<Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let retry_at = DateTime::parse_from_rfc2822(value)
        .ok()?
        .with_timezone(&Utc);
    retry_at.signed_duration_since(Utc::now()).to_std().ok()
}

pub fn build_discord_preview_request(
    candidate: &ImageCandidate,
    download_config: &DownloadConfig,
    match_config: &MatchConfig,
) -> Option<DiscordPreviewRequest> {
    if !download_config.preview.enabled {
        return None;
    }
    if candidate
        .mime_hint
        .as_deref()
        .is_some_and(|mime| !mime_is_image(mime))
    {
        return None;
    }
    if download_config.preview.skip_animated && candidate_is_animated(candidate) {
        return None;
    }
    if candidate
        .size_bytes
        .is_some_and(|bytes| bytes > u64::try_from(download_config.max_bytes).unwrap_or(u64::MAX))
    {
        return None;
    }
    if let Some(size_bytes) = candidate.size_bytes
        && size_bytes < download_config.preview.min_original_bytes
    {
        return None;
    }

    let proxy_url = candidate.proxy_url.as_deref()?;
    let mut url = Url::parse(proxy_url).ok()?;
    if url.scheme() != "https" {
        return None;
    }
    if !url.host_str().is_some_and(is_discord_host) {
        return None;
    }

    let width = candidate.metadata_width?;
    let height = candidate.metadata_height?;
    if width == 0 || height == 0 {
        return None;
    }
    let pixels = width as u64 * height as u64;
    if pixels > download_config.max_decoded_pixels {
        return None;
    }
    let aspect = width.max(height) as f32 / width.min(height) as f32;
    if aspect > match_config.local_max_aspect_ratio {
        return None;
    }

    let (target_width, target_height) = normalized_dimensions_for_match_config(
        width,
        height,
        match_config,
        download_config.preview.mode,
    );
    if target_width >= width && target_height >= height {
        return None;
    }
    let source_pixels = u64::from(width) * u64::from(height);
    let target_pixels = u64::from(target_width) * u64::from(target_height);
    let reduction_percent =
        100u64.saturating_sub(target_pixels.saturating_mul(100) / source_pixels.max(1));
    if reduction_percent < PREVIEW_MIN_PIXEL_REDUCTION_PERCENT {
        return None;
    }

    let retained_pairs = url
        .query_pairs()
        .filter(|(key, _)| key != "width" && key != "height")
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    url.set_query(None);
    {
        let mut query = url.query_pairs_mut();
        for (key, value) in retained_pairs {
            query.append_pair(&key, &value);
        }
        query.append_pair("width", &target_width.to_string());
        query.append_pair("height", &target_height.to_string());
    }

    Some(DiscordPreviewRequest {
        url: url.to_string(),
        width: target_width,
        height: target_height,
    })
}

pub fn normalized_dimensions_for_match_config(
    width: u32,
    height: u32,
    config: &MatchConfig,
    mode: PreviewDownloadMode,
) -> (u32, u32) {
    match mode {
        PreviewDownloadMode::MatchNormalized => {
            let width = width.max(1);
            let height = height.max(1);
            let width_ratio = config.local_max_width as f64 / width as f64;
            let height_ratio = config.local_max_height as f64 / height as f64;
            let area_ratio = (config.local_max_area as f64 / (width as f64 * height as f64)).sqrt();
            let ratio = width_ratio.min(height_ratio).min(area_ratio).min(1.0);
            (
                ((width as f64 * ratio).round() as u32).max(1),
                ((height as f64 * ratio).round() as u32).max(1),
            )
        }
    }
}

fn candidate_is_animated(candidate: &ImageCandidate) -> bool {
    candidate
        .mime_hint
        .as_deref()
        .is_some_and(|mime| mime.eq_ignore_ascii_case("image/gif"))
        || candidate
            .url
            .as_bytes()
            .windows(4)
            .any(|window| window.eq_ignore_ascii_case(b".gif"))
}

fn mime_is_image(mime: &str) -> bool {
    mime.as_bytes()
        .get(..6)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"image/"))
}

pub async fn hash_downloaded_image(
    downloaded: DownloadedImage,
    max_pixels: u64,
    match_config: &MatchConfig,
    decode_gate: &Arc<CpuGate>,
    mode: HashMode,
) -> Result<ImageFingerprint> {
    let match_config = match_config.clone();
    let decode_permit = decode_gate.acquire_low_priority().await?;
    tokio::task::spawn_blocking(move || {
        let _decode_permit = decode_permit;
        decode_and_hash_blocking_with_timings(
            downloaded.bytes.as_ref(),
            downloaded.byte_xxh128,
            downloaded.mime,
            max_pixels,
            &match_config,
            mode,
        )
        .map(|(fingerprint, _)| fingerprint)
    })
    .await
    .context("decode task panicked")?
}

pub async fn hash_downloaded_image_tier1(
    downloaded: DownloadedImage,
    max_pixels: u64,
    match_config: &MatchConfig,
    decode_gate: &Arc<CpuGate>,
) -> Result<StagedImageFingerprint> {
    let match_config = match_config.clone();
    let decode_permit = decode_gate.acquire_low_priority().await?;
    tokio::task::spawn_blocking(move || {
        let _decode_permit = decode_permit;
        decode_tier1_blocking_with_timings(
            downloaded.bytes.as_ref(),
            downloaded.byte_xxh128,
            downloaded.mime,
            max_pixels,
            &match_config,
        )
    })
    .await
    .context("tier-1 decode task panicked")?
}

pub async fn complete_staged_image_fingerprint(
    staged: StagedImageFingerprint,
    match_config: &MatchConfig,
    decode_gate: &Arc<CpuGate>,
    mode: HashMode,
) -> Result<(ImageFingerprint, PipelineTimings)> {
    let match_config = match_config.clone();
    let decode_permit = decode_gate.acquire_low_priority().await?;
    tokio::task::spawn_blocking(move || {
        let _decode_permit = decode_permit;
        complete_staged_fingerprint_blocking(staged, &match_config, mode)
    })
    .await
    .context("tier-2 fingerprint task panicked")
}

pub async fn prepare_ocr_payload_from_downloaded(
    bytes: bytes::Bytes,
    max_pixels: u64,
    match_config: &MatchConfig,
    decode_gate: &Arc<CpuGate>,
) -> Result<PreparedOcrCrop> {
    let match_config = match_config.clone();
    let decode_permit = decode_gate.acquire_low_priority().await?;
    tokio::task::spawn_blocking(move || {
        let _decode_permit = decode_permit;
        prepare_ocr_payload_from_bytes(bytes.as_ref(), max_pixels, &match_config)
    })
    .await
    .context("OCR crop task panicked")?
}

pub fn hash_image_bytes(
    bytes: &[u8],
    mime: Option<String>,
    max_pixels: u64,
    match_config: &MatchConfig,
    mode: HashMode,
) -> Result<ImageFingerprint> {
    Ok(hash_image_bytes_with_timings(bytes, mime, max_pixels, match_config, mode)?.0)
}

pub fn hash_image_bytes_with_timings(
    bytes: &[u8],
    mime: Option<String>,
    max_pixels: u64,
    match_config: &MatchConfig,
    mode: HashMode,
) -> Result<(ImageFingerprint, PipelineTimings)> {
    let total_started = Instant::now();
    let hash_started = Instant::now();
    let byte_xxh128 = xxh128_hex(bytes);
    let xxh128_us = hash_started.elapsed().as_micros();
    let (fingerprint, mut timings) = decode_and_hash_blocking_with_timings(
        bytes,
        byte_xxh128,
        mime,
        max_pixels,
        match_config,
        mode,
    )?;
    timings.xxh128_us = xxh128_us;
    timings.total_us = total_started.elapsed().as_micros();
    Ok((fingerprint, timings))
}

pub fn prepare_ocr_payload_from_bytes(
    bytes: &[u8],
    max_pixels: u64,
    match_config: &MatchConfig,
) -> Result<PreparedOcrCrop> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("guessing image format for OCR crops")?;
    let mut limits = Limits::default();
    limits.max_alloc = Some(max_pixels.saturating_mul(4));
    reader.limits(limits);
    let source_mime = reader
        .format()
        .and_then(image_format_mime)
        .map(str::to_owned);
    let image = reader.decode().context("decoding image for OCR crops")?;
    let (width, height) = image.dimensions();
    if width as u64 * height as u64 > max_pixels {
        bail!("decoded image exceeds pixel limit for OCR crops");
    }
    prepare_ocr_payload_from_image(&image, match_config, bytes, source_mime.as_deref())
}

pub fn source_ocr_payload(
    bytes: &[u8],
    mime: Option<&str>,
    width: u32,
    height: u32,
) -> Option<PreparedOcrCrop> {
    source_ocr_crop("full", bytes, mime, width, height)
}

fn decode_and_hash_blocking_with_timings(
    bytes: &[u8],
    byte_xxh128: String,
    mime: Option<String>,
    max_pixels: u64,
    match_config: &MatchConfig,
    mode: HashMode,
) -> Result<(ImageFingerprint, PipelineTimings)> {
    let staged =
        decode_tier1_blocking_with_timings(bytes, byte_xxh128, mime, max_pixels, match_config)?;
    Ok(complete_staged_fingerprint_blocking(
        staged,
        match_config,
        mode,
    ))
}

fn decode_tier1_blocking_with_timings(
    bytes: &[u8],
    byte_xxh128: String,
    mime: Option<String>,
    max_pixels: u64,
    match_config: &MatchConfig,
) -> Result<StagedImageFingerprint> {
    let mut timings = PipelineTimings::default();
    let decode_started = Instant::now();
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("guessing image format")?;
    let mut limits = Limits::default();
    limits.max_alloc = Some(max_pixels.saturating_mul(4));
    reader.limits(limits);

    let decoded_mime = reader
        .format()
        .and_then(image_format_mime)
        .map(str::to_owned);
    let image = reader.decode().context("decoding image")?;
    let (width, height) = image.dimensions();
    let pixels = width as u64 * height as u64;

    if pixels > max_pixels {
        bail!("decoded image exceeds pixel limit");
    }
    timings.decode_us = decode_started.elapsed().as_micros();

    let thumbnail_started = Instant::now();
    let hash_rgb = thumbnail_rgb(&image, WHOLE_HASH_MAX_DIMENSION);
    let hash_luma: GrayImage = hash_rgb.convert();
    timings.whole_thumbnail_us = thumbnail_started.elapsed().as_micros();

    let visual_started = Instant::now();
    let visual = visual_signature(&hash_rgb, &hash_luma);
    timings.visual_signature_us = visual_started.elapsed().as_micros();

    let perceptual_hashes_started = Instant::now();
    let orientation_started = Instant::now();
    let corrected_hash_luma = orientation_corrected_hash_luma(&hash_luma, match_config);
    timings.orientation_us = orientation_started.elapsed().as_micros();
    let hash_image = DynamicImage::ImageLuma8(corrected_hash_luma.unwrap_or(hash_luma));
    let phash = PHASHER.hash_image(&hash_image);
    let dhash = DHASHER.hash_image(&hash_image);
    timings.perceptual_hashes_us = perceptual_hashes_started.elapsed().as_micros();

    Ok(StagedImageFingerprint {
        fingerprint: ImageFingerprint {
            width,
            height,
            mime: decoded_mime.or(mime),
            byte_xxh128,
            phash64: hex::encode(phash.as_bytes()),
            dhash64: hex::encode(dhash.as_bytes()),
            visual,
            local_anchors: Vec::new(),
            local_hashes: Vec::new(),
        },
        timings,
        image,
    })
}

fn complete_staged_fingerprint_blocking(
    staged: StagedImageFingerprint,
    match_config: &MatchConfig,
    mode: HashMode,
) -> (ImageFingerprint, PipelineTimings) {
    let mut fingerprint = staged.fingerprint;
    let image = staged.image;
    let mut timings = staged.timings;

    let normalize_started = Instant::now();
    let mut normalized_luma = mode
        .needs_normalized_luma()
        .then(|| normalize_luma(&image, match_config))
        .flatten();
    timings.normalize_luma_us = normalize_started.elapsed().as_micros();

    let local_orientation_started = Instant::now();
    if let Some(normalized) = normalized_luma.as_ref()
        && let Some(corrected) = orientation_corrected_hash_luma(normalized, match_config)
    {
        normalized_luma = Some(corrected);
    }
    timings.orientation_us = timings
        .orientation_us
        .saturating_add(local_orientation_started.elapsed().as_micros());

    let base_tile_scorer = if let Some(normalized) = normalized_luma.as_ref() {
        let scorer_started = Instant::now();
        let scorer = TileScorer::new(normalized);
        timings.base_tile_scorer_us = scorer_started.elapsed().as_micros();
        Some(scorer)
    } else {
        None
    };

    if let Some((normalized, base_scorer)) = normalized_luma.as_ref().zip(base_tile_scorer.as_ref())
    {
        let text_grid_started = Instant::now();
        fingerprint.visual.text_grid =
            text_grid_signature_with_scorer(normalized, match_config, base_scorer);
        timings.text_grid_us = text_grid_started.elapsed().as_micros();
    }

    let (local_anchors, local_hashes) = compute_local_features(
        normalized_luma.as_ref(),
        base_tile_scorer.as_ref(),
        match_config,
        mode,
        &mut timings,
    );
    fingerprint.local_anchors = local_anchors;
    fingerprint.local_hashes = local_hashes;

    (fingerprint, timings)
}

pub fn xxh128_hex(bytes: &[u8]) -> String {
    format!("{:032x}", xxh3_128(bytes))
}

fn compute_local_features(
    normalized_luma: Option<&GrayImage>,
    base_tile_scorer: Option<&TileScorer<'_>>,
    match_config: &MatchConfig,
    mode: HashMode,
    timings: &mut PipelineTimings,
) -> (Vec<ImageAnchor>, Vec<LocalImageHash>) {
    let Some((normalized, base_scorer)) = normalized_luma.zip(base_tile_scorer) else {
        return (Vec::new(), Vec::new());
    };

    match mode {
        HashMode::Candidate { local_hashes } => {
            if local_hashes {
                let started = Instant::now();
                let local_hashes =
                    scan_local_hashes_with_base_scorer(normalized, match_config, base_scorer);
                timings.local_hashes_us = started.elapsed().as_micros();
                (Vec::new(), local_hashes)
            } else {
                (Vec::new(), Vec::new())
            }
        }
        HashMode::Specimen => {
            let started = Instant::now();
            let local_anchors =
                select_local_anchors_with_scorer(normalized, match_config, base_scorer);
            timings.local_anchors_us = started.elapsed().as_micros();
            (local_anchors, Vec::new())
        }
        HashMode::FullDiagnostics => {
            let anchors_started = Instant::now();
            let local_anchors =
                select_local_anchors_with_scorer(normalized, match_config, base_scorer);
            timings.local_anchors_us = anchors_started.elapsed().as_micros();
            let hashes_started = Instant::now();
            let local_hashes =
                scan_local_hashes_with_base_scorer(normalized, match_config, base_scorer);
            timings.local_hashes_us = hashes_started.elapsed().as_micros();
            (local_anchors, local_hashes)
        }
    }
}

fn image_format_mime(format: ImageFormat) -> Option<&'static str> {
    match format {
        ImageFormat::Png => Some("image/png"),
        ImageFormat::Jpeg => Some("image/jpeg"),
        ImageFormat::Gif => Some("image/gif"),
        ImageFormat::WebP => Some("image/webp"),
        _ => None,
    }
}

fn visual_signature(rgb: &RgbImage, luma: &GrayImage) -> ImageVisualSignature {
    let width = luma.width().max(1);
    let height = luma.height().max(1);
    let pixel_count = (width as u64 * height as u64).max(1);
    let width_usize = width as usize;
    let height_usize = height as usize;

    let mut luma_sum = 0u64;
    let mut luma_sum_sq = 0u64;
    let mut rgb_sum = [0u64; 3];
    let mut grid_sum = [0u64; 16];
    let mut grid_count = [0u64; 16];
    let rgb_pixels = rgb.as_raw();
    let rgb_full = rgb_pixels.len() >= width_usize.saturating_mul(height_usize).saturating_mul(3);
    let grid_segments = grid_segments(width_usize);

    for (y, row) in luma.as_raw().chunks_exact(width_usize).enumerate() {
        let grid_y = ((y as u32)
            .saturating_mul(4)
            .checked_div(height)
            .unwrap_or(0)
            .min(3)
            * 4) as usize;
        for (grid_x, &(start, end)) in grid_segments.iter().enumerate() {
            let (segment_sum, segment_sum_sq) = reduce_luma_segment(&row[start..end]);
            luma_sum = luma_sum.saturating_add(segment_sum);
            luma_sum_sq = luma_sum_sq.saturating_add(segment_sum_sq);
            grid_sum[grid_y + grid_x] = grid_sum[grid_y + grid_x].saturating_add(segment_sum);
            grid_count[grid_y + grid_x] =
                grid_count[grid_y + grid_x].saturating_add((end - start) as u64);
        }
    }

    if rgb_full {
        let (pixels, _) = rgb_pixels.as_chunks::<3>();
        for pixel in pixels.iter().take(width_usize.saturating_mul(height_usize)) {
            rgb_sum[0] += u64::from(pixel[0]);
            rgb_sum[1] += u64::from(pixel[1]);
            rgb_sum[2] += u64::from(pixel[2]);
        }
    } else {
        let (pixels, _) = rgb_pixels.as_chunks::<3>();
        for pixel in pixels {
            rgb_sum[0] += u64::from(pixel[0]);
            rgb_sum[1] += u64::from(pixel[1]);
            rgb_sum[2] += u64::from(pixel[2]);
        }
    }

    let luma_mean = luma_sum as f32 / pixel_count as f32;
    let luma_variance = (luma_sum_sq as f32 / pixel_count as f32) - (luma_mean * luma_mean);
    let mut grid_luma = [0u8; 16];
    for (index, value) in grid_luma.iter_mut().enumerate() {
        *value = if grid_count[index] == 0 {
            luma_mean.round().clamp(0.0, 255.0) as u8
        } else {
            (grid_sum[index] as f32 / grid_count[index] as f32)
                .round()
                .clamp(0.0, 255.0) as u8
        };
    }

    ImageVisualSignature {
        luma_mean: luma_mean.round().clamp(0.0, 255.0) as u8,
        luma_std: luma_variance.max(0.0).sqrt().round().clamp(0.0, 255.0) as u8,
        rgb_mean: [
            (rgb_sum[0] as f32 / pixel_count as f32)
                .round()
                .clamp(0.0, 255.0) as u8,
            (rgb_sum[1] as f32 / pixel_count as f32)
                .round()
                .clamp(0.0, 255.0) as u8,
            (rgb_sum[2] as f32 / pixel_count as f32)
                .round()
                .clamp(0.0, 255.0) as u8,
        ],
        grid_luma,
        text_grid: vec![0; 64],
    }
}

fn grid_segments(width: usize) -> [(usize, usize); 4] {
    let mut segments = [(0usize, 0usize); 4];
    for (index, segment) in segments.iter_mut().enumerate() {
        let start = index.saturating_mul(width).div_ceil(4);
        let end = (index + 1).saturating_mul(width).div_ceil(4);
        *segment = (start.min(width), end.min(width));
    }
    segments
}

#[cfg(feature = "nightly-simd")]
fn reduce_luma_segment(values: &[u8]) -> (u64, u64) {
    use std::simd::{Simd, num::SimdUint};

    const LANES: usize = 32;
    let mut sum = 0u64;
    let mut sum_sq = 0u64;
    let (chunks, remainder) = values.as_chunks::<LANES>();
    for chunk in chunks {
        let lanes = Simd::<u8, LANES>::from_array(*chunk).cast::<u16>();
        sum += u64::from(lanes.cast::<u32>().reduce_sum());
        sum_sq += u64::from((lanes * lanes).cast::<u32>().reduce_sum());
    }
    for &value in remainder {
        let value = u64::from(value);
        sum += value;
        sum_sq += value * value;
    }
    (sum, sum_sq)
}

#[cfg(not(feature = "nightly-simd"))]
fn reduce_luma_segment(values: &[u8]) -> (u64, u64) {
    let mut sum = 0u64;
    let mut sum_sq = 0u64;
    for &value in values {
        let value = u64::from(value);
        sum += value;
        sum_sq += value * value;
    }
    (sum, sum_sq)
}

fn text_grid_signature_with_scorer(
    image: &GrayImage,
    config: &MatchConfig,
    scorer: &TileScorer<'_>,
) -> Vec<u8> {
    let tile_w = config.local_tile_width.max(8);
    let tile_h = config.local_tile_height.max(8);
    if image.width() < tile_w || image.height() < tile_h {
        return vec![0; 64];
    }

    let stride = config.local_stride.max(4);
    let cols = ((image.width() - tile_w) / stride + 1) as usize;
    let rows = ((image.height() - tile_h) / stride + 1) as usize;
    let sample_budget = config.local_tile_budget.clamp(1, 20_000);
    let step = grid_sample_step(cols.saturating_mul(rows), sample_budget);
    let mut seen = [0u32; 64];
    let mut text_like = [0u32; 64];

    for row in (0..rows).step_by(step) {
        let y = ((row as u32) * stride).min(image.height() - tile_h);
        for col in (0..cols).step_by(step) {
            let x = ((col as u32) * stride).min(image.width() - tile_w);
            let region = tile_region(image.width(), image.height(), x, y) as usize;
            seen[region] += 1;
            if scorer.features(x, y, tile_w, tile_h).is_some() {
                text_like[region] += 1;
            }
        }
    }

    let mut grid = vec![0u8; 64];
    for (index, value) in grid.iter_mut().enumerate() {
        *value = text_like[index]
            .saturating_mul(255)
            .checked_div(seen[index])
            .unwrap_or(0)
            .min(255) as u8;
    }
    grid
}

fn grid_sample_step(total_positions: usize, budget: usize) -> usize {
    if total_positions <= budget {
        return 1;
    }
    let ratio = total_positions.div_ceil(budget);
    let mut step = 1usize;
    while step.saturating_mul(step) < ratio {
        step += 1;
    }
    step
}

fn normalize_luma(image: &DynamicImage, config: &MatchConfig) -> Option<GrayImage> {
    let (width, height) = image.dimensions();
    let width = width.max(1);
    let height = height.max(1);
    let aspect = width.max(height) as f32 / width.min(height) as f32;
    if aspect > config.local_max_aspect_ratio {
        return None;
    }

    let (resized_width, resized_height) = normalized_dimensions_for_match_config(
        width,
        height,
        config,
        PreviewDownloadMode::MatchNormalized,
    );
    let gray = image.to_luma8();
    if resized_width == width && resized_height == height {
        return Some(gray);
    }

    Some(resize_luma(&gray, resized_width, resized_height))
}

#[derive(Debug, Clone, Copy)]
struct OrientationEstimate {
    degrees: f32,
    gain: f32,
}

fn orientation_corrected_hash_luma(image: &GrayImage, config: &MatchConfig) -> Option<GrayImage> {
    if !config.perceptual_orientation_correction {
        return None;
    }
    let estimate = estimate_text_orientation(image, config)?;
    let min_effective_angle = config.perceptual_orientation_step_degrees.max(0.25);
    if estimate.degrees.abs() < min_effective_angle {
        return None;
    }
    let fill = image_mean_luma(image);
    Some(rotate_luma_bilinear(image, -estimate.degrees, fill))
}

fn estimate_text_orientation(
    image: &GrayImage,
    config: &MatchConfig,
) -> Option<OrientationEstimate> {
    let max_degrees = config.perceptual_orientation_max_degrees.clamp(0.0, 20.0);
    let step_degrees = config.perceptual_orientation_step_degrees.clamp(0.25, 5.0);
    if max_degrees < step_degrees {
        return None;
    }

    let points = orientation_edge_points(image);
    if points.len() < 128 {
        return None;
    }

    let bin_count = (((image.width() as f32).hypot(image.height() as f32)).ceil() as usize)
        .saturating_add(3)
        .max(8);
    let mut bins = vec![0u32; bin_count];
    let zero_score = orientation_projection_score(&points, 0.0, &mut bins);
    if zero_score <= f64::EPSILON {
        return None;
    }

    let mut best = OrientationEstimate {
        degrees: 0.0,
        gain: 1.0,
    };
    let steps = (max_degrees / step_degrees).floor() as i32;
    for step in -steps..=steps {
        let degrees = step as f32 * step_degrees;
        let score = orientation_projection_score(&points, degrees.to_radians(), &mut bins);
        if score > zero_score * f64::from(best.gain) {
            best = OrientationEstimate {
                degrees,
                gain: (score / zero_score) as f32,
            };
        }
    }

    (best.gain >= config.perceptual_orientation_min_gain).then_some(best)
}

#[derive(Debug, Clone, Copy)]
struct OrientationPoint {
    x: f32,
    y: f32,
    weight: u16,
}

fn orientation_edge_points(image: &GrayImage) -> Vec<OrientationPoint> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    if width < 3 || height < 3 {
        return Vec::new();
    }

    let stride = if width.max(height) >= 384 { 2 } else { 1 };
    let pixels = image.as_raw();
    let center_x = (width - 1) as f32 * 0.5;
    let center_y = (height - 1) as f32 * 0.5;
    let mut points = Vec::with_capacity(width.saturating_mul(height) / (stride * stride * 8));

    for y in (1..(height - 1)).step_by(stride) {
        let row = y * width;
        for x in (1..(width - 1)).step_by(stride) {
            let index = row + x;
            let gx = (pixels[index + 1] as i16 - pixels[index - 1] as i16).unsigned_abs();
            let gy = (pixels[index + width] as i16 - pixels[index - width] as i16).unsigned_abs();
            let magnitude = gx.saturating_add(gy);
            if magnitude < 42 {
                continue;
            }
            points.push(OrientationPoint {
                x: x as f32 - center_x,
                y: y as f32 - center_y,
                weight: magnitude.min(255),
            });
        }
    }
    points
}

fn orientation_projection_score(
    points: &[OrientationPoint],
    radians: f32,
    bins: &mut [u32],
) -> f64 {
    bins.fill(0);
    let bin_count = bins.len();
    let half = bin_count as f32 * 0.5;
    let sin = radians.sin();
    let cos = radians.cos();
    for point in points {
        let projected = (-sin).mul_add(point.x, cos * point.y) + half;
        let bin = projected.round().clamp(0.0, (bin_count - 1) as f32) as usize;
        bins[bin] += u32::from(point.weight);
    }
    bins.iter()
        .map(|&value| {
            let value = f64::from(value);
            value * value
        })
        .sum()
}

fn image_mean_luma(image: &GrayImage) -> u8 {
    let pixels = image.as_raw();
    if pixels.is_empty() {
        return 128;
    }
    let sum = pixels.iter().map(|value| u64::from(*value)).sum::<u64>();
    (sum / pixels.len() as u64).min(u64::from(u8::MAX)) as u8
}

fn rotate_luma_bilinear(image: &GrayImage, degrees: f32, fill: u8) -> GrayImage {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 || degrees.abs() < f32::EPSILON {
        return image.clone();
    }

    let radians = degrees.to_radians();
    let sin = radians.sin();
    let cos = radians.cos();
    let center_x = (width - 1) as f32 * 0.5;
    let center_y = (height - 1) as f32 * 0.5;
    let width_usize = width as usize;
    let source = image.as_raw();
    let max_x = (width - 1) as f32;
    let max_y = (height - 1) as f32;
    let x_hi = width - 1;
    let y_hi = height - 1;
    let mut output = vec![fill; width_usize.saturating_mul(height as usize)];

    for y in 0..height {
        let dy = y as f32 - center_y;
        let sin_dy = sin * dy;
        let cos_dy = cos * dy;
        let output_row = y as usize * width_usize;
        for x in 0..width {
            let dx = x as f32 - center_x;
            let source_x = cos.mul_add(dx, sin_dy) + center_x;
            let source_y = (-sin).mul_add(dx, cos_dy) + center_y;
            if source_x < 0.0 || source_x > max_x || source_y < 0.0 || source_y > max_y {
                continue;
            }
            let x0 = source_x.floor() as u32;
            let y0 = source_y.floor() as u32;
            let x1 = (x0 + 1).min(x_hi);
            let y1 = (y0 + 1).min(y_hi);
            let tx = source_x - x0 as f32;
            let ty = source_y - y0 as f32;
            let x0 = x0 as usize;
            let y0 = y0 as usize;
            let x1 = x1 as usize;
            let y1 = y1 as usize;
            let p00 = source[y0 * width_usize + x0] as f32;
            let p10 = source[y0 * width_usize + x1] as f32;
            let p01 = source[y1 * width_usize + x0] as f32;
            let p11 = source[y1 * width_usize + x1] as f32;
            let top = p00.mul_add(1.0 - tx, p10 * tx);
            let bottom = p01.mul_add(1.0 - tx, p11 * tx);
            output[output_row + x as usize] =
                top.mul_add(1.0 - ty, bottom * ty).round().clamp(0.0, 255.0) as u8;
        }
    }

    GrayImage::from_raw(width, height, output).unwrap_or_else(|| image.clone())
}

fn prepare_ocr_payload_from_image(
    image: &DynamicImage,
    match_config: &MatchConfig,
    source_bytes: &[u8],
    source_mime: Option<&str>,
) -> Result<PreparedOcrCrop> {
    let crop_byte_budget = OCR_CROP_MAX_BYTES;
    let Some(normalized) = normalize_luma(image, match_config) else {
        return encode_full_or_source_crop(image, source_bytes, source_mime, crop_byte_budget);
    };
    let Some((x, y, w, h)) = text_dense_bounds(&normalized, match_config) else {
        return encode_full_or_source_crop(image, source_bytes, source_mime, crop_byte_budget);
    };
    let padding = normalized
        .width()
        .max(normalized.height())
        .checked_div(20)
        .unwrap_or(0)
        .max(
            match_config
                .local_tile_width
                .max(match_config.local_tile_height)
                / 2,
        );
    let padded = pad_rect(x, y, w, h, padding, normalized.width(), normalized.height());
    let crop_rect = map_rect_to_original(
        padded,
        normalized.width(),
        normalized.height(),
        image.width(),
        image.height(),
    );
    if rect_covers_most_image(crop_rect, image.width(), image.height(), 97)
        && let Some(original) = source_ocr_crop(
            "full",
            source_bytes,
            source_mime,
            image.width(),
            image.height(),
        )
    {
        return Ok(original);
    }
    let crop = image.crop_imm(crop_rect.0, crop_rect.1, crop_rect.2, crop_rect.3);
    encode_ocr_crop("text_dense", crop, crop_byte_budget)
}

fn text_dense_bounds(image: &GrayImage, config: &MatchConfig) -> Option<(u32, u32, u32, u32)> {
    let tile_w = config.local_tile_width.max(8);
    let tile_h = config.local_tile_height.max(8);
    if image.width() < tile_w || image.height() < tile_h {
        return None;
    }

    let stride = config.local_stride.max(4);
    let scorer = TileScorer::new(image);
    let mut min_x = image.width();
    let mut min_y = image.height();
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut hits = Vec::new();
    let cols = ((image.width() - tile_w) / stride + 1) as usize;
    let rows = ((image.height() - tile_h) / stride + 1) as usize;
    let step = grid_sample_step(cols.saturating_mul(rows), config.local_tile_budget.max(1));
    let mut visited = 0usize;
    for row in (0..rows).step_by(step) {
        let y = ((row as u32) * stride).min(image.height() - tile_h);
        for col in (0..cols).step_by(step) {
            if visited >= config.local_tile_budget {
                break;
            }
            let x = ((col as u32) * stride).min(image.width() - tile_w);
            visited += 1;
            if scorer.basic_score(x, y, tile_w, tile_h).is_some() {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x + tile_w);
                max_y = max_y.max(y + tile_h);
                hits.push((x + tile_w / 2, y + tile_h / 2));
            }
        }
        if visited >= config.local_tile_budget {
            break;
        }
    }

    if hits.len() < 4 {
        return None;
    }
    if hits.len() < 16 {
        return Some((
            min_x,
            min_y,
            max_x.saturating_sub(min_x).max(1),
            max_y.saturating_sub(min_y).max(1),
        ));
    }

    let mut xs = hits.iter().map(|(x, _)| *x).collect::<Vec<_>>();
    let mut ys = hits.iter().map(|(_, y)| *y).collect::<Vec<_>>();
    xs.sort_unstable();
    ys.sort_unstable();
    let low_x = percentile_u32(&xs, 0.08).saturating_sub(tile_w / 2);
    let high_x = percentile_u32(&xs, 0.92).saturating_add(tile_w / 2);
    let low_y = percentile_u32(&ys, 0.08).saturating_sub(tile_h / 2);
    let high_y = percentile_u32(&ys, 0.92).saturating_add(tile_h / 2);

    Some((
        low_x.min(image.width().saturating_sub(1)),
        low_y.min(image.height().saturating_sub(1)),
        high_x.min(image.width()).saturating_sub(low_x).max(1),
        high_y.min(image.height()).saturating_sub(low_y).max(1),
    ))
}

fn percentile_u32(sorted: &[u32], quantile: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn pad_rect(
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    padding: u32,
    image_w: u32,
    image_h: u32,
) -> (u32, u32, u32, u32) {
    let x1 = x.saturating_sub(padding);
    let y1 = y.saturating_sub(padding);
    let x2 = x.saturating_add(w).saturating_add(padding).min(image_w);
    let y2 = y.saturating_add(h).saturating_add(padding).min(image_h);
    (
        x1,
        y1,
        x2.saturating_sub(x1).max(1),
        y2.saturating_sub(y1).max(1),
    )
}

fn map_rect_to_original(
    rect: (u32, u32, u32, u32),
    normalized_w: u32,
    normalized_h: u32,
    original_w: u32,
    original_h: u32,
) -> (u32, u32, u32, u32) {
    let scale_x = original_w as f64 / normalized_w.max(1) as f64;
    let scale_y = original_h as f64 / normalized_h.max(1) as f64;
    let x = (rect.0 as f64 * scale_x)
        .floor()
        .max(0.0)
        .min(original_w.saturating_sub(1) as f64) as u32;
    let y = (rect.1 as f64 * scale_y)
        .floor()
        .max(0.0)
        .min(original_h.saturating_sub(1) as f64) as u32;
    let x2 = ((rect.0 + rect.2) as f64 * scale_x).ceil().max(1.0) as u32;
    let y2 = ((rect.1 + rect.3) as f64 * scale_y).ceil().max(1.0) as u32;
    (
        x,
        y,
        x2.min(original_w).saturating_sub(x).max(1),
        y2.min(original_h).saturating_sub(y).max(1),
    )
}

fn encode_ocr_crop(label: &str, crop: DynamicImage, max_bytes: usize) -> Result<PreparedOcrCrop> {
    let mut image = constrain_ocr_crop_dimensions(crop);
    let mut quality = 88u8;
    let bytes = loop {
        let bytes = encode_jpeg(&image, quality).context("encoding OCR crop")?;
        if bytes.len() <= max_bytes {
            break bytes;
        }
        if quality > 48 {
            quality = quality.saturating_sub(10);
            continue;
        }
        let next_width = ((image.width() as f32) * 0.85).round().max(1.0) as u32;
        let next_height = ((image.height() as f32) * 0.85).round().max(1.0) as u32;
        if next_width == image.width() && next_height == image.height() {
            bail!("OCR crop could not be encoded within {max_bytes} bytes");
        }
        image = resize_dynamic_rgb(&image, next_width, next_height);
    };

    Ok(PreparedOcrCrop {
        label: label.to_owned(),
        width: image.width(),
        height: image.height(),
        mime: "image/jpeg".to_owned(),
        bytes,
    })
}

fn encode_full_or_source_crop(
    image: &DynamicImage,
    source_bytes: &[u8],
    source_mime: Option<&str>,
    max_bytes: usize,
) -> Result<PreparedOcrCrop> {
    if let Some(original) = source_ocr_crop(
        "full",
        source_bytes,
        source_mime,
        image.width(),
        image.height(),
    ) {
        return Ok(original);
    }
    encode_ocr_crop("full", image.clone(), max_bytes)
}

fn source_ocr_crop(
    label: &str,
    source_bytes: &[u8],
    source_mime: Option<&str>,
    width: u32,
    height: u32,
) -> Option<PreparedOcrCrop> {
    let mime = source_mime?;
    if source_bytes.len() > OCR_CROP_MAX_BYTES || !ocr_accepts_source_mime(mime) {
        return None;
    }
    Some(PreparedOcrCrop {
        label: label.to_owned(),
        width,
        height,
        mime: mime.to_owned(),
        bytes: source_bytes.to_vec(),
    })
}

fn ocr_accepts_source_mime(mime: &str) -> bool {
    matches!(mime, "image/jpeg" | "image/png" | "image/webp")
}

fn rect_covers_most_image(
    rect: (u32, u32, u32, u32),
    image_w: u32,
    image_h: u32,
    percent: u64,
) -> bool {
    let rect_area = rect.2 as u64 * rect.3 as u64;
    let image_area = image_w as u64 * image_h as u64;
    image_area > 0 && rect_area.saturating_mul(100) >= image_area.saturating_mul(percent)
}

fn constrain_ocr_crop_dimensions(crop: DynamicImage) -> DynamicImage {
    if crop.width().max(crop.height()) <= OCR_CROP_MAX_DIMENSION {
        return crop;
    }
    let (width, height) = fit_dimensions(
        crop.width(),
        crop.height(),
        OCR_CROP_MAX_DIMENSION,
        OCR_CROP_MAX_DIMENSION,
    );
    resize_dynamic_rgb(&crop, width, height)
}

fn encode_jpeg(image: &DynamicImage, quality: u8) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut bytes, quality);
    encoder.encode_image(image)?;
    Ok(bytes)
}

fn select_local_anchors_with_scorer(
    image: &GrayImage,
    config: &MatchConfig,
    scorer: &TileScorer<'_>,
) -> Vec<ImageAnchor> {
    if config.local_anchor_count == 0 {
        return Vec::new();
    }

    let keypoints = detect_orb_keypoints(image, config, scorer, config.local_anchor_count);
    let descriptor_patterns = brief_patterns(orb_descriptor_radius(config));
    let mut selected = Vec::new();
    let mut used_regions = [false; 64];
    for pass in 0..2 {
        for keypoint in &keypoints {
            if selected.len() >= config.local_anchor_count {
                break;
            }
            let region_index = keypoint.region as usize;
            if pass == 0 && used_regions[region_index] {
                continue;
            }
            used_regions[region_index] = true;
            let patch = keypoint.patch;
            let (pos_x, pos_y) = normalized_tile_position(
                image.width(),
                image.height(),
                patch.x,
                patch.y,
                patch.w,
                patch.h,
            );
            let descriptor = scorer.orb_descriptor_hash(
                keypoint.x,
                keypoint.y,
                patch.radius,
                &descriptor_patterns,
            );
            selected.push(ImageAnchor {
                id: format!("a{:02}", selected.len() + 1),
                x: patch.x,
                y: patch.y,
                w: patch.w,
                h: patch.h,
                pos_x,
                pos_y,
                hash: format!("{:016x}", descriptor.0),
                hash2: format!("{:016x}", descriptor.1),
                luma_mean: keypoint.features.luma_mean,
                luma_std: keypoint.features.luma_std,
                edge_density: keypoint.features.edge_density,
                kind: "orb_fast_brief".to_owned(),
                region: keypoint.region,
                max_distance: config.local_anchor_max_distance,
            });
        }
    }

    selected
}

fn scan_local_hashes_with_base_scorer(
    image: &GrayImage,
    config: &MatchConfig,
    base_scorer: &TileScorer<'_>,
) -> Vec<LocalImageHash> {
    if config.local_hash_cap == 0 {
        return Vec::new();
    }
    let descriptor_patterns = brief_patterns(orb_descriptor_radius(config));
    detect_orb_keypoints(image, config, base_scorer, config.local_hash_cap)
        .into_iter()
        .map(|keypoint| {
            let patch = keypoint.patch;
            let (pos_x, pos_y) = normalized_tile_position(
                image.width(),
                image.height(),
                patch.x,
                patch.y,
                patch.w,
                patch.h,
            );
            let descriptor = base_scorer.orb_descriptor_hash(
                keypoint.x,
                keypoint.y,
                patch.radius,
                &descriptor_patterns,
            );
            LocalImageHash {
                x: patch.x,
                y: patch.y,
                w: patch.w,
                h: patch.h,
                pos_x,
                pos_y,
                region: keypoint.region,
                hash: descriptor.0,
                hash2: descriptor.1,
                luma_mean: keypoint.features.luma_mean,
                luma_std: keypoint.features.luma_std,
                edge_density: keypoint.features.edge_density,
                scale_percent: 100,
                rotation_degrees: 0,
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct PatchRect {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    radius: i32,
}

#[derive(Debug, Clone, Copy)]
struct OrbKeypoint {
    x: u32,
    y: u32,
    patch: PatchRect,
    features: TileFeatures,
    region: u32,
    score: u32,
}

fn detect_orb_keypoints(
    image: &GrayImage,
    config: &MatchConfig,
    scorer: &TileScorer<'_>,
    limit: usize,
) -> Vec<OrbKeypoint> {
    if limit == 0 {
        return Vec::new();
    }
    let radius = orb_descriptor_radius(config);
    let border = radius.saturating_add(FAST_RADIUS);
    let width = image.width();
    let height = image.height();
    let border_u32 = u32::try_from(border).unwrap_or(u32::MAX);
    if width <= border_u32.saturating_mul(2) || height <= border_u32.saturating_mul(2) {
        return Vec::new();
    }
    let width_i32 = i32::try_from(width).unwrap_or(i32::MAX);
    let height_i32 = i32::try_from(height).unwrap_or(i32::MAX);

    let scan_width = width_i32.saturating_sub(border.saturating_mul(2)).max(0) as usize;
    let scan_height = height_i32.saturating_sub(border.saturating_mul(2)).max(0) as usize;
    let scan_budget = config
        .local_tile_budget
        .saturating_mul(16)
        .max(limit.saturating_mul(8))
        .max(1);
    let scan_step = i32::try_from(grid_sample_step(
        scan_width.saturating_mul(scan_height),
        scan_budget,
    ))
    .unwrap_or(1)
    .clamp(1, 4);
    let mut candidates = Vec::new();
    let max_candidates = limit.saturating_mul(16).max(limit);
    for y in (border..(height_i32 - border)).step_by(scan_step as usize) {
        for x in (border..(width_i32 - border)).step_by(scan_step as usize) {
            let Some(corner_score) = scorer.fast_corner_score_in_bounds(x, y) else {
                continue;
            };
            let x_u32 = u32::try_from(x).unwrap_or(0);
            let y_u32 = u32::try_from(y).unwrap_or(0);
            let patch = patch_rect_for_keypoint(width, height, x_u32, y_u32, radius);
            let Some(tile_score) = scorer.basic_score(patch.x, patch.y, patch.w, patch.h) else {
                continue;
            };
            let features = tile_score.features();
            let region = tile_region(width, height, patch.x, patch.y);
            let score = corner_score
                .saturating_add(tile_score.candidate_score())
                .saturating_add(u32::from(features.edge_density));
            candidates.push(OrbKeypoint {
                x: x_u32,
                y: y_u32,
                patch,
                features,
                region,
                score,
            });
        }
    }

    if candidates.len() > max_candidates {
        candidates
            .select_nth_unstable_by(max_candidates, |left, right| right.score.cmp(&left.score));
        candidates.truncate(max_candidates);
    }
    candidates.sort_unstable_by_key(|keypoint| std::cmp::Reverse(keypoint.score));
    suppress_keypoints(
        candidates,
        limit,
        config.local_stride.clamp(4, 24),
        width,
        height,
    )
}

fn suppress_keypoints(
    candidates: Vec<OrbKeypoint>,
    limit: usize,
    min_distance: u32,
    width: u32,
    height: u32,
) -> Vec<OrbKeypoint> {
    let cell = min_distance.max(1) as usize;
    let cols = (width as usize).div_ceil(cell).max(1);
    let rows = (height as usize).div_ceil(cell).max(1);
    let mut cell_heads = vec![usize::MAX; cols.saturating_mul(rows)];
    let mut next_in_cell = Vec::with_capacity(limit);
    let mut selected = Vec::with_capacity(limit);
    let min_distance_sq = min_distance.saturating_mul(min_distance);

    for keypoint in candidates {
        if selected.len() >= limit {
            break;
        }
        let cell_x = (keypoint.x as usize / cell).min(cols - 1);
        let cell_y = (keypoint.y as usize / cell).min(rows - 1);
        let x0 = cell_x.saturating_sub(1);
        let y0 = cell_y.saturating_sub(1);
        let x1 = (cell_x + 1).min(cols - 1);
        let y1 = (cell_y + 1).min(rows - 1);
        let mut too_close = false;
        'neighbors: for yy in y0..=y1 {
            for xx in x0..=x1 {
                let mut index = cell_heads[yy * cols + xx];
                while index != usize::MAX {
                    let existing: OrbKeypoint = selected[index];
                    if point_distance_sq(keypoint.x, keypoint.y, existing.x, existing.y)
                        < min_distance_sq
                    {
                        too_close = true;
                        break 'neighbors;
                    }
                    index = next_in_cell[index];
                }
            }
        }
        if too_close {
            continue;
        }
        let index = selected.len();
        let cell_index = cell_y * cols + cell_x;
        selected.push(keypoint);
        next_in_cell.push(cell_heads[cell_index]);
        cell_heads[cell_index] = index;
    }

    selected
}

fn point_distance_sq(ax: u32, ay: u32, bx: u32, by: u32) -> u32 {
    let dx = ax.abs_diff(bx);
    let dy = ay.abs_diff(by);
    dx.saturating_mul(dx).saturating_add(dy.saturating_mul(dy))
}

fn patch_rect_for_keypoint(width: u32, height: u32, x: u32, y: u32, radius: i32) -> PatchRect {
    let radius_u32 = radius.max(1) as u32;
    let size = radius_u32.saturating_mul(2).saturating_add(1);
    let max_x = width.saturating_sub(size);
    let max_y = height.saturating_sub(size);
    PatchRect {
        x: x.saturating_sub(radius_u32).min(max_x),
        y: y.saturating_sub(radius_u32).min(max_y),
        w: size.min(width),
        h: size.min(height),
        radius,
    }
}

fn orb_descriptor_radius(config: &MatchConfig) -> i32 {
    let patch_size = config
        .local_tile_width
        .min(config.local_tile_height)
        .clamp(24, 64);
    i32::try_from((patch_size / 2).clamp(12, 32)).unwrap_or(32)
}

const FAST_RADIUS: i32 = 3;
const FAST_THRESHOLD: i16 = 16;
const FAST_MIN_ARC: u32 = 9;
const FAST_CIRCLE: [(i32, i32); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];

#[derive(Debug)]
struct TileScorer<'a> {
    image: &'a GrayImage,
    width: usize,
    height: usize,
    stride: usize,
    sum: Vec<u32>,
    sum_sq: Vec<u64>,
    edges: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct BasicTileScore {
    mean: f32,
    contrast: f32,
    edge_density: f32,
}

#[derive(Debug, Clone, Copy)]
struct TileFeatures {
    luma_mean: u8,
    luma_std: u8,
    edge_density: u8,
}

impl<'a> TileScorer<'a> {
    fn new(image: &'a GrayImage) -> Self {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let stride = width + 1;
        let len = stride * (height + 1);
        let mut sum = vec![0u32; len];
        let mut sum_sq = vec![0u64; len];
        let mut edges = vec![0u32; len];
        let pixels = image.as_raw();

        for y in 0..height {
            let row_start = y * width;
            let next_row_start = row_start + width;
            let row_pixels = &pixels[row_start..row_start + width];
            let below_pixels =
                (y + 1 < height).then(|| &pixels[next_row_start..next_row_start + width]);
            let above_start = y * stride;
            let out_start = (y + 1) * stride;
            let mut row_sum = 0u32;
            let mut row_sum_sq = 0u64;
            let mut row_edges = 0u32;
            for x in 0..width {
                let value = u32::from(row_pixels[x]);
                row_sum += value;
                row_sum_sq += value as u64 * value as u64;

                if x + 1 < width {
                    let right = i16::from(row_pixels[x + 1]);
                    row_edges += ((value as i16 - right).unsigned_abs() > 30) as u32;
                }
                if let Some(below_pixels) = below_pixels {
                    let down = i16::from(below_pixels[x]);
                    row_edges += ((value as i16 - down).unsigned_abs() > 30) as u32;
                }

                let out = out_start + x + 1;
                let above = above_start + x + 1;
                sum[out] = sum[above] + row_sum;
                sum_sq[out] = sum_sq[above] + row_sum_sq;
                edges[out] = edges[above] + row_edges;
            }
        }

        Self {
            image,
            width,
            height,
            stride,
            sum,
            sum_sq,
            edges,
        }
    }

    fn orb_descriptor_hash(
        &self,
        center_x: u32,
        center_y: u32,
        radius: i32,
        patterns: &BriefPatterns,
    ) -> (u64, u64) {
        let center_x = i32::try_from(center_x).unwrap_or(i32::MAX);
        let center_y = i32::try_from(center_y).unwrap_or(i32::MAX);
        let (sin, cos) = if self.patch_is_in_bounds(center_x, center_y, radius) {
            self.patch_orientation_in_bounds(center_x, center_y, radius)
        } else {
            self.patch_orientation(center_x, center_y, radius)
        };

        (
            self.brief_hash(center_x, center_y, sin, cos, &patterns.first),
            self.brief_hash(center_x, center_y, sin, cos, &patterns.second),
        )
    }

    fn brief_hash(
        &self,
        center_x: i32,
        center_y: i32,
        sin: f32,
        cos: f32,
        pattern: &[(i32, i32, i32, i32); 64],
    ) -> u64 {
        let mut hash = 0u64;
        for &(ax, ay, bx, by) in pattern {
            let left = self.sample_rotated(center_x, center_y, ax, ay, sin, cos);
            let right = self.sample_rotated(center_x, center_y, bx, by, sin, cos);
            hash = (hash << 1) | u64::from(left > right);
        }
        hash
    }

    fn fast_corner_score_in_bounds(&self, x: i32, y: i32) -> Option<u32> {
        debug_assert!(x >= FAST_RADIUS);
        debug_assert!(y >= FAST_RADIUS);
        debug_assert!((x as usize) + (FAST_RADIUS as usize) < self.width);
        debug_assert!((y as usize) + (FAST_RADIUS as usize) < self.height);

        let center = i16::from(self.sample_in_bounds(x, y));
        let high = center.saturating_add(FAST_THRESHOLD);
        let low = center.saturating_sub(FAST_THRESHOLD);
        let mut bright_mask = 0u32;
        let mut dark_mask = 0u32;
        let mut score = 0u32;
        for (index, (dx, dy)) in FAST_CIRCLE.iter().enumerate() {
            let value = i16::from(self.sample_in_bounds(x + dx, y + dy));
            if value >= high {
                bright_mask |= 1 << index;
            } else if value <= low {
                dark_mask |= 1 << index;
            }
            score = score.saturating_add(u32::from(center.abs_diff(value)));
        }

        (has_contiguous_fast_arc(bright_mask) || has_contiguous_fast_arc(dark_mask))
            .then_some(score)
    }

    fn patch_is_in_bounds(&self, center_x: i32, center_y: i32, radius: i32) -> bool {
        center_x >= radius
            && center_y >= radius
            && (center_x as usize + radius as usize) < self.width
            && (center_y as usize + radius as usize) < self.height
    }

    fn patch_orientation_in_bounds(&self, center_x: i32, center_y: i32, radius: i32) -> (f32, f32) {
        #[cfg(feature = "nightly-simd")]
        {
            self.patch_orientation_in_bounds_simd(center_x, center_y, radius)
        }
        #[cfg(not(feature = "nightly-simd"))]
        {
            self.patch_orientation_in_bounds_scalar(center_x, center_y, radius)
        }
    }

    #[cfg(not(feature = "nightly-simd"))]
    fn patch_orientation_in_bounds_scalar(
        &self,
        center_x: i32,
        center_y: i32,
        radius: i32,
    ) -> (f32, f32) {
        let mut moment_x = 0i64;
        let mut moment_y = 0i64;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let value = i64::from(self.sample_in_bounds(center_x + dx, center_y + dy));
                moment_x += i64::from(dx) * value;
                moment_y += i64::from(dy) * value;
            }
        }
        moments_to_orientation(moment_x, moment_y)
    }

    #[cfg(feature = "nightly-simd")]
    fn patch_orientation_in_bounds_simd(
        &self,
        center_x: i32,
        center_y: i32,
        radius: i32,
    ) -> (f32, f32) {
        use std::simd::{
            Simd,
            num::{SimdInt, SimdUint},
        };

        const LANES: usize = 16;
        const LANES_I32: i32 = 16;
        let size = (radius.saturating_mul(2).saturating_add(1)) as usize;
        let start_x = (center_x - radius) as usize;
        let pixels = self.image.as_raw();
        let mut moment_x = 0i64;
        let mut moment_y = 0i64;
        for dy in -radius..=radius {
            let row_start = (center_y + dy) as usize * self.width + start_x;
            let row = &pixels[row_start..row_start + size];
            let (chunks, remainder) = row.as_chunks::<LANES>();
            let mut dx_base = -radius;
            for chunk in chunks {
                let values = Simd::<u8, LANES>::from_array(*chunk).cast::<i32>();
                let dxs = Simd::<i32, LANES>::from_array(std::array::from_fn(|index| {
                    dx_base + i32::try_from(index).unwrap_or(0)
                }));
                moment_x += i64::from((values * dxs).reduce_sum());
                moment_y += i64::from(dy) * i64::from(values.reduce_sum());
                dx_base += LANES_I32;
            }
            for (index, &value) in remainder.iter().enumerate() {
                let dx = dx_base + i32::try_from(index).unwrap_or(0);
                let value = i64::from(value);
                moment_x += i64::from(dx) * value;
                moment_y += i64::from(dy) * value;
            }
        }
        moments_to_orientation(moment_x, moment_y)
    }

    fn patch_orientation(&self, center_x: i32, center_y: i32, radius: i32) -> (f32, f32) {
        let mut moment_x = 0i64;
        let mut moment_y = 0i64;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let value = i64::from(self.sample_clamped(center_x + dx, center_y + dy));
                moment_x += i64::from(dx) * value;
                moment_y += i64::from(dy) * value;
            }
        }
        moments_to_orientation(moment_x, moment_y)
    }

    fn sample_rotated(
        &self,
        center_x: i32,
        center_y: i32,
        dx: i32,
        dy: i32,
        sin: f32,
        cos: f32,
    ) -> u8 {
        let x = center_x + cos.mul_add(dx as f32, -sin * dy as f32).round() as i32;
        let y = center_y + sin.mul_add(dx as f32, cos * dy as f32).round() as i32;
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            return self.sample_in_bounds(x, y);
        }
        self.sample_clamped(x, y)
    }

    fn sample_clamped(&self, x: i32, y: i32) -> u8 {
        let width = i32::try_from(self.width).unwrap_or(i32::MAX);
        let height = i32::try_from(self.height).unwrap_or(i32::MAX);
        let x = x.clamp(0, width.saturating_sub(1)) as usize;
        let y = y.clamp(0, height.saturating_sub(1)) as usize;
        self.image.as_raw()[y * self.width + x]
    }

    fn sample_in_bounds(&self, x: i32, y: i32) -> u8 {
        debug_assert!(x >= 0);
        debug_assert!(y >= 0);
        debug_assert!((x as usize) < self.width);
        debug_assert!((y as usize) < self.height);
        self.image.as_raw()[y as usize * self.width + x as usize]
    }

    fn features(&self, x: u32, y: u32, w: u32, h: u32) -> Option<TileFeatures> {
        self.basic_score(x, y, w, h).map(BasicTileScore::features)
    }

    fn basic_score(&self, x: u32, y: u32, w: u32, h: u32) -> Option<BasicTileScore> {
        let x = x as usize;
        let y = y as usize;
        let w = w as usize;
        let h = h as usize;
        let area = (w * h) as f32;
        let sum = rect_sum_u32(&self.sum, self.stride, x, y, w, h) as f32;
        let mean = sum / area;
        if !(20.0..=235.0).contains(&mean) {
            return None;
        }

        let sum_sq = rect_sum_u64(&self.sum_sq, self.stride, x, y, w, h) as f32;
        let variance = (sum_sq / area) - (mean * mean);
        if variance < 196.0 {
            return None;
        }
        let contrast = variance.max(0.0).sqrt();

        let edges = rect_sum_u32(&self.edges, self.stride, x, y, w, h);
        let edge_density = edges as f32 / area;
        if !(0.02..=0.85).contains(&edge_density) {
            return None;
        }

        Some(BasicTileScore {
            mean,
            contrast,
            edge_density,
        })
    }
}

fn moments_to_orientation(moment_x: i64, moment_y: i64) -> (f32, f32) {
    if moment_x == 0 && moment_y == 0 {
        return (0.0, 1.0);
    }
    (moment_y as f32).atan2(moment_x as f32).sin_cos()
}

fn brief_offset(bit: u32, salt: u32, radius: i32) -> (i32, i32) {
    let mut value = bit.wrapping_mul(0x9e37_79b1).rotate_left((bit % 17) + 5) ^ salt;
    value ^= value >> 16;
    value = value.wrapping_mul(0x85eb_ca6b);
    value ^= value >> 13;
    let span = radius.saturating_mul(2).saturating_add(1).max(1) as u32;
    let dx = i32::try_from(value % span).unwrap_or(0) - radius;
    let dy = i32::try_from((value / span) % span).unwrap_or(0) - radius;
    (dx, dy)
}

#[derive(Debug, Clone, Copy)]
struct BriefPatterns {
    first: [(i32, i32, i32, i32); 64],
    second: [(i32, i32, i32, i32); 64],
}

fn brief_patterns(radius: i32) -> BriefPatterns {
    BriefPatterns {
        first: brief_pattern(radius, 0x243f_6a88, 0xb7e1_5162),
        second: brief_pattern(radius, 0x9e37_79b9, 0x85eb_ca6b),
    }
}

fn brief_pattern(radius: i32, first_salt: u32, second_salt: u32) -> [(i32, i32, i32, i32); 64] {
    std::array::from_fn(|index| {
        let bit = index as u32;
        let (ax, ay) = brief_offset(bit, first_salt, radius);
        let (bx, by) = brief_offset(bit, second_salt, radius);
        (ax, ay, bx, by)
    })
}

impl BasicTileScore {
    fn features(self) -> TileFeatures {
        TileFeatures {
            luma_mean: self.mean.round().clamp(0.0, 255.0) as u8,
            luma_std: self.contrast.round().clamp(0.0, 255.0) as u8,
            edge_density: (self.edge_density * 255.0).round().clamp(0.0, 255.0) as u8,
        }
    }

    fn candidate_value(self) -> f32 {
        self.contrast + self.edge_density * 120.0
    }

    fn candidate_score(self) -> u32 {
        self.candidate_value().round().max(0.0) as u32
    }
}

fn has_contiguous_fast_arc(mask: u32) -> bool {
    let doubled = mask | (mask << 16);
    let mut run = 0u32;
    for index in 0..32 {
        if (doubled & (1 << index)) == 0 {
            run = 0;
        } else {
            run += 1;
            if run >= FAST_MIN_ARC {
                return true;
            }
        }
    }
    false
}

fn rect_sum_u64(data: &[u64], stride: usize, x: usize, y: usize, w: usize, h: usize) -> u64 {
    if w == 0 || h == 0 {
        return 0;
    }

    let x2 = x + w;
    let y2 = y + h;
    data[y2 * stride + x2] + data[y * stride + x] - data[y * stride + x2] - data[y2 * stride + x]
}

fn rect_sum_u32(data: &[u32], stride: usize, x: usize, y: usize, w: usize, h: usize) -> u32 {
    if w == 0 || h == 0 {
        return 0;
    }

    let x2 = x + w;
    let y2 = y + h;
    data[y2 * stride + x2] + data[y * stride + x] - data[y * stride + x2] - data[y2 * stride + x]
}

fn resize_luma(image: &GrayImage, width: u32, height: u32) -> GrayImage {
    use fast_image_resize::{
        FilterType, PixelType, ResizeAlg, ResizeOptions,
        images::{Image as FastImage, ImageRef as FastImageRef},
    };

    let Ok(source) =
        FastImageRef::new(image.width(), image.height(), image.as_raw(), PixelType::U8)
    else {
        return imageops::resize(image, width, height, imageops::FilterType::Triangle);
    };
    let mut destination = FastImage::new(width, height, PixelType::U8);
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));

    let resize_result = IMAGE_RESIZER.with(|cell| {
        cell.borrow_mut()
            .resize(&source, &mut destination, &options)
    });
    if resize_result.is_err() {
        return imageops::resize(image, width, height, imageops::FilterType::Triangle);
    }

    GrayImage::from_raw(width, height, destination.into_vec())
        .unwrap_or_else(|| imageops::resize(image, width, height, imageops::FilterType::Triangle))
}

fn resize_rgb(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    use fast_image_resize::{
        FilterType, PixelType, ResizeAlg, ResizeOptions,
        images::{Image as FastImage, ImageRef as FastImageRef},
    };

    let Ok(source) = FastImageRef::new(
        image.width(),
        image.height(),
        image.as_raw(),
        PixelType::U8x3,
    ) else {
        return imageops::resize(image, width, height, imageops::FilterType::Triangle);
    };
    let mut destination = FastImage::new(width, height, PixelType::U8x3);
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));

    let resize_result = IMAGE_RESIZER.with(|cell| {
        cell.borrow_mut()
            .resize(&source, &mut destination, &options)
    });
    if resize_result.is_err() {
        return imageops::resize(image, width, height, imageops::FilterType::Triangle);
    }

    RgbImage::from_raw(width, height, destination.into_vec())
        .unwrap_or_else(|| imageops::resize(image, width, height, imageops::FilterType::Triangle))
}

fn resize_dynamic_rgb(image: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    let rgb = image.to_rgb8();
    if image.width() == width && image.height() == height {
        return DynamicImage::ImageRgb8(rgb);
    }
    DynamicImage::ImageRgb8(resize_rgb(&rgb, width, height))
}

fn thumbnail_rgb(image: &DynamicImage, max_dimension: u32) -> RgbImage {
    let (width, height) = image.dimensions();
    let (target_width, target_height) = fit_dimensions(width, height, max_dimension, max_dimension);
    let rgb = image.to_rgb8();
    if target_width == width && target_height == height {
        return rgb;
    }
    resize_rgb(&rgb, target_width, target_height)
}

fn fit_dimensions(width: u32, height: u32, max_width: u32, max_height: u32) -> (u32, u32) {
    if width <= max_width && height <= max_height {
        return (width.max(1), height.max(1));
    }
    let width_ratio = max_width as f64 / width.max(1) as f64;
    let height_ratio = max_height as f64 / height.max(1) as f64;
    let ratio = width_ratio.min(height_ratio).min(1.0);
    (
        ((width as f64 * ratio).round() as u32).max(1),
        ((height as f64 * ratio).round() as u32).max(1),
    )
}

fn tile_region(width: u32, height: u32, x: u32, y: u32) -> u32 {
    let col = x.saturating_mul(8).checked_div(width).unwrap_or(0).min(7);
    let row = y.saturating_mul(8).checked_div(height).unwrap_or(0).min(7);
    row * 8 + col
}

fn normalized_tile_position(width: u32, height: u32, x: u32, y: u32, w: u32, h: u32) -> (u8, u8) {
    let center_x = x.saturating_add(w / 2);
    let center_y = y.saturating_add(h / 2);
    let pos_x = center_x
        .saturating_mul(255)
        .checked_div(width.max(1))
        .unwrap_or(0)
        .min(255) as u8;
    let pos_y = center_y
        .saturating_mul(255)
        .checked_div(height.max(1))
        .unwrap_or(0)
        .min(255) as u8;
    (pos_x, pos_y)
}

fn validate_url(raw: &str) -> Result<()> {
    let url = Url::parse(raw).context("parsing image url")?;
    match url.scheme() {
        "https" => {}
        _ => bail!("only https image urls are supported"),
    }

    let Some(host) = url.host_str() else {
        bail!("url host missing");
    };

    if let Ok(ip) = host.parse::<IpAddr>() {
        reject_private_ip(ip)?;
    }

    if !is_discord_host(host) {
        bail!("external image urls are disabled");
    }

    Ok(())
}

fn reject_private_ip(ip: IpAddr) -> Result<()> {
    let blocked = match ip {
        IpAddr::V4(ip) => {
            ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_broadcast()
                || ip.is_documentation()
                || ip.octets()[0] == 0
        }
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
        }
    };

    if blocked {
        Err(anyhow!("private ip urls are rejected"))
    } else {
        Ok(())
    }
}

pub(crate) fn is_discord_host(host: &str) -> bool {
    matches!(
        host,
        "cdn.discordapp.com"
            | "media.discordapp.net"
            | "images-ext-1.discordapp.net"
            | "images-ext-2.discordapp.net"
            | "media.discordapp.com"
    )
}

pub fn log_image_result(
    message_id: u64,
    kind: CandidateKind,
    url: &str,
    success: bool,
    elapsed_ms: u128,
    reason: Option<&str>,
) {
    let image_url = url_log_label(url);
    if success {
        info!(
            event = "image.processed",
            message_id,
            ?kind,
            image_url,
            elapsed_ms,
            "image processed"
        );
    } else {
        warn!(
            event = "image.skipped",
            message_id,
            ?kind,
            image_url,
            elapsed_ms,
            reason,
            "image skipped"
        );
    }
}

pub fn url_log_label(raw: &str) -> String {
    let Ok(url) = Url::parse(raw) else {
        return "invalid-url".to_owned();
    };
    if url.host_str().is_some_and(is_discord_host) {
        return raw.to_owned();
    }

    let Some(host) = url.host_str() else {
        return "unknown-host".to_owned();
    };
    let mut redacted = format!("{}://{}{}", url.scheme(), host, url.path());
    if redacted.ends_with('/') {
        redacted.push_str("image");
    }
    redacted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::types::{CandidateKind, ImageCandidate};
    use image::{ImageBuffer, ImageFormat, Rgb};
    use std::io::Cursor;
    use twilight_model::id::Id;

    #[tokio::test]
    async fn cpu_gate_prioritizes_high_priority_waiters() {
        let gate = Arc::new(CpuGate::new(1));
        let first_permit = gate.acquire_low_priority().await.unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        let low_gate = Arc::clone(&gate);
        let low_tx = tx.clone();
        let low = tokio::spawn(async move {
            let _permit = low_gate.acquire_low_priority().await.unwrap();
            low_tx.send("low").unwrap();
        });
        tokio::task::yield_now().await;

        let high_gate = Arc::clone(&gate);
        let high = tokio::spawn(async move {
            let _permit = high_gate.acquire_high_priority().await.unwrap();
            tx.send("high").unwrap();
        });
        tokio::task::yield_now().await;

        drop(first_permit);

        let first = tokio::time::timeout(Duration::from_secs(1), rx.recv())
            .await
            .unwrap();
        assert_eq!(first, Some("high"));
        let second = tokio::time::timeout(Duration::from_secs(1), rx.recv())
            .await
            .unwrap();
        assert_eq!(second, Some("low"));

        high.await.unwrap();
        low.await.unwrap();
    }

    fn preview_candidate() -> ImageCandidate {
        ImageCandidate {
            guild_id: Id::new(1),
            channel_id: Id::new(2),
            message_id: Id::new(3),
            candidate_index: 1,
            candidates_in_message: 1,
            author_id: Id::new(4),
            author_username: None,
            author_global_name: None,
            url: "https://cdn.discordapp.com/attachments/1/2/image.png?ex=abc&width=999".to_owned(),
            proxy_url: Some(
                "https://media.discordapp.net/attachments/1/2/image.png?ex=abc&width=999"
                    .to_owned(),
            ),
            kind: CandidateKind::Attachment,
            mime_hint: Some("image/png".to_owned()),
            size_bytes: Some(2 * 1024 * 1024),
            metadata_width: Some(1600),
            metadata_height: Some(2400),
            media_flags: None,
            verify_only: false,
            enqueued_at: None,
        }
    }

    #[test]
    fn preview_request_preserves_signed_params_and_replaces_size() {
        let candidate = preview_candidate();
        let download = DownloadConfig::default();
        let matching = MatchConfig::default();

        let preview = build_discord_preview_request(&candidate, &download, &matching).unwrap();

        assert_eq!(preview.width, 724);
        assert_eq!(preview.height, 1086);
        let url = Url::parse(&preview.url).unwrap();
        let query = url.query_pairs().collect::<Vec<_>>();
        assert!(query.contains(&("ex".into(), "abc".into())));
        assert_eq!(
            query
                .iter()
                .filter(|(key, _)| key == "width")
                .map(|(_, value)| value.as_ref())
                .collect::<Vec<_>>(),
            vec!["724"]
        );
        assert_eq!(
            query
                .iter()
                .filter(|(key, _)| key == "height")
                .map(|(_, value)| value.as_ref())
                .collect::<Vec<_>>(),
            vec!["1086"]
        );
    }

    #[test]
    fn url_log_label_preserves_full_discord_cdn_urls() {
        let raw = "https://cdn.discordapp.com/attachments/1/2/image.png?ex=abc&is=def&hm=ghi";
        assert_eq!(url_log_label(raw), raw);
    }

    #[test]
    fn url_log_label_strips_query_for_non_discord_hosts() {
        assert_eq!(
            url_log_label("https://example.com/path/image.png?secret=value"),
            "https://example.com/path/image.png"
        );
    }

    #[test]
    fn download_retry_delay_uses_exponential_backoff() {
        let mut config = DownloadConfig {
            retry_base_delay_ms: 150,
            ..DownloadConfig::default()
        };

        assert_eq!(download_retry_delay(&config, 0), Duration::from_millis(150));
        assert_eq!(download_retry_delay(&config, 1), Duration::from_millis(300));
        assert_eq!(download_retry_delay(&config, 2), Duration::from_millis(600));

        config.retry_base_delay_ms = 0;
        assert_eq!(download_retry_delay(&config, 2), Duration::ZERO);
    }

    #[test]
    fn preview_request_skips_already_normalized_images() {
        let mut candidate = preview_candidate();
        candidate.url = "https://cdn.discordapp.com/attachments/1/2/image.png".to_owned();
        candidate.proxy_url =
            Some("https://media.discordapp.net/attachments/1/2/image.png".to_owned());
        candidate.metadata_width = Some(640);
        candidate.metadata_height = Some(800);

        assert!(
            build_discord_preview_request(
                &candidate,
                &DownloadConfig::default(),
                &MatchConfig::default(),
            )
            .is_none()
        );
    }

    #[test]
    fn preview_request_skips_small_originals() {
        let mut candidate = preview_candidate();
        candidate.size_bytes = Some(DownloadConfig::default().preview.min_original_bytes - 1);

        assert!(
            build_discord_preview_request(
                &candidate,
                &DownloadConfig::default(),
                &MatchConfig::default(),
            )
            .is_none()
        );
    }

    #[test]
    fn preview_request_rejects_metadata_prefilter_failures() {
        let mut download = DownloadConfig::default();
        let matching = MatchConfig::default();

        let mut candidate = preview_candidate();
        candidate.mime_hint = Some("text/plain".to_owned());
        assert!(build_discord_preview_request(&candidate, &download, &matching).is_none());

        let mut candidate = preview_candidate();
        candidate.size_bytes = Some(download.max_bytes as u64 + 1);
        assert!(build_discord_preview_request(&candidate, &download, &matching).is_none());

        let mut candidate = preview_candidate();
        candidate.metadata_width = Some(0);
        assert!(build_discord_preview_request(&candidate, &download, &matching).is_none());

        let mut candidate = preview_candidate();
        candidate.metadata_height = None;
        assert!(build_discord_preview_request(&candidate, &download, &matching).is_none());

        let mut candidate = preview_candidate();
        candidate.proxy_url = None;
        assert!(build_discord_preview_request(&candidate, &download, &matching).is_none());

        let mut candidate = preview_candidate();
        candidate.proxy_url = Some("https://example.com/attachments/1/2/image.png".to_owned());
        assert!(build_discord_preview_request(&candidate, &download, &matching).is_none());

        let mut candidate = preview_candidate();
        candidate.proxy_url =
            Some("http://media.discordapp.net/attachments/1/2/image.png".to_owned());
        assert!(build_discord_preview_request(&candidate, &download, &matching).is_none());

        let mut candidate = preview_candidate();
        candidate.metadata_width = Some(10_000);
        candidate.metadata_height = Some(10);
        assert!(build_discord_preview_request(&candidate, &download, &matching).is_none());

        download.max_decoded_pixels = 100;
        let candidate = preview_candidate();
        assert!(build_discord_preview_request(&candidate, &download, &matching).is_none());
    }

    #[test]
    fn source_ocr_payload_uses_original_bytes_when_acceptable() {
        let bytes = b"original-image-bytes";
        let payload = source_ocr_payload(bytes, Some("image/png"), 100, 200).unwrap();

        assert_eq!(payload.label, "full");
        assert_eq!(payload.mime, "image/png");
        assert_eq!(payload.width, 100);
        assert_eq!(payload.height, 200);
        assert_eq!(payload.bytes.as_slice(), bytes);
    }

    #[test]
    fn ocr_payload_reencodes_oversized_original_under_limit() {
        let mut image = ImageBuffer::new(900, 900);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = Rgb([
                ((x * 13 + y * 7) % 251) as u8,
                ((x * 5 + y * 17) % 253) as u8,
                ((x * 19 + y * 3) % 255) as u8,
            ]);
        }
        let mut bytes = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .unwrap();
        assert!(bytes.len() > OCR_CROP_MAX_BYTES);

        let payload =
            prepare_ocr_payload_from_bytes(&bytes, 2_000_000, &MatchConfig::default()).unwrap();

        assert!(matches!(payload.label.as_str(), "full" | "text_dense"));
        assert_eq!(payload.mime, "image/jpeg");
        assert!(payload.bytes.len() <= OCR_CROP_MAX_BYTES);
        assert!(payload.width <= 900);
        assert!(payload.height <= 900);
    }

    #[test]
    fn staged_fingerprint_matches_one_shot_candidate_fingerprint() {
        let mut image = ImageBuffer::new(320, 420);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = Rgb([
                ((x * 3 + y * 5) % 251) as u8,
                ((x * 11 + y * 7) % 241) as u8,
                ((x * 17 + y * 13) % 239) as u8,
            ]);
        }
        let mut bytes = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .unwrap();
        let config = MatchConfig::default();
        let byte_xxh128 = xxh128_hex(&bytes);

        let full = decode_and_hash_blocking_with_timings(
            &bytes,
            byte_xxh128.clone(),
            Some("image/png".to_owned()),
            12_000_000,
            &config,
            HashMode::candidate(),
        )
        .unwrap()
        .0;
        let staged = decode_tier1_blocking_with_timings(
            &bytes,
            byte_xxh128,
            Some("image/png".to_owned()),
            12_000_000,
            &config,
        )
        .unwrap();
        let completed =
            complete_staged_fingerprint_blocking(staged, &config, HashMode::candidate()).0;

        assert_eq!(completed.width, full.width);
        assert_eq!(completed.height, full.height);
        assert_eq!(completed.mime, full.mime);
        assert_eq!(completed.byte_xxh128, full.byte_xxh128);
        assert_eq!(completed.phash64, full.phash64);
        assert_eq!(completed.dhash64, full.dhash64);
        assert_eq!(
            serde_json::to_value(&completed.visual).unwrap(),
            serde_json::to_value(&full.visual).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&completed.local_anchors).unwrap(),
            serde_json::to_value(&full.local_anchors).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&completed.local_hashes).unwrap(),
            serde_json::to_value(&full.local_hashes).unwrap()
        );
    }
}
