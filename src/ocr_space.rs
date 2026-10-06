use crate::image::pipeline::PreparedOcrCrop;
use anyhow::{Context, Result, anyhow, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use chrono::{DateTime, Utc};
use futures_util::{StreamExt, future::BoxFuture};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use tokio::sync::Semaphore;
use tokio::time::{Instant, sleep, timeout_at};
use tracing::warn;

pub const OCR_SPACE_API_KEY_ENV: &str = "OCR_SPACE_API_KEY";
pub const OCR_SPACE_MAX_IMAGE_BYTES: usize = 1_000_000;
const OCR_SPACE_MAX_RESPONSE_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct OcrSpaceConfig {
    pub endpoint: String,
    pub timeout_seconds: u64,
    pub total_timeout_seconds: u64,
    pub max_retries: usize,
    pub retry_base_delay_ms: u64,
    pub language: String,
    pub scale: bool,
    pub detect_orientation: bool,
}

impl Default for OcrSpaceConfig {
    fn default() -> Self {
        Self {
            endpoint: "https://api.ocr.space/parse/image".to_owned(),
            timeout_seconds: 20,
            total_timeout_seconds: 60,
            max_retries: 3,
            retry_base_delay_ms: 750,
            language: "eng".to_owned(),
            scale: true,
            detect_orientation: true,
        }
    }
}

impl OcrSpaceConfig {
    pub fn validate(&self) -> Result<()> {
        let endpoint =
            url::Url::parse(&self.endpoint).context("ocr_space.endpoint must be a valid URL")?;
        anyhow::ensure!(
            endpoint.scheme() == "https",
            "ocr_space.endpoint must use https"
        );
        anyhow::ensure!(
            endpoint
                .host_str()
                .is_some_and(|host| !host.trim().is_empty()),
            "ocr_space.endpoint must include a host"
        );
        anyhow::ensure!(
            self.timeout_seconds > 0 && self.timeout_seconds <= 120,
            "ocr_space.timeout_seconds must be between 1 and 120"
        );
        anyhow::ensure!(
            self.total_timeout_seconds > 0 && self.total_timeout_seconds <= 60,
            "ocr_space.total_timeout_seconds must be between 1 and 60"
        );
        anyhow::ensure!(
            self.max_retries <= 5,
            "ocr_space.max_retries must be at most 5"
        );
        anyhow::ensure!(
            self.retry_base_delay_ms <= 5_000,
            "ocr_space.retry_base_delay_ms must be at most 5000"
        );
        anyhow::ensure!(
            !self.language.trim().is_empty() && self.language.len() <= 16,
            "ocr_space.language must be 1-16 bytes"
        );
        Ok(())
    }
}

#[derive(Clone)]
pub struct OcrSpaceClient {
    http: Client,
    api_key: String,
    config: OcrSpaceConfig,
    attempt_gate: Option<Arc<Semaphore>>,
}

impl OcrSpaceClient {
    pub fn new(http: Client, api_key: String, config: OcrSpaceConfig) -> Result<Self> {
        Self::with_attempt_gate(http, api_key, config, None)
    }

    pub fn with_attempt_gate(
        http: Client,
        api_key: String,
        config: OcrSpaceConfig,
        attempt_gate: Option<Arc<Semaphore>>,
    ) -> Result<Self> {
        if api_key.trim().is_empty() {
            bail!("OCR.space API key is empty");
        }
        config.validate()?;
        Ok(Self {
            http,
            api_key,
            config,
            attempt_gate,
        })
    }

    pub async fn read_crop_text(&self, crop: &PreparedOcrCrop) -> Result<OcrSpaceRead> {
        if crop.bytes.len() > OCR_SPACE_MAX_IMAGE_BYTES {
            bail!(
                "OCR crop {} bytes exceeds OCR.space 1MB limit",
                crop.bytes.len()
            );
        }

        let base64_image = ocr_base64_image(crop);
        let total_deadline =
            Instant::now() + Duration::from_secs(self.config.total_timeout_seconds);
        timeout_at(
            total_deadline,
            self.read_with_retries(&base64_image, total_deadline),
        )
        .await
        .context("OCR.space request exceeded total timeout")?
    }

    async fn read_with_retries(
        &self,
        base64_image: &str,
        total_deadline: Instant,
    ) -> Result<OcrSpaceRead> {
        let mut last_error = None;
        for attempt in 0..=self.config.max_retries {
            let attempt_permit = match &self.attempt_gate {
                Some(gate) => Some(gate.acquire().await.context("OCR attempt gate closed")?),
                None => None,
            };
            // Queueing consumes the same deadline as requests and backoff.
            let Some(attempt_timeout) = self.remaining_attempt_timeout(total_deadline) else {
                break;
            };
            let result = self.try_read_crop_text(base64_image, attempt_timeout).await;
            drop(attempt_permit);
            match result {
                Ok(read) => return Ok(read),
                Err(error) if error.retryable && attempt < self.config.max_retries => {
                    let mut delay = retry_delay(&self.config, attempt)
                        .max(error.retry_after.unwrap_or_default());
                    if let Some(remaining) = total_deadline.checked_duration_since(Instant::now()) {
                        delay = delay.min(remaining);
                    } else {
                        break;
                    }
                    warn!(
                        event = "ocr_space.retry",
                        attempt = attempt + 1,
                        retry_after_ms = delay.as_millis(),
                        reason = %error.message,
                        "retrying OCR.space request"
                    );
                    last_error = Some(format!(
                        "{} (after {} attempt(s))",
                        error.message,
                        attempt + 1
                    ));
                    sleep(delay).await;
                }
                Err(error) => {
                    return Err(anyhow!(
                        "{} (after {} attempt(s))",
                        error.message,
                        attempt + 1
                    ));
                }
            }
        }

        Err(anyhow!(
            "OCR.space request exceeded total timeout{}",
            last_error.map_or_else(String::new, |error| format!("; last error: {error}"))
        ))
    }

    fn remaining_attempt_timeout(&self, total_deadline: Instant) -> Option<Duration> {
        let remaining = total_deadline.checked_duration_since(Instant::now())?;
        let per_attempt = Duration::from_secs(self.config.timeout_seconds);
        Some(remaining.min(per_attempt))
    }

    async fn try_read_crop_text(
        &self,
        base64_image: &str,
        attempt_timeout: Duration,
    ) -> Result<OcrSpaceRead, OcrSpaceError> {
        let form = [
            ("base64Image", base64_image),
            ("language", self.config.language.as_str()),
            ("OCREngine", "2"),
            ("isOverlayRequired", "false"),
            ("scale", bool_form(self.config.scale)),
            (
                "detectOrientation",
                bool_form(self.config.detect_orientation),
            ),
        ];

        let response = self
            .http
            .post(&self.config.endpoint)
            .timeout(attempt_timeout)
            .header("apikey", &self.api_key)
            .form(&form)
            .send()
            .await
            .map_err(|error| OcrSpaceError::from_reqwest(&error))?;

        let status = response.status();
        if matches!(
            status,
            StatusCode::TOO_MANY_REQUESTS | StatusCode::REQUEST_TIMEOUT
        ) || status.is_server_error()
        {
            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(parse_retry_after);
            return Err(OcrSpaceError {
                message: format!("OCR.space returned HTTP {status}"),
                retryable: true,
                retry_after,
            });
        }
        if !status.is_success() {
            return Err(OcrSpaceError {
                message: format!("OCR.space returned HTTP {status}"),
                retryable: false,
                retry_after: None,
            });
        }

        let mut bytes = bytes::BytesMut::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| OcrSpaceError {
                message: format!("failed to read OCR.space response body: {error}"),
                retryable: error.is_body() || error.is_timeout(),
                retry_after: None,
            })?;
            if bytes.len().saturating_add(chunk.len()) > OCR_SPACE_MAX_RESPONSE_BYTES {
                return Err(OcrSpaceError {
                    message: format!(
                        "OCR.space response body exceeded {OCR_SPACE_MAX_RESPONSE_BYTES} bytes"
                    ),
                    retryable: false,
                    retry_after: None,
                });
            }
            bytes.extend_from_slice(&chunk);
        }

        let payload =
            serde_json::from_slice::<OcrSpaceResponse>(&bytes).map_err(|error| OcrSpaceError {
                message: format!("failed to decode OCR.space response: {error}"),
                retryable: false,
                retry_after: None,
            })?;
        payload.into_read()
    }
}

impl crate::image::engine::OcrClient for OcrSpaceClient {
    fn read_text<'a>(
        &'a self,
        crops: &'a [PreparedOcrCrop],
    ) -> BoxFuture<'a, Result<crate::image::engine::OcrResponse>> {
        Box::pin(async move {
            let crop = crops
                .first()
                .ok_or_else(|| anyhow!("no OCR crop was prepared"))?;
            let read = self.read_crop_text(crop).await?;
            Ok(crate::image::engine::OcrResponse {
                readable: !read.text.trim().is_empty(),
                text: read.text,
            })
        })
    }
}

fn ocr_base64_image(crop: &PreparedOcrCrop) -> String {
    let prefix = format!("data:{};base64,", crop.mime);
    let mut base64_image = String::with_capacity(
        prefix
            .len()
            .saturating_add(base64::encoded_len(crop.bytes.len(), false).unwrap_or(0)),
    );
    base64_image.push_str(&prefix);
    STANDARD.encode_string(crop.bytes.as_slice(), &mut base64_image);
    base64_image
}

#[derive(Debug, Clone, Serialize)]
pub struct OcrSpaceRead {
    pub text: String,
    pub confidence: Option<f32>,
    pub exit_code: Option<i32>,
    pub processing_time_ms: Option<u128>,
}

#[derive(Debug)]
struct OcrSpaceError {
    message: String,
    retryable: bool,
    retry_after: Option<Duration>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct OcrSpaceResponse {
    #[serde(default)]
    parsed_results: Vec<OcrSpaceParsedResult>,
    #[serde(default)]
    ocr_exit_code: Option<i32>,
    #[serde(default)]
    is_errored_on_processing: bool,
    #[serde(default, deserialize_with = "deserialize_ocr_errors")]
    error_message: Vec<String>,
    #[serde(default)]
    processing_time_in_milliseconds: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct OcrSpaceParsedResult {
    #[serde(default)]
    parsed_text: String,
    #[serde(default)]
    error_message: Option<String>,
    #[serde(default)]
    file_parse_exit_code: Option<i32>,
    #[serde(default)]
    text_orientation: Option<String>,
}

impl OcrSpaceResponse {
    fn into_read(self) -> Result<OcrSpaceRead, OcrSpaceError> {
        if self.is_errored_on_processing || !self.error_message.is_empty() {
            let message = if self.error_message.is_empty() {
                "OCR.space failed to process image".to_owned()
            } else {
                self.error_message.join("; ")
            };
            return Err(OcrSpaceError {
                retryable: response_error_is_retryable(&message),
                message,
                retry_after: None,
            });
        }

        let mut errors = Vec::new();
        let mut text = String::new();
        let mut parsed_count = 0usize;
        for result in self.parsed_results {
            if let Some(error) = result
                .error_message
                .filter(|value| !value.trim().is_empty())
            {
                errors.push(error);
            }
            if !result.parsed_text.trim().is_empty() {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(result.parsed_text.trim());
            }
            let _ = result.file_parse_exit_code;
            let _ = result.text_orientation;
            parsed_count += 1;
        }

        if !errors.is_empty() && text.trim().is_empty() {
            let message = errors.join("; ");
            return Err(OcrSpaceError {
                retryable: response_error_is_retryable(&message),
                message,
                retry_after: None,
            });
        }

        let confidence = if text.trim().is_empty() || parsed_count == 0 {
            Some(0.0)
        } else {
            Some(0.80)
        };
        Ok(OcrSpaceRead {
            text,
            confidence,
            exit_code: self.ocr_exit_code,
            processing_time_ms: self
                .processing_time_in_milliseconds
                .and_then(|value| value.parse::<u128>().ok()),
        })
    }
}

impl OcrSpaceError {
    fn from_reqwest(error: &reqwest::Error) -> Self {
        Self {
            retryable: error.is_timeout() || error.is_connect() || error.is_body(),
            message: if error.is_timeout() {
                "OCR.space request timed out".to_owned()
            } else if error.is_connect() {
                "OCR.space connection failed".to_owned()
            } else {
                "OCR.space request failed".to_owned()
            },
            retry_after: None,
        }
    }
}

fn bool_form(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn retry_delay(config: &OcrSpaceConfig, attempt: usize) -> Duration {
    let shift = u32::try_from(attempt.min(6)).unwrap_or(6);
    let multiplier = 1u64.checked_shl(shift).unwrap_or(64);
    Duration::from_millis(config.retry_base_delay_ms.saturating_mul(multiplier))
}

fn parse_retry_after(value: &str) -> Option<Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds.min(300)));
    }
    let retry_at = DateTime::parse_from_rfc2822(value)
        .ok()?
        .with_timezone(&Utc);
    let delay = retry_at.signed_duration_since(Utc::now()).to_std().ok()?;
    Some(delay.min(Duration::from_secs(300)))
}

fn response_error_is_retryable(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("rate")
        || lower.contains("quota")
        || lower.contains("timeout")
        || lower.contains("try again")
        || lower.contains("server")
}

fn deserialize_ocr_errors<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Null => Ok(Vec::new()),
        serde_json::Value::String(value) if value.trim().is_empty() => Ok(Vec::new()),
        serde_json::Value::String(value) => Ok(vec![value]),
        serde_json::Value::Array(values) => Ok(values
            .into_iter()
            .filter_map(|value| value.as_str().map(str::to_owned))
            .filter(|value| !value.trim().is_empty())
            .collect()),
        other => Err(serde::de::Error::custom(format!(
            "unexpected OCR error field: {other}"
        ))),
    }
}

pub fn load_api_key_from_env() -> Result<Option<String>> {
    match std::env::var(OCR_SPACE_API_KEY_ENV) {
        Ok(value) if !value.trim().is_empty() => Ok(Some(value)),
        Ok(_) | Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).context("reading OCR_SPACE_API_KEY"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_crop() -> PreparedOcrCrop {
        PreparedOcrCrop {
            label: "test".to_owned(),
            width: 1,
            height: 1,
            mime: "image/png".to_owned(),
            bytes: vec![1, 2, 3],
        }
    }

    async fn mock_ocr(
        responses: Vec<(u16, &'static str)>,
    ) -> (
        OcrSpaceClient,
        tokio::task::JoinHandle<Vec<Instant>>,
        tokio::sync::mpsc::UnboundedReceiver<()>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/", listener.local_addr().unwrap());
        let (requests, received_requests) = tokio::sync::mpsc::unbounded_channel();
        let server = tokio::spawn(async move {
            let mut received = Vec::new();
            for (status, headers) in responses {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    let count = socket.read(&mut buf).await.unwrap();
                    assert_ne!(count, 0);
                    request.extend_from_slice(&buf[..count]);
                    if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                        let length: usize = head
                            .lines()
                            .find_map(|line| line.strip_prefix("content-length:"))
                            .unwrap()
                            .trim()
                            .parse()
                            .unwrap();
                        if request.len() >= end + 4 + length {
                            break;
                        }
                    }
                }
                received.push(Instant::now());
                let _ = requests.send(());
                let body = r#"{"ParsedResults":[{"ParsedText":"recognized text"}],"IsErroredOnProcessing":false}"#;
                let response = format!(
                    "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n{body}",
                    body.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
            received
        });
        // Local HTTP is confined to this fixture; production configuration requires HTTPS.
        let client = OcrSpaceClient {
            http: Client::builder().no_proxy().build().unwrap(),
            api_key: "test-key".to_owned(),
            config: OcrSpaceConfig {
                endpoint,
                retry_base_delay_ms: 20,
                total_timeout_seconds: 4,
                ..OcrSpaceConfig::default()
            },
            attempt_gate: None,
        };
        (client, server, received_requests)
    }

    #[tokio::test]
    async fn retries_503_with_exponential_backoff_then_succeeds() {
        let (client, server, _) = mock_ocr(vec![(503, ""), (503, ""), (200, "")]).await;
        let read = client.read_crop_text(&test_crop()).await.unwrap();
        assert_eq!(read.text, "recognized text");
        let times = server.await.unwrap();
        assert_eq!(times.len(), 3);
        assert!(times[1] - times[0] >= Duration::from_millis(20));
        assert!(times[2] - times[1] >= Duration::from_millis(40));
    }

    #[tokio::test]
    async fn exhausted_503_reports_all_four_attempts() {
        let (client, server, _) = mock_ocr(vec![(503, ""); 4]).await;
        let error = client.read_crop_text(&test_crop()).await.unwrap_err();
        assert!(error.to_string().contains("503"));
        assert!(error.to_string().contains("after 4 attempt(s)"));
        assert_eq!(server.await.unwrap().len(), 4);
    }

    #[tokio::test]
    async fn permanent_http_error_is_not_retried() {
        let (client, server, _) = mock_ocr(vec![(403, "")]).await;
        let error = client.read_crop_text(&test_crop()).await.unwrap_err();
        assert!(error.to_string().contains("403"));
        assert!(error.to_string().contains("after 1 attempt(s)"));
        assert_eq!(server.await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn retry_after_does_not_shorten_backoff() {
        let (client, server, _) = mock_ocr(vec![(429, "Retry-After: 0\r\n"), (200, "")]).await;
        client.read_crop_text(&test_crop()).await.unwrap();
        let times = server.await.unwrap();
        assert!(times[1] - times[0] >= Duration::from_millis(20));
    }

    #[tokio::test]
    async fn honors_longer_retry_after() {
        let (client, server, _) = mock_ocr(vec![(503, "Retry-After: 1\r\n"), (200, "")]).await;
        client.read_crop_text(&test_crop()).await.unwrap();
        let times = server.await.unwrap();
        assert!(times[1] - times[0] >= Duration::from_secs(1));
    }

    #[tokio::test(start_paused = true)]
    async fn total_deadline_includes_waiting_for_attempt_gate() {
        let gate = Arc::new(Semaphore::new(0));
        let client = OcrSpaceClient::with_attempt_gate(
            Client::new(),
            "test-key".to_owned(),
            OcrSpaceConfig {
                total_timeout_seconds: 1,
                ..OcrSpaceConfig::default()
            },
            Some(Arc::clone(&gate)),
        )
        .unwrap();
        let started = Instant::now();
        let result =
            tokio::time::timeout(Duration::from_secs(2), client.read_crop_text(&test_crop()))
                .await
                .expect("client deadline must expire before test deadline");
        assert!(result.unwrap_err().to_string().contains("total timeout"));
        assert_eq!(started.elapsed(), Duration::from_secs(1));
        gate.add_permits(1);
        assert!(gate.try_acquire().is_ok());
    }

    #[tokio::test]
    async fn backoff_releases_attempt_permit() {
        let (mut client, server, mut requests) = mock_ocr(vec![(503, ""), (200, "")]).await;
        let gate = Arc::new(Semaphore::new(1));
        client.attempt_gate = Some(Arc::clone(&gate));
        client.config.retry_base_delay_ms = 200;
        let task = tokio::spawn(async move { client.read_crop_text(&test_crop()).await });
        // Wait until the server receives the first attempt before checking backoff.
        requests.recv().await.unwrap();
        let permit = tokio::time::timeout(Duration::from_millis(100), gate.acquire())
            .await
            .expect("backoff must release the HTTP permit")
            .unwrap();
        assert!(!task.is_finished());
        drop(permit);
        task.await.unwrap().unwrap();
        assert_eq!(server.await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn total_deadline_stops_long_retry_after() {
        let (mut client, server, _) = mock_ocr(vec![(503, "Retry-After: 5\r\n")]).await;
        client.config.total_timeout_seconds = 1;
        let result =
            tokio::time::timeout(Duration::from_secs(2), client.read_crop_text(&test_crop()))
                .await
                .expect("total deadline must include backoff");
        assert!(result.unwrap_err().to_string().contains("total timeout"));
        assert_eq!(server.await.unwrap().len(), 1);
    }

    #[test]
    fn five_retries_are_valid_and_use_exponential_delays() {
        let config = OcrSpaceConfig {
            max_retries: 5,
            ..OcrSpaceConfig::default()
        };
        config.validate().unwrap();
        let delays: Vec<_> = (0..config.max_retries)
            .map(|attempt| retry_delay(&config, attempt).as_millis())
            .collect();
        assert_eq!(delays, [750, 1500, 3000, 6000, 12000]);
    }

    #[tokio::test]
    async fn five_retries_make_six_attempts() {
        let (mut client, server, _) = mock_ocr(vec![(503, ""); 6]).await;
        client.config.max_retries = 5;
        let error = client.read_crop_text(&test_crop()).await.unwrap_err();
        assert!(error.to_string().contains("after 6 attempt(s)"));
        assert_eq!(server.await.unwrap().len(), 6);
    }

    #[test]
    fn attempt_timeout_is_capped_by_total_deadline() {
        let client = OcrSpaceClient::new(
            Client::new(),
            "test-key".to_owned(),
            OcrSpaceConfig {
                timeout_seconds: 20,
                total_timeout_seconds: 1,
                ..OcrSpaceConfig::default()
            },
        )
        .unwrap();

        let attempt_timeout = client
            .remaining_attempt_timeout(Instant::now() + Duration::from_millis(50))
            .unwrap();

        assert!(attempt_timeout <= Duration::from_millis(50));
    }
}
