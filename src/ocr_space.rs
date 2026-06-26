use crate::image::pipeline::PreparedOcrCrop;
use anyhow::{Context, Result, anyhow, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use chrono::{DateTime, Utc};
use futures_util::{StreamExt, future::BoxFuture};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use tokio::sync::Semaphore;
use tokio::time::{Instant, sleep};
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
            total_timeout_seconds: 30,
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
            self.max_retries <= 3,
            "ocr_space.max_retries must be at most 3"
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
        let mut last_error = None;
        for attempt in 0..=self.config.max_retries {
            let Some(attempt_timeout) = self.remaining_attempt_timeout(total_deadline) else {
                break;
            };
            let attempt_permit = match &self.attempt_gate {
                Some(gate) => Some(gate.acquire().await.context("OCR attempt gate closed")?),
                None => None,
            };
            let result = self
                .try_read_crop_text(base64_image.as_str(), attempt_timeout)
                .await;
            drop(attempt_permit);
            match result {
                Ok(read) => return Ok(read),
                Err(error) if error.retryable && attempt < self.config.max_retries => {
                    let mut delay = error
                        .retry_after
                        .unwrap_or_else(|| retry_delay(&self.config, attempt));
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
                    last_error = Some(error.message);
                    sleep(delay).await;
                }
                Err(error) => return Err(anyhow!(error.message)),
            }
        }

        Err(anyhow!(last_error.unwrap_or_else(|| {
            "OCR.space request exceeded total timeout".to_owned()
        })))
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
        if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
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
