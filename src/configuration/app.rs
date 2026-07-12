#![expect(
    clippy::struct_excessive_bools,
    clippy::too_many_lines,
    reason = "Configuration mirrors TOML shape and bounds values explicitly; clippy reports if these suppressions stop being needed."
)]

use crate::{
    configuration::guild::{ScanPolicy, TextGatePolicy},
    ocr_space::OcrSpaceConfig,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{env, fmt, fs, path::Path};
use twilight_model::id::{Id, marker::UserMarker};

pub const DECODED_MEMORY_BYTES_PER_PIXEL: u64 = 12;
pub const DECODED_MEMORY_FIXED_OVERHEAD_BYTES: u64 = 2 * 1024 * 1024;

pub const fn decoded_image_memory_reservation_bytes(pixels: u64) -> u64 {
    pixels
        .saturating_mul(DECODED_MEMORY_BYTES_PER_PIXEL)
        .saturating_add(DECODED_MEMORY_FIXED_OVERHEAD_BYTES)
}

#[derive(Clone)]
pub struct Secrets {
    pub discord_token: String,
    pub specimen_hmac_secret: String,
    pub ocr_space_api_key: Option<String>,
}

impl fmt::Debug for Secrets {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Secrets")
            .field("discord_token", &"<redacted>")
            .field("specimen_hmac_secret", &"<redacted>")
            .field(
                "ocr_space_api_key",
                &self.ocr_space_api_key.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub bot: BotConfig,
    pub queue: QueueConfig,
    pub download: DownloadConfig,
    #[serde(rename = "match")]
    pub matching: MatchConfig,
    pub text_gate: TextGatePolicy,
    pub scan: RuntimeScanConfig,
    pub commands: CommandConfig,
    pub ocr_space: OcrSpaceConfig,
    pub telemetry: TelemetryConfig,
}

impl AppConfig {
    pub fn default_scan_policy(&self) -> ScanPolicy {
        ScanPolicy {
            exempt_administrators: self.scan.exempt_administrators,
            allowed_extensions: self.scan.allowed_extensions.clone(),
            max_file_bytes: u64::try_from(self.download.max_bytes).unwrap_or(u64::MAX),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct BotConfig {
    pub user_id: Option<String>,
}

impl BotConfig {
    pub fn user_id(&self) -> Result<Option<Id<UserMarker>>> {
        self.user_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| {
                value
                    .parse::<u64>()
                    .map(Id::new)
                    .with_context(|| "bot.user_id must be a Discord snowflake")
            })
            .transpose()
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct QueueConfig {
    pub max_size: usize,
    pub enqueue_timeout_ms: u64,
    pub cpu_concurrency: usize,
    pub download_concurrency: usize,
    pub ocr_concurrency: usize,
    pub max_images_per_message: usize,
    pub hash_outcome_cache_size: usize,
    pub download_memory_max_bytes: usize,
    pub decoded_image_memory_max_bytes: usize,
    pub byte_store_max_bytes: usize,
}

impl QueueConfig {
    pub fn image_worker_concurrency(&self) -> usize {
        self.download_concurrency.max(self.cpu_concurrency)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DownloadConfig {
    pub max_bytes: usize,
    pub max_decoded_pixels: u64,
    pub timeout_seconds: u64,
    pub max_retries: usize,
    pub retry_base_delay_ms: u64,
    pub warmer_enabled: bool,
    pub warmer_period_seconds: u64,
    pub preview: PreviewDownloadConfig,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreviewDownloadMode {
    #[default]
    MatchNormalized,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PreviewDownloadConfig {
    pub enabled: bool,
    pub mode: PreviewDownloadMode,
    pub skip_animated: bool,
    pub min_original_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct MatchConfig {
    pub score_threshold: f32,
    pub suspicious_score_threshold: f32,
    pub perceptual_score_weight: f32,
    pub local_anchor_score_weight: f32,
    pub dense_local_anchor_score_weight: f32,
    pub visual_signature_score_weight: f32,
    pub visual_shape_score_weight: f32,
    pub visual_shape_score_cap: f32,
    pub perceptual_score_floor: f32,
    pub local_score_full_hits: usize,
    pub local_score_full_regions: usize,
    pub local_score_full_spread: f32,
    pub visual_shape_score_full: f32,
    pub cluster_coherence_enabled: bool,
    pub cluster_hard_score: u32,
    pub cluster_chrome_ceiling_score: u32,
    pub cluster_member_score: u32,
    pub cluster_coherence_score: u32,
    pub cluster_min_size: u16,
    pub cluster_coverage_floor_permille: u16,
    pub phash64_max_distance: u32,
    pub dhash64_max_distance: u32,
    pub perceptual_hash_max_total_distance: u32,
    pub suspicious_phash64_max_distance: u32,
    pub suspicious_dhash64_max_distance: u32,
    pub suspicious_perceptual_hash_max_total_distance: u32,
    pub suspicious_perceptual_visual_support_distance_slack: u32,
    pub perceptual_orientation_correction: bool,
    pub perceptual_orientation_max_degrees: f32,
    pub perceptual_orientation_step_degrees: f32,
    pub perceptual_orientation_min_gain: f32,
    pub local_anchors_enabled: bool,
    pub local_max_width: u32,
    pub local_max_height: u32,
    pub local_max_area: u64,
    pub local_max_aspect_ratio: f32,
    pub local_tile_width: u32,
    pub local_tile_height: u32,
    pub local_stride: u32,
    pub local_tile_budget: usize,
    pub local_hash_cap: usize,
    pub local_anchor_count: usize,
    pub local_anchor_max_distance: u32,
    pub local_min_anchor_hits: usize,
    pub local_min_distinct_regions: usize,
    pub local_max_mean_distance: f32,
    pub local_luma_candidate_max_delta: u8,
    pub local_contrast_candidate_max_delta: u8,
    pub local_edge_density_candidate_max_delta: u8,
    pub local_position_candidate_max_delta: u8,
    pub visual_luma_zero_score_delta: u8,
    pub visual_color_zero_score_delta: u8,
    pub visual_grid_luma_zero_score_delta: u8,
    pub visual_text_grid_zero_score_delta: u8,
    pub geometry_min_short_edge: u32,
    pub geometry_min_area: u64,
    pub geometry_max_aspect_ratio: f32,
    pub geometry_max_aspect_delta: f32,
    pub geometry_max_width_delta: f32,
    pub geometry_max_height_delta: f32,
    pub geometry_enable_affine: bool,
    pub geometry_enable_homography: bool,
    pub geometry_model_slack: f32,
    pub geometry_max_anisotropy: f32,
    pub geometry_max_perspective: f32,
    pub geometry_affine_min_extra_inliers: usize,
    pub geometry_affine_min_extra_regions: usize,
    pub geometry_affine_max_mean_residual: f32,
    pub geometry_homography_min_extra_inliers: usize,
    pub geometry_homography_min_extra_regions: usize,
    pub geometry_homography_max_mean_residual: f32,
    pub geometry_ratio_min_margin: u8,
    pub geometry_enable_prosac_fallback: bool,
    pub geometry_prosac_max_iters: u32,
    pub geometry_prosac_min_inliers: usize,
    pub local_suspicious_min_anchor_hits: usize,
    pub local_suspicious_min_distinct_regions: usize,
    pub local_suspicious_max_mean_distance: f32,
    pub local_suspicious_unverified_support_enabled: bool,
    pub local_suspicious_unverified_support_min_anchor_hits: usize,
    pub local_suspicious_unverified_support_min_distinct_regions: usize,
    pub local_suspicious_unverified_support_min_retention_permille: u16,
    pub local_suspicious_unverified_support_max_mean_distance: f32,
    pub local_suspicious_unverified_support_max_perceptual_total_distance: u32,
    pub local_suspicious_unverified_support_max_aspect_delta: f32,
    pub local_suspicious_unverified_support_max_dimension_delta: f32,
    pub suspicious_local_luma_candidate_max_delta: u8,
    pub suspicious_local_contrast_candidate_max_delta: u8,
    pub suspicious_local_edge_density_candidate_max_delta: u8,
    pub suspicious_local_position_candidate_max_delta: u8,
    pub suspicious_visual_luma_zero_score_delta: u8,
    pub suspicious_visual_color_zero_score_delta: u8,
    pub suspicious_visual_grid_luma_zero_score_delta: u8,
    pub suspicious_visual_text_grid_zero_score_delta: u8,
    pub suspicious_geometry_min_short_edge: u32,
    pub suspicious_geometry_min_area: u64,
    pub suspicious_geometry_max_aspect_ratio: f32,
    pub suspicious_geometry_max_aspect_delta: f32,
    pub suspicious_geometry_max_width_delta: f32,
    pub suspicious_geometry_max_height_delta: f32,
    pub visual_shape_min_middle_text_percent: u8,
    pub visual_shape_min_center_text_percent: u8,
    pub visual_shape_max_center_text_percent: u8,
    pub visual_shape_max_edge_text_percent: u8,
    pub visual_shape_min_signals: usize,
    pub visual_shape_min_text_grid_mean: u8,
    pub visual_shape_max_text_grid_mean: u8,
    pub visual_shape_min_text_regions: usize,
    pub visual_shape_min_luma_mean: u8,
    pub visual_shape_max_luma_mean: u8,
    pub visual_shape_min_luma_std: u8,
    pub visual_shape_max_luma_std: u8,
    pub visual_shape_min_local_hashes: usize,
    pub visual_shape_max_rgb_spread: u8,
    pub visual_shape_sparse_max_luma_mean: u8,
    pub visual_shape_sparse_max_text_grid_mean: u8,
    pub visual_shape_sparse_min_local_hashes: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RuntimeScanConfig {
    pub exempt_administrators: bool,
    pub allowed_extensions: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CommandConfig {
    pub register_on_startup: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct TelemetryConfig {
    pub dial9: Dial9TelemetryConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Dial9TelemetryConfig {
    pub enabled: bool,
    pub trace_dir: String,
    pub max_disk_usage_mb: u64,
    pub rotation_seconds: u64,
    pub shutdown_timeout_seconds: u64,
    pub runtime_name: String,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_size: 500,
            enqueue_timeout_ms: 50,
            cpu_concurrency: 4,
            download_concurrency: 64,
            ocr_concurrency: 1,
            max_images_per_message: 4,
            hash_outcome_cache_size: 100_000,
            download_memory_max_bytes: 128 * 1024 * 1024,
            decoded_image_memory_max_bytes: 256 * 1024 * 1024,
            byte_store_max_bytes: 64 * 1024 * 1024,
        }
    }
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            max_bytes: 10 * 1024 * 1024,
            max_decoded_pixels: 12_000_000,
            timeout_seconds: 2,
            max_retries: 1,
            retry_base_delay_ms: 150,
            warmer_enabled: true,
            warmer_period_seconds: 270,
            preview: PreviewDownloadConfig::default(),
        }
    }
}

impl Default for PreviewDownloadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: PreviewDownloadMode::MatchNormalized,
            skip_animated: true,
            min_original_bytes: 256 * 1024,
        }
    }
}

impl Default for RuntimeScanConfig {
    fn default() -> Self {
        Self {
            exempt_administrators: true,
            allowed_extensions: vec!["jpg".to_owned()],
        }
    }
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            score_threshold: 63.0,
            suspicious_score_threshold: 20.0,
            perceptual_score_weight: 1.8,
            local_anchor_score_weight: 2.2,
            dense_local_anchor_score_weight: 1.4,
            visual_signature_score_weight: 0.0,
            visual_shape_score_weight: 1.0,
            visual_shape_score_cap: 10.0,
            perceptual_score_floor: 35.0,
            local_score_full_hits: 500,
            local_score_full_regions: 14,
            local_score_full_spread: 700.0,
            visual_shape_score_full: 300.0,
            cluster_coherence_enabled: true,
            cluster_hard_score: 63,
            cluster_chrome_ceiling_score: 19,
            cluster_member_score: 25,
            cluster_coherence_score: 63,
            cluster_min_size: 2,
            cluster_coverage_floor_permille: 0,
            phash64_max_distance: 16,
            dhash64_max_distance: 12,
            perceptual_hash_max_total_distance: 26,
            suspicious_phash64_max_distance: 16,
            suspicious_dhash64_max_distance: 15,
            suspicious_perceptual_hash_max_total_distance: 30,
            suspicious_perceptual_visual_support_distance_slack: 6,
            perceptual_orientation_correction: false,
            perceptual_orientation_max_degrees: 10.0,
            perceptual_orientation_step_degrees: 1.0,
            perceptual_orientation_min_gain: 1.02,
            local_anchors_enabled: true,
            local_max_width: 768,
            local_max_height: 2048,
            local_max_area: 786_432,
            local_max_aspect_ratio: 8.0,
            local_tile_width: 64,
            local_tile_height: 32,
            local_stride: 12,
            local_tile_budget: 5_000,
            local_hash_cap: 3_000,
            local_anchor_count: 512,
            local_anchor_max_distance: 12,
            local_min_anchor_hits: 80,
            local_min_distinct_regions: 10,
            local_max_mean_distance: 7.0,
            local_luma_candidate_max_delta: 55,
            local_contrast_candidate_max_delta: 50,
            local_edge_density_candidate_max_delta: 55,
            local_position_candidate_max_delta: 100,
            visual_luma_zero_score_delta: 70,
            visual_color_zero_score_delta: 90,
            visual_grid_luma_zero_score_delta: 80,
            visual_text_grid_zero_score_delta: 70,
            geometry_min_short_edge: 640,
            geometry_min_area: 350_000,
            geometry_max_aspect_ratio: 1.4,
            geometry_max_aspect_delta: 0.45,
            geometry_max_width_delta: 0.30,
            geometry_max_height_delta: 0.30,
            geometry_enable_affine: true,
            geometry_enable_homography: true,
            geometry_model_slack: 2.0,
            geometry_max_anisotropy: 1.6,
            geometry_max_perspective: 2.2,
            geometry_affine_min_extra_inliers: 1,
            geometry_affine_min_extra_regions: 0,
            geometry_affine_max_mean_residual: 22.0,
            geometry_homography_min_extra_inliers: 2,
            geometry_homography_min_extra_regions: 1,
            geometry_homography_max_mean_residual: 18.0,
            geometry_ratio_min_margin: 2,
            geometry_enable_prosac_fallback: true,
            geometry_prosac_max_iters: 64,
            geometry_prosac_min_inliers: 8,
            local_suspicious_min_anchor_hits: 20,
            local_suspicious_min_distinct_regions: 10,
            local_suspicious_max_mean_distance: 12.0,
            local_suspicious_unverified_support_enabled: true,
            local_suspicious_unverified_support_min_anchor_hits: 50,
            local_suspicious_unverified_support_min_distinct_regions: 25,
            local_suspicious_unverified_support_min_retention_permille: 200,
            local_suspicious_unverified_support_max_mean_distance: 11.75,
            local_suspicious_unverified_support_max_perceptual_total_distance: 51,
            local_suspicious_unverified_support_max_aspect_delta: 0.45,
            local_suspicious_unverified_support_max_dimension_delta: 0.55,
            suspicious_local_luma_candidate_max_delta: 70,
            suspicious_local_contrast_candidate_max_delta: 65,
            suspicious_local_edge_density_candidate_max_delta: 75,
            suspicious_local_position_candidate_max_delta: 120,
            suspicious_visual_luma_zero_score_delta: 95,
            suspicious_visual_color_zero_score_delta: 120,
            suspicious_visual_grid_luma_zero_score_delta: 110,
            suspicious_visual_text_grid_zero_score_delta: 100,
            suspicious_geometry_min_short_edge: 640,
            suspicious_geometry_min_area: 350_000,
            suspicious_geometry_max_aspect_ratio: 1.36,
            suspicious_geometry_max_aspect_delta: 0.35,
            suspicious_geometry_max_width_delta: 0.20,
            suspicious_geometry_max_height_delta: 0.20,
            visual_shape_min_middle_text_percent: 35,
            visual_shape_min_center_text_percent: 0,
            visual_shape_max_center_text_percent: 78,
            visual_shape_max_edge_text_percent: 46,
            visual_shape_min_signals: 4,
            visual_shape_min_text_grid_mean: 24,
            visual_shape_max_text_grid_mean: 170,
            visual_shape_min_text_regions: 34,
            visual_shape_min_luma_mean: 30,
            visual_shape_max_luma_mean: 92,
            visual_shape_min_luma_std: 50,
            visual_shape_max_luma_std: 80,
            visual_shape_min_local_hashes: 180,
            visual_shape_max_rgb_spread: 34,
            visual_shape_sparse_max_luma_mean: 35,
            visual_shape_sparse_max_text_grid_mean: 110,
            visual_shape_sparse_min_local_hashes: 80,
        }
    }
}

impl Default for CommandConfig {
    fn default() -> Self {
        Self {
            register_on_startup: true,
        }
    }
}

impl Default for Dial9TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            trace_dir: "target/dial9-traces".to_owned(),
            max_disk_usage_mb: 1024,
            rotation_seconds: 60,
            shutdown_timeout_seconds: 10,
            runtime_name: "discord-sightline".to_owned(),
        }
    }
}

pub fn load_config() -> Result<AppConfig> {
    let path = env::var("SIGHTLINE_CONFIG").unwrap_or_else(|_| "sightline.toml".to_owned());
    if Path::new(&path).exists() {
        return load_config_from_path(Path::new(&path));
    }
    finalize_config(AppConfig::default())
}

pub fn load_config_from_path(path: &Path) -> Result<AppConfig> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let config = toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    finalize_config(config)
}

fn finalize_config(mut config: AppConfig) -> Result<AppConfig> {
    config.text_gate.normalize();
    let scan_extensions = config.scan.allowed_extensions.clone();
    ScanPolicy::normalize_extensions_into(&scan_extensions, &mut config.scan.allowed_extensions);
    validate_config(&config)?;
    Ok(config)
}

fn validate_config(config: &AppConfig) -> Result<()> {
    config.bot.user_id()?;
    ScanPolicy {
        exempt_administrators: config.scan.exempt_administrators,
        allowed_extensions: config.scan.allowed_extensions.clone(),
        max_file_bytes: 1,
    }
    .validate()
    .context("validating scan.allowed_extensions")?;
    anyhow::ensure!(
        config.queue.max_size > 0,
        "queue.max_size must be greater than 0"
    );
    anyhow::ensure!(
        config.queue.max_size <= 20_000,
        "queue.max_size must be at most 20000"
    );
    anyhow::ensure!(
        config.queue.enqueue_timeout_ms <= 1_000,
        "queue.enqueue_timeout_ms must be at most 1000"
    );
    anyhow::ensure!(
        (1..=64).contains(&config.queue.cpu_concurrency),
        "queue.cpu_concurrency must be between 1 and 64"
    );
    anyhow::ensure!(
        (1..=256).contains(&config.queue.download_concurrency),
        "queue.download_concurrency must be between 1 and 256"
    );
    anyhow::ensure!(
        (1..=32).contains(&config.queue.ocr_concurrency),
        "queue.ocr_concurrency must be between 1 and 32"
    );
    anyhow::ensure!(
        config.queue.max_images_per_message > 0,
        "queue.max_images_per_message must be greater than 0"
    );
    anyhow::ensure!(
        config.queue.max_images_per_message <= 20,
        "queue.max_images_per_message must be at most 20"
    );
    anyhow::ensure!(
        config.queue.hash_outcome_cache_size > 0,
        "queue.hash_outcome_cache_size must be greater than 0"
    );
    anyhow::ensure!(
        config.queue.hash_outcome_cache_size <= 100_000,
        "queue.hash_outcome_cache_size must be at most 100000"
    );
    anyhow::ensure!(
        config.queue.download_memory_max_bytes >= config.download.max_bytes,
        "queue.download_memory_max_bytes must be at least download.max_bytes"
    );
    anyhow::ensure!(
        config.queue.download_memory_max_bytes <= 1024 * 1024 * 1024,
        "queue.download_memory_max_bytes must be at most 1073741824"
    );
    let max_decoded_image_bytes = usize::try_from(decoded_image_memory_reservation_bytes(
        config.download.max_decoded_pixels,
    ))
    .unwrap_or(usize::MAX);
    anyhow::ensure!(
        config.queue.decoded_image_memory_max_bytes >= max_decoded_image_bytes,
        "queue.decoded_image_memory_max_bytes must cover the maximum decoded image working set"
    );
    anyhow::ensure!(
        config.queue.decoded_image_memory_max_bytes <= 1024 * 1024 * 1024,
        "queue.decoded_image_memory_max_bytes must be at most 1073741824"
    );
    anyhow::ensure!(
        config.queue.byte_store_max_bytes > 0,
        "queue.byte_store_max_bytes must be greater than 0"
    );
    anyhow::ensure!(
        config.queue.byte_store_max_bytes <= 512 * 1024 * 1024,
        "queue.byte_store_max_bytes must be at most 536870912"
    );
    anyhow::ensure!(
        config.download.max_bytes > 0,
        "download.max_bytes must be greater than 0"
    );
    anyhow::ensure!(
        config.download.max_bytes <= 50 * 1024 * 1024,
        "download.max_bytes must be at most 52428800"
    );
    anyhow::ensure!(
        config.download.max_decoded_pixels > 0,
        "download.max_decoded_pixels must be greater than 0"
    );
    anyhow::ensure!(
        config.download.max_decoded_pixels <= 12_000_000,
        "download.max_decoded_pixels must be at most 12000000"
    );
    anyhow::ensure!(
        config.download.timeout_seconds > 0,
        "download.timeout_seconds must be greater than 0"
    );
    anyhow::ensure!(
        config.download.timeout_seconds <= 60,
        "download.timeout_seconds must be at most 60"
    );
    anyhow::ensure!(
        config.download.max_retries <= 3,
        "download.max_retries must be at most 3"
    );
    anyhow::ensure!(
        config.download.retry_base_delay_ms <= 5_000,
        "download.retry_base_delay_ms must be at most 5000"
    );
    anyhow::ensure!(
        (60..=350).contains(&config.download.warmer_period_seconds),
        "download.warmer_period_seconds must be between 60 and 350"
    );
    anyhow::ensure!(
        config.download.preview.min_original_bytes
            <= u64::try_from(config.download.max_bytes).unwrap_or(u64::MAX),
        "download.preview.min_original_bytes must be at most download.max_bytes"
    );
    anyhow::ensure!(
        config.matching.local_max_width > 0,
        "match.local_max_width must be greater than 0"
    );
    anyhow::ensure!(
        config.matching.local_max_width <= 4096,
        "match.local_max_width must be at most 4096"
    );
    anyhow::ensure!(
        config.matching.local_max_height > 0,
        "match.local_max_height must be greater than 0"
    );
    anyhow::ensure!(
        config.matching.local_max_height <= 4096,
        "match.local_max_height must be at most 4096"
    );
    anyhow::ensure!(
        config.matching.local_max_area > 0,
        "match.local_max_area must be greater than 0"
    );
    anyhow::ensure!(
        config.matching.local_max_area <= 1_048_576,
        "match.local_max_area must be at most 1048576"
    );
    anyhow::ensure!(
        config.matching.local_max_aspect_ratio.is_finite()
            && config.matching.local_max_aspect_ratio >= 1.0,
        "match.local_max_aspect_ratio must be finite and at least 1.0"
    );
    anyhow::ensure!(
        config.matching.local_tile_budget > 0,
        "match.local_tile_budget must be greater than 0"
    );
    anyhow::ensure!(
        config.matching.local_tile_budget <= 10_000,
        "match.local_tile_budget must be at most 10000"
    );
    anyhow::ensure!(
        config.matching.local_hash_cap > 0,
        "match.local_hash_cap must be greater than 0"
    );
    anyhow::ensure!(
        config.matching.local_hash_cap <= 5_000,
        "match.local_hash_cap must be at most 5000"
    );
    anyhow::ensure!(
        config.matching.local_tile_width > 0 && config.matching.local_tile_height > 0,
        "match.local_tile_width and match.local_tile_height must be greater than 0"
    );
    anyhow::ensure!(
        config.matching.local_tile_width <= 512 && config.matching.local_tile_height <= 512,
        "match.local_tile_width and match.local_tile_height must be at most 512"
    );
    anyhow::ensure!(
        config.matching.local_stride > 0,
        "match.local_stride must be greater than 0"
    );
    anyhow::ensure!(
        config.matching.local_stride <= 512,
        "match.local_stride must be at most 512"
    );
    anyhow::ensure!(
        config.matching.local_anchor_count <= 4_096,
        "match.local_anchor_count must be at most 4096"
    );
    anyhow::ensure!(
        config.matching.local_min_anchor_hits <= 4_096
            && config.matching.local_suspicious_min_anchor_hits <= 4_096,
        "match local anchor hit thresholds must be at most 4096"
    );
    anyhow::ensure!(
        config.matching.local_min_distinct_regions <= 64
            && config.matching.local_suspicious_min_distinct_regions <= 64,
        "match local distinct region thresholds must be at most 64"
    );
    anyhow::ensure!(
        config.matching.local_max_mean_distance.is_finite()
            && (0.0..=15.0).contains(&config.matching.local_max_mean_distance)
            && config
                .matching
                .local_suspicious_max_mean_distance
                .is_finite()
            && (0.0..=15.0).contains(&config.matching.local_suspicious_max_mean_distance),
        "match local mean distance thresholds must be finite and between 0 and 15"
    );
    anyhow::ensure!(
        config
            .matching
            .local_suspicious_unverified_support_min_anchor_hits
            <= 4_096,
        "match.local_suspicious_unverified_support_min_anchor_hits must be at most 4096"
    );
    anyhow::ensure!(
        config
            .matching
            .local_suspicious_unverified_support_min_distinct_regions
            <= 64,
        "match.local_suspicious_unverified_support_min_distinct_regions must be at most 64"
    );
    anyhow::ensure!(
        config
            .matching
            .local_suspicious_unverified_support_min_retention_permille
            <= 1_000,
        "match.local_suspicious_unverified_support_min_retention_permille must be at most 1000"
    );
    anyhow::ensure!(
        config
            .matching
            .local_suspicious_unverified_support_max_mean_distance
            .is_finite()
            && (0.0..=15.0).contains(
                &config
                    .matching
                    .local_suspicious_unverified_support_max_mean_distance,
            ),
        "match.local_suspicious_unverified_support_max_mean_distance must be finite and between 0 and 15"
    );
    anyhow::ensure!(
        config
            .matching
            .local_suspicious_unverified_support_max_perceptual_total_distance
            <= 96,
        "match.local_suspicious_unverified_support_max_perceptual_total_distance must be at most 96"
    );
    anyhow::ensure!(
        config
            .matching
            .local_suspicious_unverified_support_max_aspect_delta
            .is_finite()
            && (0.0..=2.0).contains(
                &config
                    .matching
                    .local_suspicious_unverified_support_max_aspect_delta,
            )
            && config
                .matching
                .local_suspicious_unverified_support_max_dimension_delta
                .is_finite()
            && (0.0..=2.0).contains(
                &config
                    .matching
                    .local_suspicious_unverified_support_max_dimension_delta,
            ),
        "match local suspicious unverified support geometry limits must be finite and between 0 and 2"
    );
    validate_perceptual_hash_distances(
        config.matching.phash64_max_distance,
        config.matching.dhash64_max_distance,
        config.matching.perceptual_hash_max_total_distance,
        "match perceptual hash distances",
    )?;
    validate_perceptual_hash_distances(
        config.matching.suspicious_phash64_max_distance,
        config.matching.suspicious_dhash64_max_distance,
        config
            .matching
            .suspicious_perceptual_hash_max_total_distance,
        "match suspicious perceptual hash distances",
    )?;
    anyhow::ensure!(
        config
            .matching
            .suspicious_perceptual_visual_support_distance_slack
            <= 16,
        "match.suspicious_perceptual_visual_support_distance_slack must be at most 16"
    );
    anyhow::ensure!(
        config.matching.local_anchor_max_distance <= 15,
        "match.local_anchor_max_distance must be at most 15"
    );
    validate_cluster_config(&config.matching)?;
    validate_geometry_match_config(
        config.matching.geometry_min_short_edge,
        config.matching.geometry_min_area,
        config.matching.geometry_max_aspect_ratio,
        config.matching.geometry_max_aspect_delta,
        config.matching.geometry_max_width_delta,
        config.matching.geometry_max_height_delta,
        "match.geometry",
    )?;
    validate_geometry_match_config(
        config.matching.suspicious_geometry_min_short_edge,
        config.matching.suspicious_geometry_min_area,
        config.matching.suspicious_geometry_max_aspect_ratio,
        config.matching.suspicious_geometry_max_aspect_delta,
        config.matching.suspicious_geometry_max_width_delta,
        config.matching.suspicious_geometry_max_height_delta,
        "match.suspicious_geometry",
    )?;
    validate_score_config(&config.matching)?;
    validate_orientation_correction_config(&config.matching)?;
    validate_geometry_model_config(&config.matching)?;
    config.text_gate.validate()?;
    validate_telemetry_config(&config.telemetry)?;
    anyhow::ensure!(
        config.matching.visual_shape_min_middle_text_percent <= 100
            && config.matching.visual_shape_min_center_text_percent <= 100
            && config.matching.visual_shape_max_center_text_percent <= 100
            && config.matching.visual_shape_max_edge_text_percent <= 100,
        "match.visual_shape text distribution percentages must be at most 100"
    );
    anyhow::ensure!(
        config.matching.visual_shape_min_center_text_percent
            <= config.matching.visual_shape_max_center_text_percent,
        "match.visual_shape_min_center_text_percent must be less than or equal to match.visual_shape_max_center_text_percent"
    );
    anyhow::ensure!(
        (1..=8).contains(&config.matching.visual_shape_min_signals),
        "match.visual_shape_min_signals must be between 1 and 8"
    );
    anyhow::ensure!(
        config.matching.visual_shape_min_text_grid_mean
            <= config.matching.visual_shape_max_text_grid_mean,
        "match.visual_shape_min_text_grid_mean must be less than or equal to match.visual_shape_max_text_grid_mean"
    );
    anyhow::ensure!(
        config.matching.visual_shape_max_text_grid_mean >= 24,
        "match.visual_shape_max_text_grid_mean must be at least 24"
    );
    anyhow::ensure!(
        config.matching.visual_shape_min_text_regions <= 64,
        "match.visual_shape_min_text_regions must be at most 64"
    );
    anyhow::ensure!(
        config.matching.visual_shape_min_luma_mean <= config.matching.visual_shape_max_luma_mean,
        "match.visual_shape_min_luma_mean must be less than or equal to match.visual_shape_max_luma_mean"
    );
    anyhow::ensure!(
        config.matching.visual_shape_min_luma_std <= config.matching.visual_shape_max_luma_std,
        "match.visual_shape_min_luma_std must be less than or equal to match.visual_shape_max_luma_std"
    );
    anyhow::ensure!(
        config.matching.visual_shape_min_local_hashes <= config.matching.local_hash_cap,
        "match.visual_shape_min_local_hashes must be at most match.local_hash_cap"
    );
    anyhow::ensure!(
        config.matching.visual_shape_sparse_max_text_grid_mean >= 24,
        "match.visual_shape_sparse_max_text_grid_mean must be at least 24"
    );
    anyhow::ensure!(
        config.matching.visual_shape_sparse_min_local_hashes <= config.matching.local_hash_cap,
        "match.visual_shape_sparse_min_local_hashes must be at most match.local_hash_cap"
    );
    config.ocr_space.validate()?;
    Ok(())
}

fn validate_cluster_config(config: &MatchConfig) -> Result<()> {
    anyhow::ensure!(
        config.matching_cluster_member_above_chrome(),
        "match.cluster_member_score must be greater than match.cluster_chrome_ceiling_score"
    );
    anyhow::ensure!(
        config.cluster_hard_score <= 1_000,
        "match.cluster_hard_score must be at most 1000"
    );
    anyhow::ensure!(
        config.cluster_coherence_score <= 1_000,
        "match.cluster_coherence_score must be at most 1000"
    );
    anyhow::ensure!(
        (2..=16).contains(&config.cluster_min_size),
        "match.cluster_min_size must be between 2 and 16"
    );
    anyhow::ensure!(
        config.cluster_coverage_floor_permille <= 1_000,
        "match.cluster_coverage_floor_permille must be at most 1000"
    );
    Ok(())
}

fn validate_score_config(config: &MatchConfig) -> Result<()> {
    for (name, value, max) in [
        ("match.score_threshold", config.score_threshold, 1_000.0),
        (
            "match.suspicious_score_threshold",
            config.suspicious_score_threshold,
            1_000.0,
        ),
        (
            "match.perceptual_score_weight",
            config.perceptual_score_weight,
            10.0,
        ),
        (
            "match.local_anchor_score_weight",
            config.local_anchor_score_weight,
            10.0,
        ),
        (
            "match.dense_local_anchor_score_weight",
            config.dense_local_anchor_score_weight,
            10.0,
        ),
        (
            "match.visual_signature_score_weight",
            config.visual_signature_score_weight,
            1.0,
        ),
        (
            "match.visual_shape_score_weight",
            config.visual_shape_score_weight,
            10.0,
        ),
        (
            "match.visual_shape_score_cap",
            config.visual_shape_score_cap,
            100.0,
        ),
        (
            "match.perceptual_score_floor",
            config.perceptual_score_floor,
            100.0,
        ),
        (
            "match.local_score_full_spread",
            config.local_score_full_spread,
            10_000.0,
        ),
        (
            "match.visual_shape_score_full",
            config.visual_shape_score_full,
            10_000.0,
        ),
    ] {
        anyhow::ensure!(
            value.is_finite() && (0.0..=max).contains(&value),
            "{name} must be finite and between 0 and {max}"
        );
    }
    anyhow::ensure!(
        config.local_score_full_hits <= 20_000,
        "match.local_score_full_hits must be at most 20000"
    );
    anyhow::ensure!(
        config.local_score_full_regions <= 64,
        "match.local_score_full_regions must be at most 64"
    );
    Ok(())
}

impl MatchConfig {
    const fn matching_cluster_member_above_chrome(&self) -> bool {
        self.cluster_member_score > self.cluster_chrome_ceiling_score
    }
}

fn validate_orientation_correction_config(config: &MatchConfig) -> Result<()> {
    anyhow::ensure!(
        config.perceptual_orientation_max_degrees.is_finite()
            && (0.0..=20.0).contains(&config.perceptual_orientation_max_degrees),
        "match.perceptual_orientation_max_degrees must be finite and between 0 and 20"
    );
    anyhow::ensure!(
        config.perceptual_orientation_step_degrees.is_finite()
            && (0.25..=5.0).contains(&config.perceptual_orientation_step_degrees),
        "match.perceptual_orientation_step_degrees must be finite and between 0.25 and 5"
    );
    anyhow::ensure!(
        config.perceptual_orientation_min_gain.is_finite()
            && (1.0..=2.0).contains(&config.perceptual_orientation_min_gain),
        "match.perceptual_orientation_min_gain must be finite and between 1 and 2"
    );
    Ok(())
}

fn validate_perceptual_hash_distances(
    phash64_max_distance: u32,
    dhash64_max_distance: u32,
    max_total_distance: u32,
    name: &str,
) -> Result<()> {
    anyhow::ensure!(
        phash64_max_distance <= 16 && dhash64_max_distance <= 16,
        "{name} must have individual pHash/dHash distances at most 16"
    );
    anyhow::ensure!(
        phash64_max_distance < 16 || dhash64_max_distance < 16,
        "{name} must keep at least one individual pHash/dHash distance below 16 for indexed lookup"
    );
    anyhow::ensure!(
        max_total_distance <= phash64_max_distance + dhash64_max_distance,
        "{name} total distance must be at most the sum of the individual caps"
    );
    Ok(())
}

fn validate_geometry_model_config(config: &MatchConfig) -> Result<()> {
    anyhow::ensure!(
        config.geometry_model_slack.is_finite()
            && (0.0..=32.0).contains(&config.geometry_model_slack),
        "match.geometry_model_slack must be finite and between 0 and 32"
    );
    anyhow::ensure!(
        config.geometry_max_anisotropy.is_finite()
            && (1.0..=8.0).contains(&config.geometry_max_anisotropy),
        "match.geometry_max_anisotropy must be finite and between 1 and 8"
    );
    anyhow::ensure!(
        config.geometry_max_perspective.is_finite()
            && (1.0..=8.0).contains(&config.geometry_max_perspective),
        "match.geometry_max_perspective must be finite and between 1 and 8"
    );
    anyhow::ensure!(
        config.geometry_affine_min_extra_inliers <= 16
            && config.geometry_homography_min_extra_inliers <= 16,
        "match geometry model extra inliers must be at most 16"
    );
    anyhow::ensure!(
        config.geometry_affine_min_extra_regions <= 16
            && config.geometry_homography_min_extra_regions <= 16,
        "match geometry model extra regions must be at most 16"
    );
    anyhow::ensure!(
        config.geometry_affine_max_mean_residual.is_finite()
            && (0.0..=64.0).contains(&config.geometry_affine_max_mean_residual)
            && config.geometry_homography_max_mean_residual.is_finite()
            && (0.0..=64.0).contains(&config.geometry_homography_max_mean_residual),
        "match geometry model residual caps must be finite and between 0 and 64"
    );
    anyhow::ensure!(
        config.geometry_ratio_min_margin <= 64,
        "match.geometry_ratio_min_margin must be at most 64"
    );
    anyhow::ensure!(
        config.geometry_prosac_max_iters <= 10_000,
        "match.geometry_prosac_max_iters must be at most 10000"
    );
    anyhow::ensure!(
        (4..=64).contains(&config.geometry_prosac_min_inliers),
        "match.geometry_prosac_min_inliers must be between 4 and 64"
    );
    Ok(())
}

fn validate_telemetry_config(config: &TelemetryConfig) -> Result<()> {
    anyhow::ensure!(
        !config.dial9.trace_dir.trim().is_empty(),
        "telemetry.dial9.trace_dir must not be empty"
    );
    anyhow::ensure!(
        (1..=102_400).contains(&config.dial9.max_disk_usage_mb),
        "telemetry.dial9.max_disk_usage_mb must be between 1 and 102400"
    );
    anyhow::ensure!(
        (1..=3600).contains(&config.dial9.rotation_seconds),
        "telemetry.dial9.rotation_seconds must be between 1 and 3600"
    );
    anyhow::ensure!(
        (1..=300).contains(&config.dial9.shutdown_timeout_seconds),
        "telemetry.dial9.shutdown_timeout_seconds must be between 1 and 300"
    );
    anyhow::ensure!(
        !config.dial9.runtime_name.trim().is_empty(),
        "telemetry.dial9.runtime_name must not be empty"
    );
    Ok(())
}

fn validate_geometry_match_config(
    min_short_edge: u32,
    min_area: u64,
    max_aspect_ratio: f32,
    max_aspect_delta: f32,
    max_width_delta: f32,
    max_height_delta: f32,
    prefix: &str,
) -> Result<()> {
    anyhow::ensure!(
        min_short_edge <= 4096,
        "{prefix}_min_short_edge must be at most 4096"
    );
    anyhow::ensure!(
        min_area <= 50_000_000,
        "{prefix}_min_area must be at most 50000000"
    );
    anyhow::ensure!(
        max_aspect_ratio.is_finite() && (1.0..=16.0).contains(&max_aspect_ratio),
        "{prefix}_max_aspect_ratio must be finite and between 1 and 16"
    );
    anyhow::ensure!(
        max_aspect_delta.is_finite() && (0.0..=16.0).contains(&max_aspect_delta),
        "{prefix}_max_aspect_delta must be finite and between 0 and 16"
    );
    anyhow::ensure!(
        max_width_delta.is_finite() && (0.0..=10.0).contains(&max_width_delta),
        "{prefix}_max_width_delta must be finite and between 0 and 10"
    );
    anyhow::ensure!(
        max_height_delta.is_finite() && (0.0..=10.0).contains(&max_height_delta),
        "{prefix}_max_height_delta must be finite and between 0 and 10"
    );
    Ok(())
}

pub fn load_secrets() -> Result<Secrets> {
    let discord_token = env::var("DISCORD_TOKEN").context("DISCORD_TOKEN is required")?;
    let specimen_hmac_secret =
        env::var("SPECIMEN_HMAC_SECRET").context("SPECIMEN_HMAC_SECRET is required")?;
    anyhow::ensure!(
        specimen_hmac_secret.len() >= 32,
        "SPECIMEN_HMAC_SECRET must be at least 32 bytes"
    );
    Ok(Secrets {
        discord_token,
        specimen_hmac_secret,
        ocr_space_api_key: crate::ocr_space::load_api_key_from_env()?,
    })
}

pub(crate) fn effective_download_config(
    base: &DownloadConfig,
    scan_policy: &crate::configuration::guild::ScanPolicy,
) -> DownloadConfig {
    let mut config = base.clone();
    config.max_bytes = config
        .max_bytes
        .min(usize::try_from(scan_policy.max_file_bytes).unwrap_or(usize::MAX));
    config.preview.min_original_bytes = config
        .preview
        .min_original_bytes
        .min(u64::try_from(config.max_bytes).unwrap_or(u64::MAX));
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_download_config_clamps_preview_minimum_to_effective_max_bytes() {
        let base = DownloadConfig {
            max_bytes: 10_000,
            preview: PreviewDownloadConfig {
                min_original_bytes: 8_000,
                ..PreviewDownloadConfig::default()
            },
            ..DownloadConfig::default()
        };
        let scan_policy = ScanPolicy {
            max_file_bytes: 4_000,
            ..ScanPolicy::default()
        };

        let effective = effective_download_config(&base, &scan_policy);

        assert_eq!(effective.max_bytes, 4_000);
        assert_eq!(effective.preview.min_original_bytes, 4_000);
    }

    #[test]
    fn download_warmer_period_is_bounded() {
        let mut config = AppConfig::default();

        config.download.warmer_period_seconds = 59;
        assert!(validate_config(&config).is_err());

        config.download.warmer_period_seconds = 351;
        assert!(validate_config(&config).is_err());

        config.download.warmer_period_seconds = 270;
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn runtime_concurrency_and_download_memory_are_bounded() {
        let mut config = AppConfig::default();

        config.queue.cpu_concurrency = 65;
        assert!(validate_config(&config).is_err());
        config.queue.cpu_concurrency = 4;

        config.queue.download_concurrency = 257;
        assert!(validate_config(&config).is_err());
        config.queue.download_concurrency = 64;

        config.queue.ocr_concurrency = 33;
        assert!(validate_config(&config).is_err());
        config.queue.ocr_concurrency = 1;

        config.queue.download_memory_max_bytes = config.download.max_bytes - 1;
        assert!(validate_config(&config).is_err());
        config.queue.download_memory_max_bytes = config.download.max_bytes;
        assert!(validate_config(&config).is_ok());

        config.queue.download_memory_max_bytes = 1024 * 1024 * 1024 + 1;
        assert!(validate_config(&config).is_err());

        let decoded_image_bytes = usize::try_from(decoded_image_memory_reservation_bytes(
            config.download.max_decoded_pixels,
        ))
        .unwrap();
        config.queue.download_memory_max_bytes = 128 * 1024 * 1024;
        config.queue.decoded_image_memory_max_bytes = decoded_image_bytes - 1;
        assert!(validate_config(&config).is_err());
        config.queue.decoded_image_memory_max_bytes = decoded_image_bytes;
        assert!(validate_config(&config).is_ok());

        config.queue.decoded_image_memory_max_bytes = 1024 * 1024 * 1024 + 1;
        assert!(validate_config(&config).is_err());
    }
}
