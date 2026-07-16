#![allow(
    clippy::ref_option,
    clippy::struct_excessive_bools,
    clippy::struct_field_names,
    clippy::too_many_lines
)]

use crate::configuration::app::MatchConfig;
use crate::configuration::storage_codec::{decode_config, encode_config};
use anyhow::{Context, Result, anyhow};
use chrono::{SecondsFormat, Utc};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::hash::{Hash, Hasher};
use twilight_model::id::{
    Id,
    marker::{ChannelMarker, GuildMarker, RoleMarker, UserMarker},
};

type HmacSha256 = Hmac<Sha256>;

pub const TIMEOUT_DURATION_SECONDS: &[u32] = &[60, 300, 600, 3_600, 86_400, 604_800];
pub(crate) const MAX_CONFIG_RECORD_ATTACHMENT_BYTES: usize = 1_000_000;
const GUILD_CONFIG_RECORD_SCHEMA: u8 = 2;
const GUILD_CONFIG_MANIFEST_SCHEMA: u8 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuildConfig {
    pub version: u8,
    pub enabled: bool,
    pub guild_id: String,
    pub ledger_channel_id: String,
    pub bot_log_channel_id: Option<String>,
    pub discord_general_log_message: String,
    #[serde(default)]
    pub discord_confirmed_log_message: String,
    #[serde(default)]
    pub discord_suspicious_log_message: String,
    #[serde(default)]
    pub discord_benign_log_message: String,
    #[serde(default)]
    pub discord_detection_log_message: String,
    pub verified_role_id: Option<String>,
    pub moderator_role_ids: Vec<String>,
    pub scan_exempt_role_ids: Vec<String>,
    pub scan_policy: ScanPolicy,
    pub detection_hyperparameters: DetectionHyperparameters,
    pub detection_policy: DetectionPolicy,
    pub text_gate_policy: TextGatePolicy,
    pub updated_at: String,
    pub updated_by_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanPolicy {
    #[serde(default = "default_exempt_administrators")]
    pub exempt_administrators: bool,
    pub allowed_extensions: Vec<String>,
    pub max_file_bytes: u64,
    // Keep new MessagePack-compatible fields at the end of this struct. Storage uses
    // sequence encoding, so trailing defaults can be absent from deployed records.
    #[serde(default)]
    pub mark_message_siblings_suspicious: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionPolicy {
    pub confirmed: DetectionRule,
    pub suspicious: DetectionRule,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextGatePolicy {
    pub enabled: bool,
    pub keyword_threshold: usize,
    pub keyword_max_distance: u8,
    pub keywords: Vec<String>,
    pub sentences: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionHyperparameters {
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionRule {
    pub threshold: DetectionThreshold,
    pub actions: DetectionActions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionThreshold {
    pub score_threshold: f32,
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
    pub cluster_coherence: bool,
    #[serde(default, rename = "cluster_promote_to_confirmed")]
    _legacy_cluster_promote_to_confirmed: bool,
    pub cluster_hard_score: u32,
    pub cluster_chrome_ceiling_score: u32,
    pub cluster_member_score: u32,
    pub cluster_coherence_score: u32,
    pub cluster_min_size: u16,
    pub cluster_coverage_floor_permille: u16,
    pub exact_xxh128: bool,
    pub perceptual_hash: bool,
    pub phash64_max_distance: u32,
    pub dhash64_max_distance: u32,
    pub perceptual_hash_max_total_distance: u32,
    pub perceptual_visual_support_distance_slack: u32,
    pub local_anchors: bool,
    pub min_anchor_hits: usize,
    pub min_distinct_regions: usize,
    pub max_mean_distance: f32,
    pub local_unverified_support: bool,
    pub local_unverified_support_min_anchor_hits: usize,
    pub local_unverified_support_min_distinct_regions: usize,
    pub local_unverified_support_min_retention_permille: u16,
    pub local_unverified_support_max_mean_distance: f32,
    pub local_unverified_support_max_perceptual_total_distance: u32,
    pub local_unverified_support_max_aspect_delta: f32,
    pub local_unverified_support_max_dimension_delta: f32,
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
    pub visual_shape: bool,
    pub visual_shape_min_signals: usize,
    pub visual_shape_min_text_grid_mean: u8,
    pub visual_shape_max_text_grid_mean: u8,
    pub visual_shape_min_text_regions: usize,
    pub visual_shape_min_luma_mean: u8,
    pub visual_shape_max_luma_mean: u8,
    pub visual_shape_min_luma_std: u8,
    pub visual_shape_max_luma_std: u8,
    pub visual_shape_min_local_hashes: usize,
    pub visual_shape_min_middle_text_percent: u8,
    pub visual_shape_min_center_text_percent: u8,
    pub visual_shape_max_center_text_percent: u8,
    pub visual_shape_max_edge_text_percent: u8,
    pub visual_shape_sparse_max_luma_mean: u8,
    pub visual_shape_max_rgb_spread: u8,
    pub visual_shape_sparse_max_text_grid_mean: u8,
    pub visual_shape_sparse_min_local_hashes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DetectionActions {
    pub delete_message: bool,
    pub remove_user_roles: bool,
    pub timeout_user: bool,
    pub timeout_seconds: u32,
    pub ban_user: bool,
    pub ban_delete_message_seconds: u32,
    #[serde(default)]
    pub kick_user: bool,
    pub add_to_specimens: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuildConfigRecord {
    pub schema: u8,
    #[serde(rename = "type")]
    pub kind: String,
    pub created_at: String,
    pub guild_id: String,
    pub config: GuildConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sig: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GuildConfigManifest {
    pub(crate) schema: u8,
    #[serde(rename = "type")]
    pub(crate) kind: String,
    pub(crate) guild_id: String,
    pub(crate) record_attachment: String,
    pub(crate) record_bytes: u32,
    pub(crate) record_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sig: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct GuildConfigRecordAttachment {
    pub(crate) filename: String,
    pub(crate) bytes: Vec<u8>,
}

impl GuildConfig {
    pub fn from_guild(guild_id: Id<GuildMarker>, ledger_channel_id: Id<ChannelMarker>) -> Self {
        Self {
            version: 2,
            enabled: false,
            guild_id: guild_id.get().to_string(),
            ledger_channel_id: ledger_channel_id.get().to_string(),
            bot_log_channel_id: None,
            discord_general_log_message: String::new(),
            discord_confirmed_log_message: String::new(),
            discord_suspicious_log_message: String::new(),
            discord_benign_log_message: String::new(),
            discord_detection_log_message: String::new(),
            verified_role_id: None,
            moderator_role_ids: Vec::new(),
            scan_exempt_role_ids: Vec::new(),
            scan_policy: ScanPolicy::default(),
            detection_hyperparameters: DetectionHyperparameters::default(),
            detection_policy: DetectionPolicy::default(),
            text_gate_policy: TextGatePolicy::default(),
            updated_at: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
            updated_by_id: "0".to_owned(),
        }
    }

    pub fn from_loaded_defaults(
        guild_id: Id<GuildMarker>,
        ledger_channel_id: Id<ChannelMarker>,
        matching: &MatchConfig,
        scan_policy: &ScanPolicy,
        text_gate_policy: &TextGatePolicy,
    ) -> Self {
        let mut config = Self::from_guild(guild_id, ledger_channel_id);
        config.scan_policy = scan_policy.clone();
        config.detection_hyperparameters = DetectionHyperparameters::from_match_config(matching);
        config.detection_policy = DetectionPolicy::from_match_config(matching);
        config.text_gate_policy = text_gate_policy.clone();
        config.normalize();
        config
    }

    pub fn touch(&mut self, user_id: Id<UserMarker>) {
        self.updated_at = Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true);
        self.updated_by_id = user_id.get().to_string();
    }

    pub fn ledger_channel_id(&self) -> Result<Id<ChannelMarker>> {
        parse_id(&self.ledger_channel_id)
    }

    pub fn bot_log_channel_id(&self) -> Option<Id<ChannelMarker>> {
        self.bot_log_channel_id.as_deref().and_then(parse_id_opt)
    }

    pub fn discord_general_log_message_content(&self) -> Option<String> {
        let content = self.discord_general_log_message.trim();
        (!content.is_empty()).then(|| content.to_owned())
    }

    pub fn discord_detection_log_message_content(&self) -> Option<String> {
        trimmed_content(&self.discord_detection_log_message)
    }

    pub fn discord_confirmed_log_message_content(&self) -> Option<String> {
        trimmed_content(&self.discord_confirmed_log_message)
            .or_else(|| self.discord_detection_log_message_content())
    }

    pub fn discord_suspicious_log_message_content(&self) -> Option<String> {
        trimmed_content(&self.discord_suspicious_log_message)
            .or_else(|| self.discord_detection_log_message_content())
    }

    pub fn discord_benign_log_message_content(&self) -> Option<String> {
        trimmed_content(&self.discord_benign_log_message)
    }

    pub fn verified_role_id(&self) -> Option<Id<RoleMarker>> {
        self.verified_role_id.as_deref().and_then(parse_id_opt)
    }

    pub fn parsed_scan_exempt_role_ids(&self) -> Vec<Id<RoleMarker>> {
        self.scan_exempt_role_ids
            .iter()
            .filter_map(|role_id| parse_id_opt(role_id))
            .collect()
    }

    pub fn detection_cache_policy_hash(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.version.hash(&mut hasher);
        if let Ok(bytes) = serde_json::to_vec(&(
            &self.scan_policy,
            &self.detection_hyperparameters,
            &self.detection_policy,
            &self.text_gate_policy,
        )) {
            bytes.hash(&mut hasher);
        }
        self.updated_at.hash(&mut hasher);
        hasher.finish()
    }

    pub fn validate(&self, expected_guild_id: Id<GuildMarker>) -> Result<()> {
        anyhow::ensure!(self.version == 2, "guild config version must be 2");
        anyhow::ensure!(
            self.guild_id == expected_guild_id.get().to_string(),
            "guild config is for guild {}, expected {}",
            self.guild_id,
            expected_guild_id.get()
        );
        let ledger_channel_id = parse_id::<ChannelMarker>(&self.ledger_channel_id)
            .context("ledger_channel_id is invalid")?;
        if let Some(bot_log_channel_id) = &self.bot_log_channel_id {
            let bot_log_channel_id = parse_id::<ChannelMarker>(bot_log_channel_id)
                .context("bot_log_channel_id is invalid")?;
            anyhow::ensure!(
                bot_log_channel_id != ledger_channel_id,
                "bot_log_channel_id must be different from ledger_channel_id"
            );
        }
        validate_optional_id::<RoleMarker>(&self.verified_role_id, "verified_role_id")?;
        anyhow::ensure!(
            self.discord_general_log_message.chars().count() <= 1900,
            "discord_general_log_message must be at most 1900 characters"
        );
        anyhow::ensure!(
            self.discord_detection_log_message.chars().count() <= 1900,
            "discord_detection_log_message must be at most 1900 characters"
        );
        anyhow::ensure!(
            self.discord_confirmed_log_message.chars().count() <= 1900,
            "discord_confirmed_log_message must be at most 1900 characters"
        );
        anyhow::ensure!(
            self.discord_suspicious_log_message.chars().count() <= 1900,
            "discord_suspicious_log_message must be at most 1900 characters"
        );
        anyhow::ensure!(
            self.discord_benign_log_message.chars().count() <= 1900,
            "discord_benign_log_message must be at most 1900 characters"
        );
        validate_id_list::<RoleMarker>(&self.moderator_role_ids, "moderator_role_ids")?;
        validate_id_list::<RoleMarker>(&self.scan_exempt_role_ids, "scan_exempt_role_ids")?;
        self.scan_policy.validate()?;
        self.detection_hyperparameters.validate()?;
        self.detection_policy.validate()?;
        self.detection_policy
            .validate_against_hyperparameters(&self.detection_hyperparameters)?;
        self.text_gate_policy.validate()?;
        Ok(())
    }

    pub fn normalize(&mut self) {
        normalize_optional_id(&mut self.bot_log_channel_id);
        normalize_optional_id(&mut self.verified_role_id);
        normalize_id_list(&mut self.moderator_role_ids);
        normalize_id_list(&mut self.scan_exempt_role_ids);
        self.discord_general_log_message = self.discord_general_log_message.trim().to_owned();
        self.discord_confirmed_log_message = self.discord_confirmed_log_message.trim().to_owned();
        self.discord_suspicious_log_message = self.discord_suspicious_log_message.trim().to_owned();
        self.discord_benign_log_message = self.discord_benign_log_message.trim().to_owned();
        self.discord_detection_log_message = self.discord_detection_log_message.trim().to_owned();
        if !self.discord_detection_log_message.is_empty() {
            if self.discord_confirmed_log_message.is_empty() {
                self.discord_confirmed_log_message
                    .clone_from(&self.discord_detection_log_message);
            }
            if self.discord_suspicious_log_message.is_empty() {
                self.discord_suspicious_log_message
                    .clone_from(&self.discord_detection_log_message);
            }
            self.discord_detection_log_message.clear();
        }
        self.scan_policy.normalize();
        self.text_gate_policy.normalize();
    }
}

fn trimmed_content(value: &str) -> Option<String> {
    let content = value.trim();
    (!content.is_empty()).then(|| content.to_owned())
}

impl Default for ScanPolicy {
    fn default() -> Self {
        Self {
            exempt_administrators: default_exempt_administrators(),
            allowed_extensions: vec!["jpg".to_owned()],
            max_file_bytes: 10 * 1024 * 1024,
            mark_message_siblings_suspicious: false,
        }
    }
}

const fn default_exempt_administrators() -> bool {
    true
}

impl ScanPolicy {
    pub fn normalize(&mut self) {
        let source = self.allowed_extensions.clone();
        Self::normalize_extensions_into(&source, &mut self.allowed_extensions);
    }

    pub fn normalize_extensions_into(source: &[String], target: &mut Vec<String>) {
        let mut normalized = Vec::with_capacity(source.len());
        for extension in source {
            let extension = normalize_image_extension(extension);
            if !extension.is_empty() && !normalized.iter().any(|existing| existing == &extension) {
                normalized.push(extension);
            }
        }
        *target = normalized;
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.max_file_bytes > 0 && self.max_file_bytes <= 50 * 1024 * 1024,
            "scan_policy.max_file_bytes must be between 1 and 52428800"
        );
        anyhow::ensure!(
            !self.allowed_extensions.is_empty() && self.allowed_extensions.len() <= 32,
            "scan_policy.allowed_extensions must contain 1-32 entries"
        );
        for extension in &self.allowed_extensions {
            anyhow::ensure!(
                !extension.is_empty()
                    && extension.len() <= 16
                    && extension
                        .chars()
                        .all(|character| character.is_ascii_alphanumeric()),
                "scan_policy.allowed_extensions entries must be 1-16 ASCII letters/digits without dots"
            );
        }
        Ok(())
    }
}

pub fn normalize_image_extension(extension: &str) -> String {
    match extension
        .trim()
        .trim_start_matches('.')
        .to_ascii_lowercase()
        .as_str()
    {
        "jpeg" | "jpe" => "jpg".to_owned(),
        other => other.to_owned(),
    }
}

impl Default for DetectionHyperparameters {
    fn default() -> Self {
        Self {
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
        }
    }
}

impl DetectionHyperparameters {
    pub fn from_match_config(config: &MatchConfig) -> Self {
        Self {
            perceptual_orientation_correction: config.perceptual_orientation_correction,
            perceptual_orientation_max_degrees: config.perceptual_orientation_max_degrees,
            perceptual_orientation_step_degrees: config.perceptual_orientation_step_degrees,
            perceptual_orientation_min_gain: config.perceptual_orientation_min_gain,
            local_anchors_enabled: config.local_anchors_enabled,
            local_max_width: config.local_max_width,
            local_max_height: config.local_max_height,
            local_max_area: config.local_max_area,
            local_max_aspect_ratio: config.local_max_aspect_ratio,
            local_tile_width: config.local_tile_width,
            local_tile_height: config.local_tile_height,
            local_stride: config.local_stride,
            local_tile_budget: config.local_tile_budget,
            local_hash_cap: config.local_hash_cap,
            local_anchor_count: config.local_anchor_count,
            local_anchor_max_distance: config.local_anchor_max_distance,
        }
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.perceptual_orientation_max_degrees.is_finite()
                && (0.0..=20.0).contains(&self.perceptual_orientation_max_degrees),
            "perceptual_orientation_max_degrees must be finite and between 0 and 20"
        );
        anyhow::ensure!(
            self.perceptual_orientation_step_degrees.is_finite()
                && (0.25..=5.0).contains(&self.perceptual_orientation_step_degrees),
            "perceptual_orientation_step_degrees must be finite and between 0.25 and 5"
        );
        anyhow::ensure!(
            self.perceptual_orientation_min_gain.is_finite()
                && (1.0..=2.0).contains(&self.perceptual_orientation_min_gain),
            "perceptual_orientation_min_gain must be finite and between 1 and 2"
        );
        anyhow::ensure!(
            self.local_max_width > 0 && self.local_max_width <= 4096,
            "local_max_width must be between 1 and 4096"
        );
        anyhow::ensure!(
            self.local_max_height > 0 && self.local_max_height <= 4096,
            "local_max_height must be between 1 and 4096"
        );
        anyhow::ensure!(
            self.local_max_area > 0 && self.local_max_area <= 1_048_576,
            "local_max_area must be between 1 and 1048576"
        );
        anyhow::ensure!(
            self.local_max_aspect_ratio.is_finite() && self.local_max_aspect_ratio >= 1.0,
            "local_max_aspect_ratio must be finite and at least 1.0"
        );
        anyhow::ensure!(
            self.local_tile_width > 0 && self.local_tile_width <= 512,
            "local_tile_width must be between 1 and 512"
        );
        anyhow::ensure!(
            self.local_tile_height > 0 && self.local_tile_height <= 512,
            "local_tile_height must be between 1 and 512"
        );
        anyhow::ensure!(
            self.local_stride > 0 && self.local_stride <= 512,
            "local_stride must be between 1 and 512"
        );
        anyhow::ensure!(
            self.local_tile_budget > 0 && self.local_tile_budget <= 10_000,
            "local_tile_budget must be between 1 and 10000"
        );
        anyhow::ensure!(
            self.local_hash_cap > 0 && self.local_hash_cap <= 5_000,
            "local_hash_cap must be between 1 and 5000"
        );
        anyhow::ensure!(
            self.local_anchor_count <= 4_096,
            "local_anchor_count must be at most 4096"
        );
        anyhow::ensure!(
            self.local_anchor_max_distance <= 15,
            "local_anchor_max_distance must be at most 15"
        );
        Ok(())
    }

    pub fn effective_match_config(&self, base: &MatchConfig) -> MatchConfig {
        let mut config = base.clone();
        config.perceptual_orientation_correction = self.perceptual_orientation_correction;
        config.perceptual_orientation_max_degrees = self.perceptual_orientation_max_degrees;
        config.perceptual_orientation_step_degrees = self.perceptual_orientation_step_degrees;
        config.perceptual_orientation_min_gain = self.perceptual_orientation_min_gain;
        config.local_anchors_enabled = self.local_anchors_enabled;
        config.local_max_width = self.local_max_width;
        config.local_max_height = self.local_max_height;
        config.local_max_area = self.local_max_area;
        config.local_max_aspect_ratio = self.local_max_aspect_ratio;
        config.local_tile_width = self.local_tile_width;
        config.local_tile_height = self.local_tile_height;
        config.local_stride = self.local_stride;
        config.local_tile_budget = self.local_tile_budget;
        config.local_hash_cap = self.local_hash_cap;
        config.local_anchor_count = self.local_anchor_count;
        config.local_anchor_max_distance = self.local_anchor_max_distance;
        config
    }
}

impl Default for DetectionPolicy {
    fn default() -> Self {
        Self {
            confirmed: DetectionRule {
                threshold: DetectionThreshold {
                    score_threshold: 63.0,
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
                    cluster_coherence: false,
                    _legacy_cluster_promote_to_confirmed: false,
                    cluster_hard_score: 63,
                    cluster_chrome_ceiling_score: 19,
                    cluster_member_score: 25,
                    cluster_coherence_score: 63,
                    cluster_min_size: 2,
                    cluster_coverage_floor_permille: 0,
                    exact_xxh128: true,
                    perceptual_hash: true,
                    phash64_max_distance: 16,
                    dhash64_max_distance: 12,
                    perceptual_hash_max_total_distance: 26,
                    perceptual_visual_support_distance_slack: 0,
                    local_anchors: true,
                    min_anchor_hits: 80,
                    min_distinct_regions: 10,
                    max_mean_distance: 7.0,
                    local_unverified_support: false,
                    local_unverified_support_min_anchor_hits: 0,
                    local_unverified_support_min_distinct_regions: 0,
                    local_unverified_support_min_retention_permille: 0,
                    local_unverified_support_max_mean_distance: 0.0,
                    local_unverified_support_max_perceptual_total_distance: 0,
                    local_unverified_support_max_aspect_delta: 0.0,
                    local_unverified_support_max_dimension_delta: 0.0,
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
                    visual_shape: false,
                    visual_shape_min_signals: 4,
                    visual_shape_min_text_grid_mean: 24,
                    visual_shape_max_text_grid_mean: 170,
                    visual_shape_min_text_regions: 34,
                    visual_shape_min_luma_mean: 30,
                    visual_shape_max_luma_mean: 92,
                    visual_shape_min_luma_std: 50,
                    visual_shape_max_luma_std: 80,
                    visual_shape_min_local_hashes: 180,
                    visual_shape_min_middle_text_percent: 35,
                    visual_shape_min_center_text_percent: 0,
                    visual_shape_max_center_text_percent: 78,
                    visual_shape_max_edge_text_percent: 46,
                    visual_shape_max_rgb_spread: 34,
                    visual_shape_sparse_max_luma_mean: 35,
                    visual_shape_sparse_max_text_grid_mean: 110,
                    visual_shape_sparse_min_local_hashes: 80,
                },
                actions: DetectionActions {
                    delete_message: true,
                    remove_user_roles: false,
                    timeout_user: false,
                    timeout_seconds: 300,
                    ban_user: false,
                    ban_delete_message_seconds: 0,
                    kick_user: false,
                    add_to_specimens: false,
                },
            },
            suspicious: DetectionRule {
                threshold: DetectionThreshold {
                    score_threshold: 20.0,
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
                    cluster_coherence: true,
                    _legacy_cluster_promote_to_confirmed: false,
                    cluster_hard_score: 63,
                    cluster_chrome_ceiling_score: 19,
                    cluster_member_score: 25,
                    cluster_coherence_score: 63,
                    cluster_min_size: 2,
                    cluster_coverage_floor_permille: 0,
                    exact_xxh128: false,
                    perceptual_hash: true,
                    phash64_max_distance: 16,
                    dhash64_max_distance: 15,
                    perceptual_hash_max_total_distance: 30,
                    perceptual_visual_support_distance_slack: 6,
                    local_anchors: true,
                    min_anchor_hits: 20,
                    min_distinct_regions: 10,
                    max_mean_distance: 12.0,
                    local_unverified_support: true,
                    local_unverified_support_min_anchor_hits: 50,
                    local_unverified_support_min_distinct_regions: 25,
                    local_unverified_support_min_retention_permille: 200,
                    local_unverified_support_max_mean_distance: 11.75,
                    local_unverified_support_max_perceptual_total_distance: 51,
                    local_unverified_support_max_aspect_delta: 0.45,
                    local_unverified_support_max_dimension_delta: 0.55,
                    local_luma_candidate_max_delta: 70,
                    local_contrast_candidate_max_delta: 65,
                    local_edge_density_candidate_max_delta: 75,
                    local_position_candidate_max_delta: 120,
                    visual_luma_zero_score_delta: 95,
                    visual_color_zero_score_delta: 120,
                    visual_grid_luma_zero_score_delta: 110,
                    visual_text_grid_zero_score_delta: 100,
                    geometry_min_short_edge: 640,
                    geometry_min_area: 350_000,
                    geometry_max_aspect_ratio: 1.36,
                    geometry_max_aspect_delta: 0.35,
                    geometry_max_width_delta: 0.20,
                    geometry_max_height_delta: 0.20,
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
                    visual_shape: true,
                    visual_shape_min_signals: 4,
                    visual_shape_min_text_grid_mean: 24,
                    visual_shape_max_text_grid_mean: 170,
                    visual_shape_min_text_regions: 34,
                    visual_shape_min_luma_mean: 30,
                    visual_shape_max_luma_mean: 92,
                    visual_shape_min_luma_std: 50,
                    visual_shape_max_luma_std: 80,
                    visual_shape_min_local_hashes: 180,
                    visual_shape_min_middle_text_percent: 35,
                    visual_shape_min_center_text_percent: 0,
                    visual_shape_max_center_text_percent: 78,
                    visual_shape_max_edge_text_percent: 46,
                    visual_shape_max_rgb_spread: 34,
                    visual_shape_sparse_max_luma_mean: 35,
                    visual_shape_sparse_max_text_grid_mean: 110,
                    visual_shape_sparse_min_local_hashes: 80,
                },
                actions: DetectionActions {
                    delete_message: false,
                    remove_user_roles: false,
                    timeout_user: false,
                    timeout_seconds: 300,
                    ban_user: false,
                    ban_delete_message_seconds: 0,
                    kick_user: false,
                    add_to_specimens: false,
                },
            },
        }
    }
}

impl Default for TextGatePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            keyword_threshold: 2,
            keyword_max_distance: 1,
            keywords: Vec::new(),
            sentences: Vec::new(),
        }
    }
}

impl DetectionPolicy {
    pub fn from_match_config(config: &MatchConfig) -> Self {
        Self {
            confirmed: DetectionRule {
                threshold: DetectionThreshold {
                    score_threshold: config.score_threshold,
                    perceptual_score_weight: config.perceptual_score_weight,
                    local_anchor_score_weight: config.local_anchor_score_weight,
                    dense_local_anchor_score_weight: config.dense_local_anchor_score_weight,
                    visual_signature_score_weight: config.visual_signature_score_weight,
                    visual_shape_score_weight: config.visual_shape_score_weight,
                    visual_shape_score_cap: config.visual_shape_score_cap,
                    perceptual_score_floor: config.perceptual_score_floor,
                    local_score_full_hits: config.local_score_full_hits,
                    local_score_full_regions: config.local_score_full_regions,
                    local_score_full_spread: config.local_score_full_spread,
                    visual_shape_score_full: config.visual_shape_score_full,
                    cluster_coherence: false,
                    _legacy_cluster_promote_to_confirmed: false,
                    cluster_hard_score: config.cluster_hard_score,
                    cluster_chrome_ceiling_score: config.cluster_chrome_ceiling_score,
                    cluster_member_score: config.cluster_member_score,
                    cluster_coherence_score: config.cluster_coherence_score,
                    cluster_min_size: config.cluster_min_size,
                    cluster_coverage_floor_permille: config.cluster_coverage_floor_permille,
                    exact_xxh128: true,
                    perceptual_hash: true,
                    phash64_max_distance: config.phash64_max_distance,
                    dhash64_max_distance: config.dhash64_max_distance,
                    perceptual_hash_max_total_distance: config.perceptual_hash_max_total_distance,
                    perceptual_visual_support_distance_slack: 0,
                    local_anchors: config.local_anchors_enabled,
                    min_anchor_hits: config.local_min_anchor_hits,
                    min_distinct_regions: config.local_min_distinct_regions,
                    max_mean_distance: config.local_max_mean_distance,
                    local_unverified_support: false,
                    local_unverified_support_min_anchor_hits: 0,
                    local_unverified_support_min_distinct_regions: 0,
                    local_unverified_support_min_retention_permille: 0,
                    local_unverified_support_max_mean_distance: 0.0,
                    local_unverified_support_max_perceptual_total_distance: 0,
                    local_unverified_support_max_aspect_delta: 0.0,
                    local_unverified_support_max_dimension_delta: 0.0,
                    local_luma_candidate_max_delta: config.local_luma_candidate_max_delta,
                    local_contrast_candidate_max_delta: config.local_contrast_candidate_max_delta,
                    local_edge_density_candidate_max_delta: config
                        .local_edge_density_candidate_max_delta,
                    local_position_candidate_max_delta: config.local_position_candidate_max_delta,
                    visual_luma_zero_score_delta: config.visual_luma_zero_score_delta,
                    visual_color_zero_score_delta: config.visual_color_zero_score_delta,
                    visual_grid_luma_zero_score_delta: config.visual_grid_luma_zero_score_delta,
                    visual_text_grid_zero_score_delta: config.visual_text_grid_zero_score_delta,
                    geometry_min_short_edge: config.geometry_min_short_edge,
                    geometry_min_area: config.geometry_min_area,
                    geometry_max_aspect_ratio: config.geometry_max_aspect_ratio,
                    geometry_max_aspect_delta: config.geometry_max_aspect_delta,
                    geometry_max_width_delta: config.geometry_max_width_delta,
                    geometry_max_height_delta: config.geometry_max_height_delta,
                    geometry_enable_affine: config.geometry_enable_affine,
                    geometry_enable_homography: config.geometry_enable_homography,
                    geometry_model_slack: config.geometry_model_slack,
                    geometry_max_anisotropy: config.geometry_max_anisotropy,
                    geometry_max_perspective: config.geometry_max_perspective,
                    geometry_affine_min_extra_inliers: config.geometry_affine_min_extra_inliers,
                    geometry_affine_min_extra_regions: config.geometry_affine_min_extra_regions,
                    geometry_affine_max_mean_residual: config.geometry_affine_max_mean_residual,
                    geometry_homography_min_extra_inliers: config
                        .geometry_homography_min_extra_inliers,
                    geometry_homography_min_extra_regions: config
                        .geometry_homography_min_extra_regions,
                    geometry_homography_max_mean_residual: config
                        .geometry_homography_max_mean_residual,
                    geometry_ratio_min_margin: config.geometry_ratio_min_margin,
                    geometry_enable_prosac_fallback: config.geometry_enable_prosac_fallback,
                    geometry_prosac_max_iters: config.geometry_prosac_max_iters,
                    geometry_prosac_min_inliers: config.geometry_prosac_min_inliers,
                    visual_shape: false,
                    visual_shape_min_signals: config.visual_shape_min_signals,
                    visual_shape_min_text_grid_mean: config.visual_shape_min_text_grid_mean,
                    visual_shape_max_text_grid_mean: config.visual_shape_max_text_grid_mean,
                    visual_shape_min_text_regions: config.visual_shape_min_text_regions,
                    visual_shape_min_luma_mean: config.visual_shape_min_luma_mean,
                    visual_shape_max_luma_mean: config.visual_shape_max_luma_mean,
                    visual_shape_min_luma_std: config.visual_shape_min_luma_std,
                    visual_shape_max_luma_std: config.visual_shape_max_luma_std,
                    visual_shape_min_local_hashes: config.visual_shape_min_local_hashes,
                    visual_shape_min_middle_text_percent: config
                        .visual_shape_min_middle_text_percent,
                    visual_shape_min_center_text_percent: config
                        .visual_shape_min_center_text_percent,
                    visual_shape_max_center_text_percent: config
                        .visual_shape_max_center_text_percent,
                    visual_shape_max_edge_text_percent: config.visual_shape_max_edge_text_percent,
                    visual_shape_sparse_max_luma_mean: config.visual_shape_sparse_max_luma_mean,
                    visual_shape_max_rgb_spread: config.visual_shape_max_rgb_spread,
                    visual_shape_sparse_max_text_grid_mean: config
                        .visual_shape_sparse_max_text_grid_mean,
                    visual_shape_sparse_min_local_hashes: config
                        .visual_shape_sparse_min_local_hashes,
                },
                actions: DetectionActions {
                    delete_message: true,
                    remove_user_roles: false,
                    timeout_user: false,
                    timeout_seconds: 300,
                    ban_user: false,
                    ban_delete_message_seconds: 0,
                    kick_user: false,
                    add_to_specimens: false,
                },
            },
            suspicious: DetectionRule {
                threshold: DetectionThreshold {
                    score_threshold: config.suspicious_score_threshold,
                    perceptual_score_weight: config.perceptual_score_weight,
                    local_anchor_score_weight: config.local_anchor_score_weight,
                    dense_local_anchor_score_weight: config.dense_local_anchor_score_weight,
                    visual_signature_score_weight: config.visual_signature_score_weight,
                    visual_shape_score_weight: config.visual_shape_score_weight,
                    visual_shape_score_cap: config.visual_shape_score_cap,
                    perceptual_score_floor: config.perceptual_score_floor,
                    local_score_full_hits: config.local_score_full_hits,
                    local_score_full_regions: config.local_score_full_regions,
                    local_score_full_spread: config.local_score_full_spread,
                    visual_shape_score_full: config.visual_shape_score_full,
                    cluster_coherence: config.cluster_coherence_enabled,
                    _legacy_cluster_promote_to_confirmed: false,
                    cluster_hard_score: config.cluster_hard_score,
                    cluster_chrome_ceiling_score: config.cluster_chrome_ceiling_score,
                    cluster_member_score: config.cluster_member_score,
                    cluster_coherence_score: config.cluster_coherence_score,
                    cluster_min_size: config.cluster_min_size,
                    cluster_coverage_floor_permille: config.cluster_coverage_floor_permille,
                    exact_xxh128: false,
                    perceptual_hash: true,
                    phash64_max_distance: config.suspicious_phash64_max_distance,
                    dhash64_max_distance: config.suspicious_dhash64_max_distance,
                    perceptual_hash_max_total_distance: config
                        .suspicious_perceptual_hash_max_total_distance,
                    perceptual_visual_support_distance_slack: config
                        .suspicious_perceptual_visual_support_distance_slack,
                    local_anchors: config.local_anchors_enabled,
                    min_anchor_hits: config.local_suspicious_min_anchor_hits,
                    min_distinct_regions: config.local_suspicious_min_distinct_regions,
                    max_mean_distance: config.local_suspicious_max_mean_distance,
                    local_unverified_support: config.local_suspicious_unverified_support_enabled,
                    local_unverified_support_min_anchor_hits: config
                        .local_suspicious_unverified_support_min_anchor_hits,
                    local_unverified_support_min_distinct_regions: config
                        .local_suspicious_unverified_support_min_distinct_regions,
                    local_unverified_support_min_retention_permille: config
                        .local_suspicious_unverified_support_min_retention_permille,
                    local_unverified_support_max_mean_distance: config
                        .local_suspicious_unverified_support_max_mean_distance,
                    local_unverified_support_max_perceptual_total_distance: config
                        .local_suspicious_unverified_support_max_perceptual_total_distance,
                    local_unverified_support_max_aspect_delta: config
                        .local_suspicious_unverified_support_max_aspect_delta,
                    local_unverified_support_max_dimension_delta: config
                        .local_suspicious_unverified_support_max_dimension_delta,
                    local_luma_candidate_max_delta: config
                        .suspicious_local_luma_candidate_max_delta,
                    local_contrast_candidate_max_delta: config
                        .suspicious_local_contrast_candidate_max_delta,
                    local_edge_density_candidate_max_delta: config
                        .suspicious_local_edge_density_candidate_max_delta,
                    local_position_candidate_max_delta: config
                        .suspicious_local_position_candidate_max_delta,
                    visual_luma_zero_score_delta: config.suspicious_visual_luma_zero_score_delta,
                    visual_color_zero_score_delta: config.suspicious_visual_color_zero_score_delta,
                    visual_grid_luma_zero_score_delta: config
                        .suspicious_visual_grid_luma_zero_score_delta,
                    visual_text_grid_zero_score_delta: config
                        .suspicious_visual_text_grid_zero_score_delta,
                    geometry_min_short_edge: config.suspicious_geometry_min_short_edge,
                    geometry_min_area: config.suspicious_geometry_min_area,
                    geometry_max_aspect_ratio: config.suspicious_geometry_max_aspect_ratio,
                    geometry_max_aspect_delta: config.suspicious_geometry_max_aspect_delta,
                    geometry_max_width_delta: config.suspicious_geometry_max_width_delta,
                    geometry_max_height_delta: config.suspicious_geometry_max_height_delta,
                    geometry_enable_affine: config.geometry_enable_affine,
                    geometry_enable_homography: config.geometry_enable_homography,
                    geometry_model_slack: config.geometry_model_slack,
                    geometry_max_anisotropy: config.geometry_max_anisotropy,
                    geometry_max_perspective: config.geometry_max_perspective,
                    geometry_affine_min_extra_inliers: config.geometry_affine_min_extra_inliers,
                    geometry_affine_min_extra_regions: config.geometry_affine_min_extra_regions,
                    geometry_affine_max_mean_residual: config.geometry_affine_max_mean_residual,
                    geometry_homography_min_extra_inliers: config
                        .geometry_homography_min_extra_inliers,
                    geometry_homography_min_extra_regions: config
                        .geometry_homography_min_extra_regions,
                    geometry_homography_max_mean_residual: config
                        .geometry_homography_max_mean_residual,
                    geometry_ratio_min_margin: config.geometry_ratio_min_margin,
                    geometry_enable_prosac_fallback: config.geometry_enable_prosac_fallback,
                    geometry_prosac_max_iters: config.geometry_prosac_max_iters,
                    geometry_prosac_min_inliers: config.geometry_prosac_min_inliers,
                    visual_shape: true,
                    visual_shape_min_signals: config.visual_shape_min_signals,
                    visual_shape_min_text_grid_mean: config.visual_shape_min_text_grid_mean,
                    visual_shape_max_text_grid_mean: config.visual_shape_max_text_grid_mean,
                    visual_shape_min_text_regions: config.visual_shape_min_text_regions,
                    visual_shape_min_luma_mean: config.visual_shape_min_luma_mean,
                    visual_shape_max_luma_mean: config.visual_shape_max_luma_mean,
                    visual_shape_min_luma_std: config.visual_shape_min_luma_std,
                    visual_shape_max_luma_std: config.visual_shape_max_luma_std,
                    visual_shape_min_local_hashes: config.visual_shape_min_local_hashes,
                    visual_shape_min_middle_text_percent: config
                        .visual_shape_min_middle_text_percent,
                    visual_shape_min_center_text_percent: config
                        .visual_shape_min_center_text_percent,
                    visual_shape_max_center_text_percent: config
                        .visual_shape_max_center_text_percent,
                    visual_shape_max_edge_text_percent: config.visual_shape_max_edge_text_percent,
                    visual_shape_sparse_max_luma_mean: config.visual_shape_sparse_max_luma_mean,
                    visual_shape_max_rgb_spread: config.visual_shape_max_rgb_spread,
                    visual_shape_sparse_max_text_grid_mean: config
                        .visual_shape_sparse_max_text_grid_mean,
                    visual_shape_sparse_min_local_hashes: config
                        .visual_shape_sparse_min_local_hashes,
                },
                actions: DetectionActions {
                    delete_message: false,
                    remove_user_roles: false,
                    timeout_user: false,
                    timeout_seconds: 300,
                    ban_user: false,
                    ban_delete_message_seconds: 0,
                    kick_user: false,
                    add_to_specimens: false,
                },
            },
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.confirmed.validate("confirmed")?;
        self.suspicious.validate("suspicious")?;
        Ok(())
    }

    fn validate_against_hyperparameters(
        &self,
        hyperparameters: &DetectionHyperparameters,
    ) -> Result<()> {
        self.confirmed
            .threshold
            .validate_feature_caps("confirmed", hyperparameters.local_hash_cap)?;
        self.suspicious
            .threshold
            .validate_feature_caps("suspicious", hyperparameters.local_hash_cap)?;
        Ok(())
    }
}

impl TextGatePolicy {
    pub fn normalize(&mut self) {
        self.keywords = normalize_text_gate_patterns(&self.keywords);
        self.sentences = normalize_text_gate_patterns(&self.sentences);
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.keyword_threshold <= 32,
            "text_gate.keyword_threshold must be at most 32"
        );
        anyhow::ensure!(
            self.keyword_max_distance <= 3,
            "text_gate.keyword_max_distance must be at most 3"
        );
        anyhow::ensure!(
            self.keywords.len() <= 64,
            "text_gate.keywords must contain at most 64 entries"
        );
        anyhow::ensure!(
            self.sentences.len() <= 32,
            "text_gate.sentences must contain at most 32 entries"
        );
        anyhow::ensure!(
            !self.enabled || !self.keywords.is_empty() || !self.sentences.is_empty(),
            "text_gate must contain at least one keyword or sentence when enabled"
        );
        for keyword in &self.keywords {
            validate_text_gate_pattern(keyword, "text_gate.keywords")?;
        }
        for sentence in &self.sentences {
            validate_text_gate_pattern(sentence, "text_gate.sentences")?;
        }
        Ok(())
    }
}

impl DetectionRule {
    fn validate(&self, name: &str) -> Result<()> {
        self.threshold.validate(name)?;
        self.actions.validate(name)?;
        Ok(())
    }
}

impl DetectionThreshold {
    fn validate(&self, name: &str) -> Result<()> {
        anyhow::ensure!(
            self.exact_xxh128 || self.perceptual_hash || self.local_anchors || self.visual_shape,
            "{name} must enable at least one detection method"
        );
        for (field, value, max) in [
            ("score_threshold", self.score_threshold, 1_000.0),
            (
                "perceptual_score_weight",
                self.perceptual_score_weight,
                10.0,
            ),
            (
                "local_anchor_score_weight",
                self.local_anchor_score_weight,
                10.0,
            ),
            (
                "dense_local_anchor_score_weight",
                self.dense_local_anchor_score_weight,
                10.0,
            ),
            (
                "visual_signature_score_weight",
                self.visual_signature_score_weight,
                1.0,
            ),
            (
                "visual_shape_score_weight",
                self.visual_shape_score_weight,
                10.0,
            ),
            ("visual_shape_score_cap", self.visual_shape_score_cap, 100.0),
            ("perceptual_score_floor", self.perceptual_score_floor, 100.0),
            (
                "local_score_full_spread",
                self.local_score_full_spread,
                10_000.0,
            ),
            (
                "visual_shape_score_full",
                self.visual_shape_score_full,
                10_000.0,
            ),
        ] {
            anyhow::ensure!(
                value.is_finite() && (0.0..=max).contains(&value),
                "{name}.{field} must be finite and between 0 and {max}"
            );
        }
        anyhow::ensure!(
            self.local_score_full_hits <= 20_000,
            "{name}.local_score_full_hits must be at most 20000"
        );
        anyhow::ensure!(
            self.local_score_full_regions <= 64,
            "{name}.local_score_full_regions must be at most 64"
        );
        self.validate_cluster(name)?;
        validate_perceptual_hash_distances(
            self.phash64_max_distance,
            self.dhash64_max_distance,
            self.perceptual_hash_max_total_distance,
            name,
        )?;
        anyhow::ensure!(
            self.perceptual_visual_support_distance_slack <= 16,
            "{name}.perceptual_visual_support_distance_slack must be at most 16"
        );
        anyhow::ensure!(
            self.min_anchor_hits <= 4_096,
            "{name}.min_anchor_hits must be at most 4096"
        );
        anyhow::ensure!(
            self.min_distinct_regions <= 64,
            "{name}.min_distinct_regions must be at most 64"
        );
        anyhow::ensure!(
            self.max_mean_distance.is_finite() && (0.0..=15.0).contains(&self.max_mean_distance),
            "{name}.max_mean_distance must be finite and between 0 and 15"
        );
        anyhow::ensure!(
            self.local_unverified_support_min_anchor_hits <= 4_096,
            "{name}.local_unverified_support_min_anchor_hits must be at most 4096"
        );
        anyhow::ensure!(
            self.local_unverified_support_min_distinct_regions <= 64,
            "{name}.local_unverified_support_min_distinct_regions must be at most 64"
        );
        anyhow::ensure!(
            self.local_unverified_support_min_retention_permille <= 1_000,
            "{name}.local_unverified_support_min_retention_permille must be at most 1000"
        );
        anyhow::ensure!(
            self.local_unverified_support_max_mean_distance.is_finite()
                && (0.0..=15.0).contains(&self.local_unverified_support_max_mean_distance),
            "{name}.local_unverified_support_max_mean_distance must be finite and between 0 and 15"
        );
        anyhow::ensure!(
            self.local_unverified_support_max_perceptual_total_distance <= 96,
            "{name}.local_unverified_support_max_perceptual_total_distance must be at most 96"
        );
        anyhow::ensure!(
            self.local_unverified_support_max_aspect_delta.is_finite()
                && (0.0..=2.0).contains(&self.local_unverified_support_max_aspect_delta),
            "{name}.local_unverified_support_max_aspect_delta must be finite and between 0 and 2"
        );
        anyhow::ensure!(
            self.local_unverified_support_max_dimension_delta
                .is_finite()
                && (0.0..=2.0).contains(&self.local_unverified_support_max_dimension_delta),
            "{name}.local_unverified_support_max_dimension_delta must be finite and between 0 and 2"
        );
        anyhow::ensure!(
            self.geometry_min_short_edge <= 4096,
            "{name}.geometry_min_short_edge must be at most 4096"
        );
        anyhow::ensure!(
            self.geometry_min_area <= 50_000_000,
            "{name}.geometry_min_area must be at most 50000000"
        );
        anyhow::ensure!(
            self.geometry_max_aspect_ratio.is_finite()
                && (1.0..=16.0).contains(&self.geometry_max_aspect_ratio),
            "{name}.geometry_max_aspect_ratio must be finite and between 1 and 16"
        );
        anyhow::ensure!(
            self.geometry_max_aspect_delta.is_finite()
                && (0.0..=16.0).contains(&self.geometry_max_aspect_delta),
            "{name}.geometry_max_aspect_delta must be finite and between 0 and 16"
        );
        anyhow::ensure!(
            self.geometry_max_width_delta.is_finite()
                && (0.0..=10.0).contains(&self.geometry_max_width_delta),
            "{name}.geometry_max_width_delta must be finite and between 0 and 10"
        );
        anyhow::ensure!(
            self.geometry_max_height_delta.is_finite()
                && (0.0..=10.0).contains(&self.geometry_max_height_delta),
            "{name}.geometry_max_height_delta must be finite and between 0 and 10"
        );
        anyhow::ensure!(
            self.geometry_model_slack.is_finite()
                && (0.0..=32.0).contains(&self.geometry_model_slack),
            "{name}.geometry_model_slack must be finite and between 0 and 32"
        );
        anyhow::ensure!(
            self.geometry_max_anisotropy.is_finite()
                && (1.0..=8.0).contains(&self.geometry_max_anisotropy),
            "{name}.geometry_max_anisotropy must be finite and between 1 and 8"
        );
        anyhow::ensure!(
            self.geometry_max_perspective.is_finite()
                && (1.0..=8.0).contains(&self.geometry_max_perspective),
            "{name}.geometry_max_perspective must be finite and between 1 and 8"
        );
        anyhow::ensure!(
            self.geometry_affine_min_extra_inliers <= 16
                && self.geometry_homography_min_extra_inliers <= 16,
            "{name} geometry model extra inliers must be at most 16"
        );
        anyhow::ensure!(
            self.geometry_affine_min_extra_regions <= 16
                && self.geometry_homography_min_extra_regions <= 16,
            "{name} geometry model extra regions must be at most 16"
        );
        anyhow::ensure!(
            self.geometry_affine_max_mean_residual.is_finite()
                && (0.0..=64.0).contains(&self.geometry_affine_max_mean_residual)
                && self.geometry_homography_max_mean_residual.is_finite()
                && (0.0..=64.0).contains(&self.geometry_homography_max_mean_residual),
            "{name} geometry model residual caps must be finite and between 0 and 64"
        );
        anyhow::ensure!(
            self.geometry_ratio_min_margin <= 64,
            "{name}.geometry_ratio_min_margin must be at most 64"
        );
        anyhow::ensure!(
            self.geometry_prosac_max_iters <= 10_000,
            "{name}.geometry_prosac_max_iters must be at most 10000"
        );
        anyhow::ensure!(
            (4..=64).contains(&self.geometry_prosac_min_inliers),
            "{name}.geometry_prosac_min_inliers must be between 4 and 64"
        );
        anyhow::ensure!(
            (1..=8).contains(&self.visual_shape_min_signals),
            "{name}.visual_shape_min_signals must be between 1 and 8"
        );
        anyhow::ensure!(
            self.visual_shape_min_text_grid_mean <= self.visual_shape_max_text_grid_mean,
            "{name}.visual_shape_min_text_grid_mean must be less than or equal to visual_shape_max_text_grid_mean"
        );
        anyhow::ensure!(
            self.visual_shape_min_text_regions <= 64,
            "{name}.visual_shape_min_text_regions must be at most 64"
        );
        anyhow::ensure!(
            self.visual_shape_min_luma_mean <= self.visual_shape_max_luma_mean,
            "{name}.visual_shape_min_luma_mean must be less than or equal to visual_shape_max_luma_mean"
        );
        anyhow::ensure!(
            self.visual_shape_min_luma_std <= self.visual_shape_max_luma_std,
            "{name}.visual_shape_min_luma_std must be less than or equal to visual_shape_max_luma_std"
        );
        anyhow::ensure!(
            self.visual_shape_min_local_hashes <= 5_000,
            "{name}.visual_shape_min_local_hashes must be at most 5000"
        );
        anyhow::ensure!(
            self.visual_shape_min_middle_text_percent <= 100
                && self.visual_shape_min_center_text_percent <= 100
                && self.visual_shape_max_center_text_percent <= 100
                && self.visual_shape_max_edge_text_percent <= 100,
            "{name}.visual_shape text distribution percentages must be at most 100"
        );
        anyhow::ensure!(
            self.visual_shape_min_center_text_percent <= self.visual_shape_max_center_text_percent,
            "{name}.visual_shape_min_center_text_percent must be less than or equal to visual_shape_max_center_text_percent"
        );
        anyhow::ensure!(
            self.visual_shape_sparse_max_text_grid_mean >= self.visual_shape_min_text_grid_mean,
            "{name}.visual_shape_sparse_max_text_grid_mean must be at least visual_shape_min_text_grid_mean"
        );
        anyhow::ensure!(
            self.visual_shape_sparse_min_local_hashes <= 5_000,
            "{name}.visual_shape_sparse_min_local_hashes must be at most 5000"
        );
        Ok(())
    }

    fn validate_feature_caps(&self, name: &str, local_hash_cap: usize) -> Result<()> {
        anyhow::ensure!(
            self.visual_shape_min_local_hashes <= local_hash_cap,
            "{name}.visual_shape_min_local_hashes must be at most local_hash_cap"
        );
        anyhow::ensure!(
            self.visual_shape_sparse_min_local_hashes <= local_hash_cap,
            "{name}.visual_shape_sparse_min_local_hashes must be at most local_hash_cap"
        );
        Ok(())
    }

    fn validate_cluster(&self, name: &str) -> Result<()> {
        anyhow::ensure!(
            self.cluster_member_score > self.cluster_chrome_ceiling_score,
            "{name}.cluster_member_score must be greater than cluster_chrome_ceiling_score"
        );
        anyhow::ensure!(
            self.cluster_hard_score <= 1_000,
            "{name}.cluster_hard_score must be at most 1000"
        );
        anyhow::ensure!(
            self.cluster_coherence_score <= 1_000,
            "{name}.cluster_coherence_score must be at most 1000"
        );
        anyhow::ensure!(
            (2..=16).contains(&self.cluster_min_size),
            "{name}.cluster_min_size must be between 2 and 16"
        );
        anyhow::ensure!(
            self.cluster_coverage_floor_permille <= 1_000,
            "{name}.cluster_coverage_floor_permille must be at most 1000"
        );
        Ok(())
    }
}

fn validate_perceptual_hash_distances(
    phash64_max_distance: u32,
    dhash64_max_distance: u32,
    max_total_distance: u32,
    name: &str,
) -> Result<()> {
    anyhow::ensure!(
        phash64_max_distance <= 16 && dhash64_max_distance <= 16,
        "{name} perceptual pHash/dHash distances must be at most 16"
    );
    anyhow::ensure!(
        phash64_max_distance < 16 || dhash64_max_distance < 16,
        "{name} perceptual pHash/dHash distances must keep at least one cap below 16 for indexed lookup"
    );
    anyhow::ensure!(
        max_total_distance <= phash64_max_distance + dhash64_max_distance,
        "{name}.perceptual_hash_max_total_distance must be at most the sum of the individual caps"
    );
    Ok(())
}

fn normalize_optional_id(value: &mut Option<String>) {
    *value = value
        .as_ref()
        .map(|id| id.trim())
        .filter(|id| !id.is_empty())
        .map(str::to_owned);
}

fn normalize_id_list(values: &mut Vec<String>) {
    let mut normalized = Vec::with_capacity(values.len());
    for value in values.iter().map(|value| value.trim()) {
        if !value.is_empty() && !normalized.iter().any(|existing| existing == value) {
            normalized.push(value.to_owned());
        }
    }
    *values = normalized;
}

impl DetectionActions {
    fn validate(&self, name: &str) -> Result<()> {
        anyhow::ensure!(
            !self.timeout_user || TIMEOUT_DURATION_SECONDS.contains(&self.timeout_seconds),
            "{name}.timeout_seconds must be one of 60, 300, 600, 3600, 86400, or 604800 when timeout_user is enabled"
        );
        anyhow::ensure!(
            self.ban_delete_message_seconds <= 604_800,
            "{name}.ban_delete_message_seconds must be at most 604800"
        );
        Ok(())
    }
}

impl GuildConfigRecord {
    pub fn new(mut config: GuildConfig) -> Self {
        config.normalize();
        Self {
            schema: GUILD_CONFIG_RECORD_SCHEMA,
            kind: "guild.config.set".to_owned(),
            created_at: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
            guild_id: config.guild_id.clone(),
            config,
            sig: None,
        }
    }

    pub fn sign(mut self, secret: &str) -> Result<Self> {
        self.sig = Some(sign_record(&self, secret)?);
        Ok(self)
    }

    pub fn verify(&self, secret: &str) -> Result<bool> {
        verify_record_signature(self, self.sig.as_deref(), secret)
    }
}

impl GuildConfigManifest {
    pub(crate) fn for_record(
        record: &GuildConfigRecord,
        attachment: &GuildConfigRecordAttachment,
    ) -> Self {
        Self {
            schema: GUILD_CONFIG_MANIFEST_SCHEMA,
            kind: "guild.config.set.manifest".to_owned(),
            guild_id: record.guild_id.clone(),
            record_attachment: attachment.filename.clone(),
            record_bytes: u32::try_from(attachment.bytes.len()).unwrap_or(u32::MAX),
            record_sha256: sha256_hex(&attachment.bytes),
            sig: None,
        }
    }

    pub(crate) fn sign(mut self, secret: &str) -> Result<Self> {
        self.sig = Some(sign_manifest(&self, secret)?);
        Ok(self)
    }

    fn verify(&self, secret: &str) -> Result<bool> {
        verify_manifest_signature(self, self.sig.as_deref(), secret)
    }

    fn validate(&self, expected_guild_id: Id<GuildMarker>) -> Result<()> {
        anyhow::ensure!(
            self.schema == GUILD_CONFIG_MANIFEST_SCHEMA,
            "unsupported guild config manifest schema {}",
            self.schema
        );
        anyhow::ensure!(
            self.kind == "guild.config.set.manifest",
            "unsupported guild config manifest type {}",
            self.kind
        );
        anyhow::ensure!(
            self.guild_id == expected_guild_id.get().to_string(),
            "guild config manifest guild_id {} does not match configured guild {}",
            self.guild_id,
            expected_guild_id.get()
        );
        anyhow::ensure!(
            !self.record_attachment.trim().is_empty(),
            "guild config manifest record_attachment must not be empty"
        );
        anyhow::ensure!(
            usize::try_from(self.record_bytes).unwrap_or(usize::MAX)
                <= MAX_CONFIG_RECORD_ATTACHMENT_BYTES,
            "guild config manifest record attachment is too large"
        );
        validate_hex(
            &self.record_sha256,
            32,
            "guild config manifest record_sha256",
        )?;
        Ok(())
    }
}

pub(crate) fn parse_and_verify_config_manifest(
    manifest_raw: &str,
    secret: &str,
    expected_guild_id: Id<GuildMarker>,
) -> Result<Option<GuildConfigManifest>> {
    let Some(manifest) = decode_config::<GuildConfigManifest>(manifest_raw)? else {
        return Ok(None);
    };

    if !manifest.verify(secret)? {
        return Ok(None);
    }
    manifest.validate(expected_guild_id)?;
    Ok(Some(manifest))
}

pub(crate) fn parse_and_verify_config_record(
    manifest: &GuildConfigManifest,
    record_bytes: &[u8],
    secret: &str,
    expected_guild_id: Id<GuildMarker>,
) -> Result<GuildConfigRecord> {
    verify_record_attachment(manifest, record_bytes)?;

    let mut record: GuildConfigRecord =
        rmp_serde::from_slice(record_bytes).context("deserializing guild config attachment")?;
    if record.schema != GUILD_CONFIG_RECORD_SCHEMA || record.kind != "guild.config.set" {
        return Err(anyhow!("unsupported guild config record"));
    }
    anyhow::ensure!(
        record.guild_id == manifest.guild_id,
        "guild config record guild_id does not match manifest"
    );
    anyhow::ensure!(
        record.guild_id == record.config.guild_id,
        "guild config record guild_id does not match embedded config"
    );

    if !record.verify(secret)? {
        return Err(anyhow!("invalid guild config record signature"));
    }
    record.config.normalize();
    record.config.validate(expected_guild_id)?;
    Ok(record)
}

pub(crate) fn config_record_attachment(
    record: &GuildConfigRecord,
) -> Result<GuildConfigRecordAttachment> {
    let bytes = rmp_serde::to_vec(record).context("serializing guild config attachment")?;
    anyhow::ensure!(
        bytes.len() <= MAX_CONFIG_RECORD_ATTACHMENT_BYTES,
        "guild config attachment exceeds storage limit"
    );
    Ok(GuildConfigRecordAttachment {
        filename: "guild_config.sightline.msgpack".to_owned(),
        bytes,
    })
}

pub(crate) fn config_manifest_to_discord(manifest: &GuildConfigManifest) -> Result<String> {
    encode_config(manifest).context("serializing guild config storage manifest")
}

pub(crate) fn signed_config_manifest_to_discord(
    record: &GuildConfigRecord,
    attachment: &GuildConfigRecordAttachment,
    secret: &str,
) -> Result<String> {
    let manifest = GuildConfigManifest::for_record(record, attachment).sign(secret)?;
    config_manifest_to_discord(&manifest)
}

pub(crate) fn threshold_summary(threshold: &DetectionThreshold) -> String {
    format!(
        "score={:.1} weights(p/local/dense/shape)={:.2}/{:.2}/{:.2}/{:.2} visual_sig_w={:.2} caps(shape)={:.1} score_shape(ph_floor={:.1},local_full={}/{}/{:.1},shape_full={:.1}), exact=`{}`, perceptual=`{}`/{}/{}/sum{} support_slack={}, anchors=`{}`/{}/{}/{:.1}, unverified_support=`{}`/{}/{}/ret{}permille/{:.2}/sum{}/geo{:.2}/{:.2}, local_candidate_filter=`{}/{}/{}/{}`, visual_zero_score=`{}/{}/{}/{}`, geometry=`{}/{}/{:.2}/{:.2}/w{:.2}/h{:.2}`, geo_models=`{}`/`{}` slack={:.1} aniso={:.1} persp={:.1} affine=+{}/+{}/{:.1} homography=+{}/+{}/{:.1} ratio_margin={} prosac=`{}`/{}/{}, cluster=`{}` hard={} chrome={} member={} coherence={} size={} coverage={}, visual_shape=`{}`/{}/{}-{}/{}/{}-{}:{}-{}/{} mid/center/edge={}/{}-{}/{} rgb_spread={} sparse={}/{}/{}",
        threshold.score_threshold,
        threshold.perceptual_score_weight,
        threshold.local_anchor_score_weight,
        threshold.dense_local_anchor_score_weight,
        threshold.visual_shape_score_weight,
        threshold.visual_signature_score_weight,
        threshold.visual_shape_score_cap,
        threshold.perceptual_score_floor,
        threshold.local_score_full_hits,
        threshold.local_score_full_regions,
        threshold.local_score_full_spread,
        threshold.visual_shape_score_full,
        threshold.exact_xxh128,
        threshold.perceptual_hash,
        threshold.phash64_max_distance,
        threshold.dhash64_max_distance,
        threshold.perceptual_hash_max_total_distance,
        threshold.perceptual_visual_support_distance_slack,
        threshold.local_anchors,
        threshold.min_anchor_hits,
        threshold.min_distinct_regions,
        threshold.max_mean_distance,
        threshold.local_unverified_support,
        threshold.local_unverified_support_min_anchor_hits,
        threshold.local_unverified_support_min_distinct_regions,
        threshold.local_unverified_support_min_retention_permille,
        threshold.local_unverified_support_max_mean_distance,
        threshold.local_unverified_support_max_perceptual_total_distance,
        threshold.local_unverified_support_max_aspect_delta,
        threshold.local_unverified_support_max_dimension_delta,
        threshold.local_luma_candidate_max_delta,
        threshold.local_contrast_candidate_max_delta,
        threshold.local_edge_density_candidate_max_delta,
        threshold.local_position_candidate_max_delta,
        threshold.visual_luma_zero_score_delta,
        threshold.visual_color_zero_score_delta,
        threshold.visual_grid_luma_zero_score_delta,
        threshold.visual_text_grid_zero_score_delta,
        threshold.geometry_min_short_edge,
        threshold.geometry_min_area,
        threshold.geometry_max_aspect_ratio,
        threshold.geometry_max_aspect_delta,
        threshold.geometry_max_width_delta,
        threshold.geometry_max_height_delta,
        threshold.geometry_enable_affine,
        threshold.geometry_enable_homography,
        threshold.geometry_model_slack,
        threshold.geometry_max_anisotropy,
        threshold.geometry_max_perspective,
        threshold.geometry_affine_min_extra_inliers,
        threshold.geometry_affine_min_extra_regions,
        threshold.geometry_affine_max_mean_residual,
        threshold.geometry_homography_min_extra_inliers,
        threshold.geometry_homography_min_extra_regions,
        threshold.geometry_homography_max_mean_residual,
        threshold.geometry_ratio_min_margin,
        threshold.geometry_enable_prosac_fallback,
        threshold.geometry_prosac_max_iters,
        threshold.geometry_prosac_min_inliers,
        threshold.cluster_coherence,
        threshold.cluster_hard_score,
        threshold.cluster_chrome_ceiling_score,
        threshold.cluster_member_score,
        threshold.cluster_coherence_score,
        threshold.cluster_min_size,
        threshold.cluster_coverage_floor_permille,
        threshold.visual_shape,
        threshold.visual_shape_min_signals,
        threshold.visual_shape_min_text_grid_mean,
        threshold.visual_shape_max_text_grid_mean,
        threshold.visual_shape_min_text_regions,
        threshold.visual_shape_min_luma_mean,
        threshold.visual_shape_max_luma_mean,
        threshold.visual_shape_min_luma_std,
        threshold.visual_shape_max_luma_std,
        threshold.visual_shape_min_local_hashes,
        threshold.visual_shape_min_middle_text_percent,
        threshold.visual_shape_min_center_text_percent,
        threshold.visual_shape_max_center_text_percent,
        threshold.visual_shape_max_edge_text_percent,
        threshold.visual_shape_max_rgb_spread,
        threshold.visual_shape_sparse_max_luma_mean,
        threshold.visual_shape_sparse_max_text_grid_mean,
        threshold.visual_shape_sparse_min_local_hashes
    )
}

pub(crate) fn actions_summary(actions: &DetectionActions) -> String {
    format!(
        "delete=`{}`, remove_roles=`{}`, timeout=`{}`/{}s, ban=`{}`, ban_delete_seconds=`{}`, kick=`{}`, add_to_specimens=`{}`",
        actions.delete_message,
        actions.remove_user_roles,
        actions.timeout_user,
        actions.timeout_seconds,
        actions.ban_user,
        actions.ban_delete_message_seconds,
        actions.kick_user,
        actions.add_to_specimens
    )
}

pub(crate) fn text_gate_policy_summary(policy: &TextGatePolicy) -> String {
    format!(
        "enabled=`{}`, keyword_threshold=`{}`, keyword_max_distance=`{}`, keywords=`{}`, sentences=`{}`",
        policy.enabled,
        policy.keyword_threshold,
        policy.keyword_max_distance,
        policy.keywords.len(),
        policy.sentences.len()
    )
}

pub(crate) fn scan_policy_summary(scan_policy: &ScanPolicy) -> String {
    format!(
        "extensions=`{}`, max_file_bytes=`{}`, exempt_administrators=`{}`, mark_message_siblings_suspicious=`{}`",
        scan_policy.allowed_extensions.join(","),
        scan_policy.max_file_bytes,
        scan_policy.exempt_administrators,
        scan_policy.mark_message_siblings_suspicious
    )
}

pub(crate) fn detection_hyperparameters_summary(
    hyperparameters: &DetectionHyperparameters,
) -> String {
    format!(
        "orientation=`{}` max={:.1} step={:.1} gain={:.2}, anchors=`{}`, normalize=`{}x{}`/area `{}`/aspect `{:.1}`, tile=`{}x{}`/stride `{}`, budget=`{}`, hash_cap=`{}`, anchor_count=`{}`, anchor_distance=`{}`",
        hyperparameters.perceptual_orientation_correction,
        hyperparameters.perceptual_orientation_max_degrees,
        hyperparameters.perceptual_orientation_step_degrees,
        hyperparameters.perceptual_orientation_min_gain,
        hyperparameters.local_anchors_enabled,
        hyperparameters.local_max_width,
        hyperparameters.local_max_height,
        hyperparameters.local_max_area,
        hyperparameters.local_max_aspect_ratio,
        hyperparameters.local_tile_width,
        hyperparameters.local_tile_height,
        hyperparameters.local_stride,
        hyperparameters.local_tile_budget,
        hyperparameters.local_hash_cap,
        hyperparameters.local_anchor_count,
        hyperparameters.local_anchor_max_distance
    )
}

fn sign_record(record: &GuildConfigRecord, secret: &str) -> Result<String> {
    if secret.is_empty() {
        return Err(anyhow!("empty HMAC secret"));
    }

    let mut unsigned = record.clone();
    unsigned.sig = None;
    let payload =
        serde_json::to_vec(&unsigned).context("serializing unsigned guild config record")?;
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).context("creating hmac")?;
    mac.update(&payload);
    Ok(hex::encode(mac.finalize().into_bytes()))
}

fn sign_manifest(manifest: &GuildConfigManifest, secret: &str) -> Result<String> {
    if secret.is_empty() {
        return Err(anyhow!("empty HMAC secret"));
    }

    let mut unsigned = manifest.clone();
    unsigned.sig = None;
    let payload =
        serde_json::to_vec(&unsigned).context("serializing unsigned guild config manifest")?;
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).context("creating hmac")?;
    mac.update(&payload);
    Ok(hex::encode(mac.finalize().into_bytes()))
}

fn verify_record_signature(
    record: &GuildConfigRecord,
    signature: Option<&str>,
    secret: &str,
) -> Result<bool> {
    let Some(signature) = signature else {
        return Ok(false);
    };
    let Ok(signature) = hex::decode(signature) else {
        return Ok(false);
    };
    if secret.is_empty() {
        return Err(anyhow!("empty HMAC secret"));
    }

    let mut unsigned = record.clone();
    unsigned.sig = None;
    let payload =
        serde_json::to_vec(&unsigned).context("serializing unsigned guild config record")?;
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).context("creating hmac")?;
    mac.update(&payload);
    Ok(mac.verify_slice(&signature).is_ok())
}

fn verify_manifest_signature(
    manifest: &GuildConfigManifest,
    signature: Option<&str>,
    secret: &str,
) -> Result<bool> {
    let Some(signature) = signature else {
        return Ok(false);
    };
    let Ok(signature) = hex::decode(signature) else {
        return Ok(false);
    };
    if secret.is_empty() {
        return Err(anyhow!("empty HMAC secret"));
    }

    let mut unsigned = manifest.clone();
    unsigned.sig = None;
    let payload =
        serde_json::to_vec(&unsigned).context("serializing unsigned guild config manifest")?;
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).context("creating hmac")?;
    mac.update(&payload);
    Ok(mac.verify_slice(&signature).is_ok())
}

fn verify_record_attachment(manifest: &GuildConfigManifest, record_bytes: &[u8]) -> Result<()> {
    anyhow::ensure!(
        record_bytes.len() == usize::try_from(manifest.record_bytes).unwrap_or(usize::MAX),
        "guild config attachment size does not match manifest"
    );
    anyhow::ensure!(
        sha256_hex(record_bytes) == manifest.record_sha256,
        "guild config attachment digest does not match manifest"
    );
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn validate_hex(value: &str, bytes: usize, name: &str) -> Result<()> {
    let decoded = hex::decode(value).with_context(|| format!("{name} must be hex"))?;
    anyhow::ensure!(decoded.len() == bytes, "{name} must be {bytes} bytes");
    Ok(())
}

fn parse_id<T>(raw: &str) -> Result<Id<T>> {
    raw.parse::<u64>()
        .map(Id::new)
        .context("parsing configured snowflake")
}

fn parse_id_opt<T>(raw: &str) -> Option<Id<T>> {
    raw.parse::<u64>().ok().map(Id::new)
}

fn validate_optional_id<T>(raw: &Option<String>, name: &str) -> Result<()> {
    if let Some(raw) = raw {
        parse_id::<T>(raw).with_context(|| format!("{name} is invalid"))?;
    }
    Ok(())
}

fn validate_id_list<T>(values: &[String], name: &str) -> Result<()> {
    anyhow::ensure!(values.len() <= 25, "{name} must contain at most 25 IDs");
    for value in values {
        parse_id::<T>(value).with_context(|| format!("{name} contains invalid snowflake"))?;
    }
    Ok(())
}

fn validate_text_gate_pattern(value: &str, name: &str) -> Result<()> {
    let trimmed = value.trim();
    anyhow::ensure!(
        !trimmed.is_empty() && trimmed.len() <= 160,
        "{name} entries must be 1-160 bytes after trimming"
    );
    anyhow::ensure!(
        !trimmed.chars().any(char::is_control),
        "{name} entries must not contain control characters"
    );
    Ok(())
}

pub(crate) fn normalize_text_gate_pattern(raw: &str) -> String {
    raw.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn normalize_text_gate_patterns(values: &[String]) -> Vec<String> {
    let mut normalized = Vec::with_capacity(values.len());
    for value in values {
        let value = normalize_text_gate_pattern(value);
        if !value.is_empty() && !normalized.iter().any(|existing| existing == &value) {
            normalized.push(value);
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use twilight_model::id::Id;

    #[test]
    fn msgpack_legacy_struct_field_round_trips() {
        #[derive(Serialize, Deserialize)]
        struct Compat {
            a: u8,
            #[serde(default, rename = "removed")]
            legacy_removed: bool,
            b: u8,
        }

        #[derive(Serialize)]
        struct Old {
            a: u8,
            removed: bool,
            b: u8,
        }

        let bytes = rmp_serde::to_vec(&Compat {
            a: 1,
            legacy_removed: true,
            b: 2,
        })
        .unwrap();
        let decoded: Compat = rmp_serde::from_slice(&bytes).unwrap();

        assert_eq!(decoded.a, 1);
        assert!(decoded.legacy_removed);
        assert_eq!(decoded.b, 2);

        let legacy_bytes = rmp_serde::to_vec(&Old {
            a: 1,
            removed: true,
            b: 2,
        })
        .unwrap();
        let decoded: Compat = rmp_serde::from_slice(&legacy_bytes).unwrap();

        assert_eq!(decoded.a, 1);
        assert!(decoded.legacy_removed);
        assert_eq!(decoded.b, 2);
    }

    #[test]
    fn msgpack_missing_trailing_default_is_backward_compatible() {
        #[derive(Deserialize)]
        struct Current {
            a: u8,
            b: u8,
            #[serde(default)]
            added: bool,
        }

        #[derive(Serialize)]
        struct Legacy {
            a: u8,
            b: u8,
        }

        let bytes = rmp_serde::to_vec(&Legacy { a: 1, b: 2 }).unwrap();
        let decoded: Current = rmp_serde::from_slice(&bytes).unwrap();

        assert_eq!(decoded.a, 1);
        assert_eq!(decoded.b, 2);
        assert!(!decoded.added);
    }

    #[test]
    fn compact_config_storage_round_trips() {
        let config = GuildConfig::from_guild(Id::new(1), Id::new(2));
        let record = GuildConfigRecord::new(config).sign("secret").unwrap();

        let attachment = config_record_attachment(&record).unwrap();
        let encoded = signed_config_manifest_to_discord(&record, &attachment, "secret").unwrap();
        assert!(encoded.starts_with(crate::configuration::storage_codec::CONFIG_PREFIX));
        assert!(!encoded.contains('{'));
        assert!(
            encoded.len() <= 2_000,
            "signed guild config manifest is {} chars, over Discord's 2000-char limit",
            encoded.len()
        );

        let manifest = parse_and_verify_config_manifest(&encoded, "secret", Id::new(1))
            .unwrap()
            .unwrap();
        let decoded =
            parse_and_verify_config_record(&manifest, &attachment.bytes, "secret", Id::new(1))
                .unwrap();
        assert_eq!(decoded.schema, 2);
        assert_eq!(decoded.config.guild_id, "1");
        assert_eq!(decoded.config.ledger_channel_id, "2");
    }

    #[test]
    fn legacy_config_defaults_message_sibling_escalation_off() {
        let config = GuildConfig::from_guild(Id::new(1), Id::new(2));
        let mut value = toml::Value::try_from(&config).unwrap();
        value
            .get_mut("scan_policy")
            .and_then(toml::Value::as_table_mut)
            .unwrap()
            .remove("mark_message_siblings_suspicious");

        let raw = toml::to_string(&value).unwrap();
        let decoded: GuildConfig = toml::from_str(&raw).unwrap();

        assert!(!decoded.scan_policy.mark_message_siblings_suspicious);
    }

    #[test]
    fn config_record_normalizes_text_gate_patterns() {
        let mut config = GuildConfig::from_guild(Id::new(1), Id::new(2));
        config.text_gate_policy.keywords = vec![
            "CLAIM".to_owned(),
            "air-drop".to_owned(),
            "claim".to_owned(),
        ];
        config.text_gate_policy.sentences = vec!["Connect   YOUR Wallet!".to_owned()];

        let record = GuildConfigRecord::new(config);

        assert_eq!(
            record.config.text_gate_policy.keywords,
            vec!["claim".to_owned(), "air drop".to_owned()]
        );
        assert_eq!(
            record.config.text_gate_policy.sentences,
            vec!["connect your wallet".to_owned()]
        );
    }

    #[test]
    fn config_normalize_migrates_legacy_detection_log_copy() {
        let mut config = GuildConfig::from_guild(Id::new(1), Id::new(2));
        config.discord_detection_log_message = "<@&123>".to_owned();

        config.normalize();

        assert_eq!(config.discord_confirmed_log_message, "<@&123>");
        assert_eq!(config.discord_suspicious_log_message, "<@&123>");
        assert!(config.discord_benign_log_message.is_empty());
        assert!(config.discord_detection_log_message.is_empty());
    }

    #[test]
    fn enabled_text_gate_requires_a_pattern() {
        let mut policy = TextGatePolicy {
            enabled: true,
            ..TextGatePolicy::default()
        };

        let source = policy.validate().unwrap_err().to_string();
        assert!(source.contains("at least one keyword or sentence"));

        policy.keywords.push("crypto casino".to_owned());
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn detection_policy_visual_hash_floors_must_fit_local_hash_cap() {
        let mut policy = DetectionPolicy::default();
        let hyperparameters = DetectionHyperparameters {
            local_hash_cap: 100,
            ..DetectionHyperparameters::default()
        };
        policy.confirmed.threshold.visual_shape_min_local_hashes = 101;

        let source = policy
            .validate_against_hyperparameters(&hyperparameters)
            .unwrap_err()
            .to_string();
        assert!(source.contains("visual_shape_min_local_hashes"));

        policy.confirmed.threshold.visual_shape_min_local_hashes = 100;
        policy
            .confirmed
            .threshold
            .visual_shape_sparse_min_local_hashes = 101;
        let source = policy
            .validate_against_hyperparameters(&hyperparameters)
            .unwrap_err()
            .to_string();
        assert!(source.contains("visual_shape_sparse_min_local_hashes"));
    }
}
