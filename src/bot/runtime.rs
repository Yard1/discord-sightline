#![allow(clippy::struct_field_names, clippy::too_many_lines)]

use crate::{
    bot::{
        admin,
        discord::{
            ADD_SPECIMEN_COMMAND, AUDIT_COMMAND, BotLogColor, BotLogEvent, CONFIG_COMMAND,
            DOCTOR_COMMAND, DiscordStorageState, EXPORT_HASHES_COMMAND, IMPORT_HASHES_COMMAND,
            IMPORT_IMAGES_COMMAND, RenderedBotLog, STATS_COMMAND, StoredSpecimen,
            VALIDATE_MESSAGE_COMMAND, VERIFY_MESSAGE_COMMAND, check_bot_permissions,
            defer_ephemeral_interaction, defer_update_interaction, edit_interaction_response,
            edit_interaction_response_data, find_database_channel, load_ledger, message_jump_link,
            register_commands, render_bot_log, respond_interaction,
        },
        effects::{DiscordEffects, TwilightDiscordEffects},
        event_stream::{BotEventStream, TwilightShardEventStream},
        extract::{
            MAX_MANUAL_SPECIMEN_IMAGES_PER_INTERACTION, extract_candidates_from_message,
            message_has_any_role, message_has_potential_image,
        },
        ledger::{SpecimenImageAttachment, SpecimenRecord},
        specimen_import::import_image_candidates,
        worker::{
            DetectionFollowup, OcrFollowupInput, OriginalAutoAddInput, detection_followup_loop,
            ocr_followup_loop, original_auto_add_loop, worker_loop,
        },
    },
    configuration::{
        app::{AppConfig, DownloadConfig, Secrets, load_secrets},
        guild::{
            DetectionPolicy, GuildConfig, GuildConfigRecord, actions_summary,
            detection_hyperparameters_summary, scan_policy_summary, text_gate_policy_summary,
            threshold_summary,
        },
    },
    image::{
        matcher::{ExactHashIndex, Matcher, MatcherScratch},
        pipeline::{CpuGate, DownloadedImage, download_image, url_log_label},
        types::{
            HashOutcomeLruCache, ImageCandidate, ImageFingerprint, ImageFingerprintTimingSample,
            ImageMatchStageMetric, ImageMetricEvent, ImagePerfSample, ImagePerfTracker,
            ImageScanDecisionMetric, ImageStageTimingSample, LruDedupe, MatchOutcome,
            TextGateResolutionMetric, Xxh128,
        },
    },
    ocr_space::OcrSpaceClient,
};
use anyhow::{Context, Result, anyhow};
use arc_swap::ArcSwap;
use dashmap::{DashMap, DashSet, mapref::entry::Entry};
use futures_util::{StreamExt, stream};
use parking_lot::{Mutex as StdMutex, RwLock as StdRwLock};
use reqwest::{Client as ReqwestClient, redirect::Policy};
use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    signal,
    sync::{Mutex as AsyncMutex, OnceCell, Semaphore, mpsc, oneshot, watch},
    task::JoinSet,
};
use tokio_util::{sync::CancellationToken, task::TaskTracker};
use tracing::{debug, info, warn};
use twilight_gateway::{Config as GatewayConfig, Event, Intents};
use twilight_http::Client as DiscordClient;
use twilight_model::{
    application::interaction::InteractionData,
    channel::message::{AllowedMentions, Message, MessageFlags, embed::EmbedFooter},
    gateway::payload::incoming::InteractionCreate,
    guild::Permissions,
    http::interaction::{InteractionResponse, InteractionResponseData, InteractionResponseType},
    id::{
        Id,
        marker::{
            ApplicationMarker, ChannelMarker, GuildMarker, MessageMarker, RoleMarker, UserMarker,
        },
    },
};

use url::Url;

type GuildLoadLocks = DashMap<Id<GuildMarker>, Arc<AsyncMutex<()>>>;
type GuildLoadBackoff = DashMap<Id<GuildMarker>, Instant>;
type OcrSingleflightMap = DashMap<OcrCacheKey, Arc<OnceCell<crate::image::engine::TextGateReport>>>;
type HashProcessingMap = DashMap<OcrCacheKey, Arc<watch::Sender<bool>>>;
type SpecimenHitCounts = DashMap<String, u64>;
type MatchedMessageSet = DashSet<MessageScopeKey>;
type ConfirmedMessageMap = DashMap<MessageScopeKey, MatchOutcome>;
type SiblingInspectionMap = DashMap<MessageImageKey, MessageSiblingInspection>;
type LoggedSiblingInspectionSet = DashSet<MessageImageKey>;
type ImageByteStoreMap = DashMap<Xxh128, Weak<ImageByteLeaseInner>>;
type SharedDownloadResult = std::result::Result<DownloadedImage, Arc<str>>;
type DownloadSingleflightMap = DashMap<String, Arc<OnceCell<SharedDownloadResult>>>;

struct MatcherScratchPool {
    available: StdMutex<Vec<MatcherScratch>>,
    max_retained: usize,
}

#[derive(Clone, Copy)]
enum RuntimeMatchVariant {
    Original,
    Preview,
}

impl MatcherScratchPool {
    fn new(max_retained: usize) -> Self {
        Self {
            available: StdMutex::new(Vec::with_capacity(max_retained)),
            max_retained: max_retained.max(1),
        }
    }

    fn take(&self) -> MatcherScratch {
        self.available.lock().pop().unwrap_or_default()
    }

    fn put(&self, scratch: MatcherScratch) {
        let mut available = self.available.lock();
        if available.len() < self.max_retained {
            available.push(scratch);
        }
    }
}

const GUILD_LOAD_INFLIGHT_BACKOFF: Duration = Duration::from_secs(10);
const GUILD_LOAD_FAILURE_BACKOFF: Duration = Duration::from_secs(60);
const GUILD_HEALTH_CHECK_PERIOD: Duration = Duration::from_secs(60);
const GUILD_PERMISSION_REFRESH_TICKS: u64 = 5;
const IMAGE_POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(350);
const DISCORD_CDN_WARMER_HOST: &str = "cdn.discordapp.com";
const DISCORD_CDN_WARMER_URL: &str = "https://cdn.discordapp.com/embed/avatars/0.png";
const DISCORD_CDN_WARMER_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct OcrCacheKey {
    pub(crate) byte_xxh128: Xxh128,
    pub(crate) policy_hash: u64,
}

pub(crate) enum HashProcessingClaim {
    Owner {
        key: OcrCacheKey,
        complete: Arc<watch::Sender<bool>>,
    },
    Waiter(watch::Receiver<bool>),
}

struct DownloadSingleflightCleanup {
    map: Arc<DownloadSingleflightMap>,
    key: String,
    cell: Arc<OnceCell<SharedDownloadResult>>,
}

#[derive(Default)]
struct DownloadHostActivity {
    cdn_discordapp_com_epoch_secs: AtomicU64,
}

impl DownloadHostActivity {
    fn touch_url(&self, url: &str) {
        let Ok(parsed) = Url::parse(url) else {
            return;
        };
        if parsed
            .host_str()
            .is_some_and(|host| host.eq_ignore_ascii_case(DISCORD_CDN_WARMER_HOST))
        {
            self.touch_cdn();
        }
    }

    fn touch_cdn(&self) {
        self.cdn_discordapp_com_epoch_secs
            .store(epoch_seconds(), Ordering::Relaxed);
    }

    fn cdn_elapsed(&self) -> Duration {
        let touched = self.cdn_discordapp_com_epoch_secs.load(Ordering::Relaxed);
        Duration::from_secs(epoch_seconds().saturating_sub(touched))
    }
}

impl Drop for DownloadSingleflightCleanup {
    fn drop(&mut self) {
        self.map
            .remove_if(&self.key, |_, existing| Arc::ptr_eq(existing, &self.cell));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct MessageScopeKey {
    pub(crate) guild_id: Id<GuildMarker>,
    pub(crate) channel_id: Id<ChannelMarker>,
    pub(crate) message_id: Id<MessageMarker>,
}

impl MessageScopeKey {
    pub(crate) fn from_candidate(candidate: &ImageCandidate) -> Self {
        Self {
            guild_id: candidate.guild_id,
            channel_id: candidate.channel_id,
            message_id: candidate.message_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct MessageImageKey {
    pub(crate) scope: MessageScopeKey,
    pub(crate) image_url: String,
}

impl MessageImageKey {
    pub(crate) fn from_candidate(candidate: &ImageCandidate) -> Self {
        Self {
            scope: MessageScopeKey::from_candidate(candidate),
            image_url: candidate.url.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MessageSiblingInspection {
    pub(crate) candidate: ImageCandidate,
    pub(crate) image_id: String,
    pub(crate) decision: String,
    pub(crate) elapsed_ms: u128,
    pub(crate) trace_id: String,
    pub(crate) error: Option<String>,
}

#[derive(Clone)]
pub(crate) struct ImageByteLease {
    inner: Arc<ImageByteLeaseInner>,
}

struct ImageByteLeaseInner {
    key: Xxh128,
    bytes: bytes::Bytes,
    store: Weak<ImageByteStore>,
    _byte_permits: Vec<tokio::sync::OwnedSemaphorePermit>,
}

impl ImageByteLease {
    pub(crate) fn bytes(&self) -> bytes::Bytes {
        self.inner.bytes.clone()
    }
}

impl Drop for ImageByteLeaseInner {
    fn drop(&mut self) {
        if let Some(store) = self.store.upgrade() {
            store.remove_if_dead(self.key);
        }
    }
}

pub(crate) struct ImageByteStore {
    entries: ImageByteStoreMap,
    byte_budget: Arc<Semaphore>,
}

impl ImageByteStore {
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            entries: DashMap::new(),
            byte_budget: Arc::new(Semaphore::new(max_bytes.max(1))),
        }
    }

    pub(crate) fn insert(
        self: &Arc<Self>,
        key: Xxh128,
        bytes: bytes::Bytes,
    ) -> Option<ImageByteLease> {
        if let Some(existing) = self.get(key) {
            return Some(existing);
        }
        if bytes.is_empty() {
            return None;
        }
        let byte_permits = acquire_byte_permits(&self.byte_budget, bytes.len())?;
        if let Some(existing) = self.get(key) {
            return Some(existing);
        }
        let inner = Arc::new(ImageByteLeaseInner {
            key,
            bytes,
            store: Arc::downgrade(self),
            _byte_permits: byte_permits,
        });
        self.entries.insert(key, Arc::downgrade(&inner));
        Some(ImageByteLease { inner })
    }

    fn get(&self, key: Xxh128) -> Option<ImageByteLease> {
        let entry = self.entries.get(&key)?;
        let Some(inner) = entry.value().upgrade() else {
            drop(entry);
            self.entries.remove(&key);
            return None;
        };
        Some(ImageByteLease { inner })
    }

    fn remove_if_dead(&self, key: Xxh128) {
        self.entries
            .remove_if(&key, |_, value| value.strong_count() == 0);
    }
}

fn acquire_byte_permits(
    budget: &Arc<Semaphore>,
    byte_count: usize,
) -> Option<Vec<tokio::sync::OwnedSemaphorePermit>> {
    const MAX_PERMITS_PER_ACQUIRE: u32 = u32::MAX;
    let mut remaining = byte_count;
    let mut permits = Vec::with_capacity(remaining.div_ceil(u32::MAX as usize));
    while remaining > 0 {
        let permits_to_take =
            u32::try_from(remaining.min(MAX_PERMITS_PER_ACQUIRE as usize)).ok()?;
        let permit = budget
            .clone()
            .try_acquire_many_owned(permits_to_take)
            .ok()?;
        permits.push(permit);
        remaining -= permits_to_take as usize;
    }
    Some(permits)
}

#[derive(Clone)]
pub(crate) struct BotState {
    pub(crate) config: AppConfig,
    pub(crate) secrets: Secrets,
    pub(crate) bot_start_id: String,
    pub(crate) application_id: Id<ApplicationMarker>,
    pub(crate) bot_user_id: Id<UserMarker>,
    pub(crate) discord: Arc<DiscordClient>,
    pub(crate) discord_effects: Arc<dyn DiscordEffects>,
    pub(crate) image_http: ReqwestClient,
    pub(crate) ocr_space: Option<Arc<OcrSpaceClient>>,
    pub(crate) matcher_gate: Arc<CpuGate>,
    matcher_scratch_pool: Arc<MatcherScratchPool>,
    pub(crate) guilds: Arc<DashMap<Id<GuildMarker>, Arc<GuildRuntime>>>,
    pub(crate) guild_load_locks: Arc<GuildLoadLocks>,
    pub(crate) guild_load_backoff: Arc<GuildLoadBackoff>,
    pub(crate) download_gate: Arc<Semaphore>,
    pub(crate) download_memory_gate: Arc<Semaphore>,
    pub(crate) decoded_image_memory_gate: Arc<Semaphore>,
    pub(crate) download_singleflight: Arc<DownloadSingleflightMap>,
    download_host_activity: Arc<DownloadHostActivity>,
    pub(crate) decode_gate: Arc<CpuGate>,
    pub(crate) image_byte_store: Arc<ImageByteStore>,
    pub(crate) dedupe: Arc<StdMutex<LruDedupe>>,
    pub(crate) hash_outcome_cache: Arc<StdMutex<HashOutcomeLruCache>>,
    pub(crate) image_metrics_tx: mpsc::Sender<ImageMetricsCommand>,
    pub(crate) database_write_tx: mpsc::Sender<DatabaseWriteRequest>,
    pub(crate) bot_log_tx: mpsc::Sender<BotLogWriteRequest>,
    pub(crate) detection_followup_tx: mpsc::Sender<DetectionFollowup>,
    pub(crate) ocr_followup_tx: mpsc::Sender<OcrFollowupInput>,
    pub(crate) original_auto_add_tx: mpsc::Sender<OriginalAutoAddInput>,
    pub(crate) image_tx: mpsc::Sender<ImageCandidate>,
    pub(crate) shutdown: CancellationToken,
    pub(crate) background_tasks: TaskTracker,
    pub(crate) interaction_gate: Arc<Semaphore>,
    pub(crate) matched_messages: Arc<MatchedMessageSet>,
    pub(crate) confirmed_messages: Arc<ConfirmedMessageMap>,
    pub(crate) sibling_inspections: Arc<SiblingInspectionMap>,
    pub(crate) logged_sibling_inspections: Arc<LoggedSiblingInspectionSet>,
}

pub(crate) struct GuildRuntime {
    pub(crate) guild_id: Id<GuildMarker>,
    pub(crate) matcher: Arc<StdRwLock<Matcher>>,
    pub(crate) exact_hash_index: Arc<StdRwLock<ExactHashIndex>>,
    pub(crate) guild_config: Arc<ArcSwap<GuildConfig>>,
    pub(crate) detection_policy_hash: Arc<AtomicU64>,
    pub(crate) guild_configured: Arc<AtomicBool>,
    pub(crate) storage: Arc<StdRwLock<DiscordStorageState>>,
    pub(crate) safe_mode: Arc<AtomicBool>,
    pub(crate) permissions_ok: Arc<AtomicBool>,
    pub(crate) ocr_singleflight: Arc<OcrSingleflightMap>,
    pub(crate) hash_processing: Arc<HashProcessingMap>,
    pub(crate) specimen_hit_counts: Arc<SpecimenHitCounts>,
    pub(crate) scan_exempt_roles: Arc<ArcSwap<Vec<Id<RoleMarker>>>>,
    pub(crate) administrator_roles: Arc<ArcSwap<Vec<Id<RoleMarker>>>>,
}

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) bot: BotState,
    pub(crate) guild_id: Id<GuildMarker>,
    pub(crate) matcher: Arc<StdRwLock<Matcher>>,
    pub(crate) exact_hash_index: Arc<StdRwLock<ExactHashIndex>>,
    pub(crate) guild_config: Arc<ArcSwap<GuildConfig>>,
    pub(crate) detection_policy_hash: Arc<AtomicU64>,
    pub(crate) guild_configured: Arc<AtomicBool>,
    pub(crate) storage: Arc<StdRwLock<DiscordStorageState>>,
    pub(crate) safe_mode: Arc<AtomicBool>,
    pub(crate) permissions_ok: Arc<AtomicBool>,
    pub(crate) hash_outcome_cache: Arc<StdMutex<HashOutcomeLruCache>>,
    pub(crate) ocr_singleflight: Arc<OcrSingleflightMap>,
    pub(crate) hash_processing: Arc<HashProcessingMap>,
    pub(crate) specimen_hit_counts: Arc<SpecimenHitCounts>,
    pub(crate) scan_exempt_roles: Arc<ArcSwap<Vec<Id<RoleMarker>>>>,
    pub(crate) administrator_roles: Arc<ArcSwap<Vec<Id<RoleMarker>>>>,
}

pub(crate) struct SpecimenWriteSuccess {
    pub(crate) specimen_id: String,
    pub(crate) ledger_message_id: Id<MessageMarker>,
    pub(crate) channel_id: Id<ChannelMarker>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SpecimenWriteLogContext {
    pub(crate) image_url: Option<String>,
    pub(crate) image_processing_ms: Option<u128>,
    pub(crate) pre_add_match: Option<String>,
}

pub(crate) enum SpecimenWriteOutcome {
    Added(SpecimenWriteSuccess),
    Duplicate,
}

pub(crate) enum DatabaseWriteRequest {
    AddSpecimen(Box<AddSpecimenWriteRequest>),
    UpsertConfig(Box<UpsertConfigWriteRequest>),
}

pub(crate) struct AddSpecimenWriteRequest {
    state: AppState,
    record: SpecimenRecord,
    image_attachments: Vec<SpecimenImageAttachment>,
    log_context: SpecimenWriteLogContext,
    respond_to: oneshot::Sender<Result<SpecimenWriteOutcome>>,
}

struct PersistedSpecimenWrite {
    state: AppState,
    record: SpecimenRecord,
    stored: StoredSpecimen,
    channel_id: Id<ChannelMarker>,
    log_context: SpecimenWriteLogContext,
    respond_to: oneshot::Sender<Result<SpecimenWriteOutcome>>,
}

pub(crate) struct UpsertConfigWriteRequest {
    state: AppState,
    record: GuildConfigRecord,
    respond_to: oneshot::Sender<Result<Option<Id<MessageMarker>>>>,
}

pub(crate) struct BotLogWriteRequest {
    channel_id: Id<ChannelMarker>,
    log: RenderedBotLog,
    kind: BotLogWriteKind,
    respond_to: Option<oneshot::Sender<Result<Id<MessageMarker>>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BotLogWriteKind {
    Standard,
    ConfigUpdate { updated_by: Id<UserMarker> },
}

pub(crate) enum ImageMetricsCommand {
    Record(ImageMetricEvent),
    RemoveGuild {
        guild_id: Id<GuildMarker>,
    },
    Snapshot {
        guild_id: Id<GuildMarker>,
        respond_to: oneshot::Sender<GuildMetricsSnapshot>,
    },
}

#[derive(Debug, Clone, Default)]
pub(crate) struct GuildMetricsSnapshot {
    pub(crate) period: GuildMetricCounters,
    pub(crate) total: GuildMetricCounters,
    pub(crate) period_perf: Option<crate::image::types::ImagePerfSnapshot>,
    pub(crate) total_perf: Option<crate::image::types::ImagePerfSnapshot>,
    pub(crate) period_timing: PipelineTimingSnapshot,
    pub(crate) total_timing: PipelineTimingSnapshot,
    pub(crate) period_timing_by_class: Vec<PipelineTimingClassSnapshot>,
    pub(crate) total_timing_by_class: Vec<PipelineTimingClassSnapshot>,
}

#[derive(Debug, Clone)]
pub(crate) struct PipelineTimingClassSnapshot {
    pub(crate) label: &'static str,
    pub(crate) timing: PipelineTimingSnapshot,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct GuildMetricCounters {
    pub(crate) images_scanned: u64,
    pub(crate) scan_failures: u64,
    pub(crate) passes: u64,
    pub(crate) hard_matches: u64,
    pub(crate) suspicious_matches: u64,
    pub(crate) hard_exact_xxh128: u64,
    pub(crate) hard_perceptual: u64,
    pub(crate) hard_local_anchors: u64,
    pub(crate) suspicious_exact_xxh128: u64,
    pub(crate) suspicious_perceptual: u64,
    pub(crate) suspicious_local_anchors: u64,
    pub(crate) ocr_calls: u64,
    pub(crate) ocr_resolved_good: u64,
    pub(crate) ocr_resolved_bad: u64,
    pub(crate) ocr_resolved_unknown: u64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct TimingDistributionSnapshot {
    pub(crate) count: u64,
    pub(crate) max_us: u64,
    pub(crate) avg_us: u64,
    pub(crate) p95_us: u64,
    pub(crate) p99_us: u64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct PipelineTimingSnapshot {
    pub(crate) preview_used: u64,
    pub(crate) preview_fallbacks: u64,
    pub(crate) total: Option<TimingDistributionSnapshot>,
    pub(crate) preview_download: Option<TimingDistributionSnapshot>,
    pub(crate) preview_fingerprint: Option<TimingDistributionSnapshot>,
    pub(crate) preview_matcher: Option<TimingDistributionSnapshot>,
    pub(crate) queue_wait: Option<TimingDistributionSnapshot>,
    pub(crate) download: Option<TimingDistributionSnapshot>,
    pub(crate) download_request: Option<TimingDistributionSnapshot>,
    pub(crate) download_body: Option<TimingDistributionSnapshot>,
    pub(crate) download_gate_wait: Option<TimingDistributionSnapshot>,
    pub(crate) flagged_cache_lookup: Option<TimingDistributionSnapshot>,
    pub(crate) exact_match_lookup: Option<TimingDistributionSnapshot>,
    pub(crate) singleflight_wait: Option<TimingDistributionSnapshot>,
    pub(crate) fingerprint: Option<TimingDistributionSnapshot>,
    pub(crate) fingerprint_pipeline: FingerprintPipelineTimingSnapshot,
    pub(crate) matcher: Option<TimingDistributionSnapshot>,
    pub(crate) ocr_crop: Option<TimingDistributionSnapshot>,
    pub(crate) progressive_eval: Option<TimingDistributionSnapshot>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FingerprintPipelineTimingSnapshot {
    pub(crate) decode: Option<TimingDistributionSnapshot>,
    pub(crate) thumbnail: Option<TimingDistributionSnapshot>,
    pub(crate) visual: Option<TimingDistributionSnapshot>,
    pub(crate) orientation: Option<TimingDistributionSnapshot>,
    pub(crate) perceptual: Option<TimingDistributionSnapshot>,
    pub(crate) normalize: Option<TimingDistributionSnapshot>,
    pub(crate) tile_scorer: Option<TimingDistributionSnapshot>,
    pub(crate) text_grid: Option<TimingDistributionSnapshot>,
    pub(crate) local_anchors: Option<TimingDistributionSnapshot>,
    pub(crate) local_hashes: Option<TimingDistributionSnapshot>,
}

impl FingerprintPipelineTimingSnapshot {
    pub(crate) fn parts(&self) -> [(&'static str, Option<&TimingDistributionSnapshot>); 10] {
        [
            ("fp_decode", self.decode.as_ref()),
            ("fp_thumb", self.thumbnail.as_ref()),
            ("fp_visual", self.visual.as_ref()),
            ("fp_orient", self.orientation.as_ref()),
            ("fp_perceptual", self.perceptual.as_ref()),
            ("fp_normalize", self.normalize.as_ref()),
            ("fp_tiles", self.tile_scorer.as_ref()),
            ("fp_text_grid", self.text_grid.as_ref()),
            ("fp_anchors", self.local_anchors.as_ref()),
            ("fp_dense", self.local_hashes.as_ref()),
        ]
    }
}

impl std::ops::Deref for AppState {
    type Target = BotState;

    fn deref(&self) -> &Self::Target {
        &self.bot
    }
}

impl AppState {
    pub(crate) fn guild_id(&self) -> Id<GuildMarker> {
        self.guild_id
    }

    pub(crate) fn active_config(&self) -> GuildConfig {
        self.guild_config.load_full().as_ref().clone()
    }

    pub(crate) fn active_config_arc(&self) -> Arc<GuildConfig> {
        self.guild_config.load_full()
    }

    pub(crate) fn detection_policy_hash(&self) -> u64 {
        self.detection_policy_hash.load(Ordering::Acquire)
    }

    pub(crate) fn ocr_singleflight_cell(
        &self,
        byte_xxh128: Xxh128,
        policy_hash: u64,
    ) -> Arc<OnceCell<crate::image::engine::TextGateReport>> {
        let key = OcrCacheKey {
            byte_xxh128,
            policy_hash,
        };
        match self.ocr_singleflight.entry(key) {
            Entry::Occupied(entry) => Arc::clone(entry.get()),
            Entry::Vacant(entry) => Arc::clone(&entry.insert(Arc::new(OnceCell::new()))),
        }
    }

    pub(crate) fn start_hash_processing(
        &self,
        byte_xxh128: Xxh128,
        policy_hash: u64,
    ) -> HashProcessingClaim {
        let key = OcrCacheKey {
            byte_xxh128,
            policy_hash,
        };
        match self.hash_processing.entry(key.clone()) {
            Entry::Occupied(entry) => HashProcessingClaim::Waiter(entry.get().subscribe()),
            Entry::Vacant(entry) => {
                let complete = Arc::new(watch::Sender::new(false));
                entry.insert(Arc::clone(&complete));
                HashProcessingClaim::Owner { key, complete }
            }
        }
    }

    pub(crate) fn finish_hash_processing(
        &self,
        key: &OcrCacheKey,
        complete: &Arc<watch::Sender<bool>>,
    ) {
        let _ = complete.send(true);
        self.hash_processing
            .remove_if(key, |_, existing| Arc::ptr_eq(existing, complete));
    }

    pub(crate) fn clear_hash_processing(&self) {
        for entry in self.hash_processing.iter() {
            let _ = entry.value().send(true);
        }
        self.hash_processing.clear();
    }

    pub(crate) fn bot_log_channel_id(&self) -> Option<Id<ChannelMarker>> {
        let config = self.guild_config.load_full();
        config.bot_log_channel_id()
    }

    pub(crate) async fn post_bot_log(
        &self,
        event: impl Into<BotLogEvent>,
    ) -> Option<Id<MessageMarker>> {
        let channel_id = self.bot_log_channel_id()?;
        let event = event.into();
        let copy_kind = event.copy_kind;
        let mut log = render_bot_log(event);
        stamp_bot_start_id(&mut log, &self.bot.bot_start_id);
        let config = self.guild_config.load_full();
        log.content = match copy_kind {
            crate::bot::discord::BotLogCopyKind::General => {
                config.discord_general_log_message_content()
            }
            crate::bot::discord::BotLogCopyKind::ConfirmedDetection => {
                config.discord_confirmed_log_message_content()
            }
            crate::bot::discord::BotLogCopyKind::SuspiciousDetection => {
                config.discord_suspicious_log_message_content()
            }
            crate::bot::discord::BotLogCopyKind::BenignDetection => {
                config.discord_benign_log_message_content()
            }
        };
        let (respond_to, response) = oneshot::channel();
        if let Err(source) = self
            .bot
            .bot_log_tx
            .send(BotLogWriteRequest {
                channel_id,
                log,
                kind: BotLogWriteKind::Standard,
                respond_to: Some(respond_to),
            })
            .await
        {
            warn!(
                event = "bot_log.enqueue_failed",
                ?source,
                "failed to enqueue bot log"
            );
            return None;
        }
        match response.await {
            Ok(Ok(message_id)) => Some(message_id),
            Ok(Err(source)) => {
                warn!(
                    event = "bot_log.post_failed",
                    ?source,
                    "failed to post bot log"
                );
                None
            }
            Err(source) => {
                warn!(
                    event = "bot_log.writer_stopped",
                    ?source,
                    "bot log writer stopped"
                );
                None
            }
        }
    }

    pub(crate) async fn edit_bot_log(
        &self,
        channel_id: Id<ChannelMarker>,
        message_id: Id<MessageMarker>,
        event: impl Into<BotLogEvent>,
    ) {
        let event = event.into();
        let copy_kind = event.copy_kind;
        let mut log = render_bot_log(event);
        stamp_bot_start_id(&mut log, &self.bot.bot_start_id);
        let config = self.guild_config.load_full();
        log.content = match copy_kind {
            crate::bot::discord::BotLogCopyKind::General => {
                config.discord_general_log_message_content()
            }
            crate::bot::discord::BotLogCopyKind::ConfirmedDetection => {
                config.discord_confirmed_log_message_content()
            }
            crate::bot::discord::BotLogCopyKind::SuspiciousDetection => {
                config.discord_suspicious_log_message_content()
            }
            crate::bot::discord::BotLogCopyKind::BenignDetection => {
                config.discord_benign_log_message_content()
            }
        };
        if let Err(source) = self
            .discord_effects
            .edit_bot_log_in_channel(channel_id, message_id, log)
            .await
        {
            warn!(
                event = "bot_log.edit_failed",
                channel_id = channel_id.get(),
                message_id = message_id.get(),
                ?source,
                "failed to edit bot log"
            );
        }
    }

    pub(crate) async fn write_specimen_record(
        &self,
        record: SpecimenRecord,
        image_attachments: Vec<SpecimenImageAttachment>,
        log_context: SpecimenWriteLogContext,
    ) -> Result<SpecimenWriteOutcome> {
        let (respond_to, response) = oneshot::channel();
        self.bot
            .database_write_tx
            .send(DatabaseWriteRequest::AddSpecimen(Box::new(
                AddSpecimenWriteRequest {
                    state: self.clone(),
                    record,
                    image_attachments,
                    log_context,
                    respond_to,
                },
            )))
            .await
            .context("queueing specimen database write")?;
        response.await.context("specimen database writer stopped")?
    }

    pub(crate) async fn upsert_config_record(
        &self,
        record: GuildConfigRecord,
    ) -> Result<Option<Id<MessageMarker>>> {
        let (respond_to, response) = oneshot::channel();
        self.bot
            .database_write_tx
            .send(DatabaseWriteRequest::UpsertConfig(Box::new(
                UpsertConfigWriteRequest {
                    state: self.clone(),
                    record,
                    respond_to,
                },
            )))
            .await
            .context("queueing config database write")?;
        response.await.context("database writer stopped")?
    }

    pub(crate) async fn metrics_snapshot(&self) -> Result<GuildMetricsSnapshot> {
        let (respond_to, response) = oneshot::channel();
        self.bot
            .image_metrics_tx
            .send(ImageMetricsCommand::Snapshot {
                guild_id: self.guild_id(),
                respond_to,
            })
            .await
            .context("queueing metrics snapshot request")?;
        response.await.context("metrics task stopped")
    }

    pub(crate) fn guild_active(&self) -> bool {
        if !self.guild_configured.load(Ordering::Acquire) {
            return false;
        }
        if self.safe_mode.load(Ordering::Acquire) {
            return false;
        }
        if !self.permissions_ok.load(Ordering::Acquire) {
            return false;
        }
        let config = self.guild_config.load_full();
        config.enabled
            && config.bot_log_channel_id.is_some()
            && config.bot_log_channel_id.as_deref() != Some(config.ledger_channel_id.as_str())
    }

    pub(crate) fn guild_accepts_specimen_writes(&self) -> bool {
        if !self.guild_configured.load(Ordering::Acquire) {
            return false;
        }
        if self.safe_mode.load(Ordering::Acquire) {
            return false;
        }
        if !self.permissions_ok.load(Ordering::Acquire) {
            return false;
        }
        let config = self.guild_config.load_full();
        config.bot_log_channel_id.is_some()
            && config.bot_log_channel_id.as_deref() != Some(config.ledger_channel_id.as_str())
    }

    pub(crate) fn matcher_len(&self) -> usize {
        self.matcher.read().len()
    }

    pub(crate) fn matcher_records(&self) -> Vec<SpecimenRecord> {
        self.matcher.read().records()
    }

    pub(crate) async fn matcher_records_snapshot(&self) -> Result<Vec<SpecimenRecord>> {
        let matcher = Arc::clone(&self.matcher);
        tokio::task::spawn_blocking(move || matcher.read().records())
            .await
            .context("matcher records snapshot task panicked")
    }

    pub(crate) fn record_specimen_hit(&self, specimen_id: &str) {
        if specimen_id == "none" {
            return;
        }
        self.specimen_hit_counts
            .entry(specimen_id.to_owned())
            .and_modify(|count| *count = count.saturating_add(1))
            .or_insert(1);
    }

    pub(crate) fn specimen_hit_count(&self, specimen_id: &str) -> u64 {
        self.specimen_hit_counts
            .get(specimen_id)
            .map_or(0, |count| *count)
    }

    pub(crate) fn specimen_ledger_link(&self, specimen_id: &str) -> Option<String> {
        let storage = self.storage.read();
        storage
            .specimens
            .iter()
            .find(|specimen| specimen.specimen_id == specimen_id)
            .map(|specimen| {
                message_jump_link(self.guild_id(), storage.channel_id, specimen.message_id)
            })
    }

    pub(crate) fn contains_specimen_xxh128(&self, byte_xxh128: &str) -> bool {
        self.exact_hash_index
            .read()
            .contains_byte_xxh128(byte_xxh128)
    }

    pub(crate) async fn add_matcher_records(&self, records: Vec<SpecimenRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let matcher = Arc::clone(&self.matcher);
        let exact_records = records.clone();
        let matcher_permit = self.matcher_gate.acquire_low_priority().await?;
        tokio::task::spawn_blocking(move || {
            let _matcher_permit = matcher_permit;
            // The matcher already owns the currently published coherence
            // threshold. Only config publication may replace it.
            matcher.write().add_batch_with_policy(records, None);
        })
        .await
        .context("matcher batch add task panicked")?;
        let mut exact_hash_index = self.exact_hash_index.write();
        for record in &exact_records {
            exact_hash_index.add_record(record);
        }
        drop(exact_hash_index);
        self.invalidate_hash_decision_state();
        Ok(())
    }

    pub(crate) async fn remove_matcher_specimen(&self, specimen_id: String) -> Result<bool> {
        // Remove from the hot exact path first. Concurrent scans that
        // already observed the old entry linearize before this removal; later
        // scans cannot return a stale exact match after this method completes.
        self.exact_hash_index.write().remove_specimen(&specimen_id);
        let matcher = Arc::clone(&self.matcher);
        let matcher_specimen_id = specimen_id.clone();
        let matcher_permit = self.matcher_gate.acquire_low_priority().await?;
        let removed = tokio::task::spawn_blocking(move || {
            let _matcher_permit = matcher_permit;
            matcher.write().remove_specimen(&matcher_specimen_id)
        })
        .await
        .context("matcher remove task panicked")?;
        if removed {
            self.invalidate_hash_decision_state();
        }
        Ok(removed)
    }

    fn invalidate_hash_decision_state(&self) {
        self.hash_outcome_cache
            .lock()
            .clear_guild(self.guild_id.get());
        self.clear_hash_processing();
    }

    pub(crate) async fn refresh_matcher_policy(&self, policy: DetectionPolicy) -> Result<()> {
        let matcher = Arc::clone(&self.matcher);
        let matcher_permit = self.matcher_gate.acquire_low_priority().await?;
        tokio::task::spawn_blocking(move || {
            let _matcher_permit = matcher_permit;
            matcher.write().set_coherence_policy(&policy);
        })
        .await
        .context("matcher policy refresh task panicked")?;
        Ok(())
    }

    pub(crate) async fn find_match_for_policy(
        &self,
        image: Arc<ImageFingerprint>,
        policy: DetectionPolicy,
    ) -> Result<Option<MatchOutcome>> {
        self.find_match_for_policy_variant(image, policy, RuntimeMatchVariant::Original)
            .await
    }

    pub(crate) async fn find_preview_match_for_policy(
        &self,
        image: Arc<ImageFingerprint>,
        policy: DetectionPolicy,
    ) -> Result<Option<MatchOutcome>> {
        self.find_match_for_policy_variant(image, policy, RuntimeMatchVariant::Preview)
            .await
    }

    async fn find_match_for_policy_variant(
        &self,
        image: Arc<ImageFingerprint>,
        policy: DetectionPolicy,
        variant: RuntimeMatchVariant,
    ) -> Result<Option<MatchOutcome>> {
        let matcher_permit = self
            .bot
            .matcher_gate
            .acquire_high_priority()
            .await
            .context("matcher gate closed")?;
        let matcher = Arc::clone(&self.matcher);
        let scratch_pool = Arc::clone(&self.bot.matcher_scratch_pool);
        tokio::task::spawn_blocking(move || {
            let _matcher_permit = matcher_permit;
            let mut scratch = scratch_pool.take();
            let outcome = match variant {
                RuntimeMatchVariant::Original => matcher.read().find_for_policy_with_scratch(
                    image.as_ref(),
                    &policy,
                    &mut scratch,
                ),
                RuntimeMatchVariant::Preview => matcher
                    .read()
                    .find_preview_for_policy_with_scratch(image.as_ref(), &policy, &mut scratch),
            };
            scratch_pool.put(scratch);
            outcome
        })
        .await
        .context("matcher task panicked")
    }

    pub(crate) fn find_exact_xxh128_for_policy(
        &self,
        byte_xxh128: Xxh128,
        policy: &DetectionPolicy,
    ) -> Option<MatchOutcome> {
        self.exact_hash_index
            .read()
            .find_for_policy(byte_xxh128, policy)
    }

    pub(crate) async fn refresh_bot_permissions(&self) -> Result<bool> {
        let config = self.active_config();
        let report =
            check_bot_permissions(&self.discord, self.guild_id(), self.bot_user_id, &config)
                .await?;
        let ok = report.ok;
        self.permissions_ok.store(ok, Ordering::Release);
        if !ok {
            warn!(
                event = "permissions.missing",
                guild_id = self.guild_id().get(),
                missing = %report.missing_summary(),
                "bot permissions are incomplete; guild scanning is inactive"
            );
        }
        Ok(ok)
    }
}

impl BotState {
    pub(crate) async fn download_image(
        &self,
        url: &str,
        mime_hint: Option<&str>,
        download_config: &DownloadConfig,
    ) -> Result<DownloadedImage> {
        self.download_host_activity.touch_url(url);
        let key = download_singleflight_key(url, mime_hint, download_config.max_bytes);
        let (cell, reused) = match self.download_singleflight.entry(key.clone()) {
            Entry::Occupied(entry) => (Arc::clone(entry.get()), true),
            Entry::Vacant(entry) => (Arc::clone(&entry.insert(Arc::new(OnceCell::new()))), false),
        };
        if reused {
            debug!(
                event = "image.download_singleflight_wait",
                image_url = %url_log_label(url),
                "waiting for in-flight image download"
            );
        }
        let _cleanup = (!reused).then(|| DownloadSingleflightCleanup {
            map: Arc::clone(&self.download_singleflight),
            key: key.clone(),
            cell: Arc::clone(&cell),
        });

        let image_http = self.image_http.clone();
        let download_gate = Arc::clone(&self.download_gate);
        let url = url.to_owned();
        let mime_hint = mime_hint.map(str::to_owned);
        let download_config = download_config.clone();
        let result = cell
            .get_or_init(|| async move {
                download_image(
                    &image_http,
                    &url,
                    mime_hint.as_deref(),
                    &download_config,
                    &download_gate,
                    &self.download_memory_gate,
                )
                .await
                .map_err(|source| Arc::<str>::from(source.to_string()))
            })
            .await
            .clone();

        result.map_err(|source| anyhow!("{}", source.as_ref()))
    }
}

fn spawn_connection_warmer(state: &BotState) {
    if !state.config.download.warmer_enabled {
        return;
    }

    let client = state.image_http.clone();
    let activity = Arc::clone(&state.download_host_activity);
    let shutdown = state.shutdown.clone();
    let period = Duration::from_secs(state.config.download.warmer_period_seconds);
    let initial_delay = warmer_initial_delay(&state.bot_start_id, period);
    state.background_tasks.spawn(async move {
        connection_warmer_loop(client, activity, shutdown, period, initial_delay).await;
    });
    info!(
        event = "download.warmer_started",
        host = DISCORD_CDN_WARMER_HOST,
        period_ms = period.as_millis(),
        pool_idle_timeout_ms = IMAGE_POOL_IDLE_TIMEOUT.as_millis(),
        "started Discord CDN connection warmer"
    );
}

async fn connection_warmer_loop(
    client: ReqwestClient,
    activity: Arc<DownloadHostActivity>,
    shutdown: CancellationToken,
    period: Duration,
    initial_delay: Duration,
) {
    if !initial_delay.is_zero() {
        tokio::select! {
            () = shutdown.cancelled() => return,
            () = tokio::time::sleep(initial_delay) => {}
        }
    }

    loop {
        if activity.cdn_elapsed() < period {
            debug!(
                event = "download.warmer_skipped",
                host = DISCORD_CDN_WARMER_HOST,
                elapsed_ms = activity.cdn_elapsed().as_millis(),
                period_ms = period.as_millis(),
                "recent real download kept CDN connection warm"
            );
        } else {
            let started = Instant::now();
            let result = client
                .head(DISCORD_CDN_WARMER_URL)
                .timeout(DISCORD_CDN_WARMER_TIMEOUT)
                .send()
                .await;
            let elapsed_ms = started.elapsed().as_millis();
            match result {
                Ok(response) => debug!(
                    event = "download.warmer_tick",
                    host = DISCORD_CDN_WARMER_HOST,
                    status = %response.status(),
                    elapsed_ms,
                    "connection warmer tick"
                ),
                Err(source) => debug!(
                    event = "download.warmer_failed",
                    host = DISCORD_CDN_WARMER_HOST,
                    elapsed_ms,
                    ?source,
                    "connection warmer failed"
                ),
            }
            activity.touch_cdn();
        }

        tokio::select! {
            () = shutdown.cancelled() => return,
            () = tokio::time::sleep(period) => {}
        }
    }
}

fn spawn_guild_health_monitor(state: &BotState) {
    let state = state.clone();
    let shutdown = state.shutdown.clone();
    state.background_tasks.clone().spawn(async move {
        Box::pin(guild_health_monitor_loop(state, shutdown)).await;
    });
}

async fn guild_health_monitor_loop(state: BotState, shutdown: CancellationToken) {
    let mut interval = tokio::time::interval(GUILD_HEALTH_CHECK_PERIOD);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    interval.tick().await;
    let mut ticks = 0_u64;

    loop {
        tokio::select! {
            () = shutdown.cancelled() => return,
            _ = interval.tick() => {}
        }
        ticks = ticks.wrapping_add(1);
        let now = Instant::now();
        let retryable = state
            .guild_load_backoff
            .iter()
            .filter(|entry| *entry.value() <= now)
            .map(|entry| *entry.key())
            .collect::<Vec<_>>();
        for guild_id in retryable {
            state.request_guild_load(guild_id);
        }

        let runtimes = state
            .guilds
            .iter()
            .map(|entry| Arc::clone(entry.value()))
            .collect::<Vec<_>>();
        let refresh_permissions = ticks.is_multiple_of(GUILD_PERMISSION_REFRESH_TICKS);
        stream::iter(runtimes)
            .for_each_concurrent(4, |runtime| async {
                if runtime.safe_mode.load(Ordering::Acquire) {
                    if state
                        .guild_load_backoff
                        .get(&runtime.guild_id)
                        .is_some_and(|retry_at| *retry_at > Instant::now())
                    {
                        return;
                    }
                    recover_safe_guild(&state, runtime).await;
                } else if refresh_permissions && runtime.guild_configured.load(Ordering::Acquire) {
                    let scoped = state.scoped_state(&runtime);
                    if let Err(source) = scoped.refresh_bot_permissions().await {
                        warn!(
                            event = "permissions.periodic_check_failed",
                            guild_id = runtime.guild_id.get(),
                            ?source,
                            "periodic bot permission refresh failed; retaining the last known state"
                        );
                    }
                }
            })
            .await;
    }
}

async fn recover_safe_guild(state: &BotState, expected: Arc<GuildRuntime>) {
    let guild_id = expected.guild_id;
    let load_lock = match state.guild_load_locks.entry(guild_id) {
        Entry::Occupied(entry) => Arc::clone(entry.get()),
        Entry::Vacant(entry) => Arc::clone(&entry.insert(Arc::new(AsyncMutex::new(())))),
    };
    let _load_guard = load_lock.lock().await;
    let still_current_and_safe = state.guilds.get(&guild_id).is_some_and(|current| {
        Arc::ptr_eq(current.value(), &expected) && current.safe_mode.load(Ordering::Acquire)
    });
    if !still_current_and_safe {
        return;
    }

    match state.load_guild_runtime(guild_id).await {
        Ok(replacement) => {
            let replaced = match state.guilds.entry(guild_id) {
                Entry::Occupied(mut entry) if Arc::ptr_eq(entry.get(), &expected) => {
                    entry.insert(replacement);
                    true
                }
                _ => false,
            };
            if replaced {
                state.guild_load_backoff.remove(&guild_id);
                info!(
                    event = "guild.safe_mode_recovered",
                    guild_id = guild_id.get(),
                    "reloaded durable guild state and left safe mode"
                );
            }
        }
        Err(source) => {
            state
                .guild_load_backoff
                .insert(guild_id, Instant::now() + GUILD_LOAD_FAILURE_BACKOFF);
            warn!(
                event = "guild.safe_mode_recovery_failed",
                guild_id = guild_id.get(),
                retry_after_ms = GUILD_LOAD_FAILURE_BACKOFF.as_millis(),
                ?source,
                "could not reload durable guild state; remaining in safe mode"
            );
        }
    }
}

fn epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn warmer_initial_delay(seed: &str, period: Duration) -> Duration {
    let jitter_window = period.as_secs().min(30);
    if jitter_window == 0 {
        return Duration::ZERO;
    }
    let hash = seed.bytes().fold(0_u64, |hash, byte| {
        hash.wrapping_mul(1_099_511_628_211)
            .wrapping_add(u64::from(byte))
    });
    Duration::from_secs(hash % jitter_window)
}

fn download_singleflight_key(url: &str, mime_hint: Option<&str>, max_bytes: usize) -> String {
    let normalized = Url::parse(url).map_or_else(
        |_| url.to_owned(),
        |mut parsed| {
            let mut retained_query = parsed
                .query_pairs()
                .map(|(key, value)| (key.into_owned(), value.into_owned()))
                .collect::<Vec<_>>();
            retained_query.sort_unstable();
            parsed.set_query(None);
            if !retained_query.is_empty() {
                let mut query = parsed.query_pairs_mut();
                for (key, value) in retained_query {
                    query.append_pair(&key, &value);
                }
            }
            parsed.to_string()
        },
    );
    format!(
        "{normalized}|mime={}|max_bytes={max_bytes}",
        mime_hint.unwrap_or("")
    )
}

fn stamp_bot_start_id(log: &mut RenderedBotLog, bot_start_id: &str) {
    let text = format!("Sightline start {bot_start_id}");
    for embed in &mut log.embeds {
        embed.footer = Some(EmbedFooter {
            icon_url: None,
            proxy_icon_url: None,
            text: text.clone(),
        });
    }
}

impl BotState {
    pub(crate) fn cached_guild(&self, guild_id: Id<GuildMarker>) -> Option<AppState> {
        self.guilds
            .get(&guild_id)
            .map(|runtime| self.scoped_state(runtime.value()))
    }

    pub(crate) fn request_guild_load(&self, guild_id: Id<GuildMarker>) {
        if self.cached_guild(guild_id).is_some() {
            return;
        }

        let now = Instant::now();
        if self
            .guild_load_backoff
            .get(&guild_id)
            .is_some_and(|retry_at| *retry_at > now)
        {
            return;
        }
        self.guild_load_backoff
            .insert(guild_id, now + GUILD_LOAD_INFLIGHT_BACKOFF);

        let state = self.clone();
        let shutdown = self.shutdown.clone();
        self.background_tasks.spawn(async move {
            let result = tokio::select! {
                () = shutdown.cancelled() => return,
                result = state.for_guild(guild_id) => result,
            };
            match result {
                Ok(_) => {
                    state.guild_load_backoff.remove(&guild_id);
                }
                Err(source) => {
                    state
                        .guild_load_backoff
                        .insert(guild_id, Instant::now() + GUILD_LOAD_FAILURE_BACKOFF);
                    warn!(
                        event = "guild.background_load_failed",
                        guild_id = guild_id.get(),
                        retry_after_ms = GUILD_LOAD_FAILURE_BACKOFF.as_millis(),
                        ?source,
                        "could not load guild runtime in background"
                    );
                }
            }
        });
    }

    pub(crate) async fn for_guild(&self, guild_id: Id<GuildMarker>) -> Result<AppState> {
        if let Some(runtime) = self.guilds.get(&guild_id) {
            return Ok(self.scoped_state(runtime.value()));
        }

        let load_lock = match self.guild_load_locks.entry(guild_id) {
            Entry::Occupied(entry) => Arc::clone(entry.get()),
            Entry::Vacant(entry) => Arc::clone(&entry.insert(Arc::new(AsyncMutex::new(())))),
        };
        let _load_guard = load_lock.lock().await;

        if let Some(runtime) = self.guilds.get(&guild_id) {
            return Ok(self.scoped_state(runtime.value()));
        }

        let runtime = match self.load_guild_runtime(guild_id).await {
            Ok(runtime) => {
                self.guild_load_backoff.remove(&guild_id);
                runtime
            }
            Err(source) => {
                self.guild_load_backoff
                    .insert(guild_id, Instant::now() + GUILD_LOAD_FAILURE_BACKOFF);
                return Err(source);
            }
        };
        let runtime = match self.guilds.entry(guild_id) {
            Entry::Occupied(entry) => Arc::clone(entry.get()),
            Entry::Vacant(entry) => Arc::clone(&entry.insert(runtime)),
        };
        Ok(self.scoped_state(&runtime))
    }

    fn state_for_storage_channel(&self, channel_id: Id<ChannelMarker>) -> Option<AppState> {
        for runtime in self.guilds.iter() {
            if runtime.value().storage.read().channel_id == channel_id {
                return Some(self.scoped_state(runtime.value()));
            }
        }

        None
    }

    fn fail_guild_storage_channel(&self, channel_id: Id<ChannelMarker>) {
        let Some(state) = self.state_for_storage_channel(channel_id) else {
            return;
        };
        let guild_id = state.guild_id();

        {
            let mut storage = state.storage.write();
            storage.config_message_id = None;
            storage.specimens.clear();
        }
        *state.matcher.write() = Matcher::default();
        *state.exact_hash_index.write() = ExactHashIndex::default();
        state
            .hash_outcome_cache
            .lock()
            .clear_guild(state.guild_id().get());
        state.ocr_singleflight.clear();
        state.clear_hash_processing();
        state.guild_configured.store(false, Ordering::Release);
        state.safe_mode.store(true, Ordering::Release);
        state.permissions_ok.store(false, Ordering::Release);
        self.guilds.remove(&guild_id);
        self.guild_load_locks.remove(&guild_id);
        self.guild_load_backoff
            .insert(guild_id, Instant::now() + GUILD_LOAD_FAILURE_BACKOFF);
        self.clear_message_inspection_for_guild(guild_id);

        warn!(
            event = "guild.database_channel_missing",
            guild_id = guild_id.get(),
            channel_id = channel_id.get(),
            "sightline-db channel disappeared; disabled guild runtime until the channel is recreated and configured"
        );
    }

    fn cleanup_removed_guild(&self, guild_id: Id<GuildMarker>) {
        let runtime = self.guilds.remove(&guild_id).map(|(_, runtime)| runtime);
        self.guild_load_locks.remove(&guild_id);
        self.guild_load_backoff.remove(&guild_id);
        self.hash_outcome_cache.lock().clear_guild(guild_id.get());
        self.clear_message_inspection_for_guild(guild_id);
        let metrics_tx = self.image_metrics_tx.clone();
        self.background_tasks.spawn(async move {
            let _ = metrics_tx
                .send(ImageMetricsCommand::RemoveGuild { guild_id })
                .await;
        });

        if let Some(runtime) = runtime {
            *runtime.matcher.write() = Matcher::default();
            *runtime.exact_hash_index.write() = ExactHashIndex::default();
            {
                let mut storage = runtime.storage.write();
                storage.config_message_id = None;
                storage.specimens.clear();
            }
            runtime.guild_configured.store(false, Ordering::Release);
            runtime.safe_mode.store(true, Ordering::Release);
            runtime.permissions_ok.store(false, Ordering::Release);
            runtime.ocr_singleflight.clear();
            let state = self.scoped_state(&runtime);
            state.clear_hash_processing();

            info!(
                event = "guild.removed_cleanup",
                guild_id = guild_id.get(),
                "cleared in-memory guild runtime after bot was removed from guild"
            );
        } else {
            info!(
                event = "guild.removed_cleanup",
                guild_id = guild_id.get(),
                "bot was removed from guild with no loaded runtime"
            );
        }
    }

    fn clear_message_inspection_for_guild(&self, guild_id: Id<GuildMarker>) {
        self.matched_messages
            .retain(|scope| scope.guild_id != guild_id);
        self.confirmed_messages
            .retain(|scope, _| scope.guild_id != guild_id);
        self.sibling_inspections
            .retain(|key, _| key.scope.guild_id != guild_id);
        self.logged_sibling_inspections
            .retain(|key| key.scope.guild_id != guild_id);
    }

    fn scoped_state(&self, runtime: &Arc<GuildRuntime>) -> AppState {
        AppState {
            bot: self.clone(),
            guild_id: runtime.guild_id,
            matcher: Arc::clone(&runtime.matcher),
            exact_hash_index: Arc::clone(&runtime.exact_hash_index),
            guild_config: Arc::clone(&runtime.guild_config),
            detection_policy_hash: Arc::clone(&runtime.detection_policy_hash),
            guild_configured: Arc::clone(&runtime.guild_configured),
            storage: Arc::clone(&runtime.storage),
            safe_mode: Arc::clone(&runtime.safe_mode),
            permissions_ok: Arc::clone(&runtime.permissions_ok),
            hash_outcome_cache: Arc::clone(&self.hash_outcome_cache),
            ocr_singleflight: Arc::clone(&runtime.ocr_singleflight),
            hash_processing: Arc::clone(&runtime.hash_processing),
            specimen_hit_counts: Arc::clone(&runtime.specimen_hit_counts),
            scan_exempt_roles: Arc::clone(&runtime.scan_exempt_roles),
            administrator_roles: Arc::clone(&runtime.administrator_roles),
        }
    }

    async fn load_guild_runtime(&self, guild_id: Id<GuildMarker>) -> Result<Arc<GuildRuntime>> {
        let ledger_channel_id = find_database_channel(&self.discord, guild_id)
            .await
            .context("finding sightline database channel")?;

        let load = load_ledger(
            &self.discord,
            &self.image_http,
            ledger_channel_id,
            guild_id,
            self.bot_user_id,
            &self.secrets.specimen_hmac_secret,
            crate::bot::discord::LedgerRecoveryConfig {
                base_match_config: &self.config.matching,
                max_decoded_pixels: self.config.download.max_decoded_pixels,
                decode_gate: &self.decode_gate,
                decoded_memory_gate: &self.decoded_image_memory_gate,
            },
        )
        .await
        .with_context(|| format!("loading guild {} ledger", guild_id.get()))?;
        let records = load.specimens;
        let loaded_guild_config = load.guild_config;
        let storage = load.storage;
        let safe_mode = false;
        let guild_configured = loaded_guild_config.is_some();
        let guild_config = loaded_guild_config.unwrap_or_else(|| {
            GuildConfig::from_loaded_defaults(
                guild_id,
                ledger_channel_id,
                &self.config.matching,
                &self.config.default_scan_policy(),
                &self.config.text_gate,
            )
        });
        let scan_exempt_roles = guild_config.parsed_scan_exempt_role_ids();
        let administrator_roles = self.load_administrator_role_ids(guild_id).await;
        let detection_policy_hash = guild_config.detection_cache_policy_hash();
        let matcher_policy = guild_config.detection_policy.clone();
        let matcher_permit = self.matcher_gate.acquire_low_priority().await?;
        let (matcher, exact_hash_index) = tokio::task::spawn_blocking(move || {
            let _matcher_permit = matcher_permit;
            let exact_hash_index = ExactHashIndex::new(&records);
            let matcher = Matcher::new_with_policy(records, &matcher_policy);
            (matcher, exact_hash_index)
        })
        .await
        .context("matcher build task panicked")?;
        info!(
            event = "guild.loaded",
            guild_id = guild_id.get(),
            specimens = matcher.len(),
            configured = guild_configured,
            safe_mode,
            "guild runtime loaded"
        );

        let runtime = Arc::new(GuildRuntime {
            guild_id,
            matcher: Arc::new(StdRwLock::new(matcher)),
            exact_hash_index: Arc::new(StdRwLock::new(exact_hash_index)),
            guild_config: Arc::new(ArcSwap::from_pointee(guild_config)),
            detection_policy_hash: Arc::new(AtomicU64::new(detection_policy_hash)),
            guild_configured: Arc::new(AtomicBool::new(guild_configured)),
            storage: Arc::new(StdRwLock::new(storage)),
            safe_mode: Arc::new(AtomicBool::new(safe_mode)),
            permissions_ok: Arc::new(AtomicBool::new(false)),
            ocr_singleflight: Arc::new(DashMap::new()),
            hash_processing: Arc::new(DashMap::new()),
            specimen_hit_counts: Arc::new(DashMap::new()),
            scan_exempt_roles: Arc::new(ArcSwap::from_pointee(scan_exempt_roles)),
            administrator_roles: Arc::new(ArcSwap::from_pointee(administrator_roles)),
        });
        let scoped = self.scoped_state(&runtime);
        if guild_configured
            && !safe_mode
            && let Err(source) = scoped.refresh_bot_permissions().await
        {
            warn!(
                event = "permissions.check_failed",
                guild_id = guild_id.get(),
                ?source,
                "failed to verify bot permissions; guild scanning is inactive"
            );
        }
        log_effective_options(&scoped).await;
        Ok(runtime)
    }

    async fn load_administrator_role_ids(&self, guild_id: Id<GuildMarker>) -> Vec<Id<RoleMarker>> {
        match self.discord.roles(guild_id).await {
            Ok(response) => match response.models().await {
                Ok(roles) => roles
                    .into_iter()
                    .filter(|role| role.permissions.contains(Permissions::ADMINISTRATOR))
                    .map(|role| role.id)
                    .collect(),
                Err(source) => {
                    warn!(
                        event = "guild.admin_roles_decode_failed",
                        guild_id = guild_id.get(),
                        ?source,
                        "failed to decode guild roles for administrator exemption cache"
                    );
                    Vec::new()
                }
            },
            Err(source) => {
                warn!(
                    event = "guild.admin_roles_load_failed",
                    guild_id = guild_id.get(),
                    ?source,
                    "failed to load guild roles for administrator exemption cache"
                );
                Vec::new()
            }
        }
    }
}

pub(crate) async fn run_bot(config: AppConfig) -> Result<()> {
    match rustls::crypto::ring::default_provider().install_default() {
        Ok(()) => {}
        Err(_) => {
            info!(
                event = "rustls.provider_already_installed",
                "rustls crypto provider was already installed"
            );
        }
    }

    let secrets = load_secrets()?;
    let bot_start_id = generate_bot_start_id()?;
    let image_pool_idle_per_host = config.queue.download_concurrency.clamp(8, 64);
    let image_http = ReqwestClient::builder()
        .redirect(Policy::none())
        .connect_timeout(Duration::from_secs(config.download.timeout_seconds.min(2)))
        .timeout(Duration::from_secs(config.download.timeout_seconds))
        .pool_max_idle_per_host(image_pool_idle_per_host)
        .pool_idle_timeout(IMAGE_POOL_IDLE_TIMEOUT)
        .tcp_keepalive(Duration::from_secs(60))
        .http2_keep_alive_interval(Duration::from_secs(30))
        .http2_keep_alive_timeout(Duration::from_secs(10))
        .http2_keep_alive_while_idle(true)
        .user_agent("discord-sightline/0.1")
        .build()
        .context("building image reqwest client")?;
    let ocr_http = ReqwestClient::builder()
        .redirect(Policy::none())
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(config.ocr_space.timeout_seconds))
        .user_agent("discord-sightline/0.1")
        .build()
        .context("building OCR reqwest client")?;
    let ocr_gate = Arc::new(Semaphore::new(config.queue.ocr_concurrency));
    let ocr_space = secrets
        .ocr_space_api_key
        .clone()
        .map(|api_key| {
            OcrSpaceClient::with_attempt_gate(
                ocr_http,
                api_key,
                config.ocr_space.clone(),
                Some(Arc::clone(&ocr_gate)),
            )
        })
        .transpose()
        .context("configuring OCR.space client")?
        .map(Arc::new);
    let discord = Arc::new(DiscordClient::new(secrets.discord_token.clone()));
    let discord_effects: Arc<dyn DiscordEffects> =
        Arc::new(TwilightDiscordEffects::new(Arc::clone(&discord)));

    let app = discord
        .current_user_application()
        .await
        .context("fetching current application")?
        .model()
        .await
        .context("decoding current application")?;
    let current_user = discord
        .current_user()
        .await
        .context("fetching current bot user")?
        .model()
        .await
        .context("decoding current bot user")?;
    let bot_user_id = config.bot.user_id()?.unwrap_or(current_user.id);

    if config.commands.register_on_startup {
        register_commands(&discord, app.id).await?;
    }

    let metrics_queue_size = config.queue.max_size.saturating_mul(4).clamp(1024, 65_536);
    let (image_metrics_tx, image_metrics_rx) = mpsc::channel(metrics_queue_size);
    let database_queue_size = config.queue.max_size.clamp(32, 65_536);
    let (database_write_tx, database_write_rx) = mpsc::channel(database_queue_size);
    let bot_log_queue_size = config.queue.max_size.saturating_mul(4).clamp(256, 65_536);
    let (bot_log_tx, bot_log_rx) = mpsc::channel(bot_log_queue_size);
    let followup_queue_size = config.queue.max_size.clamp(32, 65_536);
    let (detection_followup_tx, detection_followup_rx) = mpsc::channel(followup_queue_size);
    let (ocr_followup_tx, ocr_followup_rx) = mpsc::channel(followup_queue_size);
    let (original_auto_add_tx, original_auto_add_rx) = mpsc::channel(followup_queue_size);
    let cpu_concurrency = config.queue.cpu_concurrency;
    let cpu_gate = Arc::new(CpuGate::new(cpu_concurrency));
    let shutdown = CancellationToken::new();
    let background_tasks = TaskTracker::new();
    let download_host_activity = Arc::new(DownloadHostActivity::default());

    let (image_tx, image_rx) = mpsc::channel(config.queue.max_size);
    let state = BotState {
        download_gate: Arc::new(Semaphore::new(config.queue.download_concurrency)),
        download_memory_gate: Arc::new(Semaphore::new(config.queue.download_memory_max_bytes)),
        decoded_image_memory_gate: Arc::new(Semaphore::new(
            config.queue.decoded_image_memory_max_bytes,
        )),
        download_singleflight: Arc::new(DashMap::new()),
        download_host_activity,
        decode_gate: Arc::clone(&cpu_gate),
        image_byte_store: Arc::new(ImageByteStore::new(config.queue.byte_store_max_bytes)),
        dedupe: Arc::new(StdMutex::new(LruDedupe::new(
            config.queue.max_size.saturating_mul(20),
        ))),
        hash_outcome_cache: Arc::new(StdMutex::new(HashOutcomeLruCache::new(
            config.queue.hash_outcome_cache_size,
        ))),
        image_metrics_tx,
        database_write_tx,
        bot_log_tx,
        config,
        secrets,
        bot_start_id,
        application_id: app.id,
        bot_user_id,
        discord: Arc::clone(&discord),
        discord_effects,
        image_http,
        ocr_space,
        matcher_scratch_pool: Arc::new(MatcherScratchPool::new(cpu_concurrency)),
        matcher_gate: cpu_gate,
        guilds: Arc::new(DashMap::new()),
        guild_load_locks: Arc::new(DashMap::new()),
        guild_load_backoff: Arc::new(DashMap::new()),
        interaction_gate: Arc::new(Semaphore::new(4)),
        detection_followup_tx,
        ocr_followup_tx,
        original_auto_add_tx,
        image_tx: image_tx.clone(),
        shutdown,
        background_tasks,
        matched_messages: Arc::new(DashSet::new()),
        confirmed_messages: Arc::new(DashMap::new()),
        sibling_inspections: Arc::new(DashMap::new()),
        logged_sibling_inspections: Arc::new(DashSet::new()),
    };
    spawn_connection_warmer(&state);
    spawn_guild_health_monitor(&state);
    info!(
        event = "startup.ready",
        bot_start_id = %state.bot_start_id,
        "sightline startup complete"
    );

    let worker_task = tokio::spawn(worker_loop(state.clone(), image_rx));
    let followup_task = tokio::spawn(detection_followup_loop(
        state.shutdown.clone(),
        detection_followup_rx,
    ));
    let ocr_followup_task =
        tokio::spawn(ocr_followup_loop(state.shutdown.clone(), ocr_followup_rx));
    let original_auto_add_task = tokio::spawn(original_auto_add_loop(
        state.shutdown.clone(),
        original_auto_add_rx,
    ));
    let perf_task = tokio::spawn(performance_report_loop(image_metrics_rx));
    let database_task = tokio::spawn(database_writer_loop(database_write_rx));
    let bot_log_task = tokio::spawn(bot_log_writer_loop(
        state.discord_effects.clone(),
        bot_log_rx,
    ));

    let shutdown = state.shutdown.clone();
    let background_tasks = state.background_tasks.clone();
    let gateway_result = run_gateway(state.clone(), image_tx).await;
    shutdown.cancel();
    background_tasks.close();
    if tokio::time::timeout(Duration::from_secs(10), background_tasks.wait())
        .await
        .is_err()
    {
        warn!(
            event = "background_tasks.shutdown_timeout",
            timeout_ms = 10_000_u64,
            "timed out waiting for background tasks to stop"
        );
    }
    if let Err(source) = worker_task.await {
        warn!(
            event = "worker.join_failed",
            ?source,
            "image worker task join failed"
        );
    }
    drop(state);
    if let Err(source) = followup_task.await {
        warn!(
            event = "followup.join_failed",
            ?source,
            "detection follow-up task join failed"
        );
    }
    if let Err(source) = ocr_followup_task.await {
        warn!(
            event = "ocr_followup.join_failed",
            ?source,
            "OCR follow-up task join failed"
        );
    }
    if let Err(source) = original_auto_add_task.await {
        warn!(
            event = "original_auto_add.join_failed",
            ?source,
            "original auto-add task join failed"
        );
    }
    if let Err(source) = database_task.await {
        warn!(
            event = "database_writer.join_failed",
            ?source,
            "database writer task join failed"
        );
    }
    if let Err(source) = bot_log_task.await {
        warn!(
            event = "bot_log_writer.join_failed",
            ?source,
            "bot log writer task join failed"
        );
    }
    if let Err(source) = perf_task.await {
        warn!(
            event = "performance_writer.join_failed",
            ?source,
            "performance writer task join failed"
        );
    }
    gateway_result
}

const SPECIMEN_WRITE_DEBOUNCE: Duration = Duration::from_millis(40);
const SPECIMEN_WRITE_MAX_BATCH_DELAY: Duration = Duration::from_millis(200);
const SPECIMEN_WRITE_BATCH_LIMIT: usize = 64;

async fn database_writer_loop(mut rx: mpsc::Receiver<DatabaseWriteRequest>) {
    let mut deferred = None;
    loop {
        let request = match deferred.take() {
            Some(request) => request,
            None => match rx.recv().await {
                Some(request) => request,
                None => break,
            },
        };
        match request {
            DatabaseWriteRequest::AddSpecimen(first) => {
                let mut requests = vec![*first];
                let batch_deadline = Instant::now() + SPECIMEN_WRITE_MAX_BATCH_DELAY;
                while requests.len() < SPECIMEN_WRITE_BATCH_LIMIT {
                    let wait = batch_deadline
                        .saturating_duration_since(Instant::now())
                        .min(SPECIMEN_WRITE_DEBOUNCE);
                    if wait.is_zero() {
                        break;
                    }
                    match tokio::time::timeout(wait, rx.recv()).await {
                        Ok(Some(DatabaseWriteRequest::AddSpecimen(request))) => {
                            requests.push(*request);
                        }
                        Ok(Some(request)) => {
                            deferred = Some(request);
                            break;
                        }
                        Ok(None) | Err(_) => break,
                    }
                }
                database_write_specimen_batch(requests).await;
            }
            DatabaseWriteRequest::UpsertConfig(request) => {
                let UpsertConfigWriteRequest {
                    state,
                    record,
                    respond_to,
                } = *request;
                let result = database_upsert_config(&state, &record).await;
                let _ = respond_to.send(result);
            }
        }
    }
}

async fn database_write_specimen_batch(requests: Vec<AddSpecimenWriteRequest>) {
    let mut hashes_in_batch = HashSet::new();
    let mut persisted_by_runtime = HashMap::<(u64, usize), Vec<PersistedSpecimenWrite>>::new();

    for request in requests {
        let AddSpecimenWriteRequest {
            state,
            record,
            image_attachments,
            log_context,
            respond_to,
        } = request;
        let hash_key = (state.guild_id().get(), record.image.byte_xxh128.clone());
        if state.contains_specimen_xxh128(&record.image.byte_xxh128)
            || !hashes_in_batch.insert(hash_key.clone())
        {
            let _ = respond_to.send(Ok(SpecimenWriteOutcome::Duplicate));
            continue;
        }

        let channel_id = state.storage.read().channel_id;
        match state
            .discord_effects
            .create_ledger_record_message(
                channel_id,
                record.clone(),
                image_attachments,
                state.secrets.specimen_hmac_secret.clone(),
            )
            .await
        {
            Ok(stored) => {
                let runtime_key = (state.guild_id().get(), Arc::as_ptr(&state.matcher) as usize);
                persisted_by_runtime
                    .entry(runtime_key)
                    .or_default()
                    .push(PersistedSpecimenWrite {
                        state,
                        record,
                        stored,
                        channel_id,
                        log_context,
                        respond_to,
                    });
            }
            Err(source) => {
                hashes_in_batch.remove(&hash_key);
                let _ = respond_to.send(Err(source));
            }
        }
    }

    for (_, mut writes) in persisted_by_runtime {
        let state = writes[0].state.clone();
        let records = writes
            .iter()
            .map(|write| write.record.clone())
            .collect::<Vec<_>>();
        if let Err(source) = state.add_matcher_records(records).await {
            let message = format!("batching persisted specimens into matcher failed: {source:#}");
            for write in writes {
                let _ = write.respond_to.send(Err(anyhow!(message.clone())));
            }
            continue;
        }

        for write in writes.drain(..) {
            let success = SpecimenWriteSuccess {
                specimen_id: write.stored.specimen_id.clone(),
                ledger_message_id: write.stored.message_id,
                channel_id: write.channel_id,
            };
            write.state.storage.write().specimens.push(write.stored);
            log_specimen_database_write(&write.state, &write.record, &success, &write.log_context)
                .await;
            let _ = write
                .respond_to
                .send(Ok(SpecimenWriteOutcome::Added(success)));
        }
    }
}

async fn database_upsert_config(
    state: &AppState,
    record: &GuildConfigRecord,
) -> Result<Option<Id<MessageMarker>>> {
    let (channel_id, existing_message_id) = {
        let storage = state.storage.read();
        (storage.channel_id, storage.config_message_id)
    };
    let created_config_message_id = state
        .discord_effects
        .upsert_config_record_message(
            channel_id,
            existing_message_id,
            record.clone(),
            state.secrets.specimen_hmac_secret.clone(),
        )
        .await?;
    if let Some(message_id) = created_config_message_id {
        state.storage.write().config_message_id = Some(message_id);
    }
    log_config_database_write(state, record, created_config_message_id).await;
    Ok(created_config_message_id)
}

async fn log_specimen_database_write(
    state: &AppState,
    record: &SpecimenRecord,
    success: &SpecimenWriteSuccess,
    context: &SpecimenWriteLogContext,
) {
    info!(
        event = "database.specimen_written",
        guild_id = state.guild_id().get(),
        specimen_id = %success.specimen_id,
        ledger_message_id = success.ledger_message_id.get(),
        added_by_id = %record.source.added_by_id,
        source_author_id = %record.source.source_author_id,
        source_channel_id = %record.source.channel_id,
        source_message_id = %record.source.message_id,
        image_processing_ms = context.image_processing_ms.unwrap_or(0),
        image_processing_timed = context.image_processing_ms.is_some(),
        "specimen database record written"
    );

    let ledger_link = message_jump_link(
        state.guild_id(),
        success.channel_id,
        success.ledger_message_id,
    );
    let source_link = source_message_link(state.guild_id(), record);
    let mut event = BotLogEvent::new(
        "Database specimen added",
        format!(
            "A specimen record was written to `sightline-db` by <@{}>.",
            record.source.added_by_id
        ),
    )
    .color(BotLogColor::Success)
    .field(
        "Specimen",
        format!("`{}`\n{ledger_link}", success.specimen_id),
        false,
    )
    .field(
        "Actor",
        format!(
            "<@{}> (`{}`)",
            record.source.added_by_id, record.source.added_by_id
        ),
        true,
    )
    .field(
        "Source user",
        format!(
            "<@{}> (`{}`)",
            record.source.source_author_id, record.source.source_author_id
        ),
        true,
    )
    .field("Source message", source_link, false)
    .field(
        "Image fingerprint",
        format!(
            "xxh3-128 `{}`\npHash `{}`\ndHash `{}`\nsize `{}x{}`",
            record.image.byte_xxh128,
            record.image.phash64,
            record.image.dhash64,
            record.image.width,
            record.image.height
        ),
        false,
    );

    if let Some(image_url) = context.image_url.as_deref() {
        event = event.image_url(image_url).field("Image", image_url, false);
    }
    if let Some(processing_ms) = context.image_processing_ms {
        event = event.field(
            "Image processing time",
            format!("`{processing_ms}` ms"),
            true,
        );
    }
    if let Some(pre_add_match) = context.pre_add_match.as_deref() {
        event = event.field("Pre-add match", pre_add_match, false);
    }

    post_bot_log_for_config(state, &state.active_config(), event).await;
}

async fn log_config_database_write(
    state: &AppState,
    record: &GuildConfigRecord,
    created_message_id: Option<Id<MessageMarker>>,
) {
    let config = &record.config;
    info!(
        event = "database.config_written",
        guild_id = state.guild_id().get(),
        updated_by = %config.updated_by_id,
        enabled = config.enabled,
        database_channel_id = %config.ledger_channel_id,
        bot_log_channel_id = ?config.bot_log_channel_id,
        created_message_id = created_message_id.map(Id::get),
        "guild config database record written"
    );

    let config_message = created_message_id.map_or_else(
        || "Existing config message updated.".to_owned(),
        |message_id| {
            message_jump_link(
                state.guild_id(),
                state.storage.read().channel_id,
                message_id,
            )
        },
    );
    let event = BotLogEvent::new(
        "Database config updated",
        format!(
            "The guild configuration record was written by <@{}>.",
            config.updated_by_id
        ),
    )
    .color(BotLogColor::Success)
    .field(
        "Actor",
        format!("<@{}> (`{}`)", config.updated_by_id, config.updated_by_id),
        true,
    )
    .field("Enabled", format!("`{}`", config.enabled), true)
    .field("Config message", config_message, false)
    .field(
        "Database channel",
        format!("<#{}>", config.ledger_channel_id),
        true,
    )
    .field(
        "Bot log channel",
        config
            .bot_log_channel_id
            .as_deref()
            .map_or_else(|| "Not set".to_owned(), |id| format!("<#{id}>")),
        true,
    )
    .field(
        "Moderator roles",
        role_ids_label(&config.moderator_role_ids),
        false,
    )
    .field(
        "Scan-exempt roles",
        role_ids_label(&config.scan_exempt_role_ids),
        false,
    )
    .field(
        "Scan policy",
        scan_policy_summary(&config.scan_policy),
        false,
    )
    .field(
        "Advanced detection",
        detection_hyperparameters_summary(&config.detection_hyperparameters),
        false,
    )
    .field(
        "Detection policy",
        detection_policy_summary(&config.detection_policy),
        false,
    )
    .field(
        "Text gate",
        text_gate_policy_summary(&config.text_gate_policy),
        false,
    );

    post_bot_log_for_config(state, config, event).await;
}

async fn post_bot_log_for_config(state: &AppState, config: &GuildConfig, event: BotLogEvent) {
    let Some(channel_id) = config.bot_log_channel_id() else {
        return;
    };
    let copy_kind = event.copy_kind;
    let mut log = render_bot_log(event);
    stamp_bot_start_id(&mut log, &state.bot.bot_start_id);
    log.content = match copy_kind {
        crate::bot::discord::BotLogCopyKind::General => {
            config.discord_general_log_message_content()
        }
        crate::bot::discord::BotLogCopyKind::ConfirmedDetection => {
            config.discord_confirmed_log_message_content()
        }
        crate::bot::discord::BotLogCopyKind::SuspiciousDetection => {
            config.discord_suspicious_log_message_content()
        }
        crate::bot::discord::BotLogCopyKind::BenignDetection => {
            config.discord_benign_log_message_content()
        }
    };
    if let Err(source) = state
        .bot
        .bot_log_tx
        .send(BotLogWriteRequest {
            channel_id,
            log,
            kind: config.updated_by_id.parse::<u64>().ok().map_or(
                BotLogWriteKind::Standard,
                |updated_by| BotLogWriteKind::ConfigUpdate {
                    updated_by: Id::<UserMarker>::new(updated_by),
                },
            ),
            respond_to: None,
        })
        .await
    {
        warn!(
            event = "bot_log.enqueue_failed",
            ?source,
            "failed to enqueue bot log"
        );
    }
}

fn source_message_link(guild_id: Id<GuildMarker>, record: &SpecimenRecord) -> String {
    let channel_id = record.source.channel_id.parse::<u64>();
    let message_id = record.source.message_id.parse::<u64>();
    match (channel_id, message_id) {
        (Ok(channel_id), Ok(message_id)) => message_jump_link(
            guild_id,
            Id::<ChannelMarker>::new(channel_id),
            Id::<MessageMarker>::new(message_id),
        ),
        _ => format!(
            "channel `{}` message `{}`",
            record.source.channel_id, record.source.message_id
        ),
    }
}

fn role_ids_label(role_ids: &[String]) -> String {
    if role_ids.is_empty() {
        return "None".to_owned();
    }
    role_ids
        .iter()
        .map(|role_id| format!("<@&{role_id}>"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn detection_policy_summary(policy: &crate::configuration::guild::DetectionPolicy) -> String {
    format!(
        "confirmed=({}; {}) suspicious=({}; {})",
        threshold_summary(&policy.confirmed.threshold),
        actions_summary(&policy.confirmed.actions),
        threshold_summary(&policy.suspicious.threshold),
        actions_summary(&policy.suspicious.actions)
    )
}

async fn bot_log_writer_loop(
    discord: Arc<dyn DiscordEffects>,
    mut rx: mpsc::Receiver<BotLogWriteRequest>,
) {
    let mut last_config_logs =
        HashMap::<Id<ChannelMarker>, (Id<UserMarker>, Id<MessageMarker>)>::new();

    while let Some(request) = rx.recv().await {
        let channel_id = request.channel_id;
        let result = match request.kind {
            BotLogWriteKind::ConfigUpdate { updated_by }
                if request.respond_to.is_none()
                    && last_config_logs
                        .get(&channel_id)
                        .is_some_and(|(last_user, _)| *last_user == updated_by) =>
            {
                let (_, message_id) = last_config_logs[&channel_id];
                discord
                    .edit_bot_log_in_channel(request.channel_id, message_id, request.log)
                    .await
                    .map(|()| message_id)
            }
            _ => {
                let result = discord
                    .post_bot_log_to_channel(request.channel_id, request.log)
                    .await;
                if let Ok(message_id) = result {
                    match request.kind {
                        BotLogWriteKind::ConfigUpdate { updated_by }
                            if request.respond_to.is_none() =>
                        {
                            last_config_logs.insert(channel_id, (updated_by, message_id));
                        }
                        _ => {
                            last_config_logs.remove(&channel_id);
                        }
                    }
                } else {
                    last_config_logs.remove(&channel_id);
                }
                result
            }
        };
        if let Some(respond_to) = request.respond_to {
            let _ = respond_to.send(result);
        } else if let Err(source) = result {
            last_config_logs.remove(&channel_id);
            warn!(
                event = "bot_log.post_failed",
                channel_id = request.channel_id.get(),
                ?source,
                "failed to post bot log"
            );
        }
    }
}

async fn performance_report_loop(mut rx: mpsc::Receiver<ImageMetricsCommand>) {
    let mut trackers = HashMap::<Id<GuildMarker>, GuildMetricsTracker>::new();
    let mut report_interval = tokio::time::interval(Duration::from_hours(6));
    report_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    report_interval.tick().await;

    loop {
        tokio::select! {
            command = rx.recv() => {
                let Some(command) = command else {
                    log_all_metrics_snapshots(&trackers, true);
                    break;
                };
                match command {
                    ImageMetricsCommand::Record(event) => {
                        record_image_metric_event(&mut trackers, &event);
                    }
                    ImageMetricsCommand::RemoveGuild { guild_id } => {
                        trackers.remove(&guild_id);
                    }
                    ImageMetricsCommand::Snapshot { guild_id, respond_to } => {
                        let snapshot = trackers
                            .get(&guild_id)
                            .map_or_else(GuildMetricsSnapshot::default, GuildMetricsTracker::snapshot);
                        let _ = respond_to.send(snapshot);
                    }
                }
            }
            _ = report_interval.tick() => {
                log_all_metrics_snapshots(&trackers, false);
                for tracker in trackers.values_mut() {
                    tracker.reset_period();
                }
            }
        }
    }
}

#[derive(Debug)]
struct GuildMetricsTracker {
    period: GuildMetricCounters,
    total: GuildMetricCounters,
    period_perf: ImagePerfTracker,
    total_perf: ImagePerfTracker,
    period_timing: PipelineTimingTracker,
    total_timing: PipelineTimingTracker,
    period_timing_by_class: Vec<PipelineTimingTracker>,
    total_timing_by_class: Vec<PipelineTimingTracker>,
}

impl GuildMetricsTracker {
    const PERF_SAMPLE_CAP: usize = 512;
    const TIMING_SAMPLE_CAP: usize = 512;

    fn new() -> Self {
        Self {
            period: GuildMetricCounters::default(),
            total: GuildMetricCounters::default(),
            period_perf: ImagePerfTracker::new(Self::PERF_SAMPLE_CAP),
            total_perf: ImagePerfTracker::new(Self::PERF_SAMPLE_CAP),
            period_timing: PipelineTimingTracker::new(Self::TIMING_SAMPLE_CAP),
            total_timing: PipelineTimingTracker::new(Self::TIMING_SAMPLE_CAP),
            period_timing_by_class: new_class_timing_trackers(),
            total_timing_by_class: new_class_timing_trackers(),
        }
    }

    fn snapshot(&self) -> GuildMetricsSnapshot {
        GuildMetricsSnapshot {
            period: self.period.clone(),
            total: self.total.clone(),
            period_perf: self.period_perf.snapshot(),
            total_perf: self.total_perf.snapshot(),
            period_timing: self.period_timing.snapshot(),
            total_timing: self.total_timing.snapshot(),
            period_timing_by_class: class_timing_snapshots(&self.period_timing_by_class),
            total_timing_by_class: class_timing_snapshots(&self.total_timing_by_class),
        }
    }

    fn reset_period(&mut self) {
        self.period = GuildMetricCounters::default();
        self.period_perf = ImagePerfTracker::new(Self::PERF_SAMPLE_CAP);
        self.period_timing = PipelineTimingTracker::new(Self::TIMING_SAMPLE_CAP);
        self.period_timing_by_class = new_class_timing_trackers();
    }
}

#[derive(Debug, Clone, Copy)]
enum PipelineTimingClass {
    Pass,
    ScanFailed,
    HardExact,
    HardPerceptual,
    HardLocal,
    SuspiciousExact,
    SuspiciousPerceptual,
    SuspiciousLocal,
}

impl PipelineTimingClass {
    const ALL: [Self; 8] = [
        Self::Pass,
        Self::ScanFailed,
        Self::HardExact,
        Self::HardPerceptual,
        Self::HardLocal,
        Self::SuspiciousExact,
        Self::SuspiciousPerceptual,
        Self::SuspiciousLocal,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::ScanFailed => "scan_failed",
            Self::HardExact => "hard_exact",
            Self::HardPerceptual => "hard_perceptual",
            Self::HardLocal => "hard_local",
            Self::SuspiciousExact => "suspicious_exact",
            Self::SuspiciousPerceptual => "suspicious_perceptual",
            Self::SuspiciousLocal => "suspicious_local",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Pass => 0,
            Self::ScanFailed => 1,
            Self::HardExact => 2,
            Self::HardPerceptual => 3,
            Self::HardLocal => 4,
            Self::SuspiciousExact => 5,
            Self::SuspiciousPerceptual => 6,
            Self::SuspiciousLocal => 7,
        }
    }

    const fn from_decision(decision: ImageScanDecisionMetric) -> Self {
        match decision {
            ImageScanDecisionMetric::Pass => Self::Pass,
            ImageScanDecisionMetric::ScanFailed => Self::ScanFailed,
            ImageScanDecisionMetric::HardMatch(ImageMatchStageMetric::ExactXxh128) => {
                Self::HardExact
            }
            ImageScanDecisionMetric::HardMatch(ImageMatchStageMetric::Perceptual) => {
                Self::HardPerceptual
            }
            ImageScanDecisionMetric::HardMatch(ImageMatchStageMetric::LocalAnchors) => {
                Self::HardLocal
            }
            ImageScanDecisionMetric::Suspicious(ImageMatchStageMetric::ExactXxh128) => {
                Self::SuspiciousExact
            }
            ImageScanDecisionMetric::Suspicious(ImageMatchStageMetric::Perceptual) => {
                Self::SuspiciousPerceptual
            }
            ImageScanDecisionMetric::Suspicious(ImageMatchStageMetric::LocalAnchors) => {
                Self::SuspiciousLocal
            }
        }
    }
}

fn new_class_timing_trackers() -> Vec<PipelineTimingTracker> {
    const CLASS_TIMING_SAMPLE_CAP: usize = 128;
    PipelineTimingClass::ALL
        .into_iter()
        .map(|_| PipelineTimingTracker::new(CLASS_TIMING_SAMPLE_CAP))
        .collect()
}

fn class_timing_snapshots(trackers: &[PipelineTimingTracker]) -> Vec<PipelineTimingClassSnapshot> {
    PipelineTimingClass::ALL
        .into_iter()
        .zip(trackers)
        .map(|(class, tracker)| PipelineTimingClassSnapshot {
            label: class.label(),
            timing: tracker.snapshot(),
        })
        .collect()
}

#[derive(Debug)]
struct TimingDistributionTracker {
    samples: std::collections::VecDeque<u64>,
    sample_sum_us: u128,
    total_count: u64,
    sample_cap: usize,
}

impl TimingDistributionTracker {
    fn new(sample_cap: usize) -> Self {
        Self {
            samples: std::collections::VecDeque::new(),
            sample_sum_us: 0,
            total_count: 0,
            sample_cap,
        }
    }

    fn record_nonzero(&mut self, value_us: u64) {
        if value_us > 0 {
            self.record(value_us);
        }
    }

    fn record(&mut self, value_us: u64) {
        self.total_count += 1;
        if self.sample_cap == 0 {
            return;
        }
        if self.samples.len() == self.sample_cap
            && let Some(removed) = self.samples.pop_front()
        {
            self.sample_sum_us = self.sample_sum_us.saturating_sub(u128::from(removed));
        }
        self.samples.push_back(value_us);
        self.sample_sum_us += u128::from(value_us);
    }

    fn snapshot(&self) -> Option<TimingDistributionSnapshot> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted = self.samples.iter().copied().collect::<Vec<_>>();
        sorted.sort_unstable();
        let sample_count = sorted.len();
        Some(TimingDistributionSnapshot {
            count: self.total_count,
            max_us: *sorted.last().unwrap_or(&0),
            avg_us: average_us(self.sample_sum_us, sample_count),
            p95_us: timing_percentile_us(&sorted, 95, 100),
            p99_us: timing_percentile_us(&sorted, 99, 100),
        })
    }
}

fn average_us(sum_us: u128, sample_count: usize) -> u64 {
    let count = u128::try_from(sample_count).unwrap_or(u128::MAX).max(1);
    u64::try_from(sum_us / count).unwrap_or(u64::MAX)
}

fn timing_percentile_us(sorted: &[u64], numerator: usize, denominator: usize) -> u64 {
    if sorted.is_empty() || denominator == 0 {
        return 0;
    }
    let last = sorted.len().saturating_sub(1);
    let index = last
        .saturating_mul(numerator)
        .saturating_add(denominator / 2)
        / denominator;
    sorted[index.min(last)]
}

#[derive(Debug)]
struct PipelineTimingTracker {
    preview_used: u64,
    preview_fallbacks: u64,
    total: TimingDistributionTracker,
    preview_download: TimingDistributionTracker,
    preview_fingerprint: TimingDistributionTracker,
    preview_matcher: TimingDistributionTracker,
    queue_wait: TimingDistributionTracker,
    download: TimingDistributionTracker,
    download_request: TimingDistributionTracker,
    download_body: TimingDistributionTracker,
    download_gate_wait: TimingDistributionTracker,
    flagged_cache_lookup: TimingDistributionTracker,
    exact_match_lookup: TimingDistributionTracker,
    singleflight_wait: TimingDistributionTracker,
    fingerprint: TimingDistributionTracker,
    fingerprint_pipeline: FingerprintPipelineTimingTracker,
    matcher: TimingDistributionTracker,
    ocr_crop: TimingDistributionTracker,
    progressive_eval: TimingDistributionTracker,
}

#[derive(Debug)]
struct FingerprintPipelineTimingTracker {
    decode: TimingDistributionTracker,
    thumbnail: TimingDistributionTracker,
    visual: TimingDistributionTracker,
    orientation: TimingDistributionTracker,
    perceptual: TimingDistributionTracker,
    normalize: TimingDistributionTracker,
    tile_scorer: TimingDistributionTracker,
    text_grid: TimingDistributionTracker,
    local_anchors: TimingDistributionTracker,
    local_hashes: TimingDistributionTracker,
}

impl FingerprintPipelineTimingTracker {
    fn new(sample_cap: usize) -> Self {
        Self {
            decode: TimingDistributionTracker::new(sample_cap),
            thumbnail: TimingDistributionTracker::new(sample_cap),
            visual: TimingDistributionTracker::new(sample_cap),
            orientation: TimingDistributionTracker::new(sample_cap),
            perceptual: TimingDistributionTracker::new(sample_cap),
            normalize: TimingDistributionTracker::new(sample_cap),
            tile_scorer: TimingDistributionTracker::new(sample_cap),
            text_grid: TimingDistributionTracker::new(sample_cap),
            local_anchors: TimingDistributionTracker::new(sample_cap),
            local_hashes: TimingDistributionTracker::new(sample_cap),
        }
    }

    fn record(&mut self, sample: ImageFingerprintTimingSample) {
        self.decode.record_nonzero(sample.decode);
        self.thumbnail.record_nonzero(sample.thumbnail);
        self.visual.record_nonzero(sample.visual);
        self.orientation.record_nonzero(sample.orientation);
        self.perceptual.record_nonzero(sample.perceptual);
        self.normalize.record_nonzero(sample.normalize);
        self.tile_scorer.record_nonzero(sample.tile_scorer);
        self.text_grid.record_nonzero(sample.text_grid);
        self.local_anchors.record_nonzero(sample.local_anchors);
        self.local_hashes.record_nonzero(sample.local_hashes);
    }

    fn snapshot(&self) -> FingerprintPipelineTimingSnapshot {
        FingerprintPipelineTimingSnapshot {
            decode: self.decode.snapshot(),
            thumbnail: self.thumbnail.snapshot(),
            visual: self.visual.snapshot(),
            orientation: self.orientation.snapshot(),
            perceptual: self.perceptual.snapshot(),
            normalize: self.normalize.snapshot(),
            tile_scorer: self.tile_scorer.snapshot(),
            text_grid: self.text_grid.snapshot(),
            local_anchors: self.local_anchors.snapshot(),
            local_hashes: self.local_hashes.snapshot(),
        }
    }
}

impl PipelineTimingTracker {
    fn new(sample_cap: usize) -> Self {
        Self {
            preview_used: 0,
            preview_fallbacks: 0,
            total: TimingDistributionTracker::new(sample_cap),
            preview_download: TimingDistributionTracker::new(sample_cap),
            preview_fingerprint: TimingDistributionTracker::new(sample_cap),
            preview_matcher: TimingDistributionTracker::new(sample_cap),
            queue_wait: TimingDistributionTracker::new(sample_cap),
            download: TimingDistributionTracker::new(sample_cap),
            download_request: TimingDistributionTracker::new(sample_cap),
            download_body: TimingDistributionTracker::new(sample_cap),
            download_gate_wait: TimingDistributionTracker::new(sample_cap),
            flagged_cache_lookup: TimingDistributionTracker::new(sample_cap),
            exact_match_lookup: TimingDistributionTracker::new(sample_cap),
            singleflight_wait: TimingDistributionTracker::new(sample_cap),
            fingerprint: TimingDistributionTracker::new(sample_cap),
            fingerprint_pipeline: FingerprintPipelineTimingTracker::new(sample_cap),
            matcher: TimingDistributionTracker::new(sample_cap),
            ocr_crop: TimingDistributionTracker::new(sample_cap),
            progressive_eval: TimingDistributionTracker::new(sample_cap),
        }
    }

    fn record(&mut self, sample: &ImageStageTimingSample) {
        if sample.preview_used {
            self.preview_used += 1;
        }
        if sample.preview_fallback {
            self.preview_fallbacks += 1;
        }
        self.total.record_nonzero(sample.total_us);
        self.preview_download
            .record_nonzero(sample.preview_download_us);
        self.preview_fingerprint
            .record_nonzero(sample.preview_fingerprint_us);
        self.preview_matcher
            .record_nonzero(sample.preview_matcher_us);
        self.queue_wait.record_nonzero(sample.queue_wait_us);
        self.download.record_nonzero(sample.download_us);
        self.download_request
            .record_nonzero(sample.download_request_us);
        self.download_body.record_nonzero(sample.download_body_us);
        self.download_gate_wait
            .record_nonzero(sample.download_gate_wait_us);
        self.flagged_cache_lookup
            .record_nonzero(sample.flagged_cache_lookup_us);
        self.exact_match_lookup
            .record_nonzero(sample.exact_match_lookup_us);
        self.singleflight_wait
            .record_nonzero(sample.singleflight_wait_us);
        self.fingerprint.record_nonzero(sample.fingerprint_us);
        self.fingerprint_pipeline
            .record(sample.fingerprint_pipeline);
        self.matcher.record_nonzero(sample.matcher_us);
        self.ocr_crop.record_nonzero(sample.ocr_crop_us);
        self.progressive_eval
            .record_nonzero(sample.progressive_eval_us);
    }

    fn snapshot(&self) -> PipelineTimingSnapshot {
        PipelineTimingSnapshot {
            preview_used: self.preview_used,
            preview_fallbacks: self.preview_fallbacks,
            total: self.total.snapshot(),
            preview_download: self.preview_download.snapshot(),
            preview_fingerprint: self.preview_fingerprint.snapshot(),
            preview_matcher: self.preview_matcher.snapshot(),
            queue_wait: self.queue_wait.snapshot(),
            download: self.download.snapshot(),
            download_request: self.download_request.snapshot(),
            download_body: self.download_body.snapshot(),
            download_gate_wait: self.download_gate_wait.snapshot(),
            flagged_cache_lookup: self.flagged_cache_lookup.snapshot(),
            exact_match_lookup: self.exact_match_lookup.snapshot(),
            singleflight_wait: self.singleflight_wait.snapshot(),
            fingerprint: self.fingerprint.snapshot(),
            fingerprint_pipeline: self.fingerprint_pipeline.snapshot(),
            matcher: self.matcher.snapshot(),
            ocr_crop: self.ocr_crop.snapshot(),
            progressive_eval: self.progressive_eval.snapshot(),
        }
    }
}

fn record_image_metric_event(
    trackers: &mut HashMap<Id<GuildMarker>, GuildMetricsTracker>,
    event: &ImageMetricEvent,
) {
    let guild_id = match event {
        ImageMetricEvent::Processed(sample) => sample.guild_id,
        ImageMetricEvent::OcrCall { guild_id } | ImageMetricEvent::OcrResolved { guild_id, .. } => {
            *guild_id
        }
    };
    let tracker = trackers
        .entry(guild_id)
        .or_insert_with(GuildMetricsTracker::new);
    tracker.period.record(event);
    tracker.total.record(event);
    if let ImageMetricEvent::Processed(sample) = event {
        if let Some(timings) = sample.stage_timings.as_deref() {
            tracker.period_timing.record(timings);
            tracker.total_timing.record(timings);
            let class = PipelineTimingClass::from_decision(sample.decision).index();
            if let Some(tracker) = tracker.period_timing_by_class.get_mut(class) {
                tracker.record(timings);
            }
            if let Some(tracker) = tracker.total_timing_by_class.get_mut(class) {
                tracker.record(timings);
            }
        }
        tracker.period_perf.record(sample);
        tracker.total_perf.record(sample);
    }
}

fn log_all_metrics_snapshots(
    trackers: &HashMap<Id<GuildMarker>, GuildMetricsTracker>,
    final_dump: bool,
) {
    for (guild_id, tracker) in trackers {
        log_metrics_snapshot(*guild_id, tracker, final_dump);
    }
}

fn log_metrics_snapshot(
    guild_id: Id<GuildMarker>,
    tracker: &GuildMetricsTracker,
    final_dump: bool,
) {
    let period_perf = tracker.period_perf.snapshot();
    let total_perf = tracker.total_perf.snapshot();
    let period_timing = tracker.period_timing.snapshot();
    let total_timing = tracker.total_timing.snapshot();
    let period_timing_by_class = class_timing_snapshots(&tracker.period_timing_by_class);
    let total_timing_by_class = class_timing_snapshots(&tracker.total_timing_by_class);
    info!(
        event = "performance.image_matching",
        guild_id = guild_id.get(),
        final_dump,
        period = %tracker.period.summary(),
        total = %tracker.total.summary(),
        period_perf = %period_perf.as_ref().map_or_else(|| "none".to_owned(), performance_summary),
        total_perf = %total_perf.as_ref().map_or_else(|| "none".to_owned(), performance_summary),
        period_timing = %pipeline_timing_summary(&period_timing),
        total_timing = %pipeline_timing_summary(&total_timing),
        period_timing_by_class = %pipeline_timing_class_summary(&period_timing_by_class),
        total_timing_by_class = %pipeline_timing_class_summary(&total_timing_by_class),
        "image matching metrics summary"
    );
}

fn pipeline_timing_summary(snapshot: &PipelineTimingSnapshot) -> String {
    [
        timing_summary_part("total", snapshot.total.as_ref()),
        timing_summary_part("preview_dl", snapshot.preview_download.as_ref()),
        timing_summary_part("download", snapshot.download.as_ref()),
        timing_summary_part("fingerprint", snapshot.fingerprint.as_ref()),
        timing_summary_part("fp_decode", snapshot.fingerprint_pipeline.decode.as_ref()),
        timing_summary_part(
            "fp_perceptual",
            snapshot.fingerprint_pipeline.perceptual.as_ref(),
        ),
        timing_summary_part(
            "fp_normalize",
            snapshot.fingerprint_pipeline.normalize.as_ref(),
        ),
        timing_summary_part(
            "fp_text_grid",
            snapshot.fingerprint_pipeline.text_grid.as_ref(),
        ),
        timing_summary_part(
            "fp_dense",
            snapshot.fingerprint_pipeline.local_hashes.as_ref(),
        ),
        timing_summary_part("matcher", snapshot.matcher.as_ref()),
        timing_summary_part("queue_wait", snapshot.queue_wait.as_ref()),
        timing_summary_part("singleflight_wait", snapshot.singleflight_wait.as_ref()),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join("; ")
}

fn pipeline_timing_class_summary(classes: &[PipelineTimingClassSnapshot]) -> String {
    let parts = classes
        .iter()
        .filter_map(|class| {
            class
                .timing
                .total
                .as_ref()
                .map(|total| format!("{}:{}", class.label, timing_distribution_summary(total)))
        })
        .collect::<Vec<_>>();
    if parts.is_empty() {
        "none".to_owned()
    } else {
        parts.join("; ")
    }
}

fn timing_summary_part(
    label: &'static str,
    snapshot: Option<&TimingDistributionSnapshot>,
) -> Option<String> {
    snapshot.map(|snapshot| format!("{label}:{}", timing_distribution_summary(snapshot)))
}

fn timing_distribution_summary(snapshot: &TimingDistributionSnapshot) -> String {
    format!(
        "avg={}us p95={}us max={}us n={}",
        snapshot.avg_us, snapshot.p95_us, snapshot.max_us, snapshot.count
    )
}

fn performance_summary(snapshot: &crate::image::types::ImagePerfSnapshot) -> String {
    format!(
        "count={} success={} failure={} sample={} min={}ms max={}ms avg={:.2}ms p50={}ms p90={}ms p95={}ms p99={}ms",
        snapshot.total_count,
        snapshot.total_success,
        snapshot.total_failure,
        snapshot.sample_count,
        snapshot.min_ms,
        snapshot.max_ms,
        snapshot.avg_ms,
        snapshot.p50_ms,
        snapshot.p90_ms,
        snapshot.p95_ms,
        snapshot.p99_ms
    )
}

impl GuildMetricCounters {
    fn record(&mut self, event: &ImageMetricEvent) {
        match event {
            ImageMetricEvent::Processed(sample) => self.record_processed(sample),
            ImageMetricEvent::OcrCall { .. } => self.ocr_calls += 1,
            ImageMetricEvent::OcrResolved { resolution, .. } => match resolution {
                TextGateResolutionMetric::Good => self.ocr_resolved_good += 1,
                TextGateResolutionMetric::Bad => self.ocr_resolved_bad += 1,
                TextGateResolutionMetric::Unknown => self.ocr_resolved_unknown += 1,
            },
        }
    }

    fn record_processed(&mut self, sample: &ImagePerfSample) {
        self.images_scanned += 1;
        match sample.decision {
            ImageScanDecisionMetric::Pass => self.passes += 1,
            ImageScanDecisionMetric::ScanFailed => self.scan_failures += 1,
            ImageScanDecisionMetric::HardMatch(stage) => {
                self.hard_matches += 1;
                self.increment_hard_stage(stage);
            }
            ImageScanDecisionMetric::Suspicious(stage) => {
                self.suspicious_matches += 1;
                self.increment_suspicious_stage(stage);
            }
        }
    }

    fn increment_hard_stage(&mut self, stage: ImageMatchStageMetric) {
        match stage {
            ImageMatchStageMetric::ExactXxh128 => self.hard_exact_xxh128 += 1,
            ImageMatchStageMetric::Perceptual => self.hard_perceptual += 1,
            ImageMatchStageMetric::LocalAnchors => self.hard_local_anchors += 1,
        }
    }

    fn increment_suspicious_stage(&mut self, stage: ImageMatchStageMetric) {
        match stage {
            ImageMatchStageMetric::ExactXxh128 => self.suspicious_exact_xxh128 += 1,
            ImageMatchStageMetric::Perceptual => self.suspicious_perceptual += 1,
            ImageMatchStageMetric::LocalAnchors => self.suspicious_local_anchors += 1,
        }
    }

    pub(crate) fn summary(&self) -> String {
        format!(
            "scanned={} pass={} failed={} hard={} [exact={}, perceptual={}, local={}] suspicious={} [exact={}, perceptual={}, local={}] ocr_calls={} ocr_resolved=[good={}, bad={}, unknown={}]",
            self.images_scanned,
            self.passes,
            self.scan_failures,
            self.hard_matches,
            self.hard_exact_xxh128,
            self.hard_perceptual,
            self.hard_local_anchors,
            self.suspicious_matches,
            self.suspicious_exact_xxh128,
            self.suspicious_perceptual,
            self.suspicious_local_anchors,
            self.ocr_calls,
            self.ocr_resolved_good,
            self.ocr_resolved_bad,
            self.ocr_resolved_unknown
        )
    }
}

async fn log_effective_options(state: &AppState) {
    let configured = state.guild_configured.load(Ordering::Acquire);
    let config = state.active_config();
    info!(
        event = "options.effective",
        configured,
        enabled = config.enabled,
        database_channel_id = config.ledger_channel_id,
        bot_log_channel_id = config.bot_log_channel_id,
        verified_role_id = config.verified_role_id,
        moderator_roles = ?config.moderator_role_ids,
        scan_exempt_roles = ?config.scan_exempt_role_ids,
        allowed_extensions = ?config.scan_policy.allowed_extensions,
        max_file_bytes = config.scan_policy.max_file_bytes,
        detection_hyperparameters = ?config.detection_hyperparameters,
        detection_policy = ?config.detection_policy,
        text_gate_policy = ?config.text_gate_policy,
        queue_max_size = state.config.queue.max_size,
        enqueue_timeout_ms = state.config.queue.enqueue_timeout_ms,
        cpu_concurrency = state.config.queue.cpu_concurrency,
        download_concurrency = state.config.queue.download_concurrency,
        image_worker_concurrency = state.config.queue.image_worker_concurrency(),
        ocr_concurrency = state.config.queue.ocr_concurrency,
        max_images_per_message = state.config.queue.max_images_per_message,
        hash_outcome_cache_size = state.config.queue.hash_outcome_cache_size,
        download_memory_max_bytes = state.config.queue.download_memory_max_bytes,
        decoded_image_memory_max_bytes = state.config.queue.decoded_image_memory_max_bytes,
        byte_store_max_bytes = state.config.queue.byte_store_max_bytes,
        exempt_administrators = config.scan_policy.exempt_administrators,
        max_bytes = state.config.download.max_bytes,
        max_decoded_pixels = state.config.download.max_decoded_pixels,
        timeout_seconds = state.config.download.timeout_seconds,
        max_retries = state.config.download.max_retries,
        retry_base_delay_ms = state.config.download.retry_base_delay_ms,
        warmer_enabled = state.config.download.warmer_enabled,
        warmer_period_seconds = state.config.download.warmer_period_seconds,
        image_pool_idle_timeout_seconds = IMAGE_POOL_IDLE_TIMEOUT.as_secs(),
        ocr_space_configured = state.ocr_space.is_some(),
        ocr_space_endpoint = state.config.ocr_space.endpoint,
        ocr_space_timeout_seconds = state.config.ocr_space.timeout_seconds,
        ocr_space_total_timeout_seconds = state.config.ocr_space.total_timeout_seconds,
        ocr_space_max_retries = state.config.ocr_space.max_retries,
        ocr_space_language = state.config.ocr_space.language,
        ocr_space_scale = state.config.ocr_space.scale,
        ocr_space_detect_orientation = state.config.ocr_space.detect_orientation,
        phash64_max_distance = state.config.matching.phash64_max_distance,
        dhash64_max_distance = state.config.matching.dhash64_max_distance,
        "effective bot options"
    );

    if configured {
        state
            .post_bot_log(format!(
                "Sightline started.\nEnabled: `{}`\nDatabase channel: <#{}>\nBot log channel: {}\nModerator roles: {}\nScan-exempt roles: {}\nScan policy: {}\nAdvanced detection: {}\nActions: {}\nText gate: {}\nRuntime: max_images=`{}`, enqueue_timeout_ms=`{}`, hash_outcome_cache=`{}`, download_memory=`{}`, decoded_memory=`{}`, byte_store=`{}`, cpu_concurrency=`{}`, download_concurrency=`{}`, ocr_concurrency=`{}`, max_bytes=`{}`, max_pixels=`{}`, download_timeout=`{}s`, download_retries=`{}`, download_retry_base=`{}ms`, download_warmer=`{}`/`{}s`, pHash=`{}`, dHash=`{}`\nOCR.space: configured=`{}`, endpoint=`{}`, request_timeout=`{}s`, total_timeout=`{}s`, retries=`{}`, language=`{}`, scale=`{}`, orientation=`{}`",
                config.enabled,
                config.ledger_channel_id,
                config
                    .bot_log_channel_id
                    .as_deref()
                    .map_or_else(|| "Not set".to_owned(), |id| format!("<#{id}>")),
                config.moderator_role_ids.iter().map(|id| format!("<@&{id}>")).collect::<Vec<_>>().join(", "),
                config.scan_exempt_role_ids.iter().map(|id| format!("<@&{id}>")).collect::<Vec<_>>().join(", "),
                scan_policy_summary(&config.scan_policy),
                detection_hyperparameters_summary(&config.detection_hyperparameters),
                {
                    let confirmed = &config.detection_policy.confirmed;
                    let suspicious = &config.detection_policy.suspicious;
                    format!(
                        "confirmed=({}; {}) suspicious=({}; {})",
                        threshold_summary(&confirmed.threshold),
                        actions_summary(&confirmed.actions),
                        threshold_summary(&suspicious.threshold),
                        actions_summary(&suspicious.actions)
                    )
                },
                text_gate_policy_summary(&config.text_gate_policy),
                state.config.queue.max_images_per_message,
                state.config.queue.enqueue_timeout_ms,
                state.config.queue.hash_outcome_cache_size,
                state.config.queue.download_memory_max_bytes,
                state.config.queue.decoded_image_memory_max_bytes,
                state.config.queue.byte_store_max_bytes,
                state.config.queue.cpu_concurrency,
                state.config.queue.download_concurrency,
                state.config.queue.ocr_concurrency,
                state.config.download.max_bytes,
                state.config.download.max_decoded_pixels,
                state.config.download.timeout_seconds,
                state.config.download.max_retries,
                state.config.download.retry_base_delay_ms,
                state.config.download.warmer_enabled,
                state.config.download.warmer_period_seconds,
                state.config.matching.phash64_max_distance,
                state.config.matching.dhash64_max_distance,
                state.ocr_space.is_some(),
                state.config.ocr_space.endpoint,
                state.config.ocr_space.timeout_seconds,
                state.config.ocr_space.total_timeout_seconds,
                state.config.ocr_space.max_retries,
                state.config.ocr_space.language,
                state.config.ocr_space.scale,
                state.config.ocr_space.detect_orientation
            ))
            .await;
    }
}

fn generate_bot_start_id() -> Result<String> {
    let mut bytes = [0_u8; 16];
    getrandom::getrandom(&mut bytes)
        .map_err(|source| anyhow!("generating bot start id: {source}"))?;
    Ok(hex::encode(bytes))
}

async fn run_gateway(state: BotState, tx: mpsc::Sender<ImageCandidate>) -> Result<()> {
    let gateway_config = GatewayConfig::new(
        state.secrets.discord_token.clone(),
        Intents::GUILDS | Intents::GUILD_MESSAGES | Intents::MESSAGE_CONTENT,
    );
    let shards =
        twilight_gateway::create_recommended(&state.discord, gateway_config, |_, builder| {
            builder.build()
        })
        .await
        .context("creating gateway shards")?;

    let mut tasks = JoinSet::new();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    for shard in shards {
        let state = state.clone();
        let tx = tx.clone();
        let shutdown_rx = shutdown_rx.clone();
        tasks.spawn(async move {
            dispatch_event_stream(state, tx, TwilightShardEventStream::new(shard), shutdown_rx)
                .await;
        });
    }

    loop {
        tokio::select! {
            result = tasks.join_next(), if !tasks.is_empty() => {
                match result {
                    Some(Ok(())) => warn!(event = "gateway.shard_exited", remaining_shards = tasks.len(), "gateway shard exited"),
                    Some(Err(source)) => warn!(event = "gateway.shard_task_failed", remaining_shards = tasks.len(), ?source, "gateway shard task failed"),
                    None => {
                        warn!(event = "gateway.no_shards", "all gateway shards exited");
                        break;
                    }
                }
                if tasks.is_empty() {
                    warn!(event = "gateway.no_shards", "all gateway shards exited");
                    break;
                }
            }
            result = signal::ctrl_c() => {
                result.context("waiting for ctrl-c")?;
                info!(event = "shutdown.requested", "shutdown requested");
                break;
            }
        };
    }
    let _ = shutdown_tx.send(true);

    while let Some(result) = tasks.join_next().await {
        if let Err(source) = result {
            warn!(
                event = "gateway.shard_task_failed",
                ?source,
                "gateway shard task failed during shutdown"
            );
        }
    }

    Ok(())
}

pub(crate) async fn dispatch_event_stream<S>(
    state: BotState,
    tx: mpsc::Sender<ImageCandidate>,
    mut stream: S,
    mut shutdown: watch::Receiver<bool>,
) where
    S: BotEventStream + 'static,
{
    loop {
        let item = tokio::select! {
            _ = shutdown.changed() => {
                stream.close();
                break;
            }
            item = stream.next_event() => item,
        };

        let Some(item) = item else {
            break;
        };

        let event = match item {
            Ok(event) => event,
            Err(source) => {
                warn!(
                    event = "gateway.event_error",
                    ?source,
                    "gateway event error"
                );
                continue;
            }
        };

        if !dispatch_event(&state, &tx, event).await {
            stream.close();
            break;
        }
    }
}

pub(crate) async fn dispatch_event(
    state: &BotState,
    tx: &mpsc::Sender<ImageCandidate>,
    event: Event,
) -> bool {
    request_guild_load_for_event(state, &event);

    match event {
        Event::GatewayClose(frame) => {
            warn!(
                event = "gateway.close",
                ?frame,
                "gateway close event observed; leaving reconnect handling to the shard stream"
            );
            true
        }
        Event::GuildDelete(guild) => {
            if guild.unavailable == Some(true) {
                info!(
                    event = "guild.unavailable",
                    guild_id = guild.id.get(),
                    "guild became temporarily unavailable; keeping in-memory runtime"
                );
            } else {
                state.cleanup_removed_guild(guild.id);
            }
            true
        }
        Event::MessageCreate(message) => {
            if let Err(source) = enqueue_message_create(state, tx, &message).await {
                warn!(
                    event = "message.enqueue_failed",
                    ?source,
                    "failed to enqueue message"
                );
            }
            true
        }
        Event::MessageUpdate(message) => {
            if let Err(source) = enqueue_message_create(state, tx, &message).await {
                warn!(
                    event = "message_update.enqueue_failed",
                    ?source,
                    "failed to enqueue updated message"
                );
            }
            true
        }
        Event::MessageDelete(message) => {
            handle_ledger_message_delete(state, message.channel_id, message.id).await;
            true
        }
        Event::MessageDeleteBulk(messages) => {
            for message_id in &messages.ids {
                handle_ledger_message_delete(state, messages.channel_id, *message_id).await;
            }
            true
        }
        Event::ChannelDelete(channel) => {
            handle_channel_delete(state, channel.id);
            true
        }
        Event::InteractionCreate(interaction) => {
            let state = state.clone();
            let tx = tx.clone();
            match state.interaction_gate.clone().try_acquire_owned() {
                Ok(permit) => {
                    let shutdown = state.shutdown.clone();
                    let tracker = state.background_tasks.clone();
                    tracker.spawn(async move {
                        let _permit = permit;
                        let result = tokio::select! {
                            () = shutdown.cancelled() => return,
                            result = handle_interaction(state, tx, interaction) => result,
                        };
                        if let Err(source) = result {
                            warn!(event = "interaction.failed", ?source, "interaction failed");
                        }
                    });
                }
                Err(tokio::sync::TryAcquireError::NoPermits) => {
                    warn!(
                        event = "interaction.overload_drop",
                        "interaction queue full; dropping interaction"
                    );
                }
                Err(tokio::sync::TryAcquireError::Closed) => {
                    warn!(event = "interaction.gate_closed", "interaction gate closed");
                }
            }
            true
        }
        _ => true,
    }
}

fn request_guild_load_for_event(state: &BotState, event: &Event) {
    if matches!(event, Event::GuildDelete(guild) if guild.unavailable != Some(true)) {
        return;
    }

    if let Some(guild_id) = event.guild_id() {
        state.request_guild_load(guild_id);
    }
}

async fn enqueue_message_create(
    state: &BotState,
    tx: &mpsc::Sender<ImageCandidate>,
    message: &Message,
) -> Result<()> {
    let Some(guild_id) = message.guild_id else {
        return Ok(());
    };
    let Some(state) = state.cached_guild(guild_id) else {
        state.request_guild_load(guild_id);
        return Ok(());
    };

    if message.author.id == state.bot_user_id || !message_has_potential_image(message) {
        return Ok(());
    }

    if !state.guild_active() {
        return Ok(());
    }

    let guild_config = state.active_config();
    if should_skip_message_scan(&state, &guild_config, message) {
        return Ok(());
    }

    let candidates = extract_candidates_from_message(
        guild_id,
        message,
        state.config.queue.max_images_per_message,
        &guild_config.scan_policy.allowed_extensions,
        guild_config.scan_policy.max_file_bytes,
    );
    if candidates.is_empty() {
        return Ok(());
    }

    let scan_exempt_roles = state.scan_exempt_roles.load();
    match message_has_any_role(message, scan_exempt_roles.as_ref()) {
        Some(true) => {
            return Ok(());
        }
        Some(false) => {}
        None => warn!(
            event = "scan.exemption_roles_unavailable",
            message_id = message.id.get(),
            author_id = message.author.id.get(),
            "member roles were not present on gateway payload; scanning image"
        ),
    }

    enqueue_candidates(&state, tx, candidates).await
}

fn should_skip_message_scan(
    state: &AppState,
    guild_config: &GuildConfig,
    message: &Message,
) -> bool {
    if message.author.id == state.bot_user_id {
        return true;
    }

    let storage_channel_id = state.storage.read().channel_id;
    if message.channel_id == storage_channel_id {
        return true;
    }

    if guild_config
        .bot_log_channel_id()
        .is_some_and(|channel_id| message.channel_id == channel_id)
    {
        return true;
    }

    if guild_config.scan_policy.exempt_administrators {
        match message_member_permissions(message) {
            Some(permissions) if permissions.contains(Permissions::ADMINISTRATOR) => return true,
            Some(_) => {}
            None => {
                let administrator_roles = state.administrator_roles.load();
                match message_has_any_role(message, administrator_roles.as_ref()) {
                    Some(true) => return true,
                    Some(false) => {}
                    None => warn!(
                        event = "scan.admin_permissions_unavailable",
                        message_id = message.id.get(),
                        author_id = message.author.id.get(),
                        "administrator exemption could not be evaluated from gateway payload; scanning image"
                    ),
                }
            }
        }
    }

    false
}

fn message_member_permissions(message: &Message) -> Option<Permissions> {
    message
        .member
        .as_ref()
        .and_then(|member| member.permissions)
}

async fn enqueue_candidates(
    state: &AppState,
    tx: &mpsc::Sender<ImageCandidate>,
    candidates: Vec<ImageCandidate>,
) -> Result<()> {
    enqueue_candidates_with_key_prefix(state, tx, candidates, "").await
}

async fn enqueue_candidates_with_key_prefix(
    state: &AppState,
    tx: &mpsc::Sender<ImageCandidate>,
    candidates: Vec<ImageCandidate>,
    dedupe_prefix: &str,
) -> Result<()> {
    let policy_hash = state.detection_policy_hash();
    for candidate in candidates {
        let key = format!(
            "{}{}:{}:{}",
            dedupe_prefix,
            candidate.message_id.get(),
            policy_hash,
            candidate.url
        );
        enqueue_candidate_with_timeout(state, tx, key, candidate).await?;
    }

    Ok(())
}

async fn enqueue_candidate_with_timeout(
    state: &AppState,
    tx: &mpsc::Sender<ImageCandidate>,
    dedupe_key: String,
    mut candidate: ImageCandidate,
) -> Result<()> {
    if !state.dedupe.lock().insert_new(dedupe_key.clone()) {
        return Ok(());
    }

    let timeout_ms = state.config.queue.enqueue_timeout_ms;
    if timeout_ms == 0 {
        return match tx.try_reserve() {
            Ok(permit) => {
                candidate.enqueued_at = Some(Instant::now());
                permit.send(candidate);
                Ok(())
            }
            Err(mpsc::error::TrySendError::Full(())) => {
                state.dedupe.lock().remove(&dedupe_key);
                warn!(
                    event = "queue.full_drop",
                    guild_id = candidate.guild_id.get(),
                    channel_id = candidate.channel_id.get(),
                    message_id = candidate.message_id.get(),
                    image_url = %url_log_label(&candidate.url),
                    "image queue full; dropping candidate"
                );
                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(())) => {
                state.dedupe.lock().remove(&dedupe_key);
                Err(anyhow!("image queue closed"))
            }
        };
    }

    match tokio::time::timeout(Duration::from_millis(timeout_ms), tx.reserve()).await {
        Ok(Ok(permit)) => {
            candidate.enqueued_at = Some(Instant::now());
            permit.send(candidate);
            Ok(())
        }
        Ok(Err(_)) => {
            state.dedupe.lock().remove(&dedupe_key);
            Err(anyhow!("image queue closed"))
        }
        Err(_) => {
            state.dedupe.lock().remove(&dedupe_key);
            warn!(
                event = "queue.enqueue_timeout_drop",
                timeout_ms,
                guild_id = candidate.guild_id.get(),
                channel_id = candidate.channel_id.get(),
                message_id = candidate.message_id.get(),
                image_url = %url_log_label(&candidate.url),
                "image queue stayed full; dropping candidate"
            );
            Ok(())
        }
    }
}

async fn handle_ledger_message_delete(
    state: &BotState,
    channel_id: Id<ChannelMarker>,
    message_id: Id<MessageMarker>,
) {
    enum LedgerDelete {
        Ignore,
        Config,
        Specimen(String),
    }

    let Some(state) = state.state_for_storage_channel(channel_id) else {
        return;
    };

    let deletion = {
        let mut storage = state.storage.write();
        if storage.channel_id != channel_id {
            LedgerDelete::Ignore
        } else if storage.config_message_id == Some(message_id) {
            storage.config_message_id = None;
            LedgerDelete::Config
        } else if let Some(index) = storage
            .specimens
            .iter()
            .position(|specimen| specimen.message_id == message_id)
        {
            LedgerDelete::Specimen(storage.specimens.swap_remove(index).specimen_id)
        } else {
            LedgerDelete::Ignore
        }
    };

    match deletion {
        LedgerDelete::Ignore => {}
        LedgerDelete::Config => {
            state.guild_configured.store(false, Ordering::Release);
            warn!(
                event = "ledger.config_deleted",
                channel_id = channel_id.get(),
                message_id = message_id.get(),
                "guild config ledger message was deleted; restart or reconfigure the bot"
            );
        }
        LedgerDelete::Specimen(removed_specimen_id) => {
            let removed_from_matcher = match state
                .remove_matcher_specimen(removed_specimen_id.clone())
                .await
            {
                Ok(removed) => removed,
                Err(source) => {
                    warn!(
                        event = "ledger.specimen_matcher_remove_failed",
                        specimen_id = %removed_specimen_id,
                        ?source,
                        "failed to remove manually deleted specimen from matcher"
                    );
                    false
                }
            };
            state.specimen_hit_counts.remove(&removed_specimen_id);

            info!(
                event = "ledger.specimen_deleted",
                channel_id = channel_id.get(),
                message_id = message_id.get(),
                specimen_id = %removed_specimen_id,
                removed_from_matcher,
                decision_cache_cleared = removed_from_matcher,
                "removed manually deleted specimen from memory"
            );
            state
                .post_bot_log(format!(
                    "Specimen `{}` was removed because its ledger message `{}` was deleted.",
                    removed_specimen_id,
                    message_id.get()
                ))
                .await;
        }
    }
}

fn handle_channel_delete(state: &BotState, channel_id: Id<ChannelMarker>) {
    state.fail_guild_storage_channel(channel_id);
}

async fn handle_interaction(
    state: BotState,
    tx: mpsc::Sender<ImageCandidate>,
    interaction: Box<InteractionCreate>,
) -> Result<()> {
    let Some(guild_id) = interaction.guild_id else {
        return Ok(());
    };
    let command_name = match &interaction.data {
        Some(InteractionData::ApplicationCommand(command)) => Some(command.name.as_str()),
        _ => None,
    };
    let command_deferred = matches!(
        command_name,
        Some(
            CONFIG_COMMAND
                | DOCTOR_COMMAND
                | STATS_COMMAND
                | AUDIT_COMMAND
                | ADD_SPECIMEN_COMMAND
                | VALIDATE_MESSAGE_COMMAND
                | VERIFY_MESSAGE_COMMAND
        )
    );
    let component_deferred = match &interaction.data {
        Some(InteractionData::MessageComponent(component)) => {
            should_defer_config_component(component.custom_id.as_str())
        }
        _ => false,
    };
    let config_modal_deferred = matches!(
        &interaction.data,
        Some(InteractionData::ModalSubmit(modal)) if should_defer_modal_as_message_update(&modal.custom_id)
    );
    let ephemeral_modal_deferred = matches!(
        &interaction.data,
        Some(InteractionData::ModalSubmit(modal)) if should_defer_modal_as_ephemeral_response(&modal.custom_id)
    );
    if command_deferred || ephemeral_modal_deferred {
        defer_ephemeral_interaction(
            &state.discord,
            state.application_id,
            interaction.id,
            &interaction.token,
        )
        .await?;
    } else if component_deferred || config_modal_deferred {
        defer_update_interaction(
            &state.discord,
            state.application_id,
            interaction.id,
            &interaction.token,
        )
        .await?;
    }
    let modal_deferred = config_modal_deferred || ephemeral_modal_deferred;
    let deferred = command_deferred || component_deferred || modal_deferred;

    let state = match state.for_guild(guild_id).await {
        Ok(state) => state,
        Err(source) => {
            warn!(
                event = "guild.load_failed",
                guild_id = guild_id.get(),
                ?source,
                "could not load guild runtime for interaction"
            );
            if deferred {
                edit_interaction_response(
                    &state.discord,
                    state.application_id,
                    &interaction.token,
                    unavailable_interaction_message(),
                )
                .await?;
            } else {
                respond_unavailable_interaction(&state, &interaction).await?;
            }
            return Ok(());
        }
    };

    let Some(InteractionData::ApplicationCommand(command)) = &interaction.data else {
        return match &interaction.data {
            Some(InteractionData::MessageComponent(component))
                if component.custom_id.starts_with("config:") =>
            {
                if component_deferred {
                    admin::handle_deferred_component(&state, &interaction, component)
                        .await
                        .map(|_| ())
                } else {
                    admin::handle_component(&state, &interaction, component)
                        .await
                        .map(|_| ())
                }
            }
            Some(InteractionData::ModalSubmit(modal))
                if modal.custom_id.starts_with("config:")
                    || modal.custom_id.starts_with("specimen:") =>
            {
                if modal_deferred {
                    admin::handle_deferred_modal(&state, &interaction, modal)
                        .await
                        .map(|_| ())
                } else {
                    admin::handle_modal(&state, &interaction, modal)
                        .await
                        .map(|_| ())
                }
            }
            _ => Ok(()),
        };
    };

    if command.name == CONFIG_COMMAND
        || command.name == DOCTOR_COMMAND
        || command.name == STATS_COMMAND
        || command.name == AUDIT_COMMAND
        || command.name == IMPORT_HASHES_COMMAND
        || command.name == IMPORT_IMAGES_COMMAND
        || command.name == EXPORT_HASHES_COMMAND
    {
        let response = match admin::admin_command_response(&state, &interaction, command).await {
            Ok(Some(response)) => response,
            Ok(None) => return Ok(()),
            Err(source) => {
                warn!(
                    event = "admin_command.failed",
                    command = %command.name,
                    ?source,
                    "admin command failed"
                );
                InteractionResponse {
                    kind: InteractionResponseType::ChannelMessageWithSource,
                    data: Some(InteractionResponseData {
                        allowed_mentions: Some(AllowedMentions::default()),
                        content: Some(format!("`{}` failed: {source:#}", command.name)),
                        flags: Some(MessageFlags::EPHEMERAL),
                        ..InteractionResponseData::default()
                    }),
                }
            }
        };
        let Some(data) = response.data.as_ref() else {
            return Ok(());
        };
        if deferred {
            edit_interaction_response_data(
                &state.discord,
                state.application_id,
                &interaction.token,
                data,
            )
            .await?;
        } else {
            respond_interaction(
                &state.discord,
                state.application_id,
                interaction.id,
                &interaction.token,
                &response,
            )
            .await?;
        }
        return Ok(());
    }

    let result = match command.name.as_str() {
        ADD_SPECIMEN_COMMAND => handle_add_specimen_command(&state, &interaction).await,
        VALIDATE_MESSAGE_COMMAND => {
            handle_validate_message_command(&state, &tx, &interaction, false).await
        }
        VERIFY_MESSAGE_COMMAND => {
            handle_validate_message_command(&state, &tx, &interaction, true).await
        }
        _ => return Ok(()),
    };
    let response = match result {
        Ok(summary) => summary,
        Err(source) => {
            warn!(
                event = "message_command.failed",
                command = %command.name,
                ?source,
                "message command failed"
            );
            format!("`{}` failed: {source:#}", command.name)
        }
    };

    edit_interaction_response(
        &state.discord,
        state.application_id,
        &interaction.token,
        &response,
    )
    .await?;

    Ok(())
}

fn should_defer_config_component(custom_id: &str) -> bool {
    matches!(
        custom_id,
        "config:actions:v1"
            | "config:roles:v1"
            | "config:doctor:v1"
            | "config:home:v1"
            | "config:toggle-enabled:v1"
            | "config:bot-log-channel:v1"
            | "config:moderator-roles:v1"
            | "config:scan-exempt-roles:v1"
            | "config:verified-role:v1"
            | "config:confirmed-actions:v1"
            | "config:suspicious-actions:v1"
            | "config:confirmed-timeout:v1"
            | "config:suspicious-timeout:v1"
    )
}

fn should_defer_modal_as_message_update(custom_id: &str) -> bool {
    custom_id.starts_with("config:")
}

fn should_defer_modal_as_ephemeral_response(custom_id: &str) -> bool {
    custom_id.starts_with("specimen:")
}

fn unavailable_interaction_message() -> &'static str {
    "Sightline is unavailable in this guild. Create exactly one private text channel named `sightline-db`, then grant the bot View Channel and Send Messages permissions in both `sightline-db` and the configured bot log channel before running `/config`."
}

async fn respond_unavailable_interaction(
    state: &BotState,
    interaction: &InteractionCreate,
) -> Result<()> {
    let response = InteractionResponse {
        kind: InteractionResponseType::ChannelMessageWithSource,
        data: Some(InteractionResponseData {
            allowed_mentions: Some(AllowedMentions::default()),
            content: Some(unavailable_interaction_message().to_owned()),
            flags: Some(MessageFlags::EPHEMERAL),
            ..InteractionResponseData::default()
        }),
    };
    respond_interaction(
        &state.discord,
        state.application_id,
        interaction.id,
        &interaction.token,
        &response,
    )
    .await
}

async fn handle_add_specimen_command(
    state: &AppState,
    interaction: &InteractionCreate,
) -> Result<String> {
    if !state.guild_accepts_specimen_writes() {
        return Err(anyhow!(
            "guild is not configured, permission checks failed, or sightline-db is unavailable"
        ));
    }

    if !admin::can_moderate(state, interaction) {
        return Err(anyhow!("user lacks specimen permissions"));
    }

    let Some(InteractionData::ApplicationCommand(command)) = &interaction.data else {
        return Err(anyhow!("interaction is not an application command"));
    };
    let target_id = command
        .target_id
        .ok_or_else(|| anyhow!("message command target missing"))?;
    let target_message_id = Id::<MessageMarker>::new(target_id.get());
    let target_message = command
        .resolved
        .as_ref()
        .and_then(|resolved| resolved.messages.get(&target_message_id))
        .ok_or_else(|| anyhow!("target message was not present in resolved data"))?;
    if target_message
        .guild_id
        .is_some_and(|guild_id| guild_id != state.guild_id())
    {
        return Err(anyhow!("target message is outside the configured guild"));
    }
    if target_message.author.id == state.bot_user_id {
        return Ok("Sightline bot messages cannot be added as specimens.".to_owned());
    }

    let guild_config = state.active_config();
    let storage_channel_id = state.storage.read().channel_id;
    if target_message.channel_id == storage_channel_id
        || guild_config
            .bot_log_channel_id()
            .is_some_and(|channel_id| target_message.channel_id == channel_id)
    {
        return Ok(
            "Messages in Sightline database or bot log channels cannot be added as specimens."
                .to_owned(),
        );
    }

    let candidates = extract_candidates_from_message(
        state.guild_id(),
        target_message,
        MAX_MANUAL_SPECIMEN_IMAGES_PER_INTERACTION,
        &[],
        guild_config.scan_policy.max_file_bytes,
    );

    if candidates.is_empty() {
        return Ok("No usable images were found on that message.".to_owned());
    }

    let added_by_id = interaction
        .author_id()
        .ok_or_else(|| anyhow!("interaction author missing"))?;
    let candidate_count = candidates.len();
    let summary = import_image_candidates(
        state,
        candidates,
        added_by_id,
        &guild_config,
        "message_action",
    )
    .await;

    state
        .post_bot_log(
            BotLogEvent::new(
                "Add specimen action",
                format!(
                    "<@{}> used the message action to add image specimens from a message by <@{}>.",
                    added_by_id.get(),
                    target_message.author.id.get()
                ),
            )
            .color(if summary.failed > 0 {
                BotLogColor::Warning
            } else {
                BotLogColor::Info
            })
            .field(
                "Moderator",
                format!("<@{}> (`{}`)", added_by_id.get(), added_by_id.get()),
                true,
            )
            .field(
                "Target user",
                format!(
                    "<@{}> (`{}`, username `{}`)",
                    target_message.author.id.get(),
                    target_message.author.id.get(),
                    target_message.author.name.replace('`', "'")
                ),
                true,
            )
            .field(
                "Source message",
                message_jump_link(state.guild_id(), target_message.channel_id, target_message.id),
                false,
            )
            .field(
                "Result",
                format!(
                    "candidates `{candidate_count}`, added `{}`, exact duplicates `{}`, failed `{}`",
                    summary.added,
                    summary.exact_duplicates,
                    summary.failed
                ),
                true,
            ),
        )
        .await;

    Ok(format!(
        "Added {} specimen record(s). Exact duplicates rejected: {}. Failed: {}.",
        summary.added, summary.exact_duplicates, summary.failed
    ))
}

async fn handle_validate_message_command(
    state: &AppState,
    tx: &mpsc::Sender<ImageCandidate>,
    interaction: &InteractionCreate,
    verify_only: bool,
) -> Result<String> {
    if !state.guild_active() {
        return Err(anyhow!("guild is not configured, enabled, and writable"));
    }

    if !admin::can_moderate(state, interaction) {
        return Err(anyhow!("user lacks validation permissions"));
    }

    let Some(InteractionData::ApplicationCommand(command)) = &interaction.data else {
        return Err(anyhow!("interaction is not an application command"));
    };
    let target_message = resolved_target_message(command)?;
    if target_message
        .guild_id
        .is_some_and(|guild_id| guild_id != state.guild_id())
    {
        return Err(anyhow!("target message is outside the configured guild"));
    }
    if target_message.author.id == state.bot_user_id {
        return Ok("Sightline bot messages are not manually validated.".to_owned());
    }

    let guild_config = state.active_config();
    let storage_channel_id = state.storage.read().channel_id;
    if target_message.channel_id == storage_channel_id
        || guild_config
            .bot_log_channel_id()
            .is_some_and(|channel_id| target_message.channel_id == channel_id)
    {
        return Ok(
            "Messages in Sightline database or bot log channels are not manually validated."
                .to_owned(),
        );
    }

    let mut candidates = extract_candidates_from_message(
        state.guild_id(),
        target_message,
        usize::MAX,
        &guild_config.scan_policy.allowed_extensions,
        guild_config.scan_policy.max_file_bytes,
    );
    if verify_only {
        for candidate in &mut candidates {
            candidate.verify_only = true;
        }
    }
    if candidates.is_empty() {
        return Ok("No scan-policy-eligible images were found on that message.".to_owned());
    }

    let requested_by_id = interaction
        .author_id()
        .ok_or_else(|| anyhow!("interaction author missing"))?;
    let candidate_count = candidates.len();
    enqueue_candidates_with_key_prefix(
        state,
        tx,
        candidates,
        &format!("manual:{}:", interaction.id.get()),
    )
    .await?;

    state
        .post_bot_log(
            BotLogEvent::new(
                if verify_only {
                    "Manual verification requested"
                } else {
                    "Manual validation requested"
                },
                format!(
                    "<@{}> manually sent `{}` image(s) from a message by <@{}> through Sightline {}.",
                    requested_by_id.get(),
                    candidate_count,
                    target_message.author.id.get(),
                    if verify_only { "verification" } else { "validation" }
                ),
            )
            .color(BotLogColor::Info)
            .field(
                "Moderator",
                format!("<@{}> (`{}`)", requested_by_id.get(), requested_by_id.get()),
                true,
            )
            .field(
                "Target user",
                format!(
                    "<@{}> (`{}`, username `{}`)",
                    target_message.author.id.get(),
                    target_message.author.id.get(),
                    target_message.author.name.replace('`', "'")
                ),
                true,
            )
            .field(
                "Source message",
                message_jump_link(state.guild_id(), target_message.channel_id, target_message.id),
                false,
            )
            .field(
                "Bypass",
                "Target-author administrator and scan-exempt role filters were bypassed for this manual validation.",
                false,
            )
            .field(
                "Actions",
                if verify_only {
                    "No moderation or specimen-add actions will run."
                } else {
                    "Configured match actions will run."
                },
                false,
            ),
        )
        .await;

    if verify_only {
        Ok(format!(
            "Queued {candidate_count} image(s) for verification. No moderation or specimen-add actions will run."
        ))
    } else {
        Ok(format!(
            "Queued {candidate_count} image(s) for validation. Any configured match actions will run as if the message had just been posted."
        ))
    }
}

fn resolved_target_message(
    command: &twilight_model::application::interaction::application_command::CommandData,
) -> Result<&Message> {
    let target_id = command
        .target_id
        .ok_or_else(|| anyhow!("message command target missing"))?;
    let target_message_id = Id::<MessageMarker>::new(target_id.get());
    command
        .resolved
        .as_ref()
        .and_then(|resolved| resolved.messages.get(&target_message_id))
        .ok_or_else(|| anyhow!("target message was not present in resolved data"))
}

#[cfg(test)]
mod tests {
    use super::{
        DownloadHostActivity, ImageMetricsCommand, performance_report_loop,
        should_defer_modal_as_ephemeral_response, should_defer_modal_as_message_update,
        warmer_initial_delay,
    };
    use crate::image::types::{ImageMetricEvent, TextGateResolutionMetric};
    use std::time::Duration;
    use tokio::sync::{mpsc, oneshot};
    use twilight_model::id::{Id, marker::GuildMarker};

    #[test]
    fn config_modal_submits_defer_as_message_updates() {
        assert!(should_defer_modal_as_message_update(
            "config:advanced-modal:v2:g1:u2:pabc:c123"
        ));
        assert!(should_defer_modal_as_message_update(
            "config:log-message-modal:v2:g1:u2:pabc:c123"
        ));
        assert!(!should_defer_modal_as_ephemeral_response(
            "config:advanced-modal:v2:g1:u2:pabc:c123"
        ));
    }

    #[test]
    fn specimen_modal_submits_defer_as_ephemeral_responses() {
        assert!(should_defer_modal_as_ephemeral_response(
            "specimen:import-images-modal:v1"
        ));
        assert!(should_defer_modal_as_ephemeral_response(
            "specimen:import-hashes-modal:v1"
        ));
        assert!(!should_defer_modal_as_message_update(
            "specimen:import-images-modal:v1"
        ));
    }

    #[test]
    fn warmer_initial_delay_is_bounded_by_jitter_window() {
        let delay = warmer_initial_delay("abcdef123456", Duration::from_secs(270));

        assert!(delay < Duration::from_secs(30));
    }

    #[test]
    fn warmer_activity_tracks_only_cdn_host() {
        let activity = DownloadHostActivity::default();

        activity.touch_url("https://media.discordapp.net/attachments/1/2/image.png");
        assert!(activity.cdn_elapsed() > Duration::from_secs(1_000_000));

        activity.touch_url("https://cdn.discordapp.com/attachments/1/2/image.png");
        assert!(activity.cdn_elapsed() < Duration::from_secs(2));
    }

    #[tokio::test]
    async fn removing_guild_evicts_metrics() {
        let guild_id = Id::<GuildMarker>::new(1);
        let (tx, rx) = mpsc::channel(8);
        let task = tokio::spawn(performance_report_loop(rx));
        tx.send(ImageMetricsCommand::Record(ImageMetricEvent::OcrResolved {
            guild_id,
            resolution: TextGateResolutionMetric::Good,
        }))
        .await
        .expect("metrics receiver is running");

        let (respond_to, response) = oneshot::channel();
        tx.send(ImageMetricsCommand::Snapshot {
            guild_id,
            respond_to,
        })
        .await
        .expect("metrics receiver is running");
        assert_eq!(
            response
                .await
                .expect("snapshot response")
                .total
                .ocr_resolved_good,
            1
        );

        tx.send(ImageMetricsCommand::RemoveGuild { guild_id })
            .await
            .expect("metrics receiver is running");
        let (respond_to, response) = oneshot::channel();
        tx.send(ImageMetricsCommand::Snapshot {
            guild_id,
            respond_to,
        })
        .await
        .expect("metrics receiver is running");
        assert_eq!(
            response
                .await
                .expect("snapshot response")
                .total
                .ocr_resolved_good,
            0
        );

        drop(tx);
        task.await.expect("metrics task exits cleanly");
    }
}
