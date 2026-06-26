#![allow(clippy::too_many_lines)]

use crate::{
    bot::ledger::{
        MAX_DISCORD_STORAGE_CONTENT, MAX_SPECIMEN_ATTACHMENT_BYTES,
        MAX_SPECIMEN_RECORD_ATTACHMENT_BYTES, SpecimenImageAttachment, SpecimenManifest,
        SpecimenRecord, parse_and_verify_specimen_manifest, parse_and_verify_specimen_record,
        signed_specimen_manifest_to_discord, specimen_record_attachment,
    },
    configuration::{
        app::MatchConfig,
        guild::{
            DetectionActions, GuildConfig, GuildConfigManifest, GuildConfigRecord,
            MAX_CONFIG_RECORD_ATTACHMENT_BYTES, config_record_attachment,
            parse_and_verify_config_manifest, parse_and_verify_config_record,
            signed_config_manifest_to_discord,
        },
        storage_codec::{CONFIG_PREFIX, SPECIMEN_PREFIX},
    },
    image::{
        pipeline::{HashMode, hash_image_bytes, is_discord_host},
        types::{ImageCandidate, MatchOutcome},
    },
};
use anyhow::{Context, Result, anyhow, bail};
use bytes::{Bytes, BytesMut};
use chrono::{SecondsFormat, Utc};
use futures_util::{StreamExt, stream::FuturesUnordered};
use reqwest::Client as ReqwestClient;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tracing::{error, info, warn};
use twilight_http::{Client, request::AuditLogReason};
use twilight_model::{
    application::{
        command::{Command, CommandType},
        interaction::InteractionContextType,
    },
    channel::{
        Attachment, ChannelType,
        message::{AllowedMentions, Embed, MentionType, MessageFlags, embed::EmbedField},
        permission_overwrite::{PermissionOverwrite, PermissionOverwriteType},
    },
    guild::Permissions,
    http::{
        attachment::Attachment as HttpAttachment,
        interaction::{InteractionResponse, InteractionResponseData, InteractionResponseType},
    },
    id::{
        Id,
        marker::{
            ApplicationMarker, AttachmentMarker, ChannelMarker, CommandVersionMarker, GuildMarker,
            InteractionMarker, MessageMarker, RoleMarker, UserMarker,
        },
    },
    oauth::ApplicationIntegrationType,
    util::Timestamp,
};
use twilight_util::permission_calculator::PermissionCalculator;

pub const ADD_SPECIMEN_COMMAND: &str = "Add scam image specimen";
pub const AUDIT_COMMAND: &str = "audit";
pub const CONFIG_COMMAND: &str = "config";
pub const DOCTOR_COMMAND: &str = "doctor";
pub const EXPORT_HASHES_COMMAND: &str = "export-hashes";
pub const IMPORT_HASHES_COMMAND: &str = "import-hashes";
pub const IMPORT_IMAGES_COMMAND: &str = "import-images";
pub const VALIDATE_MESSAGE_COMMAND: &str = "Validate scam images";
pub const VERIFY_MESSAGE_COMMAND: &str = "Verify scam images";
pub const STATS_COMMAND: &str = "stats";
pub const SIGHTLINE_DB_CHANNEL_NAME: &str = "sightline-db";

#[derive(Debug, Clone, Copy)]
pub(crate) enum BotLogColor {
    Info,
    Success,
    Warning,
    Danger,
}

impl BotLogColor {
    fn value(self) -> u32 {
        match self {
            Self::Info => 5_793_266,
            Self::Success => 5_765_955,
            Self::Warning => 16_776_960,
            Self::Danger => 15_548_984,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BotLogField {
    pub(crate) name: String,
    pub(crate) value: String,
    pub(crate) inline: bool,
}

impl BotLogField {
    pub(crate) fn new(name: impl Into<String>, value: impl Into<String>, inline: bool) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
            inline,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BotLogEvent {
    pub(crate) title: String,
    pub(crate) description: String,
    pub(crate) color: BotLogColor,
    pub(crate) copy_kind: BotLogCopyKind,
    pub(crate) fields: Vec<BotLogField>,
    pub(crate) image_url: Option<String>,
    pub(crate) attachments: Vec<BotLogAttachment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BotLogCopyKind {
    General,
    ConfirmedDetection,
    SuspiciousDetection,
    BenignDetection,
}

#[derive(Debug, Clone)]
pub(crate) struct BotLogAttachment {
    pub(crate) filename: String,
    pub(crate) bytes: Vec<u8>,
}

impl BotLogEvent {
    pub(crate) fn new(title: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            description: description.into(),
            color: BotLogColor::Info,
            copy_kind: BotLogCopyKind::General,
            fields: Vec::new(),
            image_url: None,
            attachments: Vec::new(),
        }
    }

    pub(crate) fn color(mut self, color: BotLogColor) -> Self {
        self.color = color;
        self
    }

    pub(crate) fn field(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
        inline: bool,
    ) -> Self {
        self.fields.push(BotLogField::new(name, value, inline));
        self
    }

    pub(crate) fn image_url(mut self, image_url: impl Into<String>) -> Self {
        self.image_url = Some(image_url.into());
        self
    }

    pub(crate) fn confirmed_detection_copy(mut self) -> Self {
        self.copy_kind = BotLogCopyKind::ConfirmedDetection;
        self
    }

    pub(crate) fn suspicious_detection_copy(mut self) -> Self {
        self.copy_kind = BotLogCopyKind::SuspiciousDetection;
        self
    }

    pub(crate) fn benign_detection_copy(mut self) -> Self {
        self.copy_kind = BotLogCopyKind::BenignDetection;
        self
    }

    pub(crate) fn text_attachment(
        mut self,
        filename: impl Into<String>,
        contents: impl Into<String>,
    ) -> Self {
        let filename = filename.into();
        let bytes = contents.into().into_bytes();
        if bytes.len() <= 2 * 1024 * 1024 {
            self.attachments.push(BotLogAttachment { filename, bytes });
        } else {
            self.fields.push(BotLogField::new(
                "Attachment skipped",
                "Raw text details exceeded 2 MiB.",
                false,
            ));
        }
        self
    }

    pub(crate) fn json_attachment<T>(mut self, filename: impl Into<String>, value: &T) -> Self
    where
        T: serde::Serialize,
    {
        match serde_json::to_vec_pretty(value) {
            Ok(bytes) if bytes.len() <= 2 * 1024 * 1024 => {
                self.attachments.push(BotLogAttachment {
                    filename: filename.into(),
                    bytes,
                });
            }
            Ok(_) => {
                self.fields.push(BotLogField::new(
                    "Attachment skipped",
                    "JSON evidence exceeded 2 MiB.",
                    false,
                ));
            }
            Err(error) => {
                self.fields.push(BotLogField::new(
                    "Attachment skipped",
                    format!("JSON evidence could not be serialized: {error}"),
                    false,
                ));
            }
        }
        self
    }
}

impl From<String> for BotLogEvent {
    fn from(details: String) -> Self {
        let description = details.lines().next().unwrap_or("Sightline log").to_owned();
        BotLogEvent::new(truncate_chars(&description, 256), details)
    }
}

impl From<&str> for BotLogEvent {
    fn from(details: &str) -> Self {
        details.to_owned().into()
    }
}

pub(crate) struct RenderedBotLog {
    pub(crate) content: Option<String>,
    pub(crate) embeds: Vec<Embed>,
    pub(crate) attachments: Vec<HttpAttachment>,
}

#[derive(Debug, Clone)]
pub(crate) struct BotPermissionReport {
    pub(crate) ok: bool,
    pub(crate) checks: Vec<BotPermissionCheck>,
}

#[derive(Debug, Clone)]
pub(crate) struct BotPermissionCheck {
    pub(crate) label: String,
    pub(crate) ok: bool,
    pub(crate) blocks_runtime: bool,
}

impl BotPermissionReport {
    pub(crate) fn missing_summary(&self) -> String {
        let missing = self
            .checks
            .iter()
            .filter(|check| check.blocks_runtime && !check.ok)
            .map(|check| check.label.as_str())
            .collect::<Vec<_>>();
        if missing.is_empty() {
            "none".to_owned()
        } else {
            missing.join(", ")
        }
    }
}

#[derive(Debug, Clone)]
pub struct LedgerLoad {
    pub specimens: Vec<SpecimenRecord>,
    pub guild_config: Option<GuildConfig>,
    pub storage: DiscordStorageState,
}

#[derive(Debug, Clone)]
pub struct DiscordStorageState {
    pub channel_id: Id<ChannelMarker>,
    pub config_message_id: Option<Id<MessageMarker>>,
    pub specimens: Vec<StoredSpecimen>,
}

#[derive(Debug, Clone)]
pub struct StoredSpecimen {
    pub message_id: Id<MessageMarker>,
    pub specimen_id: String,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RoleRemovalOutcome {
    pub attempted: usize,
    pub removed: usize,
    pub failed: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MemberActionResults {
    pub(crate) timed_out: bool,
    pub(crate) banned: bool,
    pub(crate) kicked: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DetectionActionResults {
    pub(crate) deleted: bool,
    pub(crate) role_removal: RoleRemovalOutcome,
    pub(crate) member: MemberActionResults,
}

pub async fn register_commands(
    client: &Arc<Client>,
    application_id: Id<ApplicationMarker>,
) -> Result<()> {
    let commands = sightline_commands();
    let command_names = commands
        .iter()
        .map(|command| command.name.as_str())
        .collect::<Vec<_>>();
    client
        .interaction(application_id)
        .set_global_commands(&commands)
        .await
        .context("registering global commands")?;

    info!(
        event = "commands.registered",
        ?command_names,
        "global commands registered"
    );
    Ok(())
}

#[allow(deprecated)]
fn sightline_commands() -> Vec<Command> {
    vec![
        Command {
            application_id: None,
            contexts: Some(vec![InteractionContextType::Guild]),
            default_member_permissions: Some(Permissions::MANAGE_MESSAGES),
            dm_permission: None,
            description: String::new(),
            description_localizations: None,
            guild_id: None,
            id: None,
            integration_types: Some(vec![ApplicationIntegrationType::GuildInstall]),
            kind: CommandType::Message,
            name: ADD_SPECIMEN_COMMAND.to_owned(),
            name_localizations: None,
            nsfw: None,
            options: Vec::new(),
            version: Id::<CommandVersionMarker>::new(1),
        },
        Command {
            application_id: None,
            contexts: Some(vec![InteractionContextType::Guild]),
            default_member_permissions: Some(Permissions::MANAGE_MESSAGES),
            dm_permission: None,
            description: String::new(),
            description_localizations: None,
            guild_id: None,
            id: None,
            integration_types: Some(vec![ApplicationIntegrationType::GuildInstall]),
            kind: CommandType::Message,
            name: VALIDATE_MESSAGE_COMMAND.to_owned(),
            name_localizations: None,
            nsfw: None,
            options: Vec::new(),
            version: Id::<CommandVersionMarker>::new(1),
        },
        Command {
            application_id: None,
            contexts: Some(vec![InteractionContextType::Guild]),
            default_member_permissions: Some(Permissions::MANAGE_MESSAGES),
            dm_permission: None,
            description: String::new(),
            description_localizations: None,
            guild_id: None,
            id: None,
            integration_types: Some(vec![ApplicationIntegrationType::GuildInstall]),
            kind: CommandType::Message,
            name: VERIFY_MESSAGE_COMMAND.to_owned(),
            name_localizations: None,
            nsfw: None,
            options: Vec::new(),
            version: Id::<CommandVersionMarker>::new(1),
        },
        Command {
            application_id: None,
            contexts: Some(vec![InteractionContextType::Guild]),
            default_member_permissions: Some(Permissions::MANAGE_GUILD),
            dm_permission: None,
            description: "Open the Sightline configuration panel".to_owned(),
            description_localizations: None,
            guild_id: None,
            id: None,
            integration_types: Some(vec![ApplicationIntegrationType::GuildInstall]),
            kind: CommandType::ChatInput,
            name: CONFIG_COMMAND.to_owned(),
            name_localizations: None,
            nsfw: None,
            options: Vec::new(),
            version: Id::<CommandVersionMarker>::new(1),
        },
        Command {
            application_id: None,
            contexts: Some(vec![InteractionContextType::Guild]),
            default_member_permissions: Some(Permissions::MANAGE_MESSAGES),
            dm_permission: None,
            description: "Import specimen fingerprints from JSON or JSONL".to_owned(),
            description_localizations: None,
            guild_id: None,
            id: None,
            integration_types: Some(vec![ApplicationIntegrationType::GuildInstall]),
            kind: CommandType::ChatInput,
            name: IMPORT_HASHES_COMMAND.to_owned(),
            name_localizations: None,
            nsfw: None,
            options: Vec::new(),
            version: Id::<CommandVersionMarker>::new(1),
        },
        Command {
            application_id: None,
            contexts: Some(vec![InteractionContextType::Guild]),
            default_member_permissions: Some(Permissions::MANAGE_MESSAGES),
            dm_permission: None,
            description: "Upload specimen images for Sightline to process and store".to_owned(),
            description_localizations: None,
            guild_id: None,
            id: None,
            integration_types: Some(vec![ApplicationIntegrationType::GuildInstall]),
            kind: CommandType::ChatInput,
            name: IMPORT_IMAGES_COMMAND.to_owned(),
            name_localizations: None,
            nsfw: None,
            options: Vec::new(),
            version: Id::<CommandVersionMarker>::new(1),
        },
        Command {
            application_id: None,
            contexts: Some(vec![InteractionContextType::Guild]),
            default_member_permissions: Some(Permissions::MANAGE_MESSAGES),
            dm_permission: None,
            description: "Export current Sightline specimen fingerprints as JSONL".to_owned(),
            description_localizations: None,
            guild_id: None,
            id: None,
            integration_types: Some(vec![ApplicationIntegrationType::GuildInstall]),
            kind: CommandType::ChatInput,
            name: EXPORT_HASHES_COMMAND.to_owned(),
            name_localizations: None,
            nsfw: None,
            options: Vec::new(),
            version: Id::<CommandVersionMarker>::new(1),
        },
        Command {
            application_id: None,
            contexts: Some(vec![InteractionContextType::Guild]),
            default_member_permissions: Some(Permissions::MANAGE_GUILD),
            dm_permission: None,
            description: "Run Sightline setup diagnostics".to_owned(),
            description_localizations: None,
            guild_id: None,
            id: None,
            integration_types: Some(vec![ApplicationIntegrationType::GuildInstall]),
            kind: CommandType::ChatInput,
            name: DOCTOR_COMMAND.to_owned(),
            name_localizations: None,
            nsfw: None,
            options: Vec::new(),
            version: Id::<CommandVersionMarker>::new(1),
        },
        Command {
            application_id: None,
            contexts: Some(vec![InteractionContextType::Guild]),
            default_member_permissions: Some(Permissions::MANAGE_GUILD),
            dm_permission: None,
            description: "Show Sightline image matching statistics for this guild".to_owned(),
            description_localizations: None,
            guild_id: None,
            id: None,
            integration_types: Some(vec![ApplicationIntegrationType::GuildInstall]),
            kind: CommandType::ChatInput,
            name: STATS_COMMAND.to_owned(),
            name_localizations: None,
            nsfw: None,
            options: Vec::new(),
            version: Id::<CommandVersionMarker>::new(1),
        },
        Command {
            application_id: None,
            contexts: Some(vec![InteractionContextType::Guild]),
            default_member_permissions: Some(Permissions::MANAGE_GUILD),
            dm_permission: None,
            description: "Audit Sightline specimen quality for this guild".to_owned(),
            description_localizations: None,
            guild_id: None,
            id: None,
            integration_types: Some(vec![ApplicationIntegrationType::GuildInstall]),
            kind: CommandType::ChatInput,
            name: AUDIT_COMMAND.to_owned(),
            name_localizations: None,
            nsfw: None,
            options: Vec::new(),
            version: Id::<CommandVersionMarker>::new(1),
        },
    ]
}

pub async fn find_database_channel(
    client: &Arc<Client>,
    guild_id: Id<GuildMarker>,
) -> Result<Id<ChannelMarker>> {
    let channels = client
        .guild_channels(guild_id)
        .await
        .context("listing guild channels")?
        .models()
        .await
        .context("decoding guild channel list")?;

    let mut matches = Vec::new();
    for channel in channels {
        if channel.name.as_deref() == Some(SIGHTLINE_DB_CHANNEL_NAME)
            && matches!(
                channel.kind,
                ChannelType::GuildText | ChannelType::GuildAnnouncement
            )
        {
            ensure_database_channel_private(guild_id, channel.permission_overwrites.as_ref())?;
            matches.push(channel.id);
        }
    }

    match matches.as_slice() {
        [channel_id] => Ok(*channel_id),
        [] => bail!(
            "guild {} must contain exactly one text channel named `{}`",
            guild_id.get(),
            SIGHTLINE_DB_CHANNEL_NAME
        ),
        _ => bail!(
            "guild {} has multiple text channels named `{}`; keep exactly one",
            guild_id.get(),
            SIGHTLINE_DB_CHANNEL_NAME
        ),
    }
}

fn ensure_database_channel_private(
    guild_id: Id<GuildMarker>,
    overwrites: Option<&Vec<PermissionOverwrite>>,
) -> Result<()> {
    if database_channel_private(guild_id, overwrites) {
        return Ok(());
    }
    bail!(
        "`{SIGHTLINE_DB_CHANNEL_NAME}` must deny @everyone both View Channel and Send Messages before Sightline will use it"
    )
}

fn database_channel_private(
    guild_id: Id<GuildMarker>,
    overwrites: Option<&Vec<PermissionOverwrite>>,
) -> bool {
    let denied_everyone = overwrites
        .map_or(&[] as &[PermissionOverwrite], Vec::as_slice)
        .iter()
        .find(|overwrite| {
            overwrite.kind == PermissionOverwriteType::Role && overwrite.id.get() == guild_id.get()
        })
        .map_or(Permissions::empty(), |overwrite| overwrite.deny);
    let required_denies = Permissions::VIEW_CHANNEL | Permissions::SEND_MESSAGES;
    denied_everyone.contains(required_denies)
}

pub(crate) async fn check_bot_permissions(
    client: &Arc<Client>,
    guild_id: Id<GuildMarker>,
    bot_user_id: Id<UserMarker>,
    guild_config: &GuildConfig,
) -> Result<BotPermissionReport> {
    let member = client
        .guild_member(guild_id, bot_user_id)
        .await
        .context("fetching bot guild member")?
        .model()
        .await
        .context("decoding bot guild member")?;
    let roles = client
        .roles(guild_id)
        .await
        .context("fetching guild roles")?
        .models()
        .await
        .context("decoding guild roles")?;
    let channels = client
        .guild_channels(guild_id)
        .await
        .context("fetching guild channels")?
        .models()
        .await
        .context("decoding guild channels")?;

    let role_permissions = roles
        .iter()
        .map(|role| (role.id, role.permissions))
        .collect::<HashMap<_, _>>();
    let everyone_role_id = Id::<RoleMarker>::new(guild_id.get());
    let everyone_permissions = role_permissions
        .get(&everyone_role_id)
        .copied()
        .unwrap_or_else(Permissions::empty);
    let member_roles = member
        .roles
        .iter()
        .filter(|role_id| **role_id != everyone_role_id)
        .filter_map(|role_id| {
            role_permissions
                .get(role_id)
                .copied()
                .map(|permissions| (*role_id, permissions))
        })
        .collect::<Vec<_>>();
    let calculator =
        PermissionCalculator::new(guild_id, bot_user_id, everyone_permissions, &member_roles);
    let root_permissions = calculator.root();

    let ledger_channel_id = guild_config.ledger_channel_id()?;
    let ledger_channel = channels
        .iter()
        .find(|channel| channel.id == ledger_channel_id);
    let bot_log_channel_id = guild_config.bot_log_channel_id();
    let bot_log_channel = bot_log_channel_id
        .and_then(|channel_id| channels.iter().find(|channel| channel.id == channel_id));

    let root_required = required_root_permissions();
    let mut checks = Vec::new();
    push_permission_checks("Root", root_permissions, root_required, true, &mut checks);
    push_permission_checks(
        "Action",
        root_permissions,
        action_permissions(guild_config),
        false,
        &mut checks,
    );

    match ledger_channel {
        Some(channel) => {
            let permissions = channel_permissions(
                &calculator,
                channel.kind,
                channel.permission_overwrites.as_ref(),
            );
            push_permission_checks(
                "Database/config channel",
                permissions,
                database_channel_permissions(),
                true,
                &mut checks,
            );
            checks.push(BotPermissionCheck {
                label: "Database/config channel private from @everyone".to_owned(),
                ok: database_channel_private(guild_id, channel.permission_overwrites.as_ref()),
                blocks_runtime: true,
            });
        }
        None => checks.push(BotPermissionCheck {
            label: "Database/config channel exists".to_owned(),
            ok: false,
            blocks_runtime: true,
        }),
    }

    match bot_log_channel {
        Some(channel) => {
            let permissions = channel_permissions(
                &calculator,
                channel.kind,
                channel.permission_overwrites.as_ref(),
            );
            push_permission_checks(
                "Bot log channel",
                permissions,
                bot_log_channel_permissions(),
                true,
                &mut checks,
            );
        }
        None => checks.push(BotPermissionCheck {
            label: "Bot log channel configured and visible".to_owned(),
            ok: false,
            blocks_runtime: true,
        }),
    }

    Ok(BotPermissionReport {
        ok: checks
            .iter()
            .filter(|check| check.blocks_runtime)
            .all(|check| check.ok),
        checks,
    })
}

fn channel_permissions(
    calculator: &PermissionCalculator<'_>,
    kind: ChannelType,
    overwrites: Option<&Vec<PermissionOverwrite>>,
) -> Permissions {
    calculator
        .clone()
        .in_channel(kind, overwrites.map_or(&[], Vec::as_slice))
}

fn required_root_permissions() -> Permissions {
    Permissions::empty()
}

fn action_permissions(guild_config: &GuildConfig) -> Permissions {
    let mut permissions = Permissions::empty();

    if guild_config.enabled {
        let confirmed = &guild_config.detection_policy.confirmed.actions;
        let suspicious = &guild_config.detection_policy.suspicious.actions;

        if confirmed.delete_message || suspicious.delete_message {
            permissions |= Permissions::MANAGE_MESSAGES;
        }
        if confirmed.remove_user_roles || suspicious.remove_user_roles {
            permissions |= Permissions::MANAGE_ROLES;
        }
        if confirmed.ban_user || suspicious.ban_user {
            permissions |= Permissions::BAN_MEMBERS;
        }
        if confirmed.kick_user || suspicious.kick_user {
            permissions |= Permissions::KICK_MEMBERS;
        }
        if confirmed.timeout_user || suspicious.timeout_user {
            permissions |= Permissions::MODERATE_MEMBERS;
        }
    }

    permissions
}

fn database_channel_permissions() -> Permissions {
    Permissions::VIEW_CHANNEL
        | Permissions::SEND_MESSAGES
        | Permissions::READ_MESSAGE_HISTORY
        | Permissions::ATTACH_FILES
}

fn bot_log_channel_permissions() -> Permissions {
    Permissions::VIEW_CHANNEL
        | Permissions::SEND_MESSAGES
        | Permissions::READ_MESSAGE_HISTORY
        | Permissions::EMBED_LINKS
        | Permissions::ATTACH_FILES
}

fn push_permission_checks(
    scope: &str,
    actual: Permissions,
    required: Permissions,
    blocks_runtime: bool,
    checks: &mut Vec<BotPermissionCheck>,
) {
    for (permission, name) in permission_labels(required) {
        checks.push(BotPermissionCheck {
            label: format!("{scope}: {name}"),
            ok: actual.contains(permission),
            blocks_runtime,
        });
    }
}

fn permission_labels(required: Permissions) -> Vec<(Permissions, &'static str)> {
    [
        (Permissions::VIEW_CHANNEL, "View Channel"),
        (Permissions::SEND_MESSAGES, "Send Messages"),
        (Permissions::READ_MESSAGE_HISTORY, "Read Message History"),
        (Permissions::EMBED_LINKS, "Embed Links"),
        (Permissions::ATTACH_FILES, "Attach Files"),
        (Permissions::MANAGE_MESSAGES, "Manage Messages"),
        (Permissions::MANAGE_ROLES, "Manage Roles"),
        (Permissions::BAN_MEMBERS, "Ban Members"),
        (Permissions::MODERATE_MEMBERS, "Timeout Members"),
    ]
    .into_iter()
    .filter(|(permission, _)| required.contains(*permission))
    .collect()
}

pub async fn defer_ephemeral_interaction(
    client: &Arc<Client>,
    application_id: Id<ApplicationMarker>,
    interaction_id: Id<InteractionMarker>,
    token: &str,
) -> Result<()> {
    let response = InteractionResponse {
        kind: InteractionResponseType::DeferredChannelMessageWithSource,
        data: Some(InteractionResponseData {
            flags: Some(MessageFlags::EPHEMERAL),
            ..InteractionResponseData::default()
        }),
    };
    respond_interaction(client, application_id, interaction_id, token, &response).await
}

pub async fn defer_update_interaction(
    client: &Arc<Client>,
    application_id: Id<ApplicationMarker>,
    interaction_id: Id<InteractionMarker>,
    token: &str,
) -> Result<()> {
    let response = InteractionResponse {
        kind: InteractionResponseType::DeferredUpdateMessage,
        data: None,
    };
    respond_interaction(client, application_id, interaction_id, token, &response).await
}

pub async fn respond_interaction(
    client: &Arc<Client>,
    application_id: Id<ApplicationMarker>,
    interaction_id: Id<InteractionMarker>,
    token: &str,
    response: &InteractionResponse,
) -> Result<()> {
    client
        .interaction(application_id)
        .create_response(interaction_id, token, response)
        .await
        .context("responding to interaction")?;
    Ok(())
}

pub async fn respond_interaction_attachment(
    client: &Arc<Client>,
    application_id: Id<ApplicationMarker>,
    interaction_id: Id<InteractionMarker>,
    token: &str,
    content: &str,
    filename: &str,
    bytes: Vec<u8>,
) -> Result<()> {
    let attachment = HttpAttachment::from_bytes(filename.to_owned(), bytes, 0);
    let response = InteractionResponse {
        kind: InteractionResponseType::ChannelMessageWithSource,
        data: Some(InteractionResponseData {
            allowed_mentions: Some(AllowedMentions::default()),
            attachments: Some(vec![attachment]),
            content: Some(content.to_owned()),
            flags: Some(MessageFlags::EPHEMERAL),
            ..InteractionResponseData::default()
        }),
    };
    respond_interaction(client, application_id, interaction_id, token, &response).await
}

pub async fn edit_interaction_response(
    client: &Arc<Client>,
    application_id: Id<ApplicationMarker>,
    token: &str,
    content: &str,
) -> Result<()> {
    client
        .interaction(application_id)
        .update_response(token)
        .allowed_mentions(Some(&AllowedMentions::default()))
        .content(Some(content))
        .await
        .context("editing interaction response")?;
    Ok(())
}

pub async fn edit_interaction_response_data(
    client: &Arc<Client>,
    application_id: Id<ApplicationMarker>,
    token: &str,
    data: &InteractionResponseData,
) -> Result<()> {
    let allowed_mentions = data.allowed_mentions.clone().unwrap_or_default();
    client
        .interaction(application_id)
        .update_response(token)
        .allowed_mentions(Some(&allowed_mentions))
        .content(data.content.as_deref())
        .embeds(data.embeds.as_deref())
        .components(data.components.as_deref())
        .await
        .context("editing interaction response")?;
    Ok(())
}

#[derive(Clone, Copy)]
pub struct LedgerRecoveryConfig<'a> {
    pub base_match_config: &'a MatchConfig,
    pub max_decoded_pixels: u64,
}

pub async fn load_ledger(
    client: &Arc<Client>,
    attachment_http: &ReqwestClient,
    ledger_channel_id: Id<ChannelMarker>,
    expected_guild_id: Id<GuildMarker>,
    bot_user_id: Id<UserMarker>,
    secret: &str,
    recovery: LedgerRecoveryConfig<'_>,
) -> Result<LedgerLoad> {
    let mut loaded = LedgerAccumulator::default();
    read_storage_messages(
        client,
        attachment_http,
        ledger_channel_id,
        bot_user_id,
        expected_guild_id,
        secret,
        &mut loaded,
    )
    .await?;
    recover_stale_specimen_records(
        client,
        attachment_http,
        secret,
        expected_guild_id,
        bot_user_id,
        recovery,
        &mut loaded,
    )
    .await;

    info!(
        event = "ledger.loaded",
        specimens = loaded.specimens.len(),
        config_loaded = loaded.guild_config.is_some(),
        recovered_specimens = loaded.recovered_specimens,
        storage_messages_seen = loaded.messages_seen,
        ignored_non_bot_messages = loaded.ignored_non_bot_messages,
        "ledger loaded"
    );

    Ok(LedgerLoad {
        specimens: loaded.specimens,
        guild_config: loaded.guild_config,
        storage: DiscordStorageState {
            channel_id: ledger_channel_id,
            config_message_id: loaded.config_message_id,
            specimens: loaded.stored_specimens,
        },
    })
}

pub async fn create_ledger_record_message(
    client: &Arc<Client>,
    channel_id: Id<ChannelMarker>,
    record: &SpecimenRecord,
    image_attachments: Vec<SpecimenImageAttachment>,
    hmac_secret: &str,
) -> Result<StoredSpecimen> {
    if record.sig.is_none() {
        bail!("specimen record must be signed before storage");
    }
    let record_attachment = specimen_record_attachment(record)?;
    let body = signed_specimen_manifest_to_discord(record, &record_attachment, hmac_secret)?;
    if body.len() > MAX_DISCORD_STORAGE_CONTENT {
        bail!("single specimen manifest exceeds Discord message limit");
    }

    let mut attachments = vec![HttpAttachment::from_bytes(
        record_attachment.filename,
        record_attachment.bytes,
        0,
    )];
    attachments.extend(
        image_attachments
            .into_iter()
            .enumerate()
            .map(|(index, attachment)| {
                // Twilight owns upload bodies as Vec<u8>, so ledger image attachments
                // briefly duplicate the Bytes buffer while this HTTP request is built.
                HttpAttachment::from_bytes(
                    attachment.filename,
                    attachment.bytes.to_vec(),
                    index as u64 + 1,
                )
            }),
    );
    let mut request = client.create_message(channel_id).content(&body);
    request = request.attachments(&attachments);

    let message = request
        .await
        .context("writing specimen ledger record")?
        .model()
        .await
        .context("decoding created specimen message")?;
    Ok(StoredSpecimen {
        message_id: message.id,
        specimen_id: record.specimen_id.clone(),
    })
}

pub async fn upsert_config_record_message(
    client: &Arc<Client>,
    channel_id: Id<ChannelMarker>,
    existing_message_id: Option<Id<MessageMarker>>,
    record: &GuildConfigRecord,
    hmac_secret: &str,
) -> Result<Option<Id<MessageMarker>>> {
    let record_attachment = config_record_attachment(record)?;
    let body = signed_config_manifest_to_discord(record, &record_attachment, hmac_secret)?;
    if body.len() > MAX_DISCORD_STORAGE_CONTENT {
        bail!("guild config manifest exceeds Discord message limit");
    }

    let record_channel_id = record.config.ledger_channel_id()?;
    if record_channel_id != channel_id {
        bail!(
            "config ledger channel {} does not match active storage channel {}",
            record_channel_id.get(),
            channel_id.get()
        );
    }

    let attachments = vec![HttpAttachment::from_bytes(
        record_attachment.filename,
        record_attachment.bytes,
        0,
    )];

    if let Some(message_id) = existing_message_id {
        client
            .update_message(channel_id, message_id)
            .content(Some(&body))
            .attachments(&attachments)
            .keep_attachment_ids(&[])
            .await
            .context("editing guild config ledger record")?;
        Ok(None)
    } else {
        let message = client
            .create_message(channel_id)
            .content(&body)
            .attachments(&attachments)
            .await
            .context("writing guild config ledger record")?
            .model()
            .await
            .context("decoding created guild config message")?;
        Ok(Some(message.id))
    }
}

#[derive(Default)]
struct LedgerAccumulator {
    specimens: Vec<SpecimenRecord>,
    specimen_ids: HashSet<String>,
    guild_config: Option<GuildConfig>,
    config_message_id: Option<Id<MessageMarker>>,
    config_sort_key: Option<ConfigSortKey>,
    recoverable_specimens: Vec<RecoverableSpecimenMessage>,
    recovered_specimens: usize,
    stored_specimens: Vec<StoredSpecimen>,
    messages_seen: usize,
    ignored_non_bot_messages: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct ConfigSortKey {
    updated_at: String,
    created_at: String,
    message_id: u64,
}

#[derive(Clone, Debug)]
struct RecoverableSpecimenMessage {
    channel_id: Id<ChannelMarker>,
    message_id: Id<MessageMarker>,
    specimen_id: Option<String>,
    attachments: Vec<Attachment>,
    reason: String,
}

enum ParsedStorageMessage {
    Ignored,
    Specimen {
        message_id: Id<MessageMarker>,
        record: Box<SpecimenRecord>,
    },
    Config {
        message_id: Id<MessageMarker>,
        record: Box<GuildConfigRecord>,
    },
    RecoverableSpecimen(RecoverableSpecimenMessage),
}

struct SpecimenRecoveryContext<'a> {
    client: &'a Arc<Client>,
    attachment_http: &'a ReqwestClient,
    secret: &'a str,
    expected_guild_id: Id<GuildMarker>,
    bot_user_id: Id<UserMarker>,
    match_config: MatchConfig,
    max_decoded_pixels: u64,
}

async fn read_storage_messages(
    client: &Arc<Client>,
    attachment_http: &ReqwestClient,
    ledger_channel_id: Id<ChannelMarker>,
    bot_user_id: Id<UserMarker>,
    expected_guild_id: Id<GuildMarker>,
    secret: &str,
    loaded: &mut LedgerAccumulator,
) -> Result<()> {
    const MAX_STORAGE_MESSAGES: usize = 100_000;
    const STORAGE_ATTACHMENT_FETCH_CONCURRENCY: usize = 16;
    let mut before = None;
    let mut message_count = 0usize;

    loop {
        let response = if let Some(before_id) = before {
            client
                .channel_messages(ledger_channel_id)
                .before(before_id)
                .limit(100)
                .await
        } else {
            client.channel_messages(ledger_channel_id).limit(100).await
        };

        let messages = response
            .context("reading storage channel")?
            .models()
            .await
            .context("decoding storage channel messages")?;

        if messages.is_empty() {
            break;
        }

        before = messages.last().map(|message| message.id);
        message_count = message_count.saturating_add(messages.len());
        if message_count > MAX_STORAGE_MESSAGES {
            bail!(
                "storage channel has more than {MAX_STORAGE_MESSAGES} messages; refusing partial ledger load"
            );
        }

        let mut tasks = FuturesUnordered::new();
        for message in messages {
            loaded.messages_seen = loaded.messages_seen.saturating_add(1);
            if message.author.id != bot_user_id {
                loaded.ignored_non_bot_messages = loaded.ignored_non_bot_messages.saturating_add(1);
                continue;
            }
            tasks.push(parse_storage_message(
                attachment_http,
                message,
                bot_user_id,
                expected_guild_id,
                secret,
            ));
            if tasks.len() >= STORAGE_ATTACHMENT_FETCH_CONCURRENCY
                && let Some(parsed) = tasks.next().await
            {
                apply_parsed_storage_message(loaded, parsed);
            }
        }
        while let Some(parsed) = tasks.next().await {
            apply_parsed_storage_message(loaded, parsed);
        }
    }

    Ok(())
}

async fn parse_storage_message(
    attachment_http: &ReqwestClient,
    message: twilight_model::channel::Message,
    bot_user_id: Id<UserMarker>,
    expected_guild_id: Id<GuildMarker>,
    secret: &str,
) -> ParsedStorageMessage {
    if message.author.id != bot_user_id {
        return ParsedStorageMessage::Ignored;
    }

    if message.content.starts_with(SPECIMEN_PREFIX) {
        let manifest = match parse_and_verify_specimen_manifest(
            &message.content,
            secret,
            expected_guild_id,
        ) {
            Ok(Some(manifest)) => manifest,
            Ok(None) => {
                warn!(
                    event = "ledger.record_ignored",
                    message_id = message.id.get(),
                    "ignored invalid specimen manifest"
                );
                return ParsedStorageMessage::Ignored;
            }
            Err(source) => {
                warn!(
                    event = "ledger.record_parse_failed",
                    message_id = message.id.get(),
                    ?source,
                    "failed to parse specimen ledger manifest; will try image-attachment recovery"
                );
                return recoverable_specimen_message(&message, None, &source).map_or(
                    ParsedStorageMessage::Ignored,
                    ParsedStorageMessage::RecoverableSpecimen,
                );
            }
        };
        let record_bytes = match fetch_specimen_record_attachment(
            attachment_http,
            &message,
            &manifest,
        )
        .await
        {
            Ok(bytes) => bytes,
            Err(source) => {
                warn!(
                    event = "ledger.record_attachment_fetch_failed",
                    message_id = message.id.get(),
                    specimen_id = %manifest.specimen_id,
                    ?source,
                    "failed to fetch specimen ledger record attachment; will try image-attachment recovery"
                );
                return recoverable_specimen_message(&message, Some(manifest.specimen_id), &source)
                    .map_or(
                        ParsedStorageMessage::Ignored,
                        ParsedStorageMessage::RecoverableSpecimen,
                    );
            }
        };
        match parse_and_verify_specimen_record(&manifest, &record_bytes, secret, expected_guild_id)
        {
            Ok(record) => ParsedStorageMessage::Specimen {
                message_id: message.id,
                record: Box::new(record),
            },
            Err(source) => {
                warn!(
                    event = "ledger.record_parse_failed",
                    message_id = message.id.get(),
                    specimen_id = %manifest.specimen_id,
                    ?source,
                    "failed to parse specimen ledger record; will try image-attachment recovery"
                );
                recoverable_specimen_message(&message, Some(manifest.specimen_id), &source).map_or(
                    ParsedStorageMessage::Ignored,
                    ParsedStorageMessage::RecoverableSpecimen,
                )
            }
        }
    } else if message.content.starts_with(CONFIG_PREFIX) {
        let manifest =
            match parse_and_verify_config_manifest(&message.content, secret, expected_guild_id) {
                Ok(Some(manifest)) => manifest,
                Ok(None) => {
                    warn!(
                        event = "ledger.record_ignored",
                        message_id = message.id.get(),
                        "ignored invalid guild config manifest"
                    );
                    return ParsedStorageMessage::Ignored;
                }
                Err(source) => {
                    warn!(
                        event = "ledger.record_parse_failed",
                        message_id = message.id.get(),
                        ?source,
                        "failed to parse guild config ledger manifest"
                    );
                    return ParsedStorageMessage::Ignored;
                }
            };
        let record_bytes =
            match fetch_config_record_attachment(attachment_http, &message, &manifest).await {
                Ok(bytes) => bytes,
                Err(source) => {
                    warn!(
                        event = "ledger.config_attachment_fetch_failed",
                        message_id = message.id.get(),
                        ?source,
                        "failed to fetch guild config ledger attachment"
                    );
                    return ParsedStorageMessage::Ignored;
                }
            };
        match parse_and_verify_config_record(&manifest, &record_bytes, secret, expected_guild_id) {
            Ok(record) => ParsedStorageMessage::Config {
                message_id: message.id,
                record: Box::new(record),
            },
            Err(source) => {
                warn!(
                    event = "ledger.record_parse_failed",
                    message_id = message.id.get(),
                    ?source,
                    "failed to parse guild config ledger record"
                );
                ParsedStorageMessage::Ignored
            }
        }
    } else {
        warn!(
            event = "ledger.record_ignored",
            message_id = message.id.get(),
            "ignored unsupported ledger record"
        );
        ParsedStorageMessage::Ignored
    }
}

fn apply_parsed_storage_message(loaded: &mut LedgerAccumulator, parsed: ParsedStorageMessage) {
    match parsed {
        ParsedStorageMessage::Ignored => {}
        ParsedStorageMessage::Specimen { message_id, record } => {
            let record = *record;
            if loaded.specimen_ids.insert(record.specimen_id.clone()) {
                loaded.stored_specimens.push(StoredSpecimen {
                    message_id,
                    specimen_id: record.specimen_id.clone(),
                });
                loaded.specimens.push(record);
            } else {
                warn!(
                    event = "ledger.duplicate_specimen_ignored",
                    specimen_id = record.specimen_id,
                    message_id = message_id.get(),
                    "ignored duplicate specimen record"
                );
            }
        }
        ParsedStorageMessage::Config { message_id, record } => {
            let record = *record;
            let sort_key = ConfigSortKey {
                updated_at: record.config.updated_at.clone(),
                created_at: record.created_at.clone(),
                message_id: message_id.get(),
            };
            if loaded
                .config_sort_key
                .as_ref()
                .is_none_or(|current| &sort_key > current)
            {
                loaded.guild_config = Some(record.config);
                loaded.config_message_id = Some(message_id);
                loaded.config_sort_key = Some(sort_key);
            }
        }
        ParsedStorageMessage::RecoverableSpecimen(message) => {
            loaded.recoverable_specimens.push(message);
        }
    }
}

fn recoverable_specimen_message(
    message: &twilight_model::channel::Message,
    specimen_id: Option<String>,
    source: &anyhow::Error,
) -> Option<RecoverableSpecimenMessage> {
    message
        .attachments
        .iter()
        .any(|attachment| specimen_image_variant(&attachment.filename).is_some())
        .then(|| RecoverableSpecimenMessage {
            channel_id: message.channel_id,
            message_id: message.id,
            specimen_id,
            attachments: message.attachments.clone(),
            reason: source.to_string(),
        })
}

async fn recover_stale_specimen_records(
    client: &Arc<Client>,
    attachment_http: &ReqwestClient,
    secret: &str,
    expected_guild_id: Id<GuildMarker>,
    bot_user_id: Id<UserMarker>,
    recovery: LedgerRecoveryConfig<'_>,
    loaded: &mut LedgerAccumulator,
) {
    if loaded.recoverable_specimens.is_empty() {
        return;
    }

    let match_config = loaded.guild_config.as_ref().map_or_else(
        || recovery.base_match_config.clone(),
        |config| {
            config
                .detection_hyperparameters
                .effective_match_config(recovery.base_match_config)
        },
    );
    let context = SpecimenRecoveryContext {
        client,
        attachment_http,
        secret,
        expected_guild_id,
        bot_user_id,
        match_config,
        max_decoded_pixels: recovery.max_decoded_pixels,
    };
    let recoverable = std::mem::take(&mut loaded.recoverable_specimens);
    for stale in recoverable {
        match recover_stale_specimen_record(&context, &stale).await {
            Ok(record) => {
                if loaded.specimen_ids.insert(record.specimen_id.clone()) {
                    loaded.stored_specimens.push(StoredSpecimen {
                        message_id: stale.message_id,
                        specimen_id: record.specimen_id.clone(),
                    });
                    loaded.specimens.push(record);
                    loaded.recovered_specimens += 1;
                } else {
                    warn!(
                        event = "ledger.recovered_duplicate_specimen_ignored",
                        message_id = stale.message_id.get(),
                        "ignored recovered duplicate specimen record"
                    );
                }
            }
            Err(source) => warn!(
                event = "ledger.specimen_recovery_failed",
                message_id = stale.message_id.get(),
                reason = %stale.reason,
                ?source,
                "failed to recover stale specimen ledger record from image attachments"
            ),
        }
    }
}

async fn recover_stale_specimen_record(
    context: &SpecimenRecoveryContext<'_>,
    stale: &RecoverableSpecimenMessage,
) -> Result<SpecimenRecord> {
    let original_attachment = find_recovery_image_attachment(
        &stale.attachments,
        stale.specimen_id.as_deref(),
        "original",
    )
    .ok_or_else(|| anyhow!("recoverable specimen has no original image attachment"))?;
    let specimen_id = stale
        .specimen_id
        .clone()
        .or_else(|| specimen_id_from_image_filename(&original_attachment.filename, "original"))
        .ok_or_else(|| anyhow!("recoverable specimen id is unavailable"))?;
    let original_bytes =
        fetch_specimen_image_attachment(context.attachment_http, original_attachment)
            .await
            .context("fetching original recovery image")?;
    let image = hash_recovered_specimen_image(
        original_bytes.clone(),
        original_attachment.content_type.clone(),
        context.max_decoded_pixels,
        &context.match_config,
    )
    .await
    .context("hashing original recovery image")?;

    let preview = match find_recovery_image_attachment(
        &stale.attachments,
        Some(&specimen_id),
        "discord-preview",
    ) {
        Some(attachment) => {
            let bytes = fetch_specimen_image_attachment(context.attachment_http, attachment)
                .await
                .context("fetching Discord preview recovery image")?;
            Some(
                hash_recovered_specimen_image(
                    bytes,
                    attachment.content_type.clone(),
                    context.max_decoded_pixels,
                    &context.match_config,
                )
                .await
                .context("hashing Discord preview recovery image")?,
            )
        }
        None => None,
    };

    let record = SpecimenRecord::new_recovered(
        context.expected_guild_id,
        stale.channel_id,
        stale.message_id,
        context.bot_user_id,
        specimen_id,
        image,
        preview,
    )
    .sign(context.secret)?;
    record.validate(context.expected_guild_id)?;

    if let Err(source) =
        repair_specimen_record_message(context.client, stale, &record, context.secret).await
    {
        warn!(
            event = "ledger.specimen_repair_failed",
            message_id = stale.message_id.get(),
            specimen_id = %record.specimen_id,
            ?source,
            "recovered specimen for memory but failed to rewrite ledger msgpack"
        );
    } else {
        info!(
            event = "ledger.specimen_repaired",
            message_id = stale.message_id.get(),
            specimen_id = %record.specimen_id,
            "recovered stale specimen ledger record from image attachments"
        );
    }

    Ok(record)
}

async fn hash_recovered_specimen_image(
    bytes: Bytes,
    mime: Option<String>,
    max_decoded_pixels: u64,
    match_config: &MatchConfig,
) -> Result<crate::image::types::ImageFingerprint> {
    let match_config = match_config.clone();
    tokio::task::spawn_blocking(move || {
        hash_image_bytes(
            bytes.as_ref(),
            mime,
            max_decoded_pixels,
            &match_config,
            HashMode::Specimen,
        )
    })
    .await
    .context("specimen recovery hash task panicked")?
}

async fn repair_specimen_record_message(
    client: &Arc<Client>,
    stale: &RecoverableSpecimenMessage,
    record: &SpecimenRecord,
    secret: &str,
) -> Result<()> {
    let record_attachment = specimen_record_attachment(record)?;
    let body = signed_specimen_manifest_to_discord(record, &record_attachment, secret)?;
    if body.len() > MAX_DISCORD_STORAGE_CONTENT {
        bail!("recovered specimen manifest exceeds Discord message limit");
    }

    let keep_attachment_ids = stale
        .attachments
        .iter()
        .filter(|attachment| specimen_image_variant(&attachment.filename).is_some())
        .map(|attachment| attachment.id)
        .collect::<Vec<Id<AttachmentMarker>>>();
    let attachments = vec![HttpAttachment::from_bytes(
        record_attachment.filename,
        record_attachment.bytes,
        0,
    )];

    client
        .update_message(stale.channel_id, stale.message_id)
        .content(Some(&body))
        .attachments(&attachments)
        .keep_attachment_ids(&keep_attachment_ids)
        .await
        .context("rewriting recovered specimen ledger record")?;
    Ok(())
}

async fn fetch_specimen_image_attachment(
    attachment_http: &ReqwestClient,
    attachment: &Attachment,
) -> Result<Bytes> {
    fetch_limited_discord_attachment(
        attachment_http,
        attachment,
        MAX_SPECIMEN_ATTACHMENT_BYTES,
        "specimen image",
    )
    .await
}

fn find_recovery_image_attachment<'a>(
    attachments: &'a [Attachment],
    specimen_id: Option<&str>,
    variant: &str,
) -> Option<&'a Attachment> {
    attachments.iter().find(|attachment| {
        let Some((id, found_variant)) = specimen_image_variant(&attachment.filename) else {
            return false;
        };
        found_variant == variant && specimen_id.is_none_or(|expected| id == expected)
    })
}

fn specimen_image_variant(filename: &str) -> Option<(&str, &str)> {
    specimen_id_from_variant_filename(filename, "original")
        .map(|id| (id, "original"))
        .or_else(|| {
            specimen_id_from_variant_filename(filename, "discord-preview")
                .map(|id| (id, "discord-preview"))
        })
}

fn specimen_id_from_image_filename(filename: &str, variant: &str) -> Option<String> {
    specimen_id_from_variant_filename(filename, variant).map(str::to_owned)
}

fn specimen_id_from_variant_filename<'a>(filename: &'a str, variant: &str) -> Option<&'a str> {
    let marker = format!("_{variant}.");
    let (specimen_id, _extension) = filename.rsplit_once(&marker)?;
    (!specimen_id.trim().is_empty()).then_some(specimen_id)
}

async fn fetch_limited_discord_attachment(
    attachment_http: &ReqwestClient,
    attachment: &Attachment,
    max_bytes: usize,
    label: &str,
) -> Result<Bytes> {
    if attachment.size > max_bytes as u64 {
        bail!("{label} attachment exceeds storage limit");
    }

    let url = url::Url::parse(&attachment.url)
        .with_context(|| format!("parsing {label} attachment URL"))?;
    if url.scheme() != "https" {
        bail!("{label} attachment URL must use https");
    }
    let host = url
        .host_str()
        .ok_or_else(|| anyhow!("{label} attachment URL has no host"))?;
    if !is_discord_host(host) {
        bail!("{label} attachment URL host is not a Discord media host");
    }

    let response = attachment_http
        .get(url.clone())
        .send()
        .await
        .with_context(|| format!("fetching {label} attachment"))?;
    ensure_discord_attachment_response_url(&response, &url, label)?;
    if !response.status().is_success() {
        bail!("{label} attachment fetch returned {}", response.status());
    }
    if let Some(length) = response.content_length()
        && length > max_bytes as u64
    {
        bail!("{label} attachment response exceeds storage limit");
    }

    let attachment_size =
        usize::try_from(attachment.size).context("attachment size overflows usize")?;
    let mut body = BytesMut::with_capacity(attachment_size.min(max_bytes));
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("reading {label} attachment body"))?;
        if body.len().saturating_add(chunk.len()) > max_bytes {
            bail!("{label} attachment body exceeds storage limit");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body.freeze())
}

async fn fetch_specimen_record_attachment(
    attachment_http: &ReqwestClient,
    message: &twilight_model::channel::Message,
    manifest: &SpecimenManifest,
) -> Result<Vec<u8>> {
    let attachment = message
        .attachments
        .iter()
        .find(|attachment| attachment.filename == manifest.record_attachment.as_str())
        .ok_or_else(|| anyhow::anyhow!("specimen ledger record attachment is missing"))?;

    if attachment.size != u64::from(manifest.record_bytes) {
        bail!("specimen ledger record attachment size does not match manifest");
    }

    if attachment.size > MAX_SPECIMEN_RECORD_ATTACHMENT_BYTES as u64 {
        bail!("specimen ledger record attachment exceeds storage limit");
    }

    let url = url::Url::parse(&attachment.url).context("parsing specimen record attachment URL")?;
    if url.scheme() != "https" {
        bail!("specimen record attachment URL must use https");
    }
    let host = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("specimen record attachment URL has no host"))?;
    if !is_discord_host(host) {
        bail!("specimen record attachment URL host is not a Discord media host");
    }

    let response = attachment_http
        .get(url.clone())
        .send()
        .await
        .context("fetching specimen record attachment")?;
    ensure_discord_attachment_response_url(&response, &url, "specimen record")?;
    if !response.status().is_success() {
        bail!(
            "specimen record attachment fetch returned {}",
            response.status()
        );
    }
    if let Some(length) = response.content_length()
        && length > MAX_SPECIMEN_RECORD_ATTACHMENT_BYTES as u64
    {
        bail!("specimen record attachment response exceeds storage limit");
    }

    let attachment_size =
        usize::try_from(attachment.size).context("specimen record attachment size overflows")?;
    let mut body = BytesMut::with_capacity(attachment_size);
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading specimen record attachment body")?;
        if body.len().saturating_add(chunk.len()) > MAX_SPECIMEN_RECORD_ATTACHMENT_BYTES {
            bail!("specimen record attachment body exceeds storage limit");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body.to_vec())
}

async fn fetch_config_record_attachment(
    attachment_http: &ReqwestClient,
    message: &twilight_model::channel::Message,
    manifest: &GuildConfigManifest,
) -> Result<Vec<u8>> {
    let attachment = message
        .attachments
        .iter()
        .find(|attachment| attachment.filename == manifest.record_attachment.as_str())
        .ok_or_else(|| anyhow::anyhow!("guild config record attachment is missing"))?;

    if attachment.size != u64::from(manifest.record_bytes) {
        bail!("guild config record attachment size does not match manifest");
    }

    if attachment.size > MAX_CONFIG_RECORD_ATTACHMENT_BYTES as u64 {
        bail!("guild config record attachment exceeds storage limit");
    }

    let url = url::Url::parse(&attachment.url).context("parsing guild config attachment URL")?;
    if url.scheme() != "https" {
        bail!("guild config attachment URL must use https");
    }
    let host = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("guild config attachment URL has no host"))?;
    if !is_discord_host(host) {
        bail!("guild config attachment URL host is not a Discord media host");
    }

    let response = attachment_http
        .get(url.clone())
        .send()
        .await
        .context("fetching guild config record attachment")?;
    ensure_discord_attachment_response_url(&response, &url, "guild config")?;
    if !response.status().is_success() {
        bail!(
            "guild config record attachment fetch returned {}",
            response.status()
        );
    }
    if let Some(length) = response.content_length()
        && length > MAX_CONFIG_RECORD_ATTACHMENT_BYTES as u64
    {
        bail!("guild config record attachment response exceeds storage limit");
    }

    let attachment_size =
        usize::try_from(attachment.size).context("guild config attachment size overflows")?;
    let mut body = BytesMut::with_capacity(attachment_size);
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading guild config record attachment body")?;
        if body.len().saturating_add(chunk.len()) > MAX_CONFIG_RECORD_ATTACHMENT_BYTES {
            bail!("guild config record attachment body exceeds storage limit");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body.to_vec())
}

fn ensure_discord_attachment_response_url(
    response: &reqwest::Response,
    expected: &url::Url,
    label: &str,
) -> Result<()> {
    if response.url().as_str() != expected.as_str() {
        bail!("{label} attachment request followed a redirect");
    }
    Ok(())
}

pub async fn delete_message(
    client: &Arc<Client>,
    channel_id: Id<ChannelMarker>,
    message_id: Id<MessageMarker>,
) -> bool {
    match client.delete_message(channel_id, message_id).await {
        Ok(_) => true,
        Err(source) => {
            error!(
                event = "moderation.delete_failed",
                channel_id = channel_id.get(),
                message_id = message_id.get(),
                ?source,
                "failed to delete message"
            );
            false
        }
    }
}

pub async fn remove_member_roles(
    client: &Arc<Client>,
    guild_id: Id<GuildMarker>,
    user_id: Id<UserMarker>,
    removable_role_ids: &[Id<RoleMarker>],
) -> RoleRemovalOutcome {
    if removable_role_ids.is_empty() {
        warn!(
            event = "moderation.role_remove_skipped",
            guild_id = guild_id.get(),
            user_id = user_id.get(),
            "role removal requested but no removable roles are configured"
        );
        return RoleRemovalOutcome::default();
    }

    let member = match client.guild_member(guild_id, user_id).await {
        Ok(response) => match response.model().await {
            Ok(member) => member,
            Err(source) => {
                error!(
                    event = "moderation.member_decode_failed",
                    guild_id = guild_id.get(),
                    user_id = user_id.get(),
                    ?source,
                    "failed to decode member before role removal"
                );
                return RoleRemovalOutcome::default();
            }
        },
        Err(source) => {
            error!(
                event = "moderation.member_fetch_failed",
                guild_id = guild_id.get(),
                user_id = user_id.get(),
                ?source,
                "failed to fetch member before role removal"
            );
            return RoleRemovalOutcome::default();
        }
    };

    let roles = member
        .roles
        .into_iter()
        .filter(|role_id| removable_role_ids.contains(role_id))
        .collect::<Vec<_>>();

    let mut outcome = RoleRemovalOutcome {
        attempted: roles.len(),
        removed: 0,
        failed: 0,
    };

    for role_id in roles {
        match client
            .remove_guild_member_role(guild_id, user_id, role_id)
            .await
        {
            Ok(_) => outcome.removed += 1,
            Err(source) => {
                outcome.failed += 1;
                error!(
                    event = "moderation.role_remove_failed",
                    guild_id = guild_id.get(),
                    user_id = user_id.get(),
                    role_id = role_id.get(),
                    ?source,
                    "failed to remove member role"
                );
            }
        }
    }

    outcome
}

pub async fn ban_user(
    client: &Arc<Client>,
    guild_id: Id<GuildMarker>,
    user_id: Id<UserMarker>,
    delete_message_seconds: u32,
    reason: &str,
) -> bool {
    match client
        .create_ban(guild_id, user_id)
        .delete_message_seconds(delete_message_seconds)
        .reason(reason)
        .await
    {
        Ok(_) => true,
        Err(source) => {
            error!(
                event = "moderation.ban_failed",
                guild_id = guild_id.get(),
                user_id = user_id.get(),
                delete_message_seconds,
                ?source,
                "failed to ban user"
            );
            false
        }
    }
}

pub async fn kick_user(
    client: &Arc<Client>,
    guild_id: Id<GuildMarker>,
    user_id: Id<UserMarker>,
    reason: &str,
) -> bool {
    match client
        .remove_guild_member(guild_id, user_id)
        .reason(reason)
        .await
    {
        Ok(_) => true,
        Err(source) => {
            error!(
                event = "moderation.kick_failed",
                guild_id = guild_id.get(),
                user_id = user_id.get(),
                ?source,
                "failed to kick user"
            );
            false
        }
    }
}

pub async fn timeout_user(
    client: &Arc<Client>,
    guild_id: Id<GuildMarker>,
    user_id: Id<UserMarker>,
    timeout_seconds: u32,
    reason: &str,
) -> bool {
    let expires_at = match Timestamp::from_secs(
        Utc::now()
            .timestamp()
            .saturating_add(i64::from(timeout_seconds)),
    ) {
        Ok(timestamp) => timestamp,
        Err(source) => {
            error!(
                event = "moderation.timeout_timestamp_failed",
                guild_id = guild_id.get(),
                user_id = user_id.get(),
                timeout_seconds,
                ?source,
                "failed to build timeout timestamp"
            );
            return false;
        }
    };

    match client
        .update_guild_member(guild_id, user_id)
        .communication_disabled_until(Some(expires_at))
        .reason(reason)
        .await
    {
        Ok(_) => true,
        Err(source) => {
            error!(
                event = "moderation.timeout_failed",
                guild_id = guild_id.get(),
                user_id = user_id.get(),
                timeout_seconds,
                ?source,
                "failed to timeout user"
            );
            false
        }
    }
}

pub(crate) async fn post_bot_log_to_channel(
    client: &Arc<Client>,
    channel_id: Id<ChannelMarker>,
    log: &RenderedBotLog,
) -> Result<Id<MessageMarker>> {
    let allowed_mentions = bot_log_allowed_mentions();
    let mut request = client
        .create_message(channel_id)
        .allowed_mentions(Some(&allowed_mentions))
        .embeds(&log.embeds)
        .attachments(&log.attachments);
    if let Some(content) = log.content.as_deref() {
        request = request.content(content);
    }
    let message = request
        .await
        .context("posting bot log")?
        .model()
        .await
        .context("decoding posted bot log")?;
    Ok(message.id)
}

pub(crate) async fn edit_bot_log_in_channel(
    client: &Arc<Client>,
    channel_id: Id<ChannelMarker>,
    message_id: Id<MessageMarker>,
    log: &RenderedBotLog,
) -> Result<()> {
    let allowed_mentions = bot_log_allowed_mentions();
    let request = client
        .update_message(channel_id, message_id)
        .allowed_mentions(Some(&allowed_mentions))
        .content(log.content.as_deref())
        .embeds(Some(&log.embeds))
        .attachments(&log.attachments);
    request.await.context("editing bot log")?;
    Ok(())
}

fn bot_log_allowed_mentions() -> AllowedMentions {
    AllowedMentions {
        parse: vec![MentionType::Roles, MentionType::Users],
        ..AllowedMentions::default()
    }
}

pub(crate) fn render_bot_log(event: BotLogEvent) -> RenderedBotLog {
    let attachments = event
        .attachments
        .into_iter()
        .take(4)
        .enumerate()
        .map(|(index, attachment)| {
            HttpAttachment::from_bytes(attachment.filename, attachment.bytes, index as u64)
        })
        .collect::<Vec<_>>();
    let fields = event
        .fields
        .into_iter()
        .take(24)
        .map(|field| EmbedField {
            name: truncate_chars(&field.name, 256),
            value: truncate_chars(&field.value, 1024),
            inline: field.inline,
        })
        .collect::<Vec<_>>();

    let timestamp = Timestamp::from_secs(Utc::now().timestamp()).ok();
    let embed = Embed {
        author: None,
        color: Some(event.color.value()),
        description: Some(truncate_chars(&event.description, 4096)),
        fields,
        footer: None,
        image: event
            .image_url
            .map(|url| twilight_model::channel::message::embed::EmbedImage {
                height: None,
                proxy_url: None,
                url,
                width: None,
            }),
        kind: "rich".to_owned(),
        provider: None,
        thumbnail: None,
        timestamp,
        title: Some(truncate_chars(&event.title, 256)),
        url: None,
        video: None,
    };

    RenderedBotLog {
        content: None,
        embeds: vec![embed],
        attachments,
    }
}

pub(crate) fn message_jump_link(
    guild_id: Id<GuildMarker>,
    channel_id: Id<ChannelMarker>,
    message_id: Id<MessageMarker>,
) -> String {
    format!(
        "https://discord.com/channels/{}/{}/{}",
        guild_id.get(),
        channel_id.get(),
        message_id.get()
    )
}

pub(crate) fn user_incident_label(candidate: &ImageCandidate) -> String {
    let mut parts = vec![
        format!("<@{}>", candidate.author_id.get()),
        format!("id `{}`", candidate.author_id.get()),
    ];
    if let Some(name) = candidate.author_username.as_deref() {
        parts.push(format!("username `{}`", sanitize_inline(name)));
    }
    if let Some(name) = candidate.author_global_name.as_deref() {
        parts.push(format!("display `{}`", sanitize_inline(name)));
    }
    parts.join(", ")
}

pub(crate) fn detection_action_summary(
    actions: &DetectionActions,
    results: DetectionActionResults,
    specimen_add: Option<&str>,
) -> Vec<String> {
    let mut taken = Vec::new();

    if actions.delete_message {
        taken.push(format!("`delete_message={}`", results.deleted));
    }
    if actions.remove_user_roles {
        taken.push(format!(
            "`remove_user_roles={}/{}, failed={}`",
            results.role_removal.removed,
            results.role_removal.attempted,
            results.role_removal.failed
        ));
    }
    if actions.timeout_user {
        taken.push(format!(
            "`timeout_user={}s:{timed_out}`",
            actions.timeout_seconds,
            timed_out = results.member.timed_out
        ));
    }
    if actions.ban_user {
        taken.push(format!("`ban_user={}`", results.member.banned));
    }
    if actions.kick_user {
        taken.push(format!("`kick_user={}`", results.member.kicked));
    }
    if actions.add_to_specimens {
        taken.push(format!(
            "`add_to_specimens={}`",
            specimen_add.unwrap_or("not_attempted")
        ));
    }

    taken
}

pub(crate) fn audit_row(
    candidate: &ImageCandidate,
    image_id: &str,
    decision: &str,
    outcome: Option<&MatchOutcome>,
    actions_taken: &[String],
    process_ms: u128,
) -> String {
    let timestamp = Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true);
    let specimen_id = outcome.map_or("none", |outcome| outcome.specimen_id.as_str());
    let score = outcome.map_or_else(|| "none".to_owned(), MatchOutcome::score_label);
    let anchor_hits = outcome
        .and_then(|outcome| outcome.local_anchor_hits)
        .map_or_else(|| "0".to_owned(), |value| value.to_string());
    let distinct_regions = outcome
        .and_then(|outcome| outcome.local_distinct_regions)
        .map_or_else(|| "0".to_owned(), |value| value.to_string());
    let mean_distance = outcome
        .and_then(|outcome| outcome.local_average_distance)
        .map_or_else(|| "none".to_owned(), |value| format!("{value:.2}"));
    let candidate_file_bytes = candidate
        .size_bytes
        .map_or_else(|| "unknown".to_owned(), |value| value.to_string());
    let action_summary = if actions_taken.is_empty() {
        "none".to_owned()
    } else {
        actions_taken
            .iter()
            .map(|value| value.replace('`', ""))
            .collect::<Vec<_>>()
            .join("+")
    };

    format!(
        "ts={} message_id={} user_id={} candidate_image_id={} candidate_file_bytes={} decision={} matched_specimen_id={} score={} anchor_hits={} distinct_regions={} mean_distance={} process_ms={} action_taken={} moderator_override=none",
        timestamp,
        candidate.message_id.get(),
        candidate.author_id.get(),
        image_id,
        candidate_file_bytes,
        decision,
        specimen_id,
        score,
        anchor_hits,
        distinct_regions,
        mean_distance,
        process_ms,
        action_summary
    )
}

fn sanitize_inline(value: &str) -> String {
    truncate_chars(&value.replace('`', "'"), 80)
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_owned();
    }
    let mut truncated = value
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    truncated.push_str("...");
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specimen_image_variant_extracts_ids_with_underscores() {
        assert_eq!(
            specimen_image_variant("spm_20260629_abcdef0123456789_original.jpg"),
            Some(("spm_20260629_abcdef0123456789", "original"))
        );
        assert_eq!(
            specimen_image_variant("spm_20260629_abcdef0123456789_discord-preview.webp"),
            Some(("spm_20260629_abcdef0123456789", "discord-preview"))
        );
    }

    #[test]
    fn specimen_image_variant_rejects_non_image_record_attachment() {
        assert_eq!(
            specimen_image_variant("spm_20260629_abcdef0123456789.sightline.msgpack"),
            None
        );
        assert_eq!(specimen_image_variant("not-a-specimen.jpg"), None);
    }

    #[test]
    fn registers_message_context_commands() {
        let commands = sightline_commands();
        for command_name in [VALIDATE_MESSAGE_COMMAND, VERIFY_MESSAGE_COMMAND] {
            let command = commands
                .iter()
                .find(|command| command.name == command_name)
                .expect("message context command is registered");

            assert!(matches!(command.kind, CommandType::Message));
            assert_eq!(
                command.default_member_permissions,
                Some(Permissions::MANAGE_MESSAGES)
            );
        }
    }
}
