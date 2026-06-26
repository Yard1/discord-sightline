#![allow(clippy::too_many_lines)]

use crate::{
    bot::{
        discord::{
            AUDIT_COMMAND, CONFIG_COMMAND, DOCTOR_COMMAND, EXPORT_HASHES_COMMAND,
            IMPORT_HASHES_COMMAND, IMPORT_IMAGES_COMMAND, STATS_COMMAND, check_bot_permissions,
            edit_interaction_response_data, message_jump_link, respond_interaction,
            respond_interaction_attachment,
        },
        extract::MAX_MANUAL_SPECIMEN_IMAGES_PER_INTERACTION,
        ledger::SpecimenRecord,
        runtime::{
            AppState, GuildMetricCounters, GuildMetricsSnapshot, PipelineTimingClassSnapshot,
            PipelineTimingSnapshot, SpecimenWriteLogContext, SpecimenWriteOutcome,
            TimingDistributionSnapshot,
        },
        specimen_import::import_image_candidates,
    },
    configuration::guild::{
        DetectionActions, GuildConfig, GuildConfigRecord, ScanPolicy, TIMEOUT_DURATION_SECONDS,
        TextGatePolicy,
    },
    image::types::{CandidateKind, ExportedImageFingerprint, ImageCandidate, ImageFingerprint},
};
use anyhow::{Context, Result, anyhow};
use std::fmt::Write as _;
use std::sync::atomic::Ordering;
use std::{collections::HashMap, sync::Arc};
use tracing::info;
use twilight_model::{
    application::interaction::{
        application_command::CommandData,
        message_component::MessageComponentInteractionData,
        modal::{ModalInteractionComponent, ModalInteractionData},
    },
    channel::{
        ChannelType,
        message::{
            AllowedMentions, Component, Embed, MessageFlags,
            component::{
                ActionRow, Button, ButtonStyle, FileUpload, Label, SelectDefaultValue, SelectMenu,
                SelectMenuOption, SelectMenuType, TextInput, TextInputStyle,
            },
            embed::{EmbedField, EmbedFooter},
        },
    },
    gateway::payload::incoming::InteractionCreate,
    guild::Permissions,
    http::interaction::{InteractionResponse, InteractionResponseData, InteractionResponseType},
    id::{
        Id,
        marker::{ChannelMarker, GuildMarker, MessageMarker, RoleMarker, UserMarker},
    },
};
use xxhash_rust::xxh3::xxh3_64;

type AdminResponse = InteractionResponse;

const ADVANCED_CONFIG_CHUNK_CHARS: usize = 3600;
const ADVANCED_CONFIG_MAX_CHUNKS: usize = 5;
const ADVANCED_CONFIG_MODAL_PREFIX: &str = "config:advanced-modal:v2:";
const LOG_MESSAGE_MODAL_PREFIX: &str = "config:log-message-modal:v2:";

#[derive(Debug, Clone, Copy)]
pub(crate) enum InteractionAckMode {
    Initial,
    Deferred,
}

pub async fn admin_command_response(
    state: &AppState,
    interaction: &InteractionCreate,
    command: &CommandData,
) -> Result<Option<AdminResponse>> {
    if command.name != CONFIG_COMMAND
        && command.name != AUDIT_COMMAND
        && command.name != DOCTOR_COMMAND
        && command.name != EXPORT_HASHES_COMMAND
        && command.name != IMPORT_HASHES_COMMAND
        && command.name != IMPORT_IMAGES_COMMAND
        && command.name != STATS_COMMAND
    {
        return Ok(None);
    }

    let moderation_command = matches!(
        command.name.as_str(),
        EXPORT_HASHES_COMMAND | IMPORT_HASHES_COMMAND | IMPORT_IMAGES_COMMAND
    );
    let authorized = if moderation_command {
        can_moderate(state, interaction)
    } else {
        can_configure(state, interaction)
    };
    if !authorized {
        return Ok(Some(message_response(
            state,
            InteractionResponseType::ChannelMessageWithSource,
            Some(if moderation_command {
                "Only administrators or configured moderator roles can import or export specimens."
                    .to_owned()
            } else {
                "Only administrators or configured moderator roles can use Sightline configuration."
                    .to_owned()
            }),
            None,
            None,
        )));
    }

    let response = match command.name.as_str() {
        AUDIT_COMMAND => specimen_audit_response(state),
        DOCTOR_COMMAND => doctor_response(state, interaction).await?,
        EXPORT_HASHES_COMMAND => {
            export_hashes(state, interaction).await?;
            return Ok(None);
        }
        IMPORT_HASHES_COMMAND => import_modal(state),
        IMPORT_IMAGES_COMMAND => upload_images_modal(state),
        STATS_COMMAND => stats_response(state).await?,
        _ => config_panel(state, false),
    };

    Ok(Some(response))
}

pub async fn handle_component(
    state: &AppState,
    interaction: &InteractionCreate,
    component: &MessageComponentInteractionData,
) -> Result<bool> {
    handle_component_with_mode(state, interaction, component, InteractionAckMode::Initial).await
}

pub async fn handle_deferred_component(
    state: &AppState,
    interaction: &InteractionCreate,
    component: &MessageComponentInteractionData,
) -> Result<bool> {
    handle_component_with_mode(state, interaction, component, InteractionAckMode::Deferred).await
}

async fn handle_component_with_mode(
    state: &AppState,
    interaction: &InteractionCreate,
    component: &MessageComponentInteractionData,
    mode: InteractionAckMode,
) -> Result<bool> {
    let custom_id = component.custom_id.as_str();
    if !custom_id.starts_with("config:") {
        return Ok(false);
    }

    if !can_configure(state, interaction) {
        respond_content_with_mode(
            state,
            interaction,
            mode,
            "Only authorized config users can change settings.",
        )
        .await?;
        return Ok(true);
    }

    match custom_id {
        "config:actions:v1" => {
            respond_json_with_mode(state, interaction, mode, actions_panel(state)).await?;
        }
        "config:roles:v1" => {
            respond_json_with_mode(state, interaction, mode, roles_panel(state)).await?;
        }
        "config:log-message:v1" => {
            respond_json_with_mode(
                state,
                interaction,
                mode,
                log_message_modal(state, interaction)?,
            )
            .await?;
        }
        "config:advanced:v1" => {
            respond_json_with_mode(
                state,
                interaction,
                mode,
                advanced_modal(state, interaction)?,
            )
            .await?;
        }
        "config:doctor:v1" => {
            respond_json_with_mode(
                state,
                interaction,
                mode,
                doctor_response(state, interaction).await?,
            )
            .await?;
        }
        "config:home:v1" => {
            respond_json_with_mode(state, interaction, mode, config_panel(state, true)).await?;
        }
        "config:toggle-enabled:v1" => {
            let mut config = state.active_config();
            config.enabled = !config.enabled;
            if let Err(source) = persist_config(state, interaction, config).await {
                respond_config_error_with_mode(state, interaction, mode, source).await?;
            } else {
                respond_json_with_mode(state, interaction, mode, config_panel(state, true)).await?;
            }
        }
        "config:bot-log-channel:v1" => {
            if let Err(source) =
                update_config_from_select(state, interaction, component, custom_id).await
            {
                respond_config_error_with_mode(state, interaction, mode, source).await?;
            } else {
                respond_json_with_mode(state, interaction, mode, config_panel(state, true)).await?;
            }
        }
        "config:moderator-roles:v1" | "config:scan-exempt-roles:v1" | "config:verified-role:v1" => {
            if let Err(source) =
                update_config_from_select(state, interaction, component, custom_id).await
            {
                respond_config_error_with_mode(state, interaction, mode, source).await?;
            } else {
                respond_json_with_mode(state, interaction, mode, roles_panel(state)).await?;
            }
        }
        "config:confirmed-actions:v1"
        | "config:suspicious-actions:v1"
        | "config:confirmed-timeout:v1"
        | "config:suspicious-timeout:v1" => {
            if let Err(source) =
                update_config_from_select(state, interaction, component, custom_id).await
            {
                respond_config_error_with_mode(state, interaction, mode, source).await?;
            } else {
                respond_json_with_mode(state, interaction, mode, actions_panel(state)).await?;
            }
        }
        _ => {
            respond_content_with_mode(
                state,
                interaction,
                mode,
                "Unsupported configuration action.",
            )
            .await?;
        }
    }

    Ok(true)
}

pub async fn handle_modal(
    state: &AppState,
    interaction: &InteractionCreate,
    modal: &ModalInteractionData,
) -> Result<bool> {
    handle_modal_with_mode(state, interaction, modal, InteractionAckMode::Initial).await
}

pub async fn handle_deferred_modal(
    state: &AppState,
    interaction: &InteractionCreate,
    modal: &ModalInteractionData,
) -> Result<bool> {
    handle_modal_with_mode(state, interaction, modal, InteractionAckMode::Deferred).await
}

async fn handle_modal_with_mode(
    state: &AppState,
    interaction: &InteractionCreate,
    modal: &ModalInteractionData,
    mode: InteractionAckMode,
) -> Result<bool> {
    let custom_id = modal.custom_id.as_str();

    match custom_id {
        "specimen:import-hashes-modal:v1" => {
            if !can_moderate(state, interaction) {
                respond_content_with_mode(
                    state,
                    interaction,
                    mode,
                    "Only authorized moderators can import hashes.",
                )
                .await?;
                return Ok(true);
            }
            let summary = match import_hashes(state, interaction, modal).await {
                Ok(summary) => summary,
                Err(source) => {
                    respond_modal_error_with_mode(state, interaction, mode, source).await?;
                    return Ok(true);
                }
            };
            respond_content_with_mode(state, interaction, mode, &summary).await?;
            Ok(true)
        }
        "specimen:import-images-modal:v1" => {
            if !can_moderate(state, interaction) {
                respond_content_with_mode(
                    state,
                    interaction,
                    mode,
                    "Only authorized moderators can upload specimen images.",
                )
                .await?;
                return Ok(true);
            }
            let summary = match import_uploaded_images(state, interaction, modal).await {
                Ok(summary) => summary,
                Err(source) => {
                    respond_modal_error_with_mode(state, interaction, mode, source).await?;
                    return Ok(true);
                }
            };
            respond_content_with_mode(state, interaction, mode, &summary).await?;
            Ok(true)
        }
        id if id.starts_with(ADVANCED_CONFIG_MODAL_PREFIX) => {
            if !can_configure(state, interaction) {
                respond_content_with_mode(
                    state,
                    interaction,
                    mode,
                    "Only authorized config users can change settings.",
                )
                .await?;
                return Ok(true);
            }
            if let Err(source) = apply_advanced_modal(state, interaction, modal).await {
                respond_modal_error_with_mode(state, interaction, mode, source).await?;
                return Ok(true);
            }
            respond_json_with_mode(state, interaction, mode, config_panel(state, false)).await?;
            Ok(true)
        }
        id if id.starts_with(LOG_MESSAGE_MODAL_PREFIX) => {
            if !can_configure(state, interaction) {
                respond_content_with_mode(
                    state,
                    interaction,
                    mode,
                    "Only authorized config users can change settings.",
                )
                .await?;
                return Ok(true);
            }
            if let Err(source) = apply_log_message_modal(state, interaction, modal).await {
                respond_modal_error_with_mode(state, interaction, mode, source).await?;
                return Ok(true);
            }
            respond_json_with_mode(state, interaction, mode, config_panel(state, false)).await?;
            Ok(true)
        }
        id if id.starts_with("config:") => {
            respond_content_with_mode(
                state,
                interaction,
                mode,
                "This config modal is stale or unsupported. Reopen `/config` and try again.",
            )
            .await?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

pub fn can_moderate(state: &AppState, interaction: &InteractionCreate) -> bool {
    if !state.guild_configured.load(Ordering::Acquire) {
        return false;
    }

    let permissions = member_permissions(interaction);
    if permissions.contains(Permissions::ADMINISTRATOR) {
        return true;
    }

    let config = state.active_config();
    has_any_role(&member_role_ids(interaction), &config.moderator_role_ids)
}

fn can_configure(state: &AppState, interaction: &InteractionCreate) -> bool {
    let permissions = member_permissions(interaction);
    if permissions.contains(Permissions::ADMINISTRATOR) {
        return true;
    }

    if !state.guild_configured.load(Ordering::Acquire) {
        return permissions.contains(Permissions::MANAGE_GUILD);
    }

    let config = state.active_config();
    has_any_role(&member_role_ids(interaction), &config.moderator_role_ids)
}

async fn respond_content(
    state: &AppState,
    interaction: &InteractionCreate,
    content: &str,
) -> Result<()> {
    respond_content_with_mode(state, interaction, InteractionAckMode::Initial, content).await
}

async fn respond_content_with_mode(
    state: &AppState,
    interaction: &InteractionCreate,
    mode: InteractionAckMode,
    content: &str,
) -> Result<()> {
    respond_interaction_response(
        state,
        interaction,
        mode,
        message_response(
            state,
            InteractionResponseType::ChannelMessageWithSource,
            Some(content.to_owned()),
            None,
            None,
        ),
    )
    .await
}

async fn respond_modal_error_with_mode(
    state: &AppState,
    interaction: &InteractionCreate,
    mode: InteractionAckMode,
    source: anyhow::Error,
) -> Result<()> {
    respond_content_with_mode(
        state,
        interaction,
        mode,
        &format!("Validation failed: {source}"),
    )
    .await
}

async fn respond_config_error_with_mode(
    state: &AppState,
    interaction: &InteractionCreate,
    mode: InteractionAckMode,
    source: anyhow::Error,
) -> Result<()> {
    respond_interaction_response(
        state,
        interaction,
        mode,
        message_response(
            state,
            InteractionResponseType::UpdateMessage,
            Some(format!("Validation failed: {source}")),
            None,
            None,
        ),
    )
    .await
}

async fn respond_interaction_response(
    state: &AppState,
    interaction: &InteractionCreate,
    mode: InteractionAckMode,
    response: AdminResponse,
) -> Result<()> {
    match mode {
        InteractionAckMode::Initial => {
            respond_interaction(
                &state.discord,
                state.application_id,
                interaction.id,
                &interaction.token,
                &response,
            )
            .await
        }
        InteractionAckMode::Deferred => {
            let data = response
                .data
                .as_ref()
                .ok_or_else(|| anyhow!("deferred interaction response has no data"))?;
            edit_interaction_response_data(
                &state.discord,
                state.application_id,
                &interaction.token,
                data,
            )
            .await
        }
    }
}

async fn respond_json_with_mode(
    state: &AppState,
    interaction: &InteractionCreate,
    mode: InteractionAckMode,
    response: AdminResponse,
) -> Result<()> {
    respond_interaction_response(state, interaction, mode, response).await
}

fn message_response(
    state: &AppState,
    kind: InteractionResponseType,
    mut content: Option<String>,
    mut embeds: Option<Vec<Embed>>,
    components: Option<Vec<Component>>,
) -> AdminResponse {
    if let Some(embeds) = embeds.as_mut() {
        stamp_interaction_embeds(embeds, &state.bot.bot_start_id);
    } else if let Some(content) = content.as_mut() {
        let _ = write!(content, "\n\nSightline start `{}`", state.bot.bot_start_id);
    }

    InteractionResponse {
        kind,
        data: Some(InteractionResponseData {
            allowed_mentions: Some(no_mentions()),
            content,
            embeds,
            components,
            flags: Some(MessageFlags::EPHEMERAL),
            ..InteractionResponseData::default()
        }),
    }
}

fn modal_response(
    state: &AppState,
    custom_id: impl Into<String>,
    title: impl Into<String>,
    components: Vec<Component>,
) -> AdminResponse {
    InteractionResponse {
        kind: InteractionResponseType::Modal,
        data: Some(InteractionResponseData {
            custom_id: Some(custom_id.into()),
            title: Some(modal_title(&title.into(), &state.bot.bot_start_id)),
            components: Some(components),
            ..InteractionResponseData::default()
        }),
    }
}

fn no_mentions() -> AllowedMentions {
    AllowedMentions {
        parse: Vec::new(),
        replied_user: false,
        roles: Vec::new(),
        users: Vec::new(),
    }
}

fn action_row(components: Vec<Component>) -> Component {
    Component::ActionRow(ActionRow {
        id: None,
        components,
    })
}

fn button(label: &str, custom_id: &str, style: ButtonStyle) -> Component {
    Component::Button(Button {
        id: None,
        custom_id: Some(custom_id.to_owned()),
        disabled: false,
        emoji: None,
        label: Some(label.to_owned()),
        style,
        url: None,
        sku_id: None,
    })
}

fn channel_select(
    custom_id: &str,
    placeholder: &str,
    min_values: u8,
    max_values: u8,
    defaults: Vec<SelectDefaultValue>,
) -> Component {
    Component::SelectMenu(SelectMenu {
        id: None,
        channel_types: Some(vec![ChannelType::GuildText, ChannelType::GuildAnnouncement]),
        custom_id: custom_id.to_owned(),
        default_values: (!defaults.is_empty()).then_some(defaults),
        disabled: false,
        kind: SelectMenuType::Channel,
        max_values: Some(max_values),
        min_values: Some(min_values),
        options: None,
        placeholder: Some(placeholder.to_owned()),
        required: None,
    })
}

fn role_select(
    custom_id: &str,
    placeholder: &str,
    min_values: u8,
    max_values: u8,
    defaults: Vec<SelectDefaultValue>,
) -> Component {
    Component::SelectMenu(SelectMenu {
        id: None,
        channel_types: None,
        custom_id: custom_id.to_owned(),
        default_values: (!defaults.is_empty()).then_some(defaults),
        disabled: false,
        kind: SelectMenuType::Role,
        max_values: Some(max_values),
        min_values: Some(min_values),
        options: None,
        placeholder: Some(placeholder.to_owned()),
        required: None,
    })
}

fn text_select(
    custom_id: &str,
    placeholder: &str,
    min_values: u8,
    max_values: u8,
    options: Vec<SelectMenuOption>,
) -> Component {
    Component::SelectMenu(SelectMenu {
        id: None,
        channel_types: None,
        custom_id: custom_id.to_owned(),
        default_values: None,
        disabled: false,
        kind: SelectMenuType::Text,
        max_values: Some(max_values),
        min_values: Some(min_values),
        options: Some(options),
        placeholder: Some(placeholder.to_owned()),
        required: None,
    })
}

fn text_input(
    custom_id: &str,
    label: &str,
    value: &str,
    required: bool,
    style: TextInputStyle,
    max_length: u16,
    placeholder: Option<&str>,
) -> Component {
    #[allow(deprecated)]
    Component::TextInput(TextInput {
        id: None,
        custom_id: custom_id.to_owned(),
        label: Some(label.to_owned()),
        max_length: Some(max_length),
        min_length: Some(u16::from(required)),
        placeholder: placeholder.map(str::to_owned),
        required: Some(required),
        style,
        value: Some(value.to_owned()),
    })
}

fn modal_label(label: &str, description: Option<&str>, component: Component) -> Component {
    Component::Label(Label {
        id: None,
        label: label.to_owned(),
        description: description.map(str::to_owned),
        component: Box::new(component),
    })
}

fn rich_embed(
    state: &AppState,
    title: impl Into<String>,
    description: impl Into<String>,
    fields: Vec<EmbedField>,
) -> Embed {
    Embed {
        author: None,
        color: Some(5_793_266),
        description: Some(description.into()),
        fields,
        footer: Some(interaction_embed_footer(&state.bot.bot_start_id)),
        image: None,
        kind: "rich".to_owned(),
        provider: None,
        thumbnail: None,
        timestamp: None,
        title: Some(title.into()),
        url: None,
        video: None,
    }
}

fn stamp_interaction_embeds(embeds: &mut [Embed], bot_start_id: &str) {
    let footer = interaction_embed_footer(bot_start_id);
    for embed in embeds {
        embed.footer = Some(footer.clone());
    }
}

fn interaction_embed_footer(bot_start_id: &str) -> EmbedFooter {
    EmbedFooter {
        icon_url: None,
        proxy_icon_url: None,
        text: format!("Sightline start {bot_start_id}"),
    }
}

fn modal_title(title: &str, bot_start_id: &str) -> String {
    const MODAL_TITLE_LIMIT: usize = 45;
    let short_start = bot_start_id.chars().take(8).collect::<String>();
    let suffix = format!(" ({short_start})");
    if title.chars().count() + suffix.chars().count() <= MODAL_TITLE_LIMIT {
        return format!("{title}{suffix}");
    }
    let prefix_chars = MODAL_TITLE_LIMIT.saturating_sub(suffix.chars().count());
    let mut value = title.chars().take(prefix_chars).collect::<String>();
    value.push_str(&suffix);
    value
}

fn embed_field(name: impl Into<String>, value: impl Into<String>, inline: bool) -> EmbedField {
    EmbedField {
        name: name.into(),
        value: value.into(),
        inline,
    }
}

fn config_panel(state: &AppState, update: bool) -> AdminResponse {
    let config = state.active_config();
    let kind = if update {
        InteractionResponseType::UpdateMessage
    } else {
        InteractionResponseType::ChannelMessageWithSource
    };
    let runtime_status = format!(
        "enabled `{}`\nsafe mode `{}`\npermissions `{}`\nspecimens `{}`",
        config.enabled,
        state.safe_mode.load(Ordering::Acquire),
        state.permissions_ok.load(Ordering::Acquire),
        state.matcher_len()
    );
    let fields = vec![
        embed_field("Runtime", runtime_status, true),
        embed_field(
            "Channels",
            format!(
                "database {}\nlogs {}",
                channel_label(Some(&config.ledger_channel_id)),
                channel_label(config.bot_log_channel_id.as_deref())
            ),
            true,
        ),
        embed_field(
            "Roles",
            format!(
                "moderators `{}`\nscan-exempt `{}`\nverified {}",
                config.moderator_role_ids.len(),
                config.scan_exempt_role_ids.len(),
                role_label(config.verified_role_id.as_deref())
            ),
            true,
        ),
        embed_field(
            "Confirmed actions",
            actions_label(&config.detection_policy.confirmed.actions),
            true,
        ),
        embed_field(
            "Suspicious actions",
            actions_label(&config.detection_policy.suspicious.actions),
            true,
        ),
        embed_field(
            "Text gate",
            text_gate_policy_label(&config.text_gate_policy),
            true,
        ),
        embed_field("Scan policy", scan_policy_label(&config.scan_policy), false),
        embed_field(
            "Log copy",
            format!(
                "routine: {}\nconfirmed: {}\nsuspicious: {}\nbenign: {}",
                log_message_label(&config.discord_general_log_message),
                log_message_label(&config.discord_confirmed_log_message),
                log_message_label(&config.discord_suspicious_log_message),
                log_message_label(&config.discord_benign_log_message)
            ),
            false,
        ),
    ];
    let components = vec![
        action_row(vec![
            button(
                if config.enabled { "Disable" } else { "Enable" },
                "config:toggle-enabled:v1",
                if config.enabled {
                    ButtonStyle::Danger
                } else {
                    ButtonStyle::Success
                },
            ),
            button("Actions", "config:actions:v1", ButtonStyle::Secondary),
            button("Roles", "config:roles:v1", ButtonStyle::Secondary),
            button(
                "Log message",
                "config:log-message:v1",
                ButtonStyle::Secondary,
            ),
            button("Advanced", "config:advanced:v1", ButtonStyle::Secondary),
        ]),
        action_row(vec![channel_select(
            "config:bot-log-channel:v1",
            "Set bot log channel",
            1,
            1,
            channel_select_defaults(config.bot_log_channel_id.as_deref()),
        )]),
    ];

    message_response(
        state,
        kind,
        None,
        Some(vec![rich_embed(
            state,
            "Sightline Configuration",
            "Configure image moderation without editing local files.",
            fields,
        )]),
        Some(components),
    )
}

fn actions_panel(state: &AppState) -> AdminResponse {
    let config = state.active_config();
    message_response(
        state,
        InteractionResponseType::UpdateMessage,
        None,
        Some(vec![rich_embed(
            state,
            "Sightline Actions",
            "Pick the actions for confirmed and suspicious detections. Member actions run timeout, remove role, ban, kick. Thresholds, scan policy, text gate, timeout length, and ban message-delete period are in Advanced.",
            vec![
                embed_field(
                    "Confirmed match",
                    actions_label(&config.detection_policy.confirmed.actions),
                    false,
                ),
                embed_field(
                    "Suspicious match",
                    actions_label(&config.detection_policy.suspicious.actions),
                    false,
                ),
            ],
        )]),
        Some(vec![
            action_row(vec![text_select(
                "config:confirmed-actions:v1",
                "Actions for confirmed matches",
                0,
                6,
                action_options(&config.detection_policy.confirmed.actions),
            )]),
            action_row(vec![text_select(
                "config:suspicious-actions:v1",
                "Actions for suspicious matches",
                0,
                6,
                action_options(&config.detection_policy.suspicious.actions),
            )]),
            action_row(vec![text_select(
                "config:confirmed-timeout:v1",
                "Timeout duration for confirmed matches",
                1,
                1,
                timeout_options(
                    "Confirmed timeout",
                    config.detection_policy.confirmed.actions.timeout_seconds,
                ),
            )]),
            action_row(vec![text_select(
                "config:suspicious-timeout:v1",
                "Timeout duration for suspicious matches",
                1,
                1,
                timeout_options(
                    "Suspicious timeout",
                    config.detection_policy.suspicious.actions.timeout_seconds,
                ),
            )]),
            action_row(vec![button(
                "Back",
                "config:home:v1",
                ButtonStyle::Secondary,
            )]),
        ]),
    )
}

fn roles_panel(state: &AppState) -> AdminResponse {
    let config = state.active_config();
    message_response(
        state,
        InteractionResponseType::UpdateMessage,
        None,
        Some(vec![rich_embed(
            state,
            "Sightline Roles",
            "Pick role settings from Discord role lists.",
            vec![
                embed_field(
                    "Moderator roles",
                    role_list_label(&config.moderator_role_ids),
                    false,
                ),
                embed_field(
                    "Scan-exempt roles",
                    role_list_label(&config.scan_exempt_role_ids),
                    false,
                ),
                embed_field(
                    "Verified role",
                    role_label(config.verified_role_id.as_deref()),
                    true,
                ),
            ],
        )]),
        Some(vec![
            action_row(vec![role_select(
                "config:moderator-roles:v1",
                "Set roles allowed to configure and add/import specimens",
                0,
                10,
                role_select_defaults(&config.moderator_role_ids),
            )]),
            action_row(vec![role_select(
                "config:scan-exempt-roles:v1",
                "Set roles exempt from image scanning",
                0,
                10,
                role_select_defaults(&config.scan_exempt_role_ids),
            )]),
            action_row(vec![role_select(
                "config:verified-role:v1",
                "Set verified/member role removed by role-removal actions",
                0,
                1,
                role_select_default(config.verified_role_id.as_deref()),
            )]),
            action_row(vec![button(
                "Back",
                "config:home:v1",
                ButtonStyle::Secondary,
            )]),
        ]),
    )
}

fn specimen_audit_response(state: &AppState) -> AdminResponse {
    let records = state.matcher_records();
    let config = state.active_config();
    let exact_duplicates = exact_duplicate_groups(&records);
    let near_duplicates = near_duplicate_pairs(&records);
    let zero_hit_specimens = zero_hit_specimens(state, &records);
    let size_outliers = size_aspect_outliers(state, &config, &records);
    let summary = format!(
        "specimens `{}`\nexact duplicate groups `{}`\nnear-duplicate pairs `{}`\nzero hits since start `{}`\nsize/aspect outliers `{}`",
        records.len(),
        exact_duplicates.len(),
        near_duplicates.len(),
        zero_hit_specimens.len(),
        size_outliers.len()
    );

    message_response(
        state,
        InteractionResponseType::ChannelMessageWithSource,
        None,
        Some(vec![rich_embed(
            state,
            "Specimen Audit",
            "Process-local quality checks for the current guild specimen set.",
            vec![
                embed_field("Summary", summary, false),
                embed_field(
                    "Exact duplicates",
                    audit_lines_or_ok(
                        exact_duplicates
                            .iter()
                            .take(8)
                            .map(|group| format!("`{}`", group.join("`, `"))),
                    ),
                    false,
                ),
                embed_field(
                    "Near duplicates",
                    audit_lines_or_ok(near_duplicates.iter().take(8).map(|pair| {
                        format!(
                            "`{}` <-> `{}` pHash `{}` dHash `{}`",
                            pair.left, pair.right, pair.phash_distance, pair.dhash_distance
                        )
                    })),
                    false,
                ),
                embed_field(
                    "Zero-hit specimens",
                    audit_lines_or_ok(
                        zero_hit_specimens
                            .iter()
                            .take(12)
                            .map(|record| specimen_audit_reference(state, record)),
                    ),
                    false,
                ),
                embed_field(
                    "Size/aspect outliers",
                    audit_lines_or_ok(size_outliers.iter().take(10).cloned()),
                    false,
                ),
                embed_field(
                    "Hard-negative checks",
                    "Hard negatives are local validation data only and are not stored in Discord. Use the local validation scripts before promoting large specimen batches.",
                    false,
                ),
                embed_field(
                    "OCR policy",
                    "OCR is reserved for specimen-based suspicious matches or visual-signal-only matches with at least three visual signals. Logs include the OCR reason and whether OCR promoted or cleared the result.",
                    false,
                ),
            ],
        )]),
        None,
    )
}

fn advanced_modal(state: &AppState, interaction: &InteractionCreate) -> Result<AdminResponse> {
    let config = normalized_config_for_modal_scope(&state.active_config());
    let user_id = interaction
        .author_id()
        .ok_or_else(|| anyhow!("interaction author missing"))?;
    let scope = config_modal_scope(state.guild_id(), user_id, &state.bot.bot_start_id, &config);
    let chunks = advanced_config_chunks(&config);
    Ok(modal_response(
        state,
        format!("{ADVANCED_CONFIG_MODAL_PREFIX}{scope}"),
        "Advanced Guild Config",
        chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                action_row(vec![text_input(
                    &format!("guild_config_{index}:{scope}"),
                    &format!("Guild config TOML {}/{}", index + 1, chunks.len()),
                    chunk,
                    false,
                    TextInputStyle::Paragraph,
                    4000,
                    Some("Leave every chunk empty to reset policy defaults from the VM TOML."),
                )])
            })
            .collect(),
    ))
}

fn advanced_config_chunks(config: &GuildConfig) -> Vec<String> {
    let pretty = advanced_config_toml(config, true).unwrap_or_else(|_| String::new());
    let compact = advanced_config_toml(config, false).unwrap_or_else(|_| pretty.clone());
    let mut chunks = split_advanced_config(&pretty);
    if chunks.len() > ADVANCED_CONFIG_MAX_CHUNKS {
        chunks = split_advanced_config(&compact);
    }
    if chunks.len() > ADVANCED_CONFIG_MAX_CHUNKS {
        chunks = vec![format!(
            "# Guild config is too large for a Discord modal.\n# Use import/export tooling or reduce list sizes.\n# Current serialized length: {} characters.\n",
            compact.chars().count()
        )];
    }
    chunks
}

fn advanced_config_toml(config: &GuildConfig, pretty: bool) -> Result<String> {
    let mut value = toml::Value::try_from(config).context("serializing guild config to TOML")?;
    if let Some(table) = value.as_table_mut() {
        table.remove("discord_detection_log_message");
    }
    remove_legacy_cluster_promote_key(&mut value);
    if pretty {
        toml::to_string_pretty(&value).context("serializing advanced config TOML")
    } else {
        toml::to_string(&value).context("serializing compact advanced config TOML")
    }
}

fn remove_legacy_cluster_promote_key(value: &mut toml::Value) {
    match value {
        toml::Value::Table(table) => {
            table.remove("cluster_promote_to_confirmed");
            for (_, value) in table.iter_mut() {
                remove_legacy_cluster_promote_key(value);
            }
        }
        toml::Value::Array(values) => {
            for value in values {
                remove_legacy_cluster_promote_key(value);
            }
        }
        toml::Value::String(_)
        | toml::Value::Integer(_)
        | toml::Value::Float(_)
        | toml::Value::Boolean(_)
        | toml::Value::Datetime(_) => {}
    }
}

fn log_message_modal(state: &AppState, interaction: &InteractionCreate) -> Result<AdminResponse> {
    let config = state.active_config();
    let user_id = interaction
        .author_id()
        .ok_or_else(|| anyhow!("interaction author missing"))?;
    let scope = config_modal_scope(state.guild_id(), user_id, &state.bot.bot_start_id, &config);
    Ok(modal_response(
        state,
        format!("{LOG_MESSAGE_MODAL_PREFIX}{scope}"),
        "Discord Log Messages",
        vec![
            action_row(vec![text_input(
                &format!("discord_general_log_message:{scope}"),
                "Routine log message content",
                &config.discord_general_log_message,
                false,
                TextInputStyle::Paragraph,
                1900,
                Some("Optional plain text for non-detection bot logs."),
            )]),
            action_row(vec![text_input(
                &format!("discord_confirmed_log_message:{scope}"),
                "Confirmed match log content",
                &config.discord_confirmed_log_message,
                false,
                TextInputStyle::Paragraph,
                1900,
                Some("Optional ping/copy for hard scam image matches."),
            )]),
            action_row(vec![text_input(
                &format!("discord_suspicious_log_message:{scope}"),
                "Suspicious match log content",
                &config.discord_suspicious_log_message,
                false,
                TextInputStyle::Paragraph,
                1900,
                Some("Optional ping/copy for suspicious image logs."),
            )]),
            action_row(vec![text_input(
                &format!("discord_benign_log_message:{scope}"),
                "Benign match log content",
                &config.discord_benign_log_message,
                false,
                TextInputStyle::Paragraph,
                1900,
                Some("Optional ping/copy for OCR-cleared benign image logs."),
            )]),
        ],
    ))
}

fn import_modal(state: &AppState) -> AdminResponse {
    modal_response(
        state,
        "specimen:import-hashes-modal:v1",
        "Import Image Hashes",
        vec![action_row(vec![text_input(
            "hash_records",
            "JSON or JSONL hash records",
            "",
            true,
            TextInputStyle::Paragraph,
            4000,
            Some("{\"schema\":7,\"source_path\":\"image.png\",\"fingerprint\":{...}}"),
        )])],
    )
}

fn upload_images_modal(state: &AppState) -> AdminResponse {
    modal_response(
        state,
        "specimen:import-images-modal:v1",
        "Upload Specimen Images",
        vec![modal_label(
            "Images",
            Some(
                "Upload raw specimen images. They will be processed with the production image pipeline.",
            ),
            Component::FileUpload(FileUpload {
                id: None,
                custom_id: "specimen_images".to_owned(),
                max_values: Some(
                    u8::try_from(MAX_MANUAL_SPECIMEN_IMAGES_PER_INTERACTION).unwrap_or(10),
                ),
                min_values: Some(1),
                required: Some(true),
            }),
        )],
    )
}

fn channel_select_defaults(channel_id: Option<&str>) -> Vec<SelectDefaultValue> {
    channel_id
        .filter(|id| !id.trim().is_empty())
        .and_then(|id| id.parse::<u64>().ok())
        .map(|id| vec![SelectDefaultValue::Channel(Id::<ChannelMarker>::new(id))])
        .unwrap_or_default()
}

fn role_select_defaults(role_ids: &[String]) -> Vec<SelectDefaultValue> {
    role_ids
        .iter()
        .filter_map(|id| id.parse::<u64>().ok())
        .map(|id| SelectDefaultValue::Role(Id::<RoleMarker>::new(id)))
        .collect()
}

fn role_select_default(role_id: Option<&str>) -> Vec<SelectDefaultValue> {
    role_id
        .and_then(|id| id.parse::<u64>().ok())
        .map(|id| vec![SelectDefaultValue::Role(Id::<RoleMarker>::new(id))])
        .unwrap_or_default()
}

fn split_advanced_config(value: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for line in value.lines() {
        let line_len = line.chars().count() + 1;
        if !current.is_empty()
            && current.chars().count().saturating_add(line_len) > ADVANCED_CONFIG_CHUNK_CHARS
        {
            chunks.push(current);
            current = String::new();
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.is_empty() {
        chunks.push(current);
    }

    if chunks.is_empty() {
        chunks.push(String::new());
    }
    chunks
}

fn action_options(actions: &DetectionActions) -> Vec<SelectMenuOption> {
    [
        (
            "delete_message",
            "Delete message",
            "Delete the offending message.",
            actions.delete_message,
        ),
        (
            "remove_user_roles",
            "Remove verified role",
            "Remove the configured verified/member role.",
            actions.remove_user_roles,
        ),
        (
            "timeout_user",
            "Timeout user",
            "Apply the configured timeout duration.",
            actions.timeout_user,
        ),
        (
            "ban_user",
            "Ban user",
            "Ban and delete messages for the configured period.",
            actions.ban_user,
        ),
        (
            "kick_user",
            "Kick user",
            "Remove the user from the server without banning them.",
            actions.kick_user,
        ),
        (
            "add_to_specimens",
            "Add image to specimens",
            "Store matched non-duplicate images as new specimens.",
            actions.add_to_specimens,
        ),
    ]
    .into_iter()
    .map(|(value, label, description, default)| SelectMenuOption {
        default,
        description: Some(description.to_owned()),
        emoji: None,
        label: label.to_owned(),
        value: value.to_owned(),
    })
    .collect()
}

fn timeout_options(prefix: &str, current_seconds: u32) -> Vec<SelectMenuOption> {
    TIMEOUT_DURATION_SECONDS
        .iter()
        .map(|seconds| SelectMenuOption {
            default: *seconds == current_seconds,
            description: Some(format!("{seconds} seconds")),
            emoji: None,
            label: format!("{prefix}: {}", timeout_duration_label(*seconds)),
            value: seconds.to_string(),
        })
        .collect()
}

async fn update_config_from_select(
    state: &AppState,
    interaction: &InteractionCreate,
    component: &MessageComponentInteractionData,
    custom_id: &str,
) -> Result<()> {
    let mut config = state.active_config();
    let values = component.values.clone();

    match custom_id {
        "config:bot-log-channel:v1" => config.bot_log_channel_id = values.first().cloned(),
        "config:verified-role:v1" => config.verified_role_id = values.first().cloned(),
        "config:moderator-roles:v1" => config.moderator_role_ids = values,
        "config:scan-exempt-roles:v1" => config.scan_exempt_role_ids = values,
        "config:confirmed-actions:v1" => {
            apply_action_values(&mut config.detection_policy.confirmed.actions, &values);
        }
        "config:suspicious-actions:v1" => {
            apply_action_values(&mut config.detection_policy.suspicious.actions, &values);
        }
        "config:confirmed-timeout:v1" => {
            config.detection_policy.confirmed.actions.timeout_seconds =
                parse_selected_timeout_seconds(values.first())?;
        }
        "config:suspicious-timeout:v1" => {
            config.detection_policy.suspicious.actions.timeout_seconds =
                parse_selected_timeout_seconds(values.first())?;
        }
        _ => return Ok(()),
    }

    persist_config(state, interaction, config).await
}

fn apply_action_values(actions: &mut DetectionActions, values: &[String]) {
    actions.delete_message = values.iter().any(|value| value == "delete_message");
    actions.remove_user_roles = values.iter().any(|value| value == "remove_user_roles");
    actions.timeout_user = values.iter().any(|value| value == "timeout_user");
    actions.ban_user = values.iter().any(|value| value == "ban_user");
    actions.kick_user = values.iter().any(|value| value == "kick_user");
    actions.add_to_specimens = values.iter().any(|value| value == "add_to_specimens");
}

fn parse_selected_timeout_seconds(value: Option<&String>) -> Result<u32> {
    let value = value.ok_or_else(|| anyhow!("timeout duration selection missing"))?;
    let seconds = value
        .parse::<u32>()
        .with_context(|| format!("invalid timeout duration: {value}"))?;
    if !TIMEOUT_DURATION_SECONDS.contains(&seconds) {
        return Err(anyhow!("invalid timeout duration: {seconds}"));
    }
    Ok(seconds)
}

fn normalized_config_for_modal_scope(config: &GuildConfig) -> GuildConfig {
    let mut normalized = config.clone();
    normalized.normalize();
    normalized
}

fn config_modal_scope(
    guild_id: Id<GuildMarker>,
    user_id: Id<UserMarker>,
    bot_start_id: &str,
    config: &GuildConfig,
) -> String {
    let config_toml = toml::to_string(config).unwrap_or_default();
    let config_hash = xxh3_64(config_toml.as_bytes());
    let process = bot_start_id.chars().take(8).collect::<String>();
    format!(
        "g{}:u{}:p{}:c{config_hash:016x}",
        guild_id.get(),
        user_id.get(),
        process
    )
}

fn validate_config_modal_scope(
    state: &AppState,
    interaction: &InteractionCreate,
    modal: &ModalInteractionData,
    prefix: &str,
) -> Result<String> {
    let user_id = interaction
        .author_id()
        .ok_or_else(|| anyhow!("interaction author missing"))?;
    let provided = modal
        .custom_id
        .strip_prefix(prefix)
        .ok_or_else(|| anyhow!("invalid config modal scope"))?;
    let current = normalized_config_for_modal_scope(&state.active_config());
    let expected = config_modal_scope(state.guild_id(), user_id, &state.bot.bot_start_id, &current);
    if provided != expected {
        return Err(anyhow!(
            "this config modal is stale or belongs to another guild/user; reopen `/config` and try again"
        ));
    }
    Ok(expected)
}

async fn apply_advanced_modal(
    state: &AppState,
    interaction: &InteractionCreate,
    modal: &ModalInteractionData,
) -> Result<()> {
    let scope =
        validate_config_modal_scope(state, interaction, modal, ADVANCED_CONFIG_MODAL_PREFIX)?;
    let fields = modal_fields(modal);
    let current = normalized_config_for_modal_scope(&state.active_config());
    let raw = advanced_config_modal_value(&fields, &scope)?;
    let config = if raw.trim().is_empty() {
        loaded_default_guild_config_for_state(state, &current)
    } else {
        parse_advanced_guild_config(&raw)?
    };

    persist_config(state, interaction, config).await
}

fn parse_advanced_guild_config(raw: &str) -> Result<GuildConfig> {
    let mut config: GuildConfig = toml::from_str(raw).context("parsing advanced policy TOML")?;
    config.normalize();
    Ok(config)
}

async fn apply_log_message_modal(
    state: &AppState,
    interaction: &InteractionCreate,
    modal: &ModalInteractionData,
) -> Result<()> {
    let scope = validate_config_modal_scope(state, interaction, modal, LOG_MESSAGE_MODAL_PREFIX)?;
    let fields = modal_fields(modal);
    let mut config = state.active_config();
    modal_scoped_field(&fields, "discord_general_log_message", &scope)
        .ok_or_else(|| anyhow!("routine log message field missing from scoped modal"))?
        .trim()
        .clone_into(&mut config.discord_general_log_message);
    modal_scoped_field(&fields, "discord_confirmed_log_message", &scope)
        .ok_or_else(|| anyhow!("confirmed log message field missing from scoped modal"))?
        .trim()
        .clone_into(&mut config.discord_confirmed_log_message);
    modal_scoped_field(&fields, "discord_suspicious_log_message", &scope)
        .ok_or_else(|| anyhow!("suspicious log message field missing from scoped modal"))?
        .trim()
        .clone_into(&mut config.discord_suspicious_log_message);
    modal_scoped_field(&fields, "discord_benign_log_message", &scope)
        .ok_or_else(|| anyhow!("benign log message field missing from scoped modal"))?
        .trim()
        .clone_into(&mut config.discord_benign_log_message);
    config.discord_detection_log_message.clear();
    persist_config(state, interaction, config).await
}

fn modal_scoped_field<'a>(
    fields: &'a HashMap<String, String>,
    base_name: &str,
    scope: &str,
) -> Option<&'a String> {
    fields.get(&format!("{base_name}:{scope}"))
}

fn advanced_config_modal_value(fields: &HashMap<String, String>, scope: &str) -> Result<String> {
    let chunks = (0..ADVANCED_CONFIG_MAX_CHUNKS)
        .filter_map(|index| modal_scoped_field(fields, &format!("guild_config_{index}"), scope))
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join("\n");
    if !fields
        .keys()
        .any(|key| key.starts_with("guild_config_") && key.ends_with(scope))
    {
        return Err(anyhow!("advanced config fields missing from scoped modal"));
    }
    Ok(chunks)
}

fn loaded_default_guild_config_for_state(state: &AppState, current: &GuildConfig) -> GuildConfig {
    let storage_channel_id = state.storage.read().channel_id;
    let mut defaults = GuildConfig::from_loaded_defaults(
        state.guild_id(),
        storage_channel_id,
        &state.config.matching,
        &state.config.default_scan_policy(),
        &state.config.text_gate,
    );

    defaults.enabled = current.enabled;
    defaults
        .bot_log_channel_id
        .clone_from(&current.bot_log_channel_id);
    defaults
        .discord_general_log_message
        .clone_from(&current.discord_general_log_message);
    defaults
        .discord_confirmed_log_message
        .clone_from(&current.discord_confirmed_log_message);
    defaults
        .discord_suspicious_log_message
        .clone_from(&current.discord_suspicious_log_message);
    defaults
        .discord_benign_log_message
        .clone_from(&current.discord_benign_log_message);
    defaults
        .verified_role_id
        .clone_from(&current.verified_role_id);
    defaults
        .moderator_role_ids
        .clone_from(&current.moderator_role_ids);
    defaults
        .scan_exempt_role_ids
        .clone_from(&current.scan_exempt_role_ids);
    defaults.updated_at.clone_from(&current.updated_at);
    defaults.updated_by_id.clone_from(&current.updated_by_id);
    defaults
}

async fn persist_config(
    state: &AppState,
    interaction: &InteractionCreate,
    mut config: GuildConfig,
) -> Result<()> {
    let storage_channel_id = state.storage.read().channel_id;
    config.ledger_channel_id = storage_channel_id.get().to_string();
    config.normalize();
    validate_separate_channels(&config)?;
    config.validate(state.guild_id())?;
    let user_id = interaction
        .author_id()
        .ok_or_else(|| anyhow!("interaction author missing"))?;
    config.touch(user_id);
    let record =
        GuildConfigRecord::new(config.clone()).sign(&state.secrets.specimen_hmac_secret)?;

    if let Err(source) = state.upsert_config_record(record).await {
        state.safe_mode.store(true, Ordering::Release);
        state.permissions_ok.store(false, Ordering::Release);
        tracing::warn!(
            event = "config.persist_failed",
            guild_id = state.guild_id().get(),
            updated_by = user_id.get(),
            ?source,
            "failed to persist guild config; old runtime config left active and guild forced into safe mode"
        );
        state
            .post_bot_log(format!(
                "Config update by <@{}> could not be written to `sightline-db`; Sightline entered safe mode for this guild and kept the previous runtime config.",
                user_id.get()
            ))
            .await;
        return Err(source.context("persisting guild config"));
    }

    state.guild_config.store(Arc::new(config.clone()));
    state.detection_policy_hash.store(
        config.detection_cache_policy_hash(),
        std::sync::atomic::Ordering::Release,
    );
    if let Err(source) = state
        .refresh_matcher_policy(config.detection_policy.clone())
        .await
    {
        tracing::warn!(
            event = "matcher.policy_refresh_failed",
            guild_id = state.guild_id().get(),
            ?source,
            "failed to refresh matcher coherence policy after config update"
        );
    }
    state
        .scan_exempt_roles
        .store(Arc::new(config.parsed_scan_exempt_role_ids()));
    state.hash_outcome_cache.lock().clear();
    state.clear_hash_processing();
    state.ocr_singleflight.clear();
    state.guild_configured.store(true, Ordering::Release);
    state.safe_mode.store(false, Ordering::Release);
    let permission_report =
        match check_bot_permissions(&state.discord, state.guild_id(), state.bot_user_id, &config)
            .await
        {
            Ok(report) => {
                state.permissions_ok.store(report.ok, Ordering::Release);
                report
            }
            Err(source) => {
                tracing::warn!(
                    event = "permissions.check_failed",
                    guild_id = state.guild_id().get(),
                    ?source,
                    "failed to verify bot permissions after config update"
                );
                state.permissions_ok.store(false, Ordering::Release);
                return Err(source.context(
                    "config was saved, but the automatic post-save doctor permission check failed",
                ));
            }
        };
    let permissions_ok = permission_report.ok;

    info!(
        event = "config.updated",
        guild_id = config.guild_id,
        updated_by = user_id.get(),
        enabled = config.enabled,
        permissions_ok,
        "guild config updated and persisted"
    );
    if !permissions_ok {
        return Err(anyhow!(
            "config was saved, but automatic doctor found blocking permission issues: {}",
            permission_report.missing_summary()
        ));
    }
    Ok(())
}

async fn import_hashes(
    state: &AppState,
    interaction: &InteractionCreate,
    modal: &ModalInteractionData,
) -> Result<String> {
    if !state.guild_accepts_specimen_writes() {
        return Err(anyhow!(
            "guild is not configured, permission checks failed, or sightline-db is unavailable"
        ));
    }

    let fields = modal_fields(modal);
    let raw = fields
        .get("hash_records")
        .ok_or_else(|| anyhow!("hash_records field missing"))?;
    let records = parse_import_records(raw)?;
    let user_id = interaction
        .author_id()
        .ok_or_else(|| anyhow!("interaction author missing"))?;
    let channel_id = interaction
        .channel
        .as_ref()
        .map(|channel| channel.id)
        .ok_or_else(|| anyhow!("interaction channel missing"))?;
    let message_id = Id::<MessageMarker>::new(interaction.id.get());
    let mut added = 0usize;
    let mut exact_duplicates = 0usize;
    let mut failed = 0usize;
    for fingerprint in records {
        if state.contains_specimen_xxh128(&fingerprint.byte_xxh128) {
            exact_duplicates += 1;
            continue;
        }

        let record = SpecimenRecord::new_add(
            state.guild_id(),
            channel_id,
            message_id,
            user_id,
            user_id,
            fingerprint,
            None,
        )
        .sign(&state.secrets.specimen_hmac_secret)?;

        match state
            .write_specimen_record(record, Vec::new(), SpecimenWriteLogContext::default())
            .await
        {
            Ok(SpecimenWriteOutcome::Added(_)) => added += 1,
            Ok(SpecimenWriteOutcome::Duplicate) => exact_duplicates += 1,
            Err(error) => {
                failed += 1;
                tracing::warn!(
                    event = "specimen.hash_import_ledger_write_failed",
                    guild_id = state.guild_id().get(),
                    ?error,
                    "failed to write imported hash ledger record"
                );
            }
        }
    }

    info!(
        event = "specimen.imported",
        added,
        exact_duplicates,
        failed,
        moderator_id = user_id.get(),
        "hash import completed"
    );
    state
        .post_bot_log(format!(
            "Specimen import: moderator <@{}> (`{}`) imported `{}` record(s). Exact duplicates rejected: `{}`. Failed: `{}`. Target: guild `{}` ledger channel <#{}>.",
            user_id.get(),
            user_id.get(),
            added,
            exact_duplicates,
            failed,
            state.guild_id().get(),
            state.storage.read().channel_id.get()
        ))
        .await;

    Ok(format!(
        "Imported {added} specimen hash record(s). Exact duplicates rejected: {exact_duplicates}. Failed: {failed}."
    ))
}

async fn import_uploaded_images(
    state: &AppState,
    interaction: &InteractionCreate,
    modal: &ModalInteractionData,
) -> Result<String> {
    if !state.guild_accepts_specimen_writes() {
        return Err(anyhow!(
            "guild is not configured, permission checks failed, or sightline-db is unavailable"
        ));
    }

    let user_id = interaction
        .author_id()
        .ok_or_else(|| anyhow!("interaction author missing"))?;
    let channel_id = interaction
        .channel
        .as_ref()
        .map(|channel| channel.id)
        .ok_or_else(|| anyhow!("interaction channel missing"))?;
    let attachment_ids = modal_file_upload_ids(modal, "specimen_images");
    if attachment_ids.is_empty() {
        return Err(anyhow!("no uploaded image attachments were received"));
    }

    let attachments = modal
        .resolved
        .as_ref()
        .map(|resolved| &resolved.attachments)
        .ok_or_else(|| anyhow!("uploaded attachment metadata missing"))?;
    let guild_config = state.active_config();
    let mut candidates = Vec::new();

    for attachment_id in attachment_ids
        .into_iter()
        .take(MAX_MANUAL_SPECIMEN_IMAGES_PER_INTERACTION)
    {
        let Some(attachment) = attachments.get(&attachment_id) else {
            continue;
        };
        let url = attachment.url.as_str();
        let size_bytes = Some(attachment.size);
        if size_bytes.is_some_and(|size| size > guild_config.scan_policy.max_file_bytes) {
            continue;
        }

        let mime_hint = attachment.content_type.clone();
        let has_dimensions = attachment.width.is_some() && attachment.height.is_some();
        let is_image = mime_hint.as_deref().is_some_and(mime_starts_with_image) || has_dimensions;
        if !is_image {
            continue;
        }

        candidates.push(ImageCandidate {
            guild_id: state.guild_id(),
            channel_id,
            message_id: Id::<MessageMarker>::new(interaction.id.get()),
            candidate_index: 0,
            candidates_in_message: 0,
            author_id: user_id,
            author_username: None,
            author_global_name: None,
            url: url.to_owned(),
            proxy_url: Some(attachment.proxy_url.clone()),
            kind: CandidateKind::Attachment,
            mime_hint,
            size_bytes,
            metadata_width: attachment.width.and_then(|value| u32::try_from(value).ok()),
            metadata_height: attachment
                .height
                .and_then(|value| u32::try_from(value).ok()),
            media_flags: attachment.flags.map(|flags| flags.bits()),
            verify_only: false,
            enqueued_at: None,
        });
    }

    if candidates.is_empty() {
        return Err(anyhow!(
            "no uploaded images passed the configured size and MIME checks"
        ));
    }
    let candidate_count = u16::try_from(candidates.len()).unwrap_or(u16::MAX);
    for (index, candidate) in candidates.iter_mut().enumerate() {
        candidate.candidate_index = u16::try_from(index.saturating_add(1)).unwrap_or(u16::MAX);
        candidate.candidates_in_message = candidate_count;
    }

    let summary =
        import_image_candidates(state, candidates, user_id, &guild_config, "modal_upload").await;
    state
        .post_bot_log(format!(
            "Specimen image import: moderator <@{}> (`{}`) imported `{}` image specimen(s). Exact duplicates rejected: `{}`. Failed: `{}`. Source: `modal_upload`. Target: guild `{}` ledger channel <#{}>.",
            user_id.get(),
            user_id.get(),
            summary.added,
            summary.exact_duplicates,
            summary.failed,
            state.guild_id().get(),
            state.storage.read().channel_id.get()
        ))
        .await;
    Ok(format!(
        "Imported {} uploaded image specimen(s). Exact duplicates rejected: {}. Failed: {}.",
        summary.added, summary.exact_duplicates, summary.failed
    ))
}

async fn export_hashes(state: &AppState, interaction: &InteractionCreate) -> Result<()> {
    const MAX_EXPORT_BYTES: usize = 8 * 1024 * 1024;

    let user_id = interaction
        .author_id()
        .ok_or_else(|| anyhow!("interaction author missing"))?;
    let records = state.matcher_records_snapshot().await?;
    let mut body = Vec::with_capacity(records.len().saturating_mul(1024).min(MAX_EXPORT_BYTES));
    for record in &records {
        let source_path = specimen_export_source_path(state.guild_id(), record);
        let exported =
            ExportedImageFingerprint::new(source_path, ImageFingerprint::from(record.clone()));
        serde_json::to_writer(&mut body, &exported).context("serializing exported specimen")?;
        body.push(b'\n');
        if body.len() > MAX_EXPORT_BYTES {
            return respond_content(
                state,
                interaction,
                "The specimen export is larger than the current Discord export limit. Use the local CLI to export in batches.",
            )
            .await;
        }
    }

    let filename = format!("sightline-specimens-{}.jsonl", state.guild_id().get());
    respond_interaction_attachment(
        &state.discord,
        state.application_id,
        interaction.id,
        &interaction.token,
        &format!(
            "Exported {} specimen fingerprint record(s) from guild `{}`.",
            records.len(),
            state.guild_id().get()
        ),
        &filename,
        body,
    )
    .await?;

    info!(
        event = "specimen.exported",
        count = records.len(),
        moderator_id = user_id.get(),
        "hash export completed"
    );
    state
        .post_bot_log(format!(
            "Specimen export: moderator <@{}> (`{}`) exported `{}` record(s) from guild `{}` ledger channel <#{}>.",
            user_id.get(),
            user_id.get(),
            records.len(),
            state.guild_id().get(),
            state.storage.read().channel_id.get()
        ))
        .await;

    Ok(())
}

fn specimen_export_source_path(
    guild_id: twilight_model::id::Id<twilight_model::id::marker::GuildMarker>,
    record: &SpecimenRecord,
) -> String {
    let link = record
        .source
        .channel_id
        .parse::<u64>()
        .ok()
        .zip(record.source.message_id.parse::<u64>().ok())
        .map_or_else(
            || "unknown-source".to_owned(),
            |(channel_id, message_id)| {
                message_jump_link(guild_id, Id::new(channel_id), Id::new(message_id))
            },
        );
    format!("{}#{}", link, record.specimen_id)
}

async fn doctor_response(
    state: &AppState,
    interaction: &InteractionCreate,
) -> Result<AdminResponse> {
    let config = state.active_config();
    let app_permissions = interaction
        .app_permissions
        .unwrap_or_else(Permissions::empty);
    let matcher_len = state.matcher_len();
    let safe_mode = state.safe_mode.load(Ordering::Acquire);
    let permission_report =
        match check_bot_permissions(&state.discord, state.guild_id(), state.bot_user_id, &config)
            .await
        {
            Ok(report) => {
                state.permissions_ok.store(report.ok, Ordering::Release);
                Ok(report)
            }
            Err(source) => {
                state.permissions_ok.store(false, Ordering::Release);
                Err(source)
            }
        };
    let permissions_ok = permission_report.as_ref().is_ok_and(|report| report.ok);

    let mut lines = vec![
        check_line(
            "Database/config channel configured",
            !config.ledger_channel_id.is_empty(),
        ),
        check_line(
            "Bot log channel configured",
            config.bot_log_channel_id.is_some(),
        ),
        check_line(
            "Bot log channel separate from database",
            config.bot_log_channel_id.as_deref() != Some(config.ledger_channel_id.as_str()),
        ),
        check_line(
            "Moderator roles configured",
            !config.moderator_role_ids.is_empty(),
        ),
        check_line(
            "Bot can respond in this channel",
            app_permissions.contains(Permissions::VIEW_CHANNEL),
        ),
        check_line(
            "Bot can send in this channel",
            app_permissions.contains(Permissions::SEND_MESSAGES),
        ),
        check_line("Ledger loaded without safe mode", !safe_mode),
        check_line("Required runtime permissions", permissions_ok),
        check_line("Matcher engine initialized", true),
        info_line(&format!("Specimens loaded: {matcher_len}")),
    ];
    match permission_report {
        Ok(report) => {
            lines.extend(
                report
                    .checks
                    .iter()
                    .map(|check| check_line(&check.label, check.ok)),
            );
        }
        Err(source) => {
            lines.push(check_line("Permission inspection completed", false));
            lines.push(info_line(&format!(
                "Permission inspection error: {source:#}"
            )));
        }
    }

    Ok(message_response(
        state,
        InteractionResponseType::ChannelMessageWithSource,
        Some(format!("Doctor results:\n{}", lines.join("\n"))),
        None,
        None,
    ))
}

async fn stats_response(state: &AppState) -> Result<AdminResponse> {
    let snapshot = state.metrics_snapshot().await?;
    Ok(message_response(
        state,
        InteractionResponseType::ChannelMessageWithSource,
        None,
        Some(vec![stats_embed(state, &snapshot)]),
        None,
    ))
}

fn stats_embed(state: &AppState, snapshot: &GuildMetricsSnapshot) -> Embed {
    let mut fields = vec![
        embed_field(
            "Quality - 6h",
            quality_metrics_block(&snapshot.period),
            false,
        ),
        embed_field(
            "Quality - total",
            quality_metrics_block(&snapshot.total),
            false,
        ),
    ];
    fields.extend(runtime_metrics_fields(
        "Runtime - 6h",
        snapshot.period_perf.as_ref(),
        &snapshot.period_timing,
    ));
    fields.extend(runtime_class_metrics_fields(
        "Runtime by class - 6h",
        &snapshot.period_timing_by_class,
    ));
    fields.extend(runtime_metrics_fields(
        "Runtime - total",
        snapshot.total_perf.as_ref(),
        &snapshot.total_timing,
    ));
    fields.extend(runtime_class_metrics_fields(
        "Runtime by class - total",
        &snapshot.total_timing_by_class,
    ));
    rich_embed(
        state,
        "Sightline Stats",
        "Current six-hour period and since-start pipeline health.",
        fields,
    )
}

fn runtime_class_metrics_fields(
    label: &str,
    classes: &[PipelineTimingClassSnapshot],
) -> Vec<EmbedField> {
    let lines = classes
        .iter()
        .filter(|class| {
            class
                .timing
                .total
                .as_ref()
                .is_some_and(|timing| timing.count > 0)
        })
        .map(|class| {
            format!(
                "{}: total {}; dl {}; fp {}; matcher {}",
                class.label,
                short_timing(class.timing.total.as_ref()),
                short_timing(class.timing.download.as_ref()),
                short_timing(class.timing.fingerprint.as_ref()),
                short_timing(class.timing.matcher.as_ref())
            )
        })
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return vec![embed_field(label, "n/a", false)];
    }
    chunk_stats_lines(label, &lines)
}

fn quality_metrics_block(metrics: &GuildMetricCounters) -> String {
    let decisions = metrics
        .passes
        .saturating_add(metrics.scan_failures)
        .saturating_add(metrics.hard_matches)
        .saturating_add(metrics.suspicious_matches);
    format!(
        "scanned `{}` decisioned `{}` pass `{}` failure `{}`\nconfirmed `{}` ({}) exact `{}` perceptual `{}` local `{}`\nsuspicious `{}` ({}) exact `{}` perceptual `{}` local `{}`\nOCR calls `{}` resolved good `{}` bad `{}` unknown `{}` bad-rate `{}`",
        metrics.images_scanned,
        decisions,
        metrics.passes,
        metrics.scan_failures,
        metrics.hard_matches,
        percent_label(metrics.hard_matches, decisions),
        metrics.hard_exact_xxh128,
        metrics.hard_perceptual,
        metrics.hard_local_anchors,
        metrics.suspicious_matches,
        percent_label(metrics.suspicious_matches, decisions),
        metrics.suspicious_exact_xxh128,
        metrics.suspicious_perceptual,
        metrics.suspicious_local_anchors,
        metrics.ocr_calls,
        metrics.ocr_resolved_good,
        metrics.ocr_resolved_bad,
        metrics.ocr_resolved_unknown,
        percent_label(metrics.ocr_resolved_bad, metrics.ocr_calls.max(1)),
    )
}

fn runtime_metrics_fields(
    label: &str,
    perf: Option<&crate::image::types::ImagePerfSnapshot>,
    timing: &PipelineTimingSnapshot,
) -> Vec<EmbedField> {
    let mut fields = vec![embed_field(
        format!("{label} overview"),
        [
            perf_summary_line(perf),
            format!(
                "preview used `{}` fallback `{}`",
                timing.preview_used, timing.preview_fallbacks
            ),
        ]
        .join("\n"),
        false,
    )];
    let mut lines = vec![
        timing_line("total", timing.total.as_ref()),
        timing_line("preview_dl", timing.preview_download.as_ref()),
        timing_line("preview_fp", timing.preview_fingerprint.as_ref()),
        timing_line("preview_match", timing.preview_matcher.as_ref()),
        timing_line("queue_wait", timing.queue_wait.as_ref()),
        timing_line("download", timing.download.as_ref()),
        timing_line("dl_request", timing.download_request.as_ref()),
        timing_line("dl_body", timing.download_body.as_ref()),
        timing_line("dl_gate", timing.download_gate_wait.as_ref()),
        timing_line("flag_cache", timing.flagged_cache_lookup.as_ref()),
        timing_line("exact", timing.exact_match_lookup.as_ref()),
        timing_line("singleflight_wait", timing.singleflight_wait.as_ref()),
        timing_line("fingerprint", timing.fingerprint.as_ref()),
    ];
    lines.extend(
        timing
            .fingerprint_pipeline
            .parts()
            .into_iter()
            .map(|(label, snapshot)| timing_line(label, snapshot)),
    );
    lines.extend([
        timing_line("matcher", timing.matcher.as_ref()),
        timing_line("ocr_crop", timing.ocr_crop.as_ref()),
        timing_line("progressive", timing.progressive_eval.as_ref()),
    ]);
    fields.extend(chunk_stats_lines(&format!("{label} stages"), &lines));
    fields
}

fn chunk_stats_lines(label: &str, lines: &[String]) -> Vec<EmbedField> {
    const DISCORD_FIELD_SOFT_LIMIT: usize = 900;
    let mut fields = Vec::new();
    let mut value = String::new();
    let mut part = 1usize;
    for line in lines {
        let additional_len = line.len() + usize::from(!value.is_empty());
        if !value.is_empty()
            && value.len().saturating_add(additional_len) > DISCORD_FIELD_SOFT_LIMIT
        {
            fields.push(embed_field(
                numbered_stats_label(label, part),
                std::mem::take(&mut value),
                false,
            ));
            part += 1;
        }
        if !value.is_empty() {
            value.push('\n');
        }
        value.push_str(line);
    }
    if value.is_empty() {
        "no per-step timings yet".clone_into(&mut value);
    }
    fields.push(embed_field(numbered_stats_label(label, part), value, false));
    fields
}

fn numbered_stats_label(label: &str, part: usize) -> String {
    if part == 1 {
        label.to_owned()
    } else {
        format!("{label} {part}")
    }
}

fn perf_summary_line(perf: Option<&crate::image::types::ImagePerfSnapshot>) -> String {
    perf.map_or_else(
        || "image total `n/a`".to_owned(),
        |snapshot| {
            format!(
                "image total avg `{}` p95 `{}` p99 `{}` max `{}` n `{}`",
                ms_label_from_ms(snapshot.avg_ms),
                ms_label(snapshot.p95_ms),
                ms_label(snapshot.p99_ms),
                ms_label(snapshot.max_ms),
                snapshot.total_count,
            )
        },
    )
}

fn timing_line(name: &str, timing: Option<&TimingDistributionSnapshot>) -> String {
    timing.map_or_else(
        || format!("{name} `n/a`"),
        |snapshot| {
            format!(
                "{name} avg `{}` p95 `{}` p99 `{}` max `{}` n `{}`",
                us_label(snapshot.avg_us),
                us_label(snapshot.p95_us),
                us_label(snapshot.p99_us),
                us_label(snapshot.max_us),
                snapshot.count,
            )
        },
    )
}

fn short_timing(timing: Option<&TimingDistributionSnapshot>) -> String {
    timing.map_or_else(
        || "n/a".to_owned(),
        |snapshot| {
            format!(
                "avg {} p95 {} max {} n {}",
                us_label(snapshot.avg_us),
                us_label(snapshot.p95_us),
                us_label(snapshot.max_us),
                snapshot.count
            )
        },
    )
}

fn percent_label(numerator: u64, denominator: u64) -> String {
    if denominator == 0 {
        return "0.0%".to_owned();
    }
    let tenths = numerator.saturating_mul(1_000) / denominator;
    format!("{}.{:01}%", tenths / 10, tenths % 10)
}

fn ms_label(value: u128) -> String {
    format!("{value}ms")
}

fn ms_label_from_ms(value: f64) -> String {
    format!("{value:.1}ms")
}

fn us_label(value: u64) -> String {
    if value >= 1_000 {
        let tenths_ms = value / 100;
        format!("{}.{:01}ms", tenths_ms / 10, tenths_ms % 10)
    } else {
        format!("{value}us")
    }
}

fn parse_import_records(raw: &str) -> Result<Vec<ImageFingerprint>> {
    let trimmed = raw.trim();
    if trimmed.starts_with('[') {
        let values: Vec<ExportedImageFingerprint> = serde_json::from_str(trimmed)?;
        return values
            .into_iter()
            .map(exported_fingerprint_into_import)
            .collect();
    }

    if trimmed.starts_with('{') && !trimmed.lines().skip(1).any(|line| !line.trim().is_empty()) {
        let value: ExportedImageFingerprint = serde_json::from_str(trimmed)?;
        return Ok(vec![exported_fingerprint_into_import(value)?]);
    }

    trimmed
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let exported = serde_json::from_str::<ExportedImageFingerprint>(line)
                .context("parsing import line")?;
            exported.into_validated_fingerprint()
        })
        .collect()
}

fn mime_starts_with_image(mime: &str) -> bool {
    mime.get(..6)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("image/"))
}

fn exported_fingerprint_into_import(value: ExportedImageFingerprint) -> Result<ImageFingerprint> {
    value.into_validated_fingerprint()
}

fn modal_fields(modal: &ModalInteractionData) -> HashMap<String, String> {
    fn visit(component: &ModalInteractionComponent, fields: &mut HashMap<String, String>) {
        match component {
            ModalInteractionComponent::Label(label) => visit(&label.component, fields),
            ModalInteractionComponent::ActionRow(row) => {
                for component in &row.components {
                    visit(component, fields);
                }
            }
            ModalInteractionComponent::TextInput(input) => {
                fields.insert(input.custom_id.clone(), input.value.trim().to_owned());
            }
            _ => {}
        }
    }

    let mut fields = HashMap::new();
    for component in &modal.components {
        visit(component, &mut fields);
    }
    fields
}

fn modal_file_upload_ids(
    modal: &ModalInteractionData,
    expected_custom_id: &str,
) -> Vec<Id<twilight_model::id::marker::AttachmentMarker>> {
    fn visit(
        component: &ModalInteractionComponent,
        expected_custom_id: &str,
        ids: &mut Vec<Id<twilight_model::id::marker::AttachmentMarker>>,
    ) {
        match component {
            ModalInteractionComponent::Label(label) => {
                visit(&label.component, expected_custom_id, ids);
            }
            ModalInteractionComponent::ActionRow(row) => {
                for component in &row.components {
                    visit(component, expected_custom_id, ids);
                }
            }
            ModalInteractionComponent::FileUpload(upload)
                if upload.custom_id == expected_custom_id =>
            {
                for value in &upload.values {
                    if !ids.iter().any(|existing| existing == value) {
                        ids.push(*value);
                    }
                }
            }
            _ => {}
        }
    }

    let mut ids = Vec::new();
    for component in &modal.components {
        visit(component, expected_custom_id, &mut ids);
    }
    ids
}

fn member_permissions(interaction: &InteractionCreate) -> Permissions {
    interaction
        .member
        .as_ref()
        .and_then(|member| member.permissions)
        .unwrap_or_else(Permissions::empty)
}

fn member_role_ids(interaction: &InteractionCreate) -> Vec<String> {
    interaction
        .member
        .as_ref()
        .map(|member| {
            member
                .roles
                .iter()
                .map(|role_id| role_id.get().to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn validate_separate_channels(config: &GuildConfig) -> Result<()> {
    if config.bot_log_channel_id.is_none() {
        return Err(anyhow!("bot log channel is required"));
    }
    if config.bot_log_channel_id.as_deref() == Some(config.ledger_channel_id.as_str()) {
        return Err(anyhow!(
            "bot log channel must be separate from the database channel"
        ));
    }
    Ok(())
}

fn has_any_role(member_role_ids: &[String], allowed_role_ids: &[String]) -> bool {
    if allowed_role_ids.len() <= 8 {
        return allowed_role_ids.iter().any(|role_id| {
            member_role_ids
                .iter()
                .any(|member_role_id| member_role_id == role_id)
        });
    }
    let allowed_role_ids = allowed_role_ids
        .iter()
        .collect::<std::collections::HashSet<_>>();
    member_role_ids
        .iter()
        .any(|member_role_id| allowed_role_ids.contains(member_role_id))
}

fn channel_label(value: Option<&str>) -> String {
    value.map_or_else(|| "Not set".to_owned(), |id| format!("<#{id}>"))
}

fn role_label(value: Option<&str>) -> String {
    value.map_or_else(|| "Not set".to_owned(), |id| format!("<@&{id}>"))
}

fn role_list_label(values: &[String]) -> String {
    if values.is_empty() {
        "None".to_owned()
    } else {
        values
            .iter()
            .map(|id| format!("<@&{id}>"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn log_message_label(value: &str) -> String {
    if value.trim().is_empty() {
        "Not set".to_owned()
    } else {
        let preview = value.trim().replace('\n', " ");
        format!("Set: `{}`", truncate_label(&preview, 80))
    }
}

#[derive(Debug, Clone)]
struct NearDuplicatePair {
    left: String,
    right: String,
    phash_distance: u32,
    dhash_distance: u32,
}

fn exact_duplicate_groups(records: &[SpecimenRecord]) -> Vec<Vec<String>> {
    let mut by_hash = HashMap::<&str, Vec<String>>::new();
    for record in records {
        by_hash
            .entry(record.image.byte_xxh128.as_str())
            .or_default()
            .push(record.specimen_id.clone());
    }
    by_hash
        .into_values()
        .filter(|group| group.len() > 1)
        .collect()
}

fn near_duplicate_pairs(records: &[SpecimenRecord]) -> Vec<NearDuplicatePair> {
    const MAX_COMPARISONS: usize = 200_000;
    let mut pairs = Vec::new();
    let mut comparisons = 0usize;
    for (left_index, left) in records.iter().enumerate() {
        let Some(left_perceptual_hash) = parse_hash64(&left.image.phash64) else {
            continue;
        };
        let Some(left_difference_hash) = parse_hash64(&left.image.dhash64) else {
            continue;
        };
        for right in records.iter().skip(left_index + 1) {
            comparisons = comparisons.saturating_add(1);
            if comparisons > MAX_COMPARISONS {
                return pairs;
            }
            if left.image.byte_xxh128 == right.image.byte_xxh128 {
                continue;
            }
            let Some(right_perceptual_hash) = parse_hash64(&right.image.phash64) else {
                continue;
            };
            let Some(right_difference_hash) = parse_hash64(&right.image.dhash64) else {
                continue;
            };
            let phash_distance = (left_perceptual_hash ^ right_perceptual_hash).count_ones();
            let dhash_distance = (left_difference_hash ^ right_difference_hash).count_ones();
            if phash_distance <= 4 && dhash_distance <= 4 {
                pairs.push(NearDuplicatePair {
                    left: left.specimen_id.clone(),
                    right: right.specimen_id.clone(),
                    phash_distance,
                    dhash_distance,
                });
            }
        }
    }
    pairs
}

fn zero_hit_specimens<'a>(
    state: &AppState,
    records: &'a [SpecimenRecord],
) -> Vec<&'a SpecimenRecord> {
    records
        .iter()
        .filter(|record| state.specimen_hit_count(&record.specimen_id) == 0)
        .collect()
}

fn size_aspect_outliers(
    state: &AppState,
    config: &GuildConfig,
    records: &[SpecimenRecord],
) -> Vec<String> {
    let suspicious_max_aspect = config
        .detection_policy
        .suspicious
        .threshold
        .geometry_max_aspect_ratio;
    let confirmed_max_aspect = config
        .detection_policy
        .confirmed
        .threshold
        .geometry_max_aspect_ratio;
    let max_aspect = f64::from(suspicious_max_aspect.max(confirmed_max_aspect).max(1.0)) * 1.2;
    let min_aspect = 1.0 / max_aspect;
    let max_pixels = state.config.download.max_decoded_pixels;
    records
        .iter()
        .filter_map(|record| {
            let width = record.image.width.max(1);
            let height = record.image.height.max(1);
            let pixels = u64::from(width) * u64::from(height);
            let aspect = f64::from(width) / f64::from(height);
            let mut reasons = Vec::new();
            if pixels > max_pixels.saturating_mul(4) / 5 {
                reasons.push(format!("large `{pixels}` px"));
            }
            if aspect > max_aspect || aspect < min_aspect {
                reasons.push(format!("aspect `{aspect:.2}`"));
            }
            (!reasons.is_empty()).then(|| {
                format!(
                    "`{}` {}x{} {}",
                    record.specimen_id,
                    width,
                    height,
                    reasons.join(", ")
                )
            })
        })
        .collect()
}

fn specimen_audit_reference(state: &AppState, record: &SpecimenRecord) -> String {
    match state.specimen_ledger_link(&record.specimen_id) {
        Some(link) => format!("`{}` {}", record.specimen_id, link),
        None => format!("`{}`", record.specimen_id),
    }
}

fn audit_lines_or_ok(lines: impl Iterator<Item = String>) -> String {
    let lines = lines.collect::<Vec<_>>();
    if lines.is_empty() {
        return "No issues found.".to_owned();
    }
    truncate_label(&lines.join("\n"), 950)
}

fn parse_hash64(value: &str) -> Option<u64> {
    u64::from_str_radix(value, 16).ok()
}

fn truncate_label(value: &str, max_chars: usize) -> String {
    let mut output = String::new();
    for character in value.chars().take(max_chars) {
        output.push(character);
    }
    if value.chars().count() > max_chars {
        output.push_str("...");
    }
    output
}

fn text_gate_policy_label(policy: &TextGatePolicy) -> String {
    format!(
        "enabled `{}`\nkeywords `{}` / threshold `{}`\nsentences `{}`\nOCR typo distance `{}`",
        policy.enabled,
        policy.keywords.len(),
        policy.keyword_threshold,
        policy.sentences.len(),
        policy.keyword_max_distance
    )
}

fn scan_policy_label(policy: &ScanPolicy) -> String {
    format!(
        "extensions: `{}`; max file bytes: `{}`; exempt admins: `{}`",
        policy.allowed_extensions.join(","),
        policy.max_file_bytes,
        policy.exempt_administrators
    )
}

fn actions_label(actions: &DetectionActions) -> String {
    let mut labels = Vec::new();
    if actions.delete_message {
        labels.push("delete message".to_owned());
    }
    if actions.remove_user_roles {
        labels.push("remove roles".to_owned());
    }
    if actions.timeout_user {
        labels.push(format!(
            "timeout user for {}",
            timeout_duration_label(actions.timeout_seconds)
        ));
    }
    if actions.ban_user {
        labels.push(format!(
            "ban user, delete {}s messages",
            actions.ban_delete_message_seconds
        ));
    }
    if actions.kick_user {
        labels.push("kick user".to_owned());
    }
    if actions.add_to_specimens {
        labels.push("add to specimens".to_owned());
    }

    if labels.is_empty() {
        "No action".to_owned()
    } else {
        labels.join(", ")
    }
}

fn timeout_duration_label(seconds: u32) -> &'static str {
    match seconds {
        60 => "1m",
        300 => "5m",
        600 => "10m",
        3_600 => "1h",
        86_400 => "1 day",
        604_800 => "1 week",
        _ => "invalid duration",
    }
}

fn check_line(label: &str, ok: bool) -> String {
    format!("{} {label}", if ok { "[ok]" } else { "[fail]" })
}

fn info_line(label: &str) -> String {
    format!("[info] {label}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn optional_role_select_omits_modal_required_flag() {
        let component = role_select(
            "config:moderator-roles:v1",
            "Moderator roles",
            0,
            10,
            vec![],
        );
        let value = serde_json::to_value(component).expect("serialize role select");

        assert_eq!(value["type"], json!(6));
        assert_eq!(value["min_values"], json!(0));
        assert_eq!(value["max_values"], json!(10));
        assert!(value.get("required").is_none());
    }

    #[test]
    fn required_role_select_omits_required_flag() {
        let component = role_select("config:required-role:v1", "Required role", 1, 1, vec![]);
        let value = serde_json::to_value(component).expect("serialize role select");

        assert_eq!(value["type"], json!(6));
        assert_eq!(value["min_values"], json!(1));
        assert!(value.get("required").is_none());
    }

    #[test]
    fn optional_text_select_omits_modal_required_flag() {
        let component = text_select(
            "config:confirmed-actions:v1",
            "Actions",
            0,
            6,
            action_options(&DetectionActions::default()),
        );
        let value = serde_json::to_value(component).expect("serialize text select");

        assert_eq!(value["type"], json!(3));
        assert_eq!(value["min_values"], json!(0));
        assert!(value.get("required").is_none());
    }

    #[test]
    fn config_modal_scope_is_guild_and_user_specific() {
        let config =
            GuildConfig::from_guild(Id::<GuildMarker>::new(1), Id::<ChannelMarker>::new(2));
        let left = config_modal_scope(
            Id::<GuildMarker>::new(1),
            Id::<twilight_model::id::marker::UserMarker>::new(10),
            "0123456789abcdef",
            &config,
        );
        let different_guild = config_modal_scope(
            Id::<GuildMarker>::new(2),
            Id::<twilight_model::id::marker::UserMarker>::new(10),
            "0123456789abcdef",
            &config,
        );
        let different_user = config_modal_scope(
            Id::<GuildMarker>::new(1),
            Id::<twilight_model::id::marker::UserMarker>::new(11),
            "0123456789abcdef",
            &config,
        );

        assert_ne!(left, different_guild);
        assert_ne!(left, different_user);
        assert!(
            format!("{ADVANCED_CONFIG_MODAL_PREFIX}{left}")
                .chars()
                .count()
                <= 100
        );
        assert!(format!("{LOG_MESSAGE_MODAL_PREFIX}{left}").chars().count() <= 100);
        assert!(format!("guild_config_0:{left}").chars().count() <= 100);
    }

    #[test]
    fn advanced_config_modal_value_uses_scoped_fields_only() {
        let scope = "g1:u2:pabcdef01:c0123456789abcdef";
        let mut fields = HashMap::new();
        fields.insert(
            format!("guild_config_0:{scope}"),
            "enabled = true\n".to_owned(),
        );
        fields.insert(
            "guild_config_0:g9:u9:pabcdef01:c0123456789abcdef".to_owned(),
            "enabled = false\n".to_owned(),
        );

        assert_eq!(
            advanced_config_modal_value(&fields, scope).expect("scoped field exists"),
            "enabled = true\n"
        );
    }

    #[test]
    fn advanced_config_toml_reflects_config_panel_fields() {
        let mut config = GuildConfig::from_guild(
            Id::<twilight_model::id::marker::GuildMarker>::new(1),
            Id::<ChannelMarker>::new(2),
        );
        config.enabled = true;
        config.bot_log_channel_id = Some("3".to_owned());
        config.verified_role_id = Some("4".to_owned());
        config.moderator_role_ids = vec!["5".to_owned(), "6".to_owned()];
        config.scan_exempt_role_ids = vec!["7".to_owned()];
        config.discord_general_log_message = "general".to_owned();
        config.discord_confirmed_log_message = "confirmed".to_owned();
        config.discord_suspicious_log_message = "suspicious".to_owned();
        config.discord_benign_log_message = "benign".to_owned();
        config.discord_detection_log_message = "legacy".to_owned();
        config.scan_policy.exempt_administrators = false;
        config.detection_policy.confirmed.actions.delete_message = true;
        config.detection_policy.confirmed.actions.timeout_user = true;
        config.detection_policy.confirmed.actions.timeout_seconds = 300;
        config.detection_policy.suspicious.actions.add_to_specimens = true;

        let raw = advanced_config_chunks(&config).join("");
        assert!(!raw.starts_with("# Guild config is too large"));
        assert!(!raw.contains("discord_detection_log_message"));
        assert!(!raw.contains("cluster_promote_to_confirmed"));

        let parsed = parse_advanced_guild_config(&raw).expect("advanced config is valid TOML");
        assert!(parsed.enabled);
        assert_eq!(parsed.bot_log_channel_id.as_deref(), Some("3"));
        assert_eq!(parsed.verified_role_id.as_deref(), Some("4"));
        assert_eq!(parsed.moderator_role_ids, ["5", "6"]);
        assert_eq!(parsed.scan_exempt_role_ids, ["7"]);
        assert!(!parsed.scan_policy.exempt_administrators);
        assert_eq!(parsed.discord_general_log_message, "general");
        assert_eq!(parsed.discord_confirmed_log_message, "confirmed");
        assert_eq!(parsed.discord_suspicious_log_message, "suspicious");
        assert_eq!(parsed.discord_benign_log_message, "benign");
        assert!(parsed.detection_policy.confirmed.actions.delete_message);
        assert!(parsed.detection_policy.confirmed.actions.timeout_user);
        assert_eq!(
            parsed.detection_policy.confirmed.actions.timeout_seconds,
            300
        );
        assert!(parsed.detection_policy.suspicious.actions.add_to_specimens);
    }

    #[test]
    fn advanced_config_ignores_unknown_keys_and_migrates_legacy_copy() {
        let raw = r#"
version = 2
enabled = true
guild_id = "1"
ledger_channel_id = "2"
discord_general_log_message = ""
discord_detection_log_message = "<@&123>"
unknown_root_key = "ignored"
moderator_role_ids = []
scan_exempt_role_ids = []
updated_at = "2026-06-30T00:00:00Z"
updated_by_id = "9"

[scan_policy]
exempt_administrators = false
allowed_extensions = ["jpg"]
max_file_bytes = 1048576
unknown_scan_key = true

[detection_hyperparameters]
perceptual_orientation_correction = false
perceptual_orientation_max_degrees = 10.0
perceptual_orientation_step_degrees = 1.0
perceptual_orientation_min_gain = 1.02
local_anchors_enabled = true
local_max_width = 768
local_max_height = 2048
local_max_area = 786432
local_max_aspect_ratio = 8.0
local_tile_width = 64
local_tile_height = 32
local_stride = 12
local_tile_budget = 5000
local_hash_cap = 3000
local_anchor_count = 512
local_anchor_max_distance = 12

[detection_policy.confirmed.actions]
delete_message = false
remove_user_roles = false
timeout_user = false
timeout_seconds = 60
ban_user = false
ban_delete_message_seconds = 0
kick_user = false
add_to_specimens = false

[detection_policy.confirmed.threshold]
score_threshold = 63.0
perceptual_score_weight = 1.8
local_anchor_score_weight = 2.2
dense_local_anchor_score_weight = 1.4
visual_signature_score_weight = 0.0
visual_shape_score_weight = 1.0
visual_shape_score_cap = 10.0
perceptual_score_floor = 35.0
local_score_full_hits = 500
local_score_full_regions = 14
local_score_full_spread = 700.0
visual_shape_score_full = 300.0
cluster_coherence = true
cluster_hard_score = 63
cluster_chrome_ceiling_score = 19
cluster_member_score = 25
cluster_coherence_score = 63
cluster_min_size = 2
cluster_coverage_floor_permille = 0
exact_xxh128 = true
perceptual_hash = true
phash64_max_distance = 16
dhash64_max_distance = 12
perceptual_hash_max_total_distance = 26
perceptual_visual_support_distance_slack = 0
local_anchors = true
min_anchor_hits = 80
min_distinct_regions = 10
max_mean_distance = 7.0
local_unverified_support = false
local_unverified_support_min_anchor_hits = 0
local_unverified_support_min_distinct_regions = 0
local_unverified_support_min_retention_permille = 0
local_unverified_support_max_mean_distance = 0.0
local_unverified_support_max_perceptual_total_distance = 0
local_unverified_support_max_aspect_delta = 0.0
local_unverified_support_max_dimension_delta = 0.0
local_luma_candidate_max_delta = 55
local_contrast_candidate_max_delta = 50
local_edge_density_candidate_max_delta = 55
local_position_candidate_max_delta = 100
visual_luma_zero_score_delta = 70
visual_color_zero_score_delta = 90
visual_grid_luma_zero_score_delta = 80
visual_text_grid_zero_score_delta = 70
geometry_min_short_edge = 640
geometry_min_area = 350000
geometry_max_aspect_ratio = 1.4
geometry_max_aspect_delta = 0.45
geometry_max_width_delta = 0.3
geometry_max_height_delta = 0.3
geometry_enable_affine = true
geometry_enable_homography = true
geometry_model_slack = 2.0
geometry_max_anisotropy = 1.6
geometry_max_perspective = 2.2
geometry_affine_min_extra_inliers = 1
geometry_affine_min_extra_regions = 0
geometry_affine_max_mean_residual = 22.0
geometry_homography_min_extra_inliers = 2
geometry_homography_min_extra_regions = 1
geometry_homography_max_mean_residual = 18.0
geometry_ratio_min_margin = 2
geometry_enable_prosac_fallback = true
geometry_prosac_max_iters = 64
geometry_prosac_min_inliers = 8
visual_shape = true
visual_shape_min_signals = 3
visual_shape_min_text_grid_mean = 0
visual_shape_max_text_grid_mean = 255
visual_shape_min_text_regions = 0
visual_shape_min_luma_mean = 0
visual_shape_max_luma_mean = 255
visual_shape_min_luma_std = 0
visual_shape_max_luma_std = 255
visual_shape_min_local_hashes = 0
visual_shape_min_middle_text_percent = 0
visual_shape_min_center_text_percent = 0
visual_shape_max_center_text_percent = 100
visual_shape_max_edge_text_percent = 100
visual_shape_sparse_max_luma_mean = 255
visual_shape_max_rgb_spread = 255
visual_shape_sparse_max_text_grid_mean = 255
visual_shape_sparse_min_local_hashes = 0
future_threshold_key = 42

[detection_policy.suspicious]
actions = { delete_message = false, remove_user_roles = false, timeout_user = false, timeout_seconds = 60, ban_user = false, ban_delete_message_seconds = 0, kick_user = false, add_to_specimens = false }
threshold = { score_threshold = 20.0, perceptual_score_weight = 1.8, local_anchor_score_weight = 2.2, dense_local_anchor_score_weight = 1.4, visual_signature_score_weight = 0.0, visual_shape_score_weight = 1.0, visual_shape_score_cap = 10.0, perceptual_score_floor = 35.0, local_score_full_hits = 500, local_score_full_regions = 14, local_score_full_spread = 700.0, visual_shape_score_full = 300.0, cluster_coherence = true, cluster_hard_score = 63, cluster_chrome_ceiling_score = 19, cluster_member_score = 25, cluster_coherence_score = 63, cluster_min_size = 2, cluster_coverage_floor_permille = 0, exact_xxh128 = true, perceptual_hash = true, phash64_max_distance = 16, dhash64_max_distance = 15, perceptual_hash_max_total_distance = 30, perceptual_visual_support_distance_slack = 6, local_anchors = true, min_anchor_hits = 80, min_distinct_regions = 10, max_mean_distance = 7.0, local_unverified_support = false, local_unverified_support_min_anchor_hits = 0, local_unverified_support_min_distinct_regions = 0, local_unverified_support_min_retention_permille = 0, local_unverified_support_max_mean_distance = 0.0, local_unverified_support_max_perceptual_total_distance = 0, local_unverified_support_max_aspect_delta = 0.0, local_unverified_support_max_dimension_delta = 0.0, local_luma_candidate_max_delta = 55, local_contrast_candidate_max_delta = 50, local_edge_density_candidate_max_delta = 55, local_position_candidate_max_delta = 100, visual_luma_zero_score_delta = 70, visual_color_zero_score_delta = 90, visual_grid_luma_zero_score_delta = 80, visual_text_grid_zero_score_delta = 70, geometry_min_short_edge = 640, geometry_min_area = 350000, geometry_max_aspect_ratio = 1.4, geometry_max_aspect_delta = 0.45, geometry_max_width_delta = 0.3, geometry_max_height_delta = 0.3, geometry_enable_affine = true, geometry_enable_homography = true, geometry_model_slack = 2.0, geometry_max_anisotropy = 1.6, geometry_max_perspective = 2.2, geometry_affine_min_extra_inliers = 1, geometry_affine_min_extra_regions = 0, geometry_affine_max_mean_residual = 22.0, geometry_homography_min_extra_inliers = 2, geometry_homography_min_extra_regions = 1, geometry_homography_max_mean_residual = 18.0, geometry_ratio_min_margin = 2, geometry_enable_prosac_fallback = true, geometry_prosac_max_iters = 64, geometry_prosac_min_inliers = 8, visual_shape = true, visual_shape_min_signals = 3, visual_shape_min_text_grid_mean = 0, visual_shape_max_text_grid_mean = 255, visual_shape_min_text_regions = 0, visual_shape_min_luma_mean = 0, visual_shape_max_luma_mean = 255, visual_shape_min_luma_std = 0, visual_shape_max_luma_std = 255, visual_shape_min_local_hashes = 0, visual_shape_min_middle_text_percent = 0, visual_shape_min_center_text_percent = 0, visual_shape_max_center_text_percent = 100, visual_shape_max_edge_text_percent = 100, visual_shape_sparse_max_luma_mean = 255, visual_shape_max_rgb_spread = 255, visual_shape_sparse_max_text_grid_mean = 255, visual_shape_sparse_min_local_hashes = 0 }

[text_gate_policy]
enabled = false
keyword_threshold = 2
keyword_max_distance = 1
keywords = []
sentences = []
"#;

        let config = parse_advanced_guild_config(raw).expect("unknown TOML keys are ignored");

        assert_eq!(config.discord_confirmed_log_message, "<@&123>");
        assert_eq!(config.discord_suspicious_log_message, "<@&123>");
        assert!(config.discord_benign_log_message.is_empty());
        assert!(!config.scan_policy.exempt_administrators);
    }
}
