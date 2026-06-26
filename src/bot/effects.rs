use crate::{
    bot::{
        discord::{
            RenderedBotLog, RoleRemovalOutcome, StoredSpecimen, ban_user,
            create_ledger_record_message, delete_message, edit_bot_log_in_channel, kick_user,
            post_bot_log_to_channel, remove_member_roles, timeout_user,
            upsert_config_record_message,
        },
        ledger::{SpecimenImageAttachment, SpecimenRecord},
    },
    configuration::guild::GuildConfigRecord,
};
use anyhow::Result;
use parking_lot::Mutex;
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use twilight_http::Client as DiscordClient;
use twilight_model::id::{
    Id,
    marker::{ChannelMarker, GuildMarker, MessageMarker, RoleMarker, UserMarker},
};

type BoxFutureResult<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;
type BoxFutureValue<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub(crate) trait DiscordEffects: Send + Sync {
    fn create_ledger_record_message(
        &self,
        channel_id: Id<ChannelMarker>,
        record: SpecimenRecord,
        image_attachments: Vec<SpecimenImageAttachment>,
        hmac_secret: String,
    ) -> BoxFutureResult<'_, StoredSpecimen>;

    fn upsert_config_record_message(
        &self,
        channel_id: Id<ChannelMarker>,
        existing_message_id: Option<Id<MessageMarker>>,
        record: GuildConfigRecord,
        hmac_secret: String,
    ) -> BoxFutureResult<'_, Option<Id<MessageMarker>>>;

    fn post_bot_log_to_channel(
        &self,
        channel_id: Id<ChannelMarker>,
        log: RenderedBotLog,
    ) -> BoxFutureResult<'_, Id<MessageMarker>>;

    fn edit_bot_log_in_channel(
        &self,
        channel_id: Id<ChannelMarker>,
        message_id: Id<MessageMarker>,
        log: RenderedBotLog,
    ) -> BoxFutureResult<'_, ()>;

    fn delete_message(
        &self,
        channel_id: Id<ChannelMarker>,
        message_id: Id<MessageMarker>,
    ) -> BoxFutureValue<'_, bool>;

    fn remove_member_roles(
        &self,
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
        removable_role_ids: Vec<Id<RoleMarker>>,
    ) -> BoxFutureValue<'_, RoleRemovalOutcome>;

    fn ban_user(
        &self,
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
        delete_message_seconds: u32,
        reason: String,
    ) -> BoxFutureValue<'_, bool>;

    fn kick_user(
        &self,
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
        reason: String,
    ) -> BoxFutureValue<'_, bool>;

    fn timeout_user(
        &self,
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
        timeout_seconds: u32,
        reason: String,
    ) -> BoxFutureValue<'_, bool>;
}

pub(crate) struct TwilightDiscordEffects {
    client: Arc<DiscordClient>,
}

impl TwilightDiscordEffects {
    pub(crate) fn new(client: Arc<DiscordClient>) -> Self {
        Self { client }
    }
}

#[allow(dead_code)]
#[derive(Debug, Default)]
pub(crate) struct MockDiscordEffects {
    next_message_id: AtomicU64,
    pub(crate) calls: Mutex<Vec<MockDiscordCall>>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) enum MockDiscordCall {
    CreateLedgerRecord {
        channel_id: Id<ChannelMarker>,
        specimen_id: String,
        attachment_filenames: Vec<String>,
    },
    UpsertConfigRecord {
        channel_id: Id<ChannelMarker>,
        existing_message_id: Option<Id<MessageMarker>>,
    },
    PostBotLog {
        channel_id: Id<ChannelMarker>,
        embeds: usize,
        attachments: usize,
        has_content: bool,
    },
    EditBotLog {
        channel_id: Id<ChannelMarker>,
        message_id: Id<MessageMarker>,
        embeds: usize,
        has_content: bool,
    },
    DeleteMessage {
        channel_id: Id<ChannelMarker>,
        message_id: Id<MessageMarker>,
    },
    RemoveMemberRoles {
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
        attempted: usize,
    },
    BanUser {
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
        delete_message_seconds: u32,
    },
    KickUser {
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
    },
    TimeoutUser {
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
        timeout_seconds: u32,
    },
}

#[allow(dead_code)]
impl MockDiscordEffects {
    pub(crate) fn new() -> Self {
        Self {
            next_message_id: AtomicU64::new(1),
            calls: Mutex::new(Vec::new()),
        }
    }

    fn next_message_id(&self) -> Id<MessageMarker> {
        Id::new(self.next_message_id.fetch_add(1, Ordering::Relaxed).max(1))
    }

    fn record(&self, call: MockDiscordCall) {
        self.calls.lock().push(call);
    }
}

impl DiscordEffects for MockDiscordEffects {
    fn create_ledger_record_message(
        &self,
        channel_id: Id<ChannelMarker>,
        record: SpecimenRecord,
        image_attachments: Vec<SpecimenImageAttachment>,
        _hmac_secret: String,
    ) -> BoxFutureResult<'_, StoredSpecimen> {
        Box::pin(async move {
            let specimen_id = record.specimen_id;
            let attachment_filenames = image_attachments
                .into_iter()
                .map(|attachment| attachment.filename)
                .collect();
            self.record(MockDiscordCall::CreateLedgerRecord {
                channel_id,
                specimen_id: specimen_id.clone(),
                attachment_filenames,
            });
            Ok(StoredSpecimen {
                message_id: self.next_message_id(),
                specimen_id,
            })
        })
    }

    fn upsert_config_record_message(
        &self,
        channel_id: Id<ChannelMarker>,
        existing_message_id: Option<Id<MessageMarker>>,
        _record: GuildConfigRecord,
        _hmac_secret: String,
    ) -> BoxFutureResult<'_, Option<Id<MessageMarker>>> {
        Box::pin(async move {
            self.record(MockDiscordCall::UpsertConfigRecord {
                channel_id,
                existing_message_id,
            });
            Ok(existing_message_id.or_else(|| Some(self.next_message_id())))
        })
    }

    fn post_bot_log_to_channel(
        &self,
        channel_id: Id<ChannelMarker>,
        log: RenderedBotLog,
    ) -> BoxFutureResult<'_, Id<MessageMarker>> {
        Box::pin(async move {
            let message_id = self.next_message_id();
            self.record(MockDiscordCall::PostBotLog {
                channel_id,
                embeds: log.embeds.len(),
                attachments: log.attachments.len(),
                has_content: log.content.is_some(),
            });
            Ok(message_id)
        })
    }

    fn edit_bot_log_in_channel(
        &self,
        channel_id: Id<ChannelMarker>,
        message_id: Id<MessageMarker>,
        log: RenderedBotLog,
    ) -> BoxFutureResult<'_, ()> {
        Box::pin(async move {
            self.record(MockDiscordCall::EditBotLog {
                channel_id,
                message_id,
                embeds: log.embeds.len(),
                has_content: log.content.is_some(),
            });
            Ok(())
        })
    }

    fn delete_message(
        &self,
        channel_id: Id<ChannelMarker>,
        message_id: Id<MessageMarker>,
    ) -> BoxFutureValue<'_, bool> {
        Box::pin(async move {
            self.record(MockDiscordCall::DeleteMessage {
                channel_id,
                message_id,
            });
            true
        })
    }

    fn remove_member_roles(
        &self,
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
        removable_role_ids: Vec<Id<RoleMarker>>,
    ) -> BoxFutureValue<'_, RoleRemovalOutcome> {
        Box::pin(async move {
            self.record(MockDiscordCall::RemoveMemberRoles {
                guild_id,
                user_id,
                attempted: removable_role_ids.len(),
            });
            RoleRemovalOutcome {
                attempted: removable_role_ids.len(),
                removed: removable_role_ids.len(),
                failed: 0,
            }
        })
    }

    fn ban_user(
        &self,
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
        delete_message_seconds: u32,
        _reason: String,
    ) -> BoxFutureValue<'_, bool> {
        Box::pin(async move {
            self.record(MockDiscordCall::BanUser {
                guild_id,
                user_id,
                delete_message_seconds,
            });
            true
        })
    }

    fn kick_user(
        &self,
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
        _reason: String,
    ) -> BoxFutureValue<'_, bool> {
        Box::pin(async move {
            self.record(MockDiscordCall::KickUser { guild_id, user_id });
            true
        })
    }

    fn timeout_user(
        &self,
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
        timeout_seconds: u32,
        _reason: String,
    ) -> BoxFutureValue<'_, bool> {
        Box::pin(async move {
            self.record(MockDiscordCall::TimeoutUser {
                guild_id,
                user_id,
                timeout_seconds,
            });
            true
        })
    }
}

impl DiscordEffects for TwilightDiscordEffects {
    fn create_ledger_record_message(
        &self,
        channel_id: Id<ChannelMarker>,
        record: SpecimenRecord,
        image_attachments: Vec<SpecimenImageAttachment>,
        hmac_secret: String,
    ) -> BoxFutureResult<'_, StoredSpecimen> {
        Box::pin(async move {
            create_ledger_record_message(
                &self.client,
                channel_id,
                &record,
                image_attachments,
                &hmac_secret,
            )
            .await
        })
    }

    fn upsert_config_record_message(
        &self,
        channel_id: Id<ChannelMarker>,
        existing_message_id: Option<Id<MessageMarker>>,
        record: GuildConfigRecord,
        hmac_secret: String,
    ) -> BoxFutureResult<'_, Option<Id<MessageMarker>>> {
        Box::pin(async move {
            upsert_config_record_message(
                &self.client,
                channel_id,
                existing_message_id,
                &record,
                &hmac_secret,
            )
            .await
        })
    }

    fn post_bot_log_to_channel(
        &self,
        channel_id: Id<ChannelMarker>,
        log: RenderedBotLog,
    ) -> BoxFutureResult<'_, Id<MessageMarker>> {
        Box::pin(async move { post_bot_log_to_channel(&self.client, channel_id, &log).await })
    }

    fn edit_bot_log_in_channel(
        &self,
        channel_id: Id<ChannelMarker>,
        message_id: Id<MessageMarker>,
        log: RenderedBotLog,
    ) -> BoxFutureResult<'_, ()> {
        Box::pin(async move {
            edit_bot_log_in_channel(&self.client, channel_id, message_id, &log).await
        })
    }

    fn delete_message(
        &self,
        channel_id: Id<ChannelMarker>,
        message_id: Id<MessageMarker>,
    ) -> BoxFutureValue<'_, bool> {
        Box::pin(async move { delete_message(&self.client, channel_id, message_id).await })
    }

    fn remove_member_roles(
        &self,
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
        removable_role_ids: Vec<Id<RoleMarker>>,
    ) -> BoxFutureValue<'_, RoleRemovalOutcome> {
        Box::pin(async move {
            remove_member_roles(&self.client, guild_id, user_id, &removable_role_ids).await
        })
    }

    fn ban_user(
        &self,
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
        delete_message_seconds: u32,
        reason: String,
    ) -> BoxFutureValue<'_, bool> {
        Box::pin(async move {
            ban_user(
                &self.client,
                guild_id,
                user_id,
                delete_message_seconds,
                &reason,
            )
            .await
        })
    }

    fn kick_user(
        &self,
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
        reason: String,
    ) -> BoxFutureValue<'_, bool> {
        Box::pin(async move { kick_user(&self.client, guild_id, user_id, &reason).await })
    }

    fn timeout_user(
        &self,
        guild_id: Id<GuildMarker>,
        user_id: Id<UserMarker>,
        timeout_seconds: u32,
        reason: String,
    ) -> BoxFutureValue<'_, bool> {
        Box::pin(async move {
            timeout_user(&self.client, guild_id, user_id, timeout_seconds, &reason).await
        })
    }
}
