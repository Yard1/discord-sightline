use anyhow::{Context, Result};
use tokio::sync::mpsc;
use twilight_gateway::{CloseFrame, Event, EventTypeFlags, Shard, StreamExt as _};

fn wanted_event_types() -> EventTypeFlags {
    EventTypeFlags::CHANNEL_DELETE
        | EventTypeFlags::GATEWAY_INVALIDATE_SESSION
        | EventTypeFlags::GATEWAY_RECONNECT
        | EventTypeFlags::GUILD_DELETE
        | EventTypeFlags::INTERACTION_CREATE
        | EventTypeFlags::MESSAGE_CREATE
        | EventTypeFlags::MESSAGE_DELETE
        | EventTypeFlags::MESSAGE_DELETE_BULK
        | EventTypeFlags::MESSAGE_UPDATE
}

pub(crate) trait BotEventStream: Send {
    async fn next_event(&mut self) -> Option<Result<Event>>;

    fn close(&mut self) {}
}

pub(crate) struct TwilightShardEventStream {
    shard: Shard,
}

impl TwilightShardEventStream {
    pub(crate) fn new(shard: Shard) -> Self {
        Self { shard }
    }
}

impl BotEventStream for TwilightShardEventStream {
    async fn next_event(&mut self) -> Option<Result<Event>> {
        self.shard
            .next_event(wanted_event_types())
            .await
            .map(|result| result.context("gateway event"))
    }

    fn close(&mut self) {
        self.shard.close(CloseFrame::NORMAL);
    }
}

#[allow(dead_code)]
pub(crate) struct SyntheticEventStream {
    rx: mpsc::Receiver<Result<Event>>,
}

#[allow(dead_code)]
impl SyntheticEventStream {
    pub(crate) fn new(rx: mpsc::Receiver<Result<Event>>) -> Self {
        Self { rx }
    }

    pub(crate) fn channel(buffer: usize) -> (mpsc::Sender<Result<Event>>, Self) {
        let (tx, rx) = mpsc::channel(buffer);
        (tx, Self::new(rx))
    }
}

impl BotEventStream for SyntheticEventStream {
    async fn next_event(&mut self) -> Option<Result<Event>> {
        self.rx.recv().await
    }
}
