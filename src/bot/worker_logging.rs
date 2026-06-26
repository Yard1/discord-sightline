use crate::{
    bot::{
        discord::{BotLogColor, BotLogEvent, message_jump_link, user_incident_label},
        runtime::{AppState, MessageImageKey, MessageScopeKey, MessageSiblingInspection},
    },
    image::types::ImageCandidate,
};

pub(crate) async fn log_scan_failure(
    state: &AppState,
    candidate: &ImageCandidate,
    error: &str,
    elapsed_ms: u128,
    trace_id: &str,
) {
    let message_link = message_jump_link(
        candidate.guild_id,
        candidate.channel_id,
        candidate.message_id,
    );
    let event = BotLogEvent::new(
        "Image processing failed",
        "Sightline could not complete image processing for this candidate.",
    )
    .color(BotLogColor::Warning)
    .image_url(candidate.url.clone())
    .field("Target user", user_incident_label(candidate), false)
    .field("Source message", message_link, false)
    .field(
        "Candidate image",
        format!(
            "`{}` of `{}`\n{}",
            candidate.candidate_index, candidate.candidates_in_message, candidate.url
        ),
        false,
    )
    .field("Processing time", format!("`{elapsed_ms}` ms"), true)
    .field("Trace ID", format!("`{trace_id}`"), false)
    .field("Error", error.chars().take(1000).collect::<String>(), false);
    state.post_bot_log(event).await;
}

pub(crate) async fn mark_message_matched(state: &AppState, candidate: &ImageCandidate) {
    if !tracks_message_siblings(candidate) {
        return;
    }
    prune_message_inspection_if_large(state);
    let scope = MessageScopeKey::from_candidate(candidate);
    state.matched_messages.insert(scope);
    let siblings = state
        .sibling_inspections
        .iter()
        .filter(|entry| entry.key().scope == scope && entry.key().image_url != candidate.url)
        .map(|entry| (entry.key().clone(), entry.value().clone()))
        .collect::<Vec<_>>();
    for (key, inspection) in siblings {
        log_sibling_inspection_once(state, key, inspection).await;
    }
}

pub(crate) async fn record_nonmatching_sibling_inspection(
    state: &AppState,
    candidate: &ImageCandidate,
    image_id: String,
    decision: String,
    elapsed_ms: u128,
    trace_id: String,
    error: Option<String>,
) {
    if !tracks_message_siblings(candidate) {
        return;
    }
    prune_message_inspection_if_large(state);
    let key = MessageImageKey::from_candidate(candidate);
    let inspection = MessageSiblingInspection {
        candidate: candidate.clone(),
        image_id,
        decision,
        elapsed_ms,
        trace_id,
        error,
    };
    state
        .sibling_inspections
        .insert(key.clone(), inspection.clone());
    if state.matched_messages.contains(&key.scope) {
        log_sibling_inspection_once(state, key, inspection).await;
    }
}

async fn log_sibling_inspection_once(
    state: &AppState,
    key: MessageImageKey,
    inspection: MessageSiblingInspection,
) {
    if !state.logged_sibling_inspections.insert(key) {
        return;
    }
    prune_message_inspection_if_large(state);
    let message_link = message_jump_link(
        inspection.candidate.guild_id,
        inspection.candidate.channel_id,
        inspection.candidate.message_id,
    );
    let mut event = BotLogEvent::new(
        "Part of matched message",
        "This image did not trip Sightline, but another image in the same Discord message did.",
    )
    .color(BotLogColor::Info)
    .image_url(inspection.candidate.url.clone())
    .field(
        "Target user",
        user_incident_label(&inspection.candidate),
        false,
    )
    .field("Source message", message_link, false)
    .field(
        "Candidate image",
        format!(
            "`{}` of `{}`\n`{}`\n{}",
            inspection.candidate.candidate_index,
            inspection.candidate.candidates_in_message,
            inspection.image_id,
            inspection.candidate.url
        ),
        false,
    )
    .field("Scan result", format!("`{}`", inspection.decision), true)
    .field(
        "Processing time",
        format!("`{}` ms", inspection.elapsed_ms),
        true,
    )
    .field("Trace ID", format!("`{}`", inspection.trace_id), false);
    if let Some(error) = inspection.error {
        event = event.field("Error", error.chars().take(500).collect::<String>(), false);
    }
    state.post_bot_log(event).await;
}

fn tracks_message_siblings(candidate: &ImageCandidate) -> bool {
    candidate.candidates_in_message > 1
}

fn prune_message_inspection_if_large(state: &AppState) {
    let cap = state.config.queue.max_size.saturating_mul(20).max(1024);
    let tracked = tracked_message_state_len(state);
    if tracked <= cap {
        return;
    }
    let mut excess = tracked.saturating_sub(cap);
    prune_dash_map(&state.sibling_inspections, &mut excess);
    prune_dash_set(&state.logged_sibling_inspections, &mut excess);
    prune_dash_set(&state.matched_messages, &mut excess);
}

fn tracked_message_state_len(state: &AppState) -> usize {
    state
        .sibling_inspections
        .len()
        .saturating_add(state.logged_sibling_inspections.len())
        .saturating_add(state.matched_messages.len())
}

fn prune_dash_map<K, V>(map: &dashmap::DashMap<K, V>, excess: &mut usize)
where
    K: Clone + Eq + std::hash::Hash,
{
    if *excess == 0 {
        return;
    }
    let keys = map
        .iter()
        .take(*excess)
        .map(|entry| entry.key().clone())
        .collect::<Vec<_>>();
    prune_keys(keys, excess, |key| map.remove(key).is_some());
}

fn prune_dash_set<K>(set: &dashmap::DashSet<K>, excess: &mut usize)
where
    K: Clone + Eq + std::hash::Hash,
{
    if *excess == 0 {
        return;
    }
    let keys = set
        .iter()
        .take(*excess)
        .map(|entry| entry.key().clone())
        .collect::<Vec<_>>();
    prune_keys(keys, excess, |key| set.remove(key).is_some());
}

fn prune_keys<K>(
    keys: impl IntoIterator<Item = K>,
    excess: &mut usize,
    mut remove: impl FnMut(&K) -> bool,
) {
    for key in keys {
        if remove(&key) {
            *excess = excess.saturating_sub(1);
            if *excess == 0 {
                break;
            }
        }
    }
}
