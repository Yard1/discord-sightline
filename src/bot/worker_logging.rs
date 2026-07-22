use crate::{
    bot::{
        discord::{BotLogColor, BotLogEvent, message_jump_link, user_incident_label},
        runtime::{AppState, MessageImageKey, MessageScopeKey, MessageSiblingInspection},
    },
    image::types::{ImageCandidate, MatchOutcome},
};
use std::time::Instant;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum SiblingInspectionLogReason {
    RelatedDetection,
    AutomaticallyMarkedSuspicious,
}

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
        log_sibling_inspection_once(
            state,
            key,
            inspection,
            SiblingInspectionLogReason::RelatedDetection,
        )
        .await;
    }
}

pub(crate) async fn mark_message_scam_confirmed(
    state: &AppState,
    candidate: &ImageCandidate,
    outcome: &MatchOutcome,
) {
    if !tracks_message_siblings(candidate)
        || candidate.verify_only
        || !state
            .active_config_arc()
            .scan_policy
            .mark_message_siblings_suspicious
    {
        return;
    }
    prune_message_inspection_if_large(state);
    let scope = MessageScopeKey::from_candidate(candidate);
    state.confirmed_messages.insert(scope, outcome.clone());
    let sibling_keys = state
        .sibling_inspections
        .iter()
        .filter(|entry| entry.key().scope == scope && entry.key().image_url != candidate.url)
        .map(|entry| entry.key().clone())
        .collect::<Vec<_>>();
    for key in sibling_keys {
        if let Some((_, inspection)) = state.sibling_inspections.remove(&key) {
            let sibling = inspection.candidate.clone();
            log_sibling_inspection_once(
                state,
                key,
                inspection,
                SiblingInspectionLogReason::AutomaticallyMarkedSuspicious,
            )
            .await;
            queue_sibling_escalation(state, sibling, outcome.clone());
        }
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
    let source = state
        .confirmed_messages
        .get(&key.scope)
        .map(|entry| entry.value().clone());
    if let Some(source) = source
        && let Some((_, inspection)) = state.sibling_inspections.remove(&key)
    {
        log_sibling_inspection_once(
            state,
            key,
            inspection.clone(),
            SiblingInspectionLogReason::AutomaticallyMarkedSuspicious,
        )
        .await;
        queue_sibling_escalation(state, inspection.candidate, source);
    } else if state.matched_messages.contains(&key.scope) {
        log_sibling_inspection_once(
            state,
            key,
            inspection,
            SiblingInspectionLogReason::RelatedDetection,
        )
        .await;
    }
}

fn queue_sibling_escalation(state: &AppState, mut candidate: ImageCandidate, source: MatchOutcome) {
    if candidate.verify_only || candidate.sibling_escalation_source.is_some() {
        return;
    }

    candidate.sibling_escalation_source = Some(Box::new(source));
    candidate.enqueued_at = Some(Instant::now());
    let tx = state.bot.image_tx.clone();
    let shutdown = state.bot.shutdown.clone();
    state.bot.background_tasks.spawn(async move {
        tokio::select! {
            () = shutdown.cancelled() => {}
            result = tx.send(candidate) => {
                if let Err(source) = result {
                    tracing::warn!(
                        event = "message_sibling.enqueue_failed",
                        ?source,
                        "failed to enqueue suspicious message sibling"
                    );
                }
            }
        }
    });
}

async fn log_sibling_inspection_once(
    state: &AppState,
    key: MessageImageKey,
    inspection: MessageSiblingInspection,
    reason: SiblingInspectionLogReason,
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
    let (title, description) = sibling_inspection_log_copy(reason);
    let mut event = BotLogEvent::new(title, description)
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

fn sibling_inspection_log_copy(reason: SiblingInspectionLogReason) -> (&'static str, &'static str) {
    match reason {
        SiblingInspectionLogReason::RelatedDetection => (
            "Related image in flagged message",
            "This image did not independently trip Sightline, but another image in the same Discord message was flagged.",
        ),
        SiblingInspectionLogReason::AutomaticallyMarkedSuspicious => (
            "Automatically marked suspicious",
            "This image did not independently trip Sightline. Sightline automatically marked it suspicious because another image in the same Discord message was confirmed.",
        ),
    }
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
    prune_dash_map(&state.confirmed_messages, &mut excess);
    prune_dash_set(&state.logged_sibling_inspections, &mut excess);
    prune_dash_set(&state.matched_messages, &mut excess);
}

fn tracked_message_state_len(state: &AppState) -> usize {
    state
        .sibling_inspections
        .len()
        .saturating_add(state.confirmed_messages.len())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sibling_log_copy_distinguishes_automatic_suspicious_promotion() {
        let (related_title, related_description) =
            sibling_inspection_log_copy(SiblingInspectionLogReason::RelatedDetection);
        let (promoted_title, promoted_description) =
            sibling_inspection_log_copy(SiblingInspectionLogReason::AutomaticallyMarkedSuspicious);

        assert_eq!(related_title, "Related image in flagged message");
        assert!(related_description.contains("did not independently trip"));
        assert_eq!(promoted_title, "Automatically marked suspicious");
        assert!(promoted_description.contains("another image"));
        assert!(promoted_description.contains("confirmed"));
    }
}
