use crate::{
    configuration::guild::normalize_image_extension,
    image::{
        pipeline::is_discord_host,
        types::{CandidateKind, ImageCandidate},
    },
};
use std::collections::HashSet;
use twilight_model::channel::{Attachment, Message, message::embed::Embed};
use twilight_model::id::{
    Id,
    marker::{ChannelMarker, GuildMarker, MessageMarker, RoleMarker, UserMarker},
};

pub(crate) const MAX_MANUAL_SPECIMEN_IMAGES_PER_INTERACTION: usize = 10;

pub fn extract_candidates_from_message(
    guild_id: Id<GuildMarker>,
    message: &Message,
    max_images: usize,
    allowed_extensions: &[String],
    max_file_bytes: u64,
) -> Vec<ImageCandidate> {
    let channel_id = message.channel_id;
    let message_id = message.id;
    let author_id = message.author.id;
    let author_username = Some(message.author.name.clone());
    let author_global_name = message.author.global_name.clone();
    let mut seen = HashSet::new();
    let mut candidates = Vec::new();

    append_candidates_from_media(
        &mut candidates,
        &mut seen,
        &MediaExtractionContext {
            guild_id,
            channel_id,
            message_id,
            author_id,
            author_username: author_username.clone(),
            author_global_name: author_global_name.clone(),
            max_images,
            allowed_extensions,
            max_file_bytes,
        },
        &message.attachments,
        &message.embeds,
    );
    if candidates.len() >= max_images {
        return with_candidate_positions(candidates);
    }

    for snapshot in &message.message_snapshots {
        append_candidates_from_media(
            &mut candidates,
            &mut seen,
            &MediaExtractionContext {
                guild_id,
                channel_id,
                message_id,
                author_id,
                author_username: author_username.clone(),
                author_global_name: author_global_name.clone(),
                max_images,
                allowed_extensions,
                max_file_bytes,
            },
            &snapshot.message.attachments,
            &snapshot.message.embeds,
        );
        if candidates.len() >= max_images {
            return with_candidate_positions(candidates);
        }
    }

    with_candidate_positions(candidates)
}

#[derive(Clone)]
struct MediaExtractionContext<'a> {
    guild_id: Id<GuildMarker>,
    channel_id: Id<ChannelMarker>,
    message_id: Id<MessageMarker>,
    author_id: Id<UserMarker>,
    author_username: Option<String>,
    author_global_name: Option<String>,
    max_images: usize,
    allowed_extensions: &'a [String],
    max_file_bytes: u64,
}

fn append_candidates_from_media(
    candidates: &mut Vec<ImageCandidate>,
    seen: &mut HashSet<String>,
    context: &MediaExtractionContext<'_>,
    attachments: &[Attachment],
    embeds: &[Embed],
) {
    for attachment in attachments {
        let url = attachment.url.as_str();
        if !is_discord_hosted_url(url) {
            continue;
        }
        let mime = attachment
            .content_type
            .clone()
            .or_else(|| url_mime_hint(url));
        let size_bytes = Some(attachment.size);
        if size_bytes.is_some_and(|size| size > context.max_file_bytes) {
            continue;
        }
        if !attachment_type_allowed(url, mime.as_deref(), context.allowed_extensions) {
            continue;
        }
        let has_dimensions = attachment.width.is_some() && attachment.height.is_some();
        let is_image = mime.as_deref().is_some_and(mime_starts_with_image) || has_dimensions;

        if is_image && seen.insert(url.to_owned()) {
            candidates.push(ImageCandidate {
                guild_id: context.guild_id,
                channel_id: context.channel_id,
                message_id: context.message_id,
                candidate_index: 0,
                candidates_in_message: 0,
                author_id: context.author_id,
                author_username: context.author_username.clone(),
                author_global_name: context.author_global_name.clone(),
                url: url.to_owned(),
                proxy_url: Some(attachment.proxy_url.clone()),
                kind: CandidateKind::Attachment,
                mime_hint: mime,
                size_bytes,
                metadata_width: attachment.width.and_then(u64_to_u32),
                metadata_height: attachment.height.and_then(u64_to_u32),
                media_flags: attachment.flags.map(|flags| flags.bits()),
                verify_only: false,
                sibling_escalation_source: None,
                enqueued_at: None,
            });
        }

        if candidates.len() >= context.max_images {
            return;
        }
    }

    for embed in embeds {
        for media in embed_media(embed).into_iter().flatten() {
            if !is_discord_hosted_url(media.url) {
                continue;
            }

            if !extension_allowed(media.url, context.allowed_extensions) {
                continue;
            }

            if seen.insert(media.url.to_owned()) {
                candidates.push(ImageCandidate {
                    guild_id: context.guild_id,
                    channel_id: context.channel_id,
                    message_id: context.message_id,
                    candidate_index: 0,
                    candidates_in_message: 0,
                    author_id: context.author_id,
                    author_username: context.author_username.clone(),
                    author_global_name: context.author_global_name.clone(),
                    url: media.url.to_owned(),
                    proxy_url: media.proxy_url.map(str::to_owned),
                    kind: media.kind,
                    mime_hint: url_mime_hint(media.url),
                    size_bytes: None,
                    metadata_width: media.width,
                    metadata_height: media.height,
                    media_flags: None,
                    verify_only: false,
                    sibling_escalation_source: None,
                    enqueued_at: None,
                });
            }

            if candidates.len() >= context.max_images {
                return;
            }
        }
    }
}

fn embed_media(embed: &Embed) -> [Option<ExtractedEmbedMedia<'_>>; 2] {
    [
        embed.image.as_ref().map(|image| ExtractedEmbedMedia {
            url: image.url.as_str(),
            proxy_url: image.proxy_url.as_deref(),
            width: image.width.and_then(u64_to_u32),
            height: image.height.and_then(u64_to_u32),
            kind: CandidateKind::EmbedImage,
        }),
        embed.thumbnail.as_ref().map(|image| ExtractedEmbedMedia {
            url: image.url.as_str(),
            proxy_url: image.proxy_url.as_deref(),
            width: image.width.and_then(u64_to_u32),
            height: image.height.and_then(u64_to_u32),
            kind: CandidateKind::EmbedThumbnail,
        }),
    ]
}

#[derive(Debug, Clone, Copy)]
struct ExtractedEmbedMedia<'a> {
    url: &'a str,
    proxy_url: Option<&'a str>,
    width: Option<u32>,
    height: Option<u32>,
    kind: CandidateKind,
}

fn u64_to_u32(value: u64) -> Option<u32> {
    u32::try_from(value).ok()
}

fn with_candidate_positions(mut candidates: Vec<ImageCandidate>) -> Vec<ImageCandidate> {
    let count = u16::try_from(candidates.len()).unwrap_or(u16::MAX);
    for (index, candidate) in candidates.iter_mut().enumerate() {
        candidate.candidate_index = u16::try_from(index.saturating_add(1)).unwrap_or(u16::MAX);
        candidate.candidates_in_message = count;
    }
    candidates
}

pub fn message_has_potential_image(message: &Message) -> bool {
    media_has_potential_image(&message.attachments, &message.embeds)
        || message.message_snapshots.iter().any(|snapshot| {
            media_has_potential_image(&snapshot.message.attachments, &snapshot.message.embeds)
        })
}

fn media_has_potential_image(attachments: &[Attachment], embeds: &[Embed]) -> bool {
    attachments.iter().any(|attachment| {
        attachment.width.is_some()
            || attachment.height.is_some()
            || url_extension(attachment.url.as_str())
                .as_deref()
                .is_some_and(is_common_image_extension)
            || attachment
                .content_type
                .as_deref()
                .is_some_and(mime_starts_with_image)
    }) || embeds.iter().any(|embed| {
        embed
            .image
            .as_ref()
            .is_some_and(|image| is_discord_hosted_url(image.url.as_str()))
            || embed
                .thumbnail
                .as_ref()
                .is_some_and(|image| is_discord_hosted_url(image.url.as_str()))
    })
}

fn is_common_image_extension(extension: &str) -> bool {
    matches!(
        extension,
        "jpg" | "jpeg" | "png" | "gif" | "webp" | "bmp" | "tif" | "tiff"
    )
}

fn url_mime_hint(raw: &str) -> Option<String> {
    let mime = match url_extension(raw)?.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "tif" | "tiff" => "image/tiff",
        _ => return None,
    };
    Some(mime.to_owned())
}

pub fn message_has_any_role(message: &Message, role_ids: &[Id<RoleMarker>]) -> Option<bool> {
    if role_ids.is_empty() {
        return Some(false);
    }
    let member = message.member.as_ref()?;
    Some(
        member
            .roles
            .iter()
            .any(|member_role_id| role_ids.contains(member_role_id)),
    )
}

fn is_discord_hosted_url(raw: &str) -> bool {
    let Ok(url) = url::Url::parse(raw) else {
        return false;
    };

    url.scheme() == "https" && url.host_str().is_some_and(is_discord_host)
}

pub(crate) fn extension_allowed(raw: &str, allowed_extensions: &[String]) -> bool {
    if allowed_extensions.is_empty() {
        return true;
    }

    let Some(extension) = url_extension(raw) else {
        return false;
    };
    allowed_extensions
        .iter()
        .any(|allowed| allowed.eq_ignore_ascii_case(&extension))
}

fn attachment_type_allowed(raw: &str, mime: Option<&str>, allowed_extensions: &[String]) -> bool {
    extension_allowed(raw, allowed_extensions)
        || mime
            .and_then(mime_extension_hint)
            .is_some_and(|extension| extension_name_allowed(extension, allowed_extensions))
}

fn extension_name_allowed(extension: &str, allowed_extensions: &[String]) -> bool {
    allowed_extensions.is_empty()
        || allowed_extensions
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(extension))
}

fn mime_extension_hint(mime: &str) -> Option<&'static str> {
    if mime.eq_ignore_ascii_case("image/jpeg") || mime.eq_ignore_ascii_case("image/jpg") {
        Some("jpg")
    } else if mime.eq_ignore_ascii_case("image/png") {
        Some("png")
    } else if mime.eq_ignore_ascii_case("image/gif") {
        Some("gif")
    } else if mime.eq_ignore_ascii_case("image/webp") {
        Some("webp")
    } else if mime.eq_ignore_ascii_case("image/bmp") {
        Some("bmp")
    } else if mime.eq_ignore_ascii_case("image/tiff") {
        Some("tiff")
    } else {
        None
    }
}

fn mime_starts_with_image(mime: &str) -> bool {
    mime.get(..6)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("image/"))
}

fn url_extension(raw: &str) -> Option<String> {
    let path = url::Url::parse(raw).ok().map_or_else(
        || raw.split('?').next().unwrap_or(raw).to_owned(),
        |url| url.path().to_owned(),
    );
    let filename = path.rsplit('/').next()?;
    let extension = filename.rsplit_once('.')?.1;
    let extension = normalize_image_extension(extension);
    if extension.is_empty() {
        None
    } else {
        Some(extension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use twilight_model::{
        channel::{
            Attachment, AttachmentFlags, Message,
            message::{
                MessageType,
                embed::{Embed, EmbedImage, EmbedThumbnail},
            },
        },
        id::Id,
        user::User,
        util::Timestamp,
    };

    #[allow(deprecated)]
    fn message_with_media(attachments: Vec<Attachment>, embeds: Vec<Embed>) -> Message {
        Message {
            activity: None,
            application: None,
            application_id: None,
            attachments,
            author: User {
                accent_color: None,
                avatar: None,
                avatar_decoration: None,
                avatar_decoration_data: None,
                banner: None,
                bot: false,
                discriminator: 0,
                email: None,
                flags: None,
                global_name: Some("Global".to_owned()),
                id: Id::new(4),
                locale: None,
                mfa_enabled: None,
                name: "author".to_owned(),
                premium_type: None,
                primary_guild: None,
                public_flags: None,
                system: None,
                verified: None,
            },
            call: None,
            channel_id: Id::new(2),
            components: Vec::new(),
            content: String::new(),
            edited_timestamp: None,
            embeds,
            flags: None,
            guild_id: Some(Id::new(1)),
            id: Id::new(3),
            interaction: None,
            interaction_metadata: None,
            kind: MessageType::Regular,
            member: None,
            mention_channels: Vec::new(),
            mention_everyone: false,
            mention_roles: Vec::new(),
            mentions: Vec::new(),
            message_snapshots: Vec::new(),
            pinned: false,
            poll: None,
            reactions: Vec::new(),
            reference: None,
            referenced_message: None,
            role_subscription_data: None,
            sticker_items: Vec::new(),
            timestamp: Timestamp::from_micros(1).expect("valid timestamp"),
            thread: None,
            tts: false,
            webhook_id: None,
        }
    }

    fn attachment() -> Attachment {
        Attachment {
            content_type: Some("image/png".to_owned()),
            description: None,
            duration_secs: None,
            ephemeral: false,
            filename: "image.png".to_owned(),
            flags: Some(AttachmentFlags::IS_REMIX),
            height: Some(600),
            id: Id::new(10),
            proxy_url: "https://media.discordapp.net/attachments/1/2/image.png".to_owned(),
            size: 1234,
            title: None,
            url: "https://cdn.discordapp.com/attachments/1/2/image.png".to_owned(),
            waveform: None,
            width: Some(800),
        }
    }

    fn embed() -> Embed {
        Embed {
            author: None,
            color: None,
            description: None,
            fields: Vec::new(),
            footer: None,
            image: Some(EmbedImage {
                height: Some(1200),
                proxy_url: Some(
                    "https://media.discordapp.net/attachments/1/2/embed-image.jpg".to_owned(),
                ),
                url: "https://cdn.discordapp.com/attachments/1/2/embed-image.jpg".to_owned(),
                width: Some(900),
            }),
            kind: "rich".to_owned(),
            provider: None,
            thumbnail: Some(EmbedThumbnail {
                height: Some(400),
                proxy_url: Some(
                    "https://media.discordapp.net/attachments/1/2/embed-thumb.webp".to_owned(),
                ),
                url: "https://cdn.discordapp.com/attachments/1/2/embed-thumb.webp".to_owned(),
                width: Some(300),
            }),
            timestamp: None,
            title: None,
            url: None,
            video: None,
        }
    }

    #[test]
    fn extracts_attachment_metadata() {
        let message = message_with_media(vec![attachment()], Vec::new());

        let candidates =
            extract_candidates_from_message(Id::new(1), &message, 10, &["png".to_owned()], 10_000);

        assert_eq!(candidates.len(), 1);
        let candidate = &candidates[0];
        assert_eq!(candidate.kind, CandidateKind::Attachment);
        assert_eq!(
            candidate.proxy_url.as_deref(),
            Some("https://media.discordapp.net/attachments/1/2/image.png")
        );
        assert_eq!(candidate.mime_hint.as_deref(), Some("image/png"));
        assert_eq!(candidate.size_bytes, Some(1234));
        assert_eq!(candidate.metadata_width, Some(800));
        assert_eq!(candidate.metadata_height, Some(600));
        assert_eq!(
            candidate.media_flags,
            Some(AttachmentFlags::IS_REMIX.bits())
        );
        assert_eq!(candidate.candidate_index, 1);
        assert_eq!(candidate.candidates_in_message, 1);
    }

    #[test]
    fn extracts_embed_image_and_thumbnail_metadata() {
        let message = message_with_media(Vec::new(), vec![embed()]);

        let candidates = extract_candidates_from_message(
            Id::new(1),
            &message,
            10,
            &["jpg".to_owned(), "webp".to_owned()],
            10_000,
        );

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].kind, CandidateKind::EmbedImage);
        assert_eq!(candidates[0].mime_hint.as_deref(), Some("image/jpeg"));
        assert_eq!(candidates[0].metadata_width, Some(900));
        assert_eq!(candidates[0].metadata_height, Some(1200));
        assert_eq!(
            candidates[0].proxy_url.as_deref(),
            Some("https://media.discordapp.net/attachments/1/2/embed-image.jpg")
        );
        assert_eq!(candidates[1].kind, CandidateKind::EmbedThumbnail);
        assert_eq!(candidates[1].mime_hint.as_deref(), Some("image/webp"));
        assert_eq!(candidates[1].metadata_width, Some(300));
        assert_eq!(candidates[1].metadata_height, Some(400));
        assert_eq!(
            candidates[1].proxy_url.as_deref(),
            Some("https://media.discordapp.net/attachments/1/2/embed-thumb.webp")
        );
        assert_eq!(candidates[0].candidates_in_message, 2);
        assert_eq!(candidates[1].candidate_index, 2);
    }

    #[test]
    fn skips_attachment_when_discord_says_non_image() {
        let mut attachment = attachment();
        attachment.content_type = Some("text/plain".to_owned());
        attachment.width = None;
        attachment.height = None;
        let message = message_with_media(vec![attachment], Vec::new());

        assert!(
            extract_candidates_from_message(Id::new(1), &message, 10, &["png".to_owned()], 10_000)
                .is_empty()
        );
    }

    #[test]
    fn skips_over_limit_attachment_from_discord_size() {
        let message = message_with_media(vec![attachment()], Vec::new());

        assert!(
            extract_candidates_from_message(Id::new(1), &message, 10, &["png".to_owned()], 10)
                .is_empty()
        );
    }

    #[test]
    fn accepts_attachment_without_extension_when_mime_matches_allowed_type() {
        let mut attachment = attachment();
        attachment.url = "https://cdn.discordapp.com/attachments/1/2/image".to_owned();
        attachment.proxy_url = "https://media.discordapp.net/attachments/1/2/image".to_owned();
        attachment.content_type = Some("image/png".to_owned());
        let message = message_with_media(vec![attachment], Vec::new());

        let candidates =
            extract_candidates_from_message(Id::new(1), &message, 10, &["png".to_owned()], 10_000);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].mime_hint.as_deref(), Some("image/png"));
    }

    #[test]
    fn canonicalizes_jpeg_extension_to_jpg_allowlist() {
        let mut attachment = attachment();
        attachment.url = "https://cdn.discordapp.com/attachments/1/2/image.jpeg".to_owned();
        attachment.proxy_url = "https://media.discordapp.net/attachments/1/2/image.jpeg".to_owned();
        attachment.content_type = None;
        let message = message_with_media(vec![attachment], Vec::new());

        let candidates =
            extract_candidates_from_message(Id::new(1), &message, 10, &["jpg".to_owned()], 10_000);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].mime_hint.as_deref(), Some("image/jpeg"));
    }

    #[test]
    fn jpg_only_scan_policy_rejects_png_attachment() {
        let message = message_with_media(vec![attachment()], Vec::new());

        let candidates =
            extract_candidates_from_message(Id::new(1), &message, 10, &["jpg".to_owned()], 10_000);

        assert!(candidates.is_empty());
    }

    #[test]
    fn extracts_ten_manual_specimen_attachments_when_limit_allows() {
        let attachments = (0..MAX_MANUAL_SPECIMEN_IMAGES_PER_INTERACTION)
            .map(|index| {
                let mut attachment = attachment();
                attachment.id = Id::new(u64::try_from(index).expect("small index") + 10);
                attachment.url =
                    format!("https://cdn.discordapp.com/attachments/1/2/image-{index}.png");
                attachment.proxy_url =
                    format!("https://media.discordapp.net/attachments/1/2/image-{index}.png");
                attachment
            })
            .collect();
        let message = message_with_media(attachments, Vec::new());

        let candidates = extract_candidates_from_message(
            Id::new(1),
            &message,
            MAX_MANUAL_SPECIMEN_IMAGES_PER_INTERACTION,
            &[],
            10_000,
        );

        assert_eq!(candidates.len(), MAX_MANUAL_SPECIMEN_IMAGES_PER_INTERACTION);
        assert_eq!(
            candidates.last().map(|candidate| candidate.candidate_index),
            Some(10)
        );
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.candidates_in_message == 10)
        );
    }
}
