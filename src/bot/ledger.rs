#![allow(clippy::struct_field_names)]

use crate::{
    configuration::storage_codec::{decode_specimen, encode_specimen},
    image::types::{
        ImageAnchor, ImageFingerprint, ImageFingerprintParts, ImageVisualSignature, LocalImageHash,
    },
};
use anyhow::{Context, Result, anyhow};
use chrono::{SecondsFormat, Utc};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use twilight_model::id::{
    Id,
    marker::{ChannelMarker, GuildMarker, MessageMarker, UserMarker},
};

type HmacSha256 = Hmac<Sha256>;
pub(crate) const MAX_SPECIMEN_ATTACHMENT_BYTES: usize = 25 * 1024 * 1024;
pub(crate) const MAX_DISCORD_STORAGE_CONTENT: usize = 1_900;
pub(crate) const MAX_SPECIMEN_RECORD_ATTACHMENT_BYTES: usize = 1_000_000;
const SPECIMEN_RECORD_SCHEMA: u8 = 7;
const SPECIMEN_MANIFEST_SCHEMA: u8 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecimenRecord {
    pub schema: u8,
    #[serde(rename = "type")]
    pub kind: String,
    pub specimen_id: String,
    pub created_at: String,
    pub guild_id: String,
    pub source: SpecimenSource,
    pub image: SpecimenImage,
    pub anchors: Vec<ImageAnchor>,
    pub local_hashes: Vec<LocalImageHash>,
    pub preview: Option<SpecimenPreview>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sig: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SpecimenManifest {
    pub(crate) schema: u8,
    #[serde(rename = "type")]
    pub(crate) kind: String,
    pub(crate) specimen_id: String,
    pub(crate) guild_id: String,
    pub(crate) record_attachment: String,
    pub(crate) record_bytes: u32,
    pub(crate) record_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sig: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecimenSource {
    pub channel_id: String,
    pub message_id: String,
    pub source_author_id: String,
    pub added_by_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecimenImage {
    pub width: u32,
    pub height: u32,
    pub mime: Option<String>,
    pub byte_xxh128: String,
    pub phash64: String,
    pub dhash64: String,
    pub visual: ImageVisualSignature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecimenPreview {
    pub width: u32,
    pub height: u32,
    pub mime: Option<String>,
    pub byte_xxh128: String,
    pub phash64: String,
    pub dhash64: String,
    pub visual: ImageVisualSignature,
    pub anchors: Vec<ImageAnchor>,
    pub local_hashes: Vec<LocalImageHash>,
}

#[derive(Debug, Clone)]
pub(crate) struct SpecimenImageAttachment {
    pub(crate) filename: String,
    pub(crate) bytes: bytes::Bytes,
}

#[derive(Debug, Clone)]
pub(crate) struct SpecimenRecordAttachment {
    pub(crate) filename: String,
    pub(crate) bytes: Vec<u8>,
}

impl SpecimenImageAttachment {
    pub(crate) fn original(record: &SpecimenRecord, bytes: bytes::Bytes) -> Result<Self> {
        Self::new_variant(
            &record.specimen_id,
            "original",
            record.image.mime.as_deref(),
            bytes,
        )
    }

    pub(crate) fn discord_preview(record: &SpecimenRecord, bytes: bytes::Bytes) -> Result<Self> {
        let mime = record
            .preview
            .as_ref()
            .and_then(|preview| preview.mime.as_deref());
        Self::new_variant(&record.specimen_id, "discord-preview", mime, bytes)
    }

    fn new_variant(
        specimen_id: &str,
        label: &str,
        mime: Option<&str>,
        bytes: bytes::Bytes,
    ) -> Result<Self> {
        anyhow::ensure!(!bytes.is_empty(), "specimen image attachment is empty");
        anyhow::ensure!(
            bytes.len() <= MAX_SPECIMEN_ATTACHMENT_BYTES,
            "specimen image attachment exceeds Discord upload limit"
        );
        Ok(Self {
            filename: format!("{specimen_id}_{label}.{}", specimen_extension(mime)),
            bytes,
        })
    }
}

impl SpecimenRecord {
    pub fn new_add(
        guild_id: Id<GuildMarker>,
        source_channel_id: Id<ChannelMarker>,
        source_message_id: Id<MessageMarker>,
        source_author_id: Id<UserMarker>,
        added_by_id: Id<UserMarker>,
        image: ImageFingerprint,
        preview: Option<ImageFingerprint>,
    ) -> Self {
        let now = Utc::now();
        let created_at = now.to_rfc3339_opts(SecondsFormat::Secs, true);
        let specimen_id = format!("spm_{}_{}", now.format("%Y%m%d"), image.byte_xxh128);

        Self {
            schema: SPECIMEN_RECORD_SCHEMA,
            kind: "specimen.add".to_owned(),
            specimen_id,
            created_at,
            guild_id: guild_id.get().to_string(),
            source: SpecimenSource {
                channel_id: source_channel_id.get().to_string(),
                message_id: source_message_id.get().to_string(),
                source_author_id: source_author_id.get().to_string(),
                added_by_id: added_by_id.get().to_string(),
            },
            image: SpecimenImage {
                width: image.width,
                height: image.height,
                mime: image.mime,
                byte_xxh128: image.byte_xxh128,
                phash64: image.phash64,
                dhash64: image.dhash64,
                visual: image.visual,
            },
            anchors: image.local_anchors,
            local_hashes: image.local_hashes,
            preview: preview.map(SpecimenPreview::from),
            sig: None,
        }
    }

    pub(crate) fn new_recovered(
        guild_id: Id<GuildMarker>,
        ledger_channel_id: Id<ChannelMarker>,
        ledger_message_id: Id<MessageMarker>,
        bot_user_id: Id<UserMarker>,
        specimen_id: String,
        image: ImageFingerprint,
        preview: Option<ImageFingerprint>,
    ) -> Self {
        let mut record = Self::new_add(
            guild_id,
            ledger_channel_id,
            ledger_message_id,
            bot_user_id,
            bot_user_id,
            image,
            preview,
        );
        record.specimen_id = specimen_id;
        record
    }

    pub fn sign(mut self, secret: &str) -> Result<Self> {
        let sig = sign_record(&self, secret)?;
        self.sig = Some(sig);
        Ok(self)
    }

    pub fn verify(&self, secret: &str) -> Result<bool> {
        verify_record_signature(self, self.sig.as_deref(), secret)
    }

    pub fn validate(&self, expected_guild_id: Id<GuildMarker>) -> Result<()> {
        anyhow::ensure!(
            self.guild_id == expected_guild_id.get().to_string(),
            "specimen guild_id {} does not match configured guild {}",
            self.guild_id,
            expected_guild_id.get()
        );
        ImageFingerprint::validate_parts(ImageFingerprintParts {
            width: self.image.width,
            height: self.image.height,
            byte_xxh128: &self.image.byte_xxh128,
            phash64: &self.image.phash64,
            dhash64: &self.image.dhash64,
            visual: &self.image.visual,
            local_anchors: &self.anchors,
            local_hashes: &self.local_hashes,
        })
        .context("validating original specimen fingerprint")?;
        if let Some(preview) = &self.preview {
            preview
                .fingerprint()
                .validate()
                .context("validating Discord preview specimen fingerprint")?;
        }
        Ok(())
    }
}

impl SpecimenManifest {
    pub(crate) fn for_record(
        record: &SpecimenRecord,
        attachment: &SpecimenRecordAttachment,
    ) -> Self {
        Self {
            schema: SPECIMEN_MANIFEST_SCHEMA,
            kind: "specimen.add.manifest".to_owned(),
            specimen_id: record.specimen_id.clone(),
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

    pub(crate) fn validate(&self, expected_guild_id: Id<GuildMarker>) -> Result<()> {
        anyhow::ensure!(
            self.schema == SPECIMEN_MANIFEST_SCHEMA,
            "unsupported specimen manifest schema {}",
            self.schema
        );
        anyhow::ensure!(
            self.kind == "specimen.add.manifest",
            "unsupported specimen manifest type {}",
            self.kind
        );
        anyhow::ensure!(
            self.guild_id == expected_guild_id.get().to_string(),
            "specimen manifest guild_id {} does not match configured guild {}",
            self.guild_id,
            expected_guild_id.get()
        );
        anyhow::ensure!(
            !self.specimen_id.trim().is_empty(),
            "specimen manifest specimen_id must not be empty"
        );
        anyhow::ensure!(
            !self.record_attachment.trim().is_empty(),
            "specimen manifest record_attachment must not be empty"
        );
        anyhow::ensure!(
            usize::try_from(self.record_bytes).unwrap_or(usize::MAX)
                <= MAX_SPECIMEN_RECORD_ATTACHMENT_BYTES,
            "specimen manifest record attachment is too large"
        );
        validate_hex(&self.record_sha256, 32, "specimen manifest record_sha256")?;
        Ok(())
    }
}

impl From<ImageFingerprint> for SpecimenPreview {
    fn from(image: ImageFingerprint) -> Self {
        Self {
            width: image.width,
            height: image.height,
            mime: image.mime,
            byte_xxh128: image.byte_xxh128,
            phash64: image.phash64,
            dhash64: image.dhash64,
            visual: image.visual,
            anchors: image.local_anchors,
            local_hashes: image.local_hashes,
        }
    }
}

impl SpecimenPreview {
    pub fn fingerprint(&self) -> ImageFingerprint {
        ImageFingerprint {
            width: self.width,
            height: self.height,
            mime: self.mime.clone(),
            byte_xxh128: self.byte_xxh128.clone(),
            phash64: self.phash64.clone(),
            dhash64: self.dhash64.clone(),
            visual: self.visual.clone(),
            local_anchors: self.anchors.clone(),
            local_hashes: self.local_hashes.clone(),
        }
    }
}

fn specimen_extension(mime: Option<&str>) -> &'static str {
    let mime = mime.unwrap_or_default();
    if mime.eq_ignore_ascii_case("image/png") {
        "png"
    } else if mime.eq_ignore_ascii_case("image/gif") {
        "gif"
    } else if mime.eq_ignore_ascii_case("image/webp") {
        "webp"
    } else if mime.eq_ignore_ascii_case("image/jpeg") || mime.eq_ignore_ascii_case("image/jpg") {
        "jpg"
    } else {
        "img"
    }
}

pub(crate) fn parse_and_verify_specimen_manifest(
    manifest_raw: &str,
    secret: &str,
    expected_guild_id: Id<GuildMarker>,
) -> Result<Option<SpecimenManifest>> {
    let Some(manifest) = decode_specimen::<SpecimenManifest>(manifest_raw)? else {
        return Ok(None);
    };
    if !manifest.verify(secret)? {
        return Ok(None);
    }
    manifest.validate(expected_guild_id)?;
    Ok(Some(manifest))
}

pub(crate) fn parse_and_verify_specimen_record(
    manifest: &SpecimenManifest,
    record_bytes: &[u8],
    secret: &str,
    expected_guild_id: Id<GuildMarker>,
) -> Result<SpecimenRecord> {
    verify_record_attachment(manifest, record_bytes)?;

    let record: SpecimenRecord =
        rmp_serde::from_slice(record_bytes).context("deserializing specimen record attachment")?;
    if record.schema != SPECIMEN_RECORD_SCHEMA || record.kind != "specimen.add" {
        return Err(anyhow!("unsupported specimen record"));
    }
    anyhow::ensure!(
        record.specimen_id == manifest.specimen_id,
        "specimen record id does not match manifest"
    );
    anyhow::ensure!(
        record.guild_id == manifest.guild_id,
        "specimen record guild does not match manifest"
    );

    if record.verify(secret)? {
        record.validate(expected_guild_id)?;
        Ok(record)
    } else {
        Err(anyhow!("invalid specimen record signature"))
    }
}

pub(crate) fn specimen_record_attachment(
    record: &SpecimenRecord,
) -> Result<SpecimenRecordAttachment> {
    let bytes = rmp_serde::to_vec(record).context("serializing specimen record attachment")?;
    anyhow::ensure!(
        bytes.len() <= MAX_SPECIMEN_RECORD_ATTACHMENT_BYTES,
        "specimen record attachment exceeds storage limit"
    );
    Ok(SpecimenRecordAttachment {
        filename: format!("{}.sightline.msgpack", record.specimen_id),
        bytes,
    })
}

pub(crate) fn specimen_manifest_to_discord(manifest: &SpecimenManifest) -> Result<String> {
    encode_specimen(manifest).context("serializing specimen storage manifest")
}

pub(crate) fn signed_specimen_manifest_to_discord(
    record: &SpecimenRecord,
    attachment: &SpecimenRecordAttachment,
    secret: &str,
) -> Result<String> {
    let manifest = SpecimenManifest::for_record(record, attachment).sign(secret)?;
    specimen_manifest_to_discord(&manifest)
}

fn verify_record_attachment(manifest: &SpecimenManifest, record_bytes: &[u8]) -> Result<()> {
    anyhow::ensure!(
        record_bytes.len() == usize::try_from(manifest.record_bytes).unwrap_or(usize::MAX),
        "specimen record attachment size does not match manifest"
    );
    anyhow::ensure!(
        sha256_hex(record_bytes) == manifest.record_sha256,
        "specimen record attachment digest does not match manifest"
    );
    Ok(())
}

fn sign_record(record: &SpecimenRecord, secret: &str) -> Result<String> {
    if secret.is_empty() {
        return Err(anyhow!("empty HMAC secret"));
    }

    let mut unsigned = record.clone();
    unsigned.sig = None;
    let payload = serde_json::to_vec(&unsigned).context("serializing unsigned specimen record")?;

    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).context("creating hmac")?;
    mac.update(&payload);
    Ok(hex::encode(mac.finalize().into_bytes()))
}

fn sign_manifest(manifest: &SpecimenManifest, secret: &str) -> Result<String> {
    if secret.is_empty() {
        return Err(anyhow!("empty HMAC secret"));
    }

    let mut unsigned = manifest.clone();
    unsigned.sig = None;
    let payload =
        serde_json::to_vec(&unsigned).context("serializing unsigned specimen manifest")?;

    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).context("creating hmac")?;
    mac.update(&payload);
    Ok(hex::encode(mac.finalize().into_bytes()))
}

fn verify_record_signature(
    record: &SpecimenRecord,
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
    let payload = serde_json::to_vec(&unsigned).context("serializing unsigned specimen record")?;
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).context("creating hmac")?;
    mac.update(&payload);
    Ok(mac.verify_slice(&signature).is_ok())
}

fn verify_manifest_signature(
    manifest: &SpecimenManifest,
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
        serde_json::to_vec(&unsigned).context("serializing unsigned specimen manifest")?;
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).context("creating hmac")?;
    mac.update(&payload);
    Ok(mac.verify_slice(&signature).is_ok())
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn validate_hex(value: &str, bytes: usize, name: &str) -> Result<()> {
    let decoded = hex::decode(value).with_context(|| format!("{name} must be hex"))?;
    anyhow::ensure!(decoded.len() == bytes, "{name} must be {bytes} bytes");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use twilight_model::id::Id;

    fn test_fingerprint(byte: char, mime: Option<&str>) -> ImageFingerprint {
        ImageFingerprint {
            width: 10,
            height: 10,
            mime: mime.map(str::to_owned),
            byte_xxh128: byte.to_string().repeat(32),
            phash64: "00".repeat(8),
            dhash64: "11".repeat(8),
            visual: ImageVisualSignature::default(),
            local_anchors: Vec::new(),
            local_hashes: Vec::new(),
        }
    }

    fn anchored_fingerprint(byte: char, mime: Option<&str>, anchors: usize) -> ImageFingerprint {
        let mut fingerprint = test_fingerprint(byte, mime);
        fingerprint.width = 959;
        fingerprint.height = 1279;
        fingerprint.local_anchors = (0..anchors)
            .map(|index| {
                let index_u32 = u32::try_from(index).expect("test anchor index fits u32");
                let index_u64 = u64::try_from(index).expect("test anchor index fits u64");
                let hash = format!("{:016x}", index_u64 ^ 0x1234_5678_9abc_def0);
                let hash2 = format!("{:016x}", index_u64 ^ 0xfedc_ba98_7654_3210);
                ImageAnchor {
                    id: format!("a{:03}", index + 1),
                    x: (index_u32 * 13) % 700,
                    y: (index_u32 * 17) % 1_100,
                    w: 33,
                    h: 33,
                    pos_x: u8::try_from(index % 256).expect("test anchor pos_x fits u8"),
                    pos_y: u8::try_from((index * 3) % 256).expect("test anchor pos_y fits u8"),
                    hash,
                    hash2,
                    luma_mean: u8::try_from((index * 5) % 256)
                        .expect("test anchor luma_mean fits u8"),
                    luma_std: u8::try_from((index * 7) % 256)
                        .expect("test anchor luma_std fits u8"),
                    edge_density: u8::try_from((index * 11) % 256)
                        .expect("test anchor edge_density fits u8"),
                    kind: "orb_fast_brief".to_owned(),
                    region: u32::try_from(index % 64).expect("test anchor region fits u32"),
                    max_distance: 12,
                }
            })
            .collect();
        fingerprint
    }

    #[test]
    fn signed_records_verify_and_tampering_fails() {
        let image = test_fingerprint('a', Some("image/png"));
        let record = SpecimenRecord::new_add(
            Id::new(1),
            Id::new(2),
            Id::new(3),
            Id::new(4),
            Id::new(5),
            image,
            None,
        )
        .sign("secret")
        .unwrap();

        assert!(record.verify("secret").unwrap());

        let mut tampered = record;
        tampered.image.width = 11;
        assert!(!tampered.verify("secret").unwrap());
    }

    #[test]
    fn compact_specimen_storage_round_trips() {
        let image = test_fingerprint('a', Some("image/png"));
        let preview = test_fingerprint('b', Some("image/jpeg"));
        let specimen = SpecimenRecord::new_add(
            Id::new(1),
            Id::new(2),
            Id::new(3),
            Id::new(4),
            Id::new(5),
            image,
            Some(preview),
        )
        .sign("secret")
        .unwrap();
        let attachment = specimen_record_attachment(&specimen).unwrap();
        let encoded =
            signed_specimen_manifest_to_discord(&specimen, &attachment, "secret").unwrap();
        assert!(encoded.starts_with(crate::configuration::storage_codec::SPECIMEN_PREFIX));
        assert!(!encoded.contains('{'));

        let manifest = parse_and_verify_specimen_manifest(&encoded, "secret", Id::new(1))
            .unwrap()
            .unwrap();
        let decoded =
            parse_and_verify_specimen_record(&manifest, &attachment.bytes, "secret", Id::new(1))
                .unwrap();
        assert_eq!(decoded.specimen_id, specimen.specimen_id);
        let preview = decoded.preview.expect("preview fingerprint round trips");
        assert_eq!(preview.byte_xxh128, "b".repeat(32));
        assert_eq!(preview.mime.as_deref(), Some("image/jpeg"));
    }

    #[test]
    fn signed_specimen_manifest_stays_within_discord_message_limit() {
        let image = anchored_fingerprint('a', Some("image/png"), 512);
        let preview = anchored_fingerprint('b', Some("image/jpeg"), 512);
        let specimen = SpecimenRecord::new_add(
            Id::new(1),
            Id::new(2),
            Id::new(3),
            Id::new(4),
            Id::new(5),
            image,
            Some(preview),
        )
        .sign("secret")
        .unwrap();

        let attachment = specimen_record_attachment(&specimen).unwrap();
        let encoded =
            signed_specimen_manifest_to_discord(&specimen, &attachment, "secret").unwrap();

        assert!(
            encoded.len() <= 2_000,
            "signed specimen manifest is {} chars, over Discord's 2000-char limit",
            encoded.len()
        );
        assert!(
            encoded.len() <= MAX_DISCORD_STORAGE_CONTENT,
            "signed specimen manifest is {} chars, over the storage safety margin",
            encoded.len()
        );
        assert!(
            attachment.bytes.len() <= MAX_SPECIMEN_RECORD_ATTACHMENT_BYTES,
            "signed specimen record attachment is {} bytes, over storage cap",
            attachment.bytes.len()
        );

        let manifest = parse_and_verify_specimen_manifest(&encoded, "secret", Id::new(1))
            .unwrap()
            .unwrap();
        let decoded =
            parse_and_verify_specimen_record(&manifest, &attachment.bytes, "secret", Id::new(1))
                .unwrap();
        assert_eq!(decoded.anchors.len(), 512);
        assert_eq!(
            decoded
                .preview
                .as_ref()
                .map(|preview| preview.anchors.len()),
            Some(512)
        );
    }

    #[test]
    fn specimen_image_attachments_are_labeled_by_variant() {
        let record = SpecimenRecord::new_add(
            Id::new(1),
            Id::new(2),
            Id::new(3),
            Id::new(4),
            Id::new(5),
            test_fingerprint('a', Some("image/png")),
            Some(test_fingerprint('b', Some("image/jpeg"))),
        );

        let original =
            SpecimenImageAttachment::original(&record, bytes::Bytes::from_static(b"o")).unwrap();
        let preview =
            SpecimenImageAttachment::discord_preview(&record, bytes::Bytes::from_static(b"p"))
                .unwrap();

        assert_eq!(
            original.filename,
            format!("{}_original.png", record.specimen_id)
        );
        assert_eq!(
            preview.filename,
            format!("{}_discord-preview.jpg", record.specimen_id)
        );
    }
}
