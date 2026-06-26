use anyhow::{Context, Result, bail};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use serde::{Serialize, de::DeserializeOwned};

pub const CONFIG_PREFIX: &str = "sc1:";
pub const SPECIMEN_PREFIX: &str = "si1:";

pub fn encode_config<T: Serialize>(value: &T) -> Result<String> {
    encode_with_prefix(CONFIG_PREFIX, value)
}

pub fn encode_specimen<T: Serialize>(value: &T) -> Result<String> {
    encode_with_prefix(SPECIMEN_PREFIX, value)
}

pub fn decode_config<T: DeserializeOwned>(raw: &str) -> Result<Option<T>> {
    decode_with_prefix(CONFIG_PREFIX, raw)
}

pub fn decode_specimen<T: DeserializeOwned>(raw: &str) -> Result<Option<T>> {
    decode_with_prefix(SPECIMEN_PREFIX, raw)
}

fn encode_with_prefix<T: Serialize>(prefix: &str, value: &T) -> Result<String> {
    let bytes = rmp_serde::to_vec(value).context("serializing storage record")?;
    let mut out = String::with_capacity(prefix.len() + encoded_len(bytes.len()));
    out.push_str(prefix);
    URL_SAFE_NO_PAD.encode_string(bytes, &mut out);
    Ok(out)
}

fn decode_with_prefix<T: DeserializeOwned>(prefix: &str, raw: &str) -> Result<Option<T>> {
    let Some(encoded) = raw.strip_prefix(prefix) else {
        return Ok(None);
    };
    let bytes = URL_SAFE_NO_PAD
        .decode(encoded)
        .context("decoding storage record")?;
    if bytes.is_empty() {
        bail!("empty storage record");
    }
    Ok(Some(
        rmp_serde::from_slice(&bytes).context("deserializing storage record")?,
    ))
}

fn encoded_len(bytes: usize) -> usize {
    bytes.div_ceil(3) * 4
}
