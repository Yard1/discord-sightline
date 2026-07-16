# discord-sightline

Discord Sightline is a small Rust moderation bot that catches repeated spam
images posted from compromised Discord accounts. It keeps its matcher in memory
and stores each guild's configuration and image specimens as signed, compact
records in a private Discord channel.

## Why this exists

Compromised accounts often post a scam campaign as an image rather than as text.
Images are harder for keyword filters to catch, and they let attackers reuse a
familiar visual template while swapping out the link. Even images from
the same campaign rarely match byte for byte: re-encoding, resizing, compression
settings, metadata changes, screenshots, Discord's own preview and proxy
processing, and small layout or color edits all change the bytes.

Exact matching still helps, and Sightline checks an exact hash first because it
is cheap. But an exact hash only catches identical files, and a spammer needs
just one small change to slip past it. So Sightline treats the exact hash as the
first and cheapest gate in a wider, conservative cascade: whole-image perceptual
hashes for near-identical images, local text-anchor hashes for template-like
variants, visual and geometry checks to rule out unrelated text-heavy images, and
optional OCR confirmation for borderline cases.

## What's implemented

This is the first implementation pass. It includes:

- A Twilight gateway listener for message create/update and interactions.
- Slash commands: `/config` (admin configuration panel) and `/doctor` (setup
  diagnostics).
- Message context commands: `Add scam image specimen` and `Validate scam images`.
- Modal- and select-based guild configuration for the database channel, bot log
  channel, moderator roles, scan-exempt roles, detection policy, and Discord log
  messages.
- Separate `/import-hashes`, `/export-hashes`, and `/import-images` commands for
  specimen operations.
- Discord storage replay with HMAC verification for compact specimen and config
  records.
- A bounded image queue, CPU/download/OCR concurrency limits, XXH3 hashing, pHash/dHash
  matching, and local text-anchor matching.
- Scanning of image attachments, embed images, and embed thumbnails.
- Per-guild confirmed and suspicious thresholds, each with its own set of
  actions.

## Requirements

Sightline needs the pinned nightly toolchain in `rust-toolchain.toml`, because
the matcher uses safe portable SIMD.

## Configuration

### Secrets and config path

These come from environment variables:

```text
DISCORD_TOKEN
SPECIMEN_HMAC_SECRET
OCR_SPACE_API_KEY
SIGHTLINE_CONFIG
```

`SPECIMEN_HMAC_SECRET` must be at least 32 bytes. `OCR_SPACE_API_KEY` is only
needed if OCR is enabled in a guild.

### Runtime limits (local TOML)

Copy `sightline.example.toml` to `sightline.toml` and adjust the limits as
needed. This file sets the safety ceilings for the machine Sightline runs on:
queue size and enqueue timeout, one CPU-bound concurrency budget, one download
IO concurrency budget, one OCR IO concurrency budget, process-wide budgets for
downloaded bodies and decoded images, a separate follow-up byte store, request
timeouts, decoded pixel caps, local-anchor scan budgets, and the size of the
global LRU cache of hash results. It also configures the OCR.space endpoint,
including timeout, retry count, language, scaling, and orientation detection.

The local `[download]` section can also enable a lightweight Discord CDN
connection warmer. When `warmer_enabled = true`, Sightline uses the same image
download HTTP client to send an occasional cheap `HEAD` request to a permanent
Discord CDN avatar if real image traffic has not recently touched the CDN. The
warmer does not consume download concurrency permits and is not counted in image
pipeline metrics; its debug log latency is a pool warm/cold diagnostic.

Guild admins can tune detection from inside Discord — lower the maximum scanned
file size, adjust detection hyperparameters, and change the detection policy —
but they can never raise the machine's hard ceilings. A guild can only move to
equal or cheaper behavior.

Set `[bot].user_id` to the bot's application user ID if you want permission
diagnostics to use a fixed account. If you leave it empty, Sightline falls back
to the user ID of the running token.

### The guild database channel

For each guild, create exactly one private text channel named `sightline-db`.
Sightline finds it by name on restart and uses it as that guild's database.

Set permissions so that:

- `@everyone` is denied both View Channel and Send Messages.
- The bot has View Channel and Send Messages in both `sightline-db` and the bot
  log channel.
- The bot has Attach Files in `sightline-db`, since image-backed specimens store
  their image variants as attachments.

If either channel isn't readable and writable by the bot, the guild stays
inactive.

The first bot message in the channel is the single signed configuration record.
Each specimen is then stored as its own signed message after it. Image-backed
specimens attach the canonical original as `*_original.*`, and, when Discord
generated a resized preview, that preview as `*_discord-preview.*` on the same
message. You can delete one specimen by hand without affecting the rest of the
database. (Writing an image-backed specimen briefly holds an extra copy of each
image in memory, because Discord uploads require owned attachment buffers.)

Messages use a compact MessagePack payload encoded as base64url with a short
prefix (`sc1:` for config, `si1:` for specimens). This keeps messages text-safe
for Discord without repeating JSON field names.

Sightline reads a guild's database channel the first time it sees that guild
after starting, then keeps the config and specimen message IDs in memory. If you
delete a specimen message from a loaded channel, Sightline drops that specimen
from the in-memory matcher and clears any cached results tied to it.

The bot stays inactive until the config message exists and config is enabled. It
also stops scanning if it can't load `sightline-db`; failed loads retry every
minute instead of caching an empty runtime. A guild forced into safe mode after
a failed config write periodically reloads its last durable Discord state, and
bot permissions are refreshed every five minutes so restored permissions resume
scanning without a manual `/doctor`. Use the Enable / Disable button in `/config`
to turn scanning and moderation on or off. If you delete `sightline-db` while the
bot is running, its runtime is invalidated and retried until the replacement
channel exists and contains configuration again.

### The /config panel

`/config` opens the administrator panel, with these controls:

- **Log message** opens plain-text editors for Discord log content. Routine logs
  and confirmed, suspicious, and benign detection logs have separate copies, all
  empty by default. Put user or role mentions here when you want operator pings;
  the human-readable details stay in rich embeds.
- **Roles** opens role selectors for moderator roles, scan-exempt roles, and the
  verified/member role.
- **Actions** opens the action controls for confirmed and suspicious detections.
  Only the selected actions are taken; selecting none makes that operating point
  a dry run.
- **Advanced** opens a TOML editor for guild-specific scan and text-gate policy,
  confirmed/suspicious thresholds, timeout and ban durations, and
  image-processing hyperparameters. Machine-local values remain hard ceilings.

There are two production operating points:

- **`confirmed.*`** — high-confidence detections. Defaults to an exact match or
  strong local-anchor evidence, with message deletion enabled.
- **`suspicious.*`** — medium-confidence detections. Defaults to no destructive
  action, but is always logged.

Cluster coherence is treated as suspicious support by default. It can help route
family-like local/visual evidence to OCR, but it does not skip OCR or trigger
confirmed actions by itself.

Set `scan_policy.mark_message_siblings_suspicious = true` in the per-guild
Advanced TOML to escalate the other images in a message after any one image is
confirmed as a scam. A hard match triggers this immediately; an OCR-backed match
triggers it only after OCR confirms bad text. Each otherwise non-matching sibling
is logged as suspicious and, when the text gate is enabled, sent through OCR.
The flag defaults to `false` so existing guilds do not incur new OCR traffic.

Each operating point has its own thresholds and actions. Logs are always emitted,
and you can add pings through the editable log content. Supported timeout
durations are 60, 300, 600, 3600, 86400, and 604800 seconds (1m, 5m, 10m, 1h, 1
day, and 1 week). Message deletion runs concurrently with member actions; member
actions run in this order: timeout, remove role, ban, kick.

### OCR text confirmation

When the text gate is enabled, Sightline sends a single OCR crop to OCR.space
using `OCREngine=2` and `base64Image`. Crops are capped at 1 MB before the
request. Transient HTTP and server errors, plus HTTP 429 rate limiting, are
retried with backoff and `Retry-After` support, behind a dedicated OCR semaphore
and an overall OCR deadline.

If OCR is unavailable or can't read the text, the suspicious-image log says so.
If OCR confirms a configured sentence or enough keywords, Sightline treats the
image as a confirmed match and applies the confirmed actions and log copy.
Partial keyword hits stay suspicious, for moderator review.

### Auto-adding matched images

You can configure Sightline to add matched images back into the ledger as new
specimens. Exact byte duplicates of existing specimens are always skipped, but
fuzzy variants are kept as separate specimens. The same rule applies to manual
right-click adds, hash imports, and image uploads: only exact duplicates are
rejected. Hash-only imports have no source image, so they create fingerprint-only
entries with no attachments.

When auto-add is on for a guild, Sightline can still act from a decisive Discord
preview match. The original image is then downloaded and written as a specimen by
a bounded background worker, so moderation is not delayed by the extra download,
hashing, or database write. Detection logs may show
`add_to_specimens=deferred_original` while the later database log records the
actual specimen write.

### What gets scanned

Users with any scan-exempt role are skipped. Administrators are also skipped by
default. Set `scan_policy.exempt_administrators = false` in `/config -> Advanced`
for a guild to scan administrators there; the local TOML `[scan]` value only
sets the default for newly configured guilds and empty Advanced resets.

Use Discord permissions to control which channels are scanned — for example, by
removing the bot's View Channel permission where it shouldn't run. Sightline
always skips its own messages, the `sightline-db` channel, and the bot log
channel.

Automated scanning also respects `scan_policy.allowed_extensions`, which is
initialized from the local config and defaults to `["jpg"]`. Manual specimen
adds ignore that allowlist, but still must be valid images within the size
limits.

Only Discord-hosted attachment and embed image URLs are scanned. Non-Discord
hosts are skipped, because supporting them safely would need a DNS-pinned
connection to avoid DNS rebinding.

### Telemetry (optional)

Runtime telemetry is controlled by `[telemetry.dial9]` and is disabled by
default. When enabled, Sightline writes Tokio runtime trace segments under
`telemetry.dial9.trace_dir`, and matched/suspicious log embeds include a
per-image trace ID so you can correlate a trace with the logs. The repository
ships `.cargo/config.toml` with `tokio_unstable` enabled, since this telemetry
needs Tokio's unstable instrumentation hooks. Leave it off in production unless
you're actively collecting diagnostics.

## How matching works

Matching is a cheap cascade, designed to run on weak CPUs:

1. Exact XXH3-128.
2. Whole-image pHash/dHash.
3. Local text-anchor hashes, selected from dense, high-contrast patches.
4. Geometry verification (similarity, affine, or homography) for local/token
   evidence. Affine and homography fits require extra inlier support and tighter
   residual caps before they confirm a match. There is also a bounded PROSAC
   homography fallback for when the similarity voter can't seed a model, plus an
   optional 2-nearest-neighbor margin filter.
5. Cheap visual gates for tile brightness, contrast, and edge density, plus
   whole-image brightness, color, luma grid, text density, and coarse tile
   position.
6. Geometry gates for readable image size, broad aspect-ratio sanity, and aspect
   compatibility with the matched specimen.

Local anchors are stored with each specimen and scanned across prioritized
normalized tiles in new images; the matcher does not store synthetic hash
variants. Each image produces a decision of `confirmed`, `suspicious`, or `pass`,
according to guild policy.

Every processed image emits a compact audit row to stdout. Confirmed and
suspicious decisions always include that row in the Discord log, along with the
candidate image URL, a Discord jump link, the target user's mention/ID/names, a
link to the matched specimen when available, the actions taken, and the gates
that passed.

### Result cache

Concurrent downloads of the same Discord attachment URL are coalesced while they
are in flight, so repost bursts do not start duplicate network requests for the
same effective URL. Processed image outcomes are kept in a process-wide,
guild-scoped O(1) in-memory LRU cache capped by
`queue.hash_outcome_cache_size` (default and maximum `100000` across all
guilds). A reposted,
byte-identical image can reuse that snapshot after download and byte hashing —
before decode and local hashing — but only while the current detection and
text-gate policy still matches. A guild's entries are invalidated when its
specimens or effective matching state changes. The cache is not persisted and
rebuilds naturally as the bot scans images after a restart.

## Discord setup

Invite the bot with permissions to view and send messages and attach files in
`sightline-db`, view and send messages in the bot log channel, read message
content, manage messages, and — depending on which actions you enable — manage
roles, time out members, ban members, and kick members. Use the `bot` and
`applications.commands` scopes.

When `commands.register_on_startup = true`, the bot registers these commands on
startup:

- `/config` — opens the administrator configuration panel.
- `/doctor` — checks common permission and configuration problems.
- `/import-hashes` — opens a modal for JSON/JSONL fingerprint import.
- `/export-hashes` — returns current specimen fingerprints as an ephemeral JSONL
  attachment.
- `/import-images` — opens a file-upload modal and turns uploaded images into
  specimens.
- `/stats` — shows image-matching statistics for the current guild.
- `/audit` — audits current specimen quality signals.
- Right click a message → Apps → `Add scam image specimen` — hashes the images on
  that message and stores them as signed specimens.
- Right click a message → Apps → `Validate scam images` — sends scan-policy-eligible
  images on that message through the normal validation pipeline and configured
  match actions. This is moderator-only and bypasses target-author administrator
  and scan-exempt role filters.

Before the first config save, anyone with Administrator or Manage Guild can open
configuration. After setup, only administrators and the configured moderator
roles can configure Sightline or use its interactions.

If the `remove_user_roles` action is enabled, Sightline removes only the
configured verified/member role rather than stripping every role from the member.
If `timeout_user` is enabled, the bot needs Discord's Moderate Members
permission. If `kick_user` is enabled, the bot needs Kick Members permission.

## Importing and exporting hashes

Use `/import-hashes` and `/export-hashes` for fingerprint import and export.
Import accepts the exact fingerprint JSON emitted by the local `hash-image` and
`hash-images` commands: a single object, a JSON array of objects, or JSONL with
one object per line. Export returns the current in-memory guild specimens as an
ephemeral JSONL attachment, in the same import-compatible format.

Use `/import-images` to open a file-upload modal for specimen images. Uploaded
images go through the same production pipeline and are stored as image-backed
specimens.

```json
{"schema":7,"source_path":"specimen.png","fingerprint":{"width":0,"height":0,"mime":"image/png","byte_xxh128":"...32 hex chars...","phash64":"...16 hex chars...","dhash64":"...16 hex chars...","visual":{"luma_mean":0,"luma_std":0,"rgb_mean":[0,0,0],"grid_luma":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"text_grid":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},"local_anchors":[],"local_hashes":[]}}
```

To create a compatible record from a local image:

```bash
cargo run --release -- hash-image ./specimen.png
```

## Local image testing

You can exercise the hashing and comparison path without Discord tokens or the
gateway:

```bash
cargo run --release -- hash-image ./image.png
cargo run --release -- compare-images ./specimen.png ./candidate.png
cargo run --release -- augment-images ./specimens ./target/specimen-augmented --profile geometry
cargo run --release -- hash-images ./specimens ./fingerprints/specimens
cargo run --release -- hash-images ./target/specimen-augmented ./fingerprints/specimen_augmented
cargo run --release -- hash-images ./candidates ./fingerprints/candidates
cargo run --release -- compare-image-sets ./fingerprints/specimens ./fingerprints/candidates
cargo run --release -- compare-image-sets ./fingerprints/specimens ./fingerprints/specimen_augmented
cargo run --release -- compare-image-sets ./fingerprints/specimens ./fingerprints/hard_negatives
cargo run --release -- inspect-image ./candidate.png --fake-ocr-text "claim airdrop" --artifacts-dir ./target/sightline-inspect
cargo run --release -- export-ocr-crops-leave-one-out ./specimens ./target/specimen-ocr-crops-leave-one-out
cargo run --release -- ocr-space ./specimen.png
cargo run --release -- benchmark-images ./specimens --repeat 5 --warmup 1 --summary-only --max-preload-bytes 536870912
cargo run --release -- benchmark-matcher ./fingerprints/specimens ./fingerprints/candidates --repeat 10 --warmup 2 --summary-only
```

A few notes on the main commands:

- **`compare-images`** reads the thresholds in `sightline.toml`, hashes both
  images, and prints JSON: byte-hash equality, pHash/dHash distances,
  local-anchor diagnostics, the geometry model used, and whether the candidate
  matches.
- **`hash-image` and `hash-images`** use the same `ExportedImageFingerprint`
  schema as Discord import. `hash-images` takes one image or a directory,
  recursively hashes supported files, and writes one `.sightline.json` file per
  image with the full fingerprint (specimen anchors and candidate local hashes
  included).
- **`augment-images`** builds deterministic local validation variants from a
  source set: small rotations, center crop/zoom, horizontal and vertical shear,
  and mild perspective warps. Out-of-bounds pixels are filled with an estimated
  border color so dark specimens stay dark. It is for local validation only, not
  production; hash the variants with `hash-images` so validation still runs the
  exact production pipeline.
- **`compare-image-sets`** loads fingerprint files and compares every specimen
  against every candidate, printing matched and suspicious pairs and the
  best-scoring specimen per candidate. When a suspicious hit comes from generic
  visual shape rather than a concrete specimen, it's reported with
  `specimen_source = "generic_visual_shape"`. Hard negatives are a
  validation-only candidate set; hashed into their own directory and run as the
  candidate side, they should produce zero matches.
- **`inspect-image`** runs the staged detection engine on one image. It hashes
  through the production pipeline, evaluates the same policy as
  `compare-image-sets`, optionally evaluates the text gate via `--fake-ocr-text`,
  and writes crops plus a decision report only when `--artifacts-dir` is given.
  Production never persists crops or OCR responses.
- **`export-ocr-crops-leave-one-out`** writes the exact crop bytes that would be
  sent to OCR for each specimen, organized into leave-one-out folds (a `train/`
  set of all other specimens and a `test/` set for the held-out one). In
  production, each image produces exactly one OCR payload: the original bytes if
  they're already an acceptable, ≤1 MB payload, otherwise a conservative
  original-color JPEG crop capped at 1 MB.
- **`benchmark-images`** preloads images into memory outside the timed section
  (up to `--max-preload-bytes`) and measures only the image pipeline: decode,
  normalization, whole-image hashes, anchor selection, and local hash scanning.
  The JSON report includes wall time, hashes/second, min/max/average, and
  p50/p90/p95/p99 latencies.
- **`benchmark-matcher`** loads exported fingerprints and measures matching
  alone, with no image I/O or hashing. It reports production
  `Matcher::find_for_policy` latency and, by default, an all-pairs
  `compare_fingerprints` phase. Use `--no-pairwise` for large fingerprint sets
  when you only care about production matcher latency.

## Validation

For seeded K-fold validation, use the Python helper. It hashes images through the
Rust CLI, builds train/test folds from the exported fingerprints, runs
`compare-image-sets`, and reports TP/FP/TN/FN, accuracy, precision, recall, F1,
specificity, and false-positive/false-negative rates:

```bash
python3 scripts/validate_kfold.py --specimens ./specimens --folds 5 --seed 42
python3 scripts/validate_kfold.py --specimens ./specimens --negatives ./hard_negative_specimens --folds 5 --seed 42
cargo build --release
python3 scripts/validate_kfold.py --specimens ./specimens --folds 5 --seed 42 --binary ./target/release/discord-sightline
```

The summary reports `fully_matched_positive_sources`,
`suspicious_only_positive_sources`, and `unmatched_positive_sources`, plus
false-positive lists when negatives are provided.

For strict known-positive coverage, run leave-one-out validation. It tests every
specimen as the held-out one against all the rest and reports
`fully_matched_specimens`, `suspicious_only_specimens`, and
`unmatched_specimens`:

```bash
python3 scripts/validate_leave_one_out.py --specimens ./specimens --binary ./target/release/discord-sightline
python3 scripts/validate_leave_one_out.py --specimens ./specimens --negatives ./hard_negative_specimens --binary ./target/release/discord-sightline
```

Without `--negatives`, you measure recall and false negatives for held-out
positives; precision, specificity, and false-positive rate need known-negative
images. Hard negatives are a local validation input only and are never stored in
Discord. Reports are cached under `target/sightline-validation` by default. Set
`SIGHTLINE_CONFIG=...` (or edit `sightline.toml`) to validate different
hyperparameters, and build once and pass `--binary` to avoid paying Cargo startup
costs on every fold.

For parameter tuning, prefer the higher-level runner. It is the source of truth
for comparing candidate TOML configs, because it runs both K-fold and
leave-one-out validation, applies the same quality gates to each run, and writes
one combined report:

```bash
cargo build --release
python3 scripts/validate_tuning.py \
  --specimens ./specimens \
  --negatives ./hard_negative_specimens \
  --config sightline.toml \
  --binary ./target/release/discord-sightline \
  --no-build
```

You can compare several configs in one run:

```bash
python3 scripts/validate_tuning.py \
  --config target/sightline-tune.toml \
  --config target/sightline-sweep.toml \
  --binary ./target/release/discord-sightline \
  --no-build
```

The tuning runner requires every positive specimen to hard- or soft-match in both
validation modes, requires zero hard matches for hard negatives, limits soft
hard-negative hits, and checks precision and specificity when negatives are
provided. It does not auto-fit hyperparameters: edit a TOML config, run the
protocol, then compare `summary.json` and `summary.md` under
`target/sightline-tuning-validation`.

Add `--augment-profile geometry` when validating geometry work. That generates
deterministic transformed positives with `augment-images`, hashes them through
the production pipeline, and adds an `augmented_transforms` section showing
hard/soft/unmatched coverage by transform type. It's a reporting aid, not a
separate production path.

By default, `validate_tuning.py` caches fingerprints once and uses the Rust
`validate-threshold-sweep` path to evaluate all explicit TOML configs in memory.
This is the fastest path for threshold-only sweeps, because cached fingerprints
are keyed only by image-processing settings, not by decision thresholds. If you
need to compare configs that change fingerprint generation — normalization size,
local tile size, stride, anchor count, or local hash cap — pass
`--legacy-validation` or use separate runs.

The lower-level sweep command accepts already-generated fingerprint directories:

```bash
./target/release/discord-sightline validate-threshold-sweep \
  target/sightline-tuning-validation/fingerprints/positives \
  target/sightline-tuning-validation/fingerprints/negatives \
  --config sightline.toml \
  --config target/sightline-sweep.toml \
  --folds 5 \
  --seed 42
```

## Running

```bash
cargo run --release
```

Set at least:

```bash
export DISCORD_TOKEN='your-bot-token'
export SPECIMEN_HMAC_SECRET='at-least-32-bytes-long-random-secret'
export SIGHTLINE_CONFIG='./sightline.toml'
```

`OCR_SPACE_API_KEY` is optional unless OCR is enabled in a guild. When
`commands.register_on_startup = true`, commands are registered globally, so
invite the bot to each guild with the `bot` and `applications.commands` scopes,
create that guild's private `sightline-db` channel, and run `/config`.

## Building for deployment

Build a Linux release binary for the target architecture, usually
`x86_64-unknown-linux-musl` or `aarch64-unknown-linux-musl`.

The default release profile already enables optimized codegen, thin LTO, a single
codegen unit, stripped symbols, and abort-on-panic. If you build on the same
CPU family the binary will run on, add native CPU codegen:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

For a slower build that may squeeze out a little more runtime performance, use
the fat-LTO profile:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --profile release-fat-lto
```
