# Image pipeline

The image path is a layered fingerprint-and-match pipeline. Most of it lives in
[src/image/pipeline.rs](src/image/pipeline.rs#L460),
[src/image/matcher.rs](src/image/matcher.rs#L681), and
[src/image/geometry.rs](src/image/geometry.rs#L215).

The design is deliberately conservative, and each stage is cheaper than the one
after it. Exact and perceptual checks catch obvious duplicates. Local ORB-like
matching handles resized or mildly transformed reposts. Geometry verification
stops descriptor coincidences from turning into enforcement. And the
suspicious-only visual and OCR paths add coverage without letting weak signals
trigger confirmed actions on their own.

## 1. Input validation and download

Images are accepted only from Discord HTTPS hosts, and only within limits on
redirects, MIME type, byte size, and decoded pixel count. This keeps the scanner
bounded and avoids processing arbitrary external URLs.

When the source image is large enough to benefit from resizing, Sightline may
request a Discord proxy preview. The target dimensions come from the same local
matching limits used elsewhere, so preview fingerprints stay comparable to
normalized candidate fingerprints. Preview and original scans race each other:
a decisive preview match can still act early, but a faster original scan cancels
the preview path and uses the full-image result.

Downloads go through a shared in-flight map keyed by canonical Discord URL,
effective byte cap, and MIME hint. This coalesces duplicate simultaneous
attachment downloads without changing the later exact-byte hash semantics.

The image HTTP client also runs an optional Discord CDN connection warmer. It
uses the same reqwest client as real image downloads and sends a cheap `HEAD`
request to a permanent CDN avatar only when recent real downloads have not
touched `cdn.discordapp.com`. Warmer requests bypass the download semaphore and
image timing metrics; their debug log elapsed time is only a connection-pool
health probe.

## 2. Fingerprint construction

`hash_downloaded_image` runs decode and hashing on a blocking task, behind the
shared CPU semaphore that also gates matcher work. The resulting
`ImageFingerprint` holds `byte_xxh128`,
dimensions, MIME type, `phash64`, `dhash64`, a visual signature, and — depending
on the `HashMode` — either specimen anchors or candidate local hashes.

The decode path:

- computes a byte-level `xxh3_128` exact hash,
- decodes the image under allocation and pixel limits,
- creates a 512px-max RGB/luma thumbnail,
- builds the visual summary features,
- computes the perceptual pHash and dHash,
- optionally normalizes luma for local matching,
- builds an 8×8 text-density grid,
- and extracts local ORB-like features.

## 3. Global visual features

`visual_signature` records luma mean and standard deviation, RGB mean, and a 4×4
luma grid ([src/image/pipeline.rs](src/image/pipeline.rs#L623)). The text grid
estimates how many scanned local tiles look text-like in each of 8×8 regions
([src/image/pipeline.rs](src/image/pipeline.rs#L747)). Both are cheap, coarse
signals, used for scoring and for suspicious visual-shape detection.

## 4. Perceptual hashes

The pipeline uses two 64-bit hashes:

- **pHash** — a median hash with DCT preprocessing.
- **dHash** — a gradient hash.

A match requires both hashes to be close, not just one. The confirmed defaults
are pHash ≤ 16, dHash ≤ 12, and total ≤ 26; the suspicious point loosens this to
dHash ≤ 15 and total ≤ 30 ([src/configuration/app.rs](src/configuration/app.rs#L331)).
This balances tolerance for compression and resizing against the risk of broad
false positives.

## 5. Local feature extraction

Local features are ORB-like:

- FAST-style corner detection on luma pixels,
- tile filtering by mean brightness, contrast, and edge density,
- non-max suppression to spread keypoints out,
- orientation from intensity moments,
- two 64-bit BRIEF descriptors, sampled with deterministic random patterns
  ([src/image/pipeline.rs](src/image/pipeline.rs#L1287)).

Specimens and candidates store different things, on purpose. Specimens keep a set
of selected `ImageAnchor`s, while candidates keep the scanned `LocalImageHash`
values ([src/image/pipeline.rs](src/image/pipeline.rs#L569)). This asymmetry keeps
specimen records compact while still giving live candidates enough search
material to locate anchors after mild transformations.

## 6. Matcher indexing

`Matcher` indexes specimens several ways
([src/image/matcher.rs](src/image/matcher.rs#L268)):

- by exact byte hash,
- by pHash/dHash segment buckets,
- by local anchor descriptor buckets,
- by dense-local anchor descriptor buckets,
- and through separate preview-specimen indexes.

These segment indexes are candidate generators, not proof. They cheaply narrow
the set of specimens before the full Hamming, visual, and geometry checks run.

## 7. Decision order

For each candidate, the matcher checks the confirmed rules first, then the
suspicious rules ([src/image/matcher.rs](src/image/matcher.rs#L681)), in this
order:

- exact `xxh128`,
- perceptual hash,
- local anchors,
- dense local anchors,
- suspicious visual-shape signal,
- optional cluster-coherence support.

Exact matches short-circuit only for original fingerprints, not for Discord
preview variants.

## 8. Local matching and geometry

Local matching has two related evidence paths. The selected local-anchor path
uses curated specimen anchors and compares both 64-bit BRIEF descriptors. The
dense-local path uses the broader stored local hashes as an additional,
separately scored descriptor stage, which is useful when the selected anchors do
not provide enough coverage. Both paths are backed by the same ORB-like feature
extractor, first pull likely specimen candidates from descriptor buckets, and
then verify correspondences with luma, contrast, edge-density, and position
gates.

Verified hits then go through geometric verification
([src/image/matcher.rs](src/image/matcher.rs#L3013)). The geometry engine starts
with a similarity-transform vote, can upgrade to an affine or homography fit, and
falls back to a bounded PROSAC homography
([src/image/geometry.rs](src/image/geometry.rs#L215)). A match has to pass gates
on inlier count, distinct regions, spread, residual, and mean Hamming distance.
This is the strongest defense against accidental matches from repeated UI
or generic blocks of text.

## 9. Scoring

The per-stage scores are combined per specimen. By default, perceptual evidence
is weighted at `1.8`, selected local anchors at `2.2`, dense local anchors at
`1.4`, and visual shape at `1.0`; the confirmed threshold is `63` and the
suspicious threshold is `20`
([src/configuration/app.rs](src/configuration/app.rs#L308)). The perceptual score
rises as total Hamming distance falls, and the local score rewards hits, region
spread, spatial spread, low descriptor distance, and low residual
([src/image/matcher.rs](src/image/matcher.rs#L2596)).

## 10. Cluster coherence

A suspicious match can gain cluster-coherence support when several matched
specimens are mutually coherent in a precomputed specimen-to-specimen graph
([src/image/knn.rs](src/image/knn.rs#L1)). This is an OCR-backed escalation
signal by default, not a destructive confirmed-action shortcut. To stop weak
"shared UI" overlaps from forming a cluster, each member must exceed a ceiling
before it can take part.

## 11. OCR / text gate

Visual matching always runs first. If the result is suspicious and text-gating is
enabled, the worker prepares an OCR crop from the densest text-like region,
within size and byte limits ([src/image/pipeline.rs](src/image/pipeline.rs#L1014)).
OCR then checks the configured sentences and keywords using a bounded edit
distance ([src/image/engine.rs](src/image/engine.rs#L494)). Running visual
matching first means Sightline only pays for OCR when the visual evidence already
makes it worthwhile.
