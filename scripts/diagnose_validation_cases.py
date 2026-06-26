#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CASES = [
    ("miss", Path("specimens/image_scam.jpg")),
    ("miss", Path("specimens/scam-3-0aa2acff1753ea065e245f953df5c969.png")),
    ("fp", Path("hard_negative_specimens/9508qpbf0f9h1.png")),
    ("fp", Path("hard_negative_specimens/Capture.PNG")),
    ("fp", Path("hard_negative_specimens/HK3zVQ1XAAAymIh.png")),
    ("fp", Path("hard_negative_specimens/HLqXMsNXIAAxTnO.png")),
    ("fp", Path("hard_negative_specimens/HLszLJJawAAbG1w.jpg")),
    ("fp", Path("hard_negative_specimens/IMG_9329.png")),
    ("fp", Path("hard_negative_specimens/Screenshot_20260616_131742_YouTube.jpg")),
    ("fp", Path("hard_negative_specimens/image31.png")),
    ("fp", Path("hard_negative_specimens/image4130.jpg")),
    ("fp", Path("hard_negative_specimens/image4444.png")),
    ("fp", Path("hard_negative_specimens/image5.png")),
    ("fp", Path("hard_negative_specimens/image5234.png")),
    ("fp", Path("hard_negative_specimens/omg-tricky-tony-done-it-again-v0-i2icautceb9h1.png")),
    ("fp", Path("hard_negative_specimens/omg-tricky-tony-done-it-again-v0-zcykfqtceb9h1.png")),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprints", default="target/sightline-plan-extend-kfold/fingerprints")
    parser.add_argument("--out-dir", default="target/diagnostics-current")
    parser.add_argument("--contact-sheet", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--sparse-sweep", action="store_true")
    args = parser.parse_args()

    fingerprint_root = Path(args.fingerprints)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, source in DEFAULT_CASES:
        fingerprint = find_fingerprint(fingerprint_root, source)
        row = {
            "label": label,
            "source": source.as_posix(),
            "fingerprint": None if fingerprint is None else fingerprint.as_posix(),
        }
        if fingerprint is not None:
            row.update(fingerprint_features(fingerprint))
        rows.append(row)

    (out_dir / "validation_case_features.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))

    if args.contact_sheet:
        write_contact_sheet(rows, out_dir / "fp_miss_contact_sheet.jpg")
    if args.summary:
        write_distribution_summary(fingerprint_root, out_dir / "validation_feature_summary.json")
    if args.sparse_sweep:
        write_sparse_sweep(fingerprint_root, out_dir / "sparse_dark_sweep.json")

    return 0


def find_fingerprint(root: Path, source: Path) -> Path | None:
    for directory in (root / "positives", root / "negatives"):
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            if path.name.startswith("."):
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if Path(raw.get("source_path", "")).as_posix() == source.as_posix():
                return path
    return None


def fingerprint_features(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    fp = raw["fingerprint"]
    visual = fp["visual"]
    text_grid = visual["text_grid"]
    rows = [sum(text_grid[i * 8 : (i + 1) * 8]) for i in range(8)]
    cols = [sum(text_grid[i::8]) for i in range(8)]
    total = max(sum(text_grid), 1)
    return {
        "width": fp["width"],
        "height": fp["height"],
        "aspect": max(fp["width"], fp["height"]) / max(min(fp["width"], fp["height"]), 1),
        "luma_mean": visual["luma_mean"],
        "luma_std": visual["luma_std"],
        "rgb_mean": visual["rgb_mean"],
        "text_mean": sum(text_grid) / len(text_grid),
        "text_regions": sum(1 for value in text_grid if value >= 64),
        "middle_percent": round(sum(rows[2:6]) * 100 / total, 2),
        "center_percent": round(sum(cols[2:6]) * 100 / total, 2),
        "edge_percent": round(max(sum(cols[0:2]), sum(cols[6:8])) * 100 / total, 2),
        "local_hashes": len(fp["local_hashes"]),
        "anchors": len(fp["local_anchors"]),
    }


def write_contact_sheet(rows: list[dict], out_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit("Pillow is required for --contact-sheet") from exc

    thumb_w, thumb_h, label_h = 220, 220, 62
    cols = 5
    rows_count = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows_count * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        source = Path(row["source"])
        image = Image.open(source).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x0 = (index % cols) * thumb_w
        y0 = (index // cols) * (thumb_h + label_h)
        sheet.paste(image, (x0 + (thumb_w - image.width) // 2, y0))
        label = f"{row['label']}: {source.name[:28]}"
        feature = f"L {row.get('luma_mean', '?')}/{row.get('luma_std', '?')} T {row.get('text_mean', 0):.1f}"
        draw.text((x0 + 4, y0 + thumb_h + 4), label, fill=(0, 0, 0))
        draw.text((x0 + 4, y0 + thumb_h + 24), feature, fill=(0, 0, 0))
    sheet.save(out_path, quality=90)
    print(out_path)


def write_distribution_summary(root: Path, out_path: Path) -> None:
    positives = all_fingerprint_rows(root / "positives")
    negatives = all_fingerprint_rows(root / "negatives")
    fp_sources = {source.as_posix() for _, source in DEFAULT_CASES if source.parts[0] == "hard_negative_specimens"}
    false_positives = [row for row in negatives if row["source"] in fp_sources]
    summary = {
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "false_positive_count": len(false_positives),
        "positive": summarize_rows(positives),
        "negative": summarize_rows(negatives),
        "false_positive": summarize_rows(false_positives),
        "lowest_text_positives": sorted(
            positives, key=lambda row: (row["text_mean"], row["text_regions"])
        )[:12],
        "false_positives": sorted(false_positives, key=lambda row: row["text_mean"]),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def all_fingerprint_rows(directory: Path) -> list[dict]:
    rows = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("."):
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        source = raw.get("source_path", "")
        features = fingerprint_features(path)
        features["source"] = Path(source).as_posix()
        rows.append(features)
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    return {
        key: summarize_values([row[key] for row in rows])
        for key in (
            "luma_mean",
            "luma_std",
            "text_mean",
            "text_regions",
            "middle_percent",
            "center_percent",
            "edge_percent",
            "local_hashes",
            "aspect",
        )
    }


def summarize_values(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p25": ordered[len(ordered) // 4],
        "median": ordered[len(ordered) // 2],
        "p75": ordered[(len(ordered) * 3) // 4],
        "max": ordered[-1],
    }


def write_sparse_sweep(root: Path, out_path: Path) -> None:
    positives = all_fingerprint_rows(root / "positives")
    negatives = all_fingerprint_rows(root / "negatives")
    configs = []
    for max_luma in (24, 28, 30, 35):
        for max_text_mean in (35, 45, 55, 65):
            for max_aspect in (1.3, 1.4, 1.5):
                configs.append(
                    {
                        "max_luma": max_luma,
                        "min_text_mean": 24,
                        "max_text_mean": max_text_mean,
                        "min_text_regions": 8,
                        "min_local_hashes": 200,
                        "max_aspect": max_aspect,
                    }
                )
    results = []
    for config in configs:
        pos_hits = [row["source"] for row in positives if sparse_dark_match(row, config)]
        neg_hits = [row["source"] for row in negatives if sparse_dark_match(row, config)]
        results.append({"config": config, "positive_hits": pos_hits, "negative_hits": neg_hits})
    results.sort(key=lambda row: (len(row["negative_hits"]), -len(row["positive_hits"])))
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results[:20], indent=2))


def sparse_dark_match(row: dict, config: dict) -> bool:
    return (
        row["luma_mean"] <= config["max_luma"]
        and config["min_text_mean"] <= row["text_mean"] <= config["max_text_mean"]
        and row["text_regions"] >= config["min_text_regions"]
        and row["local_hashes"] >= config["min_local_hashes"]
        and row["aspect"] <= config["max_aspect"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
