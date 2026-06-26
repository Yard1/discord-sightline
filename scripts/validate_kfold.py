#!/usr/bin/env python3
"""Seeded K-fold validation for Sightline image specimens.

The script intentionally shells out to the Rust local CLI for hashing and
comparison so validation uses the same image pipeline and matcher code as the
bot/local commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None


CACHE_SCHEMA = 3

FINGERPRINT_DOWNLOAD_FIELDS = ("max_decoded_pixels",)
FINGERPRINT_MATCH_FIELDS = (
    "local_max_width",
    "local_max_height",
    "local_max_area",
    "local_max_aspect_ratio",
    "local_tile_width",
    "local_tile_height",
    "local_stride",
    "local_tile_budget",
    "local_hash_cap",
    "local_anchor_count",
    "local_anchor_max_distance",
)


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, other: "Confusion") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.tn += other.tn
        self.fn += other.fn

    def metrics(self) -> dict:
        total = self.tp + self.fp + self.tn + self.fn
        precision = safe_div(self.tp, self.tp + self.fp)
        recall = safe_div(self.tp, self.tp + self.fn)
        specificity = safe_div(self.tn, self.tn + self.fp)
        accuracy = safe_div(self.tp + self.tn, total)
        f1 = (
            None
            if precision is None or recall is None or precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        false_positive_rate = safe_div(self.fp, self.fp + self.tn)
        false_negative_rate = safe_div(self.fn, self.fn + self.tp)
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "total": total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run seeded K-fold validation using Sightline local CLI outputs."
    )
    parser.add_argument("--specimens", required=True, help="Directory of positive specimen images")
    parser.add_argument(
        "--negatives",
        help="Optional directory of known-negative images for precision/accuracy/FPR",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of folds, default 5")
    parser.add_argument("--seed", type=int, default=1, help="Shuffle seed, default 1")
    parser.add_argument(
        "--work-dir",
        default="target/sightline-validation",
        help="Cache/work directory, default target/sightline-validation",
    )
    parser.add_argument(
        "--binary",
        default=None,
        help="Path to compiled discord-sightline binary. Defaults to cargo run --release --",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rehash image folders even if cached fingerprints already exist",
    )
    parser.add_argument(
        "--positive-mode",
        choices=["matched", "matched-or-suspicious"],
        default="matched-or-suspicious",
        help="Which compare-image-sets result counts as a positive prediction",
    )
    parser.add_argument(
        "--keep-fold-dirs",
        action="store_true",
        help="Keep generated per-fold train/test directories",
    )
    args = parser.parse_args()

    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")

    repo = Path.cwd()
    work_dir = Path(args.work_dir)
    cache_dir = work_dir / "fingerprints"
    folds_dir = work_dir / "folds"
    results_dir = work_dir / "results"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    if folds_dir.exists() and not args.keep_fold_dirs:
        shutil.rmtree(folds_dir)
    folds_dir.mkdir(parents=True, exist_ok=True)

    command = sightline_command(args.binary)

    positive_cache = cache_dir / "positives"
    ensure_fingerprint_cache(
        command, Path(args.specimens), positive_cache, args.refresh_cache
    )
    positive_files = cached_fingerprint_files(positive_cache)
    if len(positive_files) < args.folds:
        raise SystemExit(
            f"need at least {args.folds} positive fingerprints, got {len(positive_files)}"
        )

    negative_files: list[Path] = []
    if args.negatives:
        negative_cache = cache_dir / "negatives"
        ensure_fingerprint_cache(
            command, Path(args.negatives), negative_cache, args.refresh_cache
        )
        negative_files = cached_fingerprint_files(negative_cache)

    rng = random.Random(args.seed)
    positive_folds = split_folds(shuffled(positive_files, rng), args.folds)
    negative_folds = split_folds(shuffled(negative_files, rng), args.folds)

    aggregate = Confusion()
    fold_reports = []
    fully_matched_positive_sources_all: set[str] = set()
    suspicious_only_positive_sources_all: set[str] = set()
    pass_positive_sources_all: set[str] = set()
    false_positive_sources_all: set[str] = set()
    fully_matched_false_positive_sources_all: set[str] = set()
    suspicious_only_false_positive_sources_all: set[str] = set()
    for fold_index in range(args.folds):
        fold_dir = folds_dir / f"fold_{fold_index + 1:02d}"
        train_dir = fold_dir / "train"
        positive_dir = fold_dir / "positive"
        negative_dir = fold_dir / "negative"
        recreate_dir(train_dir)
        recreate_dir(positive_dir)
        recreate_dir(negative_dir)

        train_files = [
            item
            for index, fold in enumerate(positive_folds)
            if index != fold_index
            for item in fold
        ]
        positive_test_files = positive_folds[fold_index]
        negative_test_files = (
            negative_folds[fold_index] if negative_folds else []
        )

        copy_jsons(train_files, train_dir)
        copy_jsons(positive_test_files, positive_dir)
        copy_jsons(negative_test_files, negative_dir)

        positive_compare = run_compare(command, train_dir, positive_dir)
        positive_categories = prediction_categories(positive_compare)
        positive_predictions = predicted_sources_from_categories(
            positive_categories, args.positive_mode
        )
        positive_sources = fingerprint_sources(positive_test_files)
        fully_matched_positive_sources = sorted(
            positive_sources & positive_categories["matched"]
        )
        suspicious_only_positive_sources = sorted(
            positive_sources
            & positive_categories["suspicious"]
            - positive_categories["matched"]
        )
        true_positive_sources = sorted(positive_sources & positive_predictions)
        false_negative_sources = sorted(positive_sources - positive_predictions)
        pass_positive_sources = sorted(
            positive_sources
            - positive_categories["matched"]
            - positive_categories["suspicious"]
        )
        fully_matched_positive_sources_all.update(fully_matched_positive_sources)
        suspicious_only_positive_sources_all.update(suspicious_only_positive_sources)
        pass_positive_sources_all.update(pass_positive_sources)

        fold_confusion = Confusion()
        for source in positive_sources:
            if source in positive_predictions:
                fold_confusion.tp += 1
            else:
                fold_confusion.fn += 1

        negative_compare = None
        negative_predictions: set[str] = set()
        negative_sources: set[str] = set()
        negative_categories = empty_prediction_categories()
        if negative_test_files:
            negative_compare = run_compare(command, train_dir, negative_dir)
            negative_categories = prediction_categories(negative_compare)
            negative_predictions = predicted_sources_from_categories(
                negative_categories, args.positive_mode
            )
            negative_sources = fingerprint_sources(negative_test_files)
            for source in negative_sources:
                if source in negative_predictions:
                    fold_confusion.fp += 1
                else:
                    fold_confusion.tn += 1
        false_positive_sources = sorted(negative_sources & negative_predictions)
        true_negative_sources = sorted(negative_sources - negative_predictions)
        fully_matched_false_positive_sources = sorted(
            negative_sources & negative_categories["matched"]
        )
        suspicious_only_false_positive_sources = sorted(
            negative_sources
            & negative_categories["suspicious"]
            - negative_categories["matched"]
        )
        false_positive_sources_all.update(false_positive_sources)
        fully_matched_false_positive_sources_all.update(fully_matched_false_positive_sources)
        suspicious_only_false_positive_sources_all.update(suspicious_only_false_positive_sources)

        aggregate.add(fold_confusion)
        fold_report = {
            "fold": fold_index + 1,
            "train_count": len(train_files),
            "positive_test_count": len(positive_test_files),
            "negative_test_count": len(negative_test_files),
            "metrics": fold_confusion.metrics(),
            "positive_prediction_count": len(positive_predictions),
            "negative_prediction_count": len(negative_predictions),
            "fully_matched_positive_sources": fully_matched_positive_sources,
            "suspicious_only_positive_sources": suspicious_only_positive_sources,
            "pass_positive_sources": pass_positive_sources,
            "true_positive_sources": true_positive_sources,
            "false_negative_sources": false_negative_sources,
            "false_positive_sources": false_positive_sources,
            "fully_matched_false_positive_sources": fully_matched_false_positive_sources,
            "suspicious_only_false_positive_sources": suspicious_only_false_positive_sources,
            "true_negative_count": len(true_negative_sources),
            "positive_compare_summary": compare_summary(positive_compare),
            "negative_compare_summary": compare_summary(negative_compare),
        }
        fold_reports.append(fold_report)
        (results_dir / f"fold_{fold_index + 1:02d}.json").write_text(
            json.dumps(fold_report, indent=2), encoding="utf-8"
        )

    report = {
        "specimens": str(Path(args.specimens)),
        "negatives": None if not args.negatives else str(Path(args.negatives)),
        "folds": args.folds,
        "seed": args.seed,
        "positive_mode": args.positive_mode,
        "positive_count": len(positive_files),
        "negative_count": len(negative_files),
        "aggregate": aggregate.metrics(),
        "fully_matched_positive_sources_count": len(fully_matched_positive_sources_all),
        "fully_matched_positive_sources": sorted(fully_matched_positive_sources_all),
        "suspicious_only_positive_sources_count": len(suspicious_only_positive_sources_all),
        "suspicious_only_positive_sources": sorted(suspicious_only_positive_sources_all),
        "unmatched_positive_sources_count": len(pass_positive_sources_all),
        "unmatched_positive_sources": sorted(pass_positive_sources_all),
        "false_positive_sources_count": len(false_positive_sources_all),
        "false_positive_sources": sorted(false_positive_sources_all),
        "fully_matched_false_positive_sources_count": len(
            fully_matched_false_positive_sources_all
        ),
        "fully_matched_false_positive_sources": sorted(
            fully_matched_false_positive_sources_all
        ),
        "suspicious_only_false_positive_sources_count": len(
            suspicious_only_false_positive_sources_all
        ),
        "suspicious_only_false_positive_sources": sorted(
            suspicious_only_false_positive_sources_all
        ),
        "fold_reports": fold_reports,
        "work_dir": str(work_dir),
        "command": command,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))

    if folds_dir.exists() and not args.keep_fold_dirs:
        shutil.rmtree(folds_dir)
    return 0


def sightline_command(binary: str | None) -> list[str]:
    if binary:
        return [binary]
    return ["cargo", "run", "--release", "--"]


def ensure_fingerprint_cache(
    command: list[str], input_path: Path, output_dir: Path, refresh: bool
) -> None:
    metadata_path = output_dir / ".sightline-cache.json"
    expected_metadata = fingerprint_cache_metadata(command, input_path)
    if refresh and output_dir.exists():
        shutil.rmtree(output_dir)
    if (
        output_dir.exists()
        and cached_fingerprint_files(output_dir)
        and cache_metadata_matches(metadata_path, expected_metadata)
    ):
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_json(command + ["hash-images", str(input_path), str(output_dir)])
    metadata_path.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")


def cache_metadata_matches(metadata_path: Path, expected_metadata: dict) -> bool:
    if not metadata_path.exists():
        return False
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return existing == expected_metadata


def fingerprint_cache_metadata(command: list[str], input_path: Path) -> dict:
    config_path = Path(os.environ.get("SIGHTLINE_CONFIG", "sightline.toml"))
    return {
        "schema": CACHE_SCHEMA,
        "command": command,
        "command_fingerprint": command_fingerprint(command),
        "fingerprint_config": fingerprint_config_fingerprint(config_path),
        "input": input_fingerprint(input_path),
    }


def fingerprint_config_fingerprint(config_path: Path) -> dict:
    if tomllib is None or not config_path.exists() or not config_path.is_file():
        return {"kind": "file", **file_fingerprint(config_path)}
    raw = config_path.read_bytes()
    parsed = tomllib.loads(raw.decode("utf-8"))
    download = parsed.get("download", {})
    matching = parsed.get("match", {})
    return {
        "kind": "fingerprint-fields",
        "exists": True,
        "schema": 1,
        "download": {
            field: download.get(field) for field in FINGERPRINT_DOWNLOAD_FIELDS
        },
        "match": {field: matching.get(field) for field in FINGERPRINT_MATCH_FIELDS},
    }


def command_fingerprint(command: list[str]) -> dict:
    if not command:
        return {"kind": "empty"}
    binary_path = Path(command[0])
    if binary_path.exists() and binary_path.is_file():
        return {"kind": "file", **file_fingerprint(binary_path)}
    return {"kind": "argv", "argv": command}


def file_fingerprint(path: Path) -> dict:
    if not path.exists() or not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "size": None,
            "mtime_ns": None,
            "sha256": None,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def input_fingerprint(path: Path) -> dict:
    if path.is_file():
        files = [path]
        root = path.parent
    else:
        files = sorted(file_path for file_path in path.rglob("*") if file_path.is_file())
        root = path

    entries = []
    for file_path in files:
        stat = file_path.stat()
        try:
            relative_path = file_path.relative_to(root)
        except ValueError:
            relative_path = file_path
        entries.append(
            {
                "path": relative_path.as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(path),
        "file_count": len(entries),
        "inventory_sha256": digest,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_compare(
    command: list[str],
    specimen_dir: Path,
    candidate_dir: Path,
) -> dict:
    compare_command = command + ["compare-image-sets", str(specimen_dir), str(candidate_dir)]
    return run_json(compare_command)


def cached_fingerprint_files(directory: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in directory.glob("*.json")
        if not file_path.name.startswith(".")
    )


def run_json(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            f"command: {' '.join(command)}\n"
            f"exit: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"command did not emit JSON: {' '.join(command)}\n{completed.stdout}"
        ) from exc


def split_folds(items: list[Path], folds: int) -> list[list[Path]]:
    buckets = [[] for _ in range(folds)]
    for index, item in enumerate(items):
        buckets[index % folds].append(item)
    return buckets


def shuffled(items: list[Path], rng: random.Random) -> list[Path]:
    copied = list(items)
    rng.shuffle(copied)
    return copied


def recreate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_jsons(files: Iterable[Path], output_dir: Path) -> None:
    for file_path in files:
        shutil.copy2(file_path, output_dir / file_path.name)


def predicted_sources(compare_report: dict, mode: str) -> set[str]:
    return predicted_sources_from_categories(prediction_categories(compare_report), mode)


def empty_prediction_categories() -> dict[str, set[str]]:
    return {"matched": set(), "suspicious": set()}


def prediction_categories(compare_report: dict) -> dict[str, set[str]]:
    categories = empty_prediction_categories()
    for match in compare_report.get("matches", []):
        source = match["candidate_source"]
        if bool(match.get("matched")):
            categories["matched"].add(source)
        if bool(match.get("suspicious")):
            categories["suspicious"].add(source)
    return categories


def predicted_sources_from_categories(categories: dict[str, set[str]], mode: str) -> set[str]:
    if mode == "matched":
        return set(categories["matched"])
    return set(categories["matched"]) | set(categories["suspicious"])


def fingerprint_sources(files: Iterable[Path]) -> set[str]:
    sources = set()
    for file_path in files:
        record = json.loads(file_path.read_text(encoding="utf-8"))
        sources.add(record["source_path"])
    return sources


def compare_summary(compare_report: dict | None) -> dict | None:
    if compare_report is None:
        return None
    return {
        "specimen_count": compare_report.get("specimen_count"),
        "candidate_count": compare_report.get("candidate_count"),
        "comparisons_total": compare_report.get("comparisons_total"),
        "match_rows": len(compare_report.get("matches", [])),
        "best_per_candidate": len(compare_report.get("best_per_candidate", [])),
    }


def safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    value = numerator / denominator
    if math.isnan(value) or math.isinf(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
