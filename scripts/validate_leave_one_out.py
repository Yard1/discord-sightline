#!/usr/bin/env python3
"""Leave-one-out validation for Sightline specimens.

For every positive specimen fingerprint, this script trains on every other
positive specimen and tests the held-out one. It shells out to the Rust local CLI
for hashing and comparison, matching production/local behavior by construction.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from validate_kfold import (
    Confusion,
    cached_fingerprint_files,
    compare_summary,
    copy_jsons,
    ensure_fingerprint_cache,
    fingerprint_sources,
    predicted_sources_from_categories,
    prediction_categories,
    recreate_dir,
    run_compare,
    sightline_command,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run leave-one-out validation: all specimens but one are train, "
            "the held-out specimen is test."
        )
    )
    parser.add_argument("--specimens", required=True, help="Directory of positive specimen images")
    parser.add_argument(
        "--negatives",
        help=(
            "Optional directory of known-negative images. If set, every "
            "leave-one-out train set is also compared against all negatives."
        ),
    )
    parser.add_argument(
        "--work-dir",
        default="target/sightline-leave-one-out",
        help="Cache/work directory, default target/sightline-leave-one-out",
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
        help="Keep generated per-held-out train/test directories",
    )
    args = parser.parse_args()

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
    if len(positive_files) < 2:
        raise SystemExit(
            f"need at least 2 positive fingerprints, got {len(positive_files)}"
        )

    negative_files: list[Path] = []
    if args.negatives:
        negative_cache = cache_dir / "negatives"
        ensure_fingerprint_cache(
            command, Path(args.negatives), negative_cache, args.refresh_cache
        )
        negative_files = cached_fingerprint_files(negative_cache)

    aggregate = Confusion()
    held_out_reports = []
    fully_matched_specimens = []
    suspicious_only_specimens = []
    unmatched_specimens = []
    false_positive_sources_all_folds: set[str] = set()
    fully_matched_false_positive_sources_all_folds: set[str] = set()
    suspicious_only_false_positive_sources_all_folds: set[str] = set()

    for held_out_index, held_out_file in enumerate(positive_files):
        held_out_dir = folds_dir / f"held_out_{held_out_index + 1:04d}"
        train_dir = held_out_dir / "train"
        positive_dir = held_out_dir / "positive"
        negative_dir = held_out_dir / "negative"
        recreate_dir(train_dir)
        recreate_dir(positive_dir)
        recreate_dir(negative_dir)

        train_files = [
            file_path for file_path in positive_files if file_path != held_out_file
        ]
        copy_jsons(train_files, train_dir)
        copy_jsons([held_out_file], positive_dir)
        copy_jsons(negative_files, negative_dir)

        held_out_source = single_fingerprint_source(held_out_file)
        positive_compare = run_compare(command, train_dir, positive_dir)
        positive_categories = prediction_categories(positive_compare)
        positive_predictions = predicted_sources_from_categories(
            positive_categories, args.positive_mode
        )
        fully_matched = held_out_source in positive_categories["matched"]
        suspicious_only = (
            held_out_source in positive_categories["suspicious"]
            and held_out_source not in positive_categories["matched"]
        )
        positive_prediction = held_out_source in positive_predictions

        held_out_confusion = Confusion()
        if positive_prediction:
            held_out_confusion.tp = 1
        else:
            held_out_confusion.fn = 1
            unmatched_specimens.append(held_out_source)
        if fully_matched:
            fully_matched_specimens.append(held_out_source)
        elif suspicious_only:
            suspicious_only_specimens.append(held_out_source)

        negative_compare = None
        false_positive_sources = []
        fully_matched_false_positive_sources = []
        suspicious_only_false_positive_sources = []
        true_negative_count = 0
        if negative_files:
            negative_compare = run_compare(command, train_dir, negative_dir)
            negative_categories = prediction_categories(negative_compare)
            negative_predictions = predicted_sources_from_categories(
                negative_categories, args.positive_mode
            )
            negative_sources = fingerprint_sources(negative_files)
            false_positive_sources = sorted(negative_sources & negative_predictions)
            fully_matched_false_positive_sources = sorted(
                negative_sources & negative_categories["matched"]
            )
            suspicious_only_false_positive_sources = sorted(
                negative_sources
                & negative_categories["suspicious"]
                - negative_categories["matched"]
            )
            false_positive_sources_all_folds.update(false_positive_sources)
            fully_matched_false_positive_sources_all_folds.update(
                fully_matched_false_positive_sources
            )
            suspicious_only_false_positive_sources_all_folds.update(
                suspicious_only_false_positive_sources
            )
            held_out_confusion.fp = len(false_positive_sources)
            held_out_confusion.tn = len(negative_sources) - held_out_confusion.fp
            true_negative_count = held_out_confusion.tn

        aggregate.add(held_out_confusion)
        held_out_report = {
            "held_out_index": held_out_index + 1,
            "held_out_source": held_out_source,
            "fully_matched": fully_matched,
            "suspicious_only": suspicious_only,
            "positive_prediction": positive_prediction,
            "train_count": len(train_files),
            "negative_test_count": len(negative_files),
            "metrics": held_out_confusion.metrics(),
            "positive_compare_summary": compare_summary(positive_compare),
            "negative_compare_summary": compare_summary(negative_compare),
            "false_positive_sources": false_positive_sources,
            "fully_matched_false_positive_sources": fully_matched_false_positive_sources,
            "suspicious_only_false_positive_sources": suspicious_only_false_positive_sources,
            "true_negative_count": true_negative_count,
        }
        held_out_reports.append(held_out_report)
        (results_dir / f"held_out_{held_out_index + 1:04d}.json").write_text(
            json.dumps(held_out_report, indent=2), encoding="utf-8"
        )

    report = {
        "mode": "leave_one_out",
        "specimens": str(Path(args.specimens)),
        "negatives": None if not args.negatives else str(Path(args.negatives)),
        "positive_mode": args.positive_mode,
        "positive_count": len(positive_files),
        "negative_count": len(negative_files),
        "aggregate": aggregate.metrics(),
        "fully_matched_specimens_count": len(fully_matched_specimens),
        "fully_matched_specimens": sorted(fully_matched_specimens),
        "suspicious_only_specimens_count": len(suspicious_only_specimens),
        "suspicious_only_specimens": sorted(suspicious_only_specimens),
        "unmatched_specimens_count": len(unmatched_specimens),
        "unmatched_specimens": sorted(unmatched_specimens),
        "unique_false_positive_sources_count": len(false_positive_sources_all_folds),
        "unique_false_positive_sources": sorted(false_positive_sources_all_folds),
        "fully_matched_false_positive_sources_count": len(
            fully_matched_false_positive_sources_all_folds
        ),
        "fully_matched_false_positive_sources": sorted(
            fully_matched_false_positive_sources_all_folds
        ),
        "suspicious_only_false_positive_sources_count": len(
            suspicious_only_false_positive_sources_all_folds
        ),
        "suspicious_only_false_positive_sources": sorted(
            suspicious_only_false_positive_sources_all_folds
        ),
        "held_out_reports": held_out_reports,
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


def single_fingerprint_source(file_path: Path) -> str:
    sources = fingerprint_sources([file_path])
    if len(sources) != 1:
        raise RuntimeError(f"expected one source in {file_path}, got {len(sources)}")
    return next(iter(sources))


if __name__ == "__main__":
    raise SystemExit(main())
