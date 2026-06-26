#!/usr/bin/env python3
"""Source-of-truth validation runner for Sightline matcher tuning.

This script deliberately does not mutate hyperparameters. It runs the same
K-fold and leave-one-out validators against one or more explicit TOML configs,
then applies shared quality gates. That keeps tuning auditable: config changes
are made by humans, and this script reports whether they generalize across the
positive folds and the local hard-negative set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from validate_kfold import ensure_fingerprint_cache, fingerprint_cache_metadata


DEFAULT_WORK_DIR = Path("target/sightline-tuning-validation")


@dataclass
class GateConfig:
    min_precision: float
    min_specificity: float
    max_soft_false_positive_sources: int
    min_hard_positive_ratio: float
    max_augmented_unmatched: int
    min_augmented_hard_ratio: float


@dataclass
class GateResult:
    name: str
    passed: bool
    actual: Any
    expected: str
    details: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Sightline K-fold and leave-one-out validation as one auditable "
            "parameter-tuning protocol."
        )
    )
    parser.add_argument(
        "--specimens",
        default="specimens",
        help="Directory of positive specimen images, default ./specimens",
    )
    parser.add_argument(
        "--negatives",
        default="hard_negative_specimens",
        help=(
            "Directory of local hard negatives, default ./hard_negative_specimens. "
            "Pass an empty string to disable negative validation."
        ),
    )
    parser.add_argument("--folds", type=int, default=5, help="K-fold count, default 5")
    parser.add_argument("--seed", type=int, default=42, help="K-fold shuffle seed, default 42")
    parser.add_argument(
        "--work-dir",
        default=str(DEFAULT_WORK_DIR),
        help=f"Output/cache root, default {DEFAULT_WORK_DIR}",
    )
    parser.add_argument(
        "--binary",
        default="target/release/discord-sightline",
        help=(
            "Path to compiled discord-sightline binary. Defaults to "
            "target/release/discord-sightline; use --no-build only if it already exists."
        ),
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Do not run cargo build --release before validation.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force rehashing image folders in the underlying validators.",
    )
    parser.add_argument(
        "--augment-profile",
        choices=["none", "mild", "geometry", "full"],
        default="none",
        help=(
            "Optionally generate deterministic transformed positives and report "
            "original-specimen coverage against them. Defaults to none."
        ),
    )
    parser.add_argument(
        "--augment-jpeg-quality",
        type=int,
        default=86,
        help="JPEG quality for --augment-profile outputs, default 86.",
    )
    parser.add_argument(
        "--keep-fold-dirs",
        action="store_true",
        help="Keep generated fold directories from underlying validators in --legacy-validation mode.",
    )
    parser.add_argument(
        "--legacy-validation",
        action="store_true",
        help=(
            "Run the older Python per-config/per-fold validators. By default, "
            "fingerprints are cached once and threshold sweeps run in the Rust CLI."
        ),
    )
    parser.add_argument(
        "--positive-mode",
        choices=["matched", "matched-or-suspicious"],
        default="matched-or-suspicious",
        help="Prediction mode for positives, default matched-or-suspicious.",
    )
    parser.add_argument(
        "--evaluate-all-stages",
        action="store_true",
        help=(
            "Ask the Rust validators to run every configured matcher stage for "
            "diagnostics instead of stopping at the first production decision."
        ),
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "TOML config to validate. May be passed multiple times. If omitted, "
            "uses SIGHTLINE_CONFIG when set, otherwise sightline.toml."
        ),
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.99,
        help="Required aggregate precision when negatives are present, default 0.99.",
    )
    parser.add_argument(
        "--min-specificity",
        type=float,
        default=0.99,
        help="Required aggregate specificity when negatives are present, default 0.99.",
    )
    parser.add_argument(
        "--max-soft-false-positive-sources",
        type=int,
        default=2,
        help=(
            "Maximum unique hard-negative sources allowed to be suspicious-only "
            "in each validation mode, default 2."
        ),
    )
    parser.add_argument(
        "--min-hard-positive-ratio",
        type=float,
        default=0.50,
        help=(
            "Minimum fraction of known positives that should be hard matched "
            "rather than suspicious-only, default 0.50."
        ),
    )
    parser.add_argument(
        "--max-augmented-unmatched",
        type=int,
        default=0,
        help=(
            "When --augment-profile is enabled, maximum transformed positives allowed "
            "to be neither matched nor suspicious. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--min-augmented-hard-ratio",
        type=float,
        default=0.0,
        help=(
            "When --augment-profile is enabled, optional minimum fraction of transformed "
            "positives that should be hard matched. Defaults to 0.0."
        ),
    )
    args = parser.parse_args()

    repo = Path.cwd()
    specimens = Path(args.specimens)
    negatives = Path(args.negatives) if args.negatives else None
    if not specimens.exists():
        raise SystemExit(f"specimens directory does not exist: {specimens}")
    if negatives is not None and not negatives.exists():
        raise SystemExit(f"negative directory does not exist: {negatives}")
    if not 1 <= args.augment_jpeg_quality <= 100:
        raise SystemExit("--augment-jpeg-quality must be between 1 and 100")
    if args.max_augmented_unmatched < 0:
        raise SystemExit("--max-augmented-unmatched must be non-negative")
    if not 0.0 <= args.min_augmented_hard_ratio <= 1.0:
        raise SystemExit("--min-augmented-hard-ratio must be between 0 and 1")
    if args.legacy_validation and args.augment_profile != "none":
        raise SystemExit("--augment-profile is only supported by the default fast sweep mode")

    binary = Path(args.binary)
    if not args.no_build:
        run_command(["cargo", "build", "--release"], cwd=repo)
    elif not binary.exists():
        raise SystemExit(f"binary does not exist and --no-build was set: {binary}")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    configs = config_inputs(args.config)
    gates = GateConfig(
        min_precision=args.min_precision,
        min_specificity=args.min_specificity,
        max_soft_false_positive_sources=args.max_soft_false_positive_sources,
        min_hard_positive_ratio=args.min_hard_positive_ratio,
        max_augmented_unmatched=args.max_augmented_unmatched,
        min_augmented_hard_ratio=args.min_augmented_hard_ratio,
    )

    if args.legacy_validation:
        runs = []
        for config_path in configs:
            run = run_validation_for_config(
                config_path=config_path,
                args=args,
                binary=binary,
                specimens=specimens,
                negatives=negatives,
                work_dir=work_dir,
                gates=gates,
            )
            runs.append(run)
        validation_mode = "legacy_python_validators"
    else:
        runs = run_fast_threshold_sweep(
            configs=configs,
            args=args,
            binary=binary,
            specimens=specimens,
            negatives=negatives,
            work_dir=work_dir,
            gates=gates,
        )
        validation_mode = "rust_threshold_sweep"

    report = {
        "schema": 1,
        "intent": {
            "positive_requirement": "all specimens hard or soft matched; hard matches preferred",
            "negative_requirement": (
                "no hard-negative hard matches; suspicious-only hard-negative hits near zero"
            ),
            "tuning_policy": (
                "evaluate explicit configs; do not auto-fit values to the current hard-negative set"
            ),
        },
        "inputs": {
            "specimens": str(specimens),
            "negatives": None if negatives is None else str(negatives),
            "folds": args.folds,
            "seed": args.seed,
            "positive_mode": args.positive_mode,
            "evaluate_all_stages": args.evaluate_all_stages,
            "augment_profile": args.augment_profile,
            "augment_jpeg_quality": args.augment_jpeg_quality,
            "binary": str(binary),
            "validation_mode": validation_mode,
        },
        "gates": asdict(gates),
        "runs": runs,
        "passed": all(run["passed"] for run in runs),
    }

    summary_path = work_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = work_dir / "summary.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(render_console_summary(report, summary_path, markdown_path))
    return 0 if report["passed"] else 1


def config_inputs(values: list[str]) -> list[Path]:
    if values:
        return [Path(value) for value in values]
    return [Path(os.environ.get("SIGHTLINE_CONFIG", "sightline.toml"))]


def run_fast_threshold_sweep(
    *,
    configs: list[Path],
    args: argparse.Namespace,
    binary: Path,
    specimens: Path,
    negatives: Path | None,
    work_dir: Path,
    gates: GateConfig,
) -> list[dict]:
    command = [str(binary)]
    assert_shared_fingerprint_config(configs, command, specimens, "positive specimens")
    if negatives is not None:
        assert_shared_fingerprint_config(configs, command, negatives, "hard negatives")

    cache_dir = work_dir / "fingerprints"
    positive_cache = cache_dir / "positives"
    with_sightline_config(
        configs[0],
        lambda: ensure_fingerprint_cache(
            command, specimens, positive_cache, args.refresh_cache
        ),
    )
    negative_cache = None
    if negatives is not None:
        negative_cache = cache_dir / "negatives"
        with_sightline_config(
            configs[0],
            lambda: ensure_fingerprint_cache(
                command, negatives, negative_cache, args.refresh_cache
            ),
        )
    augmented_cache = None
    if args.augment_profile != "none":
        augmented_image_dir = work_dir / "augmented_transforms" / "images"
        augmented_cache = cache_dir / "augmented_transforms"
        prepare_augmented_transform_cache(
            command=command,
            config_path=configs[0],
            specimens=specimens,
            image_dir=augmented_image_dir,
            fingerprint_dir=augmented_cache,
            profile=args.augment_profile,
            jpeg_quality=args.augment_jpeg_quality,
            refresh=args.refresh_cache,
        )

    sweep_command = [
        str(binary),
        "validate-threshold-sweep",
        str(positive_cache),
        "-" if negative_cache is None else str(negative_cache),
        "--folds",
        str(args.folds),
        "--seed",
        str(args.seed),
        "--positive-mode",
        args.positive_mode,
    ]
    for config_path in configs:
        sweep_command.extend(["--config", str(config_path)])
    if args.evaluate_all_stages:
        sweep_command.append("--evaluate-all-stages")

    sweep_report = run_json_command(sweep_command, env=os.environ.copy())
    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    sweep_report_path = results_dir / "threshold_sweep.json"
    sweep_report_path.write_text(json.dumps(sweep_report, indent=2), encoding="utf-8")

    runs = []
    for run in sweep_report["runs"]:
        kfold = run["kfold"]
        leave_one_out = run["leave_one_out"]
        config_path = Path(run["config"]["path"])
        augmented_transforms = (
            None
            if augmented_cache is None
            else run_augmented_transform_validation(
                binary=binary,
                config_path=config_path,
                specimen_cache=positive_cache,
                augmented_cache=augmented_cache,
                work_dir=work_dir,
                evaluate_all_stages=args.evaluate_all_stages,
            )
        )
        gate_results = evaluate_gates(
            kfold, leave_one_out, augmented_transforms, gates, negatives is not None
        )
        runs.append(
            {
                "config": config_report(config_path),
                "work_dir": str(work_dir),
                "passed": all(gate.passed for gate in gate_results),
                "gate_results": [asdict(gate) for gate in gate_results],
                "kfold": summarize_kfold(kfold),
                "leave_one_out": summarize_leave_one_out(leave_one_out),
                "augmented_transforms": augmented_transforms,
                "report_paths": {
                    "threshold_sweep": str(sweep_report_path),
                },
            }
        )
    return runs


def prepare_augmented_transform_cache(
    *,
    command: list[str],
    config_path: Path,
    specimens: Path,
    image_dir: Path,
    fingerprint_dir: Path,
    profile: str,
    jpeg_quality: int,
    refresh: bool,
) -> None:
    summary_path = image_dir.parent / "augment-summary.json"
    if refresh and image_dir.exists():
        remove_tree(image_dir)
    if not image_dir.exists() or not any(image_dir.glob("*.jpg")):
        image_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["SIGHTLINE_CONFIG"] = str(config_path)
        report = run_json_command(
            [
                *command,
                "augment-images",
                str(specimens),
                str(image_dir),
                "--profile",
                profile,
                "--jpeg-quality",
                str(jpeg_quality),
            ],
            env=env,
        )
        summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with_sightline_config(
        config_path,
        lambda: ensure_fingerprint_cache(command, image_dir, fingerprint_dir, refresh),
    )


def remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path)


def run_augmented_transform_validation(
    *,
    binary: Path,
    config_path: Path,
    specimen_cache: Path,
    augmented_cache: Path,
    work_dir: Path,
    evaluate_all_stages: bool,
) -> dict:
    env = os.environ.copy()
    env["SIGHTLINE_CONFIG"] = str(config_path)
    report = run_json_command(
        [
            str(binary),
            "compare-image-sets",
            str(specimen_cache),
            str(augmented_cache),
            *(["--evaluate-all-stages"] if evaluate_all_stages else []),
        ],
        env=env,
    )
    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / f"augmented_{config_slug(config_path)}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return summarize_augmented_transform_validation(report, augmented_cache, report_path)


def summarize_augmented_transform_validation(
    report: dict, augmented_cache: Path, report_path: Path
) -> dict:
    by_variant: dict[str, dict[str, int]] = {}
    by_confidence: dict[str, int] = {}
    by_geometry_model: dict[str, int] = {}
    by_passed_stage: dict[str, int] = {}
    by_passed_geometry_model: dict[str, int] = {}
    matched_sources = set()
    hard = 0
    soft = 0
    for item in report.get("best_per_candidate", []):
        source = item.get("candidate_source", "")
        matched_sources.add(source)
        variant = augmented_variant_label(source)
        bucket = by_variant.setdefault(variant, {})
        if item.get("matched"):
            hard += 1
            bucket["hard"] = bucket.get("hard", 0) + 1
        elif item.get("suspicious"):
            soft += 1
            bucket["soft"] = bucket.get("soft", 0) + 1
        else:
            bucket["unmatched"] = bucket.get("unmatched", 0) + 1
        outcome = item.get("outcome", {})
        confidence = outcome.get("confidence", "unknown")
        by_confidence[confidence] = by_confidence.get(confidence, 0) + 1
        model = outcome.get("local_geometry_model") or "none"
        by_geometry_model[model] = by_geometry_model.get(model, 0) + 1
        for step in outcome.get("diagnostics", {}).get("steps", []):
            if not step.get("passed"):
                continue
            stage = f"{step.get('threshold', 'unknown')}:{step.get('step', 'unknown')}"
            by_passed_stage[stage] = by_passed_stage.get(stage, 0) + 1
            step_model = step.get("local_geometry_model")
            if step_model:
                by_passed_geometry_model[step_model] = (
                    by_passed_geometry_model.get(step_model, 0) + 1
                )

    all_sources = fingerprint_sources(augmented_cache)
    unmatched_sources = sorted(source for source in all_sources if source not in matched_sources)
    for source in unmatched_sources:
        variant = augmented_variant_label(source)
        bucket = by_variant.setdefault(variant, {})
        bucket["unmatched"] = bucket.get("unmatched", 0) + 1

    return {
        "candidate_count": report.get("candidate_count"),
        "hard": hard,
        "soft": soft,
        "unmatched": len(unmatched_sources),
        "by_variant": by_variant,
        "by_confidence": by_confidence,
        "by_geometry_model": by_geometry_model,
        "by_passed_stage": by_passed_stage,
        "by_passed_geometry_model": by_passed_geometry_model,
        "unmatched_sources": unmatched_sources[:100],
        "report_path": str(report_path),
    }


def fingerprint_sources(fingerprint_dir: Path) -> set[str]:
    sources = set()
    for path in fingerprint_dir.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        source = record.get("source_path")
        if isinstance(source, str):
            sources.add(source)
    return sources


def augmented_variant_label(source_path: str) -> str:
    stem = Path(source_path).name.rsplit(".", 1)[0]
    if "__" in stem:
        return stem.rsplit("__", 1)[1]
    return "unknown"


def assert_shared_fingerprint_config(
    configs: list[Path], command: list[str], input_path: Path, label: str
) -> None:
    first = fingerprint_cache_metadata_for_config(configs[0], command, input_path)[
        "fingerprint_config"
    ]
    for config_path in configs[1:]:
        current = fingerprint_cache_metadata_for_config(config_path, command, input_path)[
            "fingerprint_config"
        ]
        if current != first:
            raise SystemExit(
                "fast threshold sweep can only share cached fingerprints across TOML "
                f"configs with identical image-processing settings for {label}. "
                "Use --legacy-validation for mixed image-processing configs."
            )


def fingerprint_cache_metadata_for_config(
    config_path: Path, command: list[str], input_path: Path
) -> dict:
    result: dict | None = None

    def capture() -> None:
        nonlocal result
        result = fingerprint_cache_metadata(command, input_path)

    with_sightline_config(config_path, capture)
    if result is None:
        raise RuntimeError("fingerprint metadata capture failed")
    return result


def with_sightline_config(config_path: Path, callback) -> None:
    previous = os.environ.get("SIGHTLINE_CONFIG")
    os.environ["SIGHTLINE_CONFIG"] = str(config_path)
    try:
        callback()
    finally:
        if previous is None:
            os.environ.pop("SIGHTLINE_CONFIG", None)
        else:
            os.environ["SIGHTLINE_CONFIG"] = previous


def run_validation_for_config(
    *,
    config_path: Path,
    args: argparse.Namespace,
    binary: Path,
    specimens: Path,
    negatives: Path | None,
    work_dir: Path,
    gates: GateConfig,
) -> dict:
    slug = config_slug(config_path)
    config_work_dir = work_dir / slug
    config_work_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SIGHTLINE_CONFIG"] = str(config_path)

    common_args = [
        "--specimens",
        str(specimens),
        "--binary",
        str(binary),
        "--positive-mode",
        args.positive_mode,
    ]
    if negatives is not None:
        common_args.extend(["--negatives", str(negatives)])
    if args.refresh_cache:
        common_args.append("--refresh-cache")
    if args.keep_fold_dirs:
        common_args.append("--keep-fold-dirs")

    kfold = run_json_command(
        [
            sys.executable,
            "scripts/validate_kfold.py",
            *common_args,
            "--folds",
            str(args.folds),
            "--seed",
            str(args.seed),
            "--work-dir",
            str(config_work_dir / "kfold"),
        ],
        env=env,
    )
    leave_one_out = run_json_command(
        [
            sys.executable,
            "scripts/validate_leave_one_out.py",
            *common_args,
            "--work-dir",
            str(config_work_dir / "leave_one_out"),
        ],
        env=env,
    )
    gate_results = evaluate_gates(kfold, leave_one_out, None, gates, negatives is not None)
    return {
        "config": config_report(config_path),
        "work_dir": str(config_work_dir),
        "passed": all(gate.passed for gate in gate_results),
        "gate_results": [asdict(gate) for gate in gate_results],
        "kfold": summarize_kfold(kfold),
        "leave_one_out": summarize_leave_one_out(leave_one_out),
        "report_paths": {
            "kfold": str(config_work_dir / "kfold" / "results" / "summary.json"),
            "leave_one_out": str(
                config_work_dir / "leave_one_out" / "results" / "summary.json"
            ),
        },
    }


def evaluate_gates(
    kfold: dict,
    leave_one_out: dict,
    augmented_transforms: dict | None,
    gates: GateConfig,
    has_negatives: bool,
) -> list[GateResult]:
    results = [
        gate(
            "kfold_all_positives_covered",
            kfold.get("unmatched_positive_sources_count") == 0,
            kfold.get("unmatched_positive_sources_count"),
            "0 unmatched positive sources",
            kfold.get("unmatched_positive_sources", []),
        ),
        gate(
            "leave_one_out_all_positives_covered",
            leave_one_out.get("unmatched_specimens_count") == 0,
            leave_one_out.get("unmatched_specimens_count"),
            "0 unmatched held-out specimens",
            leave_one_out.get("unmatched_specimens", []),
        ),
        gate(
            "kfold_no_hard_negative_hard_matches",
            kfold.get("fully_matched_false_positive_sources_count", 0) == 0,
            kfold.get("fully_matched_false_positive_sources_count", 0),
            "0 fully matched hard-negative sources",
            kfold.get("fully_matched_false_positive_sources", []),
        ),
        gate(
            "leave_one_out_no_hard_negative_hard_matches",
            leave_one_out.get("fully_matched_false_positive_sources_count", 0) == 0,
            leave_one_out.get("fully_matched_false_positive_sources_count", 0),
            "0 fully matched hard-negative sources",
            leave_one_out.get("fully_matched_false_positive_sources", []),
        ),
        gate(
            "kfold_soft_false_positives_near_zero",
            kfold.get("suspicious_only_false_positive_sources_count", 0)
            <= gates.max_soft_false_positive_sources,
            kfold.get("suspicious_only_false_positive_sources_count", 0),
            f"<= {gates.max_soft_false_positive_sources} suspicious-only hard-negative sources",
            kfold.get("suspicious_only_false_positive_sources", []),
        ),
        gate(
            "leave_one_out_soft_false_positives_near_zero",
            leave_one_out.get("suspicious_only_false_positive_sources_count", 0)
            <= gates.max_soft_false_positive_sources,
            leave_one_out.get("suspicious_only_false_positive_sources_count", 0),
            f"<= {gates.max_soft_false_positive_sources} suspicious-only hard-negative sources",
            leave_one_out.get("suspicious_only_false_positive_sources", []),
        ),
    ]
    results.extend(hard_positive_ratio_gates(kfold, leave_one_out, gates))
    if has_negatives:
        results.extend(metric_gates(kfold, leave_one_out, gates))
    if augmented_transforms is not None:
        results.extend(augmented_transform_gates(augmented_transforms, gates))
    return results


def augmented_transform_gates(
    augmented_transforms: dict, gates: GateConfig
) -> list[GateResult]:
    candidate_count = augmented_transforms.get("candidate_count") or 0
    hard = augmented_transforms.get("hard") or 0
    hard_ratio = safe_div(hard, candidate_count)
    return [
        gate(
            "augmented_transforms_all_covered",
            (augmented_transforms.get("unmatched") or 0) <= gates.max_augmented_unmatched,
            augmented_transforms.get("unmatched"),
            f"<= {gates.max_augmented_unmatched} unmatched transformed positives",
            augmented_transforms.get("unmatched_sources", []),
        ),
        gate(
            "augmented_transforms_hard_match_ratio",
            hard_ratio is not None and hard_ratio >= gates.min_augmented_hard_ratio,
            hard_ratio,
            f">= {gates.min_augmented_hard_ratio:.3f} hard transformed positive ratio",
            [],
        ),
    ]


def hard_positive_ratio_gates(
    kfold: dict, leave_one_out: dict, gates: GateConfig
) -> list[GateResult]:
    kfold_ratio = safe_div(
        kfold.get("fully_matched_positive_sources_count", 0),
        kfold.get("positive_count", 0),
    )
    loo_ratio = safe_div(
        leave_one_out.get("fully_matched_specimens_count", 0),
        leave_one_out.get("positive_count", 0),
    )
    return [
        gate(
            "kfold_hard_matches_preferred",
            kfold_ratio is not None and kfold_ratio >= gates.min_hard_positive_ratio,
            kfold_ratio,
            f">= {gates.min_hard_positive_ratio:.3f} hard-positive ratio",
            kfold.get("suspicious_only_positive_sources", []),
        ),
        gate(
            "leave_one_out_hard_matches_preferred",
            loo_ratio is not None and loo_ratio >= gates.min_hard_positive_ratio,
            loo_ratio,
            f">= {gates.min_hard_positive_ratio:.3f} hard-positive ratio",
            leave_one_out.get("suspicious_only_specimens", []),
        ),
    ]


def metric_gates(kfold: dict, leave_one_out: dict, gates: GateConfig) -> list[GateResult]:
    return [
        metric_gate("kfold_precision", kfold, "precision", gates.min_precision),
        metric_gate("kfold_specificity", kfold, "specificity", gates.min_specificity),
        metric_gate("leave_one_out_precision", leave_one_out, "precision", gates.min_precision),
        metric_gate(
            "leave_one_out_specificity",
            leave_one_out,
            "specificity",
            gates.min_specificity,
        ),
    ]


def metric_gate(name: str, report: dict, metric: str, minimum: float) -> GateResult:
    value = report.get("aggregate", {}).get(metric)
    return gate(
        name,
        value is not None and value >= minimum,
        value,
        f">= {minimum:.3f}",
        [],
    )


def gate(
    name: str, passed: bool, actual: Any, expected: str, details: list[str]
) -> GateResult:
    return GateResult(
        name=name,
        passed=passed,
        actual=actual,
        expected=expected,
        details=details[:50],
    )


def summarize_kfold(report: dict) -> dict:
    return {
        "positive_count": report.get("positive_count"),
        "negative_count": report.get("negative_count"),
        "aggregate": report.get("aggregate"),
        "fully_matched_positive_sources_count": report.get(
            "fully_matched_positive_sources_count"
        ),
        "suspicious_only_positive_sources_count": report.get(
            "suspicious_only_positive_sources_count"
        ),
        "unmatched_positive_sources_count": report.get("unmatched_positive_sources_count"),
        "fully_matched_false_positive_sources_count": report.get(
            "fully_matched_false_positive_sources_count"
        ),
        "suspicious_only_false_positive_sources_count": report.get(
            "suspicious_only_false_positive_sources_count"
        ),
        "unmatched_positive_sources": report.get("unmatched_positive_sources", []),
        "fully_matched_false_positive_sources": report.get(
            "fully_matched_false_positive_sources", []
        ),
        "suspicious_only_false_positive_sources": report.get(
            "suspicious_only_false_positive_sources", []
        ),
    }


def summarize_leave_one_out(report: dict) -> dict:
    return {
        "positive_count": report.get("positive_count"),
        "negative_count": report.get("negative_count"),
        "aggregate": report.get("aggregate"),
        "fully_matched_specimens_count": report.get("fully_matched_specimens_count"),
        "suspicious_only_specimens_count": report.get("suspicious_only_specimens_count"),
        "unmatched_specimens_count": report.get("unmatched_specimens_count"),
        "fully_matched_false_positive_sources_count": report.get(
            "fully_matched_false_positive_sources_count"
        ),
        "suspicious_only_false_positive_sources_count": report.get(
            "suspicious_only_false_positive_sources_count"
        ),
        "unmatched_specimens": report.get("unmatched_specimens", []),
        "fully_matched_false_positive_sources": report.get(
            "fully_matched_false_positive_sources", []
        ),
        "suspicious_only_false_positive_sources": report.get(
            "suspicious_only_false_positive_sources", []
        ),
    }


def render_console_summary(report: dict, json_path: Path, markdown_path: Path) -> str:
    lines = [
        f"Sightline tuning validation: {'PASS' if report['passed'] else 'FAIL'}",
        f"JSON report: {json_path}",
        f"Markdown report: {markdown_path}",
    ]
    for run in report["runs"]:
        lines.append(
            f"- {run['config']['path']}: {'PASS' if run['passed'] else 'FAIL'}"
        )
        for gate_result in run["gate_results"]:
            if not gate_result["passed"]:
                lines.append(
                    "  "
                    f"* {gate_result['name']} failed: actual={gate_result['actual']} "
                    f"expected {gate_result['expected']}"
                )
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    lines = [
        "# Sightline Tuning Validation",
        "",
        f"Overall: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "## Gates",
        "",
        "```json",
        json.dumps(report["gates"], indent=2),
        "```",
        "",
        "## Runs",
        "",
    ]
    for run in report["runs"]:
        lines.extend(
            [
                f"### {run['config']['path']}",
                "",
                f"Result: **{'PASS' if run['passed'] else 'FAIL'}**",
                "",
                "| Gate | Result | Actual | Expected |",
                "| --- | --- | --- | --- |",
            ]
        )
        for gate_result in run["gate_results"]:
            lines.append(
                "| {name} | {result} | `{actual}` | {expected} |".format(
                    name=gate_result["name"],
                    result="PASS" if gate_result["passed"] else "FAIL",
                    actual=gate_result["actual"],
                    expected=gate_result["expected"],
                )
            )
        lines.extend(
            [
                "",
                "#### K-fold",
                "",
                "```json",
                json.dumps(run["kfold"], indent=2),
                "```",
                "",
                "#### Leave One Out",
                "",
                "```json",
                json.dumps(run["leave_one_out"], indent=2),
                "```",
                "",
            ]
        )
        if run.get("augmented_transforms") is not None:
            lines.extend(
                [
                    "#### Augmented Transforms",
                    "",
                    "```json",
                    json.dumps(run["augmented_transforms"], indent=2),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def config_report(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
    }


def config_slug(path: Path) -> str:
    raw = path.as_posix() if str(path) else "default"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return slug or "default"


def run_command(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"command failed with exit {completed.returncode}: {' '.join(command)}")


def run_json_command(command: list[str], env: dict[str, str]) -> dict:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "validation command failed\n"
            f"command: {' '.join(command)}\n"
            f"exit: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"validation command did not emit JSON: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        ) from exc


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
