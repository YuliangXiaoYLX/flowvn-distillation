#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.flowvn_results import (
    METRIC_DIRECTIONS,
    REVISION_METRICS,
    case_metric_means,
    load_validation_csv,
    require_complete_metrics,
    validate_matching_keys,
    validate_workshop_cohort,
)


ARM_ORDER = ("supervised", "final_only", "final_trajectory", "full_kd")
ARM_LABELS = {
    "supervised": "Supervised S8",
    "final_only": "Final-only KD",
    "final_trajectory": "Final+trajectory KD",
    "full_kd": "Full KD",
}
COMPONENT_CONTRASTS = {
    "final_output_matching": ("supervised", "final_only"),
    "trajectory_matching": ("final_only", "final_trajectory"),
    "update_matching": ("final_trajectory", "full_kd"),
}
CONTRASTS = {
    **COMPONENT_CONTRASTS,
    "joint_recipe": ("supervised", "full_kd"),
}
CO_PRIMARY_METRICS = ("nrmse", "velocity_vector_rmse_cm_s")
METRIC_LABELS = {
    "nrmse": r"nRMSE $\downarrow$",
    "ssim": r"SSIM $\uparrow$",
    "relerr": r"RelErr $\downarrow$",
    "angerr": r"AngErr ($^\circ$) $\downarrow$",
    "normalized_l1": r"Norm. $L_1$ $\downarrow$",
    "velocity_vector_rmse_cm_s": r"VENC RMSE (cm/s) $\downarrow$",
}
CONTRAST_LABELS = {
    "final_output_matching": "Final-output matching",
    "trajectory_matching": "Trajectory matching",
    "update_matching": "Update matching",
    "joint_recipe": "Joint KD recipe",
}


def _mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return float(statistics.fmean(values))


def _percentile(ordered: list[float], probability: float) -> float:
    position = (len(ordered) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(
        ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    )


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    seed_material: str,
) -> list[float]:
    if int(samples) < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not values:
        raise ValueError("Cannot bootstrap an empty sequence")
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    count = len(values)
    estimates = sorted(
        _mean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(int(samples))
    )
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _favorable(metric: str, delta: float) -> bool:
    if METRIC_DIRECTIONS[metric] == "higher":
        return float(delta) > 0.0
    return float(delta) < 0.0


def _case_relative_effects(
    control_rows: list[dict],
    treatment_rows: list[dict],
    *,
    metric: str,
    usrate: int | None,
) -> dict[str, float]:
    """Mean paired group-level percentage effect within each case."""
    key_fields = ("case_id", "slice_start", "usrate")
    control = {
        tuple(row[field] for field in key_fields): float(row[metric])
        for row in control_rows
    }
    treatment = {
        tuple(row[field] for field in key_fields): float(row[metric])
        for row in treatment_rows
    }
    if set(control) != set(treatment):
        raise ValueError(f"Paired group mismatch for relative {metric} effect")
    grouped: dict[str, list[float]] = {}
    for key in sorted(control):
        if usrate is not None and int(key[2]) != int(usrate):
            continue
        control_value = float(control[key])
        if control_value == 0.0:
            raise ValueError(
                f"Cannot calculate relative {metric} effect with zero "
                f"control value: {key}"
            )
        grouped.setdefault(str(key[0]), []).append(
            100.0 * (float(treatment[key]) - control_value) / control_value
        )
    if not grouped:
        raise ValueError(
            f"No case values for relative metric={metric}, usrate={usrate}"
        )
    return {case_id: _mean(values) for case_id, values in grouped.items()}


def _ci_supports_favorable(metric: str, ci: list[float]) -> bool:
    if METRIC_DIRECTIONS[metric] == "higher":
        return float(ci[0]) > 0.0
    return float(ci[1]) < 0.0


def _ci_supports_unfavorable(metric: str, ci: list[float]) -> bool:
    if METRIC_DIRECTIONS[metric] == "higher":
        return float(ci[1]) < 0.0
    return float(ci[0]) > 0.0


def _scope_summary(rows_by_seed, *, metric: str, usrate: int | None) -> dict:
    by_seed = {}
    case_values_by_seed = {}
    for seed, rows in sorted(rows_by_seed.items()):
        values = case_metric_means(rows, metric, usrate)
        if not values:
            raise ValueError(
                f"No case values for seed={seed}, metric={metric}, usrate={usrate}"
            )
        case_values_by_seed[seed] = values
        by_seed[seed] = {
            "n_cases": len(values),
            "mean": _mean(values.values()),
        }
    cases = sorted(
        set.intersection(*(set(values) for values in case_values_by_seed.values()))
    )
    if any(set(values) != set(cases) for values in case_values_by_seed.values()):
        raise ValueError(f"Case mismatch across seeds for arm summary: {metric}")
    seed_averaged_cases = [
        _mean(case_values_by_seed[seed][case_id] for seed in case_values_by_seed)
        for case_id in cases
    ]
    return {
        "direction": METRIC_DIRECTIONS[metric],
        "n_cases": len(cases),
        "n_seeds": len(by_seed),
        "by_seed": by_seed,
        "mean_across_seeds": _mean(seed_averaged_cases),
        "seed_mean_range": [
            min(record["mean"] for record in by_seed.values()),
            max(record["mean"] for record in by_seed.values()),
        ],
    }


def _contrast_scope_summary(
    *,
    runs: dict,
    seeds: tuple[str, ...],
    control_arm: str,
    treatment_arm: str,
    metric: str,
    usrate: int | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
    contrast_name: str,
) -> dict:
    per_seed = {}
    case_effects = {}
    case_relative_effects = {}
    control_cases = {}
    treatment_cases = {}
    for seed in seeds:
        control = case_metric_means(runs[(seed, control_arm)], metric, usrate)
        treatment = case_metric_means(runs[(seed, treatment_arm)], metric, usrate)
        if set(control) != set(treatment):
            raise ValueError(
                f"Paired case mismatch for {contrast_name}, seed={seed}, metric={metric}"
            )
        cases = sorted(control)
        effects = {
            case_id: float(treatment[case_id] - control[case_id])
            for case_id in cases
        }
        relative_effects = _case_relative_effects(
            runs[(seed, control_arm)],
            runs[(seed, treatment_arm)],
            metric=metric,
            usrate=usrate,
        )
        if set(relative_effects) != set(cases):
            raise ValueError(
                f"Relative-effect case mismatch for {contrast_name}, "
                f"seed={seed}, metric={metric}"
            )
        case_effects[seed] = effects
        case_relative_effects[seed] = relative_effects
        control_cases[seed] = control
        treatment_cases[seed] = treatment
        mean_delta = _mean(effects.values())
        per_seed[seed] = {
            "n_cases": len(cases),
            "control_mean": _mean(control.values()),
            "treatment_mean": _mean(treatment.values()),
            "mean_delta_treatment_minus_control": mean_delta,
            "mean_relative_delta_percent": _mean(relative_effects.values()),
            "favorable": _favorable(metric, mean_delta),
        }

    cases = sorted(set.intersection(*(set(case_effects[s]) for s in seeds)))
    if any(set(case_effects[seed]) != set(cases) for seed in seeds):
        raise ValueError(
            f"Case mismatch across seeds for {contrast_name}, metric={metric}"
        )
    averaged_effects = [
        _mean(case_effects[seed][case_id] for seed in seeds)
        for case_id in cases
    ]
    averaged_relative_effects = [
        _mean(case_relative_effects[seed][case_id] for seed in seeds)
        for case_id in cases
    ]
    averaged_control = [
        _mean(control_cases[seed][case_id] for seed in seeds)
        for case_id in cases
    ]
    averaged_treatment = [
        _mean(treatment_cases[seed][case_id] for seed in seeds)
        for case_id in cases
    ]
    ci = _bootstrap_mean_ci(
        averaged_effects,
        samples=bootstrap_samples,
        seed_material=(
            f"{bootstrap_seed}|{contrast_name}|{metric}|"
            f"{'overall' if usrate is None else usrate}"
        ),
    )
    relative_ci = _bootstrap_mean_ci(
        averaged_relative_effects,
        samples=bootstrap_samples,
        seed_material=(
            f"{bootstrap_seed}|{contrast_name}|{metric}|"
            f"{'overall' if usrate is None else usrate}|relative"
        ),
    )
    mean_delta = _mean(averaged_effects)
    return {
        "direction": METRIC_DIRECTIONS[metric],
        "control_arm": control_arm,
        "treatment_arm": treatment_arm,
        "n_cases": len(cases),
        "n_seeds": len(seeds),
        "control_mean": _mean(averaged_control),
        "treatment_mean": _mean(averaged_treatment),
        "mean_delta_treatment_minus_control": mean_delta,
        "mean_relative_delta_percent": _mean(averaged_relative_effects),
        "case_bootstrap_95_ci_delta": ci,
        "case_bootstrap_95_ci_relative_delta_percent": relative_ci,
        "per_seed": per_seed,
        "both_seeds_favorable": all(
            record["favorable"] for record in per_seed.values()
        ),
        "bootstrap_supports_favorable": _ci_supports_favorable(metric, ci),
        "bootstrap_supports_unfavorable": _ci_supports_unfavorable(metric, ci),
    }


def analyze_revision_ablation(
    runs: dict[tuple[str, str], list[dict]],
    *,
    expected_seeds,
    expected_cases: int,
    expected_usrates,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 20260809,
) -> dict:
    seeds = tuple(str(seed) for seed in expected_seeds)
    usrates = tuple(sorted({int(value) for value in expected_usrates}))
    if len(seeds) < 2:
        raise ValueError("At least two matched seeds are required")
    expected_keys = {(seed, arm) for seed in seeds for arm in ARM_ORDER}
    actual_keys = {(str(seed), str(arm)) for seed, arm in runs}
    if actual_keys != expected_keys:
        raise ValueError(
            f"Run matrix mismatch: missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    normalized_runs = {
        (str(seed), str(arm)): rows for (seed, arm), rows in runs.items()
    }
    reference_name = None
    reference_rows = None
    for key in sorted(normalized_runs):
        rows = normalized_runs[key]
        label = f"seed={key[0]},arm={key[1]}"
        require_complete_metrics(label, rows, metrics=REVISION_METRICS)
        validate_workshop_cohort(
            label,
            rows,
            expected_groups=int(expected_cases) * len(usrates),
            expected_cases=int(expected_cases),
            expected_usrates=usrates,
            require_hashed_case_ids=True,
        )
        if reference_rows is None:
            reference_name, reference_rows = label, rows
        else:
            validate_matching_keys(reference_name, reference_rows, label, rows)

    arm_summaries = {}
    for arm in ARM_ORDER:
        rows_by_seed = {
            seed: normalized_runs[(seed, arm)] for seed in seeds
        }
        arm_summaries[arm] = {
            "overall": {
                metric: _scope_summary(rows_by_seed, metric=metric, usrate=None)
                for metric in REVISION_METRICS
            },
            "by_usrate": {
                str(usrate): {
                    metric: _scope_summary(
                        rows_by_seed, metric=metric, usrate=usrate
                    )
                    for metric in REVISION_METRICS
                }
                for usrate in usrates
            },
        }

    contrasts = {}
    for contrast_name, (control_arm, treatment_arm) in CONTRASTS.items():
        contrasts[contrast_name] = {
            "control_arm": control_arm,
            "treatment_arm": treatment_arm,
            "overall": {
                metric: _contrast_scope_summary(
                    runs=normalized_runs,
                    seeds=seeds,
                    control_arm=control_arm,
                    treatment_arm=treatment_arm,
                    metric=metric,
                    usrate=None,
                    bootstrap_samples=int(bootstrap_samples),
                    bootstrap_seed=int(bootstrap_seed),
                    contrast_name=contrast_name,
                )
                for metric in REVISION_METRICS
            },
            "by_usrate": {
                str(usrate): {
                    metric: _contrast_scope_summary(
                        runs=normalized_runs,
                        seeds=seeds,
                        control_arm=control_arm,
                        treatment_arm=treatment_arm,
                        metric=metric,
                        usrate=usrate,
                        bootstrap_samples=int(bootstrap_samples),
                        bootstrap_seed=int(bootstrap_seed),
                        contrast_name=contrast_name,
                    )
                    for metric in REVISION_METRICS
                }
                for usrate in usrates
            },
        }

    component_decisions = {}
    for contrast_name in COMPONENT_CONTRASTS:
        endpoints = contrasts[contrast_name]["overall"]
        supported_endpoints = [
            metric
            for metric in CO_PRIMARY_METRICS
            if endpoints[metric]["both_seeds_favorable"]
            and endpoints[metric]["bootstrap_supports_favorable"]
        ]
        unfavorable_endpoints = [
            metric
            for metric in CO_PRIMARY_METRICS
            if endpoints[metric]["bootstrap_supports_unfavorable"]
        ]
        if unfavorable_endpoints:
            status = "unfavorable_co_primary"
        elif supported_endpoints:
            status = "supported"
        else:
            status = "inconclusive"
        component_decisions[contrast_name] = {
            "status": status,
            "supported": status == "supported",
            "supported_co_primary_metrics": supported_endpoints,
            "unfavorable_co_primary_metrics": unfavorable_endpoints,
            "rule": (
                "At least one co-primary endpoint must favor the component arm "
                "in both seeds with a favorable case-bootstrap interval, and "
                "neither co-primary interval may exclude zero unfavorably."
            ),
        }

    return {
        "schema_version": 1,
        "analysis": "nested_kd_component_ablation",
        "cohort": {
            "n_cases": int(expected_cases),
            "usrates": list(usrates),
            "n_groups_per_run": int(expected_cases) * len(usrates),
            "seeds": list(seeds),
            "independent_unit": "validation_case",
            "bootstrap_samples": int(bootstrap_samples),
        },
        "metrics": list(REVISION_METRICS),
        "co_primary_metrics": list(CO_PRIMARY_METRICS),
        "arm_summaries": arm_summaries,
        "contrasts": contrasts,
        "component_decisions": component_decisions,
        "per_acceleration_inference": "descriptive_unadjusted_for_multiplicity",
    }


def flatten_analysis_records(analysis: dict) -> dict[str, list[dict]]:
    seeds = [str(seed) for seed in analysis["cohort"]["seeds"]]
    arm_records = []
    for arm in ARM_ORDER:
        arm_summary = analysis["arm_summaries"][arm]
        scopes = [("overall", None, arm_summary["overall"])] + [
            ("usrate", int(usrate), metrics)
            for usrate, metrics in sorted(
                arm_summary["by_usrate"].items(), key=lambda item: int(item[0])
            )
        ]
        for scope, usrate, metrics in scopes:
            for metric in REVISION_METRICS:
                summary = metrics[metric]
                row = {
                    "arm": arm,
                    "scope": scope,
                    "usrate": "" if usrate is None else int(usrate),
                    "metric": metric,
                    "direction": summary["direction"],
                    "n_cases": summary["n_cases"],
                    "n_seeds": summary["n_seeds"],
                    "mean_across_seeds": summary["mean_across_seeds"],
                    "seed_mean_min": summary["seed_mean_range"][0],
                    "seed_mean_max": summary["seed_mean_range"][1],
                }
                for seed in seeds:
                    row[f"seed_{seed}_mean"] = summary["by_seed"][seed]["mean"]
                arm_records.append(row)

    contrast_records = []
    for contrast_name in CONTRASTS:
        contrast = analysis["contrasts"][contrast_name]
        scopes = [("overall", None, contrast["overall"])] + [
            ("usrate", int(usrate), metrics)
            for usrate, metrics in sorted(
                contrast["by_usrate"].items(), key=lambda item: int(item[0])
            )
        ]
        for scope, usrate, metrics in scopes:
            for metric in REVISION_METRICS:
                summary = metrics[metric]
                row = {
                    "contrast": contrast_name,
                    "control_arm": contrast["control_arm"],
                    "treatment_arm": contrast["treatment_arm"],
                    "scope": scope,
                    "usrate": "" if usrate is None else int(usrate),
                    "metric": metric,
                    "direction": summary["direction"],
                    "n_cases": summary["n_cases"],
                    "n_seeds": summary["n_seeds"],
                    "control_mean": summary["control_mean"],
                    "treatment_mean": summary["treatment_mean"],
                    "mean_delta_treatment_minus_control": summary[
                        "mean_delta_treatment_minus_control"
                    ],
                    "mean_relative_delta_percent": summary[
                        "mean_relative_delta_percent"
                    ],
                    "bootstrap_ci95_low": summary[
                        "case_bootstrap_95_ci_delta"
                    ][0],
                    "bootstrap_ci95_high": summary[
                        "case_bootstrap_95_ci_delta"
                    ][1],
                    "relative_bootstrap_ci95_low": summary[
                        "case_bootstrap_95_ci_relative_delta_percent"
                    ][0],
                    "relative_bootstrap_ci95_high": summary[
                        "case_bootstrap_95_ci_relative_delta_percent"
                    ][1],
                    "both_seeds_favorable": summary["both_seeds_favorable"],
                    "bootstrap_supports_favorable": summary[
                        "bootstrap_supports_favorable"
                    ],
                    "bootstrap_supports_unfavorable": summary[
                        "bootstrap_supports_unfavorable"
                    ],
                }
                for seed in seeds:
                    row[f"seed_{seed}_delta"] = summary["per_seed"][seed][
                        "mean_delta_treatment_minus_control"
                    ]
                    row[f"seed_{seed}_relative_delta_percent"] = summary[
                        "per_seed"
                    ][seed]["mean_relative_delta_percent"]
                contrast_records.append(row)

    decision_records = []
    for contrast_name in COMPONENT_CONTRASTS:
        decision = analysis["component_decisions"][contrast_name]
        decision_records.append(
            {
                "contrast": contrast_name,
                "status": decision["status"],
                "supported": decision["supported"],
                "supported_co_primary_metrics": ";".join(
                    decision["supported_co_primary_metrics"]
                ),
                "unfavorable_co_primary_metrics": ";".join(
                    decision["unfavorable_co_primary_metrics"]
                ),
                "rule": decision["rule"],
            }
        )
    return {
        "arm_summaries": arm_records,
        "contrasts": contrast_records,
        "component_decisions": decision_records,
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_accepted_revision_runs(
    run_specs,
    *,
    expected_seeds,
    expected_cases: int,
    expected_usrates,
) -> tuple[dict[tuple[str, str], list[dict]], list[dict]]:
    seeds = tuple(str(seed) for seed in expected_seeds)
    usrates = tuple(sorted({int(value) for value in expected_usrates}))
    expected_matrix = {(seed, arm) for seed in seeds for arm in ARM_ORDER}
    normalized_specs = {}
    for seed, arm, csv_path, audit_path in run_specs:
        key = (str(seed), str(arm))
        if key in normalized_specs:
            raise ValueError(f"Duplicate revision run specification: {key}")
        normalized_specs[key] = (
            Path(csv_path).expanduser().resolve(),
            Path(audit_path).expanduser().resolve(),
        )
    if set(normalized_specs) != expected_matrix:
        raise ValueError(
            "Revision run specification matrix mismatch: "
            f"missing={sorted(expected_matrix - set(normalized_specs))}, "
            f"unexpected={sorted(set(normalized_specs) - expected_matrix)}"
        )

    runs = {}
    provenance = []
    for key in sorted(normalized_specs):
        csv_path, audit_path = normalized_specs[key]
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing validation CSV: {csv_path}")
        if not audit_path.is_file():
            raise FileNotFoundError(f"Missing validation audit: {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not isinstance(audit, dict):
            raise ValueError(f"Validation audit is not an object: {audit_path}")
        if audit.get("accepted") is not True:
            raise ValueError(f"Validation is not accepted: {audit_path}")
        if audit.get("paper_metrics_eligible") is not True:
            raise ValueError(f"Validation is not paper eligible: {audit_path}")
        if audit.get("physical_velocity_required") is not True:
            raise ValueError(
                f"Validation did not require physical velocity: {audit_path}"
            )
        if tuple(audit.get("required_metrics", ())) != REVISION_METRICS:
            raise ValueError(
                f"Validation required-metric contract mismatch: {audit_path}"
            )
        csv_record = audit.get("validation_csv", {})
        recorded_csv = Path(str(csv_record.get("path", ""))).expanduser()
        if not recorded_csv.is_absolute():
            recorded_csv = audit_path.parent / recorded_csv
        if recorded_csv.resolve() != csv_path:
            raise ValueError(
                f"Validation audit CSV path mismatch for {key}: "
                f"recorded={recorded_csv.resolve()}, expected={csv_path}"
            )
        csv_hash = _sha256_file(csv_path)
        if str(csv_record.get("sha256", "")).lower() != csv_hash:
            raise ValueError(
                f"Validation CSV SHA-256 mismatch for {key}: "
                f"actual={csv_hash}, recorded={csv_record.get('sha256')}"
            )
        cohort = audit.get("cohort", {})
        if int(cohort.get("n_groups", -1)) != int(expected_cases) * len(usrates):
            raise ValueError(f"Validation group count mismatch for {key}")
        if int(cohort.get("n_cases", -1)) != int(expected_cases):
            raise ValueError(f"Validation case count mismatch for {key}")
        if tuple(int(value) for value in cohort.get("usrates", ())) != usrates:
            raise ValueError(f"Validation acceleration mismatch for {key}")
        audit_provenance = audit.get("provenance", {})
        mask = audit_provenance.get("mask_backend", {})
        if int(audit_provenance.get("exit_status", -1)) != 0:
            raise ValueError(f"Validation exit status is not zero for {key}")
        if mask.get("backend") != "challenge" or mask.get("verified") is not True:
            raise ValueError(f"Challenge-mask provenance is not accepted for {key}")

        rows = load_validation_csv(csv_path)
        require_complete_metrics(str(key), rows, metrics=REVISION_METRICS)
        runs[key] = rows
        provenance.append(
            {
                "seed": key[0],
                "arm": key[1],
                "validation_csv_path": str(csv_path),
                "validation_csv_sha256": csv_hash,
                "validation_audit_path": str(audit_path),
                "validation_audit_sha256": _sha256_file(audit_path),
                "config_sha256": str(audit_provenance.get("config_sha256", "")),
                "checkpoint_sha256": str(
                    audit_provenance.get("checkpoint_sha256", "")
                ),
            }
        )
    return runs, provenance


def _csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        raise ValueError("Cannot write an empty analysis CSV")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _latex_number(value: float, metric: str, *, signed: bool = False) -> str:
    precision = 3 if metric in ("angerr", "velocity_vector_rmse_cm_s") else 4
    prefix = "+" if signed and float(value) >= 0.0 else ""
    return f"{prefix}{float(value):.{precision}f}"


def _latex_ci(summary: dict, metric: str) -> str:
    low, high = summary["case_bootstrap_95_ci_delta"]
    return (
        "["
        + _latex_number(low, metric, signed=True)
        + ", "
        + _latex_number(high, metric, signed=True)
        + "]"
    )


def _latex_by_rate_effect_number(value: float, metric: str) -> str:
    """Render small acceleration-specific effects without rounding to zero."""
    precision = 3 if metric in ("angerr", "velocity_vector_rmse_cm_s") else 6
    return f"{float(value):+.{precision}f}"


def _latex_by_rate_effect_ci(summary: dict, metric: str) -> str:
    low, high = summary["case_bootstrap_95_ci_delta"]
    return (
        "["
        + _latex_by_rate_effect_number(low, metric)
        + ", "
        + _latex_by_rate_effect_number(high, metric)
        + "]"
    )


def _latex_percent(value: float, *, signed: bool = True) -> str:
    prefix = "+" if signed and float(value) >= 0.0 else ""
    return f"{prefix}{float(value):.2f}\\%"


def _latex_relative_ci(summary: dict) -> str:
    low, high = summary["case_bootstrap_95_ci_relative_delta_percent"]
    return "[" + _latex_percent(low) + ", " + _latex_percent(high) + "]"


def _decision_label(analysis: dict, contrast_name: str) -> str:
    if contrast_name == "joint_recipe":
        return "Descriptive"
    status = analysis["component_decisions"][contrast_name]["status"]
    return {
        "supported": "Supported",
        "inconclusive": "Inconclusive",
        "unfavorable_co_primary": "Unfavorable",
    }[status]


def render_revision_latex_tables(analysis: dict) -> dict[str, str]:
    """Render deterministic, result-backed LaTeX tables for the revision."""
    seeds = [str(seed) for seed in analysis["cohort"]["seeds"]]
    usrates = [int(value) for value in analysis["cohort"]["usrates"]]

    overall_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        (
            r"\caption{Two-seed aggregate case means for every nested-ablation "
            r"arm. VENC RMSE is the VENC-scaled three-component endpoint "
            r"in cm/s. Arrows indicate the favorable direction.}"
        ),
        r"\label{tab:revision_overall}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Arm & nRMSE $\downarrow$ & SSIM $\uparrow$ & RelErr "
            r"$\downarrow$ & AngErr ($^\circ$) $\downarrow$ & Norm. $L_1$ "
            r"$\downarrow$ & VENC RMSE (cm/s) $\downarrow$ \\"
        ),
        r"\midrule",
    ]
    for arm in ARM_ORDER:
        summary = analysis["arm_summaries"][arm]["overall"]
        overall_lines.append(
            ARM_LABELS[arm]
            + " & "
            + " & ".join(
                _latex_number(summary[metric]["mean_across_seeds"], metric)
                for metric in REVISION_METRICS
            )
            + r" \\"
        )
    overall_lines.extend(
        [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    )

    seed_arm_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        (
            r"\caption{Seed-specific means for the four ordered S8 ablation "
            r"arms after averaging acceleration within case. Panels report "
            r"all six metrics for each matched training seed. Lower is "
            r"favorable except for SSIM; VENC RMSE is in cm/s.}"
        ),
        r"\label{tab:revision_seed_arm_means}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
    ]
    seed_table_metrics = (
        "normalized_l1",
        "nrmse",
        "ssim",
        "relerr",
        "angerr",
        "velocity_vector_rmse_cm_s",
    )
    panel_labels = "abcdefghijklmnopqrstuvwxyz"
    for seed_index, seed in enumerate(seeds):
        seed_arm_lines.extend(
            [
                r"\begin{tabular}{@{}lrrrrrr@{}}",
                (
                    r"\multicolumn{7}{@{}l}{\textbf{("
                    + panel_labels[seed_index]
                    + f") Seed {seed}" + r"}} \\"
                ),
                r"\toprule",
                (
                    r"Arm & Norm. $L_1$ & nRMSE & SSIM & RelErr & "
                    r"AngErr ($^\circ$) & VENC RMSE \\"
                ),
                r"\midrule",
            ]
        )
        for arm in ARM_ORDER:
            summary = analysis["arm_summaries"][arm]["overall"]
            seed_arm_lines.append(
                ARM_LABELS[arm]
                + " & "
                + " & ".join(
                    _latex_number(
                        summary[metric]["by_seed"][seed]["mean"], metric
                    )
                    for metric in seed_table_metrics
                )
                + r" \\"
            )
        seed_arm_lines.extend([r"\bottomrule", r"\end{tabular}"])
        if seed_index != len(seeds) - 1:
            seed_arm_lines.append(r"\vspace{3pt}")
    seed_arm_lines.append(r"\end{table}")

    component_full_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        (
            r"\caption{All six metrics for the ordered nested contrasts. "
            r"Effects are component-containing arm minus control after "
            r"averaging acceleration within case and then averaging the two "
            r"matched seeds; intervals are 95\% case-bootstrap confidence "
            r"intervals.}"
        ),
        r"\label{tab:revision_component_contrasts_full}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Added component & Metric & Control & Component arm & $\Delta$ (95\% CI) \\",
        r"\midrule",
    ]
    for contrast_index, contrast_name in enumerate(COMPONENT_CONTRASTS):
        contrast = analysis["contrasts"][contrast_name]
        for metric in REVISION_METRICS:
            summary = contrast["overall"][metric]
            component_full_lines.append(
                " & ".join(
                    (
                        CONTRAST_LABELS[contrast_name],
                        METRIC_LABELS[metric],
                        _latex_number(summary["control_mean"], metric),
                        _latex_number(summary["treatment_mean"], metric),
                        (
                            _latex_number(
                                summary["mean_delta_treatment_minus_control"],
                                metric,
                                signed=True,
                            )
                            + " "
                            + _latex_ci(summary, metric)
                        ),
                    )
                )
                + r" \\"
            )
        if contrast_index != len(COMPONENT_CONTRASTS) - 1:
            component_full_lines.append(r"\addlinespace")
    component_full_lines.extend(
        [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    )

    joint_overall = analysis["contrasts"]["joint_recipe"]["overall"]
    joint_overall_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        (
            r"\caption{Matched full-KD versus supervised S8, averaged within "
            r"case over acceleration and across two seeds. $\Delta$ is full "
            r"KD minus supervised S8 with a 95\% case-bootstrap confidence "
            r"interval; paired change is the mean within-case relative "
            r"effect. VENC RMSE is reported in cm/s.}"
        ),
        r"\label{tab:joint_recipe_overall}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1.8pt}",
        r"\renewcommand{\arraystretch}{0.96}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Metric & Supervised S8 & Full-KD S8 & $\Delta$ (95\% CI) & Paired change \\",
        r"\midrule",
    ]
    for metric in REVISION_METRICS:
        summary = joint_overall[metric]
        joint_overall_lines.append(
            " & ".join(
                (
                    METRIC_LABELS[metric],
                    _latex_number(summary["control_mean"], metric),
                    _latex_number(summary["treatment_mean"], metric),
                    (
                        _latex_number(
                            summary["mean_delta_treatment_minus_control"],
                            metric,
                            signed=True,
                        )
                        + " "
                        + _latex_ci(summary, metric)
                    ),
                    _latex_percent(summary["mean_relative_delta_percent"]),
                )
            )
            + r" \\"
        )
    joint_overall_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\renewcommand{\arraystretch}{1}",
            r"\end{table}",
        ]
    )

    joint_uncertainty_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        (
            r"\caption{Absolute and relative uncertainty for full KD versus "
            r"supervised S8. Effects are computed within case and acceleration "
            r"before averaging the two matched seeds. Both intervals are "
            r"95\% case-bootstrap confidence intervals; acceleration-specific "
            r"values are reported separately.}"
        ),
        r"\label{tab:joint_recipe_uncertainty}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        (
            r"Metric & Absolute $\Delta$ (95\% CI) & Paired change "
            r"& Relative 95\% CI \\"
        ),
        r"\midrule",
    ]
    for metric in REVISION_METRICS:
        summary = joint_overall[metric]
        joint_uncertainty_lines.append(
            " & ".join(
                (
                    METRIC_LABELS[metric],
                    (
                        _latex_number(
                            summary["mean_delta_treatment_minus_control"],
                            metric,
                            signed=True,
                        )
                        + " "
                        + _latex_ci(summary, metric)
                    ),
                    _latex_percent(summary["mean_relative_delta_percent"]),
                    _latex_relative_ci(summary),
                )
            )
            + r" \\"
        )
    joint_uncertainty_lines.extend(
        [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    )

    by_rate_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        (
            r"\caption{Raw two-seed case means for every revision arm and "
            r"acceleration. Acceleration-specific comparisons are descriptive "
            r"and unadjusted for multiplicity. Bold marks the favorable arm "
            r"mean within each acceleration and metric.}"
        ),
        r"\label{tab:revision_by_usrate}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{rlrrrrrr}",
        r"\toprule",
        (
            r"$R$ & Arm & nRMSE & SSIM & RelErr & AngErr & Norm. $L_1$ "
            r"& VENC RMSE (cm/s) \\"
        ),
        r"\midrule",
    ]
    for rate_index, usrate in enumerate(usrates):
        summaries = {
            arm: analysis["arm_summaries"][arm]["by_usrate"][str(usrate)]
            for arm in ARM_ORDER
        }
        best = {}
        for metric in REVISION_METRICS:
            values = [
                float(summaries[arm][metric]["mean_across_seeds"])
                for arm in ARM_ORDER
            ]
            best[metric] = (
                max(values)
                if METRIC_DIRECTIONS[metric] == "higher"
                else min(values)
            )
        for arm in ARM_ORDER:
            cells = []
            for metric in REVISION_METRICS:
                value = float(summaries[arm][metric]["mean_across_seeds"])
                rendered = _latex_number(value, metric)
                if math.isclose(value, best[metric], rel_tol=1e-12, abs_tol=1e-12):
                    rendered = rf"\textbf{{{rendered}}}"
                cells.append(rendered)
            by_rate_lines.append(
                f"{usrate} & {ARM_LABELS[arm]} & "
                + " & ".join(cells)
                + r" \\"
            )
        if rate_index != len(usrates) - 1:
            by_rate_lines.append(r"\addlinespace")
    by_rate_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\renewcommand{\arraystretch}{1}",
            r"\end{table}",
        ]
    )

    by_rate_component_lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{8pt}",
        r"\setlength{\LTleft}{\fill}",
        r"\setlength{\LTright}{\fill}",
        r"\begin{longtable}{@{}rlr@{}}",
        (
            r"\caption{Acceleration-specific ordered effects for all six "
            r"metrics. Each cell is absolute $\Delta$ [95\% case-bootstrap "
            r"CI], where $\Delta$ is component-containing arm minus control "
            r"after averaging the two matched seeds. VENC RMSE is in cm/s; "
            r"comparisons are descriptive and unadjusted for multiplicity.}"
        ),
        r"\label{tab:supp_revision_physical_by_rate}\\",
        r"\toprule",
        r"$R$ & Metric & Absolute $\Delta$ [95\% CI] \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{3}{c}{\tablename~\thetable\ (continued)} \\",
        r"\toprule",
        r"$R$ & Metric & Absolute $\Delta$ [95\% CI] \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{3}{r}{\textit{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    by_rate_metric_order = (
        "normalized_l1",
        "nrmse",
        "ssim",
        "relerr",
        "angerr",
        "velocity_vector_rmse_cm_s",
    )
    for contrast_index, contrast_name in enumerate(COMPONENT_CONTRASTS):
        by_rate_component_lines.append(
            r"\multicolumn{3}{@{}l}{\textit{"
            + CONTRAST_LABELS[contrast_name]
            + r"}} \\*"
        )
        for usrate in usrates:
            for metric_index, metric in enumerate(by_rate_metric_order):
                summary = analysis["contrasts"][contrast_name]["by_usrate"][
                    str(usrate)
                ][metric]
                by_rate_component_lines.append(
                    (str(usrate) if metric_index == 0 else "")
                    + " & "
                    + METRIC_LABELS[metric]
                    + " & "
                    + _latex_by_rate_effect_number(
                        summary["mean_delta_treatment_minus_control"],
                        metric,
                    )
                    + " "
                    + _latex_by_rate_effect_ci(summary, metric)
                    + (r" \\*" if metric_index < len(by_rate_metric_order) - 1 else r" \\")
                )
            if usrate != usrates[-1]:
                by_rate_component_lines.append(r"\addlinespace[2pt]")
        if contrast_index != len(COMPONENT_CONTRASTS) - 1:
            by_rate_component_lines.append(r"\addlinespace[4pt]")
    by_rate_component_lines.extend(
        [
            r"\end{longtable}",
            r"\endgroup",
        ]
    )

    joint_by_rate_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        (
            r"\caption{Joint-recipe effect by acceleration: Full KD versus "
            r"supervised S8. $\Delta$ is full KD minus supervised S8 after "
            r"averaging the two matched seeds; intervals are 95\% case-"
            r"bootstrap confidence intervals. These comparisons are "
            r"descriptive and unadjusted for multiplicity.}"
        ),
        r"\label{tab:joint_recipe_by_usrate}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\begin{tabular}{rllrrr}",
        r"\toprule",
        r"$R$ & Comparison & Metric & Supervised S8 & Full KD & $\Delta$ (95\% CI) \\",
        r"\midrule",
    ]
    for rate_index, usrate in enumerate(usrates):
        summaries = analysis["contrasts"]["joint_recipe"]["by_usrate"][
            str(usrate)
        ]
        for metric in REVISION_METRICS:
            summary = summaries[metric]
            joint_by_rate_lines.append(
                " & ".join(
                    (
                        str(usrate),
                        "Full KD versus supervised S8",
                        METRIC_LABELS[metric],
                        _latex_number(summary["control_mean"], metric),
                        _latex_number(summary["treatment_mean"], metric),
                        (
                            _latex_number(
                                summary["mean_delta_treatment_minus_control"],
                                metric,
                                signed=True,
                            )
                            + " "
                            + _latex_ci(summary, metric)
                        ),
                    )
                )
                + r" \\"
            )
        if rate_index != len(usrates) - 1:
            joint_by_rate_lines.append(r"\addlinespace")
    joint_by_rate_lines.extend(
        [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    )

    return {
        "overall": "\n".join(overall_lines) + "\n",
        "seed_arm_means": "\n".join(seed_arm_lines) + "\n",
        "component_contrasts_full": "\n".join(component_full_lines) + "\n",
        "joint_recipe_overall": "\n".join(joint_overall_lines) + "\n",
        "joint_recipe_uncertainty": (
            "\n".join(joint_uncertainty_lines) + "\n"
        ),
        "by_usrate": "\n".join(by_rate_lines) + "\n",
        "by_usrate_component_contrasts": (
            "\n".join(by_rate_component_lines) + "\n"
        ),
        "joint_recipe_by_usrate": "\n".join(joint_by_rate_lines) + "\n",
    }


def write_analysis_outputs(
    analysis: dict,
    *,
    output_dir: Path | str,
    prefix: str,
) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    prefix = str(prefix).strip()
    if not prefix or not prefix.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid output prefix: {prefix!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "analysis": output_dir / f"{prefix}_analysis.json",
        "arm_summaries": output_dir / f"{prefix}_arm_summaries.csv",
        "contrasts": output_dir / f"{prefix}_contrasts.csv",
        "component_decisions": output_dir / f"{prefix}_component_decisions.csv",
        "overall_latex": output_dir / f"{prefix}_overall.tex",
        "seed_arm_means_latex": (
            output_dir / f"{prefix}_seed_arm_means.tex"
        ),
        "component_contrasts_full_latex": (
            output_dir / f"{prefix}_component_contrasts_full.tex"
        ),
        "joint_recipe_overall_latex": (
            output_dir / f"{prefix}_joint_recipe_overall.tex"
        ),
        "joint_recipe_uncertainty_latex": (
            output_dir / f"{prefix}_joint_recipe_uncertainty.tex"
        ),
        "by_usrate_latex": output_dir / f"{prefix}_by_usrate.tex",
        "by_usrate_component_contrasts_latex": (
            output_dir / f"{prefix}_by_usrate_component_contrasts.tex"
        ),
        "joint_recipe_by_usrate_latex": (
            output_dir / f"{prefix}_joint_recipe_by_usrate.tex"
        ),
        "manifest": output_dir / f"{prefix}_manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Analysis outputs already exist; preserve them and choose a new "
            f"prefix or directory: {existing}"
        )

    records = flatten_analysis_records(analysis)
    latex_tables = render_revision_latex_tables(analysis)
    payloads = {
        "analysis": (
            json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
        "arm_summaries": _csv_bytes(records["arm_summaries"]),
        "contrasts": _csv_bytes(records["contrasts"]),
        "component_decisions": _csv_bytes(records["component_decisions"]),
        "overall_latex": latex_tables["overall"].encode("utf-8"),
        "seed_arm_means_latex": latex_tables["seed_arm_means"].encode(
            "utf-8"
        ),
        "component_contrasts_full_latex": latex_tables[
            "component_contrasts_full"
        ].encode("utf-8"),
        "joint_recipe_overall_latex": latex_tables[
            "joint_recipe_overall"
        ].encode("utf-8"),
        "joint_recipe_uncertainty_latex": latex_tables[
            "joint_recipe_uncertainty"
        ].encode("utf-8"),
        "by_usrate_latex": latex_tables["by_usrate"].encode("utf-8"),
        "by_usrate_component_contrasts_latex": latex_tables[
            "by_usrate_component_contrasts"
        ].encode("utf-8"),
        "joint_recipe_by_usrate_latex": latex_tables[
            "joint_recipe_by_usrate"
        ].encode("utf-8"),
    }
    output_records = []
    for label, payload in payloads.items():
        path = paths[label]
        path.write_bytes(payload)
        output_records.append(
            {
                "label": label,
                "path": str(path),
                "sha256": _sha256_bytes(payload),
                "bytes": len(payload),
            }
        )
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "purpose": "flowvn_camera_ready_nested_ablation_analysis_bundle",
        "outputs": output_records,
        "analysis_contract": {
            "independent_unit": analysis["cohort"]["independent_unit"],
            "metrics": analysis["metrics"],
            "co_primary_metrics": analysis["co_primary_metrics"],
            "bootstrap_samples": analysis["cohort"]["bootstrap_samples"],
        },
        "script": {
            "path": str(script_path),
            "sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed case-level analysis of the two-seed nested FlowVN "
            "camera-ready distillation ablation"
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=4,
        metavar=("SEED", "ARM", "VALIDATION_CSV", "VALIDATION_AUDIT"),
        required=True,
        help=(
            "Repeat for supervised, final_only, final_trajectory, and full_kd "
            "for every expected seed"
        ),
    )
    parser.add_argument("--expected-seeds", nargs="+", default=("12345", "23456"))
    parser.add_argument("--expected-cases", type=int, default=16)
    parser.add_argument(
        "--expected-usrates", type=int, nargs="+", default=(10, 20, 30, 40, 50)
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="flowvn_revision_ablation")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runs, provenance = load_accepted_revision_runs(
        args.run,
        expected_seeds=args.expected_seeds,
        expected_cases=int(args.expected_cases),
        expected_usrates=args.expected_usrates,
    )
    analysis = analyze_revision_ablation(
        runs,
        expected_seeds=args.expected_seeds,
        expected_cases=int(args.expected_cases),
        expected_usrates=args.expected_usrates,
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    analysis["inputs"] = provenance
    analysis["bootstrap_seed"] = int(args.bootstrap_seed)
    outputs = write_analysis_outputs(
        analysis,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    print(f"Revision ablation analysis: {outputs['analysis']}")
    print(f"Revision ablation manifest: {outputs['manifest']}")


if __name__ == "__main__":
    main()
