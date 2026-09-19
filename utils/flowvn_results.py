from __future__ import annotations

import csv
import hashlib
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("nrmse", "ssim", "relerr", "angerr", "normalized_l1")
PHYSICAL_METRICS = ("velocity_vector_rmse_cm_s",)
REVISION_METRICS = METRICS + PHYSICAL_METRICS
CASE_ID_CONVENTION = "basename-sha256_12(last_three_path_components)"
METRIC_DIRECTIONS = {
    "nrmse": "lower",
    "ssim": "higher",
    "relerr": "lower",
    "angerr": "lower",
    "normalized_l1": "lower",
    "velocity_vector_rmse_cm_s": "lower",
}
GROUP_FIELDS = ("case_id", "slice_start", "usrate")
REQUIRED_COLUMNS = set(GROUP_FIELDS + METRICS)


def validation_key(row: dict) -> tuple[str, int, int]:
    return (str(row["case_id"]), int(row["slice_start"]), int(row["usrate"]))


def _parse_metric(value, *, field: str, key: tuple[str, int, int]):
    if value in (None, "", "None", "null"):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {field} for validation group {key}: {value}")
    return parsed


def load_validation_csv(path: Path | str) -> list[dict]:
    path = Path(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"Missing validation columns in {path}: {missing}")

        rows = []
        keys = set()
        for line_number, raw in enumerate(reader, start=2):
            case_id = str(raw.get("case_id", "")).strip()
            if not case_id:
                raise ValueError(f"Empty case_id in {path} line {line_number}")
            try:
                row = {
                    "case_id": case_id,
                    "slice_start": int(raw["slice_start"]),
                    "usrate": int(raw["usrate"]),
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid validation group key in {path} line {line_number}"
                ) from exc

            key = validation_key(row)
            if key in keys:
                raise ValueError(f"Duplicate validation group in {path}: {key}")
            keys.add(key)
            for metric in REVISION_METRICS:
                row[metric] = _parse_metric(raw.get(metric), field=metric, key=key)
            rows.append(row)

    if not rows:
        raise ValueError(f"Validation CSV is empty: {path}")
    return rows


def require_complete_metrics(
    name: str,
    rows: list[dict],
    *,
    metrics: tuple[str, ...] = METRICS,
) -> None:
    unknown = sorted(set(metrics) - set(REVISION_METRICS))
    if unknown:
        raise ValueError(f"Unknown required metrics: {unknown}")
    missing = [
        (validation_key(row), metric)
        for row in rows
        for metric in metrics
        if row.get(metric) is None
    ]
    if missing:
        preview = ", ".join(f"{key}:{metric}" for key, metric in missing[:5])
        raise ValueError(
            f"Run {name} has {len(missing)} missing metric values; first: {preview}"
        )


def validate_matching_keys(
    left_name: str,
    left_rows: list[dict],
    right_name: str,
    right_rows: list[dict],
) -> None:
    left_keys = {validation_key(row) for row in left_rows}
    right_keys = {validation_key(row) for row in right_rows}
    if left_keys == right_keys:
        return
    left_only = sorted(left_keys - right_keys)
    right_only = sorted(right_keys - left_keys)
    raise ValueError(
        "Validation group mismatch between "
        f"{left_name} and {right_name}: "
        f"left_only={left_only[:5]} ({len(left_only)} total), "
        f"right_only={right_only[:5]} ({len(right_only)} total)"
    )


def validate_workshop_cohort(
    name: str,
    rows: list[dict],
    *,
    expected_groups: int = 80,
    expected_cases: int = 16,
    expected_usrates: tuple[int, ...] = (10, 20, 30, 40, 50),
    require_hashed_case_ids: bool = True,
) -> dict:
    expected_groups = int(expected_groups)
    expected_cases = int(expected_cases)
    expected_usrates = tuple(sorted({int(value) for value in expected_usrates}))
    if expected_groups < 1 or expected_cases < 1 or not expected_usrates:
        raise ValueError("Expected cohort dimensions must be positive and nonempty")
    if len(rows) != expected_groups:
        raise ValueError(
            f"Run {name} has {len(rows)} validation groups; expected {expected_groups}"
        )

    cases = sorted({str(row["case_id"]) for row in rows})
    if len(cases) != expected_cases:
        raise ValueError(
            f"Run {name} has {len(cases)} unique cases; expected {expected_cases}"
        )
    actual_usrates = tuple(sorted({int(row["usrate"]) for row in rows}))
    if actual_usrates != expected_usrates:
        raise ValueError(
            f"Run {name} has acceleration factors {list(actual_usrates)}; "
            f"expected {list(expected_usrates)}"
        )

    rates_by_case = defaultdict(list)
    for row in rows:
        rates_by_case[str(row["case_id"])].append(int(row["usrate"]))
    unbalanced = {
        case_id: sorted(rates)
        for case_id, rates in rates_by_case.items()
        if tuple(sorted(rates)) != expected_usrates
    }
    if unbalanced:
        preview = dict(list(sorted(unbalanced.items()))[:5])
        raise ValueError(
            f"Run {name} does not contain exactly one group per acceleration "
            f"for every case; first: {preview}"
        )

    if require_hashed_case_ids:
        invalid = [
            case_id
            for case_id in cases
            if re.fullmatch(r".+-[0-9a-f]{12}", case_id) is None
        ]
        if invalid:
            raise ValueError(
                f"Run {name} violates the hashed case-ID convention "
                f"{CASE_ID_CONVENTION}: {invalid[:5]}"
            )
    return {
        "n_groups": len(rows),
        "n_cases": len(cases),
        "usrates": list(actual_usrates),
        "case_id_convention": (
            CASE_ID_CONVENTION if require_hashed_case_ids else "not enforced"
        ),
    }


def _rows_for_scope(rows: list[dict], usrate: int | None) -> list[dict]:
    if usrate is None:
        return list(rows)
    return [row for row in rows if int(row["usrate"]) == int(usrate)]


def case_metric_means(
    rows: list[dict], metric: str, usrate: int | None = None
) -> dict[str, float]:
    if metric not in REVISION_METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    grouped = defaultdict(list)
    for row in _rows_for_scope(rows, usrate):
        value = row.get(metric)
        if value is not None:
            grouped[str(row["case_id"])].append(float(value))
    return {
        case_id: float(statistics.fmean(values))
        for case_id, values in grouped.items()
        if values
    }


def select_median_difficulty_case(
    rows: list[dict], *, usrate: int = 40, metric: str = "nrmse"
) -> dict:
    """Select a representative case from control metrics without using treatment effects."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    usrate = int(usrate)
    values = case_metric_means(rows, metric, usrate)
    if not values:
        raise ValueError(f"No finite {metric} values are available at usrate={usrate}")
    cohort_median = float(statistics.median(values.values()))
    distances = {
        case_id: abs(float(value) - cohort_median)
        for case_id, value in values.items()
    }
    minimum_distance = min(distances.values())
    tied_cases = sorted(
        case_id
        for case_id, distance in distances.items()
        if math.isclose(distance, minimum_distance, rel_tol=1e-12, abs_tol=1e-15)
    )
    case_id = tied_cases[0]
    value = values[case_id]
    matching = [
        (index, row)
        for index, row in enumerate(rows)
        if str(row["case_id"]) == case_id and int(row["usrate"]) == usrate
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Expected one row for selected case={case_id}, usrate={usrate}; "
            f"found {len(matching)}"
        )
    group_ordinal, row = matching[0]
    return {
        "case_id": case_id,
        "slice_start": int(row["slice_start"]),
        "usrate": usrate,
        "metric": metric,
        "value": float(value),
        "cohort_median": cohort_median,
        "absolute_distance": abs(float(value) - cohort_median),
        "group_ordinal": int(group_ordinal),
        "selection_source": "control_only",
        "tie_break": "lexicographically smallest stable case_id",
    }


def _descriptive(values: list[float]) -> dict:
    if not values:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summary_records(
    name: str,
    rows: list[dict],
    *,
    bootstrap_samples: int = 0,
    seed: int = 12345,
) -> list[dict]:
    records = []
    scopes = [("overall", None)] + [
        ("usrate", usrate) for usrate in sorted({int(row["usrate"]) for row in rows})
    ]
    for scope, usrate in scopes:
        scoped_rows = _rows_for_scope(rows, usrate)
        for metric in METRICS:
            case_values = case_metric_means(rows, metric, usrate)
            values = list(case_values.values())
            ci_low = None
            ci_high = None
            if values and int(bootstrap_samples) > 0:
                ci_low, ci_high = _bootstrap_mean_ci(
                    values,
                    samples=int(bootstrap_samples),
                    seed_material=(
                        f"{int(seed)}|{name}|summary|{scope}|{usrate}|{metric}"
                    ),
                )
            records.append(
                {
                    "model": str(name),
                    "scope": scope,
                    "usrate": usrate,
                    "metric": metric,
                    "direction": METRIC_DIRECTIONS[metric],
                    "n_cases": len(case_values),
                    "n_rows": sum(
                        1 for row in scoped_rows if row.get(metric) is not None
                    ),
                    **_descriptive(values),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    return records


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def _bootstrap_mean_ci(
    values: list[float], *, samples: int, seed_material: str
) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty sequence")
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    n_values = len(values)
    bootstrap_means = sorted(
        statistics.fmean(values[rng.randrange(n_values)] for _ in range(n_values))
        for _ in range(samples)
    )
    return (
        _percentile(bootstrap_means, 0.025),
        _percentile(bootstrap_means, 0.975),
    )


def _wilcoxon_pvalue(values: list[float]):
    if not values:
        return None
    if all(math.isclose(value, 0.0, abs_tol=1e-15) for value in values):
        return 1.0
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    try:
        return float(wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        return None


def paired_comparison_records(
    left_name: str,
    left_rows: list[dict],
    right_name: str,
    right_rows: list[dict],
    *,
    bootstrap_samples: int = 10000,
    seed: int = 12345,
) -> list[dict]:
    validate_matching_keys(left_name, left_rows, right_name, right_rows)
    rates = sorted({int(row["usrate"]) for row in left_rows})
    records = []
    for scope, usrate in [("overall", None)] + [
        ("usrate", value) for value in rates
    ]:
        for metric in METRICS:
            left_values = case_metric_means(left_rows, metric, usrate)
            right_values = case_metric_means(right_rows, metric, usrate)
            cases = sorted(set(left_values) & set(right_values))
            if not cases:
                continue
            left_case = [left_values[case_id] for case_id in cases]
            right_case = [right_values[case_id] for case_id in cases]
            differences = [
                left_value - right_value
                for left_value, right_value in zip(left_case, right_case)
            ]
            ci_low, ci_high = _bootstrap_mean_ci(
                differences,
                samples=int(bootstrap_samples),
                seed_material=(
                    f"{int(seed)}|{left_name}|{right_name}|{scope}|{usrate}|{metric}"
                ),
            )
            delta = float(statistics.fmean(differences))
            higher_is_better = METRIC_DIRECTIONS[metric] == "higher"
            if higher_is_better:
                improvement = delta
                improvement_ci_low, improvement_ci_high = ci_low, ci_high
            else:
                improvement = -delta
                improvement_ci_low, improvement_ci_high = -ci_high, -ci_low
            mean_right = float(statistics.fmean(right_case))
            denominator = abs(mean_right)
            if denominator > 0.0:
                relative_improvement = 100.0 * improvement / denominator
                relative_ci_low = 100.0 * improvement_ci_low / denominator
                relative_ci_high = 100.0 * improvement_ci_high / denominator
            else:
                relative_improvement = None
                relative_ci_low = None
                relative_ci_high = None
            records.append(
                {
                    "left": str(left_name),
                    "right": str(right_name),
                    "scope": scope,
                    "usrate": usrate,
                    "metric": metric,
                    "direction": METRIC_DIRECTIONS[metric],
                    "n_cases": len(cases),
                    "mean_left": float(statistics.fmean(left_case)),
                    "mean_right": mean_right,
                    "delta_left_minus_right": delta,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "improvement_signed": improvement,
                    "improvement_ci95_low": improvement_ci_low,
                    "improvement_ci95_high": improvement_ci_high,
                    "relative_improvement_percent": relative_improvement,
                    "relative_improvement_ci95_low": relative_ci_low,
                    "relative_improvement_ci95_high": relative_ci_high,
                    "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
                    "wilcoxon_pvalue": _wilcoxon_pvalue(differences),
                }
            )
    return records


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _metric_precision(metric: str) -> int:
    return 2 if metric == "angerr" else 4


def _format_latex_cell(record: dict, *, bold: bool) -> str:
    mean = record.get("mean")
    std = record.get("std")
    if mean is None or std is None:
        return "--"
    precision = _metric_precision(str(record["metric"]))
    mean_text = f"{float(mean):.{precision}f}"
    if bold:
        mean_text = rf"\textbf{{{mean_text}}}"
    return rf"{mean_text} $\pm$ {float(std):.{precision}f}"


def render_overall_latex(
    records: list[dict], model_order: list[str], labels: dict[str, str] | None = None
) -> str:
    labels = labels or {}
    overall = {
        (str(record["model"]), str(record["metric"])): record
        for record in records
        if record.get("scope") == "overall"
    }
    best = {}
    for metric in METRICS:
        candidates = [
            overall[(model, metric)]
            for model in model_order
            if (model, metric) in overall
            and overall[(model, metric)].get("mean") is not None
        ]
        if not candidates:
            best[metric] = None
        elif METRIC_DIRECTIONS[metric] == "higher":
            best[metric] = max(float(record["mean"]) for record in candidates)
        else:
            best[metric] = min(float(record["mean"]) for record in candidates)

    headers = ("nRMSE", "SSIM", "RelErr", "AngErr", "Normalized L1")
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        "Model & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for model in model_order:
        cells = []
        for metric in METRICS:
            record = overall.get((model, metric))
            if record is None:
                cells.append("--")
                continue
            target = best[metric]
            is_best = target is not None and math.isclose(
                float(record["mean"]), target, rel_tol=1e-12, abs_tol=1e-12
            )
            cells.append(_format_latex_cell(record, bold=is_best))
        label = _latex_escape(labels.get(model, model))
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "% Values are case-level mean $\\pm$ sample standard deviation.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_by_usrate_latex(
    records: list[dict], model_order: list[str], labels: dict[str, str] | None = None
) -> str:
    labels = labels or {}
    rates = sorted(
        {
            int(record["usrate"])
            for record in records
            if record.get("scope") == "usrate" and record.get("usrate") is not None
        }
    )
    indexed = {
        (str(record["model"]), int(record["usrate"]), str(record["metric"])): record
        for record in records
        if record.get("scope") == "usrate" and record.get("usrate") is not None
    }
    best = {}
    for usrate in rates:
        for metric in METRICS:
            candidates = [
                indexed[(model, usrate, metric)]
                for model in model_order
                if (model, usrate, metric) in indexed
                and indexed[(model, usrate, metric)].get("mean") is not None
            ]
            if not candidates:
                best[(usrate, metric)] = None
            elif METRIC_DIRECTIONS[metric] == "higher":
                best[(usrate, metric)] = max(
                    float(record["mean"]) for record in candidates
                )
            else:
                best[(usrate, metric)] = min(
                    float(record["mean"]) for record in candidates
                )

    headers = ("nRMSE", "SSIM", "RelErr", "AngErr", "Normalized L1")
    lines = [
        r"\begin{tabular}{lrccccc}",
        r"\toprule",
        "Model & Accel. & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for model in model_order:
        for usrate in rates:
            cells = []
            for metric in METRICS:
                record = indexed.get((model, usrate, metric))
                if record is None:
                    cells.append("--")
                    continue
                target = best[(usrate, metric)]
                is_best = target is not None and math.isclose(
                    float(record["mean"]), target, rel_tol=1e-12, abs_tol=1e-12
                )
                cells.append(_format_latex_cell(record, bold=is_best))
            label = _latex_escape(labels.get(model, model))
            lines.append(
                f"{label} & {usrate}" + " & " + " & ".join(cells) + r" \\"
            )
        if model != model_order[-1]:
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "% Values are case-level mean $\\pm$ sample standard deviation.",
        ]
    )
    return "\n".join(lines) + "\n"
