#!/usr/bin/env python3
"""Verify released aggregate arithmetic with the Python standard library.

Run from any directory: python python/verify_aggregate_release.py
No raw records, private predictions, model caches or fitted models are read.
This checks published aggregate consistency; it does not refit models or
independently establish how the underlying bootstrap samples were generated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


METRICS = (
    "roc_auc", "pr_auc", "brier", "recall20", "lift20", "ece10",
    "calibration_intercept", "calibration_slope",
)
ATOL = 1e-12
RTOL = 1e-9


class VerificationError(ValueError):
    """A required aggregate schema or identity is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def number(value: str, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"Invalid numeric value: {label}") from exc
    require(math.isfinite(result), f"Non-finite numeric value: {label}")
    return result


def integer(value: str, label: str) -> int:
    result = number(value, label)
    require(result.is_integer(), f"Expected an integer: {label}")
    return int(result)


def percentile(values: list[float], probability: float) -> float:
    """Linear sample quantile, matching numpy.quantile(method='linear')."""
    ordered = sorted(values)
    require(bool(ordered), "Cannot compute a percentile of an empty sample")
    position = (len(ordered) - 1) * probability
    left = math.floor(position)
    right = math.ceil(position)
    weight = position - left
    return ordered[left] + weight * (ordered[right] - ordered[left])


def verify(results_dir: Path) -> dict:
    report = {
        "status": "FAIL",
        "scope": "Released aggregate arithmetic; no raw or private data used",
        "tolerances": {"absolute": ATOL, "relative": RTOL},
        "percentile_method": "linear interpolation at probabilities 0.025 and 0.975",
        "seed_sd_convention": "sample standard deviation (ddof=1)",
        "sources": [],
        "checks": [],
        "failures": [],
    }

    def read(name: str, required_columns: tuple[str, ...], expected_rows: int) -> list[dict]:
        path = results_dir / name
        require(path.is_file(), f"Missing aggregate source: {name}")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise VerificationError(f"Cannot read aggregate source: {name}") from exc
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        columns = reader.fieldnames or []
        require(len(columns) == len(set(columns)), f"Duplicate column names: {name}")
        require(set(required_columns).issubset(columns), f"Required columns missing: {name}")
        rows = list(reader)
        require(len(rows) == expected_rows,
                f"Unexpected row count in {name}: {len(rows)}; expected {expected_rows}")
        report["sources"].append({
            "file": name, "rows": len(rows),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
        return rows

    def close(actual: float, expected: float, label: str) -> None:
        if not math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL):
            report["failures"].append({
                "check": label, "recomputed": actual, "reported": expected,
                "absolute_difference": abs(actual - expected),
            })

    def summary(name: str, count: int, before: int, sources: list[str], **details) -> None:
        report["checks"].append({
            "check": name, "records_checked": count,
            "status": "PASS" if len(report["failures"]) == before else "FAIL",
            "source_files": sources, **details,
        })

    try:
        draws_name = "aggregate_bootstrap_replicates.csv"
        ci_name = "primary_cluster_confidence_intervals.csv"
        pair_name = "paired_cluster_model_contrasts.csv"
        seed_name = "seed_metrics.csv"
        seed_contrast_name = "seed_contrasts.csv"
        utility_name = "utility_scenarios.csv"
        drift_name = "drift_decomposition.csv"
        draws = read(draws_name, ("replicate", "model") + METRICS, 15000)
        intervals = read(ci_name, (
            "model", "metric", "estimate", "lower", "upper", "resamples", "rows",
        ), 120)
        pairs = read(pair_name, (
            "model_A", "model_B", "metric", "difference_A_minus_B",
            "lower", "upper", "resamples",
        ), 56)
        seed_metrics = read(seed_name, ("seed", "model", "method") + METRICS, 120)
        seed_contrasts = read(seed_contrast_name, (
            "model_A", "model_B", "method", "metric", "seeds", "mean_A_minus_B",
            "sd_A_minus_B", "minimum", "maximum", "positive_differences",
            "negative_differences",
        ), 50)
        utility = read(utility_name, (
            "model", "method", "scenario", "benefit_TP", "cost_FP", "cost_FN",
            "test_TP", "test_FP", "test_FN", "utility_units", "utility_per_1000",
            "treat_none_utility", "treat_all_utility",
        ), 18)
        drift = read(drift_name, (
            "from_partition", "to_partition", "component", "estimate",
        ), 12)

        by_model: dict[str, dict[int, dict[str, float]]] = {}
        for row in draws:
            model = row["model"]
            replicate = integer(row["replicate"], "bootstrap replicate")
            group = by_model.setdefault(model, {})
            require(replicate not in group, f"Duplicate bootstrap key: {model}/{replicate}")
            group[replicate] = {metric: number(row[metric], f"{model}/{metric}")
                                for metric in METRICS}
        require(len(by_model) == 15, "Expected 15 bootstrap model/output variants")
        for model, group in by_model.items():
            require(set(group) == set(range(1000)), f"Incomplete replicate set: {model}")

        before = len(report["failures"])
        estimates: dict[tuple[str, str], float] = {}
        for row in intervals:
            model, metric = row["model"], row["metric"]
            key = (model, metric)
            require(key not in estimates, f"Duplicate primary interval: {model}/{metric}")
            require(model in by_model and metric in METRICS, f"Unknown interval key: {key}")
            require(integer(row["resamples"], "primary resamples") == 1000,
                    "Primary resample count must be 1000")
            estimates[key] = number(row["estimate"], f"primary estimate {key}")
            values = [entry[metric] for entry in by_model[model].values()]
            for bound, probability in (("lower", 0.025), ("upper", 0.975)):
                close(percentile(values, probability), number(row[bound], f"{key}/{bound}"),
                      f"primary percentile {model}/{metric}/{bound}")
        require(set(estimates) == {(model, metric) for model in by_model for metric in METRICS},
                "Primary intervals do not cover every model/metric combination")
        sample_sizes = {integer(row["rows"], "test sample size") for row in intervals}
        require(len(sample_sizes) == 1, "Primary intervals disagree on test sample size")
        sample_n = next(iter(sample_sizes))
        require(sample_n > 0, "Test sample size must be positive")
        summary("primary_percentile_intervals", 120, before, [draws_name, ci_name],
                bounds_checked=240, replicates_per_variant=1000, variants=15)

        before = len(report["failures"])
        pair_keys = set()
        for row in pairs:
            a, b, metric = row["model_A"], row["model_B"], row["metric"]
            key = (a, b, metric)
            require(key not in pair_keys, f"Duplicate paired contrast: {key}")
            pair_keys.add(key)
            require((a, metric) in estimates and (b, metric) in estimates,
                    f"Unknown paired-contrast key: {key}")
            require(integer(row["resamples"], "paired resamples") == 1000,
                    "Paired resample count must be 1000")
            # Join on replicate identifiers, never on input row order.
            values = [by_model[a][replicate][metric] - by_model[b][replicate][metric]
                      for replicate in sorted(by_model[a])]
            for bound, probability in (("lower", 0.025), ("upper", 0.975)):
                close(percentile(values, probability), number(row[bound], f"{key}/{bound}"),
                      f"paired percentile {a} minus {b}/{metric}/{bound}")
            close(estimates[(a, metric)] - estimates[(b, metric)],
                  number(row["difference_A_minus_B"], f"paired point {key}"),
                  f"paired point estimate {a} minus {b}/{metric}")
        summary("paired_model_contrasts", 56, before, [draws_name, ci_name, pair_name],
                bounds_checked=112, point_differences_checked=56,
                pairing="identical bootstrap replicate identifiers")

        before = len(report["failures"])
        by_seed: dict[tuple[str, str], dict[int, dict]] = {}
        for row in seed_metrics:
            key = (row["model"], row["method"])
            seed = integer(row["seed"], "seed identifier")
            group = by_seed.setdefault(key, {})
            require(seed not in group, f"Duplicate seed metric key: {key}/{seed}")
            group[seed] = row
        seed_keys = set()
        for row in seed_contrasts:
            a, b, method, metric = (row[k] for k in ("model_A", "model_B", "method", "metric"))
            key = (a, b, method, metric)
            require(key not in seed_keys, f"Duplicate seed contrast: {key}")
            seed_keys.add(key)
            require((a, method) in by_seed and (b, method) in by_seed and metric in METRICS,
                    f"Unknown seed-contrast key: {key}")
            ga, gb = by_seed[(a, method)], by_seed[(b, method)]
            require(set(ga) == set(gb) and len(ga) == 10,
                    f"Seed contrast requires ten matched seeds: {key}")
            require(integer(row["seeds"], "reported seed count") == 10,
                    f"Reported seed count differs from ten: {key}")
            values = [number(ga[seed][metric], f"seed metric {key}") -
                      number(gb[seed][metric], f"seed metric {key}") for seed in sorted(ga)]
            computed = {
                "mean_A_minus_B": statistics.mean(values),
                "sd_A_minus_B": statistics.stdev(values),
                "minimum": min(values), "maximum": max(values),
                "positive_differences": sum(value > 0 for value in values),
                "negative_differences": sum(value < 0 for value in values),
            }
            for field, actual in computed.items():
                close(actual, number(row[field], f"seed summary {key}/{field}"),
                      f"seed contrast {a} minus {b}/{method}/{metric}/{field}")
        summary("ten_seed_contrast_summaries", 50, before, [seed_name, seed_contrast_name],
                paired_seeds_per_contrast=10, summary_values_checked=300)

        before = len(report["failures"])
        utility_keys = set()
        for row in utility:
            key = (row["model"], row["method"], row["scenario"])
            require(key not in utility_keys, f"Duplicate utility scenario: {key}")
            utility_keys.add(key)
            tp, fp, fn = (integer(row[field], f"utility {key}/{field}")
                          for field in ("test_TP", "test_FP", "test_FN"))
            require(min(tp, fp, fn, sample_n - tp - fp - fn) >= 0,
                    f"Invalid confusion counts: {key}")
            benefit, cost_fp, cost_fn = (number(row[field], f"utility {key}/{field}")
                                         for field in ("benefit_TP", "cost_FP", "cost_FN"))
            total = benefit * tp - cost_fp * fp - cost_fn * fn
            events = tp + fn
            computed = {
                "utility_units": total,
                "utility_per_1000": total / sample_n * 1000,
                "treat_none_utility": -cost_fn * events,
                "treat_all_utility": benefit * events - cost_fp * (sample_n - events),
            }
            for field, actual in computed.items():
                close(actual, number(row[field], f"utility {key}/{field}"),
                      f"utility {key}/{field}")
        summary("utility_arithmetic_and_reference_policies", 18, before,
                [utility_name, ci_name], values_checked=72, test_rows=sample_n)

        before = len(report["failures"])
        drift_groups: dict[tuple[str, str], dict[str, float]] = {}
        required_components = {
            "total_event_rate_change", "age_cohort_composition_common_support",
            "within_stratum_rate_change_common_support", "outside_common_support_contribution",
        }
        for row in drift:
            key = (row["from_partition"], row["to_partition"])
            group = drift_groups.setdefault(key, {})
            component = row["component"]
            require(component not in group, f"Duplicate drift component: {key}/{component}")
            group[component] = number(row["estimate"], f"drift {key}/{component}")
        require(set(drift_groups) == {("train", "validation"), ("validation", "test"),
                                      ("train", "test")}, "Unexpected drift partition contrasts")
        for key, group in drift_groups.items():
            require(set(group) == required_components, f"Incomplete drift decomposition: {key}")
            total = math.fsum(value for component, value in group.items()
                              if component != "total_event_rate_change")
            close(total, group["total_event_rate_change"], f"drift component sum {key}")
        summary("drift_decomposition_identities", 3, before, [drift_name],
                components_per_contrast=3)
    except (VerificationError, UnicodeError, csv.Error) as exc:
        report["failures"].append({"check": "input_schema_or_completeness", "message": str(exc)})

    if not report["failures"] and len(report["checks"]) == 5:
        report["status"] = "PASS"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results",
                        help="Directory containing the seven released aggregate CSV files")
    parser.add_argument("--output", type=Path,
                        help="JSON report path; defaults to RESULTS_DIR/aggregate_release_verification.json")
    args = parser.parse_args()
    report = verify(args.results_dir)
    output = args.output or args.results_dir / "aggregate_release_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: aggregate release verification; report={output.name}")
    for check in report["checks"]:
        print(f"  {check['status']}: {check['check']} ({check['records_checked']} records)")
    for failure in report["failures"]:
        print(f"  FAIL: {failure['check']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
