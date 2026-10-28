"""Fresh primary and forward-OOF refit verification against retained predictions.

This verification deliberately never loads fitted-model caches. It uses the
declared scientific implementation, reconstructed records, locked seed, and
software versions, then independently executes all twenty base fits. It is an
execution-reproducibility check, not an independent implementation of estimators.
Person-level predictions and fitted objects remain in the private output folder;
the JSON report and CSV comparisons contain only aggregates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

# Set limits before importing BLAS-using packages.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))


def comparison(new, old, *, stage: str, model: str, atol: float) -> dict:
    a = np.asarray(new, dtype=np.float64)
    b = np.asarray(old, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {stage}/{model}: {a.shape} != {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"Non-finite probabilities: {stage}/{model}")
    delta = a - b
    return {
        "stage": stage, "model": model, "n": len(a),
        "max_abs_probability_difference": float(np.max(np.abs(delta))),
        "mean_abs_probability_difference": float(np.mean(np.abs(delta))),
        "rmse_probability_difference": float(np.sqrt(np.mean(delta ** 2))),
        "bitwise_equal_values": bool(np.array_equal(a, b)),
        "within_absolute_tolerance": bool(np.allclose(a, b, rtol=0, atol=atol)),
        "absolute_tolerance": atol,
    }


def align(reference: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["pid", "wave"]
    if reference.duplicated(keys).any() or frame.duplicated(keys).any():
        raise ValueError("Duplicate person-wave comparison keys")
    idx = pd.MultiIndex.from_frame(frame[keys])
    result = reference.set_index(keys).loc[idx].reset_index()
    if not np.array_equal(result.exit_label_t1.to_numpy(int), frame.exit_label_t1.to_numpy(int)):
        raise ValueError("Outcome alignment mismatch")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--code-dir", type=Path, required=True)
    ap.add_argument("--reference-private", type=Path, required=True)
    ap.add_argument("--reference-results", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--probability-atol", type=float, default=1e-10)
    ap.add_argument("--metric-atol", type=float, default=1e-10)
    args = ap.parse_args()
    sys.path.insert(0, str(args.code_dir.resolve()))
    from common import BASES, SEED, calibrated, calibrators, f1_threshold, feature_names, fit_model, metrics, predict

    out = args.output_dir
    modeldir = out / "private" / "models"
    if modeldir.exists() and any(modeldir.iterdir()):
        raise RuntimeError("The output model folder is not empty. Use a new folder for a genuinely fresh run.")
    modeldir.mkdir(parents=True, exist_ok=True)
    resultdir = out / "results"
    resultdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    d = pd.read_pickle(args.analysis)
    tr = d[d.wave.le(20)]
    va = d[d.wave.between(21, 23)]
    te = d[d.wave.between(24, 26)]
    if [len(tr), len(va), len(te)] != [86542, 25961, 27063]:
        raise ValueError("Reconstructed split counts differ from the locked benchmark")
    meta_cols = ["pid", "wave", "exit_label_t1"]
    v = va[meta_cols].copy()
    t = te[meta_cols].copy()
    reference_v = align(pd.read_pickle(args.reference_private / "valid_predictions.pkl"), v)
    reference_t = align(pd.read_pickle(args.reference_private / "test_predictions.pkl"), t)
    reference_oof = pd.read_pickle(args.reference_private / "forward_oof.pkl")
    primary_features = feature_names(tr)
    expected_features = json.loads((args.reference_results / "model_features.json").read_text())
    if primary_features != expected_features:
        raise ValueError("Primary feature selection/order differs from the locked benchmark")

    report = {
        "status": "running", "scope": "Fresh execution of twenty primary/forward base fits, one unweighted meta fit, and validation-only calibration and F1 thresholds for five primary learners.",
        "not_reexecuted": ["shuffled OOF sensitivity", "ten-seed repetitions", "other endpoint or subgroup refits", "bootstrap interval generation"],
        "source_analysis_sha256": digest(args.analysis),
        "code_sha256": {n: digest(args.code_dir / n) for n in ["common.py", "feature_definitions.py"]},
        "reference_sha256": {
            **{n: digest(args.reference_private / n) for n in ["valid_predictions.pkl", "test_predictions.pkl", "forward_oof.pkl"]},
            **{n: digest(args.reference_results / n) for n in ["model_features.json", "model_calibration_metrics.csv", "selected_thresholds.csv"]},
        },
        "python": sys.version.split()[0],
        "dependencies": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "xgboost", "joblib"]},
        "seed": SEED, "threads_per_model": 3, "fitted_model_cache_reused": False,
        "counts": {"training": len(tr), "validation": len(va), "test": len(te)},
        "probability_atol": args.probability_atol, "metric_atol": args.metric_atol,
        "fit_records": [], "probability_comparisons": [], "metric_comparisons": [],
    }

    def checkpoint():
        report["elapsed_seconds"] = time.time() - started
        write_json(args.report, report)

    def fit_fresh(name, train, cols, stage):
        print(f"FIT {stage} {name} rows={len(train)} features={len(cols)}", flush=True)
        fitted = fit_model(name, train, cols)
        joblib.dump(fitted, modeldir / f"{stage}_{name}.joblib")
        est = fitted["model"] if name == "catboost" else fitted["model"].named_steps["estimator"]
        iterations = getattr(est, "n_iter_", None)
        if isinstance(iterations, np.ndarray):
            iterations = iterations.tolist()
        report["fit_records"].append({"stage": stage, "model": name, "training_rows": len(train), "features": len(cols), "fit_seconds": fitted["seconds"], "convergence_iterations": iterations})
        checkpoint()
        return fitted

    for name in BASES:
        f = fit_fresh(name, tr, primary_features, "primary")
        v[name] = predict(f, va)
        t[name] = predict(f, te)
        for partition, frame, ref in [("validation", v, reference_v), ("test", t, reference_t)]:
            report["probability_comparisons"].append(comparison(frame[name], ref[name], stage=f"primary_{partition}", model=name, atol=args.probability_atol))
        v.to_pickle(out / "private" / "fresh_base_valid_predictions.pkl")
        t.to_pickle(out / "private" / "fresh_base_test_predictions.pkl")
        print(f"DONE primary {name} test_max_delta={report['probability_comparisons'][-1]['max_abs_probability_difference']:.3g}", flush=True)
        checkpoint()
        del f

    pieces = []
    fold_records = []
    for fend, start, end in [(7, 9, 11), (10, 12, 14), (13, 15, 17), (16, 18, 20)]:
        ftr = tr[tr.wave.le(fend)]
        fo = tr[tr.wave.between(start, end)]
        if not ftr.wave.max() + 1 < fo.wave.min():
            raise ValueError("Forward OOF timing inequality failed")
        cols = feature_names(ftr)
        p = fo[meta_cols].copy()
        ref = align(reference_oof, p)
        for name in BASES:
            f = fit_fresh(name, ftr, cols, f"forward_{fend}")
            p[name] = predict(f, fo)
            record = comparison(p[name], ref[name], stage=f"forward_{fend}", model=name, atol=args.probability_atol)
            report["probability_comparisons"].append(record)
            print(f"DONE forward_{fend} {name} max_delta={record['max_abs_probability_difference']:.3g}", flush=True)
            checkpoint()
            del f
        pieces.append(p)
        p.to_pickle(out / "private" / f"fresh_forward_{fend}.pkl")
        fold_records.append({"train_end": fend, "latest_training_outcome_wave": fend + 1, "oof_start": start, "oof_end": end, "train_n": len(ftr), "oof_n": len(fo), "feature_n": len(cols)})

    oof = pd.concat(pieces)
    oof.to_pickle(out / "private" / "fresh_forward_oof.pkl")
    meta = LogisticRegression(C=1, max_iter=2000, class_weight=None).fit(oof[BASES], oof.exit_label_t1.astype(int))
    joblib.dump(meta, modeldir / "forward_stack.joblib")
    v["forward_stack"] = meta.predict_proba(v[BASES])[:, 1]
    t["forward_stack"] = meta.predict_proba(t[BASES])[:, 1]
    report["meta_coefficients"] = {n: float(c) for n, c in zip(BASES, meta.coef_[0])}
    report["meta_intercept"] = float(meta.intercept_[0])
    report["meta_convergence_iterations"] = int(meta.n_iter_[0])
    report["forward_folds"] = fold_records

    expected_metrics = pd.read_csv(args.reference_results / "model_calibration_metrics.csv")
    expected_thresholds = pd.read_csv(args.reference_results / "selected_thresholds.csv")
    metric_rows = []
    threshold_rows = []
    for name in BASES + ["forward_stack"]:
        cal = calibrators(v[name].to_numpy(), v.exit_label_t1.astype(int).to_numpy())
        joblib.dump(cal, modeldir / f"{name}_calibrators.joblib")
        for method in ["raw", "platt", "isotonic"]:
            col = name if method == "raw" else name + "_" + method
            v[col] = calibrated(cal, v[name].to_numpy(), method)
            t[col] = calibrated(cal, t[name].to_numpy(), method)
            threshold = f1_threshold(v.exit_label_t1, v[col])
            oldth = expected_thresholds[(expected_thresholds.model == name) & (expected_thresholds.method == method)].iloc[0].threshold
            threshold_rows.append({"model": name, "method": method, "threshold": threshold, "reference_threshold": oldth, "absolute_difference": abs(threshold - oldth)})
            for partition, frame, ref in [("validation", v, reference_v), ("test", t, reference_t)]:
                report["probability_comparisons"].append(comparison(frame[col], ref[col], stage=f"{method}_{partition}", model=name, atol=args.probability_atol))
                computed = metrics(frame.exit_label_t1, frame[col], threshold)
                expected = expected_metrics[(expected_metrics.model == name) & (expected_metrics.method == method) & (expected_metrics.partition == partition)].iloc[0]
                metric_rows.append({"model": name, "method": method, "partition": partition, **computed})
                for key, value in computed.items():
                    old = float(expected[key])
                    difference = abs(float(value) - old)
                    report["metric_comparisons"].append({"model": name, "method": method, "partition": partition, "metric": key, "fresh": value, "reference": old, "absolute_difference": difference, "within_absolute_tolerance": bool(difference <= args.metric_atol)})
        print(f"CALIBRATED {name}", flush=True)

    v.to_pickle(out / "private" / "fresh_valid_predictions.pkl")
    t.to_pickle(out / "private" / "fresh_test_predictions.pkl")
    pd.DataFrame(metric_rows).to_csv(resultdir / "fresh_model_calibration_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(resultdir / "fresh_threshold_comparisons.csv", index=False)
    pd.DataFrame(report["probability_comparisons"]).to_csv(resultdir / "fresh_probability_comparisons.csv", index=False)
    pd.DataFrame(report["metric_comparisons"]).to_csv(resultdir / "fresh_metric_comparisons.csv", index=False)
    report["base_fit_count"] = len(report["fit_records"])
    report["probability_comparisons_pass"] = all(x["within_absolute_tolerance"] for x in report["probability_comparisons"])
    report["metric_comparisons_pass"] = all(x["within_absolute_tolerance"] for x in report["metric_comparisons"])
    report["threshold_comparisons_pass"] = all(x["absolute_difference"] <= args.probability_atol for x in threshold_rows)
    report["status"] = "passed" if all(report[x] for x in ["probability_comparisons_pass", "metric_comparisons_pass", "threshold_comparisons_pass"]) else "differences_detected"
    checkpoint()
    print(json.dumps({"status": report["status"], "base_fit_count": report["base_fit_count"], "elapsed_seconds": report["elapsed_seconds"], "probability_comparisons_pass": report["probability_comparisons_pass"], "metric_comparisons_pass": report["metric_comparisons_pass"], "threshold_comparisons_pass": report["threshold_comparisons_pass"]}), flush=True)


if __name__ == "__main__":
    main()
