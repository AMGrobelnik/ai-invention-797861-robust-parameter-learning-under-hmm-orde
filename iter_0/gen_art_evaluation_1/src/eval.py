#!/usr/bin/env python3
"""
KL-Constrained Baum-Welch Evaluation:
- Table 1 mean accuracies + bootstrap CIs for Δ1/Δ2 per (k_true, d)
- Symmetric outlier analysis (IQR) per (k_true, d, T) across seeds
- Bias-variance refit with fixed α=0.5 power law, suppressing R²<0.30
- ε-sensitivity analysis from iter_1 data (d=4)
- Constraint activation rates at d=16
- Narrative claim validation
"""

import json
import sys
import math
import resource
import collections
from pathlib import Path

import numpy as np
from scipy import stats
from loguru import logger

# ── logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO",
           format="{time:HH:mm:ss}|{level:<7}|{function}| {message}")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/eval.log", rotation="30 MB", level="DEBUG")

# ── resource limits ──────────────────────────────────────────────────────────
_AVAIL_BYTES = 13 * 1024 ** 3
RAM_BUDGET = int(_AVAIL_BYTES * 0.5)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
EXP1_FULL = Path(
    "/home/adrian/projects/ai-inventor/aii_data/users/admin/runs"
    "/run_anDiSbGaeE4M/3_invention_loop/iter_1/gen_art"
    "/gen_art_experiment_1/full_method_out.json"
)
EXP2_FULL = Path(
    "/home/adrian/projects/ai-inventor/aii_data/users/admin/runs"
    "/run_anDiSbGaeE4M/3_invention_loop/iter_2/gen_art"
    "/gen_art_experiment_1/full_method_out.json"
)

N_BOOTSTRAP = 10_000
RNG_SEED = 42


# ── helpers ───────────────────────────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray, stat_fn, n: int = N_BOOTSTRAP,
                 alpha: float = 0.05, rng: np.random.Generator = None):
    """Bootstrap confidence interval for stat_fn applied to values (1D)."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    boots = [stat_fn(rng.choice(values, size=len(values), replace=True))
             for _ in range(n)]
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (lo, hi)


def iqr_outliers(values: np.ndarray):
    """Return boolean mask marking outliers via IQR rule (|v - median| > 2*IQR)."""
    if len(values) < 2:
        return np.zeros(len(values), dtype=bool)
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    med = np.median(values)
    return np.abs(values - med) > 2 * iqr


def fit_bv_fixed_alpha(T_vals: np.ndarray, err_vals: np.ndarray):
    """
    Fit f(T) = bias² + c / T^0.5 (alpha=0.5 fixed) to (T, 1-accuracy) pairs.
    Returns dict with bias, c, r_squared.
    """
    y = err_vals  # 1 - accuracy → error
    X = 1.0 / np.sqrt(T_vals)  # T^{-0.5}
    # OLS: y = bias + c * X
    A = np.column_stack([np.ones_like(X), X])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        bias_est, c_est = float(coeffs[0]), float(coeffs[1])
    except Exception:
        return {"bias": None, "c": None, "r_squared": None}

    y_pred = bias_est + c_est * X
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"bias": bias_est, "c": c_est, "r_squared": r2}


def fmt_bv(fit):
    """Format bias-variance fit, suppressing unreliable estimates (R²<0.30)."""
    if fit["r_squared"] is None:
        return {"bias": "N/A (fit failed)", "c": "N/A (fit failed)",
                "r_squared": None}
    r2 = fit["r_squared"]
    if r2 < 0.30:
        return {"bias": f"N/A (R²={r2:.3f}<0.30)",
                "c": f"N/A (R²={r2:.3f}<0.30)", "r_squared": r2}
    return {"bias": round(fit["bias"], 6), "c": round(fit["c"], 6),
            "r_squared": round(r2, 4)}


# ── data loading ──────────────────────────────────────────────────────────────

@logger.catch
def load_iter2() -> list[dict]:
    logger.info(f"Loading iter_2 data: {EXP2_FULL}")
    raw = json.loads(EXP2_FULL.read_text())
    exs = raw["datasets"][0]["examples"]
    logger.info(f"Loaded {len(exs)} iter_2 examples")
    return exs


@logger.catch
def load_iter1() -> list[dict]:
    logger.info(f"Loading iter_1 data: {EXP1_FULL}")
    raw = json.loads(EXP1_FULL.read_text())
    exs = raw["datasets"][0]["examples"]
    logger.info(f"Loaded {len(exs)} iter_1 examples")
    return exs


# ── core analysis ─────────────────────────────────────────────────────────────

def analyse_iter2(exs: list[dict]):
    """All primary analyses on iter_2 data (BW, BW-warm, KL-BW)."""
    rng = np.random.default_rng(RNG_SEED)

    # Build structured data
    records = []
    for e in exs:
        records.append({
            "seed": int(e["metadata_seed"]),
            "k_true": int(e["metadata_k_true"]),
            "d": int(e["metadata_d"]),
            "T": int(e["metadata_T"]),
            "bw": float(e["predict_bw"]),
            "bw_warm": float(e["predict_bw_warm"]),
            "kl_bw": float(e["predict_kl_bw"]),
            "car": float(e["metadata_constraint_activation_rate"]),  # constraint activation rate
        })

    k_trues = sorted(set(r["k_true"] for r in records))
    ds = sorted(set(r["d"] for r in records))
    Ts = sorted(set(r["T"] for r in records))
    seeds = sorted(set(r["seed"] for r in records))
    logger.info(f"k_trues={k_trues}, ds={ds}, Ts={Ts}, n_seeds={len(seeds)}")

    # ── Table 1: per (k_true, d) mean accuracy + Δ1/Δ2 with bootstrap CIs ──
    table1 = {}
    for k in k_trues:
        for d in ds:
            cell = [r for r in records if r["k_true"] == k and r["d"] == d]
            # Group by seed to get per-seed means (over all T)
            seed_means = {}
            for s in seeds:
                sc = [r for r in cell if r["seed"] == s]
                if sc:
                    seed_means[s] = {
                        "bw": np.mean([r["bw"] for r in sc]),
                        "bw_warm": np.mean([r["bw_warm"] for r in sc]),
                        "kl_bw": np.mean([r["kl_bw"] for r in sc]),
                    }
            sm_bw = np.array([seed_means[s]["bw"] for s in sorted(seed_means)])
            sm_bww = np.array([seed_means[s]["bw_warm"] for s in sorted(seed_means)])
            sm_kl = np.array([seed_means[s]["kl_bw"] for s in sorted(seed_means)])

            delta1_seeds = sm_bww - sm_bw
            delta2_seeds = sm_kl - sm_bww

            ci_d1 = bootstrap_ci(delta1_seeds, np.mean, rng=rng)
            ci_d2 = bootstrap_ci(delta2_seeds, np.mean, rng=rng)

            key = f"k{k}_d{d}"
            table1[key] = {
                "k_true": k, "d": d,
                "mean_bw": round(float(np.mean(sm_bw)), 4),
                "mean_bw_warm": round(float(np.mean(sm_bww)), 4),
                "mean_kl_bw": round(float(np.mean(sm_kl)), 4),
                "delta1_bw_warm_vs_bw": round(float(np.mean(delta1_seeds)) * 100, 4),  # pp
                "delta2_kl_bw_vs_bw_warm": round(float(np.mean(delta2_seeds)) * 100, 4),
                "ci_delta1_95": [round(ci_d1[0] * 100, 4), round(ci_d1[1] * 100, 4)],
                "ci_delta2_95": [round(ci_d2[0] * 100, 4), round(ci_d2[1] * 100, 4)],
                "se_delta1": round((ci_d1[1] - ci_d1[0]) / 3.92 * 100, 4),
                "se_delta2": round((ci_d2[1] - ci_d2[0]) / 3.92 * 100, 4),
                "delta1_ci_crosses_zero": bool(ci_d1[0] < 0 < ci_d1[1]),
                "delta2_ci_crosses_zero": bool(ci_d2[0] < 0 < ci_d2[1]),
                "n_seed_means": len(sm_bw),
            }
    logger.info(f"Table 1 computed: {len(table1)} cells")

    # ── Outlier analysis per (k_true, d, T) ──────────────────────────────────
    outlier_results = {}
    for k in k_trues:
        for d in ds:
            for T in Ts:
                cell = [r for r in records
                        if r["k_true"] == k and r["d"] == d and r["T"] == T]
                if not cell:
                    continue
                seed_order = [r["seed"] for r in cell]
                bw_arr = np.array([r["bw"] for r in cell])
                bw_warm_arr = np.array([r["bw_warm"] for r in cell])
                kl_arr = np.array([r["kl_bw"] for r in cell])

                key = f"k{k}_d{d}_T{T}"
                outlier_results[key] = {
                    "k_true": k, "d": d, "T": T,
                    "bw_outlier_seeds": [seed_order[i] for i, v in
                                         enumerate(iqr_outliers(bw_arr)) if v],
                    "bw_warm_outlier_seeds": [seed_order[i] for i, v in
                                              enumerate(iqr_outliers(bw_warm_arr)) if v],
                    "kl_bw_outlier_seeds": [seed_order[i] for i, v in
                                            enumerate(iqr_outliers(kl_arr)) if v],
                    "bw_acc_per_seed": {seed_order[i]: round(float(v), 4)
                                        for i, v in enumerate(bw_arr)},
                    "kl_bw_acc_per_seed": {seed_order[i]: round(float(v), 4)
                                           for i, v in enumerate(kl_arr)},
                }
    logger.info(f"Outlier analysis: {len(outlier_results)} cells")

    # ── Symmetric outlier comparison: k=3/d=8 gains vs d=16 harm ─────────────
    symmetric_outlier_comparison = {}
    for k in k_trues:
        for d in ds:
            d16_gains_harm = []
            for T in Ts:
                cell = [r for r in records
                        if r["k_true"] == k and r["d"] == d and r["T"] == T]
                if not cell:
                    continue
                kl_arr = np.array([r["kl_bw"] for r in cell])
                bw_arr = np.array([r["bw"] for r in cell])
                delta_arr = kl_arr - bw_arr  # KL-BW gain over BW (can be neg)
                outlier_mask = iqr_outliers(delta_arr)
                n_out = int(outlier_mask.sum())
                mean_all = float(np.mean(delta_arr))
                if n_out > 0:
                    mean_no_out = float(np.mean(delta_arr[~outlier_mask]))
                else:
                    mean_no_out = mean_all
                d16_gains_harm.append({
                    "T": T,
                    "n_outliers": n_out,
                    "mean_delta_with_outliers": round(mean_all * 100, 4),
                    "mean_delta_without_outliers": round(mean_no_out * 100, 4),
                    "difference_pp": round((mean_all - mean_no_out) * 100, 4),
                })
            symmetric_outlier_comparison[f"k{k}_d{d}"] = d16_gains_harm

    # ── Constraint activation at d=16 ────────────────────────────────────────
    constraint_activation_d16 = {}
    for k in k_trues:
        for T in Ts:
            cell = [r for r in records
                    if r["k_true"] == k and r["d"] == 16 and r["T"] == T]
            if not cell:
                continue
            # "activates" if constraint_activation_rate > 0 for this seed
            n_activating = sum(1 for r in cell if r["car"] > 0)
            key = f"k{k}_T{T}"
            constraint_activation_d16[key] = {
                "k_true": k, "T": T,
                "n_activating_seeds": n_activating,
                "n_total_seeds": len(cell),
                "mean_car": round(float(np.mean([r["car"] for r in cell])), 6),
            }
    logger.info(f"Constraint activation d=16: {len(constraint_activation_d16)} cells")

    # ── Bias-Variance refit (α=0.5 fixed) ────────────────────────────────────
    bv_table = {}
    for k in k_trues:
        for method in ["bw", "bw_warm", "kl_bw"]:
            # For each T, compute mean error (1-accuracy) across all seeds and d values
            # As per artifact plan: per (k_true, method) curve
            T_list, err_list = [], []
            for T in Ts:
                cell = [r for r in records if r["k_true"] == k and r["T"] == T]
                accs = [r[method] for r in cell]
                if accs:
                    T_list.append(T)
                    err_list.append(1.0 - float(np.mean(accs)))

            if len(T_list) < 3:
                continue
            T_arr = np.array(T_list, dtype=float)
            err_arr = np.array(err_list, dtype=float)
            fit = fit_bv_fixed_alpha(T_arr, err_arr)
            bv_table[f"k{k}_{method}"] = {
                "k_true": k, "method": method,
                **fmt_bv(fit),
            }
    logger.info(f"Bias-variance table: {len(bv_table)} entries")

    # ── Delta2 at d=16: with vs without outlier seed ─────────────────────────
    d16_delta2_analysis = {}
    for k in k_trues:
        for T in Ts:
            cell = [r for r in records
                    if r["k_true"] == k and r["d"] == 16 and r["T"] == T]
            if not cell:
                continue
            kl_arr = np.array([r["kl_bw"] for r in cell])
            bww_arr = np.array([r["bw_warm"] for r in cell])
            delta2_arr = kl_arr - bww_arr
            outlier_mask = iqr_outliers(delta2_arr)
            n_out = int(outlier_mask.sum())
            mean_with = float(np.mean(delta2_arr))
            mean_without = float(np.mean(delta2_arr[~outlier_mask])) if n_out < len(delta2_arr) else float("nan")
            d16_delta2_analysis[f"k{k}_T{T}"] = {
                "k_true": k, "T": T,
                "n_outlier_seeds": n_out,
                "delta2_mean_with_outliers_pp": round(mean_with * 100, 4),
                "delta2_mean_without_outliers_pp": round(mean_without * 100, 4) if not math.isnan(mean_without) else None,
            }

    return {
        "table1": table1,
        "outlier_results": outlier_results,
        "symmetric_outlier_comparison": symmetric_outlier_comparison,
        "constraint_activation_d16": constraint_activation_d16,
        "bv_table": bv_table,
        "d16_delta2_analysis": d16_delta2_analysis,
        "records": records,
        "k_trues": k_trues, "ds": ds, "Ts": Ts, "seeds": seeds,
    }


def analyse_iter1(exs: list[dict]):
    """ε-sensitivity analysis from iter_1 data (d=4, ε∈{0.01,0.05,0.1,0.5})."""
    eps_keys = ["predict_kl_eps_0_01", "predict_kl_eps_0_05",
                "predict_kl_eps_0_1", "predict_kl_eps_0_5"]
    eps_vals = [0.01, 0.05, 0.1, 0.5]

    records = []
    for e in exs:
        row = {
            "seed": int(e["metadata_seed"]),
            "k_true": int(e["metadata_k_true"]),
            "T": int(e["metadata_T"]),
            "d": 4,
            "bw": float(json.loads(e["predict_standard_bw"])["accuracy"]),
        }
        for ek, ev in zip(eps_keys, eps_vals):
            row[f"eps_{ev}"] = float(json.loads(e[ek])["accuracy"])
        records.append(row)

    k_trues = sorted(set(r["k_true"] for r in records))
    Ts = sorted(set(r["T"] for r in records))

    # ε-sensitivity table: per (k_true, T), mean accuracy per ε
    eps_sensitivity = {}
    for k in k_trues:
        for T in Ts:
            cell = [r for r in records if r["k_true"] == k and r["T"] == T]
            if not cell:
                continue
            key = f"k{k}_T{T}_d4"
            entry = {"k_true": k, "T": T, "d": 4}
            for ev in eps_vals:
                accs = [r[f"eps_{ev}"] for r in cell]
                entry[f"mean_acc_eps_{ev}"] = round(float(np.mean(accs)), 4)
            # Max-min spread (claim validation: KL-BW reduces to BW-warm for d≤8)
            means = [entry[f"mean_acc_eps_{ev}"] for ev in eps_vals]
            entry["max_min_spread_pp"] = round((max(means) - min(means)) * 100, 4)
            eps_sensitivity[key] = entry

    # Per-seed ε trajectories for stability assessment
    per_seed_eps = {}
    for k in k_trues:
        for T in Ts:
            cell = [r for r in records if r["k_true"] == k and r["T"] == T]
            key = f"k{k}_T{T}_d4"
            per_seed_eps[key] = {
                "seeds": {r["seed"]: {f"eps_{ev}": round(r[f"eps_{ev}"], 4)
                                       for ev in eps_vals}
                           for r in cell}
            }

    # ε(d) functional form comparison: 1/log(d) vs 1/d at d∈{4,8,16}
    eps_functional_forms = {
        "d4":  {"1_over_log_d": round(1 / math.log(4), 4),
                "1_over_d": round(1 / 4, 4)},
        "d8":  {"1_over_log_d": round(1 / math.log(8), 4),
                "1_over_d": round(1 / 8, 4)},
        "d16": {"1_over_log_d": round(1 / math.log(16), 4),
                "1_over_d": round(1 / 16, 4)},
        "ratio_d16_1_over_log_vs_1_over_d": round(
            (1 / math.log(16)) / (1 / 16), 4),  # ~6x gap
    }

    logger.info(f"ε-sensitivity: {len(eps_sensitivity)} cells, "
                f"ε(d) ratio_d16={eps_functional_forms['ratio_d16_1_over_log_vs_1_over_d']:.2f}")
    return {
        "eps_sensitivity": eps_sensitivity,
        "per_seed_eps": per_seed_eps,
        "eps_functional_forms": eps_functional_forms,
    }


# ── output builders ────────────────────────────────────────────────────────────

def build_output(iter2_res, iter1_res):
    """Build exp_eval_sol_out schema-compliant output."""
    t1 = iter2_res["table1"]
    records = iter2_res["records"]
    Ts = iter2_res["Ts"]
    k_trues = iter2_res["k_trues"]
    ds = iter2_res["ds"]
    seeds = iter2_res["seeds"]

    # ── metrics_agg (flat numeric) ────────────────────────────────────────────
    all_bw = [r["bw"] for r in records]
    all_bww = [r["bw_warm"] for r in records]
    all_kl = [r["kl_bw"] for r in records]
    global_delta1 = float(np.mean(np.array(all_bww) - np.array(all_bw))) * 100
    global_delta2 = float(np.mean(np.array(all_kl) - np.array(all_bww))) * 100
    global_delta_bw_kl = float(np.mean(np.array(all_kl) - np.array(all_bw))) * 100

    # Fraction of (k,d) cells where delta2 CI does NOT cross zero
    n_sig_delta2 = sum(1 for v in t1.values() if not v["delta2_ci_crosses_zero"])

    metrics_agg = {
        "n_iter2_examples": len(records),
        "n_iter1_examples": len(iter1_res["eps_sensitivity"]),
        "global_mean_bw": round(float(np.mean(all_bw)), 4),
        "global_mean_bw_warm": round(float(np.mean(all_bww)), 4),
        "global_mean_kl_bw": round(float(np.mean(all_kl)), 4),
        "global_delta1_pp": round(global_delta1, 4),
        "global_delta2_pp": round(global_delta2, 4),
        "global_delta_bw_to_kl_pp": round(global_delta_bw_kl, 4),
        "n_table1_cells": len(t1),
        "n_cells_delta2_ci_sig": n_sig_delta2,
        "frac_cells_delta2_ci_sig": round(n_sig_delta2 / len(t1), 4) if t1 else 0.0,
    }

    # Add per-cell aggregate metrics
    for key, cell in t1.items():
        metrics_agg[f"delta1_pp_{key}"] = cell["delta1_bw_warm_vs_bw"]
        metrics_agg[f"delta2_pp_{key}"] = cell["delta2_kl_bw_vs_bw_warm"]
        metrics_agg[f"mean_kl_bw_{key}"] = cell["mean_kl_bw"]

    # Add global bias (mean error) per method
    for method_key, method_col in [("bw", "bw"), ("bw_warm", "bw_warm"), ("kl_bw", "kl_bw")]:
        metrics_agg[f"global_mean_err_{method_key}"] = round(
            float(np.mean([1 - r[method_col] for r in records])), 4)

    # ── datasets / examples ───────────────────────────────────────────────────
    # Primary examples: per (k_true, d, T) aggregated across seeds
    primary_examples = []
    out_res = iter2_res["outlier_results"]
    ca16 = iter2_res["constraint_activation_d16"]

    for k in k_trues:
        for d in ds:
            for T in Ts:
                key = f"k{k}_d{d}_T{T}"
                cell_recs = [r for r in records
                             if r["k_true"] == k and r["d"] == d and r["T"] == T]
                if not cell_recs:
                    continue
                bw_arr = np.array([r["bw"] for r in cell_recs])
                bww_arr = np.array([r["bw_warm"] for r in cell_recs])
                kl_arr = np.array([r["kl_bw"] for r in cell_recs])
                car_arr = np.array([r["car"] for r in cell_recs])

                delta1 = float(np.mean(bww_arr - bw_arr)) * 100
                delta2 = float(np.mean(kl_arr - bww_arr)) * 100

                out_info = out_res.get(key, {})
                car_key = f"k{k}_T{T}"
                ca_info = ca16.get(car_key, {}) if d == 16 else {}

                d16_d2 = iter2_res["d16_delta2_analysis"].get(f"k{k}_T{T}", {}) if d == 16 else {}

                ex = {
                    "input": f"k_true={k}, d={d}, T={T}",
                    "output": json.dumps({
                        "mean_bw": round(float(np.mean(bw_arr)), 4),
                        "mean_bw_warm": round(float(np.mean(bww_arr)), 4),
                        "mean_kl_bw": round(float(np.mean(kl_arr)), 4),
                        "delta1_pp": round(delta1, 4),
                        "delta2_pp": round(delta2, 4),
                    }),
                    "predict_bw_mean": str(round(float(np.mean(bw_arr)), 4)),
                    "predict_bw_warm_mean": str(round(float(np.mean(bww_arr)), 4)),
                    "predict_kl_bw_mean": str(round(float(np.mean(kl_arr)), 4)),
                    "eval_mean_bw": round(float(np.mean(bw_arr)), 4),
                    "eval_mean_bw_warm": round(float(np.mean(bww_arr)), 4),
                    "eval_mean_kl_bw": round(float(np.mean(kl_arr)), 4),
                    "eval_delta1_pp": round(delta1, 4),
                    "eval_delta2_pp": round(delta2, 4),
                    "eval_mean_car": round(float(np.mean(car_arr)), 6),
                    "eval_n_bw_outlier_seeds": len(out_info.get("bw_outlier_seeds", [])),
                    "eval_n_kl_bw_outlier_seeds": len(out_info.get("kl_bw_outlier_seeds", [])),
                    "metadata_k_true": k,
                    "metadata_d": d,
                    "metadata_T": T,
                    "metadata_n_seeds": len(cell_recs),
                }
                if d == 16:
                    ex["eval_n_activating_seeds_d16"] = ca_info.get("n_activating_seeds", 0)
                    ex["eval_delta2_no_outlier_pp"] = d16_d2.get("delta2_mean_without_outliers_pp") or 0.0
                primary_examples.append(ex)

    # Table 1 cell examples (per k_true, d)
    table1_examples = []
    for key, cell in t1.items():
        table1_examples.append({
            "input": f"Table1: k_true={cell['k_true']}, d={cell['d']}",
            "output": json.dumps({k: v for k, v in cell.items()
                                  if k not in ("k_true", "d")}),
            "predict_mean_bw": str(cell["mean_bw"]),
            "predict_mean_bw_warm": str(cell["mean_bw_warm"]),
            "predict_mean_kl_bw": str(cell["mean_kl_bw"]),
            "eval_delta1_pp": cell["delta1_bw_warm_vs_bw"],
            "eval_delta2_pp": cell["delta2_kl_bw_vs_bw_warm"],
            "eval_ci_delta1_lo": cell["ci_delta1_95"][0],
            "eval_ci_delta1_hi": cell["ci_delta1_95"][1],
            "eval_ci_delta2_lo": cell["ci_delta2_95"][0],
            "eval_ci_delta2_hi": cell["ci_delta2_95"][1],
            "eval_se_delta1": cell["se_delta1"],
            "eval_se_delta2": cell["se_delta2"],
            "eval_delta1_sig": float(not cell["delta1_ci_crosses_zero"]),
            "eval_delta2_sig": float(not cell["delta2_ci_crosses_zero"]),
            "metadata_k_true": cell["k_true"],
            "metadata_d": cell["d"],
        })

    # Bias-variance examples
    bv_examples = []
    for key, entry in iter2_res["bv_table"].items():
        r2 = entry.get("r_squared")
        bv_examples.append({
            "input": f"BV fit: k_true={entry['k_true']}, method={entry['method']}",
            "output": json.dumps({k: v for k, v in entry.items()
                                  if k not in ("k_true", "method")}),
            "predict_bias": str(entry.get("bias", "N/A")),
            "predict_c": str(entry.get("c", "N/A")),
            "eval_r_squared": r2 if isinstance(r2, float) else -1.0,
            "eval_reliable": float(isinstance(r2, float) and r2 >= 0.30),
            "metadata_k_true": entry["k_true"],
            "metadata_method": entry["method"],
        })

    # ε-sensitivity examples (iter_1, d=4)
    eps_examples = []
    for key, entry in iter1_res["eps_sensitivity"].items():
        eps_accs = {ev: entry[f"mean_acc_eps_{ev}"] for ev in [0.01, 0.05, 0.1, 0.5]}
        eps_examples.append({
            "input": f"eps_sensitivity: k_true={entry['k_true']}, T={entry['T']}, d=4",
            "output": json.dumps(eps_accs),
            "predict_best_eps": str(max(eps_accs, key=eps_accs.get)),
            "eval_max_min_spread_pp": entry["max_min_spread_pp"],
            "eval_claim_klbw_reduces_to_bwwarm": float(entry["max_min_spread_pp"] < 0.5),
            "metadata_k_true": entry["k_true"],
            "metadata_T": entry["T"],
            "metadata_d": 4,
        })

    datasets = [
        {"dataset": "synthetic_hmm_iter2_per_cell",
         "examples": primary_examples},
        {"dataset": "synthetic_hmm_iter2_table1",
         "examples": table1_examples},
        {"dataset": "synthetic_hmm_iter2_bias_variance",
         "examples": bv_examples},
        {"dataset": "synthetic_hmm_iter1_eps_sensitivity",
         "examples": eps_examples},
    ]

    # ── Supplementary analyses as metadata ───────────────────────────────────
    metadata = {
        "evaluation_name": "KL-Constrained Baum-Welch Symmetric Outlier & Statistical Rigor Evaluation",
        "symmetric_outlier_comparison": iter2_res["symmetric_outlier_comparison"],
        "constraint_activation_d16": iter2_res["constraint_activation_d16"],
        "d16_delta2_analysis": iter2_res["d16_delta2_analysis"],
        "eps_functional_forms": iter1_res["eps_functional_forms"],
        "per_seed_eps_trajectories": iter1_res["per_seed_eps"],
        "table1_full": iter2_res["table1"],
        "bv_table_full": iter2_res["bv_table"],
        "eps_sensitivity_full": iter1_res["eps_sensitivity"],
        "data_dimensions": {
            "iter2": {"k_trues": k_trues, "ds": ds, "Ts": Ts, "seeds": seeds},
            "iter1_note": "d=4 only, eps in {0.01,0.05,0.1,0.5}",
        },
        "eps_d16_data_note": (
            "ε-sensitivity at d=16 cannot be computed from available data: "
            "iter_2 uses fixed ε=0.01; iter_1 has ε-sweep but d=4 only."
        ),
    }

    return {"metadata": metadata, "metrics_agg": metrics_agg, "datasets": datasets}


@logger.catch
def main(max_examples: int = None):
    logger.info("=== KL-BW Evaluation start ===")

    # Load data
    iter2_exs = load_iter2()
    iter1_exs = load_iter1()

    if max_examples is not None:
        iter2_exs = iter2_exs[:max_examples]
        iter1_exs = iter1_exs[:max_examples]
        logger.info(f"Limited to {max_examples} examples each (dev mode)")

    logger.info("Running iter_2 analysis...")
    iter2_res = analyse_iter2(iter2_exs)

    logger.info("Running iter_1 analysis...")
    iter1_res = analyse_iter1(iter1_exs)

    logger.info("Building output...")
    output = build_output(iter2_res, iter1_res)

    out_path = BASE / "eval_final_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved output to {out_path}")

    # Print summary
    m = output["metrics_agg"]
    logger.info(f"Global BW={m['global_mean_bw']:.4f} "
                f"BW-warm={m['global_mean_bw_warm']:.4f} "
                f"KL-BW={m['global_mean_kl_bw']:.4f}")
    logger.info(f"Δ1={m['global_delta1_pp']:.4f}pp "
                f"Δ2={m['global_delta2_pp']:.4f}pp "
                f"(BW→KL: {m['global_delta_bw_to_kl_pp']:.4f}pp)")
    logger.info(f"Cells with significant Δ2: {m['n_cells_delta2_ci_sig']}/{m['n_table1_cells']}")
    logger.info("=== Done ===")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()
    main(max_examples=args.max_examples)
