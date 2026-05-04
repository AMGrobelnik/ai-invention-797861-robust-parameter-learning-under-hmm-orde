#!/usr/bin/env python3
"""HMM Order-Misspecification Experiment: KL-Constrained Baum-Welch vs Standard EM.

Generates 315 main configs (5 seeds × 3 k_true × 3 d × 7 T) plus
105 ε-sensitivity configs (d=16 only × 3 extra ε values).
"""

import json
import math
import os
import resource
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import psutil
from loguru import logger
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
_fmt = f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}"
logger.add(sys.stdout, level="INFO", format=_fmt)
WS = Path(__file__).parent
(WS / "logs").mkdir(exist_ok=True)
logger.add(str(WS / "logs" / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware / resource limits
# ---------------------------------------------------------------------------
def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


NUM_CPUS = _detect_cpus()
NUM_WORKERS = max(1, NUM_CPUS - 1)

_avail = psutil.virtual_memory().available
RAM_BUDGET = int(_avail * 0.70)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
SEEDS = list(range(5))
EXTRA_ITER = 30
K_TRUE_VALUES = [1, 2, 3]
D_VALUES = [4, 8, 16]
T_VALUES = [50, 100, 250, 500, 1000, 2000, 5000]
EPSILON_MAIN = 0.01
EPSILON_SENSITIVITY = [0.06, 0.36, 0.5]
N_RESTARTS = 3
N_BW_ITERS = 50
TOL_LAMBDA = 1e-8   # threshold: alpha < 1 - TOL_LAMBDA → constraint active
LOG_ZERO = -1e300


# ---------------------------------------------------------------------------
# HMM utilities (log-space)
# ---------------------------------------------------------------------------

def _safe_log(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.where(x > 0, np.log(x), LOG_ZERO)


def _log_forward(log_pi: np.ndarray, log_A: np.ndarray,
                 log_B: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Log-space forward pass. Returns log_alpha (T, d)."""
    T = len(obs)
    d = log_A.shape[0]
    log_alpha = np.full((T, d), LOG_ZERO)
    log_alpha[0] = log_pi + log_B[:, obs[0]]
    for t in range(1, T):
        # log sum_i exp(log_alpha[t-1,i] + log_A[i,j])  for each j
        # vals[i,j] = log_alpha[t-1,i] + log_A[i,j]
        vals = log_alpha[t - 1, :, None] + log_A   # (d, d)
        max_v = vals.max(axis=0)                    # (d,)
        log_alpha[t] = (
            max_v
            + np.log(np.exp(vals - max_v).sum(axis=0))
            + log_B[:, obs[t]]
        )
    return log_alpha


def _log_backward(log_A: np.ndarray, log_B: np.ndarray,
                  obs: np.ndarray) -> np.ndarray:
    """Log-space backward pass. Returns log_beta (T, d)."""
    T = len(obs)
    d = log_A.shape[0]
    log_beta = np.zeros((T, d))          # log(1) = 0 at T-1
    for t in range(T - 2, -1, -1):
        # log_beta[t,i] = log sum_j exp(log_A[i,j] + log_B[j,obs[t+1]] + log_beta[t+1,j])
        # vals[i,j] = log_A[i,j] + log_B[j, obs[t+1]] + log_beta[t+1,j]
        vals = log_A + log_B[:, obs[t + 1]][None, :] + log_beta[t + 1][None, :]  # (d,d)
        max_v = vals.max(axis=1)                                                   # (d,)
        log_beta[t] = max_v + np.log(np.exp(vals - max_v[:, None]).sum(axis=1))
    return log_beta


def _log_likelihood(log_alpha: np.ndarray) -> float:
    """log P(O | model) from the last row of log_alpha."""
    la = log_alpha[-1]
    max_v = la.max()
    return float(max_v + np.log(np.exp(la - max_v).sum()))


def _e_step(log_pi: np.ndarray, log_A: np.ndarray,
            log_B: np.ndarray, obs: np.ndarray):
    """E-step: compute gamma (T,d), xi (T-1,d,d), and log-likelihood."""
    T = len(obs)
    log_alpha = _log_forward(log_pi, log_A, log_B, obs)
    log_beta  = _log_backward(log_A, log_B, obs)
    log_like  = _log_likelihood(log_alpha)

    # gamma[t,i] = P(q_t=i | O)
    log_gamma = log_alpha + log_beta               # (T, d)
    max_g = log_gamma.max(axis=1, keepdims=True)
    gamma_un = np.exp(log_gamma - max_g)
    gamma = gamma_un / gamma_un.sum(axis=1, keepdims=True)

    # xi[t,i,j] = P(q_t=i, q_{t+1}=j | O)
    # log_B_next[t,j] = log_B[j, obs[t+1]]  shape (T-1, d)
    log_B_next = log_B[:, obs[1:]].T          # (T-1, d)
    # log_xi[t,i,j] = log_alpha[t,i] + log_A[i,j] + log_B_next[t,j] + log_beta[t+1,j]
    log_xi = (
        log_alpha[:-1, :, None]            # (T-1, d, 1)
        + log_A[None, :, :]                # (1,   d, d)
        + log_B_next[:, None, :]           # (T-1, 1, d)
        + log_beta[1:, None, :]            # (T-1, 1, d)
    )                                      # → (T-1, d, d)
    max_xi = log_xi.reshape(T - 1, -1).max(axis=1)[:, None, None]
    xi_un = np.exp(log_xi - max_xi)
    xi = xi_un / xi_un.reshape(T - 1, -1).sum(axis=1)[:, None, None]

    return gamma, xi, log_like


def _m_step_unconstrained(gamma: np.ndarray, xi: np.ndarray,
                          obs: np.ndarray, obs_dim: int):
    """Unconstrained M-step → (log_pi, log_A, log_B)."""
    d = gamma.shape[1]
    T = len(obs)

    # pi
    log_pi = _safe_log(gamma[0])
    lse = log_pi.max() + np.log(np.exp(log_pi - log_pi.max()).sum())
    log_pi -= lse

    # A
    A_counts = xi.sum(axis=0)                          # (d, d)
    row_sums = A_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    log_A = _safe_log(A_counts / row_sums)

    # B  — vectorised scatter-add
    B_counts = np.zeros((d, obs_dim))
    np.add.at(B_counts.T, obs, gamma)                  # B_counts[state, sym] += gamma[t, state]
    row_sums_b = B_counts.sum(axis=1, keepdims=True)
    row_sums_b = np.where(row_sums_b == 0, 1.0, row_sums_b)
    log_B = _safe_log(B_counts / row_sums_b)

    return log_pi, log_A, log_B


# ---------------------------------------------------------------------------
# KL constraint helpers
# ---------------------------------------------------------------------------

def _kl_rows(P: np.ndarray, P0: np.ndarray) -> np.ndarray:
    """Per-row KL(P[i,:] ∥ P0[i,:]), shape (n_rows,)."""
    ratio = np.where((P > 1e-300) & (P0 > 1e-300), np.log(P / P0), 0.0)
    return (P * ratio).sum(axis=1)


def _geometric_mix(P_unc: np.ndarray, P0: np.ndarray, alpha: float) -> np.ndarray:
    """Row-wise geometric interpolation P ∝ P_unc^α · P0^(1-α), renormalised."""
    P_unc_c = np.clip(P_unc, 1e-300, 1.0)
    P0_c    = np.clip(P0,    1e-300, 1.0)
    log_P = alpha * np.log(P_unc_c) + (1.0 - alpha) * np.log(P0_c)
    log_P -= log_P.max(axis=1, keepdims=True)
    P = np.exp(log_P)
    P /= P.sum(axis=1, keepdims=True)
    return P


def _find_alpha_for_kl(P_unc: np.ndarray, P0: np.ndarray,
                       epsilon: float) -> tuple[float, float]:
    """Binary-search α ∈ [0,1] so mean KL(P_mix(α) ∥ P0) ≤ ε.

    Returns (alpha, achieved_mean_kl).
    """
    P_unc_c = np.clip(P_unc, 1e-300, 1.0)
    P0_c    = np.clip(P0,    1e-300, 1.0)

    # If unconstrained already satisfies → no projection needed
    kl_unc = _kl_rows(P_unc_c, P0_c).mean()
    if kl_unc <= epsilon:
        return 1.0, float(kl_unc)

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) * 0.5
        P_mix = _geometric_mix(P_unc_c, P0_c, mid)
        kl = _kl_rows(P_mix, P0_c).mean()
        if kl <= epsilon:
            lo = mid
        else:
            hi = mid

    alpha = (lo + hi) * 0.5
    P_mix = _geometric_mix(P_unc_c, P0_c, alpha)
    achieved = float(_kl_rows(P_mix, P0_c).mean())
    return alpha, achieved


# ---------------------------------------------------------------------------
# Viterbi
# ---------------------------------------------------------------------------

def _viterbi(log_pi: np.ndarray, log_A: np.ndarray,
             log_B: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Viterbi decoding → state sequence (T,)."""
    T = len(obs)
    d = log_A.shape[0]
    delta = np.full((T, d), LOG_ZERO)
    psi   = np.zeros((T, d), dtype=int)
    delta[0] = log_pi + log_B[:, obs[0]]
    for t in range(1, T):
        scores = delta[t - 1, :, None] + log_A    # (d, d)
        psi[t] = scores.argmax(axis=0)
        delta[t] = scores.max(axis=0) + log_B[:, obs[t]]
    path = np.zeros(T, dtype=int)
    path[T - 1] = delta[T - 1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def _viterbi_accuracy(q_true: np.ndarray, q_pred: np.ndarray, d: int) -> float:
    """Best-permutation accuracy via Hungarian matching (handles label swap)."""
    C = np.zeros((d, d), dtype=int)
    for ts, ps in zip(q_true, q_pred):
        C[ts, ps] += 1
    row_ind, col_ind = linear_sum_assignment(-C)
    return float(C[row_ind, col_ind].sum()) / len(q_true)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _make_stochastic(rng: np.random.Generator, rows: int, cols: int,
                     diag_boost: float = 0.35) -> np.ndarray:
    """Random row-stochastic matrix; diagonal boost when square for spectral gap."""
    alpha = np.ones(cols) * 0.5
    M = rng.dirichlet(alpha, size=rows)
    if rows == cols:
        M = (1.0 - diag_boost) * M + diag_boost * np.eye(rows)
        M /= M.sum(axis=1, keepdims=True)
    return M


def generate_hmm_data(seed: int, k_true: int, d: int, T: int):
    """Simulate a k_true-order HMM.

    Transitions depend on the k_true most-recent states (context flattened to
    an integer index into A_true rows).  Emissions depend only on q_t.

    Returns (A_true, B_true, obs, q_true).
    """
    rng = np.random.default_rng(seed)
    n_ctx = d ** k_true
    A_true = _make_stochastic(rng, n_ctx, d)         # (d^k, d)
    B_true = _make_stochastic(rng, d, d)             # (d,  d)

    q_true = np.zeros(T, dtype=int)
    # seed first k_true states
    for i in range(k_true):
        q_true[i] = int(rng.integers(0, d))

    for t in range(k_true, T):
        ctx_idx = 0
        for lag in range(k_true):
            ctx_idx = ctx_idx * d + int(q_true[t - k_true + lag])
        q_true[t] = int(rng.choice(d, p=A_true[ctx_idx]))

    obs = np.array(
        [int(rng.choice(d, p=B_true[q_true[t]])) for t in range(T)],
        dtype=int,
    )
    return A_true, B_true, obs, q_true


# ---------------------------------------------------------------------------
# Fitting methods
# ---------------------------------------------------------------------------

def _random_init(rng: np.random.Generator, d: int, obs_dim: int):
    """Random log-parameter initialisation for a first-order HMM."""
    log_pi = _safe_log(rng.dirichlet(np.ones(d)))
    log_A  = _safe_log(rng.dirichlet(np.ones(d), size=d))
    log_B  = _safe_log(rng.dirichlet(np.ones(obs_dim), size=d))
    return log_pi, log_A, log_B


def fit_standard_bw(obs: np.ndarray, d: int, seed: int,
                    n_restarts: int = 3, n_iters: int = 50):
    """Standard Baum-Welch with multiple random restarts.

    Returns (best_model, best_log_like) where best_model = (log_pi, log_A, log_B).
    """
    obs_dim = d
    rng = np.random.default_rng(seed * 1000 + 37)
    best_ll = -np.inf
    best_model = None

    for _ in range(n_restarts):
        log_pi, log_A, log_B = _random_init(rng, d, obs_dim)
        ll_prev = -np.inf
        for _ in range(n_iters):
            try:
                gamma, xi, ll = _e_step(log_pi, log_A, log_B, obs)
                log_pi, log_A, log_B = _m_step_unconstrained(gamma, xi, obs, obs_dim)
                ll_prev = ll
            except Exception:
                break
        # Final likelihood evaluation
        try:
            _, _, ll_final = _e_step(log_pi, log_A, log_B, obs)
        except Exception:
            ll_final = ll_prev

        if best_model is None or ll_final > best_ll:
            best_ll = ll_final
            best_model = (log_pi.copy(), log_A.copy(), log_B.copy())

    # Fallback: should never be None after loop, but guard anyway
    if best_model is None:
        rng2 = np.random.default_rng(seed)
        best_model = _random_init(rng2, d, obs_dim)
        best_ll = -np.inf

    return best_model, best_ll


def fit_warm_bw(obs: np.ndarray, d: int, init_model: tuple,
                extra_iters: int = 30):
    """Warm-start BW: continue unconstrained EM from init_model."""
    log_pi, log_A, log_B = (x.copy() for x in init_model)
    obs_dim = d
    for _ in range(extra_iters):
        try:
            gamma, xi, _ = _e_step(log_pi, log_A, log_B, obs)
            log_pi, log_A, log_B = _m_step_unconstrained(gamma, xi, obs, obs_dim)
        except Exception:
            break
    return log_pi, log_A, log_B


def fit_kl_bw(obs: np.ndarray, d: int, init_model: tuple,
              epsilon: float, extra_iters: int = 30):
    """KL-constrained Baum-Welch.

    At each M-step the updated A and B are geometrically projected onto
    KL(· ∥ A_ref) ≤ ε/2 and KL(· ∥ B_ref) ≤ ε/2 respectively.
    Constraint is considered 'active' when the unconstrained update
    violates the budget (α < 1 − TOL_LAMBDA after projection).

    Returns (log_pi, log_A, log_B, constraint_activation_rate).
    """
    log_pi_ref, log_A_ref, log_B_ref = (x.copy() for x in init_model)
    log_pi, log_A, log_B = log_pi_ref.copy(), log_A_ref.copy(), log_B_ref.copy()
    obs_dim = d

    A_ref = np.exp(log_A_ref)
    B_ref = np.exp(log_B_ref)

    n_activated = 0
    eps_half = epsilon / 2.0

    for _ in range(extra_iters):
        try:
            gamma, xi, _ = _e_step(log_pi, log_A, log_B, obs)
            log_pi_unc, log_A_unc, log_B_unc = _m_step_unconstrained(
                gamma, xi, obs, obs_dim
            )
        except Exception:
            break

        A_unc = np.exp(log_A_unc)
        B_unc = np.exp(log_B_unc)

        alpha_A, _ = _find_alpha_for_kl(A_unc, A_ref, eps_half)
        alpha_B, _ = _find_alpha_for_kl(B_unc, B_ref, eps_half)

        # Constraint is active if either matrix needed to be projected
        if (alpha_A < 1.0 - TOL_LAMBDA) or (alpha_B < 1.0 - TOL_LAMBDA):
            n_activated += 1

        A_new = _geometric_mix(A_unc, A_ref, alpha_A)
        B_new = _geometric_mix(B_unc, B_ref, alpha_B)
        log_A = _safe_log(A_new)
        log_B = _safe_log(B_new)
        log_pi = log_pi_unc   # pi unconstrained (negligible effect)

    car = n_activated / extra_iters if extra_iters > 0 else 0.0
    return log_pi, log_A, log_B, car


# ---------------------------------------------------------------------------
# Single-config runner (executed in worker process)
# ---------------------------------------------------------------------------

def run_config(args: tuple) -> list[dict]:
    """Run all methods for one (seed, k_true, d, T) config."""
    seed, k_true, d, T = args
    results: list[dict] = []

    try:
        _, _, obs, q_true = generate_hmm_data(seed, k_true, d, T)

        # ── Method 1: Standard BW ──────────────────────────────────────────
        best_model, _ = fit_standard_bw(
            obs, d, seed, n_restarts=N_RESTARTS, n_iters=N_BW_ITERS
        )
        q_bw = _viterbi(*best_model, obs)
        results.append({
            "seed": seed, "k_true": k_true, "d": d, "T": T,
            "method": "BW", "epsilon": None,
            "viterbi_acc": _viterbi_accuracy(q_true, q_bw, d),
            "constraint_activation_rate": 0.0,
        })

        # ── Method 2: Warm-start BW ────────────────────────────────────────
        log_pi_w, log_A_w, log_B_w = fit_warm_bw(
            obs, d, best_model, extra_iters=EXTRA_ITER
        )
        q_w = _viterbi(log_pi_w, log_A_w, log_B_w, obs)
        results.append({
            "seed": seed, "k_true": k_true, "d": d, "T": T,
            "method": "BW-warm", "epsilon": None,
            "viterbi_acc": _viterbi_accuracy(q_true, q_w, d),
            "constraint_activation_rate": 0.0,
        })

        # ── Method 3: KL-BW (main ε) ──────────────────────────────────────
        log_pi_k, log_A_k, log_B_k, car = fit_kl_bw(
            obs, d, best_model, epsilon=EPSILON_MAIN, extra_iters=EXTRA_ITER
        )
        q_k = _viterbi(log_pi_k, log_A_k, log_B_k, obs)
        results.append({
            "seed": seed, "k_true": k_true, "d": d, "T": T,
            "method": "KL-BW", "epsilon": EPSILON_MAIN,
            "viterbi_acc": _viterbi_accuracy(q_true, q_k, d),
            "constraint_activation_rate": car,
        })

        # ── Method 4: KL-BW ε-sensitivity (d=16 only) ─────────────────────
        if d == 16:
            for eps in EPSILON_SENSITIVITY:
                log_pi_s, log_A_s, log_B_s, car_s = fit_kl_bw(
                    obs, d, best_model, epsilon=eps, extra_iters=EXTRA_ITER
                )
                q_s = _viterbi(log_pi_s, log_A_s, log_B_s, obs)
                results.append({
                    "seed": seed, "k_true": k_true, "d": d, "T": T,
                    "method": "KL-BW-sensitivity", "epsilon": eps,
                    "viterbi_acc": _viterbi_accuracy(q_true, q_s, d),
                    "constraint_activation_rate": car_s,
                })

    except Exception as exc:
        logger.error(f"Config ({seed},{k_true},{d},{T}) failed: {exc}")

    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_results(all_results: list[dict]) -> bool:
    ok = True

    main_keys = {
        (r["seed"], r["k_true"], r["d"], r["T"], r["method"])
        for r in all_results
        if r["method"] in ("BW", "BW-warm", "KL-BW")
    }
    expected_main = len(SEEDS) * len(K_TRUE_VALUES) * len(D_VALUES) * len(T_VALUES) * 3
    logger.info(f"Main configs: {len(main_keys)}/{expected_main}")
    if len(main_keys) < expected_main:
        logger.warning("Some main configs missing!")
        ok = False

    sens = [r for r in all_results if r["method"] == "KL-BW-sensitivity"]
    expected_sens = len(SEEDS) * len(K_TRUE_VALUES) * len(T_VALUES) * len(EPSILON_SENSITIVITY)
    logger.info(f"Sensitivity configs: {len(sens)}/{expected_sens}")

    for r in all_results:
        if not (0.0 <= r["viterbi_acc"] <= 1.0):
            logger.error(f"Invalid viterbi_acc: {r}")
            ok = False
        if not (0.0 <= r["constraint_activation_rate"] <= 1.0):
            logger.error(f"Invalid CAR: {r}")
            ok = False
        if r["method"] in ("BW", "BW-warm") and r["constraint_activation_rate"] != 0.0:
            logger.error(f"BW/BW-warm has non-zero CAR: {r}")
            ok = False

    logger.info(f"Validation {'PASSED' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# Output formatting (exp_gen_sol_out schema)
# ---------------------------------------------------------------------------

def results_to_method_out(all_results: list[dict]) -> dict:
    """Convert flat results list → exp_gen_sol_out schema."""
    configs: dict = defaultdict(dict)

    for r in all_results:
        key = (r["seed"], r["k_true"], r["d"], r["T"])
        method = r["method"]
        eps = r["epsilon"]

        payload = json.dumps({
            "viterbi_acc": r["viterbi_acc"],
            "constraint_activation_rate": r["constraint_activation_rate"],
            "epsilon": eps,
        })

        if method == "BW":
            configs[key]["predict_BW"] = payload
        elif method == "BW-warm":
            configs[key]["predict_BW_warm"] = payload
        elif method == "KL-BW":
            configs[key]["predict_KL_BW"] = payload
        elif method == "KL-BW-sensitivity":
            eps_key = str(eps).replace(".", "_")
            configs[key][f"predict_KL_BW_eps_{eps_key}"] = payload

    examples = []
    for (seed, k_true, d, T), preds in sorted(configs.items()):
        ex = {
            "input": json.dumps({
                "seed": seed, "k_true": k_true, "d": d, "T": T,
                "task": (
                    "HMM order misspecification: fit first-order model "
                    f"to k_true={k_true}-order process with d={d} states, T={T} obs"
                ),
            }),
            "output": json.dumps({
                "description": (
                    f"Viterbi accuracy comparing BW variants on "
                    f"k_true={k_true}, d={d}, T={T}, seed={seed}"
                )
            }),
        }
        ex.update(preds)
        examples.append(ex)

    return {
        "metadata": {
            "method_name": "KL-Constrained Baum-Welch",
            "description": (
                "HMM order-misspecification experiment comparing standard BW, "
                "warm-start BW, and KL-constrained BW"
            ),
            "seeds_used": SEEDS,
            "extra_iter": EXTRA_ITER,
            "epsilon_main": EPSILON_MAIN,
            "epsilon_sensitivity_grid": EPSILON_SENSITIVITY,
            "grid_params": {
                "k_true_values": K_TRUE_VALUES,
                "d_values": D_VALUES,
                "T_values": T_VALUES,
            },
        },
        "datasets": [
            {
                "dataset": "hmm_order_misspecification",
                "examples": examples,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@logger.catch
def main():
    logger.info(f"Workspace: {WS}")
    logger.info(f"CPUs detected: {NUM_CPUS}, workers: {NUM_WORKERS}")
    logger.info(f"RAM budget: {RAM_BUDGET / 1e9:.1f} GB")

    configs = [
        (seed, k_true, d, T)
        for seed in SEEDS
        for k_true in K_TRUE_VALUES
        for d in D_VALUES
        for T in T_VALUES
    ]
    total = len(configs)
    logger.info(f"Total configs: {total}  (+ sensitivity grid for d=16)")

    all_results: list[dict] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(run_config, cfg): cfg for cfg in configs}
        for future in as_completed(futures):
            cfg = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as exc:
                logger.error(f"Future for {cfg} raised: {exc}")
            completed += 1
            if completed % 25 == 0 or completed == total:
                logger.info(f"Progress: {completed}/{total} configs done, "
                            f"results so far: {len(all_results)}")

    logger.info(f"Total results collected: {len(all_results)}")
    validate_results(all_results)

    # ── Save full_method_out.json (artifact-plan format) ──────────────────
    full_out = {
        "seeds_used": SEEDS,
        "extra_iter": EXTRA_ITER,
        "epsilon_main": EPSILON_MAIN,
        "epsilon_sensitivity_grid": EPSILON_SENSITIVITY,
        "grid_params": {
            "k_true_values": K_TRUE_VALUES,
            "d_values": D_VALUES,
            "T_values": T_VALUES,
        },
        "results": all_results,
    }
    full_path = WS / "full_method_out.json"
    full_path.write_text(json.dumps(full_out, indent=2))
    logger.info(f"Saved full_method_out.json  ({full_path.stat().st_size / 1e3:.1f} KB)")

    # ── Save method_out.json (exp_gen_sol_out schema) ─────────────────────
    method_out = results_to_method_out(all_results)
    method_path = WS / "method_out.json"
    method_path.write_text(json.dumps(method_out, indent=2))
    logger.info(f"Saved method_out.json  ({method_path.stat().st_size / 1e3:.1f} KB)")

    logger.info("Done.")


if __name__ == "__main__":
    main()
