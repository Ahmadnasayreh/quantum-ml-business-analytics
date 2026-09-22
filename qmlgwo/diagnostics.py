"""
qmlgwo.diagnostics
==================
Trainability diagnostics: gradient-variance scaling with qubit count and depth
(barren plateaus), and a controlled comparison of circuit-parameter optimisers.

What is measured
----------------
1. :func:`gradient_variance_scan` -- ``Var[d<Z_0>/d theta_1]`` over random
   parameter initialisations as a function of qubit count and circuit depth,
   using **parameter-shift** gradients (the hardware-faithful estimator).
   McClean et al. (2018) predict ``Var ~ exp(-alpha n)``; :func:`fit_decay`
   fits that model and returns the decay constant with its R-squared.

2. :func:`optimizer_trajectories` -- loss trajectories for Adam (parameter-shift),
   SPSA and COBYLA on an identical circuit, initialisation and budget.

An important scope caveat to state in the manuscript
----------------------------------------------------
GWO in this framework optimises **hyper-parameters** (circuit depth, encoding
repetitions, entanglement topology, learning rate), not the circuit parameters
themselves -- those are still trained by Adam/SPSA/COBYLA.  Therefore GWO cannot
"avoid barren plateaus" in the sense of McClean et al.  What it *can* do, and
what :func:`plateau_avoidance_evidence` quantifies, is select architectures whose
gradient variance is measurably larger than a random or default choice.  That is
a narrower and defensible claim, and it is the one the paper should make.

References
----------
McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., Neven, H. (2018).
*Barren plateaus in quantum neural network training landscapes.*
Nature Communications 9, 4812. https://doi.org/10.1038/s41467-018-07090-4

Cerezo, M., Sone, A., Volkoff, T., Cincio, L., Coles, P. J. (2021).
*Cost function dependent barren plateaus in shallow parametrized quantum
circuits.* Nature Communications 12, 1791.
https://doi.org/10.1038/s41467-021-21728-w
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from scipy import stats as sstats

from .config import get_rng
from .quantum import FEATURE_MAPS, ansatz_shape

# --------------------------------------------------------------------------- #
# 1. Gradient variance scan                                                    #
# --------------------------------------------------------------------------- #


def _grad_sample(n_qubits: int, n_layers: int, rng, feature_map: str = "zz",
                 diff_method: str = "parameter-shift") -> tuple[float, float]:
    """One draw of the gradient at a random parameter setting.

    Returns ``(partial_k, ||grad||^2 / P)`` where ``k`` is a **uniformly random**
    parameter index.

    Why not always index 0?  ``StronglyEntanglingLayers`` applies
    ``Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)``, so parameter 0 is
    an ``RZ`` on wire 0.  At depth 1 that gate commutes with the measured
    ``Z_0``, making ``d<Z_0>/d phi_0`` identically zero by symmetry -- a
    structural zero, not a barren plateau.  Reporting it would fabricate
    "vanishing gradients" that have nothing to do with expressivity.  Sampling
    the index uniformly gives an unbiased picture of the landscape.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    fmap = FEATURE_MAPS[feature_map]

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(weights, x):
        fmap(x, n_qubits, reps=1, entanglement="linear")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    shape = ansatz_shape(n_layers, n_qubits)
    # Haar-like initialisation over the full rotation range: this is the regime
    # in which the barren-plateau result applies.
    w = pnp.array(rng.uniform(0, 2 * np.pi, shape), requires_grad=True)
    x = pnp.array(rng.uniform(-np.pi / 2, np.pi / 2, n_qubits), requires_grad=False)
    # NOTE: no argnum/argnums keyword. PennyLane renamed it (argnum -> argnums)
    # in v0.42; omitting it differentiates w.r.t. the trainable argument(s),
    # which here is only `w` (x carries requires_grad=False). Version-agnostic.
    g = np.asarray(qml.grad(circuit)(w, x)).flatten()
    k = int(rng.integers(0, g.size))
    return float(g[k]), float(np.mean(g ** 2))


def gradient_variance_scan(qubit_range: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
                           depth_range: Sequence[int] = (1, 2, 4, 6),
                           n_samples: int = 120, feature_map: str = "zz",
                           diff_method: str = "parameter-shift",
                           verbose: bool = True) -> pd.DataFrame:
    """Scan ``Var[partial <Z_0>]`` over qubit count and ansatz depth.

    Complexity: ``O(|qubits| * |depths| * n_samples * P * 2^n)`` where the factor
    ``P`` comes from parameter-shift needing two circuit evaluations per
    parameter.  Keep ``n_samples`` around 100-200 and qubits <= 10.
    """
    rows = []
    for n_q in qubit_range:
        for L in depth_range:
            rng = get_rng("barren", n_q * 100 + L)
            draws = [_grad_sample(n_q, L, rng, feature_map, diff_method)
                     for _ in range(n_samples)]
            g = np.array([d[0] for d in draws])          # random-index partial
            gsq = np.array([d[1] for d in draws])        # mean squared gradient
            rows.append({
                "n_qubits": n_q, "n_layers": L, "n_samples": n_samples,
                "grad_mean": g.mean(), "grad_var": g.var(ddof=1),
                "grad_std": g.std(ddof=1), "grad_abs_mean": np.abs(g).mean(),
                "mean_sq_grad": gsq.mean(),
                "n_parameters": int(np.prod(ansatz_shape(L, n_q))),
            })
            if verbose:
                print(f"  n_qubits={n_q} depth={L}: Var={rows[-1]['grad_var']:.3e}",
                      flush=True)
    return pd.DataFrame(rows)


def fit_decay(scan: pd.DataFrame, depth: int | None = None) -> pd.DataFrame:
    r"""Fit ``log Var = c - alpha * n`` (i.e. ``Var ~ e^{-alpha n}``) per depth.

    A significantly positive ``alpha`` with high ``R^2`` is the quantitative
    signature of a barren plateau.
    """
    rows = []
    depths = [depth] if depth is not None else sorted(scan["n_layers"].unique())
    for L in depths:
        sub = scan[scan["n_layers"] == L].sort_values("n_qubits")
        sub = sub[sub["grad_var"] > 0]
        if len(sub) < 3:
            continue
        lr = ssta_linreg(sub["n_qubits"].to_numpy(), np.log(sub["grad_var"].to_numpy()))
        rows.append({
            "n_layers": L, "decay_rate_alpha": -lr.slope,
            "intercept": lr.intercept, "r_squared": lr.rvalue ** 2,
            "p_value": lr.pvalue, "stderr": lr.stderr,
            "interpretation": ("exponential decay consistent with a barren plateau"
                               if (-lr.slope > 0 and lr.pvalue < 0.05 and lr.rvalue ** 2 > 0.8)
                               else "no significant exponential decay detected"),
        })
    return pd.DataFrame(rows)


def ssta_linreg(x, y):
    """Thin wrapper so the fit function stays readable."""
    return sstats.linregress(x, y)


# --------------------------------------------------------------------------- #
# 2. Optimizer trajectory comparison                                           #
# --------------------------------------------------------------------------- #


def optimizer_trajectories(X: np.ndarray, y: np.ndarray, n_layers: int = 3,
                           optimizers: Sequence[str] = ("adam", "spsa", "cobyla"),
                           n_epochs: int = 40, learning_rate: float = 0.1,
                           feature_map: str = "angle", seed: int = 0,
                           max_qubits: int = 6) -> pd.DataFrame:
    """Train one architecture with several circuit optimisers from a common start.

    Compares Adam, SPSA and COBYLA directly.
    All runs share data, architecture and initialisation seed, so the optimiser
    is the only varying factor.

    Comparability. The optimisers take different kinds of step: Adam and SPSA
    record one point per epoch (mean mini-batch loss), COBYLA one point per
    full-batch objective evaluation. Raw step counts are therefore NOT a common
    axis. Each trajectory carries a ``progress`` column in [0, 1] (fraction of
    that optimiser's own run), and every optimiser is additionally scored with
    the SAME metric after training -- the full-batch binary cross-entropy
    ``final_loss_full`` and training F1 -- which is the quantity to compare.
    """
    from sklearn.metrics import f1_score

    from .quantum import QuantumVQC

    rows = []
    for opt in optimizers:
        model = QuantumVQC(n_layers=n_layers, learning_rate=learning_rate,
                           n_epochs=n_epochs, optimizer=opt, feature_map=feature_map,
                           diff_method="backprop", max_qubits=max_qubits,
                           random_state=seed).fit(X, y)
        hist = model.loss_history_
        n = len(hist)
        for step, loss in enumerate(hist):
            rows.append({"optimizer": opt.upper(), "step": step,
                         "progress": step / max(n - 1, 1), "loss": loss})
        p1 = np.clip(model.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
        full = float(-np.mean(y * np.log(p1) + (1 - y) * np.log(1 - p1)))
        rows.append({"optimizer": opt.upper(), "step": n - 1, "progress": 1.0,
                     "loss": hist[-1], "final_loss_full": full,
                     "final_f1": f1_score(y, model.predict(X), zero_division=0),
                     "n_steps": n,
                     "step_unit": "objective evaluation" if opt == "cobyla" else "epoch",
                     "n_parameters": model.n_parameters_})
    return pd.DataFrame(rows)


def summarise_trajectories(traj: pd.DataFrame) -> pd.DataFrame:
    """One row per optimiser: comparable end-of-training metrics."""
    s = traj.dropna(subset=["final_loss_full"])
    return (s[["optimizer", "n_steps", "step_unit", "final_loss_full", "final_f1"]]
              .sort_values("final_loss_full").reset_index(drop=True))


# --------------------------------------------------------------------------- #
# 3. Linking GWO's architecture choices to gradient variance                   #
# --------------------------------------------------------------------------- #


def plateau_avoidance_evidence(fold_df: pd.DataFrame, scan: pd.DataFrame,
                               model_name: str = "VQC") -> pd.DataFrame:
    """Compare the gradient variance of GWO-selected depths against alternatives.

    For every outer fold this looks up the depth GWO chose, reads the measured
    ``grad_var`` at that depth, and contrasts it with the mean over the whole
    depth range.  A positive ``advantage_ratio`` is concrete evidence that the
    search prefers better-conditioned architectures -- the narrow, supportable
    version of the manuscript's claim.
    """
    sub = fold_df[(fold_df["model"] == model_name) & fold_df["best_params"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()

    import ast
    parsed = [ast.literal_eval(p) if isinstance(p, str) else p for p in sub["best_params"]]
    depths = [p.get("n_layers") for p in parsed if isinstance(p, dict)]
    depths = [d for d in depths if d is not None]
    if not depths:
        return pd.DataFrame()

    by_depth = scan.groupby("n_layers")["grad_var"].mean()
    overall = by_depth.mean()
    sel = np.mean([by_depth.get(d, np.nan) for d in depths])

    return pd.DataFrame([{
        "model": model_name,
        "n_folds": len(depths),
        "selected_depths": sorted(set(depths)),
        "modal_depth": int(pd.Series(depths).mode().iloc[0]),
        "mean_grad_var_at_selected_depth": sel,
        "mean_grad_var_over_search_range": overall,
        "advantage_ratio": sel / overall if overall > 0 else np.nan,
        "claim_supported": bool(sel > overall),
    }])


__all__ = ["gradient_variance_scan", "fit_decay", "optimizer_trajectories",
           "summarise_trajectories",
           "plateau_avoidance_evidence"]
