"""
qmlgwo.evaluate
===============
Model registry, hyper-parameter search spaces, and the nested repeated
stratified cross-validation engine.

Protocol
--------
::

    for each of  n_outer_repeats x n_outer_splits  outer folds:       <- reporting
        for each of n_inner_splits inner folds on the outer TRAIN:    <- model selection
            fit(inner-train) -> score(inner-validation)
        pick hyper-parameters maximising the mean inner-validation F1
        refit on the full outer-train, score once on the outer-test

The outer test fold is touched **exactly once per fold**, and only for
reporting.  Hyper-parameters never see it.

Every model is additionally evaluated on the *same folds* with default
hyper-parameters, which yields the paired samples required by the Wilcoxon
signed-rank test in :mod:`qmlgwo.stats`.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import ExperimentConfig, get_rng, get_seed
from .data import build_pipeline
from .optim import Param, SearchSpace, build_optimizer
from .quantum import (QuantumEmbeddingTree, QuantumKernelSVC,
                      QuantumReuploadingNN, QuantumVQC)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# 1. Metrics                                                                   #
# --------------------------------------------------------------------------- #


def compute_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    """Accuracy, Precision, Recall, F1, balanced accuracy, MCC and ROC-AUC.

    MCC and balanced accuracy are included because they are the metrics least
    distorted by residual class imbalance -- the manuscript currently leans on
    F1 alone.
    """
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            out["roc_auc"] = np.nan
    else:
        out["roc_auc"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# 2. Model registry and search spaces                                          #
# --------------------------------------------------------------------------- #


@dataclass
class ModelSpec:
    name: str
    family: str                      # 'classical' | 'quantum'
    factory: Callable[..., Any]
    space: SearchSpace | None = None
    defaults: dict[str, Any] = field(default_factory=dict)

    def make(self, params: dict[str, Any] | None = None, seed: int = 0):
        kwargs = dict(self.defaults)
        if params:
            kwargs.update({k: v for k, v in params.items() if not k.startswith("reduce__")})
        est = self.factory(**kwargs)
        if hasattr(est, "random_state"):
            try:
                est.set_params(random_state=seed)
            except Exception:
                pass
        return est


def build_registry(cfg: ExperimentConfig) -> dict[str, ModelSpec]:
    """All models with their GWO search spaces.

    The spaces below ARE the manuscript's Table "GWO search space"; call
    ``registry['VQC'].space.to_frame()`` to print it.
    """
    q_common = dict(n_epochs=cfg.n_epochs, batch_size=cfg.batch_size,
                    diff_method=cfg.diff_method, max_qubits=cfg.max_qubits)

    reg: dict[str, ModelSpec] = {
        # ---------------- classical baselines ---------------- #
        "LogReg": ModelSpec("LogReg", "classical", LogisticRegression,
            defaults=dict(max_iter=2000),
            space=SearchSpace([Param("C", "log", 1e-3, 1e2)])),
        "SVC": ModelSpec("SVC", "classical", SVC,
            defaults=dict(kernel="rbf", probability=True),
            space=SearchSpace([Param("C", "log", 1e-2, 1e2),
                               Param("gamma", "log", 1e-4, 1e1)])),
        "DecisionTree": ModelSpec("DecisionTree", "classical", DecisionTreeClassifier,
            space=SearchSpace([Param("max_depth", "int", 2, 16),
                               Param("min_samples_leaf", "int", 1, 20)])),
        "RandomForest": ModelSpec("RandomForest", "classical", RandomForestClassifier,
            defaults=dict(n_estimators=300, n_jobs=1),
            space=SearchSpace([Param("max_depth", "int", 2, 20),
                               Param("min_samples_leaf", "int", 1, 20),
                               Param("max_features", "cat", choices=["sqrt", "log2", None])])),
        "MLP": ModelSpec("MLP", "classical", MLPClassifier,
            defaults=dict(max_iter=600),
            space=SearchSpace([Param("alpha", "log", 1e-6, 1e-1),
                               Param("learning_rate_init", "log", 1e-4, 1e-1)])),

        # ---------------- quantum models ---------------- #
        "VQC": ModelSpec("VQC", "quantum", QuantumVQC,
            defaults=dict(n_layers=2, learning_rate=cfg.learning_rate,
                          feature_map="zz", map_reps=2, **q_common),
            space=SearchSpace([
                Param("n_layers", "int", 1, 6),                 # ansatz depth (reps)
                Param("learning_rate", "log", 1e-3, 5e-1),
                Param("map_reps", "int", 1, 3),                 # encoding repetitions
                Param("feature_map", "cat", choices=["zz", "angle"]),
                Param("entanglement", "cat", choices=["linear", "circular", "full"]),
                Param("init_scale", "float", 0.01, 0.5),
            ])),
        "QNN": ModelSpec("QNN", "quantum", QuantumReuploadingNN,
            defaults=dict(n_layers=3, learning_rate=cfg.learning_rate,
                          feature_map="angle", map_reps=1, **q_common),
            space=SearchSpace([
                Param("n_layers", "int", 1, 6),
                Param("learning_rate", "log", 1e-3, 5e-1),
                Param("feature_map", "cat", choices=["angle", "zz"]),
                Param("entanglement", "cat", choices=["linear", "circular"]),
                Param("init_scale", "float", 0.01, 0.5),
            ])),
        "QSVC": ModelSpec("QSVC", "quantum", QuantumKernelSVC,
            defaults=dict(C=1.0, map_reps=2, max_samples=cfg.max_kernel_samples,
                          max_qubits=cfg.max_qubits),
            space=SearchSpace([
                Param("C", "log", 1e-2, 1e2),
                Param("map_reps", "int", 1, 3),
                Param("scale", "float", 0.25, 2.0),             # encoding bandwidth
                Param("entanglement", "cat", choices=["linear", "circular", "full"]),
            ])),
        "QDT": ModelSpec("QDT", "quantum", QuantumEmbeddingTree,
            defaults=dict(max_depth=4, min_samples_split=8, map_reps=2,
                          max_qubits=cfg.max_qubits),
            space=SearchSpace([
                Param("max_depth", "int", 2, 12),
                Param("min_samples_split", "int", 2, 32),
                Param("map_reps", "int", 1, 3),
                Param("entanglement", "cat", choices=["linear", "circular", "full"]),
            ])),
    }

    try:                                            # optional dependency
        from xgboost import XGBClassifier
        reg["XGBoost"] = ModelSpec("XGBoost", "classical", XGBClassifier,
            defaults=dict(n_estimators=300, eval_metric="logloss",
                          tree_method="hist", n_jobs=1, verbosity=0),
            space=SearchSpace([Param("max_depth", "int", 2, 10),
                               Param("learning_rate", "log", 1e-3, 5e-1),
                               Param("subsample", "float", 0.5, 1.0)]))
    except ImportError:
        pass

    return reg


# --------------------------------------------------------------------------- #
# 3. Nested cross-validation                                                   #
# --------------------------------------------------------------------------- #


def _subsample(X: pd.DataFrame, y: np.ndarray, limit: int, rng) -> tuple:
    """Seeded stratified cap on training size (quantum simulation is expensive)."""
    if limit <= 0 or len(y) <= limit:
        return X, y
    idx = np.concatenate([
        rng.choice(np.flatnonzero(y == c),
                   size=max(1, int(round(limit * np.mean(y == c)))), replace=False)
        for c in np.unique(y)
    ])
    rng.shuffle(idx)
    return X.iloc[idx], y[idx]


def _inner_score(spec: ModelSpec, params: dict, X_tr: pd.DataFrame, y_tr: np.ndarray,
                 reduction: str, n_components: int, cfg: ExperimentConfig,
                 fold_seed: int) -> float:
    """Mean F1 over the inner folds -- this is the GWO fitness function."""
    skf = StratifiedKFold(n_splits=cfg.n_inner_splits, shuffle=True, random_state=fold_seed)
    k = params.get("reduce__n_components", n_components)
    scores = []
    for tr, va in skf.split(X_tr, y_tr):
        est = spec.make(params, seed=fold_seed)
        pipe, _ = build_pipeline(X_tr, est, reduction=reduction, n_components=k,
                                 balance=cfg.balance_strategy,
                                 angle_encode=spec.family == "quantum", seed=fold_seed)
        try:
            pipe.fit(X_tr.iloc[tr], y_tr[tr])
            scores.append(f1_score(y_tr[va], pipe.predict(X_tr.iloc[va]), zero_division=0))
        except Exception:
            scores.append(0.0)          # infeasible configuration -> worst fitness
    return float(np.mean(scores))


def _run_one_fold(fold_id: int, tr_idx, te_idx, X: pd.DataFrame, y: np.ndarray,
                  spec: ModelSpec, reduction: str, n_components: int,
                  cfg: ExperimentConfig, optimizer: str | None) -> dict:
    """Execute a single outer fold: (optional) HPO on train, then one test score."""
    seed = get_seed("cv_inner", fold_id)
    rng_sub = get_rng("resampling", fold_id)

    X_tr, y_tr = _subsample(X.iloc[tr_idx], y[tr_idx], cfg.max_train_samples, rng_sub)
    X_te, y_te = X.iloc[te_idx], y[te_idx]

    record: dict[str, Any] = {
        "fold": fold_id, "model": spec.name, "family": spec.family,
        "reduction": reduction.upper(), "optimizer": optimizer or "default",
        "n_train": len(y_tr), "n_test": len(y_te),
    }

    t0 = time.perf_counter()
    best_params: dict[str, Any] = {}
    if optimizer is not None and spec.space is not None:
        searcher = build_optimizer(optimizer, spec.space, cfg, get_rng(
            "gwo" if optimizer == "gwo" else ("pso" if optimizer == "pso" else "random_search"),
            fold_id))
        res = searcher.optimize(
            lambda p: _inner_score(spec, p, X_tr, y_tr, reduction, n_components, cfg, seed),
            budget=cfg.hpo_budget)
        best_params = res.best_params
        record.update({"inner_f1": res.best_score, "n_evaluations": res.n_evaluations,
                       "convergence": res.convergence, "stopped_early": res.stopped_early})
    record["search_time_s"] = time.perf_counter() - t0
    record["best_params"] = best_params

    # ---- refit on the full outer-train set, evaluate once on the outer-test ----
    t1 = time.perf_counter()
    est = spec.make(best_params, seed=seed)
    pipe, k_used = build_pipeline(X_tr, est, reduction=reduction,
                                  n_components=best_params.get("reduce__n_components",
                                                               n_components),
                                  balance=cfg.balance_strategy,
                                  angle_encode=spec.family == "quantum", seed=seed)
    try:
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        try:
            y_score = pipe.predict_proba(X_te)[:, 1]
        except Exception:
            y_score = None
        record.update(compute_metrics(y_te, y_pred, y_score))
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
        record.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
        record["status"] = "ok"
    except Exception as exc:                      # never silently return 0.0
        record.update({m: np.nan for m in
                       ("accuracy", "precision", "recall", "f1",
                        "balanced_accuracy", "mcc", "roc_auc")})
        record.update({"tn": 0, "fp": 0, "fn": 0, "tp": 0})
        record["status"] = f"failed: {type(exc).__name__}: {exc}"
    record["fit_time_s"] = time.perf_counter() - t1
    record["n_reduced_features"] = k_used
    return record


def nested_cv(X: pd.DataFrame, y: np.ndarray, spec: ModelSpec, cfg: ExperimentConfig,
              reduction: str = "lda", n_components: int = 1,
              optimizer: str | None = "gwo", n_jobs: int | None = None,
              verbose: bool = True) -> pd.DataFrame:
    """Run the full nested repeated stratified CV for one model/reduction pair.

    Returns one row per outer fold.
    """
    cv = RepeatedStratifiedKFold(n_splits=cfg.n_outer_splits,
                                 n_repeats=cfg.n_outer_repeats,
                                 random_state=get_seed("cv_outer"))
    splits = list(cv.split(X, y))
    if verbose:
        print(f"    {spec.name:<13s} {reduction.upper():<4s} "
              f"opt={optimizer or 'default':<6s} folds={len(splits)}", flush=True)

    jobs = Parallel(n_jobs=n_jobs if n_jobs is not None else cfg.n_jobs, backend="loky")(
        delayed(_run_one_fold)(i, tr, te, X, y, spec, reduction, n_components, cfg, optimizer)
        for i, (tr, te) in enumerate(splits))
    return pd.DataFrame(jobs)


def summarise(df: pd.DataFrame,
              metrics: Sequence[str] = ("accuracy", "precision", "recall", "f1",
                                        "balanced_accuracy", "mcc", "roc_auc"),
              by: Sequence[str] = ("model", "reduction", "optimizer")) -> pd.DataFrame:
    """Aggregate fold-level results into ``mean +/- std`` with 95% normal CI."""
    g = df.groupby(list(by), dropna=False)
    rows = []
    for key, sub in g:
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["n_folds"] = len(sub)
        for m in metrics:
            if m not in sub:
                continue
            v = sub[m].dropna().to_numpy()
            if len(v) == 0:
                row[f"{m}_mean"] = np.nan
                continue
            row[f"{m}_mean"] = v.mean()
            row[f"{m}_std"] = v.std(ddof=1) if len(v) > 1 else 0.0
            half = 1.96 * (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            row[f"{m}_ci95"] = f"[{v.mean() - half:.3f}, {v.mean() + half:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)


def format_table(summary: pd.DataFrame,
                 metrics: Sequence[str] = ("accuracy", "precision", "recall", "f1")) -> pd.DataFrame:
    """Manuscript-ready ``mean +/- std`` table."""
    out = summary[[c for c in ("model", "reduction", "optimizer") if c in summary]].copy()
    for m in metrics:
        out[m.capitalize()] = [
            f"{r[f'{m}_mean']:.3f} ± {r.get(f'{m}_std', 0.0):.3f}"
            for _, r in summary.iterrows()
        ]
    return out


def run_matrix(X: pd.DataFrame, y: np.ndarray, registry: dict[str, ModelSpec],
               cfg: ExperimentConfig, models: Sequence[str],
               reductions: Sequence[tuple[str, int]] = (("pca", 4), ("lda", 1)),
               optimizers: Sequence[str | None] = (None, "gwo"),
               n_jobs: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Run the full model x reduction x optimizer grid and concatenate fold rows.

    ``optimizers=(None, 'gwo')`` produces the *paired* baseline-vs-optimised
    samples on identical folds, which is what the Wilcoxon signed-rank test in
    :mod:`qmlgwo.stats` consumes.
    """
    frames = []
    for reduction, k in reductions:
        for opt in optimizers:
            for name in models:
                if name not in registry:
                    continue
                spec = registry[name]
                if opt is not None and spec.space is None:
                    continue                     # nothing to tune
                frames.append(nested_cv(X, y, spec, cfg, reduction=reduction,
                                        n_components=k, optimizer=opt,
                                        n_jobs=n_jobs, verbose=verbose))
    return pd.concat(frames, ignore_index=True)


def aggregate_confusion(fold_df: pd.DataFrame,
                        group_cols: Sequence[str] = ("model",)) -> dict[str, np.ndarray]:
    """Sum per-fold confusion counts into one 2x2 matrix per configuration."""
    out = {}
    for key, sub in fold_df.groupby(list(group_cols), dropna=False):
        label = " | ".join(map(str, key)) if isinstance(key, tuple) else str(key)
        out[label] = np.array([[sub["tn"].sum(), sub["fp"].sum()],
                               [sub["fn"].sum(), sub["tp"].sum()]], dtype=int)
    return out


def _parse_cell(v):
    """Turn a CSV-serialised list/dict back into a Python object."""
    import ast
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def load_folds(path) -> pd.DataFrame:
    """Read a fold-level CSV and restore its structured columns.

    ``best_params`` (dict) and ``convergence`` (list) are written to CSV as their
    string representation. Read back naively they stay strings, which silently
    breaks everything downstream that expects the objects: convergence plots come
    out empty and the plateau-evidence analysis finds no selected depths. Use this
    loader wherever fold results are read from disk -- in particular when resuming
    from checkpoints.
    """
    df = pd.read_csv(path)
    for col in ("best_params", "convergence"):
        if col in df.columns:
            df[col] = df[col].map(_parse_cell)
    return df


__all__ = ["ModelSpec", "build_registry", "nested_cv", "run_matrix", "summarise",
           "format_table", "compute_metrics", "aggregate_confusion", "load_folds"]
