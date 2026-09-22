"""
qmlgwo.config
=============
Centralised configuration, seeding and the reproducibility manifest.

Every stochastic component in this project draws from a SeedSequence that is
spawned deterministically from MASTER_SEED. Re-running the pipeline with the
same MASTER_SEED reproduces fold assignment and every seeded draw exactly.
Model scores agree to within numerical tolerance only: differences of up to
~0.005 F1 were observed between independent executions, consistent with
non-deterministic multithreaded floating-point arithmetic and library-version
drift. Report results from ONE complete execution and keep its fold-level
files; never mix results from separate runs.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# 1. Master seed and derived seed streams                                      #
# --------------------------------------------------------------------------- #

MASTER_SEED: int = 20250917

#: Named, independent seed streams.  Using distinct streams (rather than
#: re-using one global seed) guarantees that, e.g., changing the number of GWO
#: wolves does not perturb the CV fold assignment.
_STREAMS = (
    "cv_outer",        # outer repeated-stratified-CV fold assignment
    "cv_inner",        # inner CV used for hyper-parameter selection
    "resampling",      # undersampling / SMOTE
    "gwo",             # Grey Wolf Optimizer
    "pso",             # Particle Swarm (baseline optimiser)
    "random_search",   # Random Search (baseline optimiser)
    "circuit_init",    # variational circuit parameter initialisation
    "spsa",            # SPSA perturbation directions
    "shap",            # SHAP background sampling
    "barren",          # barren-plateau gradient-variance study
)

_seed_seq = np.random.SeedSequence(MASTER_SEED)
_CHILDREN = dict(zip(_STREAMS, _seed_seq.spawn(len(_STREAMS))))


def get_rng(stream: str, offset: int = 0) -> np.random.Generator:
    """Return an independent, reproducible ``Generator`` for a named stream.

    Parameters
    ----------
    stream : str
        One of :data:`_STREAMS`.
    offset : int, default 0
        Sub-stream index.  Use the fold index / repetition index here so that
        each fold gets its own independent but reproducible generator.

    Notes
    -----
    Time complexity O(1); space O(1).
    """
    if stream not in _CHILDREN:
        raise KeyError(f"unknown seed stream {stream!r}; expected one of {_STREAMS}")
    return np.random.default_rng(_CHILDREN[stream].spawn(offset + 1)[offset])


def get_seed(stream: str, offset: int = 0) -> int:
    """Return a 32-bit integer seed for libraries that take ``int`` seeds
    (scikit-learn ``random_state``, XGBoost, ...)."""
    return int(get_rng(stream, offset).integers(0, 2**31 - 1))


def seed_everything(seed: int = MASTER_SEED) -> None:
    """Seed all global RNGs.  Call once at notebook start."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
# 2. Experiment presets                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class ExperimentConfig:
    """Full experimental protocol.  Serialised into every results directory."""

    # ---- outer evaluation protocol ----
    n_outer_splits: int = 5
    n_outer_repeats: int = 3          # -> 15 independent estimates per model
    # ---- inner protocol for hyper-parameter selection (no test leakage) ----
    n_inner_splits: int = 3

    # ---- GWO: population, bounds, stopping criteria ----
    gwo_n_wolves: int = 8
    gwo_n_iterations: int = 12
    gwo_a_initial: float = 2.0        # a decreases linearly a_init -> 0
    gwo_patience: int = 4             # early stop: no alpha improvement
    gwo_tol: float = 1e-4

    # ---- fair-budget comparison across hyper-parameter search strategies ----
    #: All strategies receive exactly this many objective-function evaluations.
    hpo_budget: int = 96              # = gwo_n_wolves * gwo_n_iterations

    # ---- variational circuit training ----
    n_epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.08
    diff_method: str = "backprop"     # exact on simulator; "parameter-shift" for hardware-faithful
    max_qubits: int = 6

    # ---- dimensionality reduction ----
    pca_components: int = 4
    lda_components: int = 1           # binary task => min(n_comp, n_classes-1) = 1

    # ---- class imbalance ----
    balance_strategy: str = "undersample"   # {'undersample','smote','class_weight','none'}

    # ---- subsampling for tractability of the quantum simulator ----
    max_train_samples: int = 1200
    max_kernel_samples: int = 600     # fidelity kernel is O(n^2) circuit evaluations

    # ---- statistics ----
    alpha: float = 0.05
    n_bootstrap: int = 2000

    # ---- bookkeeping ----
    master_seed: int = MASTER_SEED
    results_dir: str = "results"
    figures_dir: str = "figures"
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if self.hpo_budget != self.gwo_n_wolves * self.gwo_n_iterations:
            # A fair comparison between search strategies requires equal budgets.
            self.hpo_budget = self.gwo_n_wolves * self.gwo_n_iterations

    # -- convenience ------------------------------------------------------- #
    @property
    def n_outer_folds(self) -> int:
        return self.n_outer_splits * self.n_outer_repeats

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": self.to_dict(), "environment": environment_manifest()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


#: Reduced-cost preset for smoke tests / debugging. NEVER use for paper numbers.
SMOKE = ExperimentConfig(
    n_outer_splits=3, n_outer_repeats=1, n_inner_splits=2,
    gwo_n_wolves=4, gwo_n_iterations=3, hpo_budget=12,
    n_epochs=6, max_train_samples=200, max_kernel_samples=120,
    n_bootstrap=200,
)

#: Full protocol used for the reported results.
FULL = ExperimentConfig()


# --------------------------------------------------------------------------- #
# 3. Environment manifest                                                      #
# --------------------------------------------------------------------------- #


def environment_manifest() -> dict[str, Any]:
    """Capture the exact software/hardware environment for reproducibility."""

    def _ver(mod: str) -> str | None:
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        commit = None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "git_commit": commit,
        "packages": {
            m: _ver(m)
            for m in ("numpy", "scipy", "sklearn", "pennylane",
                      "pandas", "xgboost", "shap", "imblearn", "matplotlib")
        },
    }


__all__ = [
    "MASTER_SEED", "ExperimentConfig", "SMOKE", "FULL",
    "get_rng", "get_seed", "seed_everything", "environment_manifest",
]
