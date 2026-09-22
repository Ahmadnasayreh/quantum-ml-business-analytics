"""
qmlgwo.optim
============
Hyper-parameter search: the Grey Wolf Optimizer and equal-budget baselines.

Contents
--------
* :class:`GreyWolfOptimizer` -- search space, lower and upper bounds, population
  size, stopping criterion and initialisation are explicit attributes, and
  :meth:`GreyWolfOptimizer.describe` emits the configuration table.
* :class:`RandomSearch` and :class:`ParticleSwarm` implement the same interface
  and run under an *identical* objective-evaluation budget, which is the only
  fair basis for comparing metaheuristics.

Design note
-----------
GWO is defined on a continuous domain.  Mixed search spaces (integers,
categoricals) are handled by optimising in the unit hypercube ``[0, 1]^d`` and
decoding to native parameter types.  This keeps the GWO update equations
untouched and makes every searcher operate on exactly the same geometry.

Complexity
----------
``O(n_wolves * n_iterations * C_obj)`` time, where ``C_obj`` is the cost of one
inner-CV model evaluation; ``O(n_wolves * d)`` space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 1. Search space                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class Param:
    """One tunable hyper-parameter.

    kind : {'int', 'float', 'log', 'cat'}
        ``'log'`` searches uniformly in log-space (for learning rates, ``C``).
    """
    name: str
    kind: str
    low: float | None = None
    high: float | None = None
    choices: Sequence[Any] | None = None

    def decode(self, u: float) -> Any:
        """Map ``u in [0, 1]`` to a native parameter value."""
        u = float(np.clip(u, 0.0, 1.0))
        if self.kind == "int":
            # +1 - eps so that the top bin has equal measure to the others
            return int(np.clip(int(self.low + u * (self.high - self.low + 1 - 1e-9)),
                               self.low, self.high))
        if self.kind == "float":
            return float(self.low + u * (self.high - self.low))
        if self.kind == "log":
            return float(np.exp(np.log(self.low) + u * (np.log(self.high) - np.log(self.low))))
        if self.kind == "cat":
            k = len(self.choices)
            return self.choices[int(np.clip(int(u * k - 1e-9), 0, k - 1))]
        raise ValueError(f"unknown parameter kind {self.kind!r}")

    def bounds_str(self) -> str:
        if self.kind == "cat":
            return "{" + ", ".join(map(str, self.choices)) + "}"
        fmt = "{:g}"
        return f"[{fmt.format(self.low)}, {fmt.format(self.high)}]" + \
               (" (log)" if self.kind == "log" else "")


@dataclass
class SearchSpace:
    """An ordered collection of :class:`Param`."""
    params: list[Param]

    @property
    def dim(self) -> int:
        return len(self.params)

    def decode(self, u: np.ndarray) -> dict[str, Any]:
        return {p.name: p.decode(ui) for p, ui in zip(self.params, u)}

    def to_frame(self) -> pd.DataFrame:
        """Manuscript-ready table of the search space."""
        return pd.DataFrame([
            {"Hyper-parameter": p.name, "Type": p.kind,
             "Search range (LB, UB)": p.bounds_str()}
            for p in self.params
        ])


# --------------------------------------------------------------------------- #
# 2. Result container                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class SearchResult:
    best_params: dict[str, Any]
    best_score: float
    convergence: list[float] = field(default_factory=list)   # best-so-far per iteration
    n_evaluations: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    stopped_early: bool = False
    optimizer: str = ""


class _BudgetExhausted(Exception):
    """Raised internally when the shared evaluation budget is spent."""


class _Objective:
    """Wraps the user objective, enforces the budget, caches, and logs."""

    def __init__(self, fn: Callable[[dict[str, Any]], float],
                 space: SearchSpace, budget: int):
        self.fn, self.space, self.budget = fn, space, budget
        self.n_evaluations = 0
        self.history: list[dict[str, Any]] = []
        self._cache: dict[tuple, float] = {}
        # NOTE: the budget counts *distinct* configurations, i.e. actual model
        # fits. A repeated configuration is served from cache and costs nothing,
        # which is the compute-faithful definition of a fair budget.

    def __call__(self, u: np.ndarray) -> float:
        if self.n_evaluations >= self.budget:
            raise _BudgetExhausted
        params = self.space.decode(u)
        key = tuple(sorted(params.items(), key=lambda kv: kv[0]))
        if key in self._cache:                 # identical config -> identical score
            return self._cache[key]
        score = float(self.fn(params))
        self.n_evaluations += 1
        self._cache[key] = score
        self.history.append({"eval": self.n_evaluations, "score": score, **params})
        return score


# --------------------------------------------------------------------------- #
# 3. Grey Wolf Optimizer                                                       #
# --------------------------------------------------------------------------- #


class GreyWolfOptimizer:
    r"""Grey Wolf Optimizer (Mirjalili, Mirjalili & Lewis, 2014).

    The three best solutions found so far are named :math:`\alpha, \beta,
    \delta`.  Every other wolf is repositioned toward their consensus:

    .. math::
        \vec{D}_k = |\vec{C}_k \cdot \vec{X}_k - \vec{X}|,\quad
        \vec{X}_k' = \vec{X}_k - \vec{A}_k \cdot \vec{D}_k,\quad k\in\{\alpha,\beta,\delta\}

        \vec{X}(t+1) = \tfrac{1}{3}\left(\vec{X}_\alpha' + \vec{X}_\beta' + \vec{X}_\delta'\right)

    with :math:`\vec{A} = 2a\vec{r}_1 - a`, :math:`\vec{C} = 2\vec{r}_2`,
    :math:`\vec{r}_1, \vec{r}_2 \sim U(0,1)^d`, and the exploration coefficient
    decaying **linearly**

    .. math:: a(t) = a_0\left(1 - \frac{t}{T}\right).

    :math:`|A| > 1` drives exploration (divergence from the leaders),
    :math:`|A| < 1` drives exploitation.

    Parameters
    ----------
    space : SearchSpace
    n_wolves : int, default 8
        Population size :math:`N`.
    n_iterations : int, default 12
        Maximum iterations :math:`T`.
    a_initial : float, default 2.0
    patience, tol : int, float
        Stopping criterion: stop when the :math:`\alpha` score has not improved
        by more than ``tol`` for ``patience`` consecutive iterations.
    initialization : {'lhs', 'uniform'}, default 'lhs'
        ``'lhs'`` = Latin hypercube sampling, which gives better initial
        coverage of the unit cube than i.i.d. uniform sampling at small ``N``.
    rng : numpy.random.Generator
        Seeded generator (see :func:`qmlgwo.config.get_rng`).
    """

    def __init__(self, space: SearchSpace, n_wolves: int = 8, n_iterations: int = 12,
                 a_initial: float = 2.0, patience: int = 4, tol: float = 1e-4,
                 initialization: str = "lhs",
                 rng: np.random.Generator | None = None, verbose: bool = False):
        self.space = space
        self.n_wolves = int(n_wolves)
        self.n_iterations = int(n_iterations)
        self.a_initial = float(a_initial)
        self.patience = int(patience)
        self.tol = float(tol)
        self.initialization = initialization
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.verbose = verbose

    # -- initialisation ----------------------------------------------------- #
    def _initialise(self) -> np.ndarray:
        n, d = self.n_wolves, self.space.dim
        if self.initialization == "lhs":
            cut = (np.arange(n)[:, None] + self.rng.random((n, d))) / n
            for j in range(d):                       # independent permutation / dim
                cut[:, j] = cut[self.rng.permutation(n), j]
            return cut
        return self.rng.random((n, d))

    # -- main loop ---------------------------------------------------------- #
    def optimize(self, objective: Callable[[dict[str, Any]], float],
                 budget: int | None = None) -> SearchResult:
        budget = budget if budget is not None else self.n_wolves * self.n_iterations
        obj = _Objective(objective, self.space, budget)
        # Run until the shared budget is spent: with an explicit budget the
        # iteration count -- and hence the a(t) schedule -- is derived from it,
        # so every searcher receives exactly the same compute.
        T = max(self.n_iterations, int(np.ceil(budget / self.n_wolves)))

        wolves = self._initialise()
        alpha = (np.zeros(self.space.dim), -np.inf)
        beta = (np.zeros(self.space.dim), -np.inf)
        delta = (np.zeros(self.space.dim), -np.inf)

        convergence: list[float] = []
        stalled = 0
        stopped_early = False

        try:
            for t in range(T):
                # --- evaluate and rank the pack ------------------------------
                for i in range(self.n_wolves):
                    score = obj(wolves[i])
                    if score > alpha[1]:
                        delta, beta, alpha = beta, alpha, (wolves[i].copy(), score)
                    elif score > beta[1]:
                        delta, beta = beta, (wolves[i].copy(), score)
                    elif score > delta[1]:
                        delta = (wolves[i].copy(), score)

                convergence.append(alpha[1])

                # --- stopping criterion --------------------------------------
                if len(convergence) > 1 and convergence[-1] - convergence[-2] <= self.tol:
                    stalled += 1
                    if stalled >= self.patience:
                        stopped_early = True
                        break
                else:
                    stalled = 0

                # --- position update ----------------------------------------
                a = self.a_initial * (1.0 - t / max(1, T - 1))
                new = np.empty_like(wolves)
                for i in range(self.n_wolves):
                    X = wolves[i]
                    moves = []
                    for leader, _ in (alpha, beta, delta):
                        r1, r2 = self.rng.random(self.space.dim), self.rng.random(self.space.dim)
                        A = 2.0 * a * r1 - a
                        C = 2.0 * r2
                        D = np.abs(C * leader - X)
                        moves.append(leader - A * D)
                    new[i] = np.clip(np.mean(moves, axis=0), 0.0, 1.0)
                wolves = new

                if self.verbose:
                    print(f"  GWO iter {t + 1:02d}/{self.n_iterations} "
                          f"alpha={alpha[1]:.4f} a={a:.3f} evals={obj.n_evaluations}")
        except _BudgetExhausted:
            pass

        return SearchResult(
            best_params=self.space.decode(alpha[0]), best_score=float(alpha[1]),
            convergence=convergence, n_evaluations=obj.n_evaluations,
            history=obj.history, stopped_early=stopped_early, optimizer="GWO",
        )

    # -- reporting ---------------------------------------------------------- #
    def describe(self) -> pd.DataFrame:
        """Configuration table for the manuscript."""
        rows = [
            ("Population size (N)", self.n_wolves),
            ("Maximum iterations (T)", self.n_iterations),
            ("Objective-evaluation budget", self.n_wolves * self.n_iterations),
            ("Initial exploration coefficient a0", self.a_initial),
            ("Decay schedule for a", "linear: a(t) = a0 (1 - t/T)"),
            ("Initialization strategy", self.initialization),
            ("Stopping criterion",
             f"alpha improvement <= {self.tol:g} for {self.patience} consecutive iterations"),
            ("Fitness function", "mean inner-CV F1 (positive class)"),
        ]
        return pd.DataFrame(rows, columns=["Setting", "Value"])


# --------------------------------------------------------------------------- #
# 4. Equal-budget baselines                                                    #
# --------------------------------------------------------------------------- #


class RandomSearch:
    """Uniform random search -- the reference baseline every metaheuristic must beat."""

    def __init__(self, space: SearchSpace, rng: np.random.Generator | None = None):
        self.space = space
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def optimize(self, objective, budget: int) -> SearchResult:
        obj = _Objective(objective, self.space, budget)
        best_u, best_s, convergence = None, -np.inf, []
        try:
            for _ in range(budget):
                u = self.rng.random(self.space.dim)
                s = obj(u)
                if s > best_s:
                    best_u, best_s = u, s
                convergence.append(best_s)
        except _BudgetExhausted:
            pass
        return SearchResult(self.space.decode(best_u), float(best_s), convergence,
                            obj.n_evaluations, obj.history, False, "RandomSearch")


class ParticleSwarm:
    """Standard PSO with inertia damping (Shi & Eberhart, 1998)."""

    def __init__(self, space: SearchSpace, n_particles: int = 8, n_iterations: int = 12,
                 w0: float = 0.9, w1: float = 0.4, c1: float = 1.5, c2: float = 1.5,
                 rng: np.random.Generator | None = None):
        self.space, self.n_particles, self.n_iterations = space, n_particles, n_iterations
        self.w0, self.w1, self.c1, self.c2 = w0, w1, c1, c2
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def optimize(self, objective, budget: int | None = None) -> SearchResult:
        budget = budget if budget is not None else self.n_particles * self.n_iterations
        obj = _Objective(objective, self.space, budget)
        T = max(self.n_iterations, int(np.ceil(budget / self.n_particles)))
        d = self.space.dim
        X = self.rng.random((self.n_particles, d))
        V = self.rng.uniform(-0.1, 0.1, (self.n_particles, d))
        pbest, pbest_s = X.copy(), np.full(self.n_particles, -np.inf)
        gbest, gbest_s, convergence = X[0].copy(), -np.inf, []
        try:
            for t in range(T):
                for i in range(self.n_particles):
                    s = obj(X[i])
                    if s > pbest_s[i]:
                        pbest_s[i], pbest[i] = s, X[i].copy()
                    if s > gbest_s:
                        gbest_s, gbest = s, X[i].copy()
                convergence.append(gbest_s)
                w = self.w0 - (self.w0 - self.w1) * t / max(1, T - 1)
                r1, r2 = self.rng.random((self.n_particles, d)), self.rng.random((self.n_particles, d))
                V = w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)
                V = np.clip(V, -0.5, 0.5)
                X = np.clip(X + V, 0.0, 1.0)
        except _BudgetExhausted:
            pass
        return SearchResult(self.space.decode(gbest), float(gbest_s), convergence,
                            obj.n_evaluations, obj.history, False, "PSO")


def build_optimizer(name: str, space: SearchSpace, cfg, rng) -> Any:
    """Factory used by the evaluation loop so all searchers share one budget."""
    name = name.lower()
    if name == "gwo":
        return GreyWolfOptimizer(space, cfg.gwo_n_wolves, cfg.gwo_n_iterations,
                                 cfg.gwo_a_initial, cfg.gwo_patience, cfg.gwo_tol, rng=rng)
    if name in ("random", "randomsearch"):
        return RandomSearch(space, rng=rng)
    if name == "pso":
        return ParticleSwarm(space, cfg.gwo_n_wolves, cfg.gwo_n_iterations, rng=rng)
    raise ValueError(f"unknown optimizer {name!r}")


__all__ = ["Param", "SearchSpace", "SearchResult", "GreyWolfOptimizer",
           "RandomSearch", "ParticleSwarm", "build_optimizer"]
