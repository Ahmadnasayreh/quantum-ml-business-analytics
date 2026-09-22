"""
qmlgwo.stats
============
Statistical validation of the cross-validated results.

Methods, and why each one
-------------------------
* **Wilcoxon signed-rank** -- paired, non-parametric comparison of two models on
  the *same* outer folds.  Pairing is essential: fold difficulty varies far more
  than the difference between models, so an unpaired test throws away most of
  the power.  Demsar (2006) recommends exactly this for two classifiers on one
  dataset.
* **Holm-Bonferroni** -- step-down correction.  Comparing ``m`` models pairwise
  produces ``m(m-1)/2`` tests; uncorrected p-values would be meaningless.  Holm
  is uniformly more powerful than Bonferroni and makes no independence
  assumption.
* **Cliff's delta** -- non-parametric effect size.  A p-value says an effect
  exists; ``delta`` says whether it matters.  With 15 folds a trivial difference
  can still reach significance.
* **Friedman + Nemenyi** -- omnibus rank test across ``k > 2`` models, followed
  by the post-hoc that controls the family-wise error rate; the accompanying
  critical difference supports a CD diagram.
* **BCa bootstrap** -- confidence intervals that do not assume normality of the
  fold scores.

References
----------
Demsar, J. (2006). *Statistical comparisons of classifiers over multiple data
sets.* JMLR 7, 1-30.
Benavoli, A., Corani, G., Mangili, F. (2016). *Should we really use post-hoc
tests based on mean-ranks?* JMLR 17(5), 1-10.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# 1. Effect size                                                               #
# --------------------------------------------------------------------------- #


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> tuple[float, str]:
    r"""Cliff's delta and its conventional magnitude label.

    .. math:: \delta = \frac{\#\{a_i > b_j\} - \#\{a_i < b_j\}}{n_a n_b}

    Thresholds (Romano et al., 2006): ``|d| < 0.147`` negligible,
    ``< 0.33`` small, ``< 0.474`` medium, else large.

    Complexity O(n_a n_b) time, O(1) extra space.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, "undefined"
    diff = np.sign(a[:, None] - b[None, :])
    d = float(diff.sum() / (len(a) * len(b)))
    m = abs(d)
    label = ("negligible" if m < 0.147 else "small" if m < 0.33
             else "medium" if m < 0.474 else "large")
    return d, label


# --------------------------------------------------------------------------- #
# 2. Multiple-comparison correction                                            #
# --------------------------------------------------------------------------- #


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05):
    """Step-down Holm correction.

    Returns ``(adjusted_p, reject)`` in the caller's original ordering.
    Complexity O(m log m).
    """
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj, adj < alpha


# --------------------------------------------------------------------------- #
# 3. Pairwise comparison                                                       #
# --------------------------------------------------------------------------- #


def paired_comparison(fold_df: pd.DataFrame, metric: str = "f1",
                      group_cols: Sequence[str] = ("model", "reduction", "optimizer"),
                      alpha: float = 0.05, baseline: str | None = None) -> pd.DataFrame:
    """All pairwise (or vs-baseline) paired Wilcoxon tests with Holm correction.

    Parameters
    ----------
    fold_df : DataFrame
        Fold-level output of :func:`qmlgwo.evaluate.nested_cv` (must contain a
        ``fold`` column so scores can be paired).
    baseline : str, optional
        If given, compare every configuration against this one only
        (``m - 1`` tests instead of ``m(m-1)/2``, which preserves power).
    """
    fold_df = fold_df.copy()
    fold_df["_cfg"] = fold_df[list(group_cols)].astype(str).agg(" | ".join, axis=1)
    wide = fold_df.pivot_table(index="fold", columns="_cfg", values=metric)

    cfgs = list(wide.columns)
    pairs = ([(baseline, c) for c in cfgs if c != baseline]
             if baseline is not None else list(combinations(cfgs, 2)))

    rows = []
    for a, b in pairs:
        pair = wide[[a, b]].dropna()
        x, y = pair[a].to_numpy(), pair[b].to_numpy()
        if len(x) < 3:
            continue
        if np.allclose(x, y):
            stat, p = np.nan, 1.0          # Wilcoxon is undefined for all-zero diffs
        else:
            stat, p = stats.wilcoxon(x, y, alternative="two-sided",
                                     zero_method="pratt")
        d, mag = cliffs_delta(x, y)
        rows.append({
            "config_A": a, "config_B": b, "n_pairs": len(x),
            f"{metric}_A": x.mean(), f"{metric}_B": y.mean(),
            "mean_diff": x.mean() - y.mean(),
            "median_diff": float(np.median(x - y)),
            "W": stat, "p_raw": p, "cliffs_delta": d, "effect_size": mag,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"], out["significant"] = holm_bonferroni(out["p_raw"], alpha)
    out["winner"] = np.where(~out["significant"], "n.s.",
                             np.where(out["mean_diff"] > 0, out["config_A"], out["config_B"]))
    return out.sort_values("p_holm").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Omnibus test across models                                                #
# --------------------------------------------------------------------------- #


def friedman_nemenyi(fold_df: pd.DataFrame, metric: str = "f1",
                     group_cols: Sequence[str] = ("model", "reduction", "optimizer"),
                     alpha: float = 0.05):
    """Friedman omnibus test, mean ranks, Nemenyi post-hoc and critical difference.

    Returns ``(result_dict, ranks_df, nemenyi_p_matrix)``.
    """
    df = fold_df.copy()
    df["_cfg"] = df[list(group_cols)].astype(str).agg(" | ".join, axis=1)
    wide = df.pivot_table(index="fold", columns="_cfg", values=metric).dropna(axis=0, how="any")

    k, n = wide.shape[1], wide.shape[0]
    if k < 3 or n < 3:
        return ({"error": f"need >=3 configs and >=3 folds; got k={k}, n={n}"},
                pd.DataFrame(), pd.DataFrame())

    chi2, p = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    # Higher metric = better, so rank descending (rank 1 = best).
    ranks = wide.rank(axis=1, ascending=False).mean(axis=0).sort_values()
    ranks_df = ranks.rename("mean_rank").reset_index().rename(columns={"_cfg": "config"})

    # Nemenyi critical difference: CD = q_alpha * sqrt(k(k+1) / (6n))
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
               8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
               13: 3.313, 14: 3.354, 15: 3.391}.get(k, 3.391)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    try:
        import scikit_posthocs as sp
        nemenyi = sp.posthoc_nemenyi_friedman(wide.to_numpy())
        nemenyi.index = nemenyi.columns = wide.columns
    except ImportError:
        nemenyi = pd.DataFrame()

    return ({"chi2": float(chi2), "p_value": float(p), "k_configs": int(k),
             "n_folds": int(n), "critical_difference": float(cd),
             "significant": bool(p < alpha), "alpha": alpha}, ranks_df, nemenyi)


# --------------------------------------------------------------------------- #
# 5. Bootstrap confidence intervals                                            #
# --------------------------------------------------------------------------- #


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap CI of the mean.

    Falls back to the percentile interval when the acceleration term is
    undefined (e.g. all resamples identical).  Complexity O(n_boot * n).
    """
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan, np.nan, np.nan)
    rng = rng or np.random.default_rng(0)

    theta = v.mean()
    boot = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)

    z0_prop = np.mean(boot < theta)
    if z0_prop in (0.0, 1.0):
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(theta), float(lo), float(hi)
    z0 = stats.norm.ppf(z0_prop)

    jack = np.array([np.delete(v, i).mean() for i in range(len(v))])
    jm = jack.mean()
    denom = 6.0 * ((jm - jack) ** 2).sum() ** 1.5
    a = ((jm - jack) ** 3).sum() / denom if denom > 0 else 0.0

    zl, zu = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    al = stats.norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    au = stats.norm.cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
    lo, hi = np.percentile(boot, [100 * al, 100 * au])
    return float(theta), float(lo), float(hi)


def add_bootstrap_cis(fold_df: pd.DataFrame, metric: str = "f1",
                      group_cols: Sequence[str] = ("model", "reduction", "optimizer"),
                      n_boot: int = 2000, alpha: float = 0.05,
                      rng: np.random.Generator | None = None) -> pd.DataFrame:
    """BCa interval per configuration -- the CI column for the results tables."""
    rows = []
    for key, sub in fold_df.groupby(list(group_cols), dropna=False):
        m, lo, hi = bootstrap_ci(sub[metric], n_boot, alpha, rng)
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update({f"{metric}_mean": m, f"{metric}_ci_low": lo, f"{metric}_ci_high": hi,
                    f"{metric}_ci95": f"[{lo:.3f}, {hi:.3f}]", "n_folds": len(sub)})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"{metric}_mean", ascending=False).reset_index(drop=True)


__all__ = ["cliffs_delta", "holm_bonferroni", "paired_comparison",
           "friedman_nemenyi", "bootstrap_ci", "add_bootstrap_cis"]
