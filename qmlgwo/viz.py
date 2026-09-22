"""
qmlgwo.viz
==========
Publication-quality figures.

Every function takes an explicit ``figure_number`` and writes
``figures/Figure_<n>_<slug>.{png,pdf}`` at 600 dpi. These per-dataset files are
the panels that ``make_paper_figures.py`` assembles into the manuscript figures
(see README, section 6, for the mapping).

All figures are greyscale-safe (distinct markers and line styles, colour-blind
safe palette) and use vector PDF for the camera-ready version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Okabe-Ito colour-blind-safe palette
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 600, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "DejaVu Sans",
    "savefig.bbox": "tight", "figure.constrained_layout.use": True,
})


def _save(fig, figure_number: int, slug: str, outdir: str = "figures") -> str:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"Figure_{figure_number}_{slug}"
    # dpi passed explicitly (not only via rcParams) so no library that touches
    # rcParams -- shap, for example -- can lower the resolution of a figure.
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")   # vector
    plt.close(fig)
    print(f"   saved {stem}.png / .pdf")
    return str(stem)


# --------------------------------------------------------------------------- #
# Figure: GWO convergence                                                      #
# --------------------------------------------------------------------------- #


def plot_convergence(fold_df: pd.DataFrame, figure_number: int = 2,
                     title: str = "GWO convergence", outdir: str = "figures",
                     slug: str = "gwo_convergence"):
    """Mean best-so-far fitness per iteration, with a +/-1 s.d. band over folds.

    The band distinguishes optimisation progress from the variation between
    folds, which a single curve cannot show.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for i, (model, sub) in enumerate(fold_df.groupby("model")):
        curves = [c for c in sub["convergence"] if isinstance(c, (list, np.ndarray)) and len(c)]
        if not curves:
            continue
        L = max(len(c) for c in curves)
        padded = np.array([list(c) + [c[-1]] * (L - len(c)) for c in curves], dtype=float)
        m, s = padded.mean(0), padded.std(0, ddof=1) if len(padded) > 1 else np.zeros(L)
        it = np.arange(1, L + 1)
        ax.plot(it, m, color=PALETTE[i % len(PALETTE)], marker=MARKERS[i % len(MARKERS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)], markersize=4, label=model)
        ax.fill_between(it, m - s, m + s, color=PALETTE[i % len(PALETTE)], alpha=0.15)
    ax.set_xlabel("GWO iteration")
    ax.set_ylabel("Best inner-CV $F_1$ (mean $\\pm$ 1 s.d. over folds)")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    return _save(fig, figure_number, slug, outdir)


# --------------------------------------------------------------------------- #
# Figure: confusion matrices                                                   #
# --------------------------------------------------------------------------- #


def plot_confusion_matrices(cms: dict[str, np.ndarray], figure_number: int = 3,
                            outdir: str = "figures", slug: str = "confusion_matrices",
                            normalize: bool = True, class_names=("Negative", "Positive")):
    """Grid of confusion matrices, one panel per model.

    ``cms`` maps a model name to a 2x2 matrix **summed over all outer folds**,
    which is the honest aggregate when reporting repeated CV.
    """
    n = len(cms)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows), squeeze=False)

    for ax, (name, cm) in zip(axes.ravel(), cms.items()):
        cm = np.asarray(cm, float)
        disp = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
        im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=1 if normalize else disp.max())
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                txt = f"{disp[i, j]:.2f}\n({int(cm[i, j])})" if normalize else f"{int(cm[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if disp[i, j] > 0.5 * disp.max() else "black")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0, 1], class_names, fontsize=8)
        ax.set_yticks([0, 1], class_names, fontsize=8, rotation=90, va="center")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7,
                 label="Row-normalised rate" if normalize else "Count")
    return _save(fig, figure_number, slug, outdir)


# --------------------------------------------------------------------------- #
# Figure: performance comparison with error bars                               #
# --------------------------------------------------------------------------- #


def plot_performance(summary: pd.DataFrame, metric: str = "f1", figure_number: int = 5,
                     outdir: str = "figures", slug: str = "performance",
                     title: str = "", hue: str = "reduction"):
    """Horizontal bars with 95% CI whiskers -- replaces bare point estimates."""
    df = summary.sort_values(f"{metric}_mean")
    labels = [f"{r['model']} ({r[hue]})" if hue in df else r["model"]
              for _, r in df.iterrows()]
    means = df[f"{metric}_mean"].to_numpy()
    stds = df.get(f"{metric}_std", pd.Series(np.zeros(len(df)))).to_numpy()
    err = 1.96 * stds / np.sqrt(df["n_folds"].to_numpy().clip(min=1))

    groups = df[hue].astype(str).to_numpy() if hue in df else np.array([""] * len(df))
    uniq = list(dict.fromkeys(groups))
    colors = [PALETTE[uniq.index(g) % len(PALETTE)] for g in groups]

    fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(df) + 1.6))
    ax.barh(labels, means, xerr=err, color=colors, alpha=0.85,
            error_kw=dict(ecolor="black", capsize=3, lw=1))
    ax.set_xlabel(f"{metric.upper()} (mean, 95% CI over outer folds)")
    ax.set_xlim(0, min(1.0, means.max() + err.max() + 0.08))
    if title:
        ax.set_title(title)
    if len(uniq) > 1:
        ax.legend(handles=[Line2D([0], [0], color=PALETTE[i % len(PALETTE)], lw=6, label=g)
                           for i, g in enumerate(uniq)], frameon=False, loc="lower right")
    return _save(fig, figure_number, slug, outdir)


# --------------------------------------------------------------------------- #
# Figure: barren plateau scaling                                               #
# --------------------------------------------------------------------------- #


def plot_gradient_variance(scan: pd.DataFrame, fits: pd.DataFrame | None = None,
                           figure_number: int = 6, outdir: str = "figures",
                           slug: str = "barren_plateau"):
    """``Var[partial <Z_0>]`` vs qubit count on a log axis, with fitted decay.

    This is the figure that turns the manuscript's barren-plateau assertion into
    evidence, or refutes it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    for i, (L, sub) in enumerate(scan.groupby("n_layers")):
        sub = sub.sort_values("n_qubits")
        ax.semilogy(sub["n_qubits"], sub["grad_var"], marker=MARKERS[i % len(MARKERS)],
                    color=PALETTE[i % len(PALETTE)], linestyle=LINESTYLES[i % len(LINESTYLES)],
                    label=f"depth $L$ = {L}")
        if fits is not None and not fits.empty:
            row = fits[fits["n_layers"] == L]
            if len(row):
                r = row.iloc[0]
                xs = np.linspace(sub["n_qubits"].min(), sub["n_qubits"].max(), 50)
                ax.semilogy(xs, np.exp(r["intercept"] - r["decay_rate_alpha"] * xs),
                            color=PALETTE[i % len(PALETTE)], lw=1, alpha=0.45)
    ax.set_xlabel("Number of qubits $n$")
    ax.set_ylabel(r"$\mathrm{Var}\left[\partial \langle Z_0\rangle / \partial \theta_k\right]$")
    ax.set_title("(a) Gradient variance vs. system size")
    ax.legend(frameon=False)

    ax = axes[1]
    for i, (n_q, sub) in enumerate(scan.groupby("n_qubits")):
        sub = sub.sort_values("n_layers")
        ax.semilogy(sub["n_layers"], sub["grad_var"], marker=MARKERS[i % len(MARKERS)],
                    color=PALETTE[i % len(PALETTE)], linestyle=LINESTYLES[i % len(LINESTYLES)],
                    label=f"$n$ = {n_q}")
    ax.set_xlabel("Ansatz depth $L$")
    ax.set_ylabel(r"$\mathrm{Var}\left[\partial \langle Z_0\rangle / \partial \theta_k\right]$")
    ax.set_title("(b) Gradient variance vs. circuit depth")
    ax.legend(frameon=False, ncol=2)
    return _save(fig, figure_number, slug, outdir)


# --------------------------------------------------------------------------- #
# Figure: optimizer trajectories                                               #
# --------------------------------------------------------------------------- #


def plot_optimizer_comparison(traj: pd.DataFrame, figure_number: int = 7,
                              outdir: str = "figures", slug: str = "optimizer_comparison"):
    """(a) loss vs. fraction of each optimiser's own run; (b) the comparable metric.

    Adam/SPSA steps are epochs and COBYLA steps are single objective evaluations,
    so a shared raw-step axis would squeeze two curves against the origin. Panel
    (a) shows the SHAPE of convergence on a normalised axis; panel (b) compares
    the optimisers on one common metric, full-batch cross-entropy after training.
    """
    curves = traj[traj.get("final_loss_full", pd.Series(index=traj.index)).isna()] \
        if "final_loss_full" in traj else traj
    summ = traj.dropna(subset=["final_loss_full"]) if "final_loss_full" in traj else None

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0),
                             gridspec_kw={"width_ratios": [1.6, 1.0]})
    ax = axes[0]
    for i, (opt, sub) in enumerate(curves.groupby("optimizer")):
        sub = sub.sort_values("step")
        x = sub["progress"] if "progress" in sub else sub["step"] / max(sub["step"].max(), 1)
        n = int(sub["step"].max()) + 1
        unit = "evals" if opt == "COBYLA" else "epochs"
        ax.plot(x, sub["loss"], color=PALETTE[i % len(PALETTE)],
                linestyle=LINESTYLES[i % len(LINESTYLES)], lw=1.6, label=f"{opt} ({n} {unit})")
    ax.set_xlabel("Fraction of optimisation run")
    ax.set_ylabel("Mean binary cross-entropy")
    ax.set_title("(a) Convergence profile")
    ax.legend(frameon=False)

    ax = axes[1]
    if summ is not None and len(summ):
        summ = summ.sort_values("final_loss_full")
        bars = ax.bar(summ["optimizer"], summ["final_loss_full"],
                      color=[PALETTE[i % len(PALETTE)] for i in range(len(summ))], alpha=0.85)
        for b, f1 in zip(bars, summ["final_f1"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"F1 {f1:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Full-batch cross-entropy after training")
        ax.set_title("(b) Common end-of-training metric")
    else:
        ax.axis("off")
    return _save(fig, figure_number, slug, outdir)


# --------------------------------------------------------------------------- #
# Figure: critical-difference diagram                                          #
# --------------------------------------------------------------------------- #


def plot_critical_difference(ranks: pd.DataFrame, cd: float, figure_number: int = 8,
                             outdir: str = "figures", slug: str = "critical_difference",
                             title: str = "Nemenyi critical difference"):
    """Demsar-style CD diagram: mean ranks with cliques of non-distinguishable models."""
    ranks = ranks.sort_values("mean_rank").reset_index(drop=True)
    names, r = ranks["config"].tolist(), ranks["mean_rank"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.0, 0.32 * len(names) + 2.2))
    lo, hi = np.floor(r.min() - 0.5), np.ceil(r.max() + 0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(-len(names) - 1.5, 1.6)
    ax.axis("off")

    ax.plot([lo, hi], [0.6, 0.6], color="black", lw=1.2)
    for t in np.arange(lo, hi + 0.5, 0.5):
        ax.plot([t, t], [0.6, 0.78], color="black", lw=1)
        ax.text(t, 0.95, f"{t:g}", ha="center", fontsize=8)

    for i, (nm, rv) in enumerate(zip(names, r)):
        y = -i - 0.6
        ax.plot([rv, rv], [0.6, y], color=PALETTE[i % len(PALETTE)], lw=1.1)
        ax.plot([rv, lo + 0.15], [y, y], color=PALETTE[i % len(PALETTE)], lw=1.1)
        ax.text(lo + 0.1, y, f"{nm}  ({rv:.2f})", ha="right", va="center", fontsize=8)

    # cliques: maximal runs of consecutive models whose rank span is within CD.
    # A run nested inside an earlier one (same right end) is redundant and is
    # not drawn -- standard Demsar CD diagrams show maximal cliques only.
    yc = -len(names) - 0.6
    last_j = -1
    for i in range(len(r)):
        j = i
        while j + 1 < len(r) and r[j + 1] - r[i] <= cd:
            j += 1
        if j > i and j > last_j:
            ax.plot([r[i], r[j]], [yc, yc], color="black", lw=3, solid_capstyle="butt")
            yc -= 0.28
            last_j = j

    ax.plot([lo + 0.3, lo + 0.3 + cd], [1.35, 1.35], color="black", lw=2)
    ax.text(lo + 0.3 + cd / 2, 1.5, f"CD = {cd:.2f}", ha="center", fontsize=9)
    ax.set_title(title, fontsize=11)
    return _save(fig, figure_number, slug, outdir)


# --------------------------------------------------------------------------- #
# Figure: SHAP                                                                 #
# --------------------------------------------------------------------------- #


def plot_shap_summary(pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame,
                      figure_number: int = 4, outdir: str = "figures",
                      slug: str = "shap_summary", max_background: int = 60,
                      max_explain: int = 120, seed: int = 0):
    """Model-agnostic KernelSHAP over the **whole fitted pipeline**.

    Explaining the pipeline rather than the bare estimator means the attributions
    are expressed in the original feature space, not in the post-LDA coordinate
    where there is only one axis (``LD1``) and the plot is uninformative.

    Complexity: KernelSHAP is O(n_explain * n_background * n_coalitions); keep
    both caps small.
    """
    import shap

    rng = np.random.default_rng(seed)
    bg = X_background.iloc[rng.choice(len(X_background),
                                      min(max_background, len(X_background)),
                                      replace=False)]
    ex = X_explain.iloc[rng.choice(len(X_explain),
                                   min(max_explain, len(X_explain)), replace=False)]

    f = lambda arr: pipeline.predict_proba(pd.DataFrame(arr, columns=X_background.columns))[:, 1]
    explainer = shap.KernelExplainer(f, bg, silent=True)
    sv = explainer.shap_values(ex, nsamples="auto", silent=True)

    # shap.summary_plot adds a colorbar and then calls tight_layout(); that
    # collides with the constrained-layout engine enabled globally above and
    # raises "Colorbar layout of new layout engine not compatible". Opt out here.
    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig = plt.figure(figsize=(7.0, 5.0))
        shap.summary_plot(sv, ex, show=False, plot_size=None)
        return _save(fig, figure_number, slug, outdir)


__all__ = ["plot_convergence", "plot_confusion_matrices", "plot_performance",
           "plot_gradient_variance", "plot_optimizer_comparison",
           "plot_critical_difference", "plot_shap_summary", "PALETTE"]
