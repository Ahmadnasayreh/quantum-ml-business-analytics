"""
make_paper_figures.py
=====================
Build the eight manuscript figures in their FINAL form for PeerJ submission.

Run from the repository root, after both notebooks have finished:

    %run make_paper_figures.py          (inside Jupyter)
    python make_paper_figures.py        (from a terminal)

Output: paper_figures/Figure 1.png ... Figure 8.png (600 dpi) and matching PDFs.

PeerJ rules this script implements (https://peerj.com/about/author-instructions/):
  * figures are uploaded as separate files named 'Figure N', never embedded;
  * a multi-part figure is ONE file, arranged as it will be published;
  * figures are numbered in order of first citation in the text;
  * no figure title or legend inside the image on re-review (panel labels only);
  * ideal width about 3000 px -- these are 4200-4500 px at 600 dpi;
  * colour-blind-safe palette, and never colour alone (hatching marks quantum).

Numbering follows the revised manuscript, not the notebook filenames:

    Fig. 1  methodology flowchart          Section 3.7
    Fig. 2  F1 with 95% CI                 Section 4.1   (notebook Figure_5_*)
    Fig. 3  confusion matrices             Section 4.1   (notebook Figure_3_*)
    Fig. 4  GWO convergence                Section 4.2   (notebook Figure_2_*)
    Fig. 5  gradient variance              Section 4.4   (notebook Figure_6_*)
    Fig. 6  circuit-optimiser comparison   Section 4.4   (notebook Figure_7_*)
    Fig. 7  critical-difference diagrams   Section 4.6   (notebook Figure_8_*)
    Fig. 8  SHAP summaries                 Section 4.8   (notebook Figure_4_*)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from qmlgwo.evaluate import load_folds          # noqa: E402
from qmlgwo.diagnostics import fit_decay        # noqa: E402
from qmlgwo import viz                           # noqa: E402  (sets the house style)

import matplotlib                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch   # noqa: E402
from scipy import stats                          # noqa: E402

plt.rcParams.update({"figure.constrained_layout.use": False,
                     "font.family": "DejaVu Sans", "font.size": 9})

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG_IN = ROOT / "figures"
OUT = ROOT / "paper_figures"
OUT.mkdir(exist_ok=True)

DATASETS = [  # (notebook, key, display name) in panel order (a)-(d)
    ("A", "bank_churn", "Bank Customer Churn"),
    ("A", "bank_marketing", "Bank Marketing"),
    ("B", "hr_promotion", "HR Promotion"),
    ("B", "loan_approval", "Loan Approval"),
]
QUANTUM = ["VQC", "QNN", "QSVC", "QDT"]
LETTERS = "ABCDEFGH"
C_Q, C_C = viz.PALETTE[1], viz.PALETTE[0]           # quantum orange, classical blue
Q_ALPHA = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}
DPI = 600


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def save(fig, n: int) -> None:
    """Write 'Figure n.png' (600 dpi) and 'Figure n.pdf' (vector)."""
    png, pdf = OUT / f"Figure {n}.png", OUT / f"Figure {n}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(png).size
    print(f"  Figure {n}: {w} x {h} px -> {png.name}, {pdf.name}")


def load_all() -> pd.DataFrame:
    frames = []
    for nb in ("A", "B"):
        p = RES / f"notebook_{nb}" / "folds_all.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found -- run Notebook {nb} first.")
        frames.append(load_folds(p))
    return pd.concat(frames, ignore_index=True)


def best_classical(sub: pd.DataFrame) -> str:
    return sub[~sub.model.isin(QUANTUM)].groupby("model").f1.mean().idxmax()


def panel_label(ax, i: int, name: str) -> None:
    ax.set_title(f"({LETTERS[i]}) {name}", loc="left", fontsize=10, fontweight="bold")


# --------------------------------------------------------------------------- #
# Fig. 1 -- methodology flowchart                                              #
# --------------------------------------------------------------------------- #

def figure_1() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 8.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 115); ax.axis("off")

    def box(x, y, w, h, text, fc="#FFFFFF", ec="#1F3864", bold=False, fs=8.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", wrap=True, linespacing=1.25)

    def arrow(x1, y1, x2, y2, text=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.1))
        if text:
            ax.text((x1 + x2) / 2 + 1.2, (y1 + y2) / 2, text, fontsize=7.2,
                    va="center", color="#333333", style="italic")

    # outer and inner frames
    ax.add_patch(FancyBboxPatch((3, 11.5), 95.5, 81.5, boxstyle="round,pad=0.5,rounding_size=2",
                                fc="#EEF3FA", ec="#2E5496", lw=1.3, ls="--"))
    ax.text(5, 90.2, "Outer loop: repeated stratified 5-fold CV × 3 repeats = 15 folds",
            fontsize=8.4, fontweight="bold", color="#2E5496")
    ax.add_patch(FancyBboxPatch((7, 38), 86, 44, boxstyle="round,pad=0.5,rounding_size=2",
                                fc="#FFF7E6", ec="#B8860B", lw=1.2, ls="--"))
    ax.text(9, 79.6, "Inner loop on the outer-training partition only (3-fold CV)",
            fontsize=8.2, fontweight="bold", color="#8B6508")

    box(25, 100, 50, 9, "Public benchmark dataset\n(target-leaking columns removed)", fc="#D9E2F3", bold=True)
    arrow(50, 100, 50, 93.6)

    box(10, 84, 37, 5, "Outer-training partition", fc="#FFFFFF")
    box(58, 84, 32, 5, "Outer-test partition (held out)", fc="#F2F2F2")

    box(11, 62, 78, 14.5,
        "Preprocessing pipeline, re-fitted on each training fold:\n"
        "imputation  →  one-hot encoding (rare levels grouped)  →  random undersampling\n"
        "(training rows only)  →  standardisation  →  PCA (4) or LDA (C − 1 = 1)\n"
        "→  angle rescaling to [−π/2, π/2]  (quantum models)", fc="#FFFFFF", fs=7.6)
    arrow(28, 84, 28, 76.9)

    box(11, 43, 37, 15,
        "Grey Wolf Optimizer\nN = 8 wolves, T = 12 iterations\nbudget = 96 evaluations\n"
        "fitness = mean inner-CV F1", fc="#FFFFFF", fs=7.8)
    box(52, 43, 37, 15,
        "Model: VQC · QNN · QSVC · QDT\nor classical baseline\n"
        "circuit parameters trained by\nAdam / SPSA / COBYLA", fc="#FFFFFF", fs=7.8)
    arrow(30, 62.6, 30, 58.6)
    ax.annotate("", xy=(52, 50.5), xytext=(48.2, 50.5),
                arrowprops=dict(arrowstyle="<|-|>", color="#333333", lw=1.1))
    ax.text(50.1, 53.5, "propose /\nscore", fontsize=6.8, ha="center", style="italic")

    box(18, 26, 64, 7, "Refit selected configuration on the full outer-training partition",
        fc="#FFFFFF")
    arrow(30, 42.6, 36, 33.6, "best configuration")
    box(14, 14, 72, 8.5, "Score ONCE on the outer-test partition\n"
                         "F1 · MCC · ROC-AUC · balanced accuracy", fc="#E2EFDA")
    arrow(50, 25.6, 50, 23.1)
    # held-out test data routed along the right margin, outside the inner loop,
    # so the arrow never crosses a box
    ax.plot([90.4, 95.3, 95.3], [86.5, 86.5, 18.2], color="#333333", lw=1.1)
    ax.annotate("", xy=(86.6, 18.2), xytext=(95.3, 18.2),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.1))
    ax.text(96.2, 52, "held-out test data", rotation=270, fontsize=7.2,
            va="center", color="#333333", style="italic")

    box(12, 1, 76, 8, "Across 15 folds: mean ± s.d., 95% CI (t), Wilcoxon + Holm,\n"
                      "Cliff's δ, Friedman–Nemenyi", fc="#D9E2F3", bold=True, fs=8)
    arrow(50, 14.6, 50, 9.6)
    save(fig, 1)


# --------------------------------------------------------------------------- #
# Fig. 2 -- F1 with 95% confidence intervals                                   #
# --------------------------------------------------------------------------- #

def figure_2(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.2))
    for i, (_, _, name) in enumerate(DATASETS):
        ax = axes.flat[i]
        s = df[(df.dataset == name) & (df.reduction == "LDA") & (df.optimizer == "gwo")]
        g = s.groupby("model").f1.agg(["mean", "std", "count"]).sort_values("mean")
        half = stats.t.ppf(0.975, g["count"] - 1) * g["std"] / np.sqrt(g["count"])
        q = g.index.isin(QUANTUM)
        # Point estimate + CI (Cleveland dot plot) rather than bars: the axis is
        # zoomed to the data range, and truncated bars would exaggerate differences.
        y = np.arange(len(g))
        for yi, (m, row), h, isq in zip(y, g.iterrows(), half, q):
            ax.errorbar(row["mean"], yi, xerr=h, fmt="s" if isq else "o",
                        color=C_Q if isq else C_C, ecolor="#444444", elinewidth=0.9,
                        capsize=2.5, markersize=5.5, markeredgecolor="black",
                        markeredgewidth=0.5)
        ax.set_yticks(y, g.index)
        for lab, isq in zip(ax.get_yticklabels(), q):
            lab.set_fontweight("bold" if isq else "normal")
        lo = max(0.0, (g["mean"] - half).min() - 0.03)
        hi = min(1.0, (g["mean"] + half).max() + 0.03)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("F1 score (mean and 95% CI, 15 folds)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", visible=False)
        panel_label(ax, i, name)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="s", ls="none", color=C_Q, mec="black",
                               label="Quantum model (bold label)"),
                        Line2D([], [], marker="o", ls="none", color=C_C, mec="black",
                               label="Classical baseline")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.6, w_pad=1.6)
    save(fig, 2)


# --------------------------------------------------------------------------- #
# Fig. 3 -- confusion matrices                                                 #
# --------------------------------------------------------------------------- #

def figure_3(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(7.5, 7.6))
    im = None
    for r, (_, _, name) in enumerate(DATASETS):
        s = df[(df.dataset == name) & (df.reduction == "LDA") & (df.optimizer == "gwo")]
        models = QUANTUM + [best_classical(s)]
        for c, m in enumerate(models):
            ax = axes[r, c]
            t = s[s.model == m]
            cm = np.array([[t.tn.sum(), t.fp.sum()], [t.fn.sum(), t.tp.sum()]], dtype=float)
            rate = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            im = ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{rate[a, b]:.2f}\n({int(cm[a, b]):,})", ha="center",
                            va="center", fontsize=5.8,
                            color="white" if rate[a, b] > 0.55 else "black")
            ax.set_xticks([0, 1], ["Neg", "Pos"], fontsize=6.5)
            ax.set_yticks([0, 1], ["Neg", "Pos"], fontsize=6.5)
            ax.tick_params(length=0)
            ax.grid(False)
            ax.set_title(m + ("" if m in QUANTUM else " (best classical)"), fontsize=7.2,
                         fontweight="bold" if m in QUANTUM else "normal")
            if c == 0:
                ax.set_ylabel("True class", fontsize=7)
            if r == 3:
                ax.set_xlabel("Predicted class", fontsize=6.8)
    fig.subplots_adjust(left=0.13, right=0.88, hspace=0.62, wspace=0.28, top=0.96, bottom=0.07)
    for r, (_, _, name) in enumerate(DATASETS):          # dataset label per row
        bb = axes[r, 0].get_position()
        fig.text(0.015, (bb.y0 + bb.y1) / 2, f"({LETTERS[r]}) {name}", rotation=90,
                 ha="center", va="center", fontsize=8, fontweight="bold")
    cax = fig.add_axes([0.90, 0.25, 0.018, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Row-normalised rate", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    save(fig, 3)


# --------------------------------------------------------------------------- #
# Fig. 4 -- GWO convergence                                                    #
# --------------------------------------------------------------------------- #

def figure_4(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.2))
    for i, (_, _, name) in enumerate(DATASETS):
        ax = axes.flat[i]
        s = df[(df.dataset == name) & (df.reduction == "LDA") & (df.optimizer == "gwo")]
        for j, m in enumerate(QUANTUM + [best_classical(s)]):
            curves = [c for c in s[s.model == m].convergence if isinstance(c, list) and c]
            if not curves:
                continue
            L = max(len(c) for c in curves)
            P = np.array([c + [c[-1]] * (L - len(c)) for c in curves], dtype=float)
            mu, sd = P.mean(0), P.std(0, ddof=1)
            it = np.arange(1, L + 1)
            col = viz.PALETTE[(j + 1) % len(viz.PALETTE)] if m in QUANTUM else "black"
            ax.plot(it, mu, color=col, marker=viz.MARKERS[j], markersize=3.2, lw=1.2,
                    linestyle=viz.LINESTYLES[j % len(viz.LINESTYLES)],
                    label=m if m in QUANTUM else f"{m} (best classical)")
            ax.fill_between(it, mu - sd, mu + sd, color=col, alpha=0.10)
        ax.set_xlabel("GWO iteration", fontsize=8)
        ax.set_ylabel("Best inner-CV F1 so far", fontsize=8)
        ax.tick_params(labelsize=7.5)
        panel_label(ax, i, name)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, [x.split(" (")[0] if "(" in x else x for x in l[:4]] + ["Best classical"],
               loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.5, w_pad=1.5)
    save(fig, 4)


# --------------------------------------------------------------------------- #
# Fig. 5 -- gradient variance                                                  #
# --------------------------------------------------------------------------- #

def figure_5() -> None:
    p = RES / "notebook_A" / "barren_plateau_scan.csv"
    if not p.exists():
        p = RES / "notebook_B" / "barren_plateau_scan.csv"
    scan = pd.read_csv(p)
    fits = fit_decay(scan)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3))
    ax = axes[0]
    for i, (L, sub) in enumerate(scan.groupby("n_layers")):
        sub = sub.sort_values("n_qubits")
        col = viz.PALETTE[i % len(viz.PALETTE)]
        ax.semilogy(sub.n_qubits, sub.grad_var, marker=viz.MARKERS[i], color=col,
                    linestyle="none", markersize=4.5, label=f"L = {L}")
        r = fits[fits.n_layers == L]
        if len(r):
            r = r.iloc[0]
            xs = np.linspace(sub.n_qubits.min(), sub.n_qubits.max(), 50)
            ax.semilogy(xs, np.exp(r.intercept - r.decay_rate_alpha * xs), color=col, lw=1.0,
                        linestyle=viz.LINESTYLES[i % len(viz.LINESTYLES)])
    ax.set_xlabel("Number of qubits n", fontsize=8)
    ax.set_ylabel(r"Var[$\partial\langle Z_0\rangle/\partial\theta_k$]", fontsize=8)
    ax.legend(frameon=False, fontsize=7.5, title="Depth", title_fontsize=7.5)
    panel_label(ax, 0, "Scaling with qubit count")
    ax = axes[1]
    for i, (n, sub) in enumerate(scan.groupby("n_qubits")):
        sub = sub.sort_values("n_layers")
        ax.semilogy(sub.n_layers, sub.grad_var, marker=viz.MARKERS[i % len(viz.MARKERS)],
                    color=viz.PALETTE[i % len(viz.PALETTE)], markersize=4, lw=1.0,
                    linestyle=viz.LINESTYLES[i % len(viz.LINESTYLES)], label=f"n = {n}")
    ax.set_xlabel("Ansatz depth L (layers)", fontsize=8)
    ax.set_ylabel(r"Var[$\partial\langle Z_0\rangle/\partial\theta_k$]", fontsize=8)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    panel_label(ax, 1, "Scaling with depth")
    for a in axes:
        a.tick_params(labelsize=7.5)
    fig.tight_layout(w_pad=2.0)
    save(fig, 5)


# --------------------------------------------------------------------------- #
# Fig. 6 -- circuit-parameter optimisers                                       #
# --------------------------------------------------------------------------- #

def figure_6() -> None:
    rows = []
    for nb, name in (("A", "Bank Customer Churn"), ("B", "HR Promotion")):
        p = RES / f"notebook_{nb}" / "optimizer_trajectories.csv"
        if p.exists():
            t = pd.read_csv(p)
            if "final_loss_full" in t.columns:
                rows.append((name, t))
            else:
                print(f"  Figure 6: {p} is in the old format -- re-run Notebook {nb} first.")
    if not rows:
        print("  Figure 6: SKIPPED (no valid optimizer_trajectories.csv)")
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.5, 3.0 * len(rows)),
                             gridspec_kw={"width_ratios": [1.6, 1.0]}, squeeze=False)
    k = 0
    for r, (name, t) in enumerate(rows):
        curves = t[t.final_loss_full.isna()]
        # one fixed colour/order per optimiser across ALL panels, so a colour
        # never changes meaning between the line and bar panels
        order = ["ADAM", "COBYLA", "SPSA"]
        cmap = {o: viz.PALETTE[i] for i, o in enumerate(order)}
        lmap = {o: viz.LINESTYLES[i] for i, o in enumerate(order)}
        summ = (t.dropna(subset=["final_loss_full"]).set_index("optimizer")
                  .reindex([o for o in order if o in set(t.optimizer)]).reset_index())
        ax = axes[r, 0]
        for opt in [o for o in order if o in set(curves.optimizer)]:
            sub = curves[curves.optimizer == opt].sort_values("step")
            n = int(sub.step.max()) + 1
            ax.plot(sub.progress, sub.loss, color=cmap[opt], lw=1.3,
                    linestyle=lmap[opt],
                    label=f"{opt} ({n} {'evaluations' if opt == 'COBYLA' else 'epochs'})")
        ax.set_xlabel("Fraction of optimisation run", fontsize=8)
        ax.set_ylabel("Mean binary cross-entropy", fontsize=8)
        ax.legend(frameon=False, fontsize=7)
        panel_label(ax, k, f"{name}: convergence"); k += 1
        ax = axes[r, 1]
        bars = ax.bar(summ.optimizer, summ.final_loss_full, edgecolor="black", linewidth=0.4,
                      color=[cmap[o] for o in summ.optimizer])
        for b, f1 in zip(bars, summ.final_f1):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"F1 {f1:.3f}",
                    ha="center", va="bottom", fontsize=7)
        ax.axhline(np.log(2), color="#555555", lw=0.9, ls="--")
        ax.text(-0.42, np.log(2) * 1.02, "ln 2 = constant predictor",
                fontsize=6.5, ha="left", va="bottom", color="#555555")
        ax.set_ylabel("Full-batch cross-entropy\nafter training", fontsize=8)
        ax.set_ylim(0, max(summ.final_loss_full.max(), np.log(2)) * 1.22)
        panel_label(ax, k, "final loss"); k += 1
        for a in axes[r]:
            a.tick_params(labelsize=7.5)
    fig.tight_layout(h_pad=1.8, w_pad=1.8)
    save(fig, 6)


# --------------------------------------------------------------------------- #
# Fig. 7 -- critical-difference diagrams                                       #
# --------------------------------------------------------------------------- #

def _draw_cd(ax, ranks: pd.Series, cd: float, title: str, i: int) -> None:
    names, r = list(ranks.index), ranks.values
    k = len(r)
    lo, hi = 1, k
    ax.set_xlim(lo - 0.3, hi + 0.3); ax.set_ylim(-k - 1.3, 2.0); ax.axis("off")
    ax.plot([lo, hi], [0.6, 0.6], color="black", lw=1)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [0.6, 0.85], color="black", lw=0.9)
        ax.text(t, 1.05, str(t), ha="center", fontsize=7)
    for j, (nm, rv) in enumerate(zip(names, r)):
        y = -j - 0.4
        side_left = j < (k + 1) // 2
        xe = lo - 0.15 if side_left else hi + 0.15
        col = C_Q if nm in QUANTUM else "#333333"
        ax.plot([rv, rv, xe], [0.6, y, y], color=col, lw=0.9)
        ax.text(xe - 0.05 if side_left else xe + 0.05, y, f"{nm} ({rv:.2f})",
                ha="right" if side_left else "left", va="center", fontsize=6.8,
                color=col, fontweight="bold" if nm in QUANTUM else "normal")
    yc, last = -k - 0.4, -1
    for a in range(k):
        b = a
        while b + 1 < k and r[b + 1] - r[a] <= cd:
            b += 1
        if b > a and b > last:
            ax.plot([r[a] - 0.03, r[b] + 0.03], [yc, yc], color="black", lw=2.6,
                    solid_capstyle="butt")
            yc -= 0.35; last = b
    ax.plot([lo, lo + cd], [1.65, 1.65], color="black", lw=1.8)
    ax.text(lo + cd / 2, 1.8, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_title(f"({LETTERS[i]}) {title}", loc="left", fontsize=9.5, fontweight="bold", pad=14)


def figure_7(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.8))
    for i, (_, _, name) in enumerate(DATASETS):
        s = df[(df.dataset == name) & (df.reduction == "LDA") & (df.optimizer == "gwo")]
        top = s[~s.model.isin(QUANTUM)].groupby("model").f1.mean().nlargest(3).index.tolist()
        w = s[s.model.isin(QUANTUM + top)].pivot_table(index="fold", columns="model",
                                                        values="f1").dropna()
        ranks = w.rank(axis=1, ascending=False).mean().sort_values()
        k, n = w.shape[1], w.shape[0]
        cd = Q_ALPHA[k] * np.sqrt(k * (k + 1) / (6.0 * n))
        _draw_cd(axes.flat[i], ranks, cd, name, i)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    save(fig, 7)


# --------------------------------------------------------------------------- #
# Fig. 8 -- SHAP summaries (composed from the notebooks' images)               #
# --------------------------------------------------------------------------- #

def figure_8() -> None:
    from PIL import Image, ImageDraw, ImageFont
    paths = [FIG_IN / f"notebook_{nb}" / f"Figure_4_shap_summary_{key}.png"
             for nb, key, _ in DATASETS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("  Figure 8: SKIPPED -- missing " + ", ".join(str(m) for m in missing))
        return
    ims = [Image.open(p).convert("RGB") for p in paths]
    W = max(im.width for im in ims)
    H = max(im.height for im in ims)
    lab = int(0.07 * H)                                   # space for the panel label
    canvas = Image.new("RGB", (2 * W, 2 * (H + lab)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(lab * 0.62))
    except OSError:
        font = ImageFont.load_default()
    for i, (im, (_, _, name)) in enumerate(zip(ims, DATASETS)):
        x0, y0 = (i % 2) * W, (i // 2) * (H + lab)
        canvas.paste(im, (x0 + (W - im.width) // 2, y0 + lab))
        draw.text((x0 + int(0.02 * W), y0 + int(0.12 * lab)), f"({LETTERS[i]}) {name}",
                  fill="black", font=font)
    target_w = int(7.5 * DPI)                            # 7.5 in at 600 dpi
    if canvas.width > target_w:
        canvas = canvas.resize((target_w, int(canvas.height * target_w / canvas.width)),
                               Image.LANCZOS)
    canvas.save(OUT / "Figure 8.png", dpi=(DPI, DPI))
    canvas.save(OUT / "Figure 8.pdf", "PDF", resolution=DPI)
    print(f"  Figure 8: {canvas.width} x {canvas.height} px -> Figure 8.png, Figure 8.pdf")


# --------------------------------------------------------------------------- #

if __name__ == "__main__" or "get_ipython" in globals():
    print(f"results : {RES}\nfigures : {OUT}\n")
    df = load_all()
    print(f"loaded {len(df)} fold-level records, "
          f"{df.dataset.nunique()} datasets, failures = {(df.status != 'ok').sum()}\n")
    figure_1()
    figure_2(df)
    figure_3(df)
    figure_4(df)
    figure_5()
    figure_6()
    figure_7(df)
    figure_8()
    print("\nDone. Upload the 'Figure N.png' files to PeerJ as separate figure files.")
