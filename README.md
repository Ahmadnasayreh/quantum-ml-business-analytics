# QML-GWO Business Analytics

Code for the article **"Optimizing Quantum Machine Learning for Business Analytics:
A Hybrid Grey Wolf Optimizer Framework"**.

The pipeline evaluates four quantum classifiers (VQC, data re-uploading QNN,
quantum-kernel SVC and a decision tree over a quantum embedding) against six
classical baselines on four public business datasets. A Grey Wolf Optimizer (GWO)
selects hyper-parameters inside repeated stratified nested cross-validation, with
all preprocessing fitted inside each training fold. Quantum circuits are simulated
with PennyLane.

---

## 1. Repository structure

```
QML-GWO-Business-Analytics/
├── README.md
├── requirements.txt                          exact package versions of the reported run
├── reproduce.sh                              one-command run (smoke test or full)
├── make_paper_figures.py                     builds manuscript Figures 1–8
├── Notebook_A_bank_churn_marketing.ipynb     Bank Customer Churn + Bank Marketing
├── Notebook_B_hr_loan.ipynb                  HR Promotion + Loan Approval
├── references.bib                            methodological references used in the code
├── qmlgwo/                                   the package used by both notebooks
│   ├── config.py        seeds, experiment presets (SMOKE / FULL), environment manifest
│   ├── data.py          dataset specifications and the leakage-free preprocessing pipeline
│   ├── quantum.py       feature maps, ansatz circuits, the four quantum classifiers, circuit trainers
│   ├── optim.py         Grey Wolf Optimizer, Particle Swarm Optimisation, Random Search
│   ├── evaluate.py      model registry, search spaces, nested repeated cross-validation
│   ├── stats.py         Wilcoxon, Holm, Cliff's δ, Friedman / Nemenyi, confidence intervals
│   ├── diagnostics.py   gradient-variance scan, circuit-optimiser comparison
│   └── viz.py           figures (600 dpi, PNG + PDF)
├── data/                                     place the four datasets here (see data/README.md)
└── results/                                  outputs of the reported run (see results/README.md)
```

Both notebooks have the same structure and differ only in the datasets they load.

---

## 2. Installation

The reported results were produced with **Python 3.10.11** on Windows.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
python -m ipykernel install --user --name qmlgwo
```

`requirements.txt` pins the versions recorded in the run manifest
(PennyLane 0.42.3, scikit-learn 1.7.2, NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3,
imbalanced-learn 0.14.2, XGBoost 3.2.0, SHAP 0.49.1, Matplotlib 3.10.9).

---

## 3. Data

The datasets are public and are not redistributed. `data/README.md` lists the
source of each one and the exact path at which the notebooks expect it.

---

## 4. Running

**Notebooks.** Open a notebook and run all cells. Start with `CONFIG = SMOKE`,
which exercises every cell in a few minutes, then set `CONFIG = FULL` for the
reported results. Every expensive section saves its output and skips itself on
a later run if that output exists, so an interrupted run resumes where it stopped.

**One command** (bash: Linux, macOS, or Git Bash / WSL on Windows):

```bash
./reproduce.sh smoke     # structural check
./reproduce.sh full      # the reported results
```

**Manuscript figures.** After both notebooks have finished:

```bash
python make_paper_figures.py
```

This writes `paper_figures/Figure 1.png` … `Figure 8.png` (and PDF) at 600 dpi.

**Run time.** The FULL configuration is computationally heavy. On a 28-thread
workstation the hyper-parameter search of a single outer fold took about
1,450–2,120 s for QSVC, VQC and QNN (manuscript Table 15); the complete notebooks
take many hours. Reduce `MODELS`, `gwo_n_wolves` / `gwo_n_iterations` (the budget
follows automatically) or `max_train_samples` for a faster run. Keep
`n_outer_repeats = 3`: the paired statistical tests need all 15 folds.

---

## 5. Protocol at a glance

| Setting | Value |
|---|---|
| Outer evaluation | stratified 5-fold cross-validation × 3 repeats (15 estimates) |
| Inner model selection | stratified 3-fold cross-validation on each outer-training partition |
| Training rows | capped at 1,200 per outer fold (seeded, stratified), then balanced |
| Class balancing | random undersampling, training rows only |
| Dimensionality reduction | LDA (1 component) or PCA (4 components), fitted per fold |
| Qubits | one per retained component: 1 (LDA) or 4 (PCA) |
| GWO | 8 wolves × 12 iterations, budget 96, Latin hypercube initialisation, early stopping |
| Search baselines | Particle Swarm Optimisation and Random Search at the same budget |
| Simulator | PennyLane `default.qubit`, analytic expectation values |

---

## 6. Outputs and the manuscript

| Output in `results/notebook_<A|B>/` | Manuscript |
|---|---|
| `folds_all.csv` | source of every table and figure; Supplemental Data S1 (A) and S2 (B) |
| `table_datasets.csv` | Table 2 |
| `table_gwo_settings.csv`, `table_search_space.csv` | Tables 3 and 4 |
| `table_results_<dataset>.csv`, `summary_<dataset>.csv` | Tables 5–8, Figure 2 |
| `table_gwo_effect.csv` | Table 9 |
| `table_optimizer_comparison.csv` | Table 10 |
| `barren_plateau_scan.csv`, `barren_plateau_fits.csv` | Table 11, Figure 5 |
| `optimizer_trajectories.csv` | Table 12, Figure 6 |
| `table_imbalance_ablation.csv` | Table 14 |
| `stats_*_<dataset>.csv` | Sections 4.1–4.6, Figure 7 |
| `shap_importance_<dataset>.csv` | Section 4.8, Figure 8 |

Tables 13 and 15 are aggregates of `folds_all.csv`.

The notebooks also write per-dataset panels to `figures/notebook_<A|B>/`.
Their file prefix corresponds to the manuscript figures as follows:

| Notebook file prefix | Manuscript figure |
|---|---|
| `Figure_5_performance_*` | Figure 2 |
| `Figure_3_confusion_matrices_*` | Figure 3 |
| `Figure_2_gwo_convergence_*` | Figure 4 |
| `Figure_6_barren_plateau` | Figure 5 |
| `Figure_7_optimizer_comparison` | Figure 6 |
| `Figure_8_critical_difference_*` | Figure 7 |
| `Figure_4_shap_summary_*` | Figure 8 |

Figure 1 (the evaluation protocol) is drawn by `make_paper_figures.py`.

---

## 7. Reproducibility

Every stochastic component draws from a named random stream spawned from
`MASTER_SEED = 20250917` (`qmlgwo/config.py`). Streams are independent, so
changing one component (for example the number of wolves) does not change the
fold assignment. Re-running the pipeline reproduces the fold assignment and every
seeded draw exactly; model scores agree to within about 0.005 F1 between
independent executions because of multithreaded floating-point arithmetic.
Report results from one complete execution and never mix files from separate runs.

Each results folder contains `config_and_environment.json` and `MANIFEST.json`
with the full configuration, the package versions and the hardware used.

---

## 8. Citation

If you use this code, please cite the associated article (citation details will be
added on publication).

---

## 9. License

Released under the MIT License (see `LICENSE`).
