"""
qmlgwo.data
===========
Dataset loading and a **leakage-free** preprocessing pipeline.

Why the pipeline matters
------------------------
Fitting ``StandardScaler``, ``PCA`` or -- critically -- ``LDA`` on the *entire*
dataset before the train/test split leaks information: because LDA is
*supervised*, its discriminant direction would be computed using the test
labels, and every score would be optimistically biased.  Here,
all of encoding, balancing, scaling and dimensionality reduction live inside an
``imblearn.pipeline.Pipeline``, so each cross-validation fold re-fits them on
training data only.  Resamplers in an imblearn pipeline are applied during
``fit`` but bypassed during ``transform``/``predict``, which is exactly the
required behaviour: the test fold is never rebalanced.

Dimensionality of the reduced space
-----------------------------------
For binary targets LDA yields at most ``C - 1 = 1`` component.  The pipeline
enforces this explicitly and records the reduced dimension of every fold in
the ``n_reduced_features`` column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from .config import get_seed

# --------------------------------------------------------------------------- #
# 1. Dataset specifications                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class DatasetSpec:
    """Declarative description of one benchmark dataset.

    All four datasets are publicly available benchmarks; ``source`` records the
    provenance.
    """
    key: str
    display_name: str
    path: str
    target: str
    positive_label: object = 1
    drop_columns: Sequence[str] = field(default_factory=tuple)
    #: Columns removed to prevent *target leakage* (not merely uninformative).
    leakage_columns: Sequence[str] = field(default_factory=tuple)
    read_kwargs: dict = field(default_factory=dict)
    source: str = ""

    def load(self) -> tuple[pd.DataFrame, pd.Series]:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(
                f"{self.display_name}: no file at {path}. "
                "Set DATA_DIR / the spec's `path` to your local copy."
            )
        df = pd.read_csv(path, **self.read_kwargs)
        df.columns = [c.strip() for c in df.columns]

        if self.target not in df.columns:
            raise KeyError(f"{self.display_name}: target {self.target!r} not in "
                           f"{list(df.columns)}")

        drop = [c for c in (*self.drop_columns, *self.leakage_columns) if c in df.columns]
        y_raw = df[self.target]
        X = df.drop(columns=[self.target, *drop])

        # Binarise the target deterministically.
        if y_raw.dtype == object:
            y = (y_raw.astype(str).str.strip().str.lower()
                 == str(self.positive_label).strip().lower()).astype(int)
        else:
            y = (y_raw == self.positive_label).astype(int)

        X = X.loc[:, X.nunique(dropna=False) > 1]      # drop constant columns
        return X.reset_index(drop=True), y.reset_index(drop=True).rename("target")


def default_specs(data_dir: str = "data") -> dict[str, DatasetSpec]:
    """Specs for the four benchmarks, identical to those used in the notebooks.

    Expected layout (the original file names, one folder per dataset)::

        data/Bank Customer Churn Prediction/Bank Customer Churn Prediction.csv
        data/bank-marketing-uci/bank.csv
        data/HR Analytics Employee Promotion Data/train.csv
        data/Loan Prediction Problem Dataset/train_u6lujuX_CVtuZ9i.csv

    Bank Marketing (``bank.csv``) is read with ``load_bank_marketing_raw`` in the
    notebooks, which handles the semicolon separator and quoted lines.
    """
    d = Path(data_dir)
    return {
        "bank_churn": DatasetSpec(
            key="bank_churn", display_name="Bank Customer Churn",
            path=str(d / "Bank Customer Churn Prediction" / "Bank Customer Churn Prediction.csv"),
            target="churn", positive_label=1, drop_columns=("customer_id",),
            source="Kaggle: Bank Customer Churn Prediction (public)",
        ),
        "bank_marketing": DatasetSpec(
            key="bank_marketing", display_name="Bank Marketing",
            path=str(d / "bank-marketing-uci" / "bank.csv"),
            target="y", positive_label="yes",
            leakage_columns=("duration",),   # known only after the call ends
            source="UCI ML Repository: Bank Marketing (bank.csv, 10% subset, public)",
        ),
        "hr_promotion": DatasetSpec(
            key="hr_promotion", display_name="HR Promotion",
            path=str(d / "HR Analytics Employee Promotion Data" / "train.csv"),
            target="is_promoted", positive_label=1, drop_columns=("employee_id",),
            source="Kaggle/Analytics Vidhya: HR Analytics Employee Promotion (public)",
        ),
        "loan_approval": DatasetSpec(
            key="loan_approval", display_name="Loan Approval",
            path=str(d / "Loan Prediction Problem Dataset" / "train_u6lujuX_CVtuZ9i.csv"),
            target="Loan_Status", positive_label="Y", drop_columns=("Loan_ID",),
            source="Analytics Vidhya: Loan Prediction Problem (public)",
        ),
    }


# --------------------------------------------------------------------------- #
# 2. Preprocessing blocks                                                      #
# --------------------------------------------------------------------------- #


def make_column_transformer(X: pd.DataFrame, rare_threshold: float = 0.01) -> ColumnTransformer:
    """Impute + one-hot encode categoricals (rare levels merged), impute numerics.

    ``min_frequency=rare_threshold`` performs the rare-label grouping described
    in the manuscript, but does so *per fold*, so the category vocabulary is
    learned from training data only.
    """
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]

    cat_pipe = ImbPipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=rare_threshold, sparse_output=False)),
    ])
    num_pipe = ImbPipeline([("impute", SimpleImputer(strategy="median"))])

    return ColumnTransformer(
        [("cat", cat_pipe, cat), ("num", num_pipe, num)],
        remainder="drop", verbose_feature_names_out=False,
    )


def make_balancer(strategy: str, seed: int):
    """Return a resampler, or ``'passthrough'``.

    ``strategy`` is a first-class experimental factor (see the imbalance
    ablation in the notebooks) rather than a hard-coded choice.
    """
    if strategy == "undersample":
        return RandomUnderSampler(sampling_strategy="auto", random_state=seed)
    if strategy == "smote":
        return SMOTE(random_state=seed, k_neighbors=5)
    if strategy in ("none", "class_weight"):
        return "passthrough"
    raise ValueError(f"unknown balance strategy {strategy!r}")


def make_reducer(method: str, n_components: int, n_classes: int = 2):
    """PCA / LDA with the component count clipped to what is mathematically possible."""
    method = method.lower()
    if method == "pca":
        return PCA(n_components=n_components,
                   random_state=get_seed("cv_inner")), n_components
    if method == "lda":
        k = int(min(n_components, n_classes - 1))       # binary => k = 1
        return LDA(n_components=k, solver="svd"), k
    if method in ("none", "raw"):
        return "passthrough", -1
    raise ValueError(f"unknown reduction method {method!r}")


def build_pipeline(X: pd.DataFrame, estimator, *, reduction: str = "lda",
                   n_components: int = 1, balance: str = "undersample",
                   angle_encode: bool = True, seed: int = 0,
                   n_classes: int = 2) -> tuple[ImbPipeline, int]:
    """Assemble the full leakage-free pipeline.

    Order: encode -> balance(train only) -> standardise -> reduce -> angle-scale -> model.

    ``angle_encode`` maps every reduced feature to ``[-pi/2, pi/2]`` before it is
    written into a rotation gate.  Without it, standardised values beyond
    ``+/-pi`` wrap around the Bloch sphere and two very different samples can be
    encoded into nearly identical states.

    Returns
    -------
    (pipeline, n_reduced_features)
    """
    reducer, k = make_reducer(reduction, n_components, n_classes)
    steps = [
        ("encode", make_column_transformer(X)),
        ("balance", make_balancer(balance, seed)),
        ("scale", StandardScaler()),
        ("reduce", reducer),
    ]
    if angle_encode:
        steps.append(("angle", MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))))
    steps.append(("clf", estimator))
    return ImbPipeline(steps), k


def dataset_summary(X: pd.DataFrame, y: pd.Series, spec: DatasetSpec) -> dict:
    """Row for the manuscript's dataset table."""
    pos = int(y.sum())
    return {
        "Dataset": spec.display_name,
        "Samples": len(y),
        "Features (raw)": X.shape[1],
        "Numeric": X.select_dtypes(include=[np.number]).shape[1],
        "Categorical": X.shape[1] - X.select_dtypes(include=[np.number]).shape[1],
        "Positive class": pos,
        "Positive rate": round(pos / len(y), 4),
        "Imbalance ratio": round((len(y) - pos) / max(pos, 1), 2),
        "Leakage columns removed": ", ".join(spec.leakage_columns) or "--",
        "Source": spec.source,
    }


__all__ = ["DatasetSpec", "default_specs", "build_pipeline", "make_reducer",
           "make_balancer", "make_column_transformer", "dataset_summary"]
