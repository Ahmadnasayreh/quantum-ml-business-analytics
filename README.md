# Optimizing Quantum Machine Learning for Business Analytics: A Hybrid Grey Wolf Optimizer Framework

## Description

This repository contains the implementation code for the paper submitted to **PeerJ Computer Science**. The project applies hybrid quantum-classical machine learning models to business analytics classification tasks, optimized using the Grey Wolf Optimizer (GWO) and Sine Cosine Algorithm (SCA) for hyperparameter tuning.

Two Jupyter notebooks are provided:
- **`1st_2datasets_final.ipynb`** — Experiments on Dataset 1 (Bank Customer Churn) and Dataset 2 (Bank Marketing Portugal) using ZZFeatureMap with PCA/LDA, both with and without GWO optimization.
- **`other_2_datasets_Final.ipynb`** — Experiments on Dataset 3 (Telco Customer Churn) and Dataset 4 (Loan Approval Prediction) using the same pipeline with GWO optimization.

---

## Dataset Information

- **Dataset 1 — Bank Customer Churn**
  - Source: [Kaggle – Bank Customer Churn](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)
  - Task: Binary classification (churn prediction)

- **Dataset 2 — Bank Marketing Portugal**
  - Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) — DOI: 10.24432/C5K306
  - Task: Binary classification (term deposit subscription)

- **Dataset 3 — Telco Customer Churn**
  - Source: [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
  - Task: Binary classification (churn prediction)

- **Dataset 4 — Loan Approval Prediction**
  - Source: [Kaggle – Loan Approval Prediction](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
  - Task: Binary classification (loan approval)

All datasets are publicly available third-party datasets. No personally identifiable information is included. Datasets should be downloaded from the links above and placed in the same directory as the notebooks.

---

## Code Information

- **`1st_2datasets_final.ipynb`** — Quantum ML pipeline for Bank Customer Churn and Bank Marketing Portugal datasets
- **`other_2_datasets_Final.ipynb`** — Quantum ML pipeline for Telco Customer Churn and Loan Approval Prediction datasets

### Models Implemented

**Classical Baselines:**
- Classical SVC (CSVC)
- Classical Variational Classifier / MLP (CVC)
- Classical Neural Network (CNN)
- Classical Decision Tree (CDT)

**Quantum Models:**
- Quantum Support Vector Classifier (QSVC) — using quantum kernel via Qulacs
- Variational Quantum Classifier (VQC)
- Quantum Neural Network (QNN)
- Quantum Decision Tree (QDT)

**Optimization Algorithm:**
- Grey Wolf Optimizer (GWO) — used in both notebooks for hyperparameter tuning

**Dimensionality Reduction:**
- PCA (Principal Component Analysis) — 5 components
- LDA (Linear Discriminant Analysis) — 5 components

**Quantum Feature Encoding:**
- ZZFeatureMap — 4 qubits, 2 repetitions

---

## Usage Instructions

### 1. Clone or download this repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Download datasets

Download the four datasets from the URLs listed in the Dataset Information table above and place the CSV files in the same directory as the notebooks. Expected filenames:
- `Bank Customer Churn Dataset.csv`
- `bank-additional-full.csv`
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- `loan_approval_dataset.csv`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install qulacs
pip install imbalanced-learn
pip install shap
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### 4. Run notebooks

Launch Jupyter and open either notebook:

```bash
jupyter notebook
```

Run all cells in order. Each notebook is self-contained and will:
1. Load and preprocess the datasets
2. Apply PCA and LDA dimensionality reduction
3. Train classical baseline models
4. Apply the ZZFeatureMap quantum encoding
5. Train and optimize quantum models using GWO
6. Output performance metrics and convergence plots

---

## Requirements

```
python >= 3.8
qulacs >= 0.6.0
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
imbalanced-learn >= 0.9.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scipy >= 1.7.0
shap >= 0.40.0
jupyter >= 1.0.0
```

---

## Computing Infrastructure

- **Operating System:** Compatible with Linux (Ubuntu 20.04+), macOS, and Windows 10/11
- **Python Version:** 3.8 – 3.11
- **Hardware:** CPU-based execution (no GPU required). Experiments were conducted on a standard workstation with at least 8 GB RAM.
- **Quantum Simulation:** Qulacs (CPU-based quantum circuit simulator) — no physical quantum hardware required.

> **Note:** Quantum kernel computation in QSVC is computationally intensive. For large datasets, the code automatically applies undersampling (max 500 training samples for QSVC). Full runs may take several hours depending on hardware.

---

## Methodology

### Data Preprocessing
1. Categorical features are encoded using `LabelEncoder`.
2. All features are scaled using `StandardScaler` or `MinMaxScaler`.
3. Dimensionality reduction is applied: PCA (5 components) and LDA (5 components).

### Quantum Feature Encoding
Classical features are encoded into quantum states using the **ZZFeatureMap**:
- Applies Hadamard gates and RZ rotations based on input features.
- Adds ZZ entanglement interactions between adjacent qubits.
- Extracts quantum features from state amplitudes and Pauli-Z expectation values.

### Hyperparameter Optimization
- **GWO:** 25 wolves, 50 iterations, 3 continuous parameters per quantum model.
- Position updates follow the alpha-beta-delta hierarchy of the Grey Wolf pack.
- Objective function: F1-score (binary, positive class) on a held-out validation split.

### Evaluation
Models are evaluated using:
- Accuracy, Precision, Recall, F1-Score (binary, positive class)
- ROC-AUC Score
- Confusion Matrix

**Baseline Performance Analysis:** Classical models (CSVC, CVC, CNN, CDT) serve as baselines without optimization.

**Classical vs. Hybrid Quantum Baselines:** Quantum models (QSVC, VQC, QNN, QDT) with ZZFeatureMap are compared to classical baselines under both PCA and LDA dimensionality reduction.

**Comparative Analysis:** GWO-optimized quantum models are compared against non-optimized quantum models and classical baselines to evaluate the contribution of the optimization framework.

---

## Citations

If you use this code or datasets in your research, please cite the original dataset sources:

1. Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31. https://doi.org/10.24432/C5K306

2. Telco Customer Churn. IBM Watson Analytics Sample Data. Available at: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

3. Bank Customer Churn Dataset. Available at: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn

4. Loan Approval Prediction Dataset. Available at: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

---

## License

This code is released for academic and research purposes. Please refer to the individual dataset licenses on their respective source platforms (Kaggle, UCI ML Repository) for data usage terms.

---

## Contribution Guidelines

This repository is associated with an ongoing peer-reviewed publication. External contributions are not accepted at this time. For questions or issues, please contact the corresponding author.
