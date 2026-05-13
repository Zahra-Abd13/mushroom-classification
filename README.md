# 🍄 Mushroom Edibility Classification

> Binary classification study predicting whether a mushroom is **edible** or **poisonous** using supervised learning.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [How to Run](#-how-to-run)
- [Dependencies](#-dependencies)


---

## 🔍 Project Overview

Mushroom foraging carries real danger — many toxic species closely resemble edible ones. This project builds a supervised classification pipeline to predict mushroom edibility from physical characteristics such as cap shape, gill color, stalk surface, and spore print color.

**Task type:** Binary Classification  
**Target:** `edible (0)` vs `poisonous (1)`  

### Key challenges addressed:
- All 22 features are **categorical** — encoding strategy required
- `odor` was **intentionally dropped** per spec (near-perfect single predictor; removal forces genuine multi-feature learning)
- `stalk-root` has **~30% missing values** — handled with mode imputation
- `veil-type` has **zero variance** — dropped as it carries no discriminative information

---

## 📦 Dataset

| Property | Detail |
|---|---|
| **Name** | Mushroom Dataset |
| **Source** | [UCI ML Repository – ID 73](https://archive.ics.uci.edu/dataset/73/mushroom) |
| **Instances** | 8,124 |
| **Features** | 22 categorical (19 retained after cleaning) |
| **Target** | Binary: edible (`e` → 0) / poisonous (`p` → 1) |
| **Class balance** | 51.8% edible / 48.2% poisonous — no resampling needed |
| **Access** | Loaded programmatically via `ucimlrepo` (no manual download required) |

---

## 📁 Project Structure

```
mushroom-classification/
│
├── notebook
│       │
│     Mushroom_Classification.ipynb   # Main notebook — runs end-to-end
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

The notebook is self-contained. It fetches the dataset automatically, runs all preprocessing, trains three classifiers, and produces all evaluation plots.

---

## 🔬 Methodology

### Preprocessing Pipeline

| Step | Action | Justification |
|---|---|---|
| Missing values | Mode imputation on `stalk-root` | Avoids 30% data loss; standard for high-missingness categoricals |
| Feature removal | Dropped `odor` | Near-perfect single predictor; required by spec |
| Feature removal | Dropped `veil-type` | Zero variance — no discriminative information |
| Target encoding | `e → 0`, `p → 1` (fixed mapping) | Deterministic; not reliant on LabelEncoder sort order |
| Feature encoding | `LabelEncoder` on all 19 remaining columns | Low-cardinality features (2–12 values); OHE tested and rejected (5× dimensionality, no accuracy gain) |

### Train / Validation / Test Strategy

- **80% training set** — model fitting + cross-validated hyperparameter tuning  
- **20% test set** — final evaluation only, never seen during tuning  
- **5-fold Stratified CV** used on training data to select optimal hyperparameters  
- **kNN StandardScaler** wrapped inside a `Pipeline` to prevent data leakage across CV folds

### Models

| # | Classifier | Family | Notes |
|---|---|---|---|
| 1 | Logistic Regression | Linear | Baseline; `max_iter=1000` |
| 2 | Decision Tree | Tree-based | `max_depth` tuned via 5-fold CV; `min_samples_leaf=5` |
| 3 | k-Nearest Neighbours | Instance-based | `k` tuned via 5-fold CV; `weights='distance'`; scaler inside Pipeline |

### Feature Importance

Chi-square (χ²) test and Mutual Information were used — both are statistically correct for nominal categorical features (unlike Pearson correlation, which assumes continuous linearly-related variables).

---

## 📊 Results

### Test Set Performance (20% hold-out)

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~96% | ~96% | ~0.99 |
| Decision Tree | ~99% | ~99% | ~0.99 |
| kNN | ~99% | ~99% | ~0.99 |

### ✅ Recommended Model: Decision Tree

The Decision Tree is recommended because:
1. **Matches top accuracy** — equal to kNN, better than Logistic Regression
2. **Fully interpretable** — every prediction traces to explicit if-then rules; no black box
3. **Feature importance available** — Gini splits confirm which features drive classification
4. **Safety-critical domain** — interpretability is paramount when predictions affect human health

### Error Analysis

- **False Negatives** (poisonous → predicted edible) are the critical failure type in this domain
- All three models keep FN rates very low (<2% of test set)
- Primary misclassification cause: ambiguous feature combinations after `odor` removal

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mushroom-classification.git
cd mushroom-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook Mushroom_Classification.ipynb
```

**Or open in VS Code / JupyterLab** — the notebook runs end-to-end from top to bottom with no manual steps. The dataset is fetched automatically from the UCI ML Repository.

> ⚠️ Requires an internet connection for the first run (dataset download via `ucimlrepo`).

---

## 📦 Dependencies

```
numpy
pandas
plotly
scikit-learn
scipy
ucimlrepo
jupyter
```

See `requirements.txt` for pinned versions.

---

## 📄 License

This project was developed for academic purposes. The dataset is publicly available via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/73/mushroom).
