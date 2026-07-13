# Phishing URL Detection using Machine Learning 🔒

A comparative study of classical machine learning algorithms for detecting phishing URLs. The goal: benchmark a range of models on the same feature set to identify which performs best at separating legitimate URLs from phishing attempts.

---

## Problem

Phishing URLs are one of the most common attack vectors online. This project frames phishing detection as a binary classification problem and systematically compares how different ML algorithms perform on it — measuring not just accuracy but also F1, precision, and recall, since a phishing detector's false-negative rate matters as much as its raw accuracy.

---

## Dataset

**11,054 URLs** described by **30 engineered features**, with a roughly balanced split between phishing and legitimate samples. Features capture structural and reputation signals of a URL, including:

- `PrefixSuffix-` — presence of a hyphen in the domain
- `SubDomains` — number of subdomains
- `HTTPS` — use of HTTPS
- `AnchorURL` — proportion of anchor tags pointing off-domain
- `WebsiteTraffic` — traffic-rank signal
- ...and 25 more URL-based features

Target: `class` (phishing vs. legitimate).

---

## Approach

1. **Exploratory analysis** — correlation heatmap, pairplots on key features, and class-balance check.
2. **Train/test split** — 80/20.
3. **Model comparison** — trained and evaluated a broad set of classifiers on identical features.
4. **Metrics** — accuracy, F1, precision, and recall for each model, plus per-model hyperparameter sweeps (e.g. `n_neighbors` for KNN, `max_depth` for tree-based models, `learning_rate` for boosting).

---

## Models Compared & Results

Test-set performance (accuracy and F1 on the 20% hold-out):

| Model | Accuracy | F1 |
|---|---|---|
| **Gradient Boosting** | **0.974** | **0.977** |
| Random Forest | 0.967 | 0.971 |
| Support Vector Machine | 0.964 | 0.968 |
| K-Nearest Neighbors | 0.960 | 0.964 |
| Decision Tree | 0.958 | 0.963 |
| Logistic Regression | 0.934 | 0.941 |
| Naive Bayes | 0.605 | 0.454 |

**XGBoost**, **Multi-layer Perceptron (MLP)**, and **CatBoost** were also explored in the notebook.

**Takeaway:** Ensemble methods dominated. **Gradient Boosting** was the top performer at ~97.4% test accuracy, with Random Forest close behind — consistent with the general finding that gradient-boosted trees excel on tabular, feature-engineered data. Naive Bayes underperformed badly, as expected given its strong feature-independence assumption on correlated URL features.

---

## Tech Stack

- **Python**
- **scikit-learn** — Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, MLP
- **XGBoost / CatBoost** — additional boosting models
- **pandas / NumPy** — data handling
- **Matplotlib / Seaborn** — EDA and visualization

---

## Running It

```bash
pip install pandas numpy scikit-learn xgboost catboost matplotlib seaborn
python "phishing_urls_ml (1).py"
```

Or open the original notebook in Google Colab / Jupyter.

---

## Note

This project focuses on **classical ML** with engineered URL features. I later built a separate **deep learning** approach to the same problem — a Bi-LSTM sequence model operating on raw URL characters — which is published in IJSDR.
