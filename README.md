# Credit Card Fraud Detection System

A machine learning system for detecting fraudulent credit card transactions, paired with a cost-benefit analysis quantifying the business impact of deploying the model — built as a capstone project by Saydain Sheikh.

## Overview

Credit card fraud — driven by card skimming and online phishing attacks — causes significant financial losses for banks. Many institutions still rely on manual fraud detection systems, which struggle to catch the small fraction of transactions that are actually fraudulent.

This project builds a machine learning model to automatically flag fraudulent transactions on unseen data and evaluates the model not just on predictive performance, but on its real, quantified savings to the business.

## Objective

Build a credit card fraud detection system that reduces losses from fraudulent transactions by replacing (or augmenting) manual review with a trained classifier capable of catching fraud cases that a rules-based or manual process would miss.

## Background

- A machine learning model was trained on historical transaction data to learn the patterns that distinguish fraudulent from legitimate transactions.
- The trained model was used to predict fraud on unseen (test) data.
- A **Cost-Benefit Analysis** was conducted to translate model performance into dollar-value business impact, supporting the deployment decision.

## Dataset

Two files sourced from Kaggle's [Credit Card Transactions Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection):
- `fraudTrain.csv` — 1,296,675 transactions (Jan 2019 – Dec 2020), 23 raw columns, used for model training/selection
- `fraudTest.csv` — held-out unseen transactions used for final model evaluation

Combined, the two files span 24 months of transaction history and only **7,506 of 1,296,675** training transactions (~0.58%) are fraudulent — a severe class imbalance that shapes the entire modeling approach.

## Feature Engineering

Starting from 23 raw columns (transaction time, card number, merchant, cardholder name/address, job, date of birth, exact lat/long, etc.), the following features were engineered and used for modeling:

| Feature | Description |
|---|---|
| `amt` | Transaction amount (log-transformed to correct right-skew) |
| `gender` | Cardholder gender, encoded binary (F=1, M=0) |
| `city_pop` | Cardholder's city population (log-transformed to correct right-skew) |
| `Age` | Computed from `dob` and transaction date |
| `Day_of_Week` | Day of week the transaction occurred (Mon=1 … Sun=7) |
| `Month` | Month the transaction occurred |
| `Dist` | Haversine distance (km) between the cardholder's location (`lat`/`long`) and the merchant's location (`merch_lat`/`merch_long`) |

All raw identifying/location columns (names, addresses, exact coordinates, job, timestamps) were dropped after feature extraction, leaving 7 predictive features plus the `is_fraud` target.

## Exploratory Data Analysis (EDA)

Key findings from the EDA phase:

- **Severe class imbalance**: Fraudulent transactions make up under 1% of all transactions, making fraud detection inherently difficult — the model must learn to identify rare events within a large, mostly-legitimate transaction volume.
- **Customer demographics**: The dataset skews male, with more male customers than female customers.
- **Amount vs. City Population**: A scatter plot of transaction amount against the cardholder's city population showed no clear relationship between the two features.
- **Distribution skew**: Both `amt` and `city_pop` were right-skewed (non-Gaussian) and were log-transformed to reduce skewness before modeling.
- **Correlation analysis**: A correlation heat map revealed a modest correlation (0.12) between transaction `amt` and `is_fraud` — some signal, but not enough to rely on alone.

## Methodology

1. **Train/Test Split**: 80/20 stratified split (to preserve the fraud/non-fraud ratio in both sets), followed by `StandardScaler` normalization of all numeric features.
2. **Models Compared**: **Logistic Regression**, **Random Forest**, and **XGBoost** (best number of estimators for XGBoost found via a manual search — 27 trees).
3. **Class Imbalance Handling** — each of the 3 models was trained under **4 different sampling strategies** to isolate the effect of imbalance handling:
   - **Imbalanced** (raw data, no resampling)
   - **Under-Sampling** (majority class reduced to match minority class)
   - **Over-Sampling** (minority class randomly duplicated)
   - **SMOTE** (Synthetic Minority Over-sampling Technique — synthesizes new minority-class samples)
4. **Model Selection Metric**: **Recall** was prioritized over accuracy — in fraud detection, a missed fraud case (false negative) is far more costly than a false alarm, and with <1% fraud prevalence, accuracy alone is a misleading metric (a model that predicts "no fraud" every time would still be ~99% accurate).

## Model Evaluation

### Model Comparison (validation set, all 12 model × sampling combinations)

| Model | Recall (Train) | Recall (Test) | AUC |
|---|---|---|---|
| Logistic Regression – Imbalanced | 0.000 | 0.000 | 0.84 |
| Logistic Regression – Under-Sampling | 0.744 | 0.750 | 0.75 |
| Logistic Regression – Over-Sampling | 0.774 | 0.781 | 0.84 |
| Logistic Regression – SMOTE | 0.776 | 0.786 | 0.84 |
| Random Forest – Imbalanced | 0.904 | 0.353 | 0.89 |
| Random Forest – Under-Sampling | 0.826 | 0.821 | 0.90 |
| Random Forest – Over-Sampling | 0.896 | 0.837 | 0.96 |
| **Random Forest – SMOTE** | **0.895** | **0.916** | **0.95** |
| XGBoost – Imbalanced | 0.364 | 0.316 | 0.97 |
| XGBoost – Under-Sampling | 0.906 | 0.852 | 0.92 |
| XGBoost – Over-Sampling | 0.977 | 0.834 | 0.97 |
| XGBoost – SMOTE | 0.963 | 0.960 | 0.96 |

**Takeaway**: Training on the raw imbalanced data alone produced poor or zero recall across all three model families — the models simply defaulted toward predicting "not fraud." Applying SMOTE consistently improved recall and generalization from train to test, and **Random Forest – SMOTE** was carried forward as the leading candidate based on its strong, consistent test recall (0.916) and AUC (0.95).

### Final Evaluation on Held-Out Unseen Data (`fraudTest.csv`)

The top-performing model/sampling combinations from validation were re-tested on the fully held-out `fraudTest.csv` set:

| Model | Recall | Confusion Matrix `[[TN, FP], [FN, TP]]` |
|---|---|---|
| **Random Forest – SMOTE** | **0.999** | `[[58116, 495458], [3, 2142]]` |
| Random Forest – Over-Sampling | 0.107 | `[[449555, 104019], [1916, 229]]` |
| XGBoost – SMOTE | 0.046 | `[[518438, 35136], [2046, 99]]` |
| XGBoost – Over-Sampling | 0.041 | `[[519446, 34128], [2057, 88]]` |

**Random Forest – SMOTE was selected as the final production model**, achieving a **99.9% recall** on completely unseen data — correctly flagging 2,142 of 2,145 total fraud cases and missing only 3. This came at the cost of a high false-positive rate (495,458 legitimate transactions also flagged), which directly informs the cost-benefit analysis below: false positives cost a small, fixed customer-support fee, while false negatives (missed fraud) are far more expensive — so this tradeoff favors the business.

Notably, XGBoost — despite strong validation-set metrics — did not generalize to the fully unseen test set, underscoring that validation performance alone doesn't guarantee real-world generalization, and that final holdout testing is an essential step before deployment.

## Cost-Benefit Analysis

### Before Model Deployment
| Metric | Value |
|---|---|
| Average transactions per month | 77,183 |
| Average fraudulent transactions per month | 402 |
| Average amount per fraudulent transaction | $121.00 |

### After Model Deployment (Random Forest – SMOTE)
| Metric | Value |
|---|---|
| Monthly customer support cost for fraud flagged by the model | $459 |
| Cost of fraud left undetected by the model | $0.00 |
| **Net monthly savings from deployment** | **$48,183** |

The model catches fraudulent transactions with high enough recall that essentially no fraud losses go undetected, with the remaining cost limited to customer support overhead for handling flagged transactions — translating to substantial monthly savings for the bank.

## Key Takeaways
- Class imbalance is the central challenge in fraud detection; SMOTE was critical for giving every model family enough signal on the minority (fraud) class, and was the deciding factor in the final model choice.
- Optimizing for **recall** rather than raw accuracy is the right business objective here — a missed fraud case is far more expensive than a false positive requiring extra customer support.
- Strong validation metrics don't guarantee generalization — Random Forest–SMOTE and XGBoost–SMOTE looked comparable on the validation set (AUC 0.95 vs. 0.96), but only Random Forest held up on the truly unseen test set.
- The cost-benefit framing turns a technical metric (recall) into a business metric (dollars saved), which is what ultimately justifies a model's deployment.

## Repository Structure

```
credit-card-fraud-detection/
├── data/                     # fraudTrain.csv, fraudTest.csv (not committed if large/sensitive)
├── notebooks/
│   └── Credit_Card_Fraud_Detection_System_Python_Code.ipynb   # EDA, feature engineering, modeling, cost-benefit analysis
├── reports/
│   └── Credit_Card_Fraud_Detection_System_Capstone_Presentation.pptx
├── requirements.txt
└── README.md
```

*(Update this section to match your actual folder layout once the code is pushed to the repo.)*

## How to Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebooks/Credit_Card_Fraud_Detection_System_Python_Code.ipynb
```

**Core dependencies**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn` (for SMOTE), `statsmodels`, `matplotlib`, `seaborn`

## Appendix: Raw Data Attributes

Before feature engineering, `fraudTrain.csv` contained **1,296,675 rows across 23 columns**, including transaction timestamp, card number, merchant name/category, transaction amount, cardholder name/address/job/date of birth, exact cardholder and merchant lat/long, and the `is_fraud` label. These raw fields were transformed into the 7 modeling features described above (see [Feature Engineering](#feature-engineering)).

## Author
**Saydain Sheikh**
Credit Card Fraud Detection System — Capstone Project
