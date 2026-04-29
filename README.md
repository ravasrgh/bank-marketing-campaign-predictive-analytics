# Targeted Telemarketing: Predicting Term Deposit Subscription Using Machine Learning for Banking Campaign Optimization

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-9ACD32?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Analysis-blueviolet?style=for-the-badge)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-Resampling-orange?style=for-the-badge)

---

## Table of Contents

- [Project Introduction](#project-introduction)
- [Business Problem Statement](#business-problem-statement)
- [Data Understanding](#data-understanding)
- [Analytical Workflow](#analytical-workflow--the-resampling-benchmark)
- [Evaluation Metrics & Optimization](#evaluation-metrics--optimization)
- [Key Insights (SHAP Analysis)](#key-insights-shap-analysis)
- [Business Recommendations](#business-recommendations)
- [How to Use](#how-to-use)
- [Conclusion](#conclusion)
- [Project Structure](#project-structure)

---

## Project Introduction

**PT Bank Sentral Merakyat (BSM)** is a commercial bank operating in the Indonesian market. Post-pandemic (2021-2023), Bank Indonesia aggressively raised its benchmark interest rate from 3.5% to 6.0% to curb inflation, making term deposits attractive again as an investment instrument.

BSM's Marketing Division launched a **telemarketing campaign** targeting both existing customers and new prospects to offer term deposit products. However, the current approach -- calling customers randomly from a contact list -- is expensive and inefficient:

| Problem | Consequence |
|---|---|
| **Wasted calls** to uninterested customers | Agent time and salary burned with no conversion |
| **Missed opportunities** on high-potential customers | Lost deposit margin revenue for an entire year |

The Data Science team was tasked by the Chief Marketing Officer (CMO) to build a predictive model that identifies **which customers are most likely to subscribe to a term deposit**, so that the telemarketing team can prioritize their calls and maximize conversion while minimizing costs.

*"In banking, the most expensive customer is not the one you call too many times. It's the one you never called at all."*

---

## Business Problem Statement

### The Core Question

**How can we predict whether a bank customer will subscribe to a term deposit through a telemarketing campaign, so that the Telemarketing Team can prioritize contacts to customers with the highest conversion probability?**

This is a **binary classification** problem:
- **Subscribe = Yes (1):** Customer agrees to open a term deposit
- **Subscribe = No (0):** Customer declines the offer

### The Cost of Errors - Why FN is Worse Than FP

In telemarketing campaign optimization, prediction errors have asymmetric financial consequences:

| Error Type | Scenario | Business Impact | Risk Level |
|---|---|---|---|
| **True Positive (TP)** | Predict: Subscribe, Actual: Subscribe | Relevant call, customer converted successfully | Safe |
| **True Negative (TN)** | Predict: No, Actual: No | Customer not contacted, no additional cost | Safe |
| **False Positive (FP)** | Predict: Subscribe, Actual: No | Wasted call, agent time burned | Moderate |
| **False Negative (FN)** | Predict: No, Actual: Subscribe | **Valuable customer missed, BSM loses deposit margin revenue** | Critical |

#### Why False Negatives Are Fatal

**False Negative = "The customer you never called."** BSM loses the entire annual deposit margin from a customer who would have said yes.

1. **Lost Revenue**: Each missed subscriber = Rp 500,000 in net margin per year (based on Rp 50M average deposit x 4% NIM x 25% BOPO efficiency)
2. **Cost of a Wasted Call**: Only Rp 50,000 per call (agent salary + VoIP + overhead)
3. **FN:FP Cost Ratio = 10:1**: Missing one willing customer costs as much as 10 wasted calls

### Cost Assumptions

```
Per-case cost assumptions used in ROI simulation:

  False Negative (FN) = Rp 500,000   Lost annual deposit margin from missed subscriber
  False Positive (FP) = Rp 50,000    Cost of one wasted telemarketing call
  True Positive  (TP) = Rp 50,000    Cost of call (but offset by margin gained)
  True Negative  (TN) = Rp 0         No additional cost
```

**Bottom line:** We'd rather be *overly cautious* (call a customer who declines) than *miss* a customer who would have subscribed. Minimizing False Negatives is the #1 priority.

---

## Data Understanding

### Dataset Overview

| Property | Value |
|---|---|
| **Source** | UCI Machine Learning Repository (Bank Marketing Dataset) |
| **Records** | 41,188 customers (41,176 after deduplication) |
| **Features** | 20 features + 1 target |
| **Target** | `y` (yes / no) |
| **Subscribe Rate** | ~11.3% Yes vs ~88.7% No |
| **Imbalance Ratio** | 1 : 7.8 |

### Feature Descriptions

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | `age` | Numerical | Customer age |
| 2 | `job` | Categorical | Type of job (admin, blue-collar, technician, etc.) |
| 3 | `marital` | Categorical | Marital status (married, single, divorced) |
| 4 | `education` | Ordinal | Education level (basic.4y to university.degree) |
| 5 | `default` | Categorical | Has credit in default? |
| 6 | `housing` | Categorical | Has housing loan? |
| 7 | `loan` | Categorical | Has personal loan? |
| 8 | `contact` | Categorical | Contact communication type (cellular, telephone) |
| 9 | `month` | Categorical | Last contact month |
| 10 | `day_of_week` | Categorical | Last contact day of the week |
| 11 | `duration` | Numerical | Last contact duration in seconds (**dropped -- leakage**) |
| 12 | `campaign` | Numerical | Number of contacts during this campaign |
| 13 | `pdays` | Numerical | Days since last contact from previous campaign (999 = never) |
| 14 | `previous` | Numerical | Number of contacts before this campaign |
| 15 | `poutcome` | Categorical | Outcome of previous campaign |
| 16 | `emp.var.rate` | Numerical | Employment variation rate (quarterly) |
| 17 | `cons.price.idx` | Numerical | Consumer price index (monthly) |
| 18 | `cons.conf.idx` | Numerical | Consumer confidence index (monthly) |
| 19 | `euribor3m` | Numerical | Euribor 3-month rate (daily) |
| 20 | `nr.employed` | Numerical | Number of employees (quarterly) |
| 21 | `y` | Target | Has the client subscribed to a term deposit? Yes/No |

### Data Cleaning Summary

| Issue | Action | Rationale |
|---|---|---|
| 12 duplicate rows | Removed | Exact duplicates across all columns |
| `duration` column | **Dropped** | Severe data leakage -- only available after call ends |
| `unknown` values (6 columns) | **Retained as category** | Carries predictive signal; dropping would lose ~8K rows |
| `campaign` outliers | Capped at 95th percentile (7) | High campaign pressure is counterproductive |
| `pdays = 999` | Retained as sentinel | Binary flag `was_contacted_before` engineered instead |

### Feature Engineering

| New Feature | Logic | Business Rationale |
|---|---|---|
| `was_contacted_before` | pdays != 999 = 1 | Binary flag: was the customer ever contacted before? |
| `contact_intensity` | previous / campaign | Ratio of past vs current campaign contacts |

---

## Analytical Workflow (The Resampling Benchmark)

### The Challenge

With an 88:11 imbalance ratio, a naive model that predicts "No" for everyone achieves 88.7% accuracy but catches zero actual subscribers. This is useless. We needed systematic resampling experiments to teach the model what a subscriber looks like.

### Experimental Design

We tested 6 algorithms x 5 resampling strategies = 30 combinations using 5-Fold Stratified Cross-Validation with F2-Score as the evaluation metric.

**Algorithms Tested:**

| Model | Selection Rationale |
|---|---|
| Logistic Regression | Linear baseline: fast, interpretable |
| Decision Tree | Non-linear baseline: captures feature interactions |
| Random Forest | Bagging ensemble: reduces overfitting from single trees |
| Gradient Boosting | Sequential boosting: strong at correcting prior errors |
| XGBoost | Optimized boosting: native sparse handling, L1/L2 regularization |
| LightGBM | Fast boosting: histogram-based splitting, efficient for high-cardinality categoricals |

**Resampling Strategies:**

| Strategy | Technique | How It Works |
|---|---|---|
| No Resampling | Baseline | Train on original imbalanced data |
| Over Sampling | `RandomOverSampler` | Duplicate minority samples randomly |
| SMOTE | `SMOTE` | Synthesize new minority samples via nearest neighbors |
| Under Sampling | `RandomUnderSampler` | Reduce majority samples randomly |
| Hybrid | `SMOTE + Tomek Links` | SMOTE creates synthetic samples + Tomek cleans boundaries |

### Final Model Selected

```
Algorithm  : LightGBM
Resampling : Random Over-Sampling (ROS)
Tuning     : RandomizedSearchCV (50 iterations, 5-Fold CV)
```

---

## Evaluation Metrics & Optimization

### Why F2-Score?

The F2-Score gives 2x weight to Recall over Precision. This means the model is penalized more heavily for missing actual subscribers (FN) than for making wasted calls (FP). This is aligned with our business priority of catching every potential subscriber.

**Why not pure Recall?** Maximizing Recall alone would push the model to predict *everyone* as "Subscribe", 100% Recall but ~0% Precision, which would mean calling all 41K customers -- defeating the purpose of a predictive model. F2-Score ensures balance while still prioritizing Recall.

### Dual Threshold Optimization

Beyond the standard F2-Score threshold, we implemented profit-based threshold optimization, searching for the classification threshold that maximizes total financial savings for BSM.

```
Standard threshold (F2-optimized)  : 0.52
Profit-optimized threshold         : 0.43
```

**The "Sweet Spot":** The F2-optimal threshold maximizes statistical performance, while the profit-optimal threshold maximizes the actual Rupiah savings. We recommend the profit-optimal threshold for production deployment.

### Final Model Performance

| Metric | Value | Interpretation |
|---|---|---|
| **F2-Score** | 0.5669 | Primary metric -- weighted towards catching subscribers |
| **Recall** | 63.4% | Proportion of actual subscribers detected |
| **Precision** | 39.9% | Proportion of predicted subscribers that are real |
| **ROC-AUC** | 0.8050 | Strong discrimination ability across all thresholds |
| **Net Savings** | **Rp 138.2 juta** | Savings vs. no-model baseline (on test set of 7,873 customers) |
| **ROI** | **39.8%** | Return on investment from model deployment |
| **Calls Reduced** | **74.5%** | Only 25.5% of customers need to be contacted |

**Important Context:** With the profit-optimal threshold, BSM only needs to call ~2,000 out of 7,873 test customers while still catching 69.3% of actual subscribers. Extrapolated to the full dataset of 41,176 customers, potential savings reach **Rp 722.8 juta per campaign cycle**.

---

## Key Insights (SHAP Analysis)

SHAP (SHapley Additive exPlanations) analysis reveals which features **most influence** the model's subscription prediction:

### Top Predictive Features

1. **Euribor 3-Month Rate (`euribor3m`)**: The strongest predictor. Lower interbank rates correlate with higher subscription rates -- when traditional savings yield less, customers seek term deposits for better returns.

2. **Number of Employees (`nr.employed`)**: Macro-economic proxy. Lower employment numbers indicate economic uncertainty, pushing customers toward safe deposit instruments.

3. **Previous Campaign Outcome (`poutcome`)**: Customers who subscribed in a previous campaign have >65% conversion rate -- the ultimate "low-hanging fruit" for telemarketing.

4. **Contact Type (`contact`)**: Cellular contact achieves 2x higher conversion than landline. Mobile users are more accessible and responsive.

5. **Age (`age`)**: Students (<25) and retirees (>65) show the highest subscription rates, but for different reasons -- students seek first-time savings products, retirees seek stable income.

6. **Campaign Contacts (`campaign`)**: More is not better. Beyond 5-6 contacts, conversion rate drops -- aggressive calling is counterproductive.

These insights directly inform the business recommendations below. They're not just statistical curiosities, they're actionable intelligence for the Marketing and Telemarketing teams.

---

## Business Recommendations

### For the Telemarketing Operations Team

1. **Tiered Priority Calling List**
   - Tier 1: Customers with `poutcome=success` + cellular contact -- conversion rate >65%
   - Tier 2: Students/retirees with higher education, no active loans
   - Tier 3: All other customers predicted positive by the model

2. **Contact Limit Policy**
   - Enforce a maximum of 5-6 contacts per customer per campaign
   - Data proves that excessive calling is counterproductive

3. **Channel Strategy**
   - Prioritize cellular numbers over landline
   - Invest in updating customer mobile number database

### For the Marketing & Strategy Team

4. **Campaign Timing Optimization**
   - Concentrate campaign budget in March, September, October, December
   - These months historically show 2-3x higher conversion rates

5. **Macro-Economic Triggers**
   - Monitor JIBOR rates and employment indicators
   - When rates drop or employment weakens, intensify deposit campaigns -- customers are more receptive

6. **Segment-Specific Scripting**
   - Develop tailored scripts for Tier 1 (emphasize loyalty rewards) vs Tier 2 (emphasize security/stability)

### For the Technology Team

7. **Model Deployment & Monitoring**
   - Integrate model scoring into the CRM system for real-time priority ranking
   - Retrain every 6-12 months with fresh BSM customer data
   - Implement A/B testing: 80% model-guided, 20% random control group
   - Monitor for data drift using PSI (Population Stability Index)

---

## How to Use

### Prerequisites

```bash
pip install pandas numpy scikit-learn lightgbm xgboost imbalanced-learn shap matplotlib seaborn statsmodels missingno
```

### Loading the Saved Model

```python
import pickle
import json
import pandas as pd

# Load the trained model pipeline
model = pickle.load(open('bsm_final_model_v1.pkl', 'rb'))

# Load model metadata (features, thresholds, metrics)
with open('bsm_model_metadata_v1.json', 'r') as f:
    metadata = json.load(f)

print(f"Model: {metadata['model_name']} + {metadata['resampler']}")
print(f"F2 Threshold: {metadata['threshold_f2']}")
print(f"Profit Threshold: {metadata['threshold_profit']}")
```

### Making Predictions

```python
# Example: predict on new customer data
new_customer = pd.DataFrame({
    'age': [35],
    'job': ['admin.'],
    'marital': ['married'],
    'education': ['university.degree'],
    'default': ['no'],
    'housing': ['yes'],
    'loan': ['no'],
    'contact': ['cellular'],
    'month': ['oct'],
    'day_of_week': ['thu'],
    'campaign': [2],
    'pdays': [999],
    'previous': [0],
    'poutcome': ['nonexistent'],
    'emp.var.rate': [-1.1],
    'cons.price.idx': [93.994],
    'cons.conf.idx': [-36.4],
    'euribor3m': [0.879],
    'nr.employed': [5099.1],
    'was_contacted_before': [0],
    'contact_intensity': [0.0]
})

# Get probability
prob = model.predict_proba(new_customer)[:, 1]

# Apply profit-optimized threshold
threshold = metadata['threshold_profit']  # 0.43
prediction = (prob >= threshold).astype(int)

print(f"Subscribe Probability: {prob[0]:.2%}")
print(f"Prediction (threshold={threshold}): {'SUBSCRIBE' if prediction[0] else 'NO SUBSCRIBE'}")
```

### Running the Full Notebook

```bash
# 1. Clone the repository
git clone https://github.com/ravasrgh/bank-marketing-campaign-predictive-analytics.git
cd bank-marketing-campaign-predictive-analytics

# 2. Install dependencies
pip install pandas numpy scikit-learn lightgbm xgboost imbalanced-learn shap matplotlib seaborn statsmodels missingno

# 3. Open and run the notebook
jupyter notebook BSM_Bank_Marketing_FinalProject.ipynb

# Or open in VS Code with the Jupyter extension and click "Run All"
# Expected runtime: ~15-20 minutes (ML benchmarking + SHAP is compute-intensive)
```

---

## Conclusion

### What We Built

A machine learning-powered customer targeting system for BSM's term deposit telemarketing campaign that enables the Telemarketing Team to prioritize calls based on conversion probability, reducing wasted calls by 74.5% while still capturing 69.3% of potential subscribers.

### Key Results

| Metric | Value |
|---|---|
| Best Model | LightGBM + Random Over-Sampling (ROS) |
| ROC-AUC | 0.8050 (strong discriminative ability) |
| F2-Score | 0.5669 |
| Net Savings | Rp 138.2 juta on test set |
| ROI | 39.8% positive return |
| Projected Annual Savings | Rp 722.8 juta per campaign cycle |

### Financial Impact

```
Without Model:  Call all 7,873 customers = Rp 347M in wasted call costs
With Model:     Call only 2,007 customers = Rp 138.2M net savings

Extrapolated to full portfolio (41,176 customers): Rp 722.8 juta savings
```

### Limitations & Future Work

| Limitation | Proposed Solution |
|---|---|
| Dataset based on European data (2008-2013) | Re-train with actual BSM Indonesia customer data |
| Static threshold | Implement dynamic threshold recalibration per quarter |
| No customer lifetime value (CLV) | Incorporate CLV for more accurate ROI projection |
| Macro indicators highly correlated (VIF >40) | Monitor with VIF checks; consider PCA for linear models |
| Single-model approach | Explore stacking ensembles for improved performance |

---

## Project Structure

```
bank-marketing-campaign-predictive-analytics/
|
|-- README.md                                # This file
|-- BSM_Bank_Marketing_FinalProject.ipynb    # Main analysis notebook (A-G, fully documented)
|-- bank-additional-full.csv                 # Raw dataset (41,188 records)
|-- bsm_model_ready_v1.csv                  # Cleaned dataset for modeling (41,176 records)
|-- bsm_final_model_v1.pkl                  # Trained model pipeline (LightGBM + ROS)
|-- bsm_model_metadata_v1.json              # Model info, thresholds, metrics
```
