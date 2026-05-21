# geldium-delinquency-prediction
 End-to-end credit delinquency risk model                   built for Tata iQ job simulation on Forage.                   XGBoost · SHAP · Python
# Geldium Credit Delinquency Prediction
**Tata iQ Data Analytics Job Simulation — Forage**

## What This Project Does
Predicts which credit customers are at risk of going delinquent
using an end-to-end machine learning pipeline built in Python.

## Results
| Metric | Score |
|--------|-------|
| AUC-ROC | 0.61 |
| Recall (delinquents caught) | 75% at threshold 0.79 |
| False Negatives | 4 of 16 |

## The Pipeline
1. **EDA** — Analysed 500 customer records across 19 variables
2. **Cleaning** — Fixed missing values, inconsistent labels, outliers
3. **Feature Engineering** — Created 5 composite payment behaviour features
4. **Modeling** — XGBoost classifier with SMOTE class balancing
5. **Evaluation** — Threshold tuning, confusion matrix, AUC-ROC
6. **Explainability** — SHAP values for per-customer explanations

## Key Findings
- No single feature correlates with delinquency above 0.05
- Risk emerges from combinations of variables, not individual ones
- Payment behaviour (6-month history) was the strongest signal
- Standard accuracy is misleading — an 84% accurate model 
  caught zero delinquents due to class imbalance

## Tech Stack
- Python 3.11
- pandas, numpy
- scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- SHAP
- Matplotlib

## How to Run
1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Open `Final_Model.ipynb` in Jupyter
4. Note: Dataset not included (Forage simulation data).
   Replace the load step with your own dataset path.

## SHAP Feature Importance
![SHAP Bar Chart](shap_bar.png)
![SHAP Direction Chart](shap_dot.png)

## Author
Vinith B. — [LinkedIn](your-linkedin-url-here)
