# Kaggle 50 Startups Profit Prediction: CRISP-DM Pipeline

This repository contains a complete, robust machine learning pipeline for predicting startup profitability based on budget allocations across R&D, Administration, and Marketing expenditures, alongside geographical location (State).

---

## 🚀 Key Results & Insights

*   **Optimal Model**: **Lasso Regression** trained on a 2-feature subset (`[R&D Spend, Marketing Spend]`) achieves peak generalizability.
    *   **Test Set $R^2$**: `94.74%`
    *   **Test Set MAE**: `$6,453.44`
*   **R&D Spend is the Dominant Driver**: Both linear coefficients and Random Forest feature importance verify that R&D investment is the single most critical predictor of profit ($92.71\%$ importance).
*   **Administration Spend is Noise**: Adding Administration spend to the model decreases model performance (RMSE increases by **11.4%**), proving it behaves as noise with respect to profit.
*   **State is a Proxy**: Dummy variables for state location (California, Florida, New York) show minimal impact, indicating geography has little direct causal effect on profitability.

---

## 📊 Performance Comparison (All 5 Algorithms)

Below is the performance comparison showing how 5 different algorithms behave as features are added sequentially:
1.  `[R&D Spend]`
2.  `[R&D Spend, Marketing Spend]`
3.  `[R&D Spend, Marketing Spend, New York]`
4.  `[R&D Spend, Marketing Spend, New York, Florida]`
5.  `[R&D Spend, Marketing Spend, New York, Florida, Administration]`

![All-in-One Performance Plot](./plots/allinone.png)

---

## 📁 Repository Structure

```text
├── data.csv                   # Kaggle 50 Startups source dataset
├── design.md                  # Project requirements and CRISP-DM design system
├── solve_50_startups.py       # Core pipeline script (EDA, tuning, report generation)
├── generate_summary_plot.py   # Code for annotated single-model summary plot
├── generate_allinone_plot.py  # Code for multi-model all-in-one comparisons
├── startup_profit_model.pkl   # Serialized Lasso Regression production pipeline
├── business_report.md         # Generated CRISP-DM business report
├── README.md                  # Project overview (this file)
├── hw6.md                     # Homework submission summary
└── plots/                     # Output directory for plots
    ├── histograms.png
    ├── scatter_plots.png
    ├── categorical_plots.png
    ├── correlation_heatmap.png
    ├── feature_selection_annotated.png
    └── allinone.png
```

---

## 🛠️ Installation & Execution

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### 2. Run the Core Machine Learning Pipeline
This command executes EDA, fits models across 5 experiments (E1 to E5) with GridSearchCV 5-fold CV, saves the best model to `startup_profit_model.pkl`, and creates `business_report.md`.
```bash
python solve_50_startups.py
```

### 3. Generate Comparative Plots
```bash
python generate_summary_plot.py
python generate_allinone_plot.py
```

---

## 🔮 Deployment & Inference
You can directly import and call `predict_profit` in Python:

```python
from solve_50_startups import predict_profit

# Predict profit for a startup:
# R&D Spend = $100k, Admin = $90k, Marketing = $200k, State = 'New York'
predicted = predict_profit(
    rd_spend=100000.0,
    admin_spend=90000.0,
    marketing_spend=200000.0,
    state='New York'
)
print(f"Predicted Profit: ${predicted:,.2f}")
```
