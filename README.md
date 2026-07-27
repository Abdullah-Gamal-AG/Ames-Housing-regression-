# [Project Name] — House Price Prediction

Predicting [target variable, e.g. sale price] using the [dataset name, e.g. Ames Housing] dataset with classical ML models (Linear/Ridge/Lasso, Random Forest, Gradient Boosting).

## Overview

- **Problem type:** Regression
- **Dataset:** [source/link, size — e.g. N rows, M features]
- **Target:** `SalePrice` (log-transformed via `log1p` for training, inverse-transformed for evaluation)
- **Best model:** [model name] — R² = [x.xxx], RMSE = [xxxx]

## Dataset

[1–2 sentences on where the data comes from and what it represents. Link to source if public.]

| | |
|---|---|
| Rows | [2030] |
| Features (raw) | [80] |
| Features (after encoding) | [258] |
| Missing data | [6.5]% before imputation |

## Pipeline

1. **Cleaning & imputation**
   - Zero-fill: features where missing = absence of the attribute (e.g. no garage → 0 garage area)
   - `"None"`-fill: categorical features where missing = absence of the feature
   - Median/mode fill: remaining numeric/categorical gaps
   - Special case: `Garage Yr Blt` imputed from `Year Built`

2. **Feature engineering**
   - `TotalSF` = basement + 1st floor + 2nd floor square footage
   - `TotalBath` = full baths + 0.5×half baths (incl. basement)
   - `Age` = year sold − year built
   - `RemodAge` = year sold − year remodeled
   - `TotalGarage` = garage cars × garage area

3. **Encoding**
   - One-hot encoding (drop-first) for nominal categoricals
   - Ordinal encoding for quality/condition scales (`Po`→`Ex`, `NA`→`Ex`, 1–10)

4. **Scaling**
   - `StandardScaler` applied before linear models (fit on train, applied to test)

5. **Models trained**
   - Linear Regression
   - Ridge (α = [1.0])
   - Lasso (α = [0.0005])
   - Random Forest (n_estimators = [100])
   - Gradient Boosting (GridSearchCV over [params])

6. **Evaluation**
   - Metrics: MSE, RMSE, MAE, R², 5-fold CV R²
   - Predictions inverse-transformed from log space before scoring

## Results

| Model | R² | RMSE | MAE | CV R² |
|---|---|---|---|---|
| Linear |0.8128 |38739.9374 |14750.2579 |0.8070 |
| Ridge |0.8252 |37432.3118 |14724.9747 |0.8191 |
| Lasso |0.8537 |34244.5129 |14704.9932 |0.8516 |
| Random Forest |0.9174 |25740.6202 |15593.7641 |0.8783 |
| Gradient Boosting |0.9347 |22872.5185 |13946.6845 |0.9045 |

[Fill in after running. Note which model won and why, in one sentence — don't editorialize.]
Gradient Boosting was the best-performing model, achieving the highest R² and CV R² while also obtaining the lowest RMSE and MAE among all evaluated models.
### Feature importance (Gradient Boosting)

Top features: [list top 5–10 from `importance.sort_values(ascending=False)`]
Overall Qual         0.353972
TotalSF              0.318387
TotalBath            0.051858
Kitchen Qual         0.033709
Central Air_Y        0.020560
## Visualizations

### Model Comparison
Shows the performance of all trained models across R², Cross-Validation R², MAE, and RMSE.
![Model Comparison](images/results.png)

---

### Actual vs Predicted Prices

Comparison between the predicted and actual house prices for each model.

![Prediction Scatter](images/predictions.png)

---

### Correlation Heatmap

Correlation matrix of the numerical features after preprocessing.

![Correlation Heatmap](images/correlation_heatmap.png)

---

### Feature Importance (Gradient Boosting)

The most influential features used by the best-performing Gradient Boosting model.

![Feature Importance](images/feature_importance.png)

---

### Prediction VS Actual

Scatter plots comparing the predicted and actual house prices for each model. A closer alignment to the diagonal line indicates better predictive performance.

![EDA](images/prediction_vs_actual.png)
## Project structure

```
.
├── data.csv                # raw dataset (not committed if large/licensed)
├── main.py                 # pipeline: load → clean → engineer → encode → train → evaluate
├── requirements.txt
└── README.md
└── figures
    ├── feature_importance.png
    ├── model_comparison.png
    ├── prediction_vs_actual.png
    ├── residuals.png
    └── missing_values.png
```

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
```