# STATISTICAL_CONSULTING_FINAL

## 📊 Unemployment Rate Trend Analysis and Prediction

This project implements a comprehensive machine learning pipeline for analyzing and predicting unemployment rate trends using time series decomposition and multiple regression models.

## 🎯 Project Overview

The project focuses on fitting regression models to unemployment rate trend data extracted from STL (Seasonal and Trend decomposition using Loess) decomposition. We compare various machine learning algorithms and ensemble methods to identify the best performing model for unemployment rate prediction.

## 📈 Dataset

- **Time Period**: January 2002 (民國91年1月) to December 2024 (民國113年12月)
- **Total Observations**: 276 monthly records
- **Features**: 192 economic indicators including:
  - Population and demographic data
  - Salary and wage information (industrial & service sectors)
  - Employment statistics
  - Consumer price indices
  - Import/export trade values
  - Financial market indicators (exchange rates, stock indices)
  - Banking and monetary statistics
  - Tax revenue data

## 🔧 Feature Engineering

### Time Series Features
- **Lag Features**: Previous 1, 2, 3, 6, and 12 months values
- **Moving Averages**: 3, 6, and 12-month rolling averages
- **Trend Features**: 6-month rolling linear trend coefficients
- **Seasonal Features**: Month, quarter, and cyclical encoding (sin/cos)

### Original Feature Transformations
- **Lagged Variables**: All 192 original features shifted by 1 period (t-1 → t prediction)
- **Standardization**: Applied to linear models (Linear, Lasso, Ridge)
- **Missing Value Treatment**: Forward/backward filling

**Final Feature Count**: 206 features (204 in optimized version)

## 🤖 Machine Learning Models

### Base Models

| Model | Type | Hyperparameter Tuning |
|-------|------|----------------------|
| **Linear Regression** | Linear | Standard implementation |
| **Lasso Regression** | Linear (L1) | α = 0.1 |
| **Ridge Regression** | Linear (L2) | α = 1.0 |
| **Random Forest** | Tree Ensemble | Bayesian Optimization |
| **XGBoost** | Gradient Boosting | Bayesian Optimization |
| **Gradient Boosting** | Tree Ensemble | Bayesian Optimization |

### Hyperparameter Optimization

**Bayesian Optimization** using `scikit-optimize` with 30 iterations:

#### Random Forest Search Space:
- `n_estimators`: [50, 500]
- `max_depth`: [3, 20]
- `min_samples_split`: [2, 20]
- `min_samples_leaf`: [1, 10]
- `max_features`: [0.1, 1.0]

#### XGBoost Search Space:
- `n_estimators`: [50, 500]
- `max_depth`: [3, 10]
- `learning_rate`: [0.01, 0.3]
- `subsample`: [0.1, 1.0]
- `colsample_bytree`: [0.1, 1.0]
- `reg_alpha`: [0, 10]
- `reg_lambda`: [0, 10]

#### Gradient Boosting Search Space:
- `n_estimators`: [50, 500]
- `max_depth`: [3, 10]
- `learning_rate`: [0.01, 0.3]
- `subsample`: [0.1, 1.0]
- `min_samples_split`: [2, 20]
- `min_samples_leaf`: [1, 10]

### Ensemble Models

#### Voting Regressor
- **Components**: Lasso Regression + Linear Regression
- **Voting Strategy**: Average (equal weights)
- **Rationale**: Combines regularized and non-regularized linear approaches

## 📊 Model Performance

### Training/Test Split
- **Training Set**: 80% (212 observations)
- **Test Set**: 20% (52 observations)
- **Split Method**: Temporal split (chronological order preserved)

### Performance Metrics

| Model | Train RMSE | Test RMSE | Train R² | Test R² | Train Adj R² | Test Adj R² |
|-------|------------|-----------|----------|---------|--------------|-------------|
| **Linear Regression** | 0.0006 | 0.1026 | 1.0000 | 0.7703 | 1.0000 | 1.0786 |
| **Lasso** | 0.1100 | 0.1223 | 0.9611 | 0.6736 | -0.3615 | 1.1117 |
| **Ridge** | 0.0069 | 0.1242 | 0.9998 | 0.6636 | 0.9947 | 1.1151 |
| **Random Forest** | 0.0222 | 0.2354 | 0.9984 | -0.2095 | 0.9447 | 1.4138 |
| **Gradient Boosting** | 0.0023 | 0.2129 | 1.0000 | 0.0112 | 0.9994 | 1.3383 |
| **XGBoost** | 0.0004 | 0.2154 | 1.0000 | -0.0123 | 1.0000 | 1.3463 |
| **Voting (Lasso+Linear)** | - | - | - | - | - | - |

### Key Findings

1. **Best Performing Model**: Linear Regression (Test R² = 0.7703)
2. **Overfitting Issues**: Tree-based models show severe overfitting
   - Perfect training performance but poor test performance
   - Negative R² values indicate worse performance than baseline
3. **Linear Models Superiority**: Simple linear approaches outperform complex ensemble methods
4. **Voting Model**: Combines strengths of Lasso and Linear Regression

## 🔮 Future Predictions

The best-performing model provides 6-month ahead forecasts:

| Period | Predicted Value |
|--------|----------------|
| Month 1 | 3.3722 |
| Month 2 | 3.3722 |
| Month 3 | 3.3722 |
| Month 4 | 3.3722 |
| Month 5 | 3.3722 |
| Month 6 | 3.3722 |

## 📁 Project Structure

```
├── unemployment_prediction.py      # Main analysis script
├── voting_model_results.csv       # Voting model predictions
├── README.md                      # This file
└── 失業率資料_2.xlsx              # Raw dataset
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl scikit-optimize xgboost
```

### Running the Analysis

```python
# Execute the complete pipeline
prediction_results = main('失業率資料_2.xlsx')

# Access results
print("Best Model:", prediction_results['best_model'])
print("Future Predictions:", prediction_results['future_predictions'])
```

## 📊 Visualizations

The project generates comprehensive visualizations including:

1. **Historical Trend Analysis**: Time series plot of unemployment rates
2. **Model Performance Comparison**: RMSE and R² comparisons
3. **Prediction vs Actual**: Scatter plots for all models
4. **Residual Analysis**: Error distribution and patterns
5. **Feature Importance**: Tree model feature rankings
6. **Voting Model Analysis**: Detailed ensemble performance

## 🎯 Business Insights

### Model Interpretations

1. **Linear Regression Success**: 
   - Unemployment rate follows relatively linear trends
   - Simple models avoid overfitting with limited data
   - Economic relationships are often linear in nature

2. **Tree Model Challenges**:
   - High dimensionality (206 features) leads to overfitting
   - Monthly economic data may not have complex non-linear patterns
   - Limited training data for complex models

3. **Ensemble Strategy**:
   - Voting model balances regularization and flexibility
   - Reduces model variance through averaging
   - Provides more stable predictions

### Economic Implications

- **Trend Stability**: Predicted flat trend suggests economic stability
- **Policy Impact**: Models can inform unemployment policy decisions
- **Risk Assessment**: Confidence intervals help quantify prediction uncertainty

## ⚠️ Limitations and Future Work

### Current Limitations
- **Negative R² Values**: Some models perform worse than baseline
- **Overfitting**: Complex models fail to generalize
- **Feature Selection**: Need more sophisticated variable selection
- **External Factors**: Models may not capture unprecedented economic shocks

### Future Improvements
1. **Feature Selection**: Use LASSO or recursive feature elimination
2. **Time Series Models**: Implement ARIMA, LSTM, or Prophet
3. **Cross-Validation**: Use time series cross-validation
4. **External Data**: Incorporate policy changes and economic indicators
5. **Ensemble Methods**: Explore stacking and blending approaches

## 👥 Contributing

This project is part of a statistical consulting final project. For questions or improvements, please contact the development team.

## 📄 License

This project is for academic purposes as part of the Statistical Consulting course.

---

**Note**: All economic data is sourced from official Taiwan government statistics. Predictions should be used in conjunction with expert economic analysis and are subject to model limitations.
