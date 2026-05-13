# Project Overview: Rossmann Sales Forecasting
This project aims to build a robust predictive pipeline for Rossmann, one of Europe’s largest drugstore chains. The core challenge is to forecast daily sales for over 1,100 stores across Germany, considering factors like promotions, competition, school holidays, and seasonality.

## Phase 1: Establishing the Baseline (Current)
As the initial stage of this project, I am implementing Facebook Prophet as our baseline model. Despite being a time-series forecasting problem, retail data is often heavily influenced by non-linear events and multiple seasonalities, which Prophet is specifically designed to handle.

### Why Prophet for the Baseline?

- Ease of Interpretability: It allows us to clearly decompose trends and seasonal effects (weekly, yearly).
- Handling Holidays: The model inherently supports holiday effects, which is crucial for Rossmann's promotional calendar.
- Robustness to Missing Data: It deals well with outliers and shifts in trends caused by events like store refurbishments or new competitors.

## Success Metrics
To evaluate the model's performance, we are primarily using **sMAPE (symmetric Mean Absolute Percentage Error)**, as it provides a balanced perspective on both over and under-forecasting errors across different store scales.

## Roadmap
[] Data Cleaning and EDA.

[ ] Baseline implementation (Prophet).

[ ] Hyperparameter tuning via MLflow.

[ ] Comparison with advanced Gradient Boosting models (XGBoost/LightGBM).