![Status](https://img.shields.io/badge/status-in%20development-yellow)

# Rossmann Sales Forecasting

This project builds a robust predictive pipeline for Rossmann, one of Europe's largest drugstore chains. The core challenge is to forecast daily sales for over 1,100 stores across Germany, considering factors like promotions, competition, school holidays, and seasonality.

## Architecture (v2.0)

The pipeline is designed as a **weekly scheduled batch system**, intentionally avoiding real-time infrastructure that would add operational complexity without benefit for this forecasting cadence. It runs every Monday at 3AM, producing a fresh set of predictions once the previous week's sales data is available.

> This architecture reflects the decisions made for v2.0. As the project evolves, for example toward segmented models, additional data sources, or lower-latency requirements, some of these choices may be revisited.

<img width="1360" height="1240" alt="Image" src="https://github.com/user-attachments/assets/d085bada-58b1-49c8-b65f-4c73739eb34d" />

### Technology Decisions

**Prefect: Orchestration**

Manages the weekly schedule, task-level logging, and failure notifications. Chosen for consistency with existing pipelines in this project's ecosystem, eliminating the need to operate a second orchestration tool.

**S3 + Parquet: Storage**

Weekly sales data is stored as Parquet files partitioned by week. Parquet's columnar compression keeps storage costs low and read performance high for range-based queries. The format is natively compatible with every other tool in this stack, and the transition from local disk to S3 requires no changes to downstream code.

**DuckDB: Feature Engineering**

Runs in-process inside the Prefect flow with no server or infrastructure to manage. Reads Parquet files directly and handles 1,100+ stores across multiple years of history in seconds. SQL-native interface makes feature logic transparent and easy to audit.

**MLflow: Model Registry and Experiment Tracking**

Provides versioned model artifacts, experiment metadata, and a clear promotion path from candidate to production. All training runs, including sMAPE scores, hyperparameters, and training windows, are logged for reproducibility.

**PostgreSQL (Supabase): Predictions Storage**

Predictions are written to a structured table with `store_id`, `date`, `predicted_value`, `model_version`, and `run_date` columns. PostgreSQL was chosen for its queryability, durability, and straightforward integration with dashboards and downstream consumers. Supabase is already part of the project's infrastructure, avoiding the need for an additional service.

## Modeling Approach

### Phase 1: Baseline with Prophet (Completed)

Facebook Prophet was used as the baseline model. Despite this being a time-series problem, Rossmann's sales are heavily influenced by non-linear events and multiple seasonalities, which Prophet is specifically designed to handle. It also provides clean decomposition of trend, weekly seasonality, and yearly seasonality, making results interpretable from the start.

Prophet's built-in holiday support was particularly relevant given Rossmann's promotional calendar.

However, Prophet is a curve-fitting model that fits trend and seasonality independently per time series, with no ability to learn shared patterns across stores. This meant that stores with unusual behavior required isolation into segments to produce acceptable results, and a single global Prophet model performed poorly.

The figure below shows the latest results of building models for different groups of Rossmann Stores.

<img width="1269" height="502" alt="Image" src="https://github.com/user-attachments/assets/3d16a4ed-bf3c-4ce7-90c5-0e7c66fbffd0" />


### Phase 2: Global XGBoost Model (Completed)

Moving from Prophet to XGBoost addressed the core limitation identified in Phase 1. XGBoost learns from features across all stores simultaneously, allowing it to implicitly discover store-level behavioral differences through features such as competition distance, store type, and promotional flags, without requiring explicit segmentation. A single global model trained on all stores outperformed the segmented Prophet approach.

All experiments were tracked in MLflow, with sMAPE as the primary evaluation metric.
The figure below depicts the XGBoost model performance across all the stores.

**Current best result: 12.2% sMAPE**

<img width="1851" height="356" alt="Image" src="https://github.com/user-attachments/assets/e70df35d-73e1-4323-ad77-ce6b487a7b7c" />

## Success Metric

**sMAPE (Symmetric Mean Absolute Percentage Error)** provides a balanced view of both over- and under-forecasting errors across stores of different sales scales, making it more appropriate than MAE or RMSE for a multi-store comparison.

## What Is Deliberately Excluded

| Tool | Reason excluded |
|---|---|
| Feature store (Feast, Tecton) | Only one model currently in production; becomes worth considering when a second model needs overlapping features |
| Kafka / Redis | No real-time ingestion or low-latency prediction serving |
| Kubernetes | Weekly batch job does not justify container orchestration overhead |
| Airflow | Redundant given Prefect is already in use |