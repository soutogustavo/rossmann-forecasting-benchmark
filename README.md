![Status](https://img.shields.io/badge/status-in%20development-yellow)

# Rossmann Sales Forecasting

This project builds a robust predictive pipeline for Rossmann, one of Europe's largest drugstore chains. The core challenge is to forecast daily sales for over 1,100 stores across Germany, considering factors like promotions, competition, school holidays, and seasonality.

## Architecture (v2.0)

The pipeline is designed as a **weekly scheduled batch system** running on AWS, intentionally avoiding real-time infrastructure that would add operational complexity without benefit for this forecasting cadence. It runs every Monday at 3AM, producing a fresh set of predictions once the previous week's sales data is available.

> This architecture reflects the decisions made for v2.0. As the project evolves, for example toward segmented models, additional data sources, or lower-latency requirements, some of these choices may be revisited.

<img width="1520" height="1480" alt="Architecture v2.0" src="https://github.com/user-attachments/assets/68944600-a2bc-4e8d-8d3e-58295bfbdac6" />

### Technology Decisions

**EventBridge: Scheduling**

Triggers the ECS Fargate task every Monday at 3AM via a cron rule. Chosen over a self-managed scheduler because it is fully managed, costs nothing at this cadence, and integrates natively with ECS without additional infrastructure.

**ECR + ECS Fargate: Compute**

The pipeline is packaged as a Docker container, stored in ECR, and executed by ECS Fargate. Fargate was chosen over Lambda because it imposes no 15-minute execution limit and places no constraints on memory or local storage, making it more appropriate as the pipeline grows. There is no server or cluster to manage. If the pipeline later splits into multiple independent pipelines, an orchestration layer such as Prefect with an ECS worker pool can be introduced without changing the container logic.

**S3 + Parquet: Storage**

Weekly sales data is stored as Parquet files partitioned by week. Parquet's columnar compression keeps storage costs low and read performance high for range-based queries. S3 also serves as the artifact store for MLflow model files, centralizing storage under a single AWS service.

**AWS Secrets Manager: Credentials**

Database credentials and configuration are retrieved at runtime from Secrets Manager rather than stored in environment variables or container images. This is the standard security pattern for production pipelines on AWS and signals credential hygiene to anyone reviewing the codebase.

**DuckDB: Feature Engineering**

Runs in-process inside the container with no server or infrastructure to manage. Reads Parquet files directly from S3 and handles 1,100+ stores across multiple years of history in seconds. SQL-native interface makes feature logic transparent and easy to audit.

**MLflow: Model Registry and Experiment Tracking**

Provides versioned model artifacts stored on S3, experiment metadata, and a clear promotion path from candidate to production. All training runs, including sMAPE scores, hyperparameters, and training windows, are logged for reproducibility.

**RDS PostgreSQL: Predictions Storage**

Predictions are written to a structured table with `store_id`, `date`, `predicted_value`, `model_version`, and `run_date` columns. RDS PostgreSQL was chosen over Redshift because the predictions table is modest in size and does not require a data warehousing solution. It is the managed AWS equivalent of the Supabase Postgres used during local development, requiring no changes to query logic.

**SNS: Notifications**

Publishes a success or failure message at the end of each pipeline run. Decoupled from the pipeline logic so notification targets (email, Slack, PagerDuty) can be changed without touching the pipeline code.

**CloudWatch: Observability**

Captures container logs automatically from ECS Fargate. Metrics and alarms can be configured to alert on task failures or anomalous prediction volumes without additional tooling.

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

**Current best result: 12.2% sMAPE**

<img width="1851" height="356" alt="Image" src="https://github.com/user-attachments/assets/e70df35d-73e1-4323-ad77-ce6b487a7b7c" />

## Success Metric

**sMAPE (Symmetric Mean Absolute Percentage Error)** provides a balanced view of both over- and under-forecasting errors across stores of different sales scales, making it more appropriate than MAE or RMSE for a multi-store comparison.

## What Is Deliberately Excluded

| Tool | Reason excluded |
|---|---|
| Feature store (Feast, Tecton) | Only one model currently in production; becomes worth considering when a second model needs overlapping features |
| Kafka / Kinesis | No real-time ingestion or low-latency prediction serving |
| Redshift | Predictions table is modest in size; a data warehouse adds cost and complexity without benefit at this scale |
| SageMaker | Adds significant operational overhead; MLflow on S3 covers model registry and experiment tracking at lower cost and complexity |
| Prefect | Redundant for a single pipeline; scheduling is handled by EventBridge, observability by CloudWatch. Worth revisiting if the project grows to multiple independent pipelines |
| Kubernetes / EKS | Weekly batch job does not justify container orchestration at this scale; ECS Fargate provides managed containers without cluster management |