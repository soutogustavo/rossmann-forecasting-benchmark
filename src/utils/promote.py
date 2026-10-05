""" Results promotion utilities for Rossmann Forecasting Benchmark"""

import logging
import tempfile
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import PipelineConfig
from src.utils.preprocessing import preprocess_rossmann_data

logger = logging.getLogger(__name__)


def register_and_promote(
    model: xgb.XGBRegressor,
    feature_builder: preprocess_rossmann_data,
    X_example: pd.DataFrame,
    run_id: str,
    cfg: PipelineConfig,
) -> None:

    with tempfile.TemporaryDirectory() as tmp:
        fb_path = Path(tmp) / "feature_builder.joblib"
        joblib.dump(feature_builder, fb_path)
        mlflow.log_artifact(str(fb_path), artifact_path="model_extras")

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=X_example.head(5),
        registered_model_name=None if cfg.dry_run else cfg.model_name,
    )

    if cfg.dry_run:
        logger.info("Dry run: model logged but not registered")
        return

    client = MlflowClient()
    new_version = client.search_model_versions(
        f"name='{cfg.model_name}' and run_id='{run_id}'"
    )[0].version

    alias = "champion" if cfg.auto_promote else "challenger"
    logger.info("Registering %s v%s with alias '%s'",
                cfg.model_name, new_version, alias)

    if alias == "champion":
        try:
            previous = client.get_model_version_by_alias(
                cfg.model_name, "champion"
            ).version
            client.set_model_version_tag(
                cfg.model_name, new_version, "previous_champion", previous
            )
        except MlflowException:
            pass

    client.set_registered_model_alias(cfg.model_name, alias, new_version)

    logger.info(
        "Registered %s v%s with alias '%s'",
        cfg.model_name, new_version, alias
    )
