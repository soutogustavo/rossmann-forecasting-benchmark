""" Results promotion utilities for Rossmann Forecasting Benchmark"""

import json
import logging
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import xgboost as xgb
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.config import PipelineConfig
from src.utils.preprocessing import preprocess_rossmann_data

logger = logging.getLogger(__name__)


def register_and_promote(
    model: xgb.XGBRegressor,
    feature_builder: preprocess_rossmann_data,
    artifacts: dict,
    X_example: pd.DataFrame,
    run_id: str,
    cfg: PipelineConfig,
) -> None:

    X_sig = signature_frame(X_example)
    signature = infer_signature(X_sig, X_example.tail(100))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        artifacts["cluster_map"].to_parquet(tmp / "cluster_map.parquet", index=False)
        (tmp / "schema.json").write_text(
            json.dumps(schema_to_json(artifacts["schema"]), indent=2))
        (tmp / "feature_config.json").write_text(
            json.dumps(artifacts["feature_config"], indent=2))
        mlflow.log_artifacts(str(tmp), artifact_path="model_extras")

    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        signature=signature,
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


def schema_to_json(schema: dict) -> dict:
    """Convert schema to JSON format.

    Args:
        schema (dict): Schema to convert.

    Returns:
        dict: Schema in JSON format.
    """
    dtypes = {}

    for col, dt in schema["dtypes"].items():
        if isinstance(dt, pd.CategoricalDtype):
            dtypes[col] = {
                "type": "category",
                "categories": dt.categories.tolist()
            }
        else:
            dtypes[col] = {"type": str(dt)}

    return {
        "columns": schema["columns"],
        "dtypes": dtypes
    }


def signature_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Copy of X with categories converted
       to the type of values (only for the signature).

    Args:
        X (pd.DataFrame): DataFrame to convert.

    Returns:
        pd.DataFrame: DataFrame with categories converted to the type of values.
    """
    X = X.copy()
    for col in X.select_dtypes("category"):
        X[col] = X[col].astype(X[col].cat.categories.dtype)
    return X
