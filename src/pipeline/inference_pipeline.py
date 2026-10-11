"""Inference pipeline: forecast dos próximos `horizon_days`
    dias com o modelo @champion.
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.config import PipelineConfig, load_config
from src.utils.preprocessing import build_base_features, set_cluster_feature

logger = logging.getLogger(__name__)

HISTORY_DAYS = 180  # covers the biggest lag (84) + the biggest window (28), with slack


def schema_from_json(data: dict) -> dict:
    """Inverse of schema_to_json: categories return to being CategoricalDtype."""
    dtypes = {}
    for col, spec in data["dtypes"].items():
        if spec["type"] == "category":
            cats = pd.Index(spec["categories"], dtype=spec.get("categories_dtype"))
            dtypes[col] = pd.CategoricalDtype(categories=cats)
        else:
            dtypes[col] = spec["type"]
    return {"columns": data["columns"], "dtypes": dtypes}


def load_champion(model_name: str):
    """Champion model + artifacts from the SAME run."""
    client = MlflowClient()
    version = client.get_model_version_by_alias(model_name, "champion")
    model = mlflow.xgboost.load_model(f"models:/{model_name}@champion")

    extras = Path(mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path="model_extras"
    ))
    cluster_map = pd.read_parquet(extras / "cluster_map.parquet")
    schema = schema_from_json(json.loads((extras / "schema.json").read_text()))

    logger.info("Champion: %s v%s (run %s)", model_name, version.version, version.run_id)
    return model, cluster_map, schema, version.version


def load_inputs(cfg: PipelineConfig):
    """History up to T + future covariates from T+1 to T+H + stores table.

    Args:
        cfg (PipelineConfig): pipeline configuration

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: history, future, stores
    """
    T = pd.Timestamp(cfg.cutoff_date)

    history = pd.read_parquet(
        f"{cfg.data_uri}/data-raw/",
        filters=[("Date", ">", T - pd.Timedelta(days=HISTORY_DAYS)), ("Date", "<=", T)],
    )
    history["Date"] = pd.to_datetime(history["Date"])
    if history["Date"].max() > T:
        raise ValueError("History contains dates after cutoff")

    future = pd.read_parquet(f"{cfg.data_uri}/data-future/cutoff={T:%Y-%m-%d}/")
    future["Date"] = pd.to_datetime(future["Date"])

    leaked = {"Sales", "Customers"} & set(future.columns)
    if leaked:
        raise ValueError(f"Future covariates contain {leaked}")
    first_day = T + pd.Timedelta(days=1)
    if future["Date"].min() != first_day or future["Date"].nunique() != cfg.horizon_days:
        raise ValueError(f"Future covariates do not cover {first_day.date()} + {cfg.horizon_days} days")

    stores = pd.read_parquet(f"{cfg.data_uri}/data-store/")
    return history, future, stores


def check_predictions(out: pd.DataFrame, n_stores: int, horizon: int) -> None:
    """Stops everything if the output is incomplete or implausible.

    Args:
        out (pd.DataFrame): predictions DataFrame
        n_stores (int): number of stores
        horizon (int): forecast horizon
    """
    expected = n_stores * horizon
    if len(out) != expected:
        raise ValueError(f"{len(out)} predictions, expected {expected}")
    if out["predicted_value"].isna().any():
        raise ValueError("Predictions with NaN")
    if (out["predicted_value"] < 0).any():
        raise ValueError("Negative predictions")
    if out.duplicated(["store_id", "date"]).any():
        raise ValueError("(store_id, date) duplicate")


def run(cfg: PipelineConfig) -> pd.DataFrame:
    """Generates and saves the forecast for T+1 to T+H.

    Args:
        cfg (PipelineConfig): pipeline configuration

    Returns:
        pd.DataFrame: predictions DataFrame
    """
    T = pd.Timestamp(cfg.cutoff_date)
    logger.info("Starting inference in %s | cutoff %s", cfg.env, T.date())

    model, cluster_map, schema, version = load_champion(cfg.model_name)
    history, future, stores = load_inputs(cfg)

    # History + future rows (Sales empty) pass through the SAME training function
    frame = pd.concat([history, future], ignore_index=True)
    features = build_base_features(
        sales=frame,
        stores=stores,
        horizon=cfg.horizon_days
    )
    fut = features[features["date"] > T]

    fut = set_cluster_feature(fut, cluster_map)

    # Same columns, in the same order and with the same categories as training
    X = fut[schema["columns"]].copy()
    for col, dtype in schema["dtypes"].items():
        if isinstance(dtype, pd.CategoricalDtype):
            X[col] = pd.Categorical(X[col], categories=dtype.categories)

    preds = np.where(fut["open"] == 1, model.predict(X), 0.0)

    out = pd.DataFrame({
        "store_id": fut["store"].astype(int).to_numpy(),
        "date": fut["date"].to_numpy(),
        "predicted_value": preds,
        "model_version": str(version),
        "run_date": datetime.now(tz=ZoneInfo("Europe/Berlin")).date(),
    })
    check_predictions(out, n_stores=future["Store"].nunique(), horizon=cfg.horizon_days)

    path = f"{cfg.data_uri}/data-predictions/cutoff={T:%Y-%m-%d}/predictions.parquet"
    out.to_parquet(path, index=False)
    logger.info("%d predictions for %d stores recorded in %s", len(out), future["Store"].nunique(), path)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Default: last Sunday (the pipeline runs on Mondays).

    Args:
        argv (list[str], optional): arguments to parse. Defaults to None.

    Returns:
        argparse.Namespace: parsed arguments
    """
    today = datetime.now(tz=ZoneInfo("Europe/Berlin")).date()
    last_sunday = today - timedelta(days=today.weekday() + 1)

    parser = argparse.ArgumentParser(description="Rossmann inference pipeline")
    parser.add_argument("--env", choices=["dev", "staging", "prod"], default="dev")
    parser.add_argument("--cutoff-date", type=date.fromisoformat, default=last_sunday)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)

    try:
        cfg = load_config(args.env, args.cutoff_date)
        run(cfg)
    except Exception:
        logger.exception("Inference pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
