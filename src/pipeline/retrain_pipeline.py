"""Retrain pipeline"""

import argparse
import logging
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.config import PipelineConfig, load_config
from src.utils.data import load_data
from src.utils.evaluation import (
    GateResult,
    RossmannEvaluation,
    get_champion_backtest_smape,
    quality_gate,
    sanity_check_refit,
    seasonal_naive_forecast,
    smape,
)
from src.utils.preprocessing import (
    extended_preprocessing_rossmann_xgb,
    merge_sales_and_store_data,
    preprocess_rossmann_data,
    preprocess_rossmann_store_data,
    split_train_val_data,
)
from src.utils.processing import build_model, cluster_rossmann_stores
from src.utils.promote import register_and_promote
from src.utils.validation import validate_data

logger = logging.getLogger(__name__)


MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": 1,
    "objective": "reg:tweedie",
    "tweedie_variance_power": 1.5,
    "tree_method": "hist",
    "enable_categorical": True,
    "eval_names": ["train", "val"],
    "eval_metric": None,
}


VALID_ENVS = ["dev", "staging", "prod"]
RUN_SCOPED_FIELDS = {"cutoff_date", "dry_run"}


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    """Parse command-line arguments"""

    # Default is the Monday of the current week,
    #  as we usually run this pipeline on Mondays.
    current_date = datetime.now(tz=ZoneInfo("Europe/Berlin")).date()
    weekday = current_date.weekday()
    if (6 - weekday) == 0:
        current_date = current_date - timedelta(days=7)
    else:
        current_date = current_date - timedelta(days=weekday + 1)

    parser = argparse.ArgumentParser(
        description="Rossmann retraining pipeline"
    )
    parser.add_argument(
        "--env", choices=VALID_ENVS, default="dev"
    )
    parser.add_argument(
        "--cutoff-date", type=date.fromisoformat, default=current_date
    )
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args(argv)


def run(cfg: PipelineConfig) -> GateResult:
    """Run the re-train pipeline"""
    logger.info("Starting re-train pipeline in %s environment", cfg.env)
    logger.info("Cutoff date: %s", cfg.cutoff_date)

    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=f"retrain_{cfg.cutoff_date:%Y-%m-%d}") as mlrun:
        mlflow.set_tags({"env": cfg.env, "pipeline": "retraining"})
        mlflow.log_params(
            {**{k: str(v) for k, v in asdict(cfg).items()},
            **MODEL_PARAMS}
        )

        sales, stores = load_data(cfg)
        validate_data(sales, stores, cfg)

        sales_store = merge_sales_and_store_data(
            sales=sales,
            store=stores,
            id_feature="Store"
        )
        ptrain = preprocess_rossmann_data(df=sales_store)
        pstore = preprocess_rossmann_store_data(data=stores)

        mlflow.log_params({
            "n_rows": len(ptrain),
            "n_stores": ptrain["store"].nunique(),
            "data_start": str(ptrain["date"].min().date())
        })

        store_features = cluster_rossmann_stores(data=ptrain, store_data=pstore)
        train_encoded = extended_preprocessing_rossmann_xgb(
            data=ptrain,
            profile_data=store_features,
            categorical_features=["storetype", "assortment", "cluster"]
        )

        holdout_start = pd.Timestamp(
            cfg.cutoff_date - timedelta(days=cfg.horizon_days - 1)
        )
        X_train, y_train, X_val, y_val, X_val_naive = split_train_val_data(
            data=train_encoded,
            date_threshold=holdout_start
        )

        masks_by_length = {
            len(y_train): X_train["open"].values,
            len(y_val): X_val["open"].values,
        }

        rossmann_eval = RossmannEvaluation(masks_by_length=masks_by_length)
        MODEL_PARAMS["eval_metric"] = rossmann_eval.smape_adjusted

        logger.info("Training model with parameters: %s", MODEL_PARAMS)
        logger.info("X_train shape: %s", X_train.shape)
        logger.info("y_train shape: %s", y_train.shape)
        logger.info("X_val shape: %s", X_val.shape)
        logger.info("y_val shape: %s", y_val.shape)

        backtest_model = build_model(MODEL_PARAMS).fit(
            X_train, y_train,
            eval_set=[(X_train, y_train),(X_val, y_val)],
            verbose=50
        )

        preds = backtest_model.predict(X_val)
        preds = np.where(X_val['open'] == 0, 0, preds)

        candidate_smape = smape(y_true=y_val, y_pred=preds)
        naive_smape = smape(
            y_true=y_val,
            y_pred=seasonal_naive_forecast(X_val_naive, X_val_naive)
        )

        champion_smape = get_champion_backtest_smape(cfg.model_name)
        print(f"sMAPE = {candidate_smape:.3f}")
        print(f"Naive sMAPE = {naive_smape:.3f}")

        metrics = {"backtest_smape": candidate_smape, "naive_smape": naive_smape}
        if champion_smape is not None:
            metrics["champion_backtest_smape"] = champion_smape
        mlflow.log_metrics(metrics)
        logger.info("Backtest sMAPE: candidate=%.4f naive=%.4f champion=%s",
                    candidate_smape, naive_smape, f"{champion_smape:.4f}" if champion_smape else "n/a")

        gate = quality_gate(candidate_smape, naive_smape, champion_smape, cfg)
        mlflow.set_tags({"gate_passed": str(gate.passed), "gate_reasons": " | ".join(gate.reasons) or "none"})
        if not gate.passed:
            logger.warning("Candidate rejected:\n  - %s", "\n  - ".join(gate.reasons))
            return gate

        final_model = build_model(MODEL_PARAMS).fit(
                train_encoded[X_train.columns],
                train_encoded["sales"],
                verbose=50
            )

        recent_mask = (train_encoded["date"] > pd.Timestamp(
            cfg.cutoff_date - timedelta(days=28)))
        sanity_check_refit(
            final_model,
            train_encoded[X_train.columns],
            train_encoded["sales"],
            recent_mask
        )

        register_and_promote(
            final_model,
            preprocess_rossmann_data,
            train_encoded[X_train.columns],
            mlrun.info.run_id,
            cfg
        )



def main(argv: list[str] | None = None) -> int:
    """Main entrypoint for retraining pipeline execution."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        args = parse_args(argv)
        cfg = load_config(args.env, args.cutoff_date)
        run(cfg)
        return 0
    except (ValueError, argparse.ArgumentError) as exc:
        logger.error("Failed to execute retraining pipeline: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
