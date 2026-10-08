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
    get_champion_backtest_smape,
    quality_gate,
    sanity_check_refit,
    seasonal_naive_forecast,
    smape,
)
from src.utils.preprocessing import (
    build_schema,
    merge_sales_and_store_data,
    preprocess_rossmann_data,
    preprocess_rossmann_store_data,
    preprocessing_for_xgb,
    split_input_data,
)
from src.utils.processing import (
    build_evaluation_rossmann_instance,
    build_model,
    clustering_split_datasets,
)
from src.utils.promote import register_and_promote
from src.utils.validation import validate_data

logger = logging.getLogger(__name__)


MODEL_PARAMS = {
    "n_estimators": 800,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": 1,
    "objective": "reg:tweedie",
    "tweedie_variance_power": 1.5,
    "tree_method": "hist",
    "enable_categorical": True,
    "early_stopping_rounds": 50,
    "eval_metric": None,
}

VALID_ENVS = ["dev", "staging", "prod"]
TARGET_NAME = "sales"
CAT_FEATURES = ["storetype", "assortment"]
FEATURES = [
    "store",
    "dayofweek",
    "open",
    "promo",
    "storetype",
    "assortment",
    "competitiondistance",
    "competitionopensincemonth",
    "competitionopensinceyear",
    "promo2",
    "year",
    "month",
    "competitionopenmonth",
    "competitionopenyear",
    "days_to_competitor",
    "cluster",
    "is_holiday",
    "sales_60",
    "sales_61",
    "sales_62",
    "sales_mean",
    "sales_sum",
    "sales_std"
]

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

        prep_data = preprocessing_for_xgb(
            data=ptrain,
            horizon_size=cfg.horizon_days,
            target_feature=TARGET_NAME,
            categorical_features=CAT_FEATURES
        )

        train, es, holdout = split_input_data(
            data=prep_data,
            cutoff=cfg.cutoff_date,
            horizon=cfg.horizon_days
        )
        train_es = pd.concat([train, es], axis=0)
        whole = pd.concat([train, train_es, holdout], axis=0)

        processed = clustering_split_datasets(
            train=train.copy(),
            es=es.copy(),
            train_es=train_es.copy(),
            holdout=holdout.copy(),
            whole=whole.copy(),
            pstore=pstore.copy()
        )

        X_train = processed["train"][FEATURES]
        y_train = processed["train"][TARGET_NAME]
        X_es = processed["es"][FEATURES]
        y_es = processed["es"][TARGET_NAME]
        rossmann_eval_s1 = build_evaluation_rossmann_instance(
            X_train, y_train, X_es, y_es
        )
        MODEL_PARAMS["eval_metric"] = rossmann_eval_s1.smape_adjusted

        backtest_model = build_model(MODEL_PARAMS).fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train),(X_es, y_es)],
            verbose=50
        )

        best_n = backtest_model.best_iteration + 1
        mlflow.log_metric("best_iteration", backtest_model.best_iteration)

        preds = backtest_model.predict(X_es[FEATURES])
        preds = np.where(X_es["open"] == 0, 0, preds)
        candidate_smape = smape(y_true=y_es, y_pred=preds)
        logger.info("sMAPE = %s", f"{candidate_smape:.4f}")

        X_train = processed["train_es"][FEATURES]
        y_train = processed["train_es"][TARGET_NAME]
        X_val = processed["holdout"][FEATURES]
        y_val = processed["holdout"][TARGET_NAME]
        rossmann_eval_s2 = build_evaluation_rossmann_instance(
            X_train, y_train, X_val, y_val
        )
        MODEL_PARAMS["eval_metric"] = rossmann_eval_s2.smape_adjusted

        refit_params = {**MODEL_PARAMS, "n_estimators": best_n}
        refit_params.pop("early_stopping_rounds", None)

        gate_model = build_model(refit_params).fit(
            X_train,
            y_train,
            verbose=50
        )

        preds = gate_model.predict(X_val[FEATURES])
        preds = np.where(X_val["open"] == 0, 0, preds)
        gate_smape = smape(y_true=y_val, y_pred=preds)
        logger.info("sMAPE (gate) = %f", gate_smape)

        naive_preds = seasonal_naive_forecast(
            history=processed["train_es"],
            target=processed["holdout"]
        )
        naive_preds = np.where(X_val["open"] == 0, 0, naive_preds)
        naive_smape = smape(y_true=y_val, y_pred=naive_preds)
        logger.info("sMAPE (naive) = %s", f"{naive_smape:.4f}")

        champion_smape = get_champion_backtest_smape(cfg.model_name)

        metrics = {
            "backtest_smape": candidate_smape,
            "naive_smape": naive_smape
        }

        if champion_smape is not None:
            metrics["champion_backtest_smape"] = champion_smape
        mlflow.log_metrics(metrics)
        logger.info("Backtest sMAPE: candidate=%.4f naive=%.4f champion=%s",
                    candidate_smape, naive_smape,
                    f"{champion_smape:.4f}" if champion_smape else "n/a")

        gate = quality_gate(candidate_smape, naive_smape, champion_smape, cfg)
        mlflow.set_tags({
            "gate_passed": str(gate.passed),
            "gate_reasons": " | ".join(gate.reasons) or "none"
        })
        if not gate.passed:
            logger.warning(
                "Candidate rejected:\n  - %s", "\n  - ".join(gate.reasons)
            )
            return gate

        X_train_final = processed["whole"]
        y_train_final = processed["whole"][TARGET_NAME]

        schema_final = build_schema(X_train_final[FEATURES])

        refit_params = {**MODEL_PARAMS, "n_estimators": best_n}
        refit_params.pop("early_stopping_rounds", None)

        final_model = build_model(refit_params).fit(
                X_train_final[FEATURES],
                y_train_final,
                verbose=50
            )

        mlflow.log_params({
            "final_n_estimators": best_n,
            "final_train_end": str(X_train_final["date"].max().date()),
        })
        mlflow.log_metric("final_train_rows", len(X_train_final))

        recent = (
            (X_train_final["date"] > pd.Timestamp(
                cfg.cutoff_date) - pd.Timedelta(days=28))
            & (X_train_final["open"] == 1)
        )
        sanity_check_refit(
            final_model,
            X_train_final[FEATURES],
            y_train_final,
            recent
        )

        register_and_promote(
            model=final_model,
            feature_builder=preprocess_rossmann_data,
            artifacts = {"schema": schema_final},
            X_example=X_train_final[FEATURES],
            run_id=mlrun.info.run_id,
            cfg=cfg
        )

        return gate

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
        gate = run(cfg)

    except (ValueError, argparse.ArgumentError) as exc:
        logger.error("Failed to execute retraining pipeline: %s", exc)

    if not gate.passed:
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
