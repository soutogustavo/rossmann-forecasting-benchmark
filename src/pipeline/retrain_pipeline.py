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
import xgboost as xgb
from dotenv import load_dotenv

from src.config import PipelineConfig, load_config
from src.utils.data import load_data
from src.utils.evaluation import (
    GateResult,
    get_champion_skill,
    quality_gate,
    sanity_check_refit,
    seasonal_naive_forecast,
    skill_score,
    smape,
)
from src.utils.preprocessing import (
    build_base_features,
    build_schema,
    preprocess_rossmann_store_data,
    same_weekday_lags,
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
    "n_jobs": -1,
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
    "sales_lag_63",
    "sales_lag_70",
    "sales_lag_77",
    "sales_lag_84",
    "sales_roll7_mean",
    "sales_roll7_std",
    "sales_roll28_mean",
    "sales_roll28_std",
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


def find_best_n(processed: dict) -> tuple[int, float]:
    """Find the best number of trees for the model using
    early stopping on the ES block.

    Args:
        processed (dict): Dictionary with the train, es, train_es,
        holdout and whole data.

    Returns:
        tuple[int, float]: The best number of iterations for the model
        and the sMAPE in the ES block.
    """
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
    logger.info("sMAPE (early stopping) = %s", f"{candidate_smape:.4f}")

    return best_n, candidate_smape


def evaluate_on_holdout(processed: dict, best_n: int) -> tuple[float, float]:
    """Evaluate candidate and naive models on the holdout set.

    Args:
        processed (dict): Dictionary with the train, es, train_es,
        holdout and whole data.
        best_n (int): The best number of iterations for the model.

    Returns:
        tuple[float, float]: (holdout_smape, naive_smape), both on open days.
    """
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
    logger.info("sMAPE (holdout) = %f", gate_smape)

    naive_preds = seasonal_naive_forecast(
        history=processed["train_es"],
        target=processed["holdout"]
    )
    naive_preds = np.where(X_val["open"] == 0, 0, naive_preds)
    naive_smape = smape(y_true=y_val, y_pred=naive_preds)
    logger.info("sMAPE (naive) = %s", f"{naive_smape:.4f}")

    return gate_smape, naive_smape


def train_final_model(
    processed: dict,
    best_n: int,
    cutoff_date: date
) -> tuple[xgb.XGBRegressor, dict]:
    """Retrain final model with best number of iterations and run sanity checks.

    Args:
        processed (dict): Dictionary with the train, es, train_es,
        holdout and whole data.
        best_n (int): The best number of iterations for the model.
        cutoff_date (date): The cutoff date for the model.

    Returns:
        tuple[xgb.XGBRegressor, dict]: The final model and the schema of the data.
    """
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
            cutoff_date) - pd.Timedelta(days=28))
        & (X_train_final["open"] == 1)
    )
    sanity_check_refit(
        model=final_model,
        X=X_train_final[FEATURES],
        y=y_train_final,
        recent_mask=recent
    )

    return final_model, schema_final, X_train_final


def check_champion_backtest(
    es_smape: float,
    holdout_smape: float,
    naive_smape: float,
    cfg: PipelineConfig,
) -> float | None:
    """Log the candidate's backtest metrics and return the champion's skill.

    Args:
        es_smape: sMAPE of the candidate on the early-stopping block (diagnostic only).
        holdout_smape: sMAPE of the candidate on the holdout.
        naive_smape: sMAPE of the seasonal naive on the same holdout.
        cfg: Pipeline configuration.

    Returns:
        The champion's holdout skill, or None if there is no champion
        or it was registered before skill was logged.
    """
    skill = skill_score(holdout_smape, naive_smape)
    champion_skill = get_champion_skill(cfg.model_name)

    metrics = {
        "es_smape": es_smape,
        "holdout_smape": holdout_smape,
        "naive_smape": naive_smape,
        "holdout_skill": skill,
    }
    if champion_skill is not None:
        metrics["champion_skill"] = champion_skill
    mlflow.log_metrics(metrics)

    logger.info(
        "Holdout: candidate sMAPE=%.4f naive=%.4f skill=%.3f champion skill=%s",
        holdout_smape,
        naive_smape,
        skill,
        f"{champion_skill:.3f}" if champion_skill is not None else "n/a",
    )
    return champion_skill


def check_quality_results(
    holdout_smape: float,
    naive_smape: float,
    champion_skill: float | None,
    cfg: PipelineConfig,
) -> GateResult:
    """Run the quality gate, tag the MLflow run and log a rejection.

    Args:
        holdout_smape (float): sMAPE of the candidate on the holdout.
        naive_smape (float): sMAPE of the seasonal naive on the same holdout.
        champion_skill (float | None): holdout skill of the current champion,
            or None if there is none to compare against.
        cfg (PipelineConfig): Configuration for the pipeline.

    Returns:
        GateResult: Whether the candidate passed, the reasons if not,
        and the candidate's skill.
    """
    gate = quality_gate(holdout_smape, naive_smape, champion_skill, cfg)
    mlflow.set_tags({
        "gate_passed": str(gate.passed),
        "gate_reasons": " | ".join(gate.reasons) or "none",
    })
    if not gate.passed:
        logger.warning(
            "Candidate rejected:\n  - %s", "\n  - ".join(gate.reasons)
        )

    return gate


def run(cfg: PipelineConfig) -> GateResult:
    """Run the re-train pipeline.

    Args:
        cfg (PipelineConfig): Configuration for the pipeline.

    Returns:
        GateResult: Result of the quality gates.
    """
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

        prep_data = build_base_features(
            sales=sales, stores=stores, horizon=cfg.horizon_days,
            target_feature=TARGET_NAME,
            categorical_features=CAT_FEATURES
        )
        pstore = preprocess_rossmann_store_data(data=stores)

        mlflow.log_params({
            "n_rows": len(prep_data),
            "n_stores": prep_data["store"].nunique(),
            "data_start": str(prep_data["date"].min().date())
        })

        train, es, holdout = split_input_data(
            data=prep_data,
            cutoff=cfg.cutoff_date,
            horizon=cfg.horizon_days
        )
        train_es = pd.concat([train, es], axis=0)
        whole = pd.concat([train_es, holdout], axis=0)

        processed = clustering_split_datasets(
            train=train.copy(),
            es=es.copy(),
            train_es=train_es.copy(),
            holdout=holdout.copy(),
            whole=whole.copy(),
            pstore=pstore.copy()
        )

        best_n, es_smape = find_best_n(processed)

        holdout_smape, naive_smape = evaluate_on_holdout(
            processed=processed,
            best_n=best_n
        )

        champion_skill = check_champion_backtest(
            es_smape=es_smape,
            holdout_smape=holdout_smape,
            naive_smape=naive_smape,
            cfg=cfg,
        )

        gate = check_quality_results(
            holdout_smape=holdout_smape,
            naive_smape=naive_smape,
            champion_skill=champion_skill,
            cfg=cfg,
        )

        final_model, schema_final, X_train_final = train_final_model(
            processed=processed,
            best_n=best_n,
            cutoff_date=cfg.cutoff_date
        )

        register_and_promote(
            model=final_model,
            artifacts = {
                "schema": schema_final,
                "cluster_map": processed["cluster_map"],
                "feature_config": {
                    "features": FEATURES,
                    "horizon_size": cfg.horizon_days,
                    "lags": same_weekday_lags(cfg.horizon_days),
                    "rolling_windows": [7, 28],
                    "best_n": best_n,
                }
            },
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
