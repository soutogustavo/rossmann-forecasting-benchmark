""" Evaluation module for Rossmann Forecasting Benchmark"""

import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import PipelineConfig
from src.utils.validation import GateResult

logger = logging.getLogger(__name__)


def smape(y_true, y_pred, eps=1e-8):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_true (pd.Series): True values.
        y_pred (pd.Series): Predicted values.
        eps (float): Small constant to avoid division by zero.

    Returns:
        float: SMAPE value.
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2

    return 100 * np.mean(np.abs(y_true - y_pred) / np.maximum(denom, eps))


class RossmannEvaluation:
    """Class for evaluation metrics on Rossmann Forecasting Benchmark"""

    def __init__(self, masks_by_length=None):
        self.masks_by_length = masks_by_length

    def smape_adjusted(self, y_true, y_pred):
        """Adjusted SMAPE by masking zeros

        Args:
            y_true (pd.Series): True values.
            y_pred (pd.Series): Predicted values.

        Returns:
            float: SMAPE value.
        """
        mask = self.masks_by_length[len(y_true)]
        pred_fixed = np.where(mask == 0, 0, y_pred)

        return smape(y_true, pred_fixed)


class ModelSanityError(Exception):
    """Modelo treinou, mas produz saídas implausíveis."""


def skill_score(model_smape: float, naive_smape: float) -> float:
    """The extent to which the model outperforms the naive approach
        within the same window (1 = perfect, 0 = equal to the naive approach).

    Args:
        model_smape (float): sMAPE of the model.
        naive_smape (float): sMAPE of the naive model.

    Returns:
        float: Skill score.
    """
    return 1 - model_smape / naive_smape


def quality_gate(
    holdout_smape: float,
    naive_smape: float,
    champion_skill: float | None,
    cfg: PipelineConfig
) -> GateResult:
    """Apply quality gates to the candidate model.

    Args:
        holdout_smape (float): sMAPE of the candidate model.
        naive_smape (float): sMAPE of the naive model.
        champion_skill (float | None): Skill of the champion model.
        cfg (PipelineConfig): Configuration for the pipeline.

    Returns:
        GateResult: Result of the quality gates.
    """

    skill = skill_score(holdout_smape, naive_smape)
    reasons = []


    if skill < cfg.min_skill:
        reasons.append(
            f"Skill {skill:.3f} below minimum {cfg.min_skill:.2f} vs naive on the same window"
        )

    if champion_skill is None:
        logger.warning("Champion has no holdout_skill metric: skipping champion comparison")
    elif skill < champion_skill - cfg.max_skill_drop:
        reasons.append(
            f"Skill {skill:.3f} more than {cfg.max_skill_drop:.2f} below champion skill {champion_skill:.3f}"
        )

    return GateResult(passed=not reasons, reasons=reasons, skill=skill)


def seasonal_naive_forecast(
    history: pd.DataFrame,
    target: pd.DataFrame
) -> np.ndarray:
    """
    For each (store, day of week), repeat the last observed sale with the store open.

    Args:
        history (pd.DataFrame): Historical sales data.
        target (pd.DataFrame): Target data.

    Returns:
        np.ndarray: Seasonal naive forecasts.
    """
    last_by_dow = (
        history.loc[history["open"] == 1]
        .assign(dow=lambda d: d["date"].dt.dayofweek)
        .sort_values("date")
        .groupby(["store", "dayofweek"], observed=True)["sales"]
        .last()
        .rename("naive_pred")
    )
    preds = target.assign(dow=target["date"].dt.dayofweek).join(
        last_by_dow, on=["store", "dayofweek"]
    )["naive_pred"]

    return preds.fillna(0).to_numpy()


def sanity_check_refit(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    recent_mask: pd.Series,
    tolerance: float = 0.20,
) -> None:
    """In-sample check: DOES NOT measure generalization, only catches broken
    models (features all NaN, target misaligned, etc.).

    Args:
        model (xgb.XGBRegressor): The XGBoost model.
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        recent_mask (pd.Series): Mask for recent data.
        tolerance (float): Tolerance for mean prediction / mean actual ratio.
    """
    pred = model.predict(X[recent_mask])
    if not np.isfinite(pred).all():
        raise ModelSanityError("Refit model produced non-finite predictions")
    ratio = pred.mean() / y[recent_mask].mean()
    if abs(ratio - 1) > tolerance:
        raise ModelSanityError(
            f"Mean prediction / mean actual = {ratio:.2f} "
            f"(tolerance ±{tolerance:.0%})"
        )
    logger.info("Refit sanity check passed (mean ratio %.3f)", ratio)


def get_champion_skill(model_name: str) -> float | None:
    """
    Get the holdout skill metric recorded when the current champion was trained.

    Args:
        model_name (str): Name of the model.

    Returns:
        float | None: Skill value.
    """
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        return None

    return client.get_run(version.run_id).data.metrics.get("holdout_skill")


def get_champion_backtest_smape(model_name: str) -> float | None:
    """
    Get the backtest metric recorded when the current champion was trained.

    Args:
        model_name (str): Name of the model.

    Returns:
        float | None: sMAPE value.
    """
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        logger.info("No champion found — first run for '%s'", model_name)
        return None
    return client.get_run(version.run_id).data.metrics.get("backtest_smape")
