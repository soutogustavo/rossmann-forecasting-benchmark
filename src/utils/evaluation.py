""" Evaluation module for Rossmann Forecasting Benchmark"""

import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from prophet import Prophet

from src.config import PipelineConfig
from src.utils.preprocessing import split_cluster_data
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


def evaluate_model_performance_on_stores(
    model: Prophet, cluster_ptrain: pd.DataFrame, cluster_features:list,
    train_size:float=0.8, margin:float=1.5,
):
    """Evaluate the model performance on the cluster data.

    Args:
        model (Prophet): The Prophet model.
        cluster_ptrain (pd.DataFrame): The cluster data.
        train_size (float): The size of the training set.
        margin (float): The margin to apply to the maximum sales by day.

    Returns:
        tuple(dict, list): The training and testing sets and cap by day.
    """

    scores_smapes = []
    store_forecasts = {}
    store_forecasts_df = pd.DataFrame()

    for sid in cluster_ptrain["Store"].unique():
        store_data = cluster_ptrain[cluster_ptrain["Store"] == sid]
        store_x = store_data[cluster_features]
        store_x.reset_index(drop=True, inplace=True)

        _, store_x_test, cap_by_day = split_cluster_data(
            train_size=train_size,
            agg_x=store_x,
            margin=margin
        )

        future = model.make_future_dataframe(periods=store_x_test.shape[0])
        future = future.loc[:store_x.shape[0]-1]
        future["Open"] = store_x["Open"]
        future["Promo"] = store_x["Promo"]
        future["DayOfWeek"] = store_x["DayOfWeek"]
        future["cap"] = future["DayOfWeek"].map(cap_by_day)
        future["floor"] = 0

        store_forecast = model.predict(future)

        check_res =  store_x_test.join(store_forecast[["yhat"]])
        check_res["yhat"] = check_res["yhat"].mask(check_res["Open"] == 0, 0)
        check_res = check_res.round(3)
        ind_smape = smape(check_res["y"], check_res["yhat"])
        scores_smapes.append(ind_smape)

        store_forecasts[sid] = check_res

        check_res["Store"] = sid
        store_forecasts_df = pd.concat([
            store_forecasts_df, check_res], ignore_index=True)

    return store_forecasts_df, scores_smapes


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


def quality_gate(
    candidate_smape: float, naive_smape: float,
    champion_smape: float | None, cfg: PipelineConfig
) -> GateResult:
    """Apply quality gates to the candidate model.

    Args:
        candidate_smape (float): sMAPE of the candidate model.
        naive_smape (float): sMAPE of the naive model.
        champion_smape (float | None): sMAPE of the champion model.
        cfg (PipelineConfig): Configuration for the pipeline.

    Returns:
        GateResult: Result of the quality gates.
    """

    reasons: list[str] = []

    # (a) Same window, fair comparison: the model needs
    #     to be worth it compared to the naive one.
    if candidate_smape > naive_smape * (1 - cfg.min_improvement_vs_naive):
        reasons.append(
            f"Candidate sMAPE {candidate_smape:.4f} is not {cfg.min_improvement_vs_naive:.0%} "
            f"better than seasonal naive {naive_smape:.4f}"
        )

    # (b) Different windows (champion backtest was in a
    #     different period): that's why
    #     it's a degradation tolerance, not a strict comparison.
    if champion_smape is not None and candidate_smape > champion_smape * (1 + cfg.max_degradation_vs_champion):
        reasons.append(
            f"Candidate sMAPE {candidate_smape:.4f} degraded more than "
            f"{cfg.max_degradation_vs_champion:.0%} vs champion backtest {champion_smape:.4f}"
        )

    return GateResult(passed=not reasons, reasons=reasons)


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
        .groupby(["store", "dayofweek"])["sales"]
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
