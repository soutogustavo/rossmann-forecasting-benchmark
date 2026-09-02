""" Evaluation module for Rossmann Forecasting Benchmark"""

import numpy as np
import pandas as pd
from prophet import Prophet

from src.preprocessing import split_cluster_data


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
        future = future.loc[:store_x.shape[0]-1] # make sure it does not go beyond the test set
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
