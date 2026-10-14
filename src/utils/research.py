"""Functions that can be used during model research phase"""

import numpy as np
import pandas as pd
from prophet import Prophet

from src.utils.evaluation import smape
from src.utils.preprocessing import split_cluster_data


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


def calculate_store_factors(
    model: Prophet,
    train_data: pd.DataFrame,
    window_size: int = 60
):
    """Calculate store factors for the Rossmann dataset.

    Args:
        model (Prophet): The Prophet model.
        train_data (pd.DataFrame): Training data.
        window_size (int, optional): Window size for the rolling statistics. Defaults to 60.

    Returns:
        dict: Dictionary with store factors.
    """

    stores = train_data["store"].unique()

    store_factors = {}
    for sid in stores:

        store_data = train_data[train_data["store"] == sid].copy()
        store_data = store_data.sort_values("date").tail(window_size).reset_index(drop=True)
        store_data.rename(columns={"date":"ds", "sales": "y"}, inplace=True)
        store_data['cap'] = float(store_data['y'].max() * 1.5)
        store_data['floor'] = 0.0

        factor_data = store_data.head(window_size // 2)

        features = ["ds", "cap", "floor"] + list(model.extra_regressors.keys())
        factor_forecast = model.predict(factor_data[features])

        store_factor = (
            factor_data["y"].values /
            factor_forecast["yhat"].values
        ).mean()

        store_factors[int(sid)] = float(abs(store_factor))

    return store_factors
