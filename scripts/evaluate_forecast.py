"""Replay (only dev): compares the forecast of a cutoff with the actual sales.

In production, this would run ~60 days after the forecast. In replay, the realized
is already in data-raw, so it can be measured immediately after.

Usage:
    python -m scripts.evaluate_forecast --cutoff-date 2015-05-31
"""

import argparse
from datetime import date

import mlflow
import pandas as pd
from dotenv import load_dotenv

from src.config import load_config
from src.utils.evaluation import smape

EXPERIMENT = "rossmann/forecast-monitoring"


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Evaluates the forecast against the realized"
    )
    parser.add_argument("--env", default="dev")
    parser.add_argument("--cutoff-date", type=date.fromisoformat, required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.env, args.cutoff_date)
    T = pd.Timestamp(cfg.cutoff_date)
    last_day = T + pd.Timedelta(days=cfg.horizon_days)

    preds = pd.read_parquet(f"{cfg.data_uri}/data-predictions/cutoff={T:%Y-%m-%d}/")
    actual = pd.read_parquet(
        f"{cfg.data_uri}/data-raw/",
        columns=["Store", "Date", "Sales", "Open"],
        filters=[("Date", ">", T), ("Date", "<=", last_day)],
    )
    actual["Date"] = pd.to_datetime(actual["Date"])

    df = preds.merge(
        actual, left_on=["store_id", "date"], right_on=["Store", "Date"],
        how="inner", validate="one_to_one",
    )
    if len(df) != len(preds):
        print(
            f"Warning: {len(preds) - len(df)} predictions without corresponding actual sales"
        )

    df = df[df["Open"] == 1]
    df["week"] = (df["date"] - T).dt.days.sub(1) // 7 + 1

    overall = smape(df["Sales"].to_numpy(), df["predicted_value"].to_numpy())
    by_week = {
        int(week): smape(g["Sales"].to_numpy(), g["predicted_value"].to_numpy())
        for week, g in df.groupby("week")
    }

    print(f"Realized sMAPE (open days): {overall:.4f}")
    for week, value in by_week.items():
        print(f"  week {week}: {value:.4f}")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"eval_{T:%Y-%m-%d}"):
        mlflow.set_tags({
            "cutoff": str(T.date()),
            "model_version": str(preds["model_version"].iloc[0]),
        })
        mlflow.log_metric("realized_smape_open", overall)
        for week, value in by_week.items():
            mlflow.log_metric("realized_smape_open_by_week", value, step=week)


if __name__ == "__main__":
    main()
