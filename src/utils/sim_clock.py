"""Simulated clock: the current cutoff lives in an SSM Parameter."""

from datetime import date, timedelta

import boto3

STEP = timedelta(days=7)


def read_clock(param_name: str) -> date:
    ssm = boto3.client("ssm")
    value = ssm.get_parameter(Name=param_name)["Parameter"]["Value"]
    return date.fromisoformat(value)


def advance_clock(param_name: str, current: date) -> date:
    nxt = current + STEP
    boto3.client("ssm").put_parameter(
        Name=param_name,
        Value=nxt.isoformat(),
        Overwrite=True
    )
    return nxt
