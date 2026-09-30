"""Configuration file for the AWS Data Pipeline."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

CONFIG_DIR = Path("config")

@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline configuration"""

    env: str
    cutoff_date: date
    experiment_name: str
    model_name: str
    data_uri: str
    auto_promote: bool
    horizon_days: int = 60
    min_improvement_vs_naive: float = 0.10
    max_degradation_vs_champion: float = 0.05
    dry_run: bool = False


def load_config(env: str, cutoff_date: date) -> PipelineConfig:
    """CLI + YAML -> PipelineConfig.

    Args:
        env (str): Environment name
        cutoff_date (date): Cutoff date

    Returns:
        PipelineConfig: Pipeline configuration
    """
    with open(CONFIG_DIR / f"{env}.yaml") as f:
        values = yaml.safe_load(f)
    return PipelineConfig(env=env, cutoff_date=cutoff_date, **values)
