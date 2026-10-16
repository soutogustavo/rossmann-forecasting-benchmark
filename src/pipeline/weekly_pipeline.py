"""Weekly job: retrain and forecast. In Dev, the cutoff comes from a simulated clock."""

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from src.pipeline import inference_pipeline, retrain_pipeline
from src.utils.sim_clock import advance_clock, read_clock

logger = logging.getLogger(__name__)


def last_sunday() -> date:
    today = datetime.now(tz=ZoneInfo("Europe/Berlin")).date()
    return today - timedelta(days=today.weekday() + 1)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rossmann weekly pipeline")
    parser.add_argument("--env", choices=["dev", "staging", "prod"], default="dev")
    parser.add_argument("--cutoff-date", type=date.fromisoformat, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    clock_param = os.getenv("SIM_CLOCK_PARAM")
    use_clock = args.cutoff_date is None and clock_param is not None

    if use_clock:
        cutoff = read_clock(clock_param)
        sim_end = date.fromisoformat(os.environ["SIM_END_DATE"])
        if cutoff > sim_end:
            logger.info(
                "Simulation finished (cutoff %s > end %s): nothing to do", cutoff, sim_end
            )
            return 0
    else:
        cutoff = args.cutoff_date or last_sunday()

    step_argv = ["--env", args.env, "--cutoff-date", cutoff.isoformat()]
    logger.info(
        "Weekly pipeline | env=%s cutoff=%s clock=%s", args.env, cutoff, use_clock
    )

    retrain_code = retrain_pipeline.main(step_argv)
    if retrain_code == 1:
        logger.error("Retrain failed: skipping forecast")
        code = 1
    else:
        if retrain_code == 2:
            logger.warning("Candidate rejected by gate: forecasting with current champion")
        inference_code = inference_pipeline.main(step_argv)
        code = inference_code if inference_code != 0 else retrain_code

    if use_clock and code != 1:
        nxt = advance_clock(clock_param, cutoff)
        logger.info("Clock advanced: %s -> %s", cutoff, nxt)

    logger.info("Weekly pipeline finished | cutoff=%s exit_code=%d", cutoff, code)
    return code


if __name__ == "__main__":
    sys.exit(main())
