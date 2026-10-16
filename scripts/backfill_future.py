"""Generate future covariates for all simulation cutoffs."""

from datetime import date, timedelta

from scripts.simulate_future import main as simulate

cutoff, end = date(2015, 2, 1), date(2015, 5, 31)
while cutoff <= end:
    simulate(["--env", "dev", "--cutoff-date", cutoff.isoformat()])
    cutoff += timedelta(days=7)
