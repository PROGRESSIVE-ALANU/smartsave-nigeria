"""Research metrics computed from simulation outputs.

Every number here is a statistic of *simulated* outcomes under stated
assumptions. None of them is an empirical observation about real households.

Definitions
-----------
success rate               % of simulations with school-fee shortfall = 0
mean shortfall             average shortfall over ALL simulations (0 counted)
mean shortfall if failed   average shortfall among simulations that failed
funds at deadline          money usable for fees just before paying them
final savings              education fund + general savings at end of horizon
cash-flow deficit month    income < essential costs in that month
                           (depends on conditions only, identical across
                           strategies by construction)
unfunded-essential month   essential costs could not be fully paid even after
                           using allowed savings
worst case                 the single worst simulated outcome (max shortfall)
percentiles                P5, P25, P50, P75, P95 of the stated quantity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator import SimulationResult, StrategyResult
from src.strategies import STRATEGY_LABELS

PERCENTILES = (5, 25, 50, 75, 95)


def _strategy_row(result: SimulationResult, r: StrategyResult) -> dict[str, float]:
    failed = ~r.met_deadline
    unfunded_month = r.unfunded_essentials > 0.5
    deficit = result.cashflow_deficit
    final = r.total_balance[:, -1]
    row: dict[str, float] = {
        "strategy": r.strategy,
        "label": STRATEGY_LABELS.get(r.strategy, r.strategy),
        "success_rate_pct": 100.0 * float(r.met_deadline.mean()),
        "mean_shortfall": float(r.shortfall.mean()),
        "mean_shortfall_if_failed": float(r.shortfall[failed].mean()) if failed.any() else 0.0,
        "worst_case_shortfall": float(r.shortfall.max()),
        "median_funds_at_deadline": float(np.median(r.funds_at_deadline)),
        "median_final_savings": float(np.median(final)),
        "pct_sims_with_cashflow_deficit": 100.0 * float(deficit.any(axis=1).mean()),
        "mean_cashflow_deficit_months": float(deficit.sum(axis=1).mean()),
        "pct_sims_with_unfunded_essentials": 100.0 * float(unfunded_month.any(axis=1).mean()),
        "mean_unfunded_essential_months": float(unfunded_month.sum(axis=1).mean()),
        "mean_unfunded_essential_amount": float(r.unfunded_essentials.sum(axis=1).mean()),
        "mean_total_leakage": float(r.leakage.sum(axis=1).mean()),
        "mean_total_nonessential_cut": float(r.nonessential_cut.sum(axis=1).mean()),
        "mean_education_fund_used_for_essentials": float(
            r.education_used_for_essentials.sum(axis=1).mean()),
    }
    for q in PERCENTILES:
        row[f"funds_at_deadline_p{q}"] = float(np.percentile(r.funds_at_deadline, q))
    row["shortfall_p95"] = float(np.percentile(r.shortfall, 95))
    return row


def summarize(result: SimulationResult) -> pd.DataFrame:
    """One row of summary metrics per strategy."""
    return pd.DataFrame(
        [_strategy_row(result, r) for r in result.strategies.values()]
    )


def percentile_table(result: SimulationResult, quantity: str = "funds_at_deadline") -> pd.DataFrame:
    """Percentiles of a per-simulation quantity, one row per strategy.

    quantity: "funds_at_deadline", "shortfall" or "final_savings".
    """
    rows = []
    for name, r in result.strategies.items():
        if quantity == "final_savings":
            values = r.total_balance[:, -1]
        else:
            values = getattr(r, quantity)
        row = {"strategy": STRATEGY_LABELS.get(name, name), "worst": None}
        row["worst"] = float(values.max() if quantity == "shortfall" else values.min())
        for q in PERCENTILES:
            row[f"P{q}"] = float(np.percentile(values, q))
        row["mean"] = float(values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(result: SimulationResult, a: str, b: str) -> dict[str, float]:
    """Compare two strategies simulation-by-simulation (same random futures).

    Because both strategies faced identical conditions in each simulation,
    differences here come only from the decision rules.
    """
    ra, rb = result.strategies[a], result.strategies[b]
    ma, mb = ra.met_deadline, rb.met_deadline
    return {
        "both_meet_pct": 100.0 * float((ma & mb).mean()),
        "only_a_meets_pct": 100.0 * float((ma & ~mb).mean()),
        "only_b_meets_pct": 100.0 * float((~ma & mb).mean()),
        "neither_meets_pct": 100.0 * float((~ma & ~mb).mean()),
        "mean_funds_difference_a_minus_b": float(
            (ra.funds_at_deadline - rb.funds_at_deadline).mean()),
    }


def mean_paths(result: SimulationResult) -> pd.DataFrame:
    """Mean and 10th/90th percentile of total savings by month and strategy."""
    rows = []
    for name, r in result.strategies.items():
        total = r.total_balance
        for month in range(result.horizon + 1):
            col = total[:, month]
            rows.append({
                "strategy": name,
                "month": month,
                "mean_total_savings": float(col.mean()),
                "p10_total_savings": float(np.percentile(col, 10)),
                "p90_total_savings": float(np.percentile(col, 90)),
            })
    return pd.DataFrame(rows)


def outcomes_frame(result: SimulationResult) -> pd.DataFrame:
    """One row per (simulation, strategy): handy for CSV export."""
    frames = []
    deficit_months = result.cashflow_deficit.sum(axis=1)
    shocks = result.paths.unexpected_expenses.sum(axis=1)
    for name, r in result.strategies.items():
        frames.append(pd.DataFrame({
            "simulation": np.arange(result.n_simulations),
            "strategy": name,
            "funds_at_deadline": r.funds_at_deadline,
            "fees_paid": r.fees_paid,
            "shortfall": r.shortfall,
            "met_deadline": r.met_deadline,
            "final_savings": r.total_balance[:, -1],
            "cashflow_deficit_months": deficit_months,
            "unfunded_essential_months": (r.unfunded_essentials > 0.5).sum(axis=1),
            "total_unexpected_expenses": shocks,
            "total_leakage": r.leakage.sum(axis=1),
        }))
    return pd.concat(frames, ignore_index=True)
