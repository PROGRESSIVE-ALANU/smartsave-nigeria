"""Month-by-month cash-flow simulation engine.

The same engine produces both the deterministic "expected" projection (one
path, all randomness switched off) and the Monte Carlo experiments (many
paths). Every strategy is run on exactly the same ``ScenarioPaths``.

Accounts
--------
G  general savings (starts at initial savings)
E  earmarked education fund (starts at existing school-fee savings)
Neither account may go below zero, and there is no borrowing. When money runs
out for essentials, the unmet amount is recorded as ``unfunded_essentials``
rather than being hidden in a negative balance.

Monthly order of operations (t = 0 .. horizon-1, deadline month D)
------------------------------------------------------------------
1. Pay essentials X_t from income Y_t. A gap max(X_t - Y_t, 0) is covered
   from G, then from E (if allowed), and any remainder is unfunded.
2. A_t = max(Y_t - X_t, 0) is allocated by the strategy (see strategies.py).
   The required monthly saving before the deadline is
       R_t = max(F - E_t, 0) / (D - t),
   i.e. it is recalculated each month from the current fund balance and the
   months left, so a household that falls behind tries to catch up.
3. At the end of month D the fees F are paid from E, then from G (if
   allowed). Any part that cannot be paid is the school-fee shortfall.
After the deadline (if horizon > D), R_t = 0 and households keep living;
months after the deadline only affect cash-flow and final-savings metrics.

Accounting identity (checked in the tests)
------------------------------------------
initial G + initial E + sum(income)
    = final G + final E + essentials paid + nonessential spent
      + leakage + fees paid
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from src.budget import Household
from src.scenarios import Scenario, ScenarioPaths, apply_scenario, generate_paths
from src.strategies import STRATEGIES, StrategyParams, allocate


@dataclass
class StrategyResult:
    """Outputs for one strategy. Monthly arrays have shape (n, horizon);
    balance arrays have shape (n, horizon + 1) with column 0 = start."""

    strategy: str
    education_balance: np.ndarray
    general_balance: np.ndarray
    nonessential_spent: np.ndarray
    education_contribution: np.ndarray
    general_saving: np.ndarray
    leakage: np.ndarray
    unfunded_essentials: np.ndarray
    education_used_for_essentials: np.ndarray
    general_used_for_essentials: np.ndarray
    nonessential_cut: np.ndarray  # planned minus actually spent
    funds_at_deadline: np.ndarray  # (n,) money usable for fees, before paying
    fees_paid: np.ndarray  # (n,)
    shortfall: np.ndarray  # (n,)

    @property
    def total_balance(self) -> np.ndarray:
        return self.education_balance + self.general_balance

    @property
    def met_deadline(self) -> np.ndarray:
        # Tolerance of half a naira absorbs floating-point rounding.
        return self.shortfall <= 0.5


@dataclass
class SimulationResult:
    household: Household  # after scenario adjustments
    scenario: Scenario
    params: StrategyParams
    n_simulations: int
    horizon: int
    deadline: int
    seed: int
    paths: ScenarioPaths
    strategies: dict[str, StrategyResult] = field(default_factory=dict)

    @property
    def cashflow_deficit(self) -> np.ndarray:
        """(n, horizon) True where income < essential costs that month.

        This depends only on the economic conditions, not on the strategy."""
        return self.paths.income < self.paths.essentials - 1e-9


def run_strategy(
    strategy: str,
    h: Household,
    paths: ScenarioPaths,
    params: StrategyParams,
    deadline: int,
) -> StrategyResult:
    """Run one strategy over pre-generated paths (see module docstring)."""
    n, horizon = paths.shape
    fees = h.school_fees

    E = np.full(n, h.existing_school_savings, dtype=float)
    G = np.full(n, h.initial_savings, dtype=float)

    edu_bal = np.zeros((n, horizon + 1))
    gen_bal = np.zeros((n, horizon + 1))
    edu_bal[:, 0], gen_bal[:, 0] = E, G
    monthly = {k: np.zeros((n, horizon)) for k in (
        "N", "C", "S", "U", "unfunded", "E_used", "G_used", "cut")}
    funds_at_deadline = np.zeros(n)
    fees_paid = np.zeros(n)
    shortfall = np.zeros(n)

    essentials = paths.essentials
    for t in range(horizon):
        Y = paths.income[:, t]
        X = essentials[:, t]

        # 1. Essentials first; cover any gap from savings, never by borrowing.
        gap = np.maximum(X - Y, 0.0)
        from_G = np.minimum(gap, G)
        G = G - from_G
        gap = gap - from_G
        if params.allow_education_fund_for_essentials:
            from_E = np.minimum(gap, E)
            E = E - from_E
            gap = gap - from_E
        else:
            from_E = np.zeros(n)

        # 2. Strategy allocates what is left after essentials.
        available = np.maximum(Y - X, 0.0)
        if t < deadline:
            required = np.maximum(fees - E, 0.0) / (deadline - t)
        else:
            required = np.zeros(n)
        a = allocate(strategy, available, paths.nonessential_planned[:, t], required, params)
        E = E + a.education
        G = G + a.general_saving

        # 3. Pay fees at the end of the deadline month.
        if t == deadline - 1:
            usable_G = G if params.use_general_savings_for_fees else np.zeros(n)
            funds_at_deadline = E + usable_G
            pay_E = np.minimum(fees, E)
            E = E - pay_E
            remaining = fees - pay_E
            pay_G = np.minimum(remaining, usable_G)
            G = G - pay_G
            remaining = remaining - pay_G
            fees_paid = pay_E + pay_G
            shortfall = remaining

        monthly["N"][:, t] = a.nonessential
        monthly["C"][:, t] = a.education
        monthly["S"][:, t] = a.general_saving
        monthly["U"][:, t] = a.leakage
        monthly["unfunded"][:, t] = gap
        monthly["E_used"][:, t] = from_E
        monthly["G_used"][:, t] = from_G
        monthly["cut"][:, t] = np.minimum(
            paths.nonessential_planned[:, t], available) - a.nonessential
        edu_bal[:, t + 1] = E
        gen_bal[:, t + 1] = G

    return StrategyResult(
        strategy=strategy,
        education_balance=edu_bal,
        general_balance=gen_bal,
        nonessential_spent=monthly["N"],
        education_contribution=monthly["C"],
        general_saving=monthly["S"],
        leakage=monthly["U"],
        unfunded_essentials=monthly["unfunded"],
        education_used_for_essentials=monthly["E_used"],
        general_used_for_essentials=monthly["G_used"],
        nonessential_cut=monthly["cut"],
        funds_at_deadline=funds_at_deadline,
        fees_paid=fees_paid,
        shortfall=shortfall,
    )


def simulate(
    household: Household,
    scenario: Scenario,
    params: StrategyParams,
    n_simulations: int = 1000,
    horizon: int | None = None,
    seed: int = 42,
    strategies: tuple[str, ...] = STRATEGIES,
) -> SimulationResult:
    """Run every strategy on the same simulated futures.

    ``horizon`` defaults to the (scenario-adjusted) deadline. It must be at
    least the deadline, and the deadline must be at least 1 month away.
    """
    h = apply_scenario(household, scenario)
    deadline = h.months_until_due
    if deadline < 1:
        raise ValueError(
            "Fees are due immediately; there is nothing to simulate. Use the "
            "affordability calculator for the immediate-deadline case."
        )
    horizon = deadline if horizon is None else int(horizon)
    if horizon < deadline:
        raise ValueError(
            f"Horizon ({horizon}) must be at least the deadline ({deadline} months)."
        )
    paths = generate_paths(h, scenario, n_simulations, horizon, seed)
    result = SimulationResult(
        household=h, scenario=scenario, params=params, n_simulations=n_simulations,
        horizon=horizon, deadline=deadline, seed=seed, paths=paths,
    )
    for strategy in strategies:
        result.strategies[strategy] = run_strategy(strategy, h, paths, params, deadline)
    return result


def deterministic_scenario(scenario: Scenario) -> Scenario:
    """Same scenario with every random component switched off.

    Scheduled events (medical bill, food price jump, income loss, inflation)
    are kept because they are known in advance in the scenario definition.
    """
    return replace(scenario, income_cv=0.0, shock_probability=0.0, shock_severity=0.0)


def project_deterministic(
    household: Household,
    params: StrategyParams,
    scenario: Scenario | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Month-by-month expected projection for every strategy (one path).

    Returns a long DataFrame with one row per (strategy, month), month 0
    being the starting balances. Returns an empty frame if fees are due now.
    """
    scenario = deterministic_scenario(scenario or Scenario())
    if apply_scenario(household, scenario).months_until_due < 1:
        return pd.DataFrame()
    result = simulate(household, scenario, params, n_simulations=1, horizon=horizon, seed=0)
    return projection_frame(result, sim_index=0)


def projection_frame(result: SimulationResult, sim_index: int = 0) -> pd.DataFrame:
    """Tidy table of one simulated path (all strategies), for charts/CSV."""
    rows = []
    p = result.paths
    for name, r in result.strategies.items():
        for month in range(result.horizon + 1):
            row = {
                "strategy": name,
                "month": month,
                "education_fund": r.education_balance[sim_index, month],
                "general_savings": r.general_balance[sim_index, month],
                "total_savings": r.total_balance[sim_index, month],
                # Same as total_savings except in the deadline month, where it
                # shows the balance just before the fees are paid.
                "total_before_fees": r.total_balance[sim_index, month]
                + (r.fees_paid[sim_index] if month == result.deadline else 0.0),
            }
            if month > 0:
                t = month - 1
                row.update(
                    income=p.income[sim_index, t],
                    essentials=p.essentials[sim_index, t],
                    unexpected_expenses=p.unexpected_expenses[sim_index, t],
                    nonessential_spent=r.nonessential_spent[sim_index, t],
                    education_contribution=r.education_contribution[sim_index, t],
                    general_saving=r.general_saving[sim_index, t],
                    leakage=r.leakage[sim_index, t],
                    unfunded_essentials=r.unfunded_essentials[sim_index, t],
                    is_deadline_month=(month == result.deadline),
                )
            rows.append(row)
    return pd.DataFrame(rows)
