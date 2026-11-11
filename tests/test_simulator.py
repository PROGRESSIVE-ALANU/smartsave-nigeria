"""Phase 3 tests: scenarios, simulation engine, and metrics."""

from dataclasses import replace

import numpy as np
import pytest

from src.budget import Household, example_household
from src.metrics import mean_paths, outcomes_frame, paired_comparison, percentile_table, summarize
from src.scenarios import Scenario, apply_scenario, generate_paths, lognormal_sigma, preset_scenarios
from src.simulator import project_deterministic, simulate
from src.strategies import (
    EXPENSE_PRIORITIZATION,
    PLANNED_SAVINGS,
    SAVE_REMAINDER,
    STRATEGIES,
    StrategyParams,
)

P = StrategyParams(remainder_saving_rate=0.6, max_nonessential_cut=0.3)
RANDOM = Scenario(
    name="random", income_cv=0.3, shock_probability=0.2,
    shock_severity=80_000, monthly_inflation=0.02,
)


def make(**overrides) -> Household:
    base = example_household().to_dict()
    base.update(overrides)
    return Household.from_mapping(base)


# ----------------------------------------------------------- scenarios
def test_lognormal_income_has_mean_one_and_target_cv():
    h = example_household()
    paths = generate_paths(h, Scenario(income_cv=0.25), 200_000, 1, seed=1)
    m = paths.income[:, 0] / h.monthly_income
    assert m.mean() == pytest.approx(1.0, abs=0.005)
    assert m.std() / m.mean() == pytest.approx(0.25, abs=0.005)
    assert (m > 0).all()
    assert lognormal_sigma(0) == 0


def test_shock_size_and_frequency():
    h = example_household()
    paths = generate_paths(h, Scenario(shock_probability=0.3, shock_severity=50_000),
                           100_000, 1, seed=2)
    shocks = paths.unexpected_expenses[:, 0]
    occurred = shocks > 0
    assert occurred.mean() == pytest.approx(0.3, abs=0.01)
    assert shocks[occurred].min() >= 25_000 and shocks[occurred].max() <= 75_000
    assert shocks[occurred].mean() == pytest.approx(50_000, rel=0.01)


def test_scheduled_events():
    h = example_household()
    s = Scenario(medical_expense=80_000, medical_expense_month=2,
                 food_price_increase=0.2, food_increase_month=3,
                 income_loss_fraction=0.5, income_loss_start=2, income_loss_duration=2)
    p = generate_paths(h, s, 1, 5, seed=0)
    np.testing.assert_allclose(p.unexpected_expenses[0], [0, 80_000, 0, 0, 0])
    np.testing.assert_allclose(p.income[0], [380_000, 190_000, 190_000, 380_000, 380_000])
    assert p.base_essentials[0, 2] - p.base_essentials[0, 1] == pytest.approx(24_000)


def test_presets_exist_and_apply():
    h = example_household()
    presets = preset_scenarios(h)
    assert len(presets) == 8
    assert apply_scenario(h, presets["Delayed school-fee deadline"]).months_until_due == 8
    low = apply_scenario(h, presets["Low initial savings"])
    assert low.initial_savings == 0 and low.existing_school_savings == 0
    high = apply_scenario(h, presets["High essential-expense burden"])
    assert high.essential_expenses == pytest.approx(1.25 * h.essential_expenses)


def test_invalid_scenario_rejected():
    with pytest.raises(ValueError):
        Scenario(shock_probability=1.5)


# -------------------------------------------------------- reproducibility
def test_same_seed_same_results():
    a = simulate(example_household(), RANDOM, P, 500, horizon=8, seed=123)
    b = simulate(example_household(), RANDOM, P, 500, horizon=8, seed=123)
    for s in STRATEGIES:
        np.testing.assert_array_equal(a.strategies[s].shortfall, b.strategies[s].shortfall)
        np.testing.assert_array_equal(a.strategies[s].total_balance, b.strategies[s].total_balance)


def test_different_seed_different_results():
    a = simulate(example_household(), RANDOM, P, 500, seed=1)
    b = simulate(example_household(), RANDOM, P, 500, seed=2)
    assert not np.array_equal(a.paths.income, b.paths.income)


# ------------------------------------------------------- fair comparison
def test_all_strategies_face_identical_conditions():
    """Common random numbers: one set of paths shared by all strategies, and
    strategy-independent quantities are identical across strategies."""
    r = simulate(example_household(), RANDOM, P, 300, seed=7)
    deficits = summarize(r)["pct_sims_with_cashflow_deficit"]
    assert deficits.nunique() == 1
    # Running a strategy alone gives exactly the same result as in the group.
    alone = simulate(example_household(), RANDOM, P, 300, seed=7, strategies=(PLANNED_SAVINGS,))
    np.testing.assert_array_equal(
        alone.strategies[PLANNED_SAVINGS].shortfall, r.strategies[PLANNED_SAVINGS].shortfall)


# ------------------------------------------------ calculation correctness
def test_deterministic_hand_calculation():
    """Example household, r = 0.6, 5 months, no shocks.
    S1: saves 0.6 * 60k = 36k/month -> 20k + 180k general + 30k edu = 230k
    S2: earmarks 44k/month -> edu 250k; general 20k + 5 * 0.6 * 16k = 68k
    """
    r = simulate(example_household(), Scenario(), P, 1, seed=0)
    s1, s2, s3 = (r.strategies[s] for s in STRATEGIES)
    assert s1.funds_at_deadline[0] == pytest.approx(230_000)
    assert s1.shortfall[0] == pytest.approx(20_000)
    assert s2.funds_at_deadline[0] == pytest.approx(318_000)
    assert s2.shortfall[0] == 0
    assert s2.education_balance[0, 4] == pytest.approx(30_000 + 4 * 44_000)
    np.testing.assert_allclose(s3.funds_at_deadline, s2.funds_at_deadline)  # no cut needed


def test_no_leakage_means_s1_equals_s2():
    params = StrategyParams(remainder_saving_rate=1.0)
    r = simulate(example_household(), RANDOM, params, 400, seed=3)
    np.testing.assert_allclose(
        r.strategies[SAVE_REMAINDER].funds_at_deadline,
        r.strategies[PLANNED_SAVINGS].funds_at_deadline, atol=1e-6)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_accounting_identity_holds(strategy):
    """No money is created or lost: inflows = outflows + change in balances."""
    r = simulate(example_household(), RANDOM, P, 400, horizon=10, seed=11)
    s = r.strategies[strategy]
    p = r.paths
    essentials_paid = (p.essentials - s.unfunded_essentials).sum(axis=1)
    lhs = s.total_balance[:, 0] + p.income.sum(axis=1)
    rhs = (s.total_balance[:, -1] + essentials_paid + s.nonessential_spent.sum(axis=1)
           + s.leakage.sum(axis=1) + s.fees_paid)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-3)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_balances_never_negative_and_fees_consistent(strategy):
    harsh = replace(RANDOM, shock_probability=0.6, shock_severity=300_000)
    r = simulate(example_household(), harsh, P, 500, seed=5)
    s = r.strategies[strategy]
    assert (s.education_balance >= -1e-6).all()
    assert (s.general_balance >= -1e-6).all()
    np.testing.assert_allclose(s.fees_paid + s.shortfall, example_household().school_fees)


# ----------------------------------------------------------- edge cases
def test_income_below_essentials_no_strategy_helps():
    h = make(monthly_income=200_000)
    r = simulate(h, Scenario(), P, 1, seed=0)
    shortfalls = [r.strategies[s].shortfall[0] for s in STRATEGIES]
    assert shortfalls[0] == shortfalls[1] == shortfalls[2] == 250_000
    for s in STRATEGIES:
        assert r.strategies[s].unfunded_essentials.sum() > 0  # recorded, not hidden
    summary = summarize(r)
    assert (summary["success_rate_pct"] == 0).all()
    assert (summary["pct_sims_with_cashflow_deficit"] == 100).all()


def test_zero_savings_household_simulates():
    h = make(initial_savings=0, existing_school_savings=0)
    r = simulate(h, Scenario(), P, 1, seed=0)
    # S2 earmarks 50k/month (250k / 5) out of 60k room -> exactly funded
    assert r.strategies[PLANNED_SAVINGS].funds_at_deadline[0] == pytest.approx(
        250_000 + 0.6 * 10_000 * 5)


def test_already_funded_household():
    h = make(existing_school_savings=300_000)
    calm = simulate(h, Scenario(), P, 1, seed=0)
    for s in STRATEGIES:
        assert calm.strategies[s].education_contribution.sum() == 0
        assert calm.strategies[s].met_deadline.all()
    # Under shocks the fund may be raided for essentials; only then does a
    # planning strategy start rebuilding it.
    r = simulate(h, RANDOM, P, 500, seed=0)
    s2 = r.strategies[PLANNED_SAVINGS]
    contributed = s2.education_contribution.sum(axis=1) > 0
    raided = s2.education_used_for_essentials.sum(axis=1) > 0
    assert not (contributed & ~raided).any()


def test_immediate_deadline_cannot_be_simulated():
    with pytest.raises(ValueError, match="immediately"):
        simulate(make(months_until_due=0), Scenario(), P, 10)
    assert project_deterministic(make(months_until_due=0), P).empty


def test_horizon_shorter_than_deadline_rejected():
    with pytest.raises(ValueError, match="Horizon"):
        simulate(example_household(), Scenario(), P, 10, horizon=3)


def test_horizon_beyond_deadline_runs_and_stops_earmarking():
    r = simulate(example_household(), Scenario(), P, 1, horizon=8, seed=0)
    assert r.strategies[PLANNED_SAVINGS].education_contribution[0, 5:].sum() == 0


def test_unexpected_expenses_reduce_outcomes():
    base = simulate(example_household(), Scenario(income_cv=0.1), P, 2000, seed=9)
    shocked = simulate(example_household(),
                       Scenario(income_cv=0.1, shock_probability=0.3, shock_severity=100_000),
                       P, 2000, seed=9)
    # S1 saves a fixed share of leftovers, so a shock can only lower its funds.
    assert (shocked.strategies[SAVE_REMAINDER].funds_at_deadline
            <= base.strategies[SAVE_REMAINDER].funds_at_deadline + 1e-6).all()
    for s in STRATEGIES:
        assert (shocked.strategies[s].shortfall.mean()
                >= base.strategies[s].shortfall.mean())


def test_medical_expense_scenario_hand_calculation():
    # 80k bill in month 2: income 380k covers it (essentials 350k), but the
    # money left after essentials falls from 110k to 30k that month.
    s = Scenario(medical_expense=80_000, medical_expense_month=2)
    r = simulate(example_household(), s, P, 1, seed=0)
    s1 = r.strategies[SAVE_REMAINDER]
    # month 2: A = 30k, nonessential 30k, nothing left to save
    assert s1.general_saving[0, 1] == 0
    assert s1.funds_at_deadline[0] == pytest.approx(230_000 - 36_000)


def test_protecting_education_fund_records_unfunded_need():
    h = make(monthly_income=250_000)  # 20k monthly gap
    protected = StrategyParams(allow_education_fund_for_essentials=False)
    r = simulate(h, Scenario(), protected, 1, seed=0)
    s = r.strategies[PLANNED_SAVINGS]
    assert s.education_used_for_essentials.sum() == 0
    assert s.education_balance[0, -2] == 30_000  # fund untouched before paying fees
    # 20k general savings cover one month; 4 months x 20k unfunded
    assert s.unfunded_essentials.sum() == pytest.approx(80_000)


def test_prioritization_never_worse_than_planned_on_funds():
    r = simulate(example_household(), RANDOM, P, 2000, seed=4)
    s2 = r.strategies[PLANNED_SAVINGS].funds_at_deadline
    s3 = r.strategies[EXPENSE_PRIORITIZATION].funds_at_deadline
    assert r.strategies[EXPENSE_PRIORITIZATION].met_deadline.mean() >= \
        r.strategies[PLANNED_SAVINGS].met_deadline.mean()
    assert s3.mean() >= s2.mean() - 1e-6


# ---------------------------------------------------------------- metrics
def test_metrics_shapes_and_ranges():
    r = simulate(example_household(), RANDOM, P, 300, horizon=8, seed=0)
    summary = summarize(r)
    assert len(summary) == 3
    assert summary["success_rate_pct"].between(0, 100).all()
    assert (summary["worst_case_shortfall"] >= summary["mean_shortfall"]).all()
    assert (summary["funds_at_deadline_p5"] <= summary["funds_at_deadline_p95"]).all()
    pct = percentile_table(r, "shortfall")
    assert list(pct.columns[:2]) == ["strategy", "worst"]
    comp = paired_comparison(r, PLANNED_SAVINGS, SAVE_REMAINDER)
    total = (comp["both_meet_pct"] + comp["only_a_meets_pct"]
             + comp["only_b_meets_pct"] + comp["neither_meets_pct"])
    assert total == pytest.approx(100)
    assert len(mean_paths(r)) == 3 * 9
    assert len(outcomes_frame(r)) == 900


def test_deterministic_projection_frame():
    df = project_deterministic(example_household(), P)
    assert set(df["strategy"]) == set(STRATEGIES)
    assert df["month"].max() == 5
    s2 = df[(df.strategy == PLANNED_SAVINGS) & (df.month == 4)]
    assert float(s2["education_fund"].iloc[0]) == pytest.approx(206_000)
