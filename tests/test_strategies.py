"""Phase 2 tests: strategy decision rules and the affordability calculator."""

import numpy as np
import pytest

from src.budget import Household, example_household
from src.strategies import (
    EXPENSE_PRIORITIZATION,
    PLANNED_SAVINGS,
    SAVE_REMAINDER,
    STRATEGIES,
    Affordability,
    StrategyParams,
    allocate,
    classify_affordability,
)

P = StrategyParams(remainder_saving_rate=0.5, max_nonessential_cut=0.3)


def make(**overrides) -> Household:
    base = example_household().to_dict()
    base.update(overrides)
    return Household.from_mapping(base)


# --------------------------------------------------------------- allocate
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_allocation_parts_are_nonnegative_and_sum_to_available(strategy):
    rng = np.random.default_rng(0)
    A = rng.uniform(-50_000, 200_000, size=500)  # includes deficits
    planned = rng.uniform(0, 100_000, size=500)
    req = rng.uniform(0, 120_000, size=500)
    a = allocate(strategy, A, planned, req, P)
    total = a.nonessential + a.education + a.general_saving + a.leakage
    np.testing.assert_allclose(total, np.maximum(A, 0), atol=1e-6)
    for part in (a.nonessential, a.education, a.general_saving, a.leakage):
        assert (part >= -1e-9).all()


def test_save_remainder_rule():
    a = allocate(SAVE_REMAINDER, 110_000, 50_000, 44_000, P)
    assert a.nonessential == 50_000
    assert a.education == 0
    assert a.general_saving == pytest.approx(30_000)  # 0.5 * 60k
    assert a.leakage == pytest.approx(30_000)


def test_planned_savings_rule_reserves_requirement_before_leakage():
    a = allocate(PLANNED_SAVINGS, 110_000, 50_000, 44_000, P)
    assert a.nonessential == 50_000
    assert a.education == pytest.approx(44_000)
    assert a.general_saving == pytest.approx(8_000)  # 0.5 * 16k
    assert a.leakage == pytest.approx(8_000)


def test_planned_savings_does_not_cut_nonessentials():
    # only 20k left after nonessentials; requirement 44k
    a = allocate(PLANNED_SAVINGS, 70_000, 50_000, 44_000, P)
    assert a.nonessential == 50_000
    assert a.education == pytest.approx(20_000)


def test_prioritization_cuts_only_as_much_as_needed():
    # need 44k, room is 30k -> cut nonessential by 14k (28% < 30% limit)
    a = allocate(EXPENSE_PRIORITIZATION, 80_000, 50_000, 44_000, P)
    assert a.nonessential == pytest.approx(36_000)
    assert a.education == pytest.approx(44_000)
    assert a.leakage == pytest.approx(0)


def test_prioritization_respects_cut_limit():
    # need 44k, room 20k -> would need a 24k cut but limit is 30% of 50k = 15k
    a = allocate(EXPENSE_PRIORITIZATION, 70_000, 50_000, 44_000, P)
    assert a.nonessential == pytest.approx(35_000)
    assert a.education == pytest.approx(35_000)


def test_prioritization_no_cut_when_not_needed():
    a = allocate(EXPENSE_PRIORITIZATION, 110_000, 50_000, 44_000, P)
    assert a.nonessential == 50_000


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_nothing_available_means_nothing_saved(strategy):
    """Savings can never be created from money needed for essentials."""
    a = allocate(strategy, -30_000, 50_000, 44_000, P)
    assert a.nonessential == 0 and a.education == 0
    assert a.general_saving == 0 and a.leakage == 0


def test_without_leakage_s1_and_s2_save_the_same_total():
    params = StrategyParams(remainder_saving_rate=1.0)
    s1 = allocate(SAVE_REMAINDER, 110_000, 50_000, 44_000, params)
    s2 = allocate(PLANNED_SAVINGS, 110_000, 50_000, 44_000, params)
    assert s1.education + s1.general_saving == pytest.approx(
        s2.education + s2.general_saving
    )


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        allocate("magic", 1, 1, 1, P)


def test_strategy_params_validated():
    with pytest.raises(ValueError):
        StrategyParams(remainder_saving_rate=1.2)


# ----------------------------------------------------------- affordability
def test_feasible_example():
    r = classify_affordability(example_household(), 0.3)
    assert r.category is Affordability.FEASIBLE
    assert r.required_monthly_saving == pytest.approx(44_000)
    assert r.needed_cut_fraction == 0.0


def test_already_funded():
    r = classify_affordability(make(existing_school_savings=300_000))
    assert r.category is Affordability.ALREADY_FUNDED
    assert r.required_monthly_saving == 0


def test_feasible_only_with_cuts():
    # after essentials 110k, nonessential 100k -> 10k/month without cuts.
    # start 50k; need 200k more over 5 months = 40k/month -> allowed N = 70k,
    # i.e. a 30% cut, exactly at the limit.
    h = make(nonessential=100_000)
    r = classify_affordability(h, max_nonessential_cut=0.3)
    assert r.category is Affordability.FEASIBLE_WITH_CUTS
    assert r.needed_cut_fraction == pytest.approx(0.30)
    # a lower limit makes it infeasible
    r2 = classify_affordability(h, max_nonessential_cut=0.2)
    assert r2.category is Affordability.NOT_FEASIBLE
    assert "beyond the stated limit" in r2.explanation


def test_not_feasible_when_income_below_essentials():
    r = classify_affordability(make(monthly_income=200_000), 1.0)
    assert r.category is Affordability.NOT_FEASIBLE
    assert r.needed_cut_fraction is None
    assert "income" in r.explanation.lower()


def test_not_feasible_even_with_zero_nonessential():
    # 10k/month after essentials, fees far away from reach
    h = make(monthly_income=280_000, school_fees=900_000)
    r = classify_affordability(h, 1.0)
    assert r.category is Affordability.NOT_FEASIBLE
    assert r.needed_cut_fraction is None


def test_immediate_deadline_classification():
    covered = make(months_until_due=0, school_fees=40_000)  # 50k held
    assert classify_affordability(covered).category is Affordability.FEASIBLE
    short = make(months_until_due=0)
    r = classify_affordability(short)
    assert r.category is Affordability.NOT_FEASIBLE
    assert "due now" in r.explanation


def test_general_savings_switch_matters():
    # 20k general + 30k edu; fees 40k, due now
    h = make(months_until_due=0, school_fees=40_000)
    r = classify_affordability(h, use_general_savings_for_fees=False)
    assert r.category is Affordability.NOT_FEASIBLE
