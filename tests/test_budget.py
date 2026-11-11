"""Phase 1 tests: input validation and budget arithmetic."""

import math

import pytest

from src.budget import (
    Household,
    best_case_monthly_net_saving,
    example_household,
    format_naira,
    required_monthly_saving,
    spending_breakdown,
    summarize_budget,
)
from src.validation import (
    validate_household_inputs,
    validate_simulation_settings,
    validate_strategy_parameters,
)


def make(**overrides) -> Household:
    """Build a valid household, overriding selected fields."""
    base = example_household().to_dict()
    base.update(overrides)
    return Household.from_mapping(base)


# ------------------------------------------------------------- formatting
def test_format_naira_thousands_and_symbol():
    assert format_naira(1234567.4) == "₦1,234,567"
    assert format_naira(0) == "₦0"
    assert format_naira(-2500) == "−₦2,500"
    assert format_naira(-0.0001) == "₦0"  # no negative zero
    assert format_naira(1500.256, decimals=2) == "₦1,500.26"


# ------------------------------------------------------------- validation
def test_example_household_is_valid():
    report = example_household().validate()
    assert report.ok, report.errors


@pytest.mark.parametrize(
    "field,value",
    [
        ("monthly_income", -1),
        ("food", float("nan")),
        ("rent", float("inf")),
        ("school_fees", "abc"),
        ("nonessential", None),
        ("existing_school_savings", True),  # booleans are not money
    ],
)
def test_invalid_money_values_are_errors(field, value):
    raw = example_household().to_dict()
    raw[field] = value
    report = validate_household_inputs(raw)
    assert not report.ok
    with pytest.raises(ValueError):
        Household.from_mapping(raw)


def test_missing_field_is_error():
    raw = example_household().to_dict()
    del raw["utilities"]
    report = validate_household_inputs(raw)
    assert any("missing" in e for e in report.errors)


@pytest.mark.parametrize("months", [-1, 2.5, 61])
def test_invalid_months(months):
    raw = example_household().to_dict()
    raw["months_until_due"] = months
    assert not validate_household_inputs(raw).ok


def test_household_size_must_be_positive_integer():
    raw = example_household().to_dict()
    raw["household_size"] = 0
    assert not validate_household_inputs(raw).ok


def test_income_below_essentials_is_warning_not_error():
    raw = example_household().to_dict()
    raw["monthly_income"] = 200_000  # essentials are 270,000
    report = validate_household_inputs(raw)
    assert report.ok
    assert any("income problem" in w for w in report.warnings)


def test_fees_already_funded_warning():
    raw = example_household().to_dict()
    raw["existing_school_savings"] = 300_000
    report = validate_household_inputs(raw)
    assert report.ok and any("already cover" in w for w in report.warnings)


def test_immediate_deadline_warning():
    raw = example_household().to_dict()
    raw["months_until_due"] = 0
    report = validate_household_inputs(raw)
    assert report.ok and any("immediately" in w for w in report.warnings)


def test_strategy_parameter_validation():
    assert validate_strategy_parameters(0.5, 0.3).ok
    assert not validate_strategy_parameters(1.5, 0.3).ok
    assert not validate_strategy_parameters(0.5, -0.1).ok


def test_simulation_settings_validation():
    ok = validate_simulation_settings(1000, 12, 6, 0.1, 0.1, 50_000, 0.01, 42)
    assert ok.ok
    # horizon shorter than deadline
    assert not validate_simulation_settings(1000, 4, 6, 0.1, 0.1, 50_000, 0.01, 42).ok
    # too many simulations
    assert not validate_simulation_settings(20_000, 12, 6, 0.1, 0.1, 0, 0.0, 1).ok
    # immediate deadline cannot be simulated
    assert not validate_simulation_settings(100, 12, 0, 0.1, 0.1, 0, 0.0, 1).ok
    # probability outside [0, 1]
    assert not validate_simulation_settings(100, 12, 6, 0.1, 1.2, 0, 0.0, 1).ok


# ------------------------------------------------------ required saving
def test_required_monthly_saving_formula():
    assert required_monthly_saving(250_000, 30_000, 5) == pytest.approx(44_000)


def test_required_saving_when_already_funded():
    assert required_monthly_saving(100_000, 150_000, 4) == 0.0
    assert required_monthly_saving(100_000, 100_000, 0) == 0.0


def test_required_saving_immediate_deadline_returns_full_outstanding():
    assert required_monthly_saving(100_000, 40_000, 0) == 60_000


def test_required_saving_negative_months_rejected():
    with pytest.raises(ValueError):
        required_monthly_saving(100_000, 0, -1)


# ------------------------------------------------------- summary numbers
def test_summary_example_values():
    s = summarize_budget(example_household())
    assert s.total_income == 380_000
    assert s.essential_expenses == 270_000
    assert s.nonessential_expenses == 50_000
    assert s.surplus_after_essentials == 110_000
    assert s.monthly_surplus == 60_000
    assert s.required_monthly_saving == pytest.approx(44_000)
    # best case: 20k + 30k + 5 * 60k = 350k
    assert s.best_case_funds_at_deadline == pytest.approx(350_000)
    assert s.projected_shortfall == 0
    assert not s.is_deficit


def test_zero_savings_household():
    h = make(initial_savings=0, existing_school_savings=0)
    s = summarize_budget(h)
    assert s.outstanding_need == 250_000
    assert s.best_case_funds_at_deadline == pytest.approx(300_000)


def test_negative_cash_flow_drains_savings_then_records_unfunded_need():
    # essentials 270k, income 250k -> -20k per month, start with 50k total
    h = make(monthly_income=250_000)
    assert best_case_monthly_net_saving(h) == -20_000
    s = summarize_budget(h)
    assert s.is_deficit
    # 50k - 5 * 20k = -50k -> funds floor at 0, 50k unfunded, full shortfall
    assert s.best_case_funds_at_deadline == 0
    assert s.unfunded_essentials_best_case == pytest.approx(50_000)
    assert s.projected_shortfall == 250_000


def test_nonessential_overspend_cannot_create_negative_saving():
    # 110k left after essentials but 150k nonessential planned
    h = make(nonessential=150_000)
    assert best_case_monthly_net_saving(h) == 0.0
    s = summarize_budget(h)
    assert s.monthly_surplus == -40_000  # as budgeted
    assert s.best_case_funds_at_deadline == 50_000  # savings untouched


def test_immediate_deadline_summary():
    h = make(months_until_due=0)
    s = summarize_budget(h)
    assert s.due_immediately
    assert s.best_case_funds_at_deadline == 50_000
    assert s.projected_shortfall == 200_000


def test_essential_burden():
    assert example_household().essential_burden == pytest.approx(270 / 380)
    assert math.isinf(make(monthly_income=0).essential_burden)


def test_spending_breakdown_sums_to_total_spending():
    h = example_household()
    assert sum(spending_breakdown(h).values()) == pytest.approx(
        h.essential_expenses + h.nonessential
    )
