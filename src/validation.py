"""Input validation.

The goal of this module is not only to reject bad input but to *explain*
why a combination of inputs is impossible or economically inconsistent, in
language a beginner can follow.

Two levels of feedback are produced:

* errors   -- the inputs cannot be used (negative money, a fractional number
              of months, missing values). Calculations must not run.
* warnings -- the inputs are valid but describe a difficult or unusual
              situation (income below essential costs, fees already funded,
              fees due immediately). Calculations run, and the warning is
              shown next to the results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

# Monetary fields of a household, with a human-readable label.
MONEY_FIELDS: dict[str, str] = {
    "monthly_income": "Monthly household income",
    "initial_savings": "Initial (general) savings",
    "rent": "Rent / housing",
    "food": "Food",
    "transport": "Transportation",
    "utilities": "Utilities",
    "healthcare": "Healthcare",
    "other_essential": "Other essential expenses",
    "nonessential": "Nonessential spending",
    "school_fees": "School fees due",
    "existing_school_savings": "Existing school-fee savings",
}

ESSENTIAL_FIELDS: tuple[str, ...] = (
    "rent",
    "food",
    "transport",
    "utilities",
    "healthcare",
    "other_essential",
)

# Upper bound on the planning window for the calculator. The Monte Carlo
# simulator has its own (tighter) 1-24 month horizon limit.
MAX_MONTHS_UNTIL_DUE = 60

# A sanity ceiling, not an economic claim: values above this are almost
# certainly typing mistakes (an extra few zeros) for a household budget.
MAX_REASONABLE_NAIRA = 1_000_000_000_000  # one trillion naira


@dataclass
class ValidationReport:
    """Collects errors and warnings produced while checking inputs."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when there are no blocking errors (warnings are allowed)."""
        return not self.errors

    def extend(self, other: "ValidationReport") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


def _as_number(value: Any) -> float | None:
    """Convert a value to float, returning None if impossible or not finite."""
    if isinstance(value, bool):  # bool is a subclass of int; reject it
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _check_integer(
    report: ValidationReport,
    raw: Mapping[str, Any],
    key: str,
    label: str,
    minimum: int,
    maximum: int,
) -> int | None:
    number = _as_number(raw.get(key))
    if number is None:
        report.errors.append(f"{label} must be a number.")
        return None
    if not float(number).is_integer():
        report.errors.append(f"{label} must be a whole number (got {number}).")
        return None
    integer = int(number)
    if integer < minimum or integer > maximum:
        report.errors.append(
            f"{label} must be between {minimum} and {maximum} (got {integer})."
        )
        return None
    return integer


def validate_household_inputs(raw: Mapping[str, Any]) -> ValidationReport:
    """Validate the raw household inputs entered by a user.

    ``raw`` is a mapping such as the one produced by the Streamlit form or a
    row of ``data/example_households.csv``.
    """
    report = ValidationReport()
    values: dict[str, float] = {}

    for key, label in MONEY_FIELDS.items():
        if key not in raw or raw.get(key) is None:
            report.errors.append(f"{label} is missing.")
            continue
        number = _as_number(raw[key])
        if number is None:
            report.errors.append(f"{label} must be a finite number.")
            continue
        if number < 0:
            report.errors.append(
                f"{label} cannot be negative (got {number:,.0f}). Enter costs "
                "and savings as positive amounts; the model works out "
                "deficits itself."
            )
            continue
        if number > MAX_REASONABLE_NAIRA:
            report.errors.append(
                f"{label} looks unrealistically large ({number:,.0f}). "
                "Please check for extra zeros."
            )
            continue
        values[key] = number

    _check_integer(report, raw, "household_size", "Household size", 1, 50)
    months = _check_integer(
        report, raw, "months_until_due", "Months until school fees are due",
        0, MAX_MONTHS_UNTIL_DUE,
    )

    # Consistency warnings only make sense if the numbers themselves parsed.
    if report.errors:
        return report

    income = values["monthly_income"]
    essentials = sum(values[k] for k in ESSENTIAL_FIELDS)
    nonessential = values["nonessential"]
    fees = values["school_fees"]
    edu_savings = values["existing_school_savings"]

    if income == 0:
        report.warnings.append(
            "Monthly income is zero. Every expense must come from savings, "
            "and the model will record unfunded needs once savings run out."
        )
    if essentials > income:
        report.warnings.append(
            f"Essential expenses ({essentials:,.0f}) exceed income "
            f"({income:,.0f}). This is an income problem, not a budgeting "
            "problem: no allocation rule can create savings here, and the "
            "household must draw down savings or leave some needs unfunded."
        )
    elif income > 0 and essentials / income >= 0.9:
        report.warnings.append(
            f"Essential expenses take {essentials / income:.0%} of income. "
            "Very little money is left for any strategy to work with."
        )
    if essentials + nonessential > income >= essentials:
        report.warnings.append(
            "Planned nonessential spending is larger than what remains after "
            "essentials. The model will only let the household spend what it "
            "actually has (no borrowing), so nonessential spending will be "
            "lower than planned."
        )
    if fees == 0:
        report.warnings.append("School fees are zero, so there is no goal to fund.")
    elif edu_savings >= fees:
        report.warnings.append(
            "Existing school-fee savings already cover the fees. The required "
            "monthly saving is zero (provided those savings are not needed "
            "for emergencies)."
        )
    if months == 0 and edu_savings < fees:
        report.warnings.append(
            "Fees are due immediately (0 months). There is no time left to "
            "save; only money already held can be used."
        )
    return report


def validate_strategy_parameters(
    remainder_saving_rate: Any, max_nonessential_cut: Any
) -> ValidationReport:
    """Validate behavioural parameters shared by the budgeting strategies."""
    report = ValidationReport()
    for value, label in (
        (remainder_saving_rate, "Remainder saving rate"),
        (max_nonessential_cut, "Maximum nonessential reduction"),
    ):
        number = _as_number(value)
        if number is None:
            report.errors.append(f"{label} must be a number.")
        elif not 0.0 <= number <= 1.0:
            report.errors.append(f"{label} must be between 0% and 100%.")
    return report


def validate_simulation_settings(
    n_simulations: Any,
    horizon_months: Any,
    deadline_month: Any,
    income_cv: Any,
    shock_probability: Any,
    shock_severity: Any,
    monthly_inflation: Any,
    seed: Any,
    *,
    max_simulations: int = 10_000,
    max_horizon: int = 24,
) -> ValidationReport:
    """Validate Monte Carlo settings (ranges follow the project brief)."""
    report = ValidationReport()
    raw = {
        "n": n_simulations,
        "h": horizon_months,
        "d": deadline_month,
        "seed": seed,
    }
    n = _check_integer(report, raw, "n", "Number of simulations", 1, max_simulations)
    h = _check_integer(report, raw, "h", "Forecast horizon (months)", 1, max_horizon)
    d = _check_integer(
        report, raw, "d", "School-fee deadline month", 0, MAX_MONTHS_UNTIL_DUE
    )
    _check_integer(report, raw, "seed", "Random seed", 0, 2**32 - 1)
    del n  # only validated

    if d is not None and d == 0:
        report.errors.append(
            "Fees are due immediately, so there is nothing to simulate: the "
            "outcome is decided by money already held. See the Education "
            "Planner for the immediate-deadline calculation."
        )
    elif h is not None and d is not None and d > h:
        report.errors.append(
            f"The forecast horizon ({h} months) ends before the school-fee "
            f"deadline (month {d}). Increase the horizon to at least {d}."
        )

    for value, label, low, high in (
        (income_cv, "Income variability", 0.0, 2.0),
        (shock_probability, "Unexpected-expense probability", 0.0, 1.0),
        (monthly_inflation, "Monthly expense inflation", -0.5, 1.0),
    ):
        number = _as_number(value)
        if number is None:
            report.errors.append(f"{label} must be a number.")
        elif not low <= number <= high:
            report.errors.append(f"{label} must be between {low} and {high}.")

    severity = _as_number(shock_severity)
    if severity is None or severity < 0:
        report.errors.append("Unexpected-expense severity must be a non-negative amount.")
    return report
