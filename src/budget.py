"""Household data model and monthly budget arithmetic.

All amounts are in Nigerian naira (NGN) per month unless stated otherwise.

Key definitions (used consistently across the project)
------------------------------------------------------
Essential expenses   E_ss = rent + food + transport + utilities + healthcare
                            + other essentials
Surplus after essentials       S_ess = income - E_ss        (may be negative)
Monthly surplus / deficit      S     = income - E_ss - nonessential
Outstanding education need     need  = max(fees - existing school savings, 0)
Required monthly saving        R     = need / months remaining

The "best-case" projection in this module assumes every naira of surplus is
saved, income and prices never change, and nothing unexpected happens. It is
an *upper bound* used for orientation, not a forecast. Strategy-specific and
uncertain projections live in ``strategies`` and ``simulator``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from src.validation import (
    ESSENTIAL_FIELDS,
    ValidationReport,
    validate_household_inputs,
)

NAIRA = "₦"  # ₦


def format_naira(amount: float, decimals: int = 0) -> str:
    """Format an amount as naira with thousands separators.

    >>> format_naira(1234567.4)
    '₦1,234,567'
    >>> format_naira(-2500)
    '−₦2,500'
    """
    rounded = round(float(amount), decimals)
    if rounded == 0:
        rounded = 0.0  # avoid "-₦0" from tiny negative floats
    sign = "−" if rounded < 0 else ""  # proper minus sign
    return f"{sign}{NAIRA}{abs(rounded):,.{decimals}f}"


@dataclass(frozen=True)
class Household:
    """A (fictional) household's monthly budget and school-fee goal."""

    monthly_income: float
    initial_savings: float
    household_size: int
    rent: float
    food: float
    transport: float
    utilities: float
    healthcare: float
    other_essential: float
    nonessential: float
    school_fees: float
    months_until_due: int
    existing_school_savings: float
    name: str = "Custom household"

    # ------------------------------------------------------------------ build
    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Household":
        """Validate raw inputs and build a Household.

        Raises ``ValueError`` listing every blocking problem if the inputs are
        invalid. Warnings are not raised; call ``validate_household_inputs``
        separately to display them.
        """
        report = validate_household_inputs(raw)
        if not report.ok:
            raise ValueError("Invalid household inputs:\n- " + "\n- ".join(report.errors))
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name == "name":
                kwargs["name"] = str(raw.get("name", "Custom household"))
            elif f.name in ("household_size", "months_until_due"):
                kwargs[f.name] = int(float(raw[f.name]))
            else:
                kwargs[f.name] = float(raw[f.name])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> ValidationReport:
        return validate_household_inputs(self.to_dict())

    # ------------------------------------------------------------ properties
    @property
    def essential_expenses(self) -> float:
        return float(sum(getattr(self, k) for k in ESSENTIAL_FIELDS))

    @property
    def essential_breakdown(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in ESSENTIAL_FIELDS}

    @property
    def essential_burden(self) -> float:
        """Share of income consumed by essentials (inf if income is zero)."""
        if self.monthly_income == 0:
            return float("inf") if self.essential_expenses > 0 else 0.0
        return self.essential_expenses / self.monthly_income


def required_monthly_saving(
    school_fees: float, existing_savings: float, months_remaining: int
) -> float:
    """Required Monthly Saving = (Fees − Existing Education Savings) / Months.

    Edge cases:
    * Existing savings already cover the fees  -> 0.
    * months_remaining == 0 (due immediately)  -> the whole outstanding
      amount, because it is all needed *now* rather than spread over months.
    """
    if months_remaining < 0:
        raise ValueError("months_remaining cannot be negative")
    outstanding = max(float(school_fees) - float(existing_savings), 0.0)
    if outstanding == 0:
        return 0.0
    if months_remaining == 0:
        return outstanding
    return outstanding / months_remaining


@dataclass(frozen=True)
class BudgetSummary:
    """Headline numbers for the Household Budget dashboard."""

    total_income: float
    essential_expenses: float
    nonessential_expenses: float
    surplus_after_essentials: float
    monthly_surplus: float  # negative = monthly deficit as budgeted
    outstanding_need: float
    required_monthly_saving: float
    best_case_funds_at_deadline: float
    projected_shortfall: float
    unfunded_essentials_best_case: float
    due_immediately: bool

    @property
    def is_deficit(self) -> bool:
        return self.monthly_surplus < 0


def best_case_monthly_net_saving(h: Household) -> float:
    """Net change in savings per month if all surplus is saved.

    * Income covers essentials: the household can save
      ``max(income - essentials - nonessential, 0)``. Nonessential spending
      can never exceed what is left after essentials (no borrowing), so a
      planned nonessential overspend simply means zero saving, not a loss.
    * Income does not cover essentials: savings fall by the essential gap.
    """
    after_essentials = h.monthly_income - h.essential_expenses
    if after_essentials >= 0:
        return max(after_essentials - h.nonessential, 0.0)
    return after_essentials  # negative: savings are drawn down


def summarize_budget(h: Household) -> BudgetSummary:
    """Compute dashboard figures under the best-case (all surplus saved) rule."""
    essentials = h.essential_expenses
    after_essentials = h.monthly_income - essentials
    monthly_surplus = after_essentials - h.nonessential
    outstanding = max(h.school_fees - h.existing_school_savings, 0.0)
    required = required_monthly_saving(
        h.school_fees, h.existing_school_savings, h.months_until_due
    )

    # Total money available for fees at the deadline in the best case. If the
    # monthly net is negative, savings are drawn down and cannot go below 0;
    # anything beyond that is an unfunded essential need (not hidden debt).
    start = h.initial_savings + h.existing_school_savings
    net = best_case_monthly_net_saving(h)
    raw_end = start + net * h.months_until_due
    funds = max(raw_end, 0.0)
    unfunded = max(-raw_end, 0.0)
    shortfall = max(h.school_fees - funds, 0.0)

    return BudgetSummary(
        total_income=h.monthly_income,
        essential_expenses=essentials,
        nonessential_expenses=h.nonessential,
        surplus_after_essentials=after_essentials,
        monthly_surplus=monthly_surplus,
        outstanding_need=outstanding,
        required_monthly_saving=required,
        best_case_funds_at_deadline=funds,
        projected_shortfall=shortfall,
        unfunded_essentials_best_case=unfunded,
        due_immediately=(h.months_until_due == 0 and outstanding > 0),
    )


def spending_breakdown(h: Household) -> dict[str, float]:
    """Monthly outflows by category, used by the income-vs-spending chart."""
    labels = {
        "rent": "Rent / housing",
        "food": "Food",
        "transport": "Transport",
        "utilities": "Utilities",
        "healthcare": "Healthcare",
        "other_essential": "Other essentials",
    }
    out = {labels[k]: v for k, v in h.essential_breakdown.items()}
    out["Nonessential"] = h.nonessential
    return out


def example_household() -> Household:
    """A FICTIONAL example household used as the app's reset/default values.

    The numbers are invented for illustration. They are not calibrated to,
    and should not be read as representative of, Nigerian households.
    """
    return Household(
        name="Fictional example: tight but workable",
        monthly_income=380_000,
        initial_savings=20_000,
        household_size=5,
        rent=70_000,
        food=120_000,
        transport=35_000,
        utilities=20_000,
        healthcare=10_000,
        other_essential=15_000,
        nonessential=50_000,
        school_fees=250_000,
        months_until_due=5,
        existing_school_savings=30_000,
    )
