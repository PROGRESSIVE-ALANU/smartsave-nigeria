"""Budgeting strategies and the education affordability calculator.

How a month works (shared by every strategy)
--------------------------------------------
1. Income arrives.
2. Essential expenses are paid first. If income is not enough, the gap is
   covered from general savings, then (optionally) from the education fund;
   anything still uncovered is recorded as an *unfunded essential need*.
   No strategy can touch money needed for essentials.
3. Money available after essentials:  A = max(income - essentials, 0).
4. The strategy splits A into four non-negative parts that always add up
   exactly to A:

       N  nonessential spending actually made
       C  contribution to the earmarked education fund
       S  saving into general savings
       U  "leakage": leftover money that is neither planned nor saved

Leakage and the remainder saving rate r
---------------------------------------
Money that is left over *without a label* tends to get spent on small,
unplanned things. The model represents this with one transparent parameter,
the remainder saving rate r (0-100%): of any unlabelled leftover L, the
household saves r*L and the rest, (1-r)*L, leaks away. r is an assumption,
not a measured fact; the experiments vary it. When r = 1 there is no leakage
and Strategy 1 and Strategy 2 accumulate the same total savings, which is an
important honesty check on the model.

The three decision rules
------------------------
S1 Save the Remainder
    N = min(planned nonessential, A);  C = 0;  leftover L = A - N.
S2 Planned Education Savings
    N = min(planned nonessential, A);  C = min(R, A - N);  L = A - N - C.
    R is the required monthly saving, recalculated each month from the
    education fund's current balance and the months left. The earmarked part
    is protected from leakage; nonessential spending is not cut.
S3 Expense Prioritization
    Like S2, but if A - N is too small to fund R, nonessential spending is
    reduced -- only as much as needed and never below
    (1 - max_cut) * planned nonessential. C = min(R, A - N).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.budget import Household, required_monthly_saving

SAVE_REMAINDER = "save_remainder"
PLANNED_SAVINGS = "planned_savings"
EXPENSE_PRIORITIZATION = "expense_prioritization"

STRATEGIES: tuple[str, ...] = (SAVE_REMAINDER, PLANNED_SAVINGS, EXPENSE_PRIORITIZATION)

STRATEGY_LABELS: dict[str, str] = {
    SAVE_REMAINDER: "S1: Save the Remainder",
    PLANNED_SAVINGS: "S2: Planned Education Savings",
    EXPENSE_PRIORITIZATION: "S3: Expense Prioritization",
}

STRATEGY_RULES: dict[str, str] = {
    SAVE_REMAINDER: (
        "Pay essentials, spend the usual amount on nonessentials, and save a "
        "share (the remainder saving rate) of whatever is left over. Nothing "
        "is set aside specifically for school fees in advance."
    ),
    PLANNED_SAVINGS: (
        "Pay essentials and usual nonessentials, then move the required "
        "monthly amount into a separate school-fee fund before anything "
        "else happens to the leftover. If there is not enough left, put in "
        "what is possible and recalculate next month. Nonessential spending "
        "is not cut."
    ),
    EXPENSE_PRIORITIZATION: (
        "Same as Planned Education Savings, but if the leftover is not "
        "enough for the required amount, cut nonessential spending -- only "
        "as much as needed, and never more than the stated maximum cut. "
        "Essential spending is never cut."
    ),
}


@dataclass(frozen=True)
class StrategyParams:
    """Behavioural and accounting assumptions shared by all strategies.

    remainder_saving_rate
        Share of unlabelled leftover money that is actually saved (r).
    max_nonessential_cut
        Largest fraction by which S3 may reduce nonessential spending.
    allow_education_fund_for_essentials
        If True, the education fund can be drawn on to pay essential needs
        once general savings are exhausted (realistic in emergencies). If
        False, the fund is protected and the unmet need is recorded instead.
    use_general_savings_for_fees
        If True, general savings may be used to pay fees at the deadline.
    """

    remainder_saving_rate: float = 0.6
    max_nonessential_cut: float = 0.3
    allow_education_fund_for_essentials: bool = True
    use_general_savings_for_fees: bool = True

    def __post_init__(self) -> None:
        for name in ("remainder_saving_rate", "max_nonessential_cut"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass
class Allocation:
    """Result of splitting money available after essentials (arrays)."""

    nonessential: np.ndarray
    education: np.ndarray
    general_saving: np.ndarray
    leakage: np.ndarray


def allocate(
    strategy: str,
    available: np.ndarray,
    nonessential_planned: np.ndarray,
    required: np.ndarray,
    params: StrategyParams,
) -> Allocation:
    """Apply one strategy's decision rule to money left after essentials.

    All inputs are arrays of the same shape (one entry per simulated
    household) or scalars. Guarantees, for every entry:
        N + C + S + U == A,   and all four parts are >= 0.
    """
    A = np.maximum(np.asarray(available, dtype=float), 0.0)
    planned = np.maximum(np.asarray(nonessential_planned, dtype=float), 0.0)
    req = np.maximum(np.asarray(required, dtype=float), 0.0)
    A, planned, req = np.broadcast_arrays(A, planned, req)

    full_nonessential = np.minimum(planned, A)  # cannot spend money you lack

    if strategy == SAVE_REMAINDER:
        N = full_nonessential
        C = np.zeros_like(A)
    elif strategy == PLANNED_SAVINGS:
        N = full_nonessential
        C = np.minimum(req, A - N)
    elif strategy == EXPENSE_PRIORITIZATION:
        floor = np.minimum(planned * (1.0 - params.max_nonessential_cut), A)
        room = A - full_nonessential
        # Cut just enough to fund `req` (A - req), but not below the floor.
        cut_level = np.maximum(floor, A - req)
        N = np.where(room >= req, full_nonessential, np.minimum(cut_level, full_nonessential))
        C = np.minimum(req, A - N)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    leftover = np.maximum(A - N - C, 0.0)  # clip tiny negative float noise
    S = params.remainder_saving_rate * leftover
    U = leftover - S
    return Allocation(nonessential=N, education=C, general_saving=S, leakage=U)


# ---------------------------------------------------------------------------
# Affordability calculator
# ---------------------------------------------------------------------------
class Affordability(str, Enum):
    ALREADY_FUNDED = "Already funded"
    FEASIBLE = "Feasible under the modeled conditions"
    FEASIBLE_WITH_CUTS = "Feasible only with reduced discretionary expenses"
    NOT_FEASIBLE = "Not feasible without additional resources or reduced obligations"


DISCLAIMER = (
    "This classification assumes constant income and prices, no unexpected "
    "expenses, and that every naira not spent is saved. It describes the "
    "arithmetic of the entered numbers; it is not a guarantee of future "
    "affordability. Use the Economic Simulation page to see how uncertainty "
    "changes the picture."
)


@dataclass(frozen=True)
class AffordabilityResult:
    category: Affordability
    explanation: str
    outstanding_need: float
    required_monthly_saving: float
    months_remaining: int
    surplus_after_essentials: float
    funds_without_cuts: float
    funds_with_max_cuts: float
    funds_with_no_nonessential: float
    needed_cut_fraction: float | None  # None = cuts alone cannot close the gap
    max_cut: float


def _funds_at_deadline(h: Household, nonessential: float, use_general: bool) -> float:
    """Money usable for fees at the deadline if nonessential spending is
    held at ``nonessential`` and every remaining naira is earmarked."""
    after = h.monthly_income - h.essential_expenses
    months = h.months_until_due
    if after >= 0:
        net = max(after - nonessential, 0.0)
        edu_end = h.existing_school_savings + net * months
        gen_end = h.initial_savings
    else:
        # Essentials exceed income: general savings are drawn first, then the
        # education fund. Neither can go below zero (no borrowing).
        drain = -after * months
        gen_end = max(h.initial_savings - drain, 0.0)
        remaining = max(drain - h.initial_savings, 0.0)
        edu_end = max(h.existing_school_savings - remaining, 0.0)
    return edu_end + (gen_end if use_general else 0.0)


def classify_affordability(
    h: Household,
    max_nonessential_cut: float = 0.3,
    use_general_savings_for_fees: bool = True,
) -> AffordabilityResult:
    """Classify whether the school-fee goal is arithmetically reachable."""
    if not 0.0 <= max_nonessential_cut <= 1.0:
        raise ValueError("max_nonessential_cut must be in [0, 1]")

    fees = h.school_fees
    months = h.months_until_due
    after = h.monthly_income - h.essential_expenses
    outstanding = max(fees - h.existing_school_savings, 0.0)
    required = required_monthly_saving(fees, h.existing_school_savings, months)
    use_g = use_general_savings_for_fees

    f_none = _funds_at_deadline(h, h.nonessential, use_g)
    f_cut = _funds_at_deadline(h, h.nonessential * (1 - max_nonessential_cut), use_g)
    f_zero = _funds_at_deadline(h, 0.0, use_g)

    # Smallest cut fraction c with funds(N0 * (1 - c)) >= fees.
    needed_cut: float | None
    if f_none >= fees:
        needed_cut = 0.0
    elif months == 0 or after <= 0 or f_zero < fees or h.nonessential == 0:
        needed_cut = None
    else:
        start = h.existing_school_savings + (h.initial_savings if use_g else 0.0)
        required_net = (fees - start) / months
        allowed_nonessential = after - required_net
        needed_cut = float(np.clip(1 - allowed_nonessential / h.nonessential, 0.0, 1.0))

    if outstanding == 0:
        category = Affordability.ALREADY_FUNDED
        text = (
            "Existing school-fee savings already cover the fees. The goal is "
            "met as long as those savings are not needed for emergencies "
            "before the deadline."
        )
    elif f_none >= fees:
        category = Affordability.FEASIBLE
        if months == 0:
            text = "Fees are due now and the money already held covers them."
        else:
            text = (
                f"After essentials and usual nonessential spending, the "
                f"household could set aside enough each month (required: "
                f"₦{required:,.0f}/month) to reach the fees by the deadline."
            )
    elif f_cut >= fees:
        category = Affordability.FEASIBLE_WITH_CUTS
        text = (
            f"The goal is only reachable if nonessential spending is reduced "
            f"by about {needed_cut:.0%} (the stated limit is "
            f"{max_nonessential_cut:.0%}). Essential spending is not cut."
        )
    else:
        category = Affordability.NOT_FEASIBLE
        if after < 0:
            text = (
                "Income does not cover essential expenses, so no budgeting "
                "rule can create savings. This is an income or obligations "
                "problem: it would need more income, support, a lower fee, "
                "a later deadline, or lower essential costs."
            )
        elif months == 0:
            text = (
                "Fees are due now and the money already held is not enough. "
                "There is no time left for any budgeting strategy to help."
            )
        elif needed_cut is not None:
            text = (
                f"Reaching the fees would require cutting nonessential "
                f"spending by about {needed_cut:.0%}, beyond the stated "
                f"limit of {max_nonessential_cut:.0%}."
            )
        else:
            text = (
                "Even with zero nonessential spending, the money left after "
                "essentials is not enough by the deadline. Budgeting alone "
                "cannot close this gap; additional resources or reduced "
                "obligations would be needed."
            )

    return AffordabilityResult(
        category=category,
        explanation=text,
        outstanding_need=outstanding,
        required_monthly_saving=required,
        months_remaining=months,
        surplus_after_essentials=after,
        funds_without_cuts=f_none,
        funds_with_max_cuts=f_cut,
        funds_with_no_nonessential=f_zero,
        needed_cut_fraction=needed_cut,
        max_cut=max_nonessential_cut,
    )
