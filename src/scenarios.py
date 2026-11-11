"""Economic scenarios and random shock generation.

A ``Scenario`` is a bundle of transparent, configurable assumptions about the
economic environment. ``generate_paths`` turns a household + scenario into
month-by-month arrays of income, essential costs and planned nonessential
spending for many simulated "possible futures".

Random components (all drawn with NumPy's ``default_rng``)
----------------------------------------------------------
Income multiplier   m_t ~ LogNormal with E[m_t] = 1 and CV = income_cv.
                    If X = exp(s*Z - s^2/2) with Z ~ N(0,1), then E[X] = 1
                    and CV(X) = sqrt(exp(s^2) - 1); solving for s gives
                    s = sqrt(ln(1 + cv^2)). Income is therefore unbiased on
                    average but can be high or low in any month, and is never
                    negative.
Unexpected expense  Each month, with probability p, an extra essential cost
                    occurs. Its size is uniform between 50% and 150% of the
                    stated severity (so its average equals the severity).

Common random numbers
---------------------
The same three standard random arrays (Z, occurrence draws, size draws) are
generated for a given seed regardless of parameter values, and the same
paths are used for every strategy. Strategies are therefore compared on
identical simulated futures, and changing e.g. the income CV rescales the
same underlying draws rather than drawing new ones.

Scheduled (deterministic) events -- a medical bill in a given month, a food
price jump, a temporary income loss -- are applied on top of the random parts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.budget import Household
from src.validation import ESSENTIAL_FIELDS


@dataclass(frozen=True)
class Scenario:
    """Configurable economic assumptions. All values are editable in the app.

    Months are counted from 1 (the first simulated month).
    """

    name: str = "Stable income and stable expenses"
    description: str = "No randomness and no scheduled shocks."
    # --- random components
    income_cv: float = 0.0  # coefficient of variation of monthly income
    shock_probability: float = 0.0  # chance per month of an unexpected expense
    shock_severity: float = 0.0  # average naira size of an unexpected expense
    monthly_inflation: float = 0.0  # price growth per month (essentials + nonessential)
    # --- scheduled events
    food_price_increase: float = 0.0  # e.g. 0.2 = food costs rise 20%
    food_increase_month: int = 1
    medical_expense: float = 0.0  # one-off essential cost in naira
    medical_expense_month: int = 1
    income_loss_fraction: float = 0.0  # e.g. 0.5 = half of income lost
    income_loss_start: int = 1
    income_loss_duration: int = 0  # months
    # --- structural adjustments to the household
    deadline_delay: int = 0  # extra months before fees are due
    essential_multiplier: float = 1.0  # scales every essential category
    savings_multiplier: float = 1.0  # scales initial and school-fee savings

    def __post_init__(self) -> None:
        checks = [
            (0 <= self.income_cv <= 2, "income_cv must be in [0, 2]"),
            (0 <= self.shock_probability <= 1, "shock_probability must be in [0, 1]"),
            (self.shock_severity >= 0, "shock_severity must be >= 0"),
            (-0.5 <= self.monthly_inflation <= 1, "monthly_inflation must be in [-0.5, 1]"),
            (self.food_price_increase >= -1, "food_price_increase must be >= -1"),
            (self.medical_expense >= 0, "medical_expense must be >= 0"),
            (0 <= self.income_loss_fraction <= 1, "income_loss_fraction must be in [0, 1]"),
            (self.income_loss_duration >= 0, "income_loss_duration must be >= 0"),
            (self.deadline_delay >= 0, "deadline_delay must be >= 0"),
            (self.essential_multiplier >= 0, "essential_multiplier must be >= 0"),
            (self.savings_multiplier >= 0, "savings_multiplier must be >= 0"),
            (
                min(self.food_increase_month, self.medical_expense_month,
                    self.income_loss_start) >= 1,
                "event months start at 1",
            ),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)


def apply_scenario(h: Household, s: Scenario) -> Household:
    """Return the household as modified by the scenario's structural changes."""
    essentials = {k: getattr(h, k) * s.essential_multiplier for k in ESSENTIAL_FIELDS}
    return replace(
        h,
        **essentials,
        initial_savings=h.initial_savings * s.savings_multiplier,
        existing_school_savings=h.existing_school_savings * s.savings_multiplier,
        months_until_due=h.months_until_due + s.deadline_delay,
    )


def preset_scenarios(h: Household) -> dict[str, Scenario]:
    """The eight illustrative scenarios from the project brief.

    Sizes of one-off events are expressed relative to the household's own
    income so that presets make sense for any household. All numbers are
    illustrative assumptions, not estimates of Nigerian conditions.
    """
    income = h.monthly_income
    return {
        "Stable income and stable expenses": Scenario(),
        "Variable monthly income": Scenario(
            name="Variable monthly income",
            description="Income varies randomly month to month (CV 25%), same average.",
            income_cv=0.25,
        ),
        "Unexpected medical expense": Scenario(
            name="Unexpected medical expense",
            description="A one-off medical bill equal to 30% of monthly income in month 2.",
            medical_expense=round(0.30 * income, -3),
            medical_expense_month=2,
        ),
        "Increase in food prices": Scenario(
            name="Increase in food prices",
            description="Food costs jump 20% from month 2 onward.",
            food_price_increase=0.20,
            food_increase_month=2,
        ),
        "Delayed school-fee deadline": Scenario(
            name="Delayed school-fee deadline",
            description="The fee deadline moves 3 months later.",
            deadline_delay=3,
        ),
        "High essential-expense burden": Scenario(
            name="High essential-expense burden",
            description="Every essential category costs 25% more.",
            essential_multiplier=1.25,
        ),
        "Low initial savings": Scenario(
            name="Low initial savings",
            description="The household starts with no general or school-fee savings.",
            savings_multiplier=0.0,
        ),
        "Temporary loss of income": Scenario(
            name="Temporary loss of income",
            description="Half of income is lost for 2 months starting in month 2.",
            income_loss_fraction=0.5,
            income_loss_start=2,
            income_loss_duration=2,
        ),
    }


@dataclass
class ScenarioPaths:
    """Simulated monthly series, shape (n_simulations, horizon)."""

    income: np.ndarray
    base_essentials: np.ndarray  # scheduled essential costs (incl. inflation, food jump)
    unexpected_expenses: np.ndarray  # random + scheduled one-off shocks
    nonessential_planned: np.ndarray

    @property
    def essentials(self) -> np.ndarray:
        return self.base_essentials + self.unexpected_expenses

    @property
    def shape(self) -> tuple[int, int]:
        return self.income.shape


def lognormal_sigma(cv: float) -> float:
    """sigma of log-income giving a mean-one multiplier with the given CV."""
    return float(np.sqrt(np.log1p(cv**2)))


def generate_paths(
    h: Household, s: Scenario, n_simulations: int, horizon: int, seed: int
) -> ScenarioPaths:
    """Generate income and expense paths for an (already adjusted) household.

    ``h`` should normally be ``apply_scenario(original, s)``.
    """
    if n_simulations < 1 or horizon < 1:
        raise ValueError("n_simulations and horizon must be >= 1")
    rng = np.random.default_rng(seed)
    shape = (n_simulations, horizon)
    # Always draw all three arrays in the same order (common random numbers).
    z = rng.standard_normal(shape)
    occurrence_draw = rng.random(shape)
    size_draw = rng.random(shape)

    months = np.arange(1, horizon + 1)  # 1-indexed month numbers
    price_index = (1.0 + s.monthly_inflation) ** (months - 1)

    # Income
    sigma = lognormal_sigma(s.income_cv)
    multiplier = np.exp(sigma * z - 0.5 * sigma**2) if sigma > 0 else np.ones(shape)
    loss_active = (months >= s.income_loss_start) & (
        months < s.income_loss_start + s.income_loss_duration
    )
    income_factor = np.where(loss_active, 1.0 - s.income_loss_fraction, 1.0)
    income = h.monthly_income * multiplier * income_factor

    # Scheduled essential costs: food may jump; everything inflates.
    food_factor = np.where(months >= s.food_increase_month, 1.0 + s.food_price_increase, 1.0)
    other_essentials = h.essential_expenses - h.food
    base_row = (other_essentials + h.food * food_factor) * price_index
    base_essentials = np.broadcast_to(base_row, shape).copy()

    # Unexpected expenses (treated as essential: e.g. medical, repairs).
    occurs = occurrence_draw < s.shock_probability
    sizes = s.shock_severity * (0.5 + size_draw)
    unexpected = np.where(occurs, sizes, 0.0)
    if s.medical_expense > 0 and s.medical_expense_month <= horizon:
        unexpected[:, s.medical_expense_month - 1] += s.medical_expense

    nonessential = np.broadcast_to(h.nonessential * price_index, shape).copy()
    return ScenarioPaths(
        income=income,
        base_essentials=base_essentials,
        unexpected_expenses=unexpected,
        nonessential_planned=nonessential,
    )
