"""Phase 4 smoke tests: every Streamlit page renders without errors.

Uses Streamlit's built-in headless testing API (no browser needed).
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.charts import (
    income_vs_spending,
    outcome_distribution,
    projected_savings,
    sensitivity,
    shock_impact,
    strategy_comparison,
    target_vs_available,
)

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["Home", "Household Budget", "Education Planner", "Strategy Comparison",
         "Economic Simulation", "Research & Methodology"]


def fresh_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = fresh_app()
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, at.exception
    expected = "SmartSave Nigeria" if page == "Home" else page
    assert any(expected in t.value for t in at.title)


def test_structurally_insufficient_profile_is_flagged():
    at = fresh_app()
    at.sidebar.selectbox(key="profile_choice").set_value(
        "D: Structurally insufficient income").run()
    at.sidebar.button[1].click().run()  # "Load profile"
    at.sidebar.radio(key="page").set_value("Education Planner").run()
    assert not at.exception
    assert any("Not feasible" in e.value for e in at.error)


def test_reset_restores_fictional_example():
    at = fresh_app()
    at.sidebar.selectbox(key="profile_choice").set_value("A: Comfortable margin").run()
    at.sidebar.button[1].click().run()
    assert at.session_state.household["monthly_income"] == 600_000
    at.sidebar.button[0].click().run()  # "Reset to fictional example"
    assert at.session_state.household["monthly_income"] == 380_000


def test_horizon_shorter_than_deadline_is_explained():
    at = fresh_app()
    at.sidebar.radio(key="page").set_value("Economic Simulation").run()
    at.slider[1].set_value(2)  # forecast horizon below the 5-month deadline
    at.button[0].click().run()  # form submit
    assert not at.exception
    assert any("horizon" in e.value.lower() for e in at.error)


def test_chart_functions_return_figures():
    import pandas as pd

    from src.budget import example_household
    from src.metrics import summarize
    from src.scenarios import Scenario
    from src.simulator import project_deterministic, simulate
    from src.strategies import STRATEGIES, StrategyParams

    h = example_household()
    p = StrategyParams()
    res = simulate(h, Scenario(income_cv=0.2, shock_probability=0.1, shock_severity=50_000),
                   p, 200, seed=1)
    figs = [
        income_vs_spending(h),
        projected_savings(project_deterministic(h, p), h.school_fees, 5),
        target_vs_available({s: 1.0 for s in STRATEGIES}, h.school_fees),
        strategy_comparison(summarize(res)),
        outcome_distribution(res),
        shock_impact(pd.DataFrame({"shock_probability": [0, 0.1] * 3,
                                   "strategy": [s for s in STRATEGIES for _ in (0, 1)],
                                   "success_rate_pct": [100, 90] * 3})),
        sensitivity(pd.DataFrame({"essential_burden": [0.5, 0.9] * 3, "income_cv": 0.0,
                                  "strategy": [s for s in STRATEGIES for _ in (0, 1)],
                                  "success_rate_pct": [100, 0] * 3})),
    ]
    for fig in figs:
        ax = fig.axes[0]
        assert ax.get_title(loc="left"), "every chart needs a title"
        assert ax.get_xlabel(), "every chart needs an x-axis label"
