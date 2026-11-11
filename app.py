"""SmartSave Nigeria -- Streamlit application.

Run with:  streamlit run app.py

The app is a thin interface over the ``src`` package: every number shown is
computed by the same functions that the tests and experiment scripts use.
No data is sent anywhere; nothing is stored beyond the browser session.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from src import charts
from src.budget import Household, example_household, format_naira, summarize_budget
from src.metrics import mean_paths, outcomes_frame, paired_comparison, percentile_table, summarize
from src.profiles import load_profiles, profile_descriptions
from src.scenarios import Scenario, apply_scenario, preset_scenarios
from src.simulator import deterministic_scenario, project_deterministic, projection_frame, simulate
from src.strategies import (
    DISCLAIMER,
    PLANNED_SAVINGS,
    SAVE_REMAINDER,
    STRATEGIES,
    STRATEGY_LABELS,
    STRATEGY_RULES,
    Affordability,
    StrategyParams,
    classify_affordability,
)
from src.validation import validate_household_inputs, validate_simulation_settings

OUTPUTS = Path(__file__).resolve().parent / "outputs"

st.set_page_config(page_title="SmartSave Nigeria", page_icon="₦", layout="wide")

PAGES = [
    "Home",
    "Household Budget",
    "Education Planner",
    "Strategy Comparison",
    "Economic Simulation",
    "Research & Methodology",
]

GLOSSARY = {
    "Essential expenses": "Spending the household cannot reasonably avoid this month: housing, food, transport to work/school, utilities, healthcare.",
    "Nonessential (discretionary) spending": "Spending that could be reduced without immediate harm, e.g. entertainment, eating out, non-urgent purchases.",
    "Surplus / deficit": "Income minus spending. A deficit means spending is larger than income, so savings must be used or needs go unmet.",
    "Earmarking": "Labelling money for a specific purpose (here: school fees) and keeping it in a separate fund.",
    "Leakage": "Money that was left over but was neither planned nor saved, so it was spent on small unplanned things. Controlled by the remainder saving rate.",
    "Required monthly saving": "(School fees − existing school-fee savings) ÷ months until the deadline.",
    "Shortfall": "The part of the school fees that cannot be paid at the deadline.",
    "Monte Carlo simulation": "Running the same budget many times under randomly varied conditions to see the range of possible outcomes, not just one forecast.",
    "Random seed": "A number that fixes the random draws, so the same inputs and seed always reproduce the same results.",
    "Percentile (P5, P50, P95)": "P5 is the value below which the worst 5% of simulated outcomes fall; P50 is the median.",
    "Coefficient of variation (CV)": "Standard deviation divided by the average. Income CV 20% means income typically moves about 20% above or below its average.",
}

MONEY_INPUTS = [
    ("monthly_income", "Monthly household income (₦)"),
    ("initial_savings", "Initial general savings (₦)"),
    ("rent", "Rent / housing (₦/month)"),
    ("food", "Food (₦/month)"),
    ("transport", "Transportation (₦/month)"),
    ("utilities", "Utilities (₦/month)"),
    ("healthcare", "Healthcare (₦/month)"),
    ("other_essential", "Other essential expenses (₦/month)"),
    ("nonessential", "Nonessential spending (₦/month)"),
    ("school_fees", "School fees due (₦)"),
    ("existing_school_savings", "Existing school-fee savings (₦)"),
]


# --------------------------------------------------------------------- state
def _example_dict() -> dict:
    return example_household().to_dict()


def init_state() -> None:
    if "household" not in st.session_state:
        st.session_state.household = _example_dict()
    # Bumped only on reset/profile load so input widgets are re-created with
    # the new values; ordinary edits keep the same widget keys.
    st.session_state.setdefault("hh_version", 0)


def reset_to_example() -> None:
    st.session_state.household = _example_dict()
    st.session_state.hh_version += 1
    for key in ("p_rate", "p_cut", "p_allow", "p_gen"):
        st.session_state.pop(key, None)


def load_profile(name: str) -> None:
    profiles = load_profiles()
    st.session_state.household = profiles[name].to_dict() | {"name": name}
    st.session_state.hh_version += 1


def current_household() -> tuple[Household | None, list[str], list[str]]:
    raw = st.session_state.household
    report = validate_household_inputs(raw)
    if not report.ok:
        return None, report.errors, report.warnings
    return Household.from_mapping(raw), report.errors, report.warnings


def strategy_params() -> StrategyParams:
    return StrategyParams(
        remainder_saving_rate=st.session_state.get("p_rate", 60) / 100,
        max_nonessential_cut=st.session_state.get("p_cut", 30) / 100,
        allow_education_fund_for_essentials=st.session_state.get("p_allow", True),
        use_general_savings_for_fees=st.session_state.get("p_gen", True),
    )


def require_household() -> Household | None:
    h, errors, _ = current_household()
    if h is None:
        st.error("The household inputs have problems. Fix them on the Household Budget page:")
        for e in errors:
            st.write(f"- {e}")
    return h


def show_warnings(warnings: list[str]) -> None:
    for w in warnings:
        st.warning(w)


def show_fig(fig) -> None:
    st.pyplot(fig, clear_figure=True)
    charts.close(fig)


def money_table(df: pd.DataFrame, money_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in money_cols:
        if c in out:
            out[c] = out[c].map(format_naira)
    return out


# ------------------------------------------------------------------- sidebar
def sidebar() -> str:
    st.sidebar.title("SmartSave Nigeria")
    page = st.sidebar.radio("Navigate", PAGES, key="page")
    st.sidebar.divider()
    st.sidebar.subheader("Strategy assumptions")
    st.sidebar.slider(
        "Remainder saving rate (%)", 0, 100, 60, 5, key="p_rate",
        help="Share of unlabelled leftover money that actually gets saved. The rest "
             "'leaks' into unplanned spending. This is an assumption, not a measured fact.",
    )
    st.sidebar.slider(
        "Maximum nonessential cut for S3 (%)", 0, 100, 30, 5, key="p_cut",
        help="The most that Strategy 3 may reduce nonessential spending in a month.",
    )
    st.sidebar.checkbox(
        "Education fund may pay for essential emergencies", True, key="p_allow",
        help="If unticked, the fund is protected and any unmet essential need is recorded instead.",
    )
    st.sidebar.checkbox("General savings may be used for fees", True, key="p_gen")
    st.sidebar.divider()
    st.sidebar.button("Reset to fictional example", on_click=reset_to_example,
                      use_container_width=True)
    names = list(load_profiles().keys())
    choice = st.sidebar.selectbox("Fictional profile", names, key="profile_choice")
    st.sidebar.button("Load profile", on_click=load_profile, args=(choice,),
                      use_container_width=True)
    st.sidebar.caption(profile_descriptions()[choice])
    st.sidebar.caption("All profiles are fictional and not representative of Nigerian households.")
    return page


# --------------------------------------------------------------------- pages
def page_home() -> None:
    st.title("SmartSave Nigeria")
    st.subheader("A household financial planning and education affordability simulator")
    st.markdown(
        """
This educational tool explores one question:

> **To what extent can structured household budgeting and advance savings planning
> improve educational expense preparedness under different income and economic conditions?**

It compares three transparent budgeting rules on the *same* simulated economic
conditions and shows when planning helps, and when it cannot help because income
is simply too low.

**What it is**
- Explainable formulas and seeded, reproducible simulations (NumPy).
- Fictional example households, clearly labelled.
- No AI, machine learning or automated recommendations: every number can be traced to a formula on the *Research & Methodology* page.

**What it is not**
- Not financial advice and not a forecast of any real household.
- Not empirical evidence: simulations show what follows from the stated assumptions, not what real families do.
- It does not assume hardship comes from poor budgeting. Many households cannot afford fees even with excellent management, and the model shows those cases explicitly.

**How to use it**
1. *Household Budget*: enter (or load) a fictional budget.
2. *Education Planner*: see the required monthly saving and whether the goal is arithmetically reachable.
3. *Strategy Comparison*: compare the three rules without uncertainty.
4. *Economic Simulation*: add randomness and shocks and run a Monte Carlo experiment.
5. *Research & Methodology*: hypotheses, formulas, assumptions and limitations.

No account is needed and no personal information is collected or stored.
"""
    )
    with st.expander("Glossary of financial terms"):
        for term, text in GLOSSARY.items():
            st.markdown(f"**{term}.** {text}")


def page_budget() -> None:
    st.title("Household Budget")
    st.caption("Enter monthly amounts in naira. Use fictional or properly anonymized numbers only.")
    hh = st.session_state.household
    st.write(f"Current profile: **{hh.get('name', 'Custom household')}**")

    v = st.session_state.hh_version
    new = dict(hh)
    cols = st.columns(3)
    for i, (key, label) in enumerate(MONEY_INPUTS):
        with cols[i % 3]:
            new[key] = st.number_input(label, min_value=0.0, value=float(hh[key]),
                                       step=1_000.0, format="%.0f", key=f"in_{key}_{v}")
    with cols[2]:
        new["household_size"] = st.number_input(
            "Household size (people)", 1, 50, int(hh["household_size"]),
            key=f"in_size_{v}", help="Descriptive only; not used in calculations.")
        new["months_until_due"] = st.number_input(
            "Months until school fees are due", 0, 60, int(hh["months_until_due"]),
            key=f"in_months_{v}")
    if new != hh:
        name = hh.get("name", "Custom household")
        if not name.endswith("(edited)"):
            name = f"{name} (edited)"
        st.session_state.household = new | {"name": name}

    h, errors, warnings = current_household()
    if h is None:
        for e in errors:
            st.error(e)
        return
    show_warnings(warnings)

    s = summarize_budget(h)
    st.subheader("Summary")
    c = st.columns(4)
    c[0].metric("Total monthly income", format_naira(s.total_income))
    c[1].metric("Essential expenses", format_naira(s.essential_expenses),
                f"{h.essential_burden:.0%} of income" if h.monthly_income else None,
                delta_color="off")
    c[2].metric("Nonessential expenses", format_naira(s.nonessential_expenses))
    c[3].metric("Monthly surplus / deficit", format_naira(s.monthly_surplus),
                "deficit" if s.is_deficit else "surplus",
                delta_color="inverse" if s.is_deficit else "normal")
    c = st.columns(3)
    c[0].metric(
        "Required monthly education saving" if not s.due_immediately else "Needed immediately",
        format_naira(s.required_monthly_saving))
    c[1].metric("Best-case savings at deadline", format_naira(s.best_case_funds_at_deadline),
                help="Assumes every naira of surplus is saved, constant prices and income, "
                     "and no surprises. An upper bound, not a forecast.")
    c[2].metric("Projected school-fee shortfall", format_naira(s.projected_shortfall),
                help="Fees minus best-case savings at the deadline (zero if covered).")
    if s.unfunded_essentials_best_case > 0:
        st.error(
            f"Even after using all savings, {format_naira(s.unfunded_essentials_best_case)} of "
            "essential needs would go unfunded by the deadline. The model records this "
            "explicitly instead of hiding it as debt."
        )
    show_fig(charts.income_vs_spending(h))
    st.caption(
        "What this shows: where the entered monthly income goes. What it does not show: "
        "how spending changes over time or under shocks (see the Simulation page)."
    )


def page_planner() -> None:
    st.title("Education Planner")
    h = require_household()
    if h is None:
        return
    params = strategy_params()
    a = classify_affordability(h, params.max_nonessential_cut, params.use_general_savings_for_fees)

    c = st.columns(4)
    c[0].metric("School fees", format_naira(h.school_fees))
    c[1].metric("Still needed", format_naira(a.outstanding_need))
    c[2].metric("Months remaining", f"{a.months_remaining}")
    c[3].metric("Required per month" if a.months_remaining else "Required now",
                format_naira(a.required_monthly_saving))
    st.latex(r"\text{Required monthly saving} = \frac{\text{Fees} - \text{Existing education savings}}{\text{Months remaining}}")

    box = {Affordability.ALREADY_FUNDED: st.success, Affordability.FEASIBLE: st.success,
           Affordability.FEASIBLE_WITH_CUTS: st.warning,
           Affordability.NOT_FEASIBLE: st.error}[a.category]
    box(f"**{a.category.value}.** {a.explanation}")
    st.caption(DISCLAIMER)

    t = pd.DataFrame({
        "Spending assumption": ["Usual nonessential spending",
                                f"Nonessential cut by {a.max_cut:.0%} (limit)",
                                "No nonessential spending at all"],
        "Funds usable for fees at deadline": [a.funds_without_cuts, a.funds_with_max_cuts,
                                              a.funds_with_no_nonessential],
    })
    t["Covers fees?"] = t["Funds usable for fees at deadline"] >= h.school_fees - 0.5
    st.dataframe(money_table(t, ["Funds usable for fees at deadline"]), hide_index=True)

    if h.months_until_due == 0:
        st.info("Fees are due now, so there is no savings path to project.")
        return
    st.subheader("Projected savings by strategy (no uncertainty)")
    proj = project_deterministic(h, params)
    show_fig(charts.projected_savings(proj, h.school_fees, h.months_until_due))
    # Funds at the deadline = balances just before the fee payment.
    det = simulate(h, Scenario(), params, 1, seed=0)
    funds = {s: float(det.strategies[s].funds_at_deadline[0]) for s in STRATEGIES}
    show_fig(charts.target_vs_available(funds, h.school_fees))
    st.caption(
        "What these show: the arithmetic path of savings if income and prices stay "
        "exactly as entered. The drop at the deadline is the fee payment. What they do not "
        "establish: what will actually happen, since real months vary (see the Simulation page)."
    )
    st.download_button("Download projection (CSV)", proj.to_csv(index=False),
                       "household_projection.csv", "text/csv")


def page_strategies() -> None:
    st.title("Strategy Comparison")
    h = require_household()
    if h is None:
        return
    params = strategy_params()
    cols = st.columns(3)
    for col, s in zip(cols, STRATEGIES):
        col.markdown(f"**{STRATEGY_LABELS[s]}**")
        col.write(STRATEGY_RULES[s])
    st.info(
        f"All strategies pay essentials first and never borrow. Current assumptions: "
        f"remainder saving rate {params.remainder_saving_rate:.0%}, maximum nonessential "
        f"cut {params.max_nonessential_cut:.0%}. Change them in the sidebar."
    )
    if h.months_until_due == 0:
        st.info("Fees are due now, so strategies have no time to differ.")
        return

    presets = preset_scenarios(h)
    name = st.selectbox("Deterministic scenario (scheduled events only)", list(presets))
    scenario = presets[name]
    st.caption(scenario.description)
    adj = apply_scenario(h, scenario)
    result = simulate(h, deterministic_scenario(scenario), params, 1, seed=0)

    rows = []
    for s, r in result.strategies.items():
        rows.append({
            "Strategy": STRATEGY_LABELS[s],
            "Funds at deadline": r.funds_at_deadline[0],
            "Shortfall": r.shortfall[0],
            "Deadline met": bool(r.met_deadline[0]),
            "Total earmarked contributions": r.education_contribution.sum(),
            "Leakage (unplanned spending)": r.leakage.sum(),
            "Nonessential spending cut": r.nonessential_cut.sum(),
            "Unfunded essential needs": r.unfunded_essentials.sum(),
        })
    df = pd.DataFrame(rows)
    st.dataframe(money_table(df, [c for c in df.columns if c not in ("Strategy", "Deadline met")]),
                 hide_index=True)
    funds = {s: float(r.funds_at_deadline[0]) for s, r in result.strategies.items()}
    show_fig(charts.target_vs_available(funds, adj.school_fees))
    show_fig(charts.projected_savings(projection_frame(result), adj.school_fees, adj.months_until_due))
    st.caption(
        "Why the strategies differ: S2 and S3 protect the required amount from leakage; S3 "
        "can also cut nonessentials within the limit. If the remainder saving rate is 100%, "
        "S1 and S2 reach the same total, so in this model planning helps only by preventing "
        "leakage or by cutting nonessentials. When essentials consume the whole income, all "
        "three strategies give the same result."
    )
    st.download_button("Download comparison (CSV)", df.to_csv(index=False),
                       "strategy_comparison.csv", "text/csv")


SCENARIO_FIELDS = {
    "food_price_increase": ("Food price increase (fraction, e.g. 0.2 = +20%)", 0.0, 3.0, 0.05),
    "food_increase_month": ("Month food prices rise", 1, 24, 1),
    "medical_expense": ("Scheduled medical expense (₦)", 0.0, 1e10, 5_000.0),
    "medical_expense_month": ("Month of medical expense", 1, 24, 1),
    "income_loss_fraction": ("Income lost during loss period (fraction)", 0.0, 1.0, 0.05),
    "income_loss_start": ("Income loss starts in month", 1, 24, 1),
    "income_loss_duration": ("Income loss duration (months)", 0, 24, 1),
    "deadline_delay": ("Delay school-fee deadline by (months)", 0, 24, 1),
    "essential_multiplier": ("Essential-expense multiplier", 0.0, 5.0, 0.05),
    "savings_multiplier": ("Starting-savings multiplier", 0.0, 5.0, 0.05),
}


@st.cache_data(show_spinner=False)
def cached_simulation(hh: dict, scenario: dict, params: dict, n: int, horizon: int, seed: int):
    h = Household.from_mapping(hh)
    return simulate(h, Scenario(**scenario), StrategyParams(**params), n, horizon, seed)


@st.cache_data(show_spinner=False)
def cached_sweep(hh: dict, scenario: dict, params: dict, n: int, horizon: int, seed: int,
                 field: str, values: tuple) -> pd.DataFrame:
    h = Household.from_mapping(hh)
    rows = []
    for v in values:
        if field == "essential_burden":
            scale = v * h.monthly_income / max(h.essential_expenses, 1e-9)
            sc = Scenario(**(scenario | {"essential_multiplier": scale}))
        else:
            sc = Scenario(**(scenario | {field: v}))
        try:
            res = simulate(h, sc, StrategyParams(**params), n, horizon, seed)
        except ValueError:
            continue
        for _, row in summarize(res).iterrows():
            rows.append({field: v, "strategy": row["strategy"],
                         "success_rate_pct": row["success_rate_pct"]})
    return pd.DataFrame(rows)


def page_simulation() -> None:
    st.title("Economic Simulation")
    h = require_household()
    if h is None:
        return
    params = strategy_params()
    presets = preset_scenarios(h)
    preset_name = st.selectbox("Start from scenario", list(presets), key="sim_preset")
    base = presets[preset_name]
    st.caption(base.description + " All values below are editable.")
    pk = list(presets).index(preset_name)

    with st.form("sim_form"):
        c = st.columns(4)
        n = c[0].slider("Number of simulations", 100, 10_000, 2_000, 100)
        horizon = c[1].slider("Forecast horizon (months)", 1, 24,
                              min(24, max(1, h.months_until_due + base.deadline_delay)))
        seed = c[2].number_input("Random seed", 0, 2**32 - 1, 42)
        income_cv = c[3].slider("Income variability (CV, %)", 0, 100,
                                int(round(base.income_cv * 100)) or 15, 5)
        c = st.columns(3)
        shock_p = c[0].slider("Unexpected-expense probability per month (%)", 0, 100,
                              int(round(base.shock_probability * 100)) or 10, 1)
        severity = c[1].number_input(
            "Unexpected-expense severity: average size (₦)", 0.0, 1e10,
            float(base.shock_severity or round(0.15 * h.monthly_income, -3)), 5_000.0,
            format="%.0f", help="Each shock is uniform between 50% and 150% of this amount.")
        inflation = c[2].slider("Monthly expense inflation (%)", -5.0, 10.0,
                                float(base.monthly_inflation * 100), 0.1)
        with st.expander("Scheduled events and structural changes"):
            sc_vals = {}
            cols = st.columns(2)
            for i, (key, (label, lo, hi, step)) in enumerate(SCENARIO_FIELDS.items()):
                default = getattr(base, key)
                with cols[i % 2]:
                    if isinstance(step, int):
                        sc_vals[key] = int(st.number_input(label, int(lo), int(hi), int(default),
                                                           step, key=f"sc{pk}_{key}"))
                    else:
                        sc_vals[key] = float(st.number_input(label, float(lo), float(hi),
                                                             float(default), float(step),
                                                             key=f"sc{pk}_{key}"))
        submitted = st.form_submit_button("Run simulation", type="primary")

    scenario_dict = asdict(base) | sc_vals | {
        "income_cv": income_cv / 100, "shock_probability": shock_p / 100,
        "shock_severity": severity, "monthly_inflation": inflation / 100,
    }
    deadline = h.months_until_due + scenario_dict["deadline_delay"]
    report = validate_simulation_settings(n, horizon, deadline, scenario_dict["income_cv"],
                                          scenario_dict["shock_probability"], severity,
                                          scenario_dict["monthly_inflation"], seed)
    if not report.ok:
        for e in report.errors:
            st.error(e)
        return
    del submitted  # form values only change on submit; results are cached
    try:
        scenario = Scenario(**scenario_dict)
    except ValueError as exc:
        st.error(str(exc))
        return
    sdict, pdict, seed = asdict(scenario), asdict(params), int(seed)
    with st.spinner("Simulating..."):
        res = cached_simulation(h.to_dict(), sdict, pdict, n, horizon, seed)
    st.caption(f"Showing results for: {sdict['name']}, {n:,} simulations, horizon "
               f"{horizon} months, seed {seed}. Same inputs + seed ⇒ identical results.")
    summary = summarize(res)

    st.subheader("Summary (simulated)")
    show = summary[["label", "success_rate_pct", "mean_shortfall", "mean_shortfall_if_failed",
                    "worst_case_shortfall", "median_funds_at_deadline", "median_final_savings",
                    "pct_sims_with_cashflow_deficit", "mean_unfunded_essential_months",
                    "mean_total_leakage", "mean_total_nonessential_cut"]].rename(columns={
        "label": "Strategy", "success_rate_pct": "Deadline met (%)",
        "mean_shortfall": "Avg shortfall", "mean_shortfall_if_failed": "Avg shortfall when failed",
        "worst_case_shortfall": "Worst-case shortfall",
        "median_funds_at_deadline": "Median funds at deadline",
        "median_final_savings": "Median final savings",
        "pct_sims_with_cashflow_deficit": "Sims with a cash-flow deficit (%)",
        "mean_unfunded_essential_months": "Avg months with unfunded essentials",
        "mean_total_leakage": "Avg leakage", "mean_total_nonessential_cut": "Avg nonessential cut",
    })
    money = ["Avg shortfall", "Avg shortfall when failed", "Worst-case shortfall",
             "Median funds at deadline", "Median final savings", "Avg leakage",
             "Avg nonessential cut"]
    st.dataframe(money_table(show.round(2), money), hide_index=True)
    st.caption(
        "A cash-flow deficit month is one where income is below essential costs. It depends "
        "only on the simulated conditions, so it is identical for every strategy: a fairness check."
    )

    comp = paired_comparison(res, PLANNED_SAVINGS, SAVE_REMAINDER)
    st.markdown(
        f"**Paired comparison, S2 vs S1 (same futures):** both meet the deadline in "
        f"{comp['both_meet_pct']:.1f}% of simulations, only S2 in {comp['only_a_meets_pct']:.1f}%, "
        f"only S1 in {comp['only_b_meets_pct']:.1f}%, neither in {comp['neither_meets_pct']:.1f}%."
    )

    st.subheader("Percentiles of funds at the deadline (simulated)")
    st.dataframe(money_table(percentile_table(res, "funds_at_deadline"),
                             ["worst", "P5", "P25", "P50", "P75", "P95", "mean"]), hide_index=True)

    show_fig(charts.strategy_comparison(summary))
    st.caption("Shows: how often each rule met the deadline in these simulations. "
               "Does not show: real-world probabilities; these depend entirely on the assumptions.")
    show_fig(charts.outcome_distribution(res))
    st.caption("Shows: the spread of possible outcomes. The share to the left of the fee line "
               "is the failure rate. Does not show: which outcome will happen.")
    show_fig(charts.savings_fan(mean_paths(res), res.household.school_fees, res.deadline))

    st.subheader("Sensitivity analyses (same seed, smaller runs)")
    sweep_n = min(n, 1_000)
    sweep_base = sdict
    probs = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
    shock_df = cached_sweep(h.to_dict(), sweep_base, pdict, sweep_n, horizon, seed,
                            "shock_probability", probs)
    show_fig(charts.shock_impact(shock_df))
    st.caption(f"Each point re-runs {sweep_n:,} simulations with only the shock probability "
               "changed (common random numbers). Shows how sensitive success is to shocks "
               "of the chosen severity.")
    burdens = tuple(round(x, 2) for x in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05])
    rows = []
    for cv in (0.0, sdict["income_cv"]) if sdict["income_cv"] > 0 else (0.0,):
        df = cached_sweep(h.to_dict(), sweep_base | {"income_cv": cv}, pdict, sweep_n,
                          horizon, seed, "essential_burden", burdens)
        df["income_cv"] = cv
        rows.append(df)
    sens = pd.concat(rows, ignore_index=True)
    show_fig(charts.sensitivity(sens))
    st.caption("Essential costs are rescaled so that they equal the stated share of income; "
               "everything else is unchanged. Shows where planning stops mattering: once "
               "essentials absorb nearly all income, the lines converge.")

    c = st.columns(2)
    c[0].download_button("Download summary (CSV)", summary.to_csv(index=False),
                         "simulation_summary.csv", "text/csv")
    c[1].download_button("Download every simulated outcome (CSV)",
                         outcomes_frame(res).to_csv(index=False),
                         "simulation_outcomes.csv", "text/csv")
    with st.expander("Modeling assumptions for this simulation"):
        st.markdown(ASSUMPTIONS_MD)


ASSUMPTIONS_MD = """
- Income each month = average income × a random multiplier with mean 1 (lognormal, so never negative), times any scheduled income-loss factor. Income does not grow.
- Essential and nonessential costs grow by the monthly inflation rate; school fees are fixed in naira.
- Unexpected expenses occur independently each month with the stated probability; their size is uniform between 50% and 150% of the severity. They are treated as essential.
- Essentials are always paid first: from income, then general savings, then (if allowed) the education fund. Any remaining unmet amount is recorded as an unfunded essential need. There is no borrowing and no negative balance.
- Nonessential spending can never exceed what is left after essentials.
- Of any unlabelled leftover, only the remainder saving rate is saved; the rest is leakage.
- Fees are paid at the end of the deadline month, from the education fund and then (if allowed) general savings. Any unpaid part is the shortfall.
- All strategies face exactly the same simulated months (common random numbers).
- Savings earn no interest; there are no transfers, gifts, loans or extra jobs. These omissions are deliberate simplifications.
"""


def page_research() -> None:
    st.title("Research & Methodology")
    st.markdown("### Research question")
    st.markdown("> To what extent can structured household budgeting and advance savings "
                "planning improve educational expense preparedness under different income and "
                "economic conditions?")
    st.markdown("### Hypotheses (to be tested, not assumed)")
    st.markdown(
        """
- **H1 (leakage).** When some unlabelled leftover money is spent (remainder saving rate < 100%), earmarking (S2) meets the deadline at least as often as saving the remainder (S1). *Reasoning:* earmarked money is protected from leakage.
- **H2 (no free lunch).** When there is no leakage (rate = 100%), S1 and S2 accumulate identical totals. *Reasoning:* the model allocates the same money; only its label changes. This is a property of the model, so the test checks the model, not the world.
- **H3 (shocks).** More frequent or larger unexpected expenses lower success rates for every strategy; the gap between strategies may widen or narrow.
- **H4 (income volatility).** Higher income variability lowers success when the average surplus is close to the requirement.
- **H5 (burden).** The advantage of planning shrinks as essentials approach 100% of income, because the money that strategies can allocate shrinks to zero.
- **H6 (structural unaffordability).** If income after essentials, with zero nonessential spending and no leakage, cannot reach the fees by the deadline, no strategy succeeds in the deterministic case.
"""
    )
    st.markdown("### Mathematical model")
    st.latex(r"A_t = \max(Y_t - X_t,\,0) \qquad R_t = \frac{\max(F - E_t,\,0)}{D - t}")
    st.latex(r"A_t = N_t + C_t + S_t + U_t, \qquad S_t = r\,L_t,\ U_t=(1-r)\,L_t,\ L_t = A_t - N_t - C_t")
    st.markdown(
        r"""
- $Y_t$ income, $X_t$ essential costs, $A_t$ money available after essentials, $F$ fees, $D$ deadline month, $E_t$ education fund, $r$ remainder saving rate.
- **S1:** $N_t=\min(\bar N_t, A_t)$, $C_t=0$. **S2:** $N_t=\min(\bar N_t,A_t)$, $C_t=\min(R_t, A_t-N_t)$. **S3:** $N_t=\min\big(\bar N_t,\max(A_t-R_t,\,(1-\rho)\bar N_t)\big)$ (capped by $A_t$), $C_t=\min(R_t,A_t-N_t)$.
- Income multiplier: $m_t=\exp(\sigma Z_t-\sigma^2/2)$, $\sigma=\sqrt{\ln(1+\mathrm{CV}^2)}$, so $E[m_t]=1$.
- Shortfall at the deadline: $\max\big(F - E_D - G_D\cdot\mathbb 1[\text{general savings allowed}],\,0\big)$.
"""
    )
    st.markdown("### Experimental design")
    st.markdown(
        """
- **Independent variables:** strategy; essential-expense burden; income variability; unexpected-expense probability and severity; remainder saving rate; household profile.
- **Dependent variables:** % of simulations meeting the deadline; mean shortfall; median funds at the deadline; months with unfunded essentials.
- **Controlled:** seed, number of simulations, deadline, fee amount, all other inputs; every strategy runs on identical simulated months.
- **Reproduce:** `python experiments/run_experiments.py`, then `python experiments/make_figures.py`.
"""
    )
    st.markdown("### Assumptions")
    st.markdown(ASSUMPTIONS_MD)
    st.markdown("### Limitations")
    st.markdown(
        """
- Synthetic, fictional inputs; not calibrated to Nigerian data. Results show consequences of assumptions, not real behaviour.
- Leakage is a single fixed parameter; real behaviour is richer (social obligations, family support, informal savings groups, credit).
- No borrowing, interest, transfers, or extra work: real coping strategies are omitted.
- Simulation results are not causal evidence about real households.
"""
    )
    st.markdown("### Experiment outputs")
    csvs = sorted(OUTPUTS.glob("*.csv")) if OUTPUTS.exists() else []
    if not csvs:
        st.info("No experiment outputs yet. Run `python experiments/run_experiments.py`.")
    for path in csvs:
        with st.expander(path.name):
            df = pd.read_csv(path)
            st.dataframe(df.head(200), hide_index=True)
            st.download_button(f"Download {path.name}", path.read_bytes(), path.name,
                               "text/csv", key=f"dl_{path.name}")
    figs = sorted((OUTPUTS / "figures").glob("*.png")) if (OUTPUTS / "figures").exists() else []
    for f in figs:
        st.image(str(f), caption=f.name + " (simulated)")


def main() -> None:
    init_state()
    page = sidebar()
    {
        "Home": page_home,
        "Household Budget": page_budget,
        "Education Planner": page_planner,
        "Strategy Comparison": page_strategies,
        "Economic Simulation": page_simulation,
        "Research & Methodology": page_research,
    }[page]()


main()
