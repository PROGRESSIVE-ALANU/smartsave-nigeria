"""Regenerate figures and summary tables from the experiment CSV outputs.

Usage (from the project root, after run_experiments.py):
    python experiments/make_figures.py

Reads outputs/e*.csv and writes:
    outputs/figures/*.png   charts (all labelled as simulated)
    outputs/tables/*.md     Markdown tables to paste into the report template

This script performs no simulation itself; it only reshapes and plots the
numbers that run_experiments.py produced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import charts  # noqa: E402
from src.strategies import STRATEGY_LABELS  # noqa: E402

OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"


def set_output_dir(path: Path) -> None:
    global OUT, FIG, TAB
    OUT, FIG, TAB = path, path / "figures", path / "tables"


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal Markdown table writer (avoids an extra dependency)."""
    def fmt(v):
        if isinstance(v, float):
            if abs(v) >= 1000:
                return f"{v:,.0f}"
            return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"
        return str(v)
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows]) + "\n"


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    charts.close(fig)
    print(f"figure  {name}")


def write_table(df: pd.DataFrame, name: str, title: str) -> None:
    text = (f"### {title}\n\n*Simulated results from fictional households; not empirical "
            f"observations.*\n\n" + to_markdown(df))
    (TAB / name).write_text(text)
    print(f"table   {name}")


def short(s: str) -> str:
    return STRATEGY_LABELS.get(s, s).split(":")[0]


def tidy(t: pd.DataFrame) -> pd.DataFrame:
    """Rename strategy columns to S1/S2/S3 and put them in that order."""
    t = t.reset_index() if t.index.name or isinstance(t.index, pd.MultiIndex) else t
    ids = [c for c in t.columns if c not in STRATEGY_LABELS]
    strat = [c for c in STRATEGY_LABELS if c in t.columns]
    t = t[ids + strat]
    t.columns = ids + [short(c) for c in strat]
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate figures and tables.")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs",
                        help="folder containing the experiment CSVs")
    set_output_dir(parser.parse_args().out)
    required = ["e1_profiles_scenarios.csv", "e2_burden_sensitivity.csv",
                "e3_leakage_sensitivity.csv", "e4_shock_sensitivity.csv",
                "e5_income_volatility.csv"]
    missing = [f for f in required if not (OUT / f).exists()]
    if missing:
        sys.exit(f"Missing {missing}. Run: python experiments/run_experiments.py")
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    e1 = pd.read_csv(OUT / required[0])
    e2 = pd.read_csv(OUT / required[1])
    e3 = pd.read_csv(OUT / required[2])
    e4 = pd.read_csv(OUT / required[3])
    e5 = pd.read_csv(OUT / required[4])

    # ---- E1: profiles under the stable scenario + background uncertainty
    stable = e1[(e1.scenario == "Stable income and stable expenses") & e1.background_uncertainty]
    save(charts.grouped_success(stable, "profile",
                                "Deadline success by fictional profile (stable scenario + "
                                "background uncertainty)", "Fictional household profile"),
         "fig1_profiles.png")
    t = stable.pivot_table(index=["profile", "essential_burden", "baseline_affordability"],
                           columns="strategy", values="success_rate_pct").reset_index()
    t = tidy(t)
    write_table(t, "table1_profiles.md",
                "Table 1. % of simulations meeting the deadline, by profile")

    # ---- E1: scenarios for profile B, with background uncertainty
    b = e1[(e1.profile == "B: Tight but workable") & e1.background_uncertainty]
    save(charts.grouped_success(b, "scenario",
                                "Profile B: deadline success by scenario (with background "
                                "uncertainty)"), "fig2_scenarios_profile_B.png")
    t = b.pivot_table(index="scenario", columns="strategy", values="success_rate_pct",
                      sort=False).reset_index()
    t = tidy(t)
    write_table(t, "table2_scenarios_profile_B.md",
                "Table 2. Profile B: % meeting the deadline by scenario")

    # deterministic scenario outcomes for all profiles (shortfall in naira)
    det = e1[~e1.background_uncertainty].pivot_table(
        index=["profile", "scenario"], columns="strategy", values="mean_shortfall",
        sort=False).reset_index()
    det = tidy(det)
    write_table(det, "table3_deterministic_shortfalls.md",
                "Table 3. Shortfall in naira without background uncertainty "
                "(single deterministic run; 'Variable monthly income' is a mean over simulations)")

    # ---- E2: burden sensitivity
    save(charts.sensitivity(e2, facet="income_cv"), "fig3_burden_sensitivity.png")
    t = e2[e2.income_cv == 0.15].pivot_table(index="essential_burden", columns="strategy",
                                             values="success_rate_pct").reset_index()
    t = tidy(t)
    write_table(t, "table4_burden.md",
                "Table 4. % meeting the deadline vs essential burden (income CV 15%)")

    # ---- E3: leakage
    lb = e3[e3.profile == "B: Tight but workable"]
    fig = charts.shock_impact(lb, x="remainder_saving_rate",
                              xlabel="Remainder saving rate (share of unlabelled leftover saved)",
                              title="Profile B: success vs the leakage assumption")
    save(fig, "fig4_leakage_profile_B.png")
    t = e3.pivot_table(index=["profile", "remainder_saving_rate"], columns="strategy",
                       values="success_rate_pct").reset_index()
    t = tidy(t)
    write_table(t, "table5_leakage.md",
                "Table 5. % meeting the deadline vs remainder saving rate")

    # ---- E4: shocks
    mid = e4[e4.shock_severity_share_of_income == 0.25]
    save(charts.shock_impact(mid, title="Profile B: success vs unexpected expenses "
                                        "(average size 25% of monthly income)"),
         "fig5_shock_impact_profile_B.png")
    t = e4.pivot_table(index=["shock_severity_share_of_income", "shock_probability"],
                       columns="strategy", values="success_rate_pct").reset_index()
    t = tidy(t)
    write_table(t, "table6_shocks.md",
                "Table 6. Profile B: % meeting the deadline vs unexpected expenses")

    # ---- E5: income volatility
    vb = e5[e5.profile == "B: Tight but workable"]
    fig = charts.shock_impact(vb, x="income_cv",
                              xlabel="Income variability (coefficient of variation)",
                              title="Profile B: success vs income variability")
    save(fig, "fig6_income_volatility_profile_B.png")
    t = e5.pivot_table(index=["profile", "income_cv"], columns="strategy",
                       values="success_rate_pct").reset_index()
    t = tidy(t)
    write_table(t, "table7_income_volatility.md",
                "Table 7. % meeting the deadline vs income variability")
    print("done")


if __name__ == "__main__":
    main()
