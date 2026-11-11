"""Phase 5 tests: experiment scripts are reproducible end to end."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILES = ["e1_profiles_scenarios.csv", "e2_burden_sensitivity.csv",
         "e3_leakage_sensitivity.csv", "e4_shock_sensitivity.csv",
         "e5_income_volatility.csv"]


def run_script(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "experiments" / script), *args],
                   check=True, cwd=ROOT, capture_output=True)


def test_experiments_reproduce_exactly_and_figures_build(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run_script("run_experiments.py", "--n", "60", "--seed", "5", "--out", str(a))
    run_script("run_experiments.py", "--n", "60", "--seed", "5", "--out", str(b))
    for f in FILES:
        pd.testing.assert_frame_equal(pd.read_csv(a / f), pd.read_csv(b / f))
    run_script("make_figures.py", "--out", str(a))
    assert len(list((a / "figures").glob("*.png"))) == 6
    assert len(list((a / "tables").glob("*.md"))) == 7


def test_experiment_contains_planning_helps_and_nothing_helps_cases(tmp_path):
    """The brief requires at least one case where planning improves results
    and one where no strategy overcomes insufficient income. This checks the
    generated data actually contains both (it does not force them)."""
    run_script("run_experiments.py", "--n", "300", "--seed", "1", "--out", str(tmp_path))
    e1 = pd.read_csv(tmp_path / FILES[0])
    stable = e1[(e1.scenario == "Stable income and stable expenses") & ~e1.background_uncertainty]
    rate = stable.pivot_table(index="profile", columns="strategy", values="success_rate_pct")
    b = rate.loc["B: Tight but workable"]
    assert b["planned_savings"] > b["save_remainder"]
    assert (rate.loc["D: Structurally insufficient income"] == 0).all()
