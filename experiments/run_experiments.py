"""Reproducible strategy-comparison experiments.

Usage (from the project root):
    python experiments/run_experiments.py              # default: 5,000 sims, seed 2026
    python experiments/run_experiments.py --n 1000 --seed 7

Writes CSV files to outputs/ plus outputs/experiment_metadata.json. Every
number in those files is produced by executing the simulation engine; none is
typed in by hand. Re-running with the same arguments reproduces them exactly.

Experimental design
-------------------
Independent variables : strategy (S1/S2/S3), household profile (different
                        income-to-essential ratios), scenario, essential
                        burden, income variability, shock probability and
                        severity, remainder saving rate.
Dependent variables   : % of simulations meeting the deadline, mean shortfall,
                        median funds at the deadline, months with unfunded
                        essentials.
Controls              : the same seed (so identical simulated months) for
                        every strategy and every level of a varied factor;
                        fixed n, fees, deadline and all non-varied inputs.

Experiments
-----------
E1 profiles_scenarios   5 fictional profiles x 8 preset scenarios, each with
                        and without background uncertainty.
E2 burden_sensitivity   essential burden 40%..110% of income x income CV.
E3 leakage_sensitivity  remainder saving rate 0..100% x profiles.
E4 shock_sensitivity    shock probability x shock severity (profile B).
E5 income_volatility    income CV 0..60% x profiles.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.budget import Household  # noqa: E402
from src.metrics import summarize  # noqa: E402
from src.profiles import load_profiles  # noqa: E402
from src.scenarios import Scenario, preset_scenarios  # noqa: E402
from src.simulator import simulate  # noqa: E402
from src.strategies import StrategyParams, classify_affordability  # noqa: E402

METRIC_COLUMNS = [
    "strategy", "success_rate_pct", "mean_shortfall", "mean_shortfall_if_failed",
    "worst_case_shortfall", "median_funds_at_deadline", "funds_at_deadline_p5",
    "funds_at_deadline_p95", "median_final_savings", "pct_sims_with_cashflow_deficit",
    "pct_sims_with_unfunded_essentials", "mean_unfunded_essential_months",
    "mean_total_leakage", "mean_total_nonessential_cut",
]

BASE_PARAMS = StrategyParams(remainder_saving_rate=0.6, max_nonessential_cut=0.3)


def background_uncertainty(h: Household) -> dict:
    """Moderate background randomness added on top of a scenario (assumption)."""
    return {
        "income_cv": 0.15,
        "shock_probability": 0.10,
        "shock_severity": round(0.15 * h.monthly_income, -3),
        "monthly_inflation": 0.01,
    }


def run(h: Household, scenario: Scenario, params: StrategyParams, n: int, seed: int,
        /, **labels) -> pd.DataFrame:
    """Simulate one condition; ``labels`` become leading identifier columns."""
    result = simulate(h, scenario, params, n_simulations=n, seed=seed)
    df = summarize(result)[METRIC_COLUMNS]
    for key, value in reversed(list(labels.items())):
        df.insert(0, key, value)
    return df


def e1_profiles_scenarios(profiles, n, seed) -> pd.DataFrame:
    frames = []
    for pname, h in profiles.items():
        afford = classify_affordability(h, BASE_PARAMS.max_nonessential_cut).category.value
        for sname, sc in preset_scenarios(h).items():
            for uncertain in (False, True):
                scenario = replace(sc, **background_uncertainty(h)) if uncertain else sc
                if uncertain and sc.income_cv > 0:  # keep the preset's own volatility
                    scenario = replace(scenario, income_cv=sc.income_cv)
                random = uncertain or scenario.income_cv > 0 or scenario.shock_probability > 0
                frames.append(run(
                    h, scenario, BASE_PARAMS, n if random else 1, seed,
                    profile=pname, essential_burden=round(h.essential_burden, 3),
                    baseline_affordability=afford, scenario=sname,
                    background_uncertainty=uncertain,
                ))
    return pd.concat(frames, ignore_index=True)


def e2_burden_sensitivity(base: Household, n, seed) -> pd.DataFrame:
    frames = []
    for burden in np.round(np.arange(0.40, 1.101, 0.05), 2):
        for cv in (0.0, 0.15, 0.30):
            scale = burden * base.monthly_income / base.essential_expenses
            sc = Scenario(name="burden sweep", essential_multiplier=float(scale),
                          income_cv=cv, shock_probability=0.10,
                          shock_severity=round(0.15 * base.monthly_income, -3))
            frames.append(run(base, sc, BASE_PARAMS, n, seed,
                              essential_burden=float(burden), income_cv=cv))
    return pd.concat(frames, ignore_index=True)


def e3_leakage_sensitivity(profiles, n, seed) -> pd.DataFrame:
    frames = []
    for pname, h in profiles.items():
        sc = Scenario(name="background", **background_uncertainty(h))
        for rate in (0.0, 0.25, 0.5, 0.75, 1.0):
            params = replace(BASE_PARAMS, remainder_saving_rate=rate)
            frames.append(run(h, sc, params, n, seed, profile=pname,
                              remainder_saving_rate=rate))
    return pd.concat(frames, ignore_index=True)


def e4_shock_sensitivity(h: Household, n, seed) -> pd.DataFrame:
    frames = []
    for severity_share in (0.10, 0.25, 0.50):
        for p in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
            sc = Scenario(name="shock sweep", income_cv=0.10, shock_probability=p,
                          shock_severity=round(severity_share * h.monthly_income, -3))
            frames.append(run(h, sc, BASE_PARAMS, n, seed,
                              shock_severity_share_of_income=severity_share,
                              shock_probability=p))
    return pd.concat(frames, ignore_index=True)


def e5_income_volatility(profiles, n, seed) -> pd.DataFrame:
    frames = []
    for pname, h in profiles.items():
        for cv in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6):
            sc = Scenario(name="volatility sweep", income_cv=cv)
            frames.append(run(h, sc, BASE_PARAMS, n, seed, profile=pname, income_cv=cv))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=5_000, help="simulations per condition")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    profiles = load_profiles()
    base = profiles["B: Tight but workable"]
    start = time.time()
    experiments = {
        "e1_profiles_scenarios.csv": lambda: e1_profiles_scenarios(profiles, args.n, args.seed),
        "e2_burden_sensitivity.csv": lambda: e2_burden_sensitivity(base, args.n, args.seed),
        "e3_leakage_sensitivity.csv": lambda: e3_leakage_sensitivity(profiles, args.n, args.seed),
        "e4_shock_sensitivity.csv": lambda: e4_shock_sensitivity(base, args.n, args.seed),
        "e5_income_volatility.csv": lambda: e5_income_volatility(profiles, args.n, args.seed),
    }
    for filename, fn in experiments.items():
        df = fn()
        df.to_csv(args.out / filename, index=False, float_format="%.4f")
        print(f"wrote {filename:32s} {len(df):5d} rows")

    metadata = {
        "description": "SmartSave Nigeria strategy experiments. SIMULATED data from "
                       "FICTIONAL households; not empirical observations.",
        "command": " ".join(["python"] + sys.argv),
        "n_simulations_per_condition": args.n,
        "seed": args.seed,
        "base_strategy_params": asdict(BASE_PARAMS),
        "background_uncertainty_rule": "income CV 15%, shock probability 10%/month, "
                                       "shock severity 15% of monthly income, inflation 1%/month",
        "note_e1": "Rows with background_uncertainty=False are single deterministic runs, "
                   "except the 'Variable monthly income' preset, which is random by design.",
        "profiles_file": "data/example_households.csv",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "runtime_seconds": round(time.time() - start, 2),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"done in {metadata['runtime_seconds']} s -> {args.out}")


if __name__ == "__main__":
    main()
