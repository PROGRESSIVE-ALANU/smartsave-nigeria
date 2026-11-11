# Development phases, verification checklist, and troubleshooting

This file explains how the project was built in five phases, what each phase does, how
to check it yourself, and what remains unresolved. Run every command from the project
root (`smartsave-nigeria/`) with your virtual environment active.

---

## Phase 1: Budget calculations and validation

**Files:** `src/validation.py`, `src/budget.py`, `tests/test_budget.py`

**Logic.** A `Household` holds the monthly budget and the fee goal. From it:

- Essential expenses = rent + food + transport + utilities + healthcare + other essentials
- Surplus after essentials = income − essentials (can be negative)
- Monthly surplus/deficit = income − essentials − nonessential
- Required monthly saving = (fees − existing school savings) ÷ months remaining,
  which is 0 if the fees are already funded and the whole outstanding amount if fees
  are due now (0 months)
- Best-case funds at the deadline = initial savings + school savings + months × net
  monthly saving, floored at zero. If income doesn't cover essentials, savings drain,
  and anything beyond zero is reported as *unfunded essential need*, never as a
  negative balance.

Validation separates **errors** (negative money, NaN, fractional months, missing
fields: calculations refuse to run) from **warnings** (income below essentials, fees
already funded, deadline now: calculations run and the situation is explained).

**Verify:** `python -m pytest tests/test_budget.py -v`

**Limitation:** the best-case figure assumes perfect discipline and no surprises. It is
an upper bound, so the app labels it that way.

---

## Phase 2: Strategies and affordability

**Files:** `src/strategies.py`, `tests/test_strategies.py`

**Logic.** After essentials, money available $A$ is split into nonessential spending
$N$, an education contribution $C$, general saving $S$ and leakage $U$, always
summing exactly to $A$. The three rules differ only in how they choose $N$ and $C$
(see README). The affordability classifier compares funds at the deadline under usual
spending, under the maximum allowed cut, and with zero nonessential spending, and
reports the cut that would actually be needed.

**Verify:** `python -m pytest tests/test_strategies.py -v`. Key checks:

- the parts always sum to $A$ and are never negative
- no strategy saves anything when income doesn't cover essentials
- S3 never cuts beyond its limit
- with no leakage, S1 and S2 save the same total

**Limitation:** the S2 requirement targets the education fund only, so a household with
large general savings may earmark more than strictly necessary. This is deliberate and
documented.

---

## Phase 3: Scenarios, simulation engine, metrics

**Files:** `src/scenarios.py`, `src/simulator.py`, `src/metrics.py`, `tests/test_simulator.py`

**Logic.** `generate_paths` builds income and expense arrays of shape
(simulations × months) from three standard random arrays drawn in a fixed order with
`numpy.random.default_rng(seed)`. Every strategy runs on the *same* arrays (common
random numbers). `run_strategy` steps month by month (essentials → allocation → fee
payment at the deadline), vectorized across simulations. Metrics cover success rate,
mean and worst shortfall, percentiles, median final savings, cash-flow deficit months,
unfunded-essential months, leakage, and nonessential cuts, plus a paired comparison of
two strategies on identical futures.

**Verify:** `python -m pytest tests/test_simulator.py -v`. Highlights:

- same seed gives identical arrays; different seed gives different arrays
- hand-calculated deterministic case (S1 ₦230,000 vs S2 ₦318,000 for the example household)
- the **accounting identity** holds to the naira in every simulation: no money is created or lost
- balances never go negative even under extreme shocks
- strategy-independent metrics are identical across strategies (fairness check)

**Limitations:** shocks are independent month to month; income does not grow; savings
earn no interest; there are no transfers, loans or savings groups.

---

## Phase 4: Streamlit app and charts

**Files:** `app.py`, `src/charts.py`, `src/profiles.py`, `tests/test_app.py`

**Logic.** The app is a thin layer over `src/`. Household inputs live in
`st.session_state.household`, so they persist across pages. Simulations are cached by
their exact inputs (`st.cache_data`), so changing nothing reruns nothing. Charts use one
fixed colour **and** line style per strategy, so they stay distinguishable in
grayscale and for colour-blind readers, and every simulated chart carries a
"Simulated" note.

**Verify:**

```bash
python -m pytest tests/test_app.py -v     # every page renders, profile load, reset, horizon error
streamlit run app.py                      # then click through every page
```

**Limitation:** Matplotlib charts in Streamlit are static images (no hover tooltips).
That is the cost of following the Matplotlib requirement.

---

## Phase 5: Experiments, report template, docs

**Files:** `experiments/run_experiments.py`, `experiments/make_figures.py`,
`tests/test_experiments.py`, `reports/research_report_template.md`, `README.md`

**Logic.** Five experiments (E1–E5) write CSVs; a separate script turns CSVs into
figures and Markdown tables without simulating anything. `experiment_metadata.json`
records the seed, n, assumptions, and package versions.

**Verify:** `python -m pytest tests/test_experiments.py -v` (runs the pipeline twice and
checks the CSVs are identical).

---

## Verification checklist

| # | Command | Expected behaviour |
|---|---|---|
| 1 | `python --version` | 3.11 or newer |
| 2 | `pip install -r requirements.txt` | installs without errors |
| 3 | `python -m pytest` | `91 passed` (takes roughly 10–20 s) |
| 4 | `python experiments/run_experiments.py` | five "wrote e…csv" lines, then "done in … s" |
| 5 | `python experiments/make_figures.py` | six "figure" lines and seven "table" lines, then "done" |
| 6 | Run step 4 again and compare `outputs/e1_profiles_scenarios.csv` | identical file (only the timestamp in metadata changes) |
| 7 | `streamlit run app.py` | browser opens at http://localhost:8501 |
| 8 | Household Budget: set income to 200000 | warning that essentials exceed income; deficit shown; unfunded essentials reported |
| 9 | Household Budget: set months until due to 0 | "Needed immediately" metric; Planner says fees are due now |
| 10 | Sidebar: load profile D, open Education Planner | red "Not feasible…" box explaining it is an income problem |
| 11 | Sidebar: set remainder saving rate to 100%, open Strategy Comparison | S1 and S2 reach identical funds |
| 12 | Economic Simulation: run twice with the same seed | identical numbers |
| 13 | Economic Simulation: set horizon below the deadline | clear error asking for a longer horizon |
| 14 | Sidebar: "Reset to fictional example" | inputs return to the example household |
| 15 | Download buttons | CSV files open in a spreadsheet |
| 16 | `pandoc reports/research_report_template.md -s --mathjax -o reports/research_report.html` | HTML report you can print to PDF |

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`**: run commands from the project root.
  For pytest, `pytest.ini` sets the path automatically; run `python -m pytest`, not
  `pytest` from another folder.
- **`streamlit: command not found`**: activate the virtual environment, or use
  `python -m streamlit run app.py`.
- **Port 8501 already in use**: `streamlit run app.py --server.port 8502`.
- **Windows PowerShell won't activate the venv**: run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.
- **₦ shows as a box in charts**: your Matplotlib font lacks the glyph. The default
  DejaVu Sans includes it. If you changed fonts, switch back, or install DejaVu.
- **App feels slow at 10,000 simulations**: the main run is fast. The sensitivity charts
  run about 27 extra simulations (capped at 1,000 runs each) and are cached after the first view.
- **Results differ from a classmate's**: check seed, n, every sidebar assumption, and
  package versions (recorded in `outputs/experiment_metadata.json`). Different NumPy
  major versions can change random streams.
- **Pandoc PDF fails with a font error**: use the HTML route, or install a LaTeX
  distribution and the DejaVu fonts.
