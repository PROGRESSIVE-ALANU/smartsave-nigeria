# SmartSave Nigeria

**A Household Financial Planning and Education Affordability Simulator**

**Live app:** https://smartsave-nigeria.streamlit.app

SmartSave Nigeria is an educational, fully local Python application that asks whether
structured budgeting and advance savings planning help a household meet a school-fee
deadline, and when they cannot help because income is simply too low. It combines
household budget arithmetic, three transparent budgeting strategies, and seeded Monte
Carlo simulation, with a Streamlit interface and reproducible experiment scripts.

There is **no artificial intelligence, machine learning, chatbot, or automated
recommendation** in the application. Every number comes from a formula documented in
the code and on the app's *Research & Methodology* page.

> All households in this project are **fictional**. Results are **simulated** consequences
> of stated assumptions, not observations of real Nigerian households.

---

## Why the problem matters

School fees arrive as large, lumpy payments with fixed deadlines, while household
income and expenses arrive month by month and can be disrupted by medical bills, price
increases, or lost income. A household can have income and still be unable to pay fees
on time if the money is not set aside early enough. Other households cannot pay
regardless of how carefully they budget. A useful tool has to tell these two situations
apart, and this one is designed to.

## Research question

> To what extent can structured household budgeting and advance savings planning
> improve educational expense preparedness under different income and economic
> conditions?

Sub-questions: saving only the remainder vs reserving early; the effect of unexpected
expenses; inconsistent income; whether the advantage of budgeting shrinks as essentials
consume most income; and when fees are mathematically unaffordable under every strategy.

---

## Installation

Requires **Python 3.11 or newer**.

```bash
cd smartsave-nigeria
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Then open the URL it prints (normally http://localhost:8501). Everything runs locally.
No account is needed and nothing is stored outside your browser session.

## Run the tests

```bash
python -m pytest
```

There are 91 tests covering validation, calculations, strategies, simulation
reproducibility, fairness of comparisons, the accounting identity, all app pages, and
end-to-end reproducibility of the experiments.

## Reproduce the experiments

```bash
python experiments/run_experiments.py      # writes outputs/*.csv + experiment_metadata.json
python experiments/make_figures.py         # writes outputs/figures/*.png + outputs/tables/*.md
```

Options: `--n 5000` (simulations per condition), `--seed 2026`, `--out outputs`.
The same arguments always produce identical CSV files (a test checks this).
The default run takes a few seconds.

---

## How to use the app

| Page | What you do there |
|---|---|
| **Home** | Introduction, research question, limitations, glossary of terms. |
| **Household Budget** | Enter income, savings, each expense category, fees and deadline. See totals, surplus/deficit, the required monthly saving, best-case funds at the deadline, the projected shortfall, and an income-vs-spending chart. Invalid or inconsistent inputs are explained. |
| **Education Planner** | The affordability classification (*feasible* / *feasible only with reduced discretionary spending* / *not feasible without additional resources*), the funds available under each spending assumption, deterministic projections per strategy, and a CSV download. |
| **Strategy Comparison** | Plain-language rules for the three strategies and a side-by-side comparison under any preset scenario, without randomness. |
| **Economic Simulation** | Pick a scenario, edit every value, set simulations (100–10,000), horizon (1–24 months), income variability, shock probability/severity, inflation and seed. See success rates, shortfalls, percentiles, worst cases, distributions, and sensitivity charts. Download the summary and every simulated outcome as CSV. |
| **Research & Methodology** | Hypotheses, formulas, assumptions, experimental design, limitations, and the experiment outputs if you have generated them. |

The **sidebar** holds the strategy assumptions (remainder saving rate, maximum
nonessential cut, whether the education fund may pay for emergencies, whether general
savings may pay fees), a **Reset to fictional example** button, and a loader for the five
fictional profiles in `data/example_households.csv`.

---

## The model in one page

Each simulated month:

1. **Income** $Y_t$ arrives: average income × a lognormal multiplier with mean 1 and the
   chosen coefficient of variation, × any scheduled income loss.
2. **Essentials** $X_t$ are paid first (inflated, with any food-price jump, random shocks,
   and scheduled medical bills). A gap is covered from general savings, then (optionally)
   the education fund. Anything left is recorded as an **unfunded essential need**.
   There is no borrowing and no balance ever goes negative.
3. **Money after essentials** $A_t = \max(Y_t - X_t, 0)$ is split by the strategy into
   nonessential spending $N$, an education contribution $C$, general saving $S$, and
   **leakage** $U$ (unplanned spending of unlabelled leftovers). $N+C+S+U = A$ always.
4. At the end of the deadline month the fees are paid from the education fund, then
   general savings. Anything unpaid is the **shortfall**.

Required monthly saving, recalculated every month:
$R_t = \max(F - E_t, 0) / (D - t)$.

| Strategy | Rule |
|---|---|
| **S1 Save the Remainder** | Usual nonessentials; save a share $r$ of what is left. |
| **S2 Planned Education Savings** | Usual nonessentials; earmark $\min(R_t, \text{what is left})$ first; save share $r$ of the rest. |
| **S3 Expense Prioritization** | Like S2, but cut nonessentials just enough to fund $R_t$, never by more than the stated limit. |

**Important property:** if $r = 100\%$ (no leakage), S1 and S2 accumulate exactly the
same total. In this model planning helps *only* by protecting money from leakage and
(for S3) by cutting discretionary spending. A test enforces this.

All strategies are run on **identical simulated months** (common random numbers), so
differences come only from the decision rules.

## How to interpret results

- A **success rate** is the share of *simulated* futures in which fees were fully paid on
  time, given the assumptions you set. It is not a real-world probability.
- Look at **unfunded essential needs** and **nonessential cuts** alongside success.
  A higher success rate can coincide with more hardship (for example, when variable income
  forces nonessential spending down).
- The **affordability class** assumes constant income and prices and perfect saving
  discipline. It describes arithmetic, not a guarantee.
- Simulation results are **not causal evidence**. They show what follows from the
  assumptions. Changing the assumptions (especially the remainder saving rate) can change
  the conclusions, which is why the experiments vary them.

## Limitations

- Inputs are synthetic and not calibrated to Nigerian data.
- Leakage is modelled by one fixed parameter; real behaviour is far richer.
- No borrowing, interest, family transfers, savings groups (e.g. *ajo/esusu*), or extra
  work. These real coping mechanisms are deliberately omitted for transparency.
- Fees are fixed in naira; income does not grow with inflation.
- Months are independent: no persistence in income shocks.

## Data provenance

- `data/example_households.csv`: **fictional** profiles invented for illustration.
  They are not representative of Nigerian households.
- No external datasets are used. If you add public data (for example, official inflation
  statistics), record the source, URL, access date and table in the report template, and
  keep it clearly separate from synthetic inputs.
- Do not put real, identifiable household records in this repository. `.gitignore`
  excludes `data/private/` if you need a local place for anonymized data.

## Project structure

```
smartsave-nigeria/
├── app.py                         Streamlit interface (6 pages)
├── requirements.txt
├── pytest.ini
├── src/
│   ├── validation.py              input checks + plain-language explanations
│   ├── budget.py                  Household model, budget summary, ₦ formatting
│   ├── strategies.py              S1/S2/S3 decision rules, affordability classifier
│   ├── scenarios.py               scenarios, presets, random path generation
│   ├── simulator.py               month-by-month engine (deterministic + Monte Carlo)
│   ├── metrics.py                 success rates, shortfalls, percentiles, paired comparison
│   ├── charts.py                  Matplotlib figures
│   └── profiles.py                loads the fictional profiles CSV
├── data/example_households.csv    fictional profiles
├── experiments/
│   ├── run_experiments.py         E1–E5 → outputs/*.csv
│   └── make_figures.py            outputs/*.csv → figures + Markdown tables
├── tests/                         91 pytest tests
├── reports/research_report_template.md
├── docs/PHASES.md                 phase-by-phase explanation + verification checklist
└── outputs/                       generated results
```

`src/validation.py` and `src/profiles.py` are additions to the proposed structure: the
brief asks for a separate validation module, and profile loading is shared by the app
and the experiments.

## Converting the research report to PDF (optional, free)

Using [Pandoc](https://pandoc.org/) (free):

```bash
# Option A: HTML, then your browser's Print → Save as PDF (no LaTeX needed)
pandoc reports/research_report_template.md -s --mathjax -o reports/research_report.html

# Option B: direct PDF (needs a LaTeX distribution such as TeX Live or MiKTeX;
# xelatex handles the ₦ symbol with a Unicode font)
pandoc reports/research_report_template.md -o reports/research_report.pdf \
  --pdf-engine=xelatex -V mainfont="DejaVu Serif" -V geometry:margin=2.5cm
```

## Authorship and AI-assistance disclosure

This codebase was produced with substantial assistance from an AI coding assistant
(Anthropic's Claude), working from a detailed specification written by the project owner.
If you present this project (portfolio, application, competition), describe accurately
which parts you designed, wrote, modified, verified, or extended yourself, and follow
the rules of the venue you submit to. The application itself contains no AI features.

## License

MIT. See `LICENSE`.
