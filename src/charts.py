"""Matplotlib charts for the app and the experiment scripts.

Every function returns a ``matplotlib.figure.Figure`` built from real
calculation or simulation outputs passed in as arguments -- nothing is
hardcoded. Charts of simulated quantities say "Simulated" in the title.

Style choices: one fixed colour per strategy (never re-ordered), thin marks,
light grid, value text in neutral ink, one y-axis per chart.
"""

from __future__ import annotations

from typing import Mapping

import matplotlib

matplotlib.use("Agg")  # headless backend: works in Streamlit, scripts and tests

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from src.budget import Household, spending_breakdown  # noqa: E402
from src.strategies import STRATEGIES, STRATEGY_LABELS  # noqa: E402

# Fixed colour per strategy (validated categorical palette, slots 1-3).
STRATEGY_COLORS = {
    STRATEGIES[0]: "#2a78d6",  # S1 blue
    STRATEGIES[1]: "#eb6834",  # S2 orange
    STRATEGIES[2]: "#1baf7a",  # S3 aqua
}
# Secondary encoding so identical lines stay distinguishable (and for CVD).
STRATEGY_LINESTYLES = {STRATEGIES[0]: "-", STRATEGIES[1]: "--", STRATEGIES[2]: ":"}
STRATEGY_MARKERS = {STRATEGIES[0]: "o", STRATEGIES[1]: "s", STRATEGIES[2]: "^"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e4e3df"
ESSENTIAL_COLOR = "#2a78d6"
NONESSENTIAL_COLOR = "#eb6834"
SIM_NOTE = "Simulated under stated assumptions; not observed data."


def _naira_axis(value: float, _pos=None) -> str:
    """Compact naira tick labels: ₦250k, ₦1.2M."""
    a = abs(value)
    sign = "−" if value < 0 else ""
    if a >= 1_000_000:
        return f"{sign}₦{a / 1_000_000:.1f}M"
    if a >= 1_000:
        return f"{sign}₦{a / 1_000:.0f}k"
    return f"{sign}₦{a:.0f}"


NAIRA_FMT = FuncFormatter(_naira_axis)


def _style(ax, title: str, xlabel: str, ylabel: str, note: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=14 if note else 8)
    if note:
        ax.text(0, 1.01, note, transform=ax.transAxes, fontsize=8,
                color=INK_SECONDARY, va="bottom")
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def _label(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy)


# 1 ---------------------------------------------------------------------
def income_vs_spending(h: Household) -> Figure:
    """Monthly income versus spending by category (inputs, not simulated)."""
    items = spending_breakdown(h)
    labels = list(items.keys())
    values = np.array(list(items.values()))
    colors = [NONESSENTIAL_COLOR if k == "Nonessential" else ESSENTIAL_COLOR for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.6)
    ax.set_yticks(y, labels)
    for yi, v in zip(y, values):
        ax.text(v, yi, f"  ₦{v:,.0f}", va="center", fontsize=8, color=INK_SECONDARY)
    ax.axvline(h.monthly_income, color=INK, linewidth=1.5, linestyle="--")
    ax.text(h.monthly_income, y.max() + 0.6, f" Income ₦{h.monthly_income:,.0f}",
            fontsize=8, color=INK, va="bottom")
    total = values.sum()
    ax.axvline(total, color=INK_MUTED, linewidth=1, linestyle=":")
    ax.text(total, -0.9, f"Total spending ₦{total:,.0f} ", fontsize=8,
            color=INK_SECONDARY, ha="right")
    ax.xaxis.set_major_formatter(NAIRA_FMT)
    ax.set_xlim(0, max(h.monthly_income, total, values.max()) * 1.25 + 1)
    ax.set_ylim(-1.3, len(labels) + 0.2)
    _style(ax, "Monthly income versus planned spending", "Naira per month (₦)", "",
           note="Blue = essential, orange = nonessential. Based on entered values.")
    fig.tight_layout()
    return fig


# 2 ---------------------------------------------------------------------
def projected_savings(projection: pd.DataFrame, fees: float, deadline: int,
                      simulated: bool = False) -> Figure:
    """Total savings by month for each strategy (one projected path).

    The deadline-month point is the balance just before fees are paid."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    col = "total_before_fees" if "total_before_fees" in projection else "total_savings"
    for strategy, group in projection.groupby("strategy", sort=False):
        group = group[group["month"] <= deadline]
        ax.plot(group["month"], group[col], linewidth=2,
                linestyle=STRATEGY_LINESTYLES.get(strategy, "-"),
                marker=STRATEGY_MARKERS.get(strategy, "o"), markersize=5,
                color=STRATEGY_COLORS.get(strategy), label=_label(strategy))
    ax.axhline(fees, color=INK, linestyle="--", linewidth=1)
    ax.text(0, fees, " School-fee target", va="bottom", fontsize=8, color=INK)
    ax.axvline(deadline, color=INK_MUTED, linestyle=":", linewidth=1)
    ax.text(deadline, ax.get_ylim()[1], " Deadline", fontsize=8,
            color=INK_SECONDARY, va="top", ha="right")
    ax.yaxis.set_major_formatter(NAIRA_FMT)
    ax.set_xticks([m for m in sorted(projection["month"].unique()) if m <= deadline])
    title = "Projected total savings by month"
    note = (SIM_NOTE if simulated else
            "Deterministic projection: constant income and prices, no random shocks. "
            "Last point = funds just before paying fees.")
    _style(ax, title, "Month (0 = today)", "Education fund + general savings (₦)", note)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


# 3 ---------------------------------------------------------------------
def target_vs_available(funds: Mapping[str, float], fees: float,
                        simulated: bool = False) -> Figure:
    """Money usable for fees at the deadline versus the fee target."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    names = list(funds.keys())
    values = [funds[n] for n in names]
    x = np.arange(len(names))
    ax.bar(x, values, color=[STRATEGY_COLORS.get(n) for n in names], width=0.55)
    ax.axhline(fees, color=INK, linestyle="--", linewidth=1.2)
    ax.text(-0.45, fees, f" Fees ₦{fees:,.0f}", ha="left", va="bottom",
            fontsize=8, color=INK)
    for xi, v in zip(x, values):
        gap = v - fees
        tag = "covered" if gap >= -0.5 else f"short ₦{-gap:,.0f}"
        ax.text(xi, v * 0.5, f"₦{v:,.0f}\n({tag})", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(x, [_label(n) for n in names], fontsize=8)
    ax.yaxis.set_major_formatter(NAIRA_FMT)
    ax.set_ylim(0, max(max(values), fees) * 1.3 + 1)
    _style(ax, "School-fee target versus funds available at the deadline", "Strategy",
           "Funds usable for fees (₦)",
           SIM_NOTE if simulated else "Deterministic projection (no random shocks).")
    fig.tight_layout()
    return fig


# 4 ---------------------------------------------------------------------
def strategy_comparison(summary: pd.DataFrame, title_suffix: str = "") -> Figure:
    """Success rate and mean shortfall per strategy (two panels, one axis each)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    x = np.arange(len(summary))
    colors = [STRATEGY_COLORS.get(s) for s in summary["strategy"]]
    short_labels = [lbl.split(":")[0] for lbl in summary["label"]]

    ax1.bar(x, summary["success_rate_pct"], color=colors, width=0.55)
    for xi, v in zip(x, summary["success_rate_pct"]):
        ax1.text(xi, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
    ax1.set_ylim(0, 110)
    ax1.yaxis.set_major_formatter(PercentFormatter())
    ax1.set_xticks(x, short_labels)
    _style(ax1, "Deadline met" + title_suffix, "Strategy", "% of simulations", SIM_NOTE)

    ax2.bar(x, summary["mean_shortfall"], color=colors, width=0.55)
    for xi, v in zip(x, summary["mean_shortfall"]):
        ax2.text(xi, v, f"₦{v:,.0f}", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
    ax2.yaxis.set_major_formatter(NAIRA_FMT)
    ax2.set_ylim(0, max(summary["mean_shortfall"].max() * 1.25, 1))
    ax2.set_xticks(x, short_labels)
    _style(ax2, "Average school-fee shortfall", "Strategy", "₦ (all simulations)", SIM_NOTE)
    fig.tight_layout()
    return fig


# 5 ---------------------------------------------------------------------
def outcome_distribution(result, bins: int = 40) -> Figure:
    """Distribution of funds at the deadline across simulations."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    all_values = np.concatenate([r.funds_at_deadline for r in result.strategies.values()])
    lo, hi = float(all_values.min()), float(all_values.max())
    if hi - lo < 1:
        hi = lo + 1
    edges = np.linspace(lo, hi, bins + 1)
    n = result.n_simulations
    for name, r in result.strategies.items():
        counts, _ = np.histogram(r.funds_at_deadline, bins=edges)
        ax.stairs(100 * counts / n, edges, color=STRATEGY_COLORS.get(name),
                  linestyle=STRATEGY_LINESTYLES.get(name, "-"),
                  linewidth=2, label=_label(name))
    fees = result.household.school_fees
    ax.axvline(fees, color=INK, linestyle="--", linewidth=1.2)
    ax.text(fees, ax.get_ylim()[1], " Fees", fontsize=8, color=INK, va="top")
    ax.xaxis.set_major_formatter(NAIRA_FMT)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    _style(ax, f"Distribution of funds at the deadline ({n:,} simulations)",
           "Funds usable for fees at the deadline (₦)", "% of simulations", SIM_NOTE)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.tight_layout()
    return fig


def savings_fan(mean_paths_df: pd.DataFrame, fees: float, deadline: int) -> Figure:
    """Mean total savings by month with a 10th-90th percentile band."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for strategy, g in mean_paths_df.groupby("strategy", sort=False):
        c = STRATEGY_COLORS.get(strategy)
        ax.fill_between(g["month"], g["p10_total_savings"], g["p90_total_savings"],
                        color=c, alpha=0.12, linewidth=0)
        ax.plot(g["month"], g["mean_total_savings"], color=c, linewidth=2,
                linestyle=STRATEGY_LINESTYLES.get(strategy, "-"),
                label=_label(strategy) + " (mean)")
    ax.axhline(fees, color=INK, linestyle="--", linewidth=1)
    ax.axvline(deadline, color=INK_MUTED, linestyle=":", linewidth=1)
    ax.text(deadline, ax.get_ylim()[1], " Deadline", fontsize=8, color=INK_SECONDARY, va="top")
    ax.yaxis.set_major_formatter(NAIRA_FMT)
    _style(ax, "Simulated savings by month (mean, shaded 10th–90th percentile)",
           "Month (0 = today)", "Total savings (₦)", SIM_NOTE)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.tight_layout()
    return fig


# 6 ---------------------------------------------------------------------
def shock_impact(df: pd.DataFrame, x: str = "shock_probability",
                 xlabel: str = "Monthly probability of an unexpected expense",
                 title: str = "Impact of unexpected expenses on meeting the school-fee deadline",
                 ) -> Figure:
    """Success rate as unexpected expenses become more likely (long df with
    columns: x, strategy, success_rate_pct)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for strategy, g in df.groupby("strategy", sort=False):
        g = g.sort_values(x)
        ax.plot(g[x], g["success_rate_pct"], linewidth=2,
                linestyle=STRATEGY_LINESTYLES.get(strategy, "-"),
                marker=STRATEGY_MARKERS.get(strategy, "o"), markersize=5,
                color=STRATEGY_COLORS.get(strategy), label=_label(strategy))
    ax.set_ylim(-3, 103)
    ax.yaxis.set_major_formatter(PercentFormatter())
    if df[x].max() <= 1:
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    _style(ax, title, xlabel, "% of simulations meeting deadline", SIM_NOTE)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# 7 ---------------------------------------------------------------------
def sensitivity(df: pd.DataFrame, facet: str = "income_cv") -> Figure:
    """Success rate versus essential-expense burden, one panel per value of
    ``facet`` (long df: essential_burden, strategy, success_rate_pct, facet)."""
    facets = sorted(df[facet].unique())
    fig, axes = plt.subplots(1, len(facets), figsize=(3.6 * len(facets), 3.9),
                             sharey=True, squeeze=False)
    for ax, value in zip(axes[0], facets):
        sub = df[df[facet] == value]
        for strategy, g in sub.groupby("strategy", sort=False):
            g = g.sort_values("essential_burden")
            ax.plot(g["essential_burden"], g["success_rate_pct"], linewidth=2,
                    linestyle=STRATEGY_LINESTYLES.get(strategy, "-"),
                    color=STRATEGY_COLORS.get(strategy), label=_label(strategy))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_ylim(-3, 103)
        _style(ax, f"Income CV = {value:.0%}", "Essentials as % of income",
               "% meeting deadline" if ax is axes[0][0] else "")
    axes[0][0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Sensitivity of simulated success rates to essential-expense burden",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.905, SIM_NOTE, fontsize=8, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def close(fig: Figure) -> None:
    plt.close(fig)


def grouped_success(df: pd.DataFrame, category: str, title: str,
                    xlabel: str = "") -> Figure:
    """Grouped bars: success rate per strategy within each category
    (long df with columns: category, strategy, success_rate_pct)."""
    cats = list(dict.fromkeys(df[category]))
    strategies = [s for s in STRATEGIES if s in set(df["strategy"])]
    width = 0.8 / max(len(strategies), 1)
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(cats)), 4.4))
    x = np.arange(len(cats))
    for i, s in enumerate(strategies):
        vals = [float(df[(df[category] == c) & (df["strategy"] == s)]["success_rate_pct"].iloc[0])
                for c in cats]
        ax.bar(x + (i - (len(strategies) - 1) / 2) * width, vals, width * 0.92,
               color=STRATEGY_COLORS[s], label=_label(s),
               hatch={0: "", 1: "//", 2: ".."}.get(i, ""), edgecolor="white", linewidth=0)
    import textwrap
    ax.set_xticks(x, [textwrap.fill(str(c), 16) for c in cats], fontsize=8)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(PercentFormatter())
    _style(ax, title, xlabel, "% of simulations meeting deadline", SIM_NOTE)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    return fig
