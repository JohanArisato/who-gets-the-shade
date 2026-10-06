"""Figures for 'Who Gets the Shade?' -> figures/*.png"""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
INK, INK2, GRID, BLUE, ORANGE, GREEN = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)


def main():
    g = gpd.read_file(ROOT / "data" / "uhf42_shade_results.geojson")
    r = json.loads((ROOT / "data" / "results.json").read_text())

    # 1. canopy vs surface temperature
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(g.tree_canopy_pct, g.surface_temp_f, s=36, color=BLUE, edgecolor="white", linewidth=.8)
    b = np.polyfit(g.tree_canopy_pct, g.surface_temp_f, 1)
    xs = np.linspace(g.tree_canopy_pct.min(), g.tree_canopy_pct.max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=INK2, lw=1.5)
    h = r["heat_on_canopy"]
    ax.set(xlabel="Tree canopy (% of land, 2014)", ylabel="Summer surface temperature (°F, 2009)",
           title=f"More canopy, cooler surfaces ({h['ols_slope']} °F per point, R² {h['ols_r2']})")
    save(fig, "01_canopy_vs_temperature.png")

    # 2. heat illness vs poverty and temperature
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, col, lab, rr, c in [(axes[0], "poverty_pct", "Poverty (%, 2013–17)", r["heat_illness"]["r_poverty"], ORANGE),
                                (axes[1], "surface_temp_f", "Surface temperature (°F)", r["heat_illness"]["r_temperature"], BLUE)]:
        ax.scatter(g[col], g.heat_ed_rate, s=34, color=c, edgecolor="white", linewidth=.8)
        ax.set(xlabel=lab, title=f"r = {rr:.2f}")
    pr = g[g.uhf == 501].iloc[0]
    axes[1].annotate("Port Richmond", (pr.surface_temp_f, pr.heat_ed_rate), xytext=(6, -2),
                     textcoords="offset points", fontsize=8, color=INK2)
    axes[0].set_ylabel("Heat-stress ED visits per 100k (age-adj.)")
    fig.suptitle("Heat illness follows poverty, not surface heat", fontweight="bold", x=0.08, ha="left")
    save(fig, "02_heat_illness_drivers.png")

    # 3. ranking rules
    rules = r["ranking_rules"]
    labels = {"canopy_gap": "Canopy gap", "hottest_surface": "Hottest surfaces",
              "canopy_gap_and_heat": "Canopy gap + heat", "heat_illness": "Heat illness",
              "equity_weighted": "Equity-weighted"}
    names = list(labels)
    vals = [rules[n]["poorest_third_in_top10"] for n in names]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.barh(range(len(names)), vals, color=[BLUE if n != "equity_weighted" else GREEN for n in names], height=.6)
    for i, v in enumerate(vals):
        ax.text(v + .15, i, f"{v} of 10", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(names)), [labels[n] for n in names])
    ax.invert_yaxis()
    ax.set(xlim=(0, 10.5), xlabel="Poorest-third neighborhoods among the first 10 planted",
           title="The ranking rule is the policy")
    ax.grid(axis="y", visible=False)
    save(fig, "03_ranking_rules.png")

    # 4. maps: heat illness and robust priorities
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    g.plot(column="heat_ed_rate", cmap="Blues", ax=axes[0], edgecolor="white", linewidth=.5, legend=True,
           legend_kwds={"shrink": .6, "label": "Heat ED visits per 100k"})
    axes[0].set_title("Where people get sick from heat")
    g.plot(column="top10_share", cmap="Greens", ax=axes[1], edgecolor="white", linewidth=.5, legend=True,
           vmin=0, vmax=1, legend_kwds={"shrink": .6, "label": "Share of 5,000 weightings in top 10"})
    g[g.top10_share >= .9].boundary.plot(ax=axes[1], color=INK, linewidth=1.4)
    axes[1].set_title("Robust planting priorities (outlined: ≥90%)")
    for ax in axes:
        ax.set_axis_off()
    save(fig, "04_maps.png")

    # 5. prediction
    p = r["prediction"]
    fig, ax = plt.subplots(figsize=(5.4, 3))
    bars = [p["r2_random_5fold"], p["r2_leave_one_borough_out"]]
    ax.bar([0, 1], bars, color=[BLUE, ORANGE], width=.55)
    ax.axhline(0, color=INK2, lw=1)
    for i, v in enumerate(bars):
        ax.text(i, v + (.03 if v >= 0 else -.07), f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks([0, 1], ["Random 5-fold", "Whole borough held out"])
    ax.set(ylabel="Cross-validated R²", title="A model that looks useful does not travel")
    ax.grid(axis="x", visible=False)
    save(fig, "05_prediction_transfer.png")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
