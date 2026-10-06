"""Guard the published numbers and the data-quality fix."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_42_neighborhoods_and_bensonhurst_rekeyed():
    d = pd.read_csv(ROOT / "data" / "uhf42_shade.csv")
    assert len(d) == 42 and d.uhf.is_unique
    assert {208, 209} <= set(d.uhf)
    assert d.loc[d.uhf == 209, "name"].str.contains("Bensonhurst").all()


def test_published_results_reproduce():
    r = json.loads((ROOT / "data" / "results.json").read_text())
    assert r["heat_on_canopy"]["ols_slope"] == -0.174
    assert r["heat_on_canopy"]["ols_r2"] == 0.425
    assert round(r["heat_illness"]["r_poverty"], 2) == 0.59
    assert round(r["heat_illness"]["r_temperature"], 2) == 0.20
    assert r["ranking_rules"]["canopy_gap"]["poorest_third_in_top10"] == 4
    assert r["ranking_rules"]["equity_weighted"]["poorest_third_in_top10"] == 9
    assert r["prediction"]["r2_leave_one_borough_out"] < 0 < r["prediction"]["r2_random_5fold"]
    assert len(r["robustness"]["robust_priorities"]) == 6
