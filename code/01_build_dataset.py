"""Build the 42-neighborhood analysis table from public NYC data.

Inputs (data/raw/, downloaded by `python code/00_download.py`):
  All_EHDP_data.csv   NYC DOHMH Environment & Health Data Portal export
  UHF42.geo.json      United Hospital Fund neighborhood boundaries

Output:
  data/uhf42_shade.csv and data/uhf42_shade.geojson

Data note: in the portal export, Bensonhurst - Bay Ridge (UHF 209) is filed
under code 208 (Canarsie - Flatlands). Rows are re-keyed by neighborhood
name and the script asserts every code matches its boundary name.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

INDICATORS = {  # indicator_id: (column, time period or None = average all years)
    688: ("surface_temp_f", "2009"),     # daytime summer surface temperature, Landsat
    708: ("tree_canopy_pct", "2014"),
    690: ("vegetation_pct", "2010"),
    221: ("poverty_pct", "2013-17"),     # % below federal poverty level, ACS
    537: ("heat_ed_rate", None),         # heat-stress ED visits per 100k, age-adjusted
}


# Portal names that differ from the boundary file beyond spacing/case.
ALIASES = {"fordhambronxpk": "fordhambronxpark", "washingtonheights": "washingtonheightsinwood",
           "rockaways": "rockaway"}


def norm(name: str) -> str:
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    return ALIASES.get(key, key)


def main():
    geo = gpd.read_file(RAW / "UHF42.geo.json")
    geo = geo[geo.GEOCODE > 0].rename(columns={"GEOCODE": "uhf", "GEONAME": "name",
                                                "BOROUGH": "borough"})
    name_to_code = {norm(n): c for n, c in zip(geo.name, geo.uhf)}

    df = pd.read_csv(RAW / "All_EHDP_data.csv", low_memory=False)
    df = df[(df.geo_type_name == "UHF42") & df.indicator_id.isin(INDICATORS)].copy()
    # Re-key every row by neighborhood NAME (fixes the 208/209 filing error).
    df["uhf"] = df.geo_place_name.map(norm).map(name_to_code)
    missing = df.loc[df.uhf.isna(), "geo_place_name"].unique()
    assert len(missing) == 0, f"Unmatched neighborhood names: {missing}"
    n_fixed = int((df.uhf.astype(int) != pd.to_numeric(df.geo_join_id)).sum())

    cols = {}
    for ind, (col, period) in INDICATORS.items():
        d = df[df.indicator_id == ind]
        if period:
            d = d[d.time_period.astype(str) == period]
            assert d.uhf.is_unique, f"{col}: duplicate neighborhoods"
            cols[col] = d.set_index("uhf").data_value
        else:  # unweighted mean of the yearly age-adjusted rates
            cols[col] = d.groupby("uhf").data_value.mean()
            cols["heat_ed_years"] = d.groupby("uhf").time_period.nunique()
    tab = pd.DataFrame(cols)
    tab.index = tab.index.astype(int)

    out = geo.merge(tab, left_on="uhf", right_index=True, how="left")
    assert len(out) == 42 and out[list(c for c, _ in INDICATORS.values())].notna().all().all()
    for code, name in zip(out.uhf, out.name):          # final consistency check
        assert name_to_code[norm(name)] == code

    out = out.sort_values("uhf")
    out.drop(columns="geometry").to_csv(ROOT / "data" / "uhf42_shade.csv", index=False)
    out.to_file(ROOT / "data" / "uhf42_shade.geojson", driver="GeoJSON")
    print(f"42 neighborhoods written; {n_fixed} rows re-keyed by name.")


if __name__ == "__main__":
    main()
