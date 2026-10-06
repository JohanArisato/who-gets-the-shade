"""Spatial analysis for 'Who Gets the Shade?'. Writes data/results.json and
data/uhf42_shade_results.csv.

1. Spatial clustering: global Moran's I and local LISA (KNN k=5, 9,999 permutations)
2. Heat ~ canopy: OLS, spatial lag and spatial error models
3. Heat illness ~ temperature + poverty
4. Prediction: random 5-fold CV vs leave-one-borough-out CV
5. Five ranking rules for "which neighborhoods get trees first", plus 5,000
   random weightings of the equity-weighted score
"""
import json
from pathlib import Path

import esda
import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import KNN
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score
from spreg import OLS, ML_Error, ML_Lag

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
VARS = ["surface_temp_f", "tree_canopy_pct", "vegetation_pct", "poverty_pct", "heat_ed_rate"]


def z(s):
    return (s - s.mean()) / s.std(ddof=0)


def main():
    g = gpd.read_file(ROOT / "data" / "uhf42_shade.geojson").sort_values("uhf").reset_index(drop=True)
    pts = g.to_crs(2263).geometry.centroid
    w = KNN.from_array(np.column_stack([pts.x, pts.y]), k=5)
    w.transform = "r"
    res = {"n": len(g)}

    # 1. clustering -----------------------------------------------------------
    res["morans_i"] = {}
    for v in VARS:
        np.random.seed(SEED)
        mi = esda.Moran(g[v].values, w, permutations=9999)
        lisa = esda.Moran_Local(g[v].values, w, permutations=9999, seed=SEED)
        labels = np.array(["ns", "HH", "LH", "LL", "HL"])[np.where(lisa.p_sim < 0.05, lisa.q, 0)]
        g[f"lisa_{v}"] = labels
        res["morans_i"][v] = {"I": round(mi.I, 3), "p_sim": round(mi.p_sim, 4)}

    # 2. heat ~ canopy ---------------------------------------------------------
    y = g[["surface_temp_f"]].values
    X = g[["tree_canopy_pct"]].values
    ols = OLS(y, X, w=w, spat_diag=True, name_y="surface_temp_f", name_x=["tree_canopy_pct"])
    lag = ML_Lag(y, X, w=w, name_y="surface_temp_f", name_x=["tree_canopy_pct"])
    err = ML_Error(y, X, w=w, name_y="surface_temp_f", name_x=["tree_canopy_pct"])
    res["heat_on_canopy"] = {
        "ols_slope": round(float(ols.betas[1][0]), 3), "ols_r2": round(float(ols.r2), 3),
        "ols_p": float(ols.t_stat[1][1]),
        "lag_slope": round(float(lag.betas[1][0]), 3), "lag_rho": round(float(lag.rho), 3),
        "error_slope": round(float(err.betas[1][0]), 3), "error_lambda": round(float(err.lam), 3),
        "ols_aic": round(float(ols.aic), 1), "lag_aic": round(float(lag.aic), 1),
        "error_aic": round(float(err.aic), 1),
    }

    # 3. heat illness ----------------------------------------------------------
    y2 = g[["heat_ed_rate"]].values
    X2 = g[["surface_temp_f", "poverty_pct"]].values
    ols2 = OLS(y2, X2, w=w, spat_diag=True, name_y="heat_ed_rate", name_x=["surface_temp_f", "poverty_pct"])
    res["heat_illness"] = {
        "r_poverty": round(float(g.heat_ed_rate.corr(g.poverty_pct)), 3),
        "r_temperature": round(float(g.heat_ed_rate.corr(g.surface_temp_f)), 3),
        "r_canopy": round(float(g.heat_ed_rate.corr(g.tree_canopy_pct)), 3),
        "coef_temperature": round(float(ols2.betas[1][0]), 3), "p_temperature": round(float(ols2.t_stat[1][1]), 3),
        "coef_poverty": round(float(ols2.betas[2][0]), 3), "p_poverty": float(ols2.t_stat[2][1]),
        "r2": round(float(ols2.r2), 3),
    }

    # 4. prediction: random folds vs unseen borough ----------------------------
    feats = ["surface_temp_f", "tree_canopy_pct", "vegetation_pct", "poverty_pct"]
    Xp, yp = g[feats].values, g.heat_ed_rate.values
    rand = cross_val_predict(LinearRegression(), Xp, yp, cv=KFold(5, shuffle=True, random_state=SEED))
    lobo = cross_val_predict(LinearRegression(), Xp, yp, cv=LeaveOneGroupOut(), groups=g.borough)
    res["prediction"] = {"features": feats, "r2_random_5fold": round(r2_score(yp, rand), 3),
                         "r2_leave_one_borough_out": round(r2_score(yp, lobo), 3),
                         "r2_lobo_by_borough": {b: round(r2_score(yp[g.borough == b], lobo[g.borough == b]), 3)
                                                for b in sorted(g.borough.unique())}}

    # 5. ranking rules ---------------------------------------------------------
    poorest = g.poverty_pct.rank(ascending=False, method="first") <= len(g) / 3
    comps = pd.DataFrame({"canopy_gap": z(-g.tree_canopy_pct), "heat": z(g.surface_temp_f),
                          "poverty": z(g.poverty_pct), "heat_illness": z(g.heat_ed_rate)})
    rules = {
        "canopy_gap": comps.canopy_gap,
        "hottest_surface": comps.heat,
        "canopy_gap_and_heat": comps[["canopy_gap", "heat"]].mean(axis=1),
        "heat_illness": comps.heat_illness,
        "equity_weighted": comps.mean(axis=1),
    }
    res["ranking_rules"] = {}
    for name, score in rules.items():
        top = score.rank(ascending=False, method="first") <= 10
        g[f"rank_{name}"] = score.rank(ascending=False, method="first").astype(int)
        res["ranking_rules"][name] = {
            "poorest_third_in_top10": int((top & poorest).sum()),
            "top10": g.loc[top].sort_values(f"rank_{name}").name.tolist()}

    rng = np.random.default_rng(SEED)
    W = rng.dirichlet(np.ones(comps.shape[1]), size=5000)
    scores = comps.values @ W.T                              # 42 x 5000
    ranks = (-scores).argsort(axis=0).argsort(axis=0) + 1
    g["top10_share"] = (ranks <= 10).mean(axis=1)
    robust = g.loc[g.top10_share >= 0.9].sort_values("top10_share", ascending=False)
    res["robustness"] = {"weightings": 5000, "threshold": 0.9,
                         "robust_priorities": robust.name.tolist(),
                         "top10_share": dict(zip(robust.name, robust.top10_share.round(3)))}
    res["port_richmond"] = g.loc[g.uhf == 501, ["surface_temp_f", "heat_ed_rate"]].round(2).iloc[0].to_dict()

    (ROOT / "data" / "results.json").write_text(json.dumps(res, indent=2))
    g.drop(columns="geometry").to_csv(ROOT / "data" / "uhf42_shade_results.csv", index=False)
    g.to_file(ROOT / "data" / "uhf42_shade_results.geojson", driver="GeoJSON")
    print(json.dumps({k: res[k] for k in ["heat_on_canopy", "heat_illness", "prediction"]}, indent=1))
    print({k: v["poorest_third_in_top10"] for k, v in res["ranking_rules"].items()})
    print(res["robustness"]["robust_priorities"])


if __name__ == "__main__":
    main()
