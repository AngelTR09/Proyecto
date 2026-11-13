# -*- coding: utf-8 -*-
"""Fase 3 (docs/PROTOCOLO.md, sección 8): modelo con normalización por régimen, entrenado con los cuatro train_FD00X.

    python experimentos_regimenes.py --etapa cv       # validación cruzada por motor, normalizador reajustado en cada pliegue
    python experimentos_regimenes.py --etapa final    # entrena con todo y guarda models/modelo_rul_regimenes.joblib

No lee ninguna prueba. La evaluación confirmatoria sobre test_FD002 y test_FD004 (reservadas) la hace evaluar_final.py.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor

from app.features import FEATURES, build_features_multi
from entrenar_modelo import MLP_PARAMS, XGB_PARAMS, mlp_reg, pipe, xgb_reg
from experimentos_multiconjunto import rmse_ic, sesgos
from ml_utils import CAP, EVAL_MAX, N_SPLITS, SEED, all_metrics, load_train, pool, repeated_group_folds
from regimenes import RegimeNormalizer

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def datos(data_dir):
    return pool([(s, load_train(data_dir, s)) for s in SUBSETS])


def ensamble():
    return VotingRegressor([("xgb", pipe(xgb_reg(**XGB_PARAMS))), ("mlp", pipe(mlp_reg(**MLP_PARAMS)))])


def cv(data_dir):
    df = datos(data_dir)
    yr, sub, g = df["RUL"].to_numpy(float), df["subset"].to_numpy(), df["unit"].to_numpy()
    y = np.minimum(yr, CAP)
    print(f"{df['unit'].nunique()} motores, {len(df)} ciclos", flush=True)
    oof = np.zeros(len(df))
    folds = repeated_group_folds(g, sub, N_SPLITS, SEED)
    for k, (tr, va) in enumerate(folds):
        norm = RegimeNormalizer().fit(df.iloc[tr])
        X = build_features_multi(norm.transform(df))
        oof[va] = ensamble().fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
        print(f"  pliegue {k + 1}/{N_SPLITS}", flush=True)
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/oof_regimenes.npy", oof)
    zona = yr <= EVAL_MAX
    p = np.clip(oof, 0, EVAL_MAX)
    p90 = float(np.quantile((p - yr)[zona], 0.90))
    rows = []
    for s in SUBSETS + ["total"]:
        m = zona & ((sub == s) | (s == "total"))
        r = all_metrics(yr[m], p[m])
        lo, hi = rmse_ic(yr[m], p[m], g[m])
        fr = [all_metrics(yr[m & np.isin(np.arange(len(g)), va)], p[m & np.isin(np.arange(len(g)), va)])["RMSE"] for _, va in folds
              if (m & np.isin(np.arange(len(g)), va)).any()]
        rows.append({"conjunto": s, "RMSE": r["RMSE"], "RMSE_sd_pliegues": float(np.std(fr, ddof=1)), "ic95_inf": lo, "ic95_sup": hi,
                     "MAE": r["MAE"], "R2": r["R2"], **sesgos(oof[(sub == s) | (s == "total")], yr[(sub == s) | (s == "total")]),
                     "cobertura_margen_p90": float(np.mean((p - yr)[m] <= p90)), "p90_margen": p90})
    t = pd.DataFrame(rows)
    os.makedirs("docs/resultados", exist_ok=True)
    t.to_csv("docs/resultados/fase3_regimenes_cv.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(t.round(2).to_string(index=False))


def final(data_dir):
    df = datos(data_dir)
    yr = df["RUL"].to_numpy(float)
    norm = RegimeNormalizer().fit(df)
    X = build_features_multi(norm.transform(df))
    model = ensamble().fit(X, np.minimum(yr, CAP))
    oof = np.load("outputs/oof_regimenes.npy")
    zona = yr <= EVAL_MAX
    p90 = float(np.quantile((np.clip(oof, 0, EVAL_MAX) - yr)[zona], 0.90))
    bundle = {"model": model, "features": FEATURES, "rul_cap": CAP, "name": "Ensamble XGBoost + MLP con normalización por régimen",
              "normalizer": norm, "overestimate_p90": p90, "entrenado_con": SUBSETS}
    joblib.dump(bundle, "models/modelo_rul_regimenes.joblib", compress=3)
    print("Guardado models/modelo_rul_regimenes.joblib | margen P90:", round(p90, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", choices=["cv", "final"], required=True)
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    cv(a.data_dir) if a.etapa == "cv" else final(a.data_dir)
