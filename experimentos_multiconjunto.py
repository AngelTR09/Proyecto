# -*- coding: utf-8 -*-
"""Fase 2 (docs/PROTOCOLO.md, sección 7): ¿entrenar con FD001 + FD003 arregla la transferencia?

    python experimentos_multiconjunto.py --etapa entrenar    # validación cruzada repetida (lenta; guarda caché)
    python experimentos_multiconjunto.py --etapa analisis    # tablas, intervalos y comparaciones pareadas

Variantes (todas con el mismo ensamble XGBoost + MLP y recorte a CAP):
  M0 solo train_FD001 | M1 train_FD001 + train_FD003 | M2 = M1 + motores de prueba etiquetados de FD001 y FD003.
Se evalúa siempre sobre motores de entrenamiento (vida completa) que quedaron fuera del pliegue, zona RUL <= 125.
No lee las pruebas de FD002 ni FD004 (`load_test_labeled` lo impide).
"""
import argparse
import os

import numpy as np
import pandas as pd

from app.features import build_features_multi
from entrenar_modelo import MLP_PARAMS, XGB_PARAMS, mlp_reg, pipe, xgb_reg
from ml_utils import (CAP, EVAL_MAX, N_SPLITS, SEED, all_metrics, load_test_labeled, load_train, paired_bootstrap_diff,
                      pool, repeated_group_folds)

SEEDS = [0, 1, 2]
VARIANTS = ["M0", "M1", "M2"]
OOF = "outputs/oof_multi"
ZONES = [(-1, 25, "0-25"), (25, 50, "26-50"), (50, 75, "51-75"), (75, 100, "76-100"), (100, 125, "101-125")]


def datos(data_dir):
    base = pool([("FD001", load_train(data_dir, "FD001")), ("FD003", load_train(data_dir, "FD003"))])
    extra = pool([("FD001", load_test_labeled(data_dir, "FD001")), ("FD003", load_test_labeled(data_dir, "FD003"))])
    extra["unit"] += 10_000                      # que no choquen con los motores de entrenamiento
    return base, extra


def ajustar_predecir(Xa, ya, Xb):
    m1 = pipe(xgb_reg(**XGB_PARAMS)).fit(Xa, ya)
    m2 = pipe(mlp_reg(**MLP_PARAMS)).fit(Xa, ya)
    return (m1.predict(Xb) + m2.predict(Xb)) / 2


def entrenar(data_dir):
    base, extra = datos(data_dir)
    X, Xe = build_features_multi(base), build_features_multi(extra)
    y, ye = base["RUL"].clip(upper=CAP).to_numpy(), extra["RUL"].clip(upper=CAP).to_numpy()
    sub = base["subset"].to_numpy()
    os.makedirs(OOF, exist_ok=True)
    print(f"motores: {base['unit'].nunique()} de entrenamiento + {extra['unit'].nunique()} de prueba etiquetados | filas {len(base)} + {len(extra)}", flush=True)
    for s in SEEDS:
        folds = repeated_group_folds(base["unit"].to_numpy(), sub, N_SPLITS, SEED + s)
        for v in VARIANTS:
            path = f"{OOF}/{v}_semilla{s}.npy"
            if os.path.exists(path):
                continue
            oof = np.zeros(len(base))
            for k, (tr, va) in enumerate(folds):
                if v == "M0":
                    tr = tr[sub[tr] == "FD001"]
                Xa, ya = X.iloc[tr], y[tr]
                if v == "M2":
                    Xa, ya = pd.concat([Xa, Xe]), np.concatenate([ya, ye])
                oof[va] = ajustar_predecir(Xa, ya, X.iloc[va])
                print(f"  semilla {s} {v} pliegue {k + 1}/{N_SPLITS}", flush=True)
            np.save(path, oof)


def rmse_ic(y, p, groups, n=2000, alpha=0.05):
    """IC 95 % del RMSE remuestreando MOTORES (no filas)."""
    rng = np.random.default_rng(SEED)
    _, inv = np.unique(groups, return_inverse=True)
    k = inv.max() + 1
    cnt, sse = np.bincount(inv, minlength=k).astype(float), np.bincount(inv, weights=(p - y) ** 2, minlength=k)
    r = [np.sqrt(sse[i].sum() / cnt[i].sum()) for i in (rng.integers(0, k, k) for _ in range(n))]
    return float(np.quantile(r, alpha / 2)), float(np.quantile(r, 1 - alpha / 2))


def sesgos(p, y):
    p = np.clip(p, 0, EVAL_MAX)
    return {f"sesgo_{lab}": float(np.mean(p[(y > lo) & (y <= hi)] - y[(y > lo) & (y <= hi)])) for lo, hi, lab in ZONES}


def analisis(data_dir):
    base, _ = datos(data_dir)
    yr, sub, g = base["RUL"].to_numpy(float), base["subset"].to_numpy(), base["unit"].to_numpy()
    zona = yr <= EVAL_MAX
    oof = {v: [np.load(f"{OOF}/{v}_semilla{s}.npy") for s in SEEDS] for v in VARIANTS}
    avg = {v: np.mean(oof[v], 0) for v in VARIANTS}
    rows = []
    for v in VARIANTS:
        p_zone = np.clip(avg[v], 0, EVAL_MAX)
        p90 = float(np.quantile((p_zone - yr)[zona & (sub == "FD001")], 0.90))   # margen como en el modelo desplegado (FD001)
        for conj in ["FD001", "FD003", "total"]:
            m = zona & ((sub == conj) | (conj == "total"))
            fold_rmse = []
            for s in SEEDS:
                for _, va in repeated_group_folds(g, sub, N_SPLITS, SEED + s):
                    mm = np.zeros(len(g), bool); mm[va] = True; mm &= m
                    if mm.any():
                        fold_rmse.append(all_metrics(yr[mm], np.clip(oof[v][s][mm], 0, EVAL_MAX))["RMSE"])
            r = all_metrics(yr[m], p_zone[m])
            lo, hi = rmse_ic(yr[m], p_zone[m], g[m])
            rows.append({"variante": v, "conjunto": conj, "RMSE": r["RMSE"], "RMSE_sd_pliegues": float(np.std(fold_rmse, ddof=1)),
                         "RMSE_ic95_inf": lo, "RMSE_ic95_sup": hi, "MAE": r["MAE"], "NASA": r["NASA"],
                         **sesgos(avg[v][sub == conj] if conj != "total" else avg[v], yr[sub == conj] if conj != "total" else yr),
                         "cobertura_margen_p90": float(np.mean((p_zone - yr)[m] <= p90)), "p90_margen": p90})
    tbl = pd.DataFrame(rows)
    os.makedirs("docs/resultados", exist_ok=True)
    tbl.to_csv("docs/resultados/fase2_multiconjunto.csv", index=False)
    diffs = []
    for a, b in [("M1", "M0"), ("M2", "M1")]:
        for conj in ["FD001", "FD003"]:
            m = zona & (sub == conj)
            d, lo, hi = paired_bootstrap_diff(yr[m], np.clip(avg[a][m], 0, EVAL_MAX), np.clip(avg[b][m], 0, EVAL_MAX), g[m])
            diffs.append({"comparacion": f"{a} - {b}", "conjunto": conj, "dif_RMSE": d, "ic95_inf": lo, "ic95_sup": hi})
    dif = pd.DataFrame(diffs)
    dif.to_csv("docs/resultados/fase2_diferencias.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(tbl.round(2).to_string(index=False)); print(); print(dif.round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", choices=["entrenar", "analisis"], required=True)
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    np.random.seed(SEED)
    entrenar(a.data_dir) if a.etapa == "entrenar" else analisis(a.data_dir)
