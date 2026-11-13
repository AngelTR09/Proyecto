# -*- coding: utf-8 -*-
"""Sesgo por el recorte de la RUL (limitación 3). Todo con validación cruzada por motor sobre el entrenamiento.

    python experimentos_recorte.py --etapa cap                 # compara CAP = 100, 110, 125, 140, 150
    python experimentos_recorte.py --etapa calibracion --cap 125   # isotónica sobre predicciones fuera de pliegue
    python experimentos_recorte.py --etapa cuantil --cap 125       # cota con regresión cuantílica vs margen fijo

Nunca lee archivos de prueba (docs/PROTOCOLO.md). Las predicciones fuera de pliegue se guardan en
`outputs/oof/` para poder repetir las etapas 2 y 3 sin reentrenar.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from app.features import build_features_multi
from entrenar_modelo import MLP_PARAMS, XGB_PARAMS, mlp_reg, pipe, xgb_reg
from ml_utils import N_SPLITS, SEED, all_metrics, cv_predict, cv_report, load_train, paired_bootstrap_diff, rmse

EVAL_MAX = 125                                  # zona de interés: RUL real <= 125, igual en todas las comparaciones
OUT, OOF = "outputs", "outputs/oof"
ZONES = [(-1, 25, "0-25"), (25, 50, "26-50"), (50, 75, "51-75"), (75, 100, "76-100"), (100, 125, "101-125")]


def contexto(data_dir):
    train = load_train(data_dir)
    X = build_features_multi(train)
    return train, X, train["RUL"].astype(float), train["unit"].to_numpy()


def oof_ensamble(X, y_raw, groups, cap):
    """Predicciones fuera de pliegue del ensamble XGBoost + MLP entrenado con la RUL acotada a `cap`."""
    path = f"{OOF}/ensamble_cap{cap}.npy"
    if os.path.exists(path):
        return np.load(path)
    y = y_raw.clip(upper=cap)
    o = (cv_predict(pipe(xgb_reg(**XGB_PARAMS)), X, y, groups) + cv_predict(pipe(mlp_reg(**MLP_PARAMS)), X, y, groups)) / 2
    os.makedirs(OOF, exist_ok=True)
    np.save(path, o)
    return o


def sesgo_por_zona(pred, y_raw):
    p = np.clip(pred, 0, EVAL_MAX)
    return {f"sesgo_{lab}": float(np.mean(p[(y_raw > lo) & (y_raw <= hi)] - y_raw[(y_raw > lo) & (y_raw <= hi)]))
            for lo, hi, lab in ZONES}


def etapa_cap(data_dir, caps):
    _, X, y_raw, groups = contexto(data_dir)
    yr = y_raw.to_numpy()
    zona = yr <= EVAL_MAX
    oofs = {c: oof_ensamble(X, y_raw, groups, c) for c in caps}
    ref = 125 if 125 in oofs else caps[0]
    rows = []
    for c, o in oofs.items():
        r = cv_report(o, y_raw, groups)
        d, lo, hi = paired_bootstrap_diff(yr[zona], np.clip(o[zona], 0, EVAL_MAX), np.clip(oofs[ref][zona], 0, EVAL_MAX), groups[zona])
        rows.append({"CAP": c, "RMSE": r["RMSE"], "RMSE_sd_pliegues": r["RMSE_std"], "MAE": r["MAE"], "NASA": r["NASA"],
                     **sesgo_por_zona(o, yr),
                     "RMSE_vs_objetivo_acotado": rmse(np.minimum(yr, c), np.clip(o, 0, c)),
                     f"dif_RMSE_vs_{ref}": d, "dif_inf95": lo, "dif_sup95": hi})
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/recorte_cap.csv", index=False)


def etapa_calibracion(data_dir, cap):
    _, X, y_raw, groups = contexto(data_dir)
    yr = y_raw.to_numpy()
    zona = yr <= EVAL_MAX
    o = oof_ensamble(X, y_raw, groups, cap)
    target = np.minimum(yr, cap)
    variantes = {"sin calibrar": o}
    # Cada pliegue se calibra con las predicciones fuera de pliegue de los otros cuatro (los motores del pliegue
    # evaluado no intervienen en el ajuste de la isotónica).
    for nombre, usar_zona, y_fit in [("isotónica (solo zona <= 125, objetivo = RUL real)", True, yr),
                                     ("isotónica (todas las filas, objetivo acotado)", False, target)]:
        cal = o.copy()
        for tr, va in GroupKFold(n_splits=N_SPLITS).split(o, o, groups):
            m = zona[tr] if usar_zona else np.ones(len(tr), bool)
            cal[va] = IsotonicRegression(out_of_bounds="clip").fit(o[tr][m], y_fit[tr][m]).predict(o[va])
        variantes[nombre] = cal
    rows = []
    base = variantes["sin calibrar"]
    for nombre, p in variantes.items():
        r = cv_report(p, y_raw, groups)
        d, lo, hi = paired_bootstrap_diff(yr[zona], np.clip(p[zona], 0, EVAL_MAX), np.clip(base[zona], 0, EVAL_MAX), groups[zona])
        rows.append({"variante": nombre, "RMSE": r["RMSE"], "RMSE_sd_pliegues": r["RMSE_std"], "MAE": r["MAE"], "NASA": r["NASA"],
                     **sesgo_por_zona(p, yr), "dif_RMSE_vs_sin_calibrar": d, "dif_inf95": lo, "dif_sup95": hi})
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    pd.DataFrame(rows).to_csv(f"{OUT}/recorte_calibracion_cap{cap}.csv", index=False)


def etapa_cuantil(data_dir, cap):
    _, X, y_raw, groups = contexto(data_dir)
    yr = y_raw.to_numpy()
    zona = yr <= EVAL_MAX
    o = np.clip(oof_ensamble(X, y_raw, groups, cap), 0, EVAL_MAX)
    path = f"{OOF}/cuantil10_cap{cap}.npy"
    if os.path.exists(path):
        q = np.load(path)
    else:
        q = cv_predict(pipe(xgb_reg(objective="reg:quantileerror", quantile_alpha=0.1, **XGB_PARAMS)), X, y_raw.clip(upper=cap), groups)
        os.makedirs(OOF, exist_ok=True)
        np.save(path, q)
    q = np.clip(q, 0, EVAL_MAX)
    fijo = np.zeros(len(o))
    for tr, va in GroupKFold(n_splits=N_SPLITS).split(o, o, groups):     # margen P90 calculado con los otros pliegues
        t = tr[zona[tr]]
        fijo[va] = np.maximum(o[va] - np.quantile(o[t] - yr[t], 0.90), 0)
    rows = []
    for nombre, b in [("margen fijo (P90 de la sobreestimación)", fijo), ("regresión cuantílica α = 0,1", q)]:
        fila = {"cota": nombre, "cobertura_total": float(np.mean(yr[zona] >= b[zona])),
                "holgura_media": float(np.mean(yr[zona] - b[zona]))}
        for lo, hi, lab in ZONES:
            m = (yr > lo) & (yr <= hi)
            fila[f"cobertura_{lab}"] = float(np.mean(yr[m] >= b[m]))
        rows.append(fila)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fila.items()})
    pd.DataFrame(rows).to_csv(f"{OUT}/recorte_cuantil_cap{cap}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", choices=["cap", "calibracion", "cuantil"], required=True)
    ap.add_argument("--cap", type=int, default=125)
    ap.add_argument("--caps", type=int, nargs="+", default=[100, 110, 125, 140, 150])
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    np.random.seed(SEED)
    if a.etapa == "cap":
        etapa_cap(a.data_dir, a.caps)
    elif a.etapa == "calibracion":
        etapa_calibracion(a.data_dir, a.cap)
    else:
        etapa_cuantil(a.data_dir, a.cap)
