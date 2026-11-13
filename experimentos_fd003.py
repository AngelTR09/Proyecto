# -*- coding: utf-8 -*-
"""Diagnóstico de la caída en FD003 usando SOLO train_FD003 (no toca test_FD003 ni RUL_FD003).

Separa dos explicaciones posibles del mal resultado confirmatorio:
  A) Transferencia: el modelo entrenado con FD001 no reconoce el segundo modo de falla.
     -> se mide el modelo congelado sobre los ciclos de train_FD003.
  B) Variables: las 14 columnas elegidas en FD001 no llevan señal suficiente de FD003.
     -> se entrena y valida con GroupKFold dentro de train_FD003 (mismo ensamble y variables).
Si B da un error normal y A da uno alto, el problema es de transferencia entre modos de falla.

    python experimentos_fd003.py
"""
import joblib
import numpy as np
import pandas as pd

from app.features import build_features_multi
from entrenar_modelo import MLP_PARAMS, XGB_PARAMS, mlp_reg, pipe, xgb_reg
from ml_utils import CAP, EVAL_MAX, all_metrics, cv_predict, cv_report, load_train

ZONES = [(-1, 25, "0-25"), (25, 50, "26-50"), (50, 75, "51-75"), (75, 100, "76-100"), (100, 125, "101-125")]


def zonas(p, y):
    return {f"sesgo_{lab}": float(np.mean(np.clip(p, 0, EVAL_MAX)[(y > lo) & (y <= hi)] - y[(y > lo) & (y <= hi)])) for lo, hi, lab in ZONES}


tr = load_train("data", "FD003")
X, y_raw, groups = build_features_multi(tr), tr["RUL"].astype(float), tr["unit"].to_numpy()
yr = y_raw.to_numpy()
z = yr <= EVAL_MAX
print(f"FD003 entrenamiento: {tr['unit'].nunique()} motores, {len(tr)} ciclos")

b = joblib.load("models/modelo_rul_fd001.joblib")
pa = np.clip(b["model"].predict(X[b["features"]]), 0, EVAL_MAX)
A = {"caso": "A) modelo FD001 congelado sobre train_FD003", **all_metrics(yr[z], pa[z]), **zonas(pa, yr)}

y = y_raw.clip(upper=CAP)
o = (cv_predict(pipe(xgb_reg(**XGB_PARAMS)), X, y, groups) + cv_predict(pipe(mlp_reg(**MLP_PARAMS)), X, y, groups)) / 2
r = cv_report(o, y_raw, groups)
B = {"caso": "B) entrenar y validar dentro de FD003 (GroupKFold)", "MAE": r["MAE"], "RMSE": r["RMSE"], "R2": r["R2"], "NASA": r["NASA"],
     "RMSE_sd_pliegues": r["RMSE_std"], **zonas(o, yr)}
tbl = pd.DataFrame([A, B])
tbl.to_csv("docs/resultados/diagnostico_fd003.csv", index=False)
pd.set_option("display.width", 250)
print(tbl.round(2).T.to_string())
