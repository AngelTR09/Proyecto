# -*- coding: utf-8 -*-
"""Figura de la evaluación final a partir de las predicciones que dejó `evaluar_final.py` (no vuelve a evaluar nada)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for a, (sub, tit) in zip(ax, [("FD001", "FD001 (prueba ya vista; referencia)"), ("FD003", "FD003 (conjunto virgen; otro modo de falla)")]):
    d = pd.read_csv(f"docs/resultados/final_{sub}_predicciones.csv")
    m = json.load(open(f"docs/resultados/final_{sub}.json", encoding="utf8"))["contra_RUL_real"]
    a.scatter(d["RUL_real"], d["RUL_pred"], alpha=.7, color="#4C72B0")
    lim = [0, max(d["RUL_real"].max(), d["RUL_pred"].max()) + 5]
    a.plot(lim, lim, "r--", label="predicción perfecta")
    a.set_title(f"{tit}\nRMSE {m['RMSE']:.1f} · MAE {m['MAE']:.1f}", fontsize=10)
    a.set_xlabel("RUL real (ciclos)")
ax[0].set_ylabel("RUL predicha (ciclos)"); ax[0].legend()
plt.tight_layout(); plt.savefig("docs/figuras/prueba_resultados.png", dpi=130)
