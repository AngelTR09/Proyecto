# -*- coding: utf-8 -*-
"""Figura de la evaluación final a partir de las predicciones que dejó `evaluar_final.py` (no vuelve a evaluar nada)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = "docs/resultados"
PANELES = [("final_FD003", "FD003 con el modelo de un solo modo (FD001)\n(conjunto virgen en ese momento)"),
           ("final_fase2_FD003", "FD003 con el modelo FD001 + FD003\n(ya no virgen: solo referencia)"),
           ("final_regimenes_FD002", "FD002 con normalización por régimen\n(conjunto virgen)"),
           ("final_regimenes_FD004", "FD004 con normalización por régimen\n(conjunto virgen)")]
fig, ax = plt.subplots(2, 2, figsize=(11, 8.6), sharex=True, sharey=True)
for a, (base, tit) in zip(ax.ravel(), PANELES):
    d = pd.read_csv(f"{R}/{base}_predicciones.csv")
    j = json.load(open(f"{R}/{base}.json", encoding="utf8"))
    z = d["RUL_real"] <= 125
    rmse125 = (((d["RUL_pred"].clip(upper=125) - d["RUL_real"])[z]) ** 2).mean() ** .5
    a.scatter(d["RUL_real"], d["RUL_pred"], alpha=.55, s=14, color="#4C72B0")
    a.plot([0, 200], [0, 200], "r--", lw=1)
    a.axvline(125, color="gray", ls=":", lw=1)
    a.set_title(f"{tit}\nRMSE con RUL real ≤ 125: {rmse125:.1f}", fontsize=9.5)
    a.set_xlim(0, 200); a.set_ylim(0, 150)
for a in ax[1]: a.set_xlabel("RUL real (ciclos)")
for a in ax[:, 0]: a.set_ylabel("RUL predicha (ciclos)")
plt.tight_layout(); plt.savefig("docs/figuras/prueba_resultados.png", dpi=130)
