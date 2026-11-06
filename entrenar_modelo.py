# -*- coding: utf-8 -*-
"""Entrenamiento y evaluación del predictor de RUL (NASA C-MAPSS FD001).

Reemplaza como flujo principal al notebook exportado de Colab (`rul_cmapss_fd001.py`, que queda como línea
base del informe v1). Cada bloque indica la semana del curso de Machine Learning que lo justifica.

    python entrenar_modelo.py                 # búsqueda de hiperparámetros incluida (≈15 min)
    python entrenar_modelo.py --sin-busqueda  # usa los hiperparámetros ya encontrados (≈5 min)
    python entrenar_modelo.py --rapido        # prueba de humo (omite SVR y Random Forest)

Genera `models/modelo_rul_fd001.joblib` (lo que usa la API), tablas en `outputs/` y figuras en `docs/figuras/`.
"""
import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBClassifier, XGBRegressor

from app.explain import xgb_component
from app.features import FEATURES, build_features_multi, feature_label
from ml_utils import (CAP, FIG_DIR, N_SPLITS, SEED, all_metrics, bootstrap_ci, cv_predict, cv_report,
                      last_rows, load_data, rmse)

ALARM_HORIZON = 30          # "falla en <= 30 ciclos" (semana 3: clasificación binaria)

# Resultado de la búsqueda con GroupKFold (semana 7); `--sin-busqueda` los reutiliza.
XGB_PARAMS = dict(n_estimators=600, learning_rate=0.02, max_depth=6, subsample=0.8, colsample_bytree=0.6,
                  min_child_weight=10, reg_lambda=20)
MLP_PARAMS = dict(hidden_layer_sizes=(64, 32), alpha=1, learning_rate_init=1e-3, batch_size=256)
SPACE_XGB = {"m__n_estimators": [200, 400, 600], "m__learning_rate": [0.02, 0.03, 0.05], "m__max_depth": [3, 4, 5, 6],
             "m__subsample": [0.6, 0.8, 1.0], "m__colsample_bytree": [0.4, 0.6, 0.8],
             "m__min_child_weight": [1, 5, 10], "m__reg_lambda": [1, 5, 20]}
SPACE_MLP = {"m__hidden_layer_sizes": [(64,), (64, 32), (128, 64), (128, 64, 32)], "m__alpha": [1e-3, 1e-2, 1e-1, 1],
             "m__learning_rate_init": [1e-3, 3e-3], "m__batch_size": [128, 256]}


def pipe(est):
    # El escalado va dentro del Pipeline: se ajusta solo con el entrenamiento de cada pliegue (semana 2).
    return Pipeline([("sc", StandardScaler()), ("m", est)])


def xgb_reg(**kw):
    return XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0, **kw)


def mlp_reg(**kw):
    return MLPRegressor(early_stopping=True, max_iter=300, random_state=SEED, **kw)


def savefig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--sin-busqueda", action="store_true")
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    np.random.seed(SEED)

    # ------------------------------------------------------------ datos y variables (semanas 1 y 2)
    train, test, rul_test = load_data(args.data_dir)
    t0 = time.time()
    X = build_features_multi(train)
    X_test = last_rows(test, build_features_multi(test))
    y_raw = train["RUL"].astype(float)
    y = y_raw.clip(upper=CAP)
    groups = train["unit"].to_numpy()
    assert len(X_test) == len(rul_test) == test["unit"].nunique()
    print(f"Variables: {X.shape[1]} | filas train {len(X)} | motores test {len(X_test)} | {time.time()-t0:.0f}s")

    # ------------------------------------------------------------ familias de modelos (semanas 3 a 6)
    models = {
        "Ridge": pipe(Ridge(alpha=10, random_state=SEED)),
        "XGBoost": pipe(xgb_reg(n_estimators=200, learning_rate=0.03, max_depth=5, subsample=0.7, colsample_bytree=0.7)),
        "MLP": pipe(mlp_reg(hidden_layer_sizes=(64, 32), alpha=1e-3)),
    }
    if not args.rapido:
        models["RandomForest"] = pipe(RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5,
                                                            n_jobs=-1, random_state=SEED))
        models["SVR"] = pipe(SVR(C=10, gamma="scale", epsilon=1.0))

    rows, oofs = [], {}
    for name, m in models.items():
        t0 = time.time()
        oofs[name] = cv_predict(m, X, y, groups)
        r = cv_report(oofs[name], y_raw, groups)
        rows.append({"modelo": name, **r, "seg": time.time() - t0})
        print(f"{name:14s} CV RMSE {r['RMSE']:.2f} ± {r['RMSE_std']:.2f} | MAE {r['MAE']:.2f} | NASA {r['NASA']:.0f}")

    # ------------------------------------------------------------ hiperparámetros (semana 7)
    cv = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))
    px, pm = dict(XGB_PARAMS), dict(MLP_PARAMS)
    if not args.sin_busqueda:
        sx = RandomizedSearchCV(pipe(xgb_reg()), SPACE_XGB, n_iter=16 if not args.rapido else 4, cv=cv,
                                scoring="neg_root_mean_squared_error", random_state=SEED, refit=False).fit(X, y)
        px = {k.replace("m__", ""): v for k, v in sx.best_params_.items()}
        sm = RandomizedSearchCV(pipe(mlp_reg()), SPACE_MLP, n_iter=10 if not args.rapido else 3, cv=cv,
                                scoring="neg_root_mean_squared_error", random_state=SEED, refit=False).fit(X, y)
        pm = {k.replace("m__", ""): v for k, v in sm.best_params_.items()}
    print("XGBoost ajustado:", px, "\nMLP ajustado:", pm)

    tuned = {"XGBoost (ajustado)": pipe(xgb_reg(**px)), "MLP (ajustado)": pipe(mlp_reg(**pm))}
    for name, m in tuned.items():
        oofs[name] = cv_predict(m, X, y, groups)
    # Promedio de los dos modelos ajustados (idea de ensamble de la semana 4). Como cada uno se entrena por
    # separado en cada pliegue, el promedio de sus predicciones fuera de pliegue es la validación del ensamble.
    oofs["Ensamble XGB+MLP"] = (oofs["XGBoost (ajustado)"] + oofs["MLP (ajustado)"]) / 2
    for name in list(tuned) + ["Ensamble XGB+MLP"]:
        r = cv_report(oofs[name], y_raw, groups)
        rows.append({"modelo": name, **r, "seg": np.nan})
        print(f"{name:20s} CV RMSE {r['RMSE']:.2f} ± {r['RMSE_std']:.2f} | MAE {r['MAE']:.2f} | NASA {r['NASA']:.0f}")
    cv_tbl = pd.DataFrame(rows)
    cv_tbl.to_csv("outputs/cv_modelos.csv", index=False)

    plt.figure(figsize=(8, 4))
    t = cv_tbl.sort_values("RMSE")
    plt.barh(t["modelo"], t["RMSE"], xerr=t["RMSE_std"], color="#4C72B0")
    plt.gca().invert_yaxis()
    plt.xlabel("RMSE en validación cruzada por motor (ciclos)")
    plt.title("Comparación de modelos (zona RUL ≤ 125)")
    savefig("cv_modelos.png")

    # ------------------------------------------------------------ elección por validación cruzada, no por la prueba
    cand = cv_tbl[cv_tbl["modelo"].isin(list(tuned) + ["Ensamble XGB+MLP"])].sort_values("RMSE")
    best = cand.iloc[0]["modelo"]
    print("Modelo elegido por validación cruzada:", best)
    xgb_final = pipe(xgb_reg(**px))
    mlp_final = pipe(mlp_reg(**pm))
    if best == "Ensamble XGB+MLP":
        final = VotingRegressor([("xgb", xgb_final), ("mlp", mlp_final)])
        final_name = "Ensamble XGBoost + MLP"
    elif best.startswith("XGBoost"):
        final, final_name = xgb_final, "XGBoost"
    else:
        final, final_name = mlp_final, "MLP"
    final.fit(X, y)

    # ------------------------------------------------------------ prueba, una sola vez (semana 7)
    pred = np.clip(final.predict(X_test), 0, None)
    base = clone(models["Ridge"]).fit(X, y)
    pred_ridge = np.clip(base.predict(X_test), 0, None)
    ci = bootstrap_ci(rul_test, pred)
    test_res = pd.DataFrame({final_name: all_metrics(rul_test, pred), "Ridge (referencia)": all_metrics(rul_test, pred_ridge)}).T
    print(test_res.round(3))
    print("IC 95 % (bootstrap sobre motores):", {k: tuple(round(x, 2) for x in v) for k, v in ci.items()})
    test_res.to_csv("outputs/resultados_test.csv")
    pd.DataFrame(ci, index=["inf", "sup"]).T.to_csv("outputs/resultados_test_ic95.csv")
    pd.DataFrame({"unit": np.arange(1, len(pred) + 1), "RUL_real": rul_test, "RUL_pred": pred}).to_csv(
        "outputs/predicciones_test.csv", index=False)

    err = pred - rul_test
    bins = pd.cut(rul_test, [-1, 25, 50, 75, 100, 200], labels=["0-25", "26-50", "51-75", "76-100", ">100"])
    by_zone = pd.DataFrame({"zona": bins, "abs": np.abs(err), "err": err}).groupby("zona", observed=True).agg(
        n=("err", "size"), MAE=("abs", "mean"), sesgo=("err", "mean"))
    by_zone.to_csv("outputs/error_por_zona.csv")
    print(by_zone.round(2))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].scatter(rul_test, pred, alpha=0.7)
    lim = [0, max(rul_test.max(), pred.max()) + 5]
    ax[0].plot(lim, lim, "r--"); ax[0].set_xlabel("RUL real"); ax[0].set_ylabel("RUL predicha"); ax[0].set_title("Prueba: predicho vs. real")
    ax[1].hist(err, bins=20, color="#C44E52"); ax[1].axvline(0, color="k")
    ax[1].set_title("Error (predicha − real)\n> 0 = sobreestima (más riesgoso)"); ax[1].set_xlabel("ciclos")
    ax[2].bar(by_zone.index.astype(str), by_zone["MAE"], color="#4C72B0"); ax[2].set_title("MAE por zona de RUL real"); ax[2].set_xlabel("RUL real")
    savefig("prueba_resultados.png")

    # Margen conservador de la API: percentil 90 de la sobreestimación en validación (no en la prueba).
    ens_oof = oofs[best] if best in oofs else oofs["Ensamble XGB+MLP"]
    zone = (y_raw <= CAP).to_numpy()
    over = np.clip(ens_oof[zone], 0, CAP) - y_raw.to_numpy()[zone]
    p90 = float(np.quantile(over, 0.90))
    print(f"Sobreestimación P90 en validación: {p90:.1f} ciclos | en prueba: {np.quantile(err, 0.9):.1f} ciclos")

    # ------------------------------------------------------------ alarma binaria (semanas 3, 5 y 7)
    yb = (y_raw.to_numpy() <= ALARM_HORIZON).astype(int)
    yb_test = (rul_test <= ALARM_HORIZON).astype(int)
    clf = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.7,
                        random_state=SEED, n_jobs=-1, verbosity=0)
    oof_p = np.zeros(len(yb))
    for a, b in cv:
        oof_p[b] = clone(clf).fit(X.iloc[a], yb[a]).predict_proba(X.iloc[b])[:, 1]
    ths = np.arange(0.05, 0.96, 0.01)
    f2 = [precision_recall_fscore_support(yb, oof_p >= t, beta=2, average="binary", zero_division=0)[2] for t in ths]
    th = float(ths[int(np.argmax(f2))])      # F2: perder una falla cuesta más que una falsa alarma
    clf.fit(X, yb)
    p_test = clf.predict_proba(X_test)[:, 1]
    rows = []
    for nombre, yy, pp in [("validación cruzada (filas)", yb, oof_p), ("prueba (100 motores)", yb_test, p_test)]:
        pr, rc, f1, _ = precision_recall_fscore_support(yy, pp >= th, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(yy, pp >= th).ravel()
        rows.append({"conjunto": nombre, "umbral": th, "AUC": roc_auc_score(yy, pp), "AP": average_precision_score(yy, pp),
                     "precision": pr, "recall": rc, "F1": f1, "Brier": brier_score_loss(yy, pp), "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    alarm_tbl = pd.DataFrame(rows)
    alarm_tbl.to_csv("outputs/alarma_clasificador.csv", index=False)
    print(alarm_tbl.round(3).to_string(index=False))

    # ------------------------------------------------------------ curva de aprendizaje (semana 7)
    rng = np.random.default_rng(SEED)
    lc = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        trn_r, val_r = [], []
        for a, b in cv:
            units = np.unique(groups[a])
            sel = rng.choice(units, max(int(len(units) * frac), 2), replace=False)
            ia = a[np.isin(groups[a], sel)]
            m = clone(xgb_final).fit(X.iloc[ia], y.iloc[ia])
            ka = (y_raw.iloc[ia] <= CAP).to_numpy(); kb = (y_raw.iloc[b] <= CAP).to_numpy()
            trn_r.append(rmse(y_raw.iloc[ia][ka], np.clip(m.predict(X.iloc[ia][ka]), 0, CAP)))
            val_r.append(rmse(y_raw.iloc[b][kb], np.clip(m.predict(X.iloc[b][kb]), 0, CAP)))
        lc.append({"motores_entrenamiento": int(len(np.unique(groups)) * 0.8 * frac), "RMSE_entrenamiento": np.mean(trn_r),
                   "RMSE_validacion": np.mean(val_r), "sd_validacion": np.std(val_r)})
    lc = pd.DataFrame(lc)
    lc.to_csv("outputs/curva_aprendizaje.csv", index=False)
    plt.figure(figsize=(6, 4))
    plt.plot(lc["motores_entrenamiento"], lc["RMSE_entrenamiento"], "o-", label="entrenamiento")
    plt.errorbar(lc["motores_entrenamiento"], lc["RMSE_validacion"], lc["sd_validacion"], fmt="o-", label="validación (motores nuevos)")
    plt.xlabel("motores usados para entrenar"); plt.ylabel("RMSE (ciclos)"); plt.legend(); plt.title("Curva de aprendizaje (XGBoost)")
    savefig("curva_aprendizaje.png")
    print(lc.round(2).to_string(index=False))

    # ------------------------------------------------------------ interpretabilidad global: SHAP del componente XGBoost (semana 7)
    comp = xgb_component(final)
    Xs = comp.named_steps["sc"].transform(X[zone])
    contrib = comp.named_steps["m"].get_booster().predict(xgb.DMatrix(Xs, feature_names=list(X.columns)), pred_contribs=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=X.columns).sort_values(ascending=False)
    imp.to_csv("outputs/shap_importancia.csv", header=["mean_abs_shap"])
    top = imp.head(15)[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh([feature_label(c) for c in top.index], top.values, color="#4C72B0")
    plt.xlabel("|contribución SHAP| media (ciclos de RUL)"); plt.title("Variables que más pesan en la predicción")
    savefig("shap_global.png")

    # ------------------------------------------------------------ guardado
    bundle = {"model": final, "features": FEATURES, "rul_cap": CAP, "name": final_name,
              "classifier": clf, "alarm_horizon": ALARM_HORIZON, "alarm_threshold": th,
              "overestimate_p90": p90,
              "test_metrics": {k: float(v) for k, v in all_metrics(rul_test, pred).items()}}
    joblib.dump(bundle, "models/modelo_rul_fd001.joblib", compress=3)
    with open("outputs/resumen.json", "w", encoding="utf-8") as fh:
        json.dump({"modelo": final_name, "xgb": px, "mlp": {k: (list(v) if isinstance(v, tuple) else v) for k, v in pm.items()},
                   "p90_validacion": p90, "umbral_alarma": th, "test": bundle["test_metrics"]}, fh, indent=2, ensure_ascii=False)
    print("Guardado: models/modelo_rul_fd001.joblib")


if __name__ == "__main__":
    main()
