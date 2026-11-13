"""Utilidades compartidas por `entrenar_modelo.py` y `experimentos_semanas.py`.

Datos, métricas y validación cruzada agrupada por motor (semana 7: validación de modelos).
"""
import os
import time
import warnings
import zipfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

SEED = 42
CAP = 110            # la RUL de entrenamiento se acota a 110 ciclos (RUL "piece-wise"); elegido por CV (ver docs/PROTOCOLO.md)
EVAL_MAX = 125       # zona de evaluación: RUL real <= 125, fija para que las cifras sean comparables entre valores de CAP
N_SPLITS = 5
COLS = ["unit", "cycle"] + [f"setting{i}" for i in range(1, 4)] + [f"sensor{i}" for i in range(1, 22)]
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def files_for(subset="FD001"):
    return [f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"]


FILES = files_for("FD001")   # compatibilidad
FIG_DIR = "docs/figuras"


# ---------------------------------------------------------------- datos
def _locate(data_dir, name):
    for root, _, files in os.walk(data_dir):
        if name in files:
            return os.path.join(root, name)
    return None


def ensure_data(data_dir="data", subset="FD001"):
    """Busca los archivos de C-MAPSS en `data_dir`; si faltan, intenta descargarlos."""
    FILES = files_for(subset)
    if all(_locate(data_dir, f) for f in FILES):
        return
    os.makedirs(data_dir, exist_ok=True)
    zips = [
        "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
        "https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
    ]
    for url in zips:
        try:
            print("Descargando:", url)
            urllib.request.urlretrieve(url, os.path.join(data_dir, "cmapss.zip"))
            with zipfile.ZipFile(os.path.join(data_dir, "cmapss.zip")) as zf:
                zf.extractall(data_dir)
            for r, _, fs in os.walk(data_dir):          # el zip de NASA trae otro zip dentro
                for f in fs:
                    if f.lower().endswith(".zip") and f != "cmapss.zip":
                        with zipfile.ZipFile(os.path.join(r, f)) as zf:
                            zf.extractall(r)
            if all(_locate(data_dir, f) for f in FILES):
                return
        except Exception as e:
            print("  No se pudo descargar:", e)
    mirror = "https://raw.githubusercontent.com/ericlrf/rul/main/CMAPSSData/"   # copia pública en GitHub
    try:
        for f in FILES:
            print("Descargando (espejo GitHub):", f)
            urllib.request.urlretrieve(mirror + f, os.path.join(data_dir, f))
    except Exception as e:
        print("  No se pudo descargar:", e)
    if not all(_locate(data_dir, f) for f in FILES):
        raise FileNotFoundError(f"Coloca {FILES} dentro de la carpeta '{data_dir}/'.")


def _read(data_dir, name):
    return pd.read_csv(_locate(data_dir, name), sep=r"\s+", header=None, names=COLS)


def load_train(data_dir="data", subset="FD001"):
    """Solo el entrenamiento, con la columna RUL = T_max - cycle. No toca ningún archivo de prueba."""
    ensure_data(data_dir, subset)
    train = _read(data_dir, f"train_{subset}.txt").sort_values(["unit", "cycle"]).reset_index(drop=True)
    train["RUL"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    return train


def load_test(data_dir="data", subset="FD001"):
    """Devuelve (test, rul_test). Solo debe usarlo `evaluar_final.py` (ver docs/PROTOCOLO.md)."""
    ensure_data(data_dir, subset)
    test = _read(data_dir, f"test_{subset}.txt").sort_values(["unit", "cycle"]).reset_index(drop=True)
    rul = pd.read_csv(_locate(data_dir, f"RUL_{subset}.txt"), header=None)[0].to_numpy(float)
    return test, rul


def last_rows(df, X):
    """Variables del último ciclo observado de cada motor (así se evalúa el conjunto de prueba)."""
    return X.loc[df.groupby("unit").tail(1).index].reset_index(drop=True)


# ---------------------------------------------------------------- métricas
def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def nasa_score(y_true, y_pred):
    """Puntaje asimétrico de NASA: sobreestimar la RUL (d > 0) penaliza más que subestimarla."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def all_metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": rmse(y, p), "R2": r2_score(y, p), "NASA": nasa_score(y, p)}


# ---------------------------------------------------------------- validación cruzada por motor
def cv_predict(model, X, y, groups, n_splits=N_SPLITS):
    """Predicciones fuera de pliegue (OOF). `GroupKFold` evita que ciclos de un mismo motor caigan
    en entrenamiento y validación a la vez."""
    oof = np.zeros(len(X))
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict(X.iloc[va])
    return oof


def cv_report(oof, y_raw, groups, n_splits=N_SPLITS):
    """Media y desviación entre pliegues, sobre la zona de interés (RUL real <= EVAL_MAX).

    Se compara contra la RUL real sin acotar: la zona cercana a la falla es la que importa para decidir
    mantenimiento, y los ciclos de motor sano (RUL > EVAL_MAX) diluirían el error.
    """
    y_raw = pd.Series(np.asarray(y_raw))
    rows = []
    for _, va in GroupKFold(n_splits=n_splits).split(oof, oof, groups):
        keep = (y_raw.iloc[va] <= EVAL_MAX).to_numpy()
        rows.append(all_metrics(y_raw.iloc[va].to_numpy()[keep], np.clip(oof[va][keep], 0, EVAL_MAX)))
    df = pd.DataFrame(rows)
    out = df.mean().to_dict()
    out["RMSE_std"] = float(df["RMSE"].std())
    return out


def bootstrap_ci(y_true, y_pred, n=2000, seed=SEED, alpha=0.05):
    """Intervalo de confianza (percentiles) de MAE, RMSE y NASA remuestreando los motores de prueba."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    stats = {"MAE": [], "RMSE": [], "NASA": []}
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        stats["MAE"].append(mean_absolute_error(y_true[i], y_pred[i]))
        stats["RMSE"].append(rmse(y_true[i], y_pred[i]))
        stats["NASA"].append(nasa_score(y_true[i], y_pred[i]))
    return {k: (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))) for k, v in stats.items()}


def paired_bootstrap_diff(y_true, pred_a, pred_b, groups, n=2000, seed=SEED, alpha=0.05):
    """Intervalo de la diferencia de RMSE (A - B) remuestreando los MISMOS motores para ambos modelos.

    Mucho más sensible que comparar dos intervalos independientes. Devuelve (diferencia, inf, sup);
    diferencia negativa = A es mejor. Usa solo las filas pasadas (p. ej. la zona RUL <= CAP).
    """
    rng = np.random.default_rng(seed)
    y_true, pa, pb, groups = map(np.asarray, (y_true, pred_a, pred_b, groups))
    units, inv = np.unique(groups, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(units)).astype(float)
    sa = np.bincount(inv, weights=(pa - y_true) ** 2, minlength=len(units))
    sb = np.bincount(inv, weights=(pb - y_true) ** 2, minlength=len(units))
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(units), len(units))
        diffs.append(np.sqrt(sa[i].sum() / cnt[i].sum()) - np.sqrt(sb[i].sum() / cnt[i].sum()))
    point = float(np.sqrt(sa.sum() / cnt.sum()) - np.sqrt(sb.sum() / cnt.sum()))
    return point, float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))
