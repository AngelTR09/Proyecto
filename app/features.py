"""Variables de entrada del modelo (fuente única para el entrenamiento y la API).

El script de entrenamiento (`entrenar_modelo.py`) importa este mismo módulo, así que las variables que
ve el modelo al entrenar son exactamente las que calcula el servicio. Antes la lógica estaba duplicada
y había que mantener las dos copias a mano.

Tres familias de variables por sensor, todas causales (la fila i solo usa los ciclos 0..i del motor):

- ``rel0``: valor del sensor menos la media de los primeros ``BASELINE_CYCLES`` ciclos del propio motor.
  Cada motor sale de fábrica con valores algo distintos; restar su línea base deja solo el desgaste.
- ``ewm30``: media móvil exponencial (span 30). Quita ruido sin perder la tendencia.
- ``slope30``: pendiente de una recta ajustada a los últimos 30 ciclos. Mide qué tan rápido se degrada.

Requisito de la API: el historial debe empezar en el primer ciclo del motor, porque la línea base se
calcula con sus primeros 10 ciclos.
"""
import re

import numpy as np
import pandas as pd

SETTINGS = ["setting1", "setting2"]
SENSORS = [f"sensor{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
INPUT_COLS = SETTINGS + SENSORS      # columnas que debe traer cada ciclo

BASELINE_CYCLES = 10                 # ciclos iniciales que definen la "línea base" del motor
EWM_SPAN = 30
SLOPE_WINDOW = 30
MIN_HISTORY = 30                     # ciclos mínimos para que las variables de ventana sean completas

KINDS = ("rel0", "ewm30", "slope30")
FEATURES = SETTINGS + [f"{s}_{k}" for s in SENSORS for k in KINDS]

_DERIVED = re.compile(r"^(sensor\d+)_(rel0|ewm30|slope30)$")


def _slope(a):
    n = len(a)
    t = np.arange(n) - (n - 1) / 2.0
    return (t * a).sum() / (t * t).sum()


def _baseline(x: pd.Series) -> pd.Series:
    """Media de los primeros BASELINE_CYCLES ciclos; antes de completarlos, media acumulada (causal)."""
    cum = x.expanding(min_periods=1).mean()
    base = cum.copy()
    if len(cum) > BASELINE_CYCLES:
        base.iloc[BASELINE_CYCLES:] = cum.iloc[BASELINE_CYCLES - 1]
    return base


def build_features(history: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    """Calcula las variables del modelo para cada ciclo de UN motor.

    `history` debe estar ordenado por ciclo y contener INPUT_COLS. La fila i de la salida solo depende
    de las filas 0..i, así que predecir sobre toda la trayectoria equivale a predecir ciclo a ciclo.
    """
    out = {}
    for name in features:
        m = _DERIVED.match(name)
        if m is None:
            out[name] = history[name].astype(float).to_numpy()
            continue
        sensor, kind = m.groups()
        x = history[sensor].astype(float).reset_index(drop=True)
        if kind == "rel0":
            v = x - _baseline(x)
        elif kind == "ewm30":
            v = x.ewm(span=EWM_SPAN, adjust=False).mean()
        else:  # slope30
            v = x.rolling(SLOPE_WINDOW, min_periods=2).apply(_slope, raw=True).fillna(0.0)
        out[name] = v.to_numpy()
    return pd.DataFrame(out, index=history.index)[features]


def build_features_multi(df: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    """Igual que `build_features`, pero para un DataFrame con varios motores (columnas `unit` y `cycle`).

    Devuelve las variables con el mismo índice y orden que `df`.
    """
    parts = [build_features(g.sort_values("cycle"), features) for _, g in df.groupby("unit")]
    return pd.concat(parts).loc[df.index]


# Nombres legibles para las explicaciones (SHAP) que muestra la interfaz.
_KIND_TEXT = {
    "rel0": "cambio respecto a su línea base inicial",
    "ewm30": "nivel suavizado (30 ciclos)",
    "slope30": "tendencia de los últimos 30 ciclos",
}


def feature_label(name: str) -> str:
    m = _DERIVED.match(name)
    if m is None:
        return name.replace("setting", "Ajuste operativo ")
    sensor, kind = m.groups()
    return f"Sensor {sensor.replace('sensor', '')}: {_KIND_TEXT[kind]}"
