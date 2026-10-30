"""Variables de entrada del modelo.

Replica la ingeniería de características de `rul_cmapss_fd001.py` (sección 4.3): ventanas causales
por motor, que solo usan el ciclo actual y los anteriores. Si esta lógica cambia en el entrenamiento,
debe cambiar aquí igual; de lo contrario las predicciones se degradan sin mostrar ningún error.
"""
import re

import numpy as np
import pandas as pd

SETTINGS = ["setting1", "setting2"]
SENSORS = [f"sensor{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
INPUT_COLS = SETTINGS + SENSORS      # columnas que debe traer cada ciclo
MIN_HISTORY = 10                     # tamaño de la ventana más larga

_DERIVED = re.compile(r"^(sensor\d+)_(mean5|mean10|std10|diff|slope10)$")


def _slope(a):
    n = len(a)
    t = np.arange(n) - (n - 1) / 2.0
    return (t * a).sum() / (t * t).sum()


def build_features(history: pd.DataFrame, features: list[str]) -> pd.DataFrame:
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
        x = history[sensor].astype(float)
        if kind == "mean5":
            v = x.rolling(5, min_periods=1).mean()
        elif kind == "mean10":
            v = x.rolling(10, min_periods=1).mean()
        elif kind == "std10":
            v = x.rolling(10, min_periods=2).std().fillna(0.0)
        elif kind == "diff":
            v = x.diff().fillna(0.0)
        else:  # slope10
            v = x.rolling(10, min_periods=2).apply(_slope, raw=True).fillna(0.0)
        out[name] = v.to_numpy()
    return pd.DataFrame(out, index=history.index)[features]
