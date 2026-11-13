# -*- coding: utf-8 -*-
"""Normalización por régimen de operación (FD002 y FD004 tienen 6 condiciones de vuelo; FD001 y FD003, una).

En esos conjuntos los sensores saltan según la condición, y ese salto tapa el desgaste: sin normalizar, la
correlación de los sensores con la vida restante es casi 0; normalizando dentro de cada régimen llega a -0,77.

    1. Se agrupan setting1-3 con K-Means (k = 6; recupera exactamente las 6 condiciones).
    2. Cada sensor se estandariza con la media y la desviación de su régimen, calculadas SOLO con los datos de ajuste.
    3. Recién después se calculan rel0, ewm30 y slope30 (app.features) sobre los valores normalizados.

Este módulo es de investigación: la API desplegada no lo usa (asume un solo régimen).
"""
import numpy as np
from sklearn.cluster import KMeans

from app.features import SENSORS

SETTINGS3 = ["setting1", "setting2", "setting3"]


class RegimeNormalizer:
    def __init__(self, k=6, seed=42):
        self.k, self.seed = k, seed

    def fit(self, df):
        S = df[SETTINGS3].to_numpy(float)
        self.mu_s_, self.sd_s_ = S.mean(0), S.std(0) + 1e-9
        self.km_ = KMeans(self.k, n_init=10, random_state=self.seed).fit((S - self.mu_s_) / self.sd_s_)
        lab = self.km_.labels_
        X = df[SENSORS].to_numpy(float)
        self.mu_ = np.vstack([X[lab == r].mean(0) for r in range(self.k)])
        self.sd_ = np.vstack([X[lab == r].std(0) for r in range(self.k)]) + 1e-9
        return self

    def regimes(self, df):
        return self.km_.predict((df[SETTINGS3].to_numpy(float) - self.mu_s_) / self.sd_s_)

    def transform(self, df):
        lab = self.regimes(df)
        out = df.copy()
        out[SENSORS] = (df[SENSORS].to_numpy(float) - self.mu_[lab]) / self.sd_[lab]
        return out
