"""Explicaciones de la predicción (semana 7: interpretabilidad; la API las entrega como parte de la semana 14).

Usa los valores SHAP exactos que calcula XGBoost de forma nativa (`pred_contribs=True`), sin depender de la
librería `shap` (que añade dependencias pesadas al contenedor). Para cada fila:

    predicción del componente XGBoost = valor base + suma de las contribuciones de cada variable
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import VotingRegressor

from .features import feature_label


def xgb_component(model):
    """Devuelve el Pipeline (escalador + XGBoost) dentro del modelo guardado."""
    if isinstance(model, VotingRegressor):
        return model.named_estimators_["xgb"]
    return model


def contributions(model, X: pd.DataFrame):
    """Matriz de contribuciones (n_filas x n_variables) y valor base, en ciclos de RUL."""
    comp = xgb_component(model)
    Xs = comp.named_steps["sc"].transform(X)
    dm = xgb.DMatrix(Xs, feature_names=list(X.columns))
    c = comp.named_steps["m"].get_booster().predict(dm, pred_contribs=True)
    return c[:, :-1], float(c[0, -1])


def top_drivers(model, X_row: pd.DataFrame, k: int = 8):
    """Las k variables que más empujan la predicción de una fila, con signo."""
    c, base = contributions(model, X_row)
    c = c[0]
    order = np.argsort(-np.abs(c))[:k]
    drivers = [{"feature": X_row.columns[i], "label": feature_label(X_row.columns[i]),
                "value": float(X_row.iloc[0, i]), "contribution": float(c[i])} for i in order]
    return {"base": base, "drivers": drivers, "xgb_prediction": float(base + c.sum())}
