"""Pruebas de las variables del modelo: causalidad y coincidencia entre entrenamiento y servicio."""
import numpy as np
import pandas as pd
import pytest

from app.features import (BASELINE_CYCLES, FEATURES, INPUT_COLS, build_features, build_features_multi,
                          feature_label)


def _engine(n=60, seed=0, unit=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n).cumsum() * 0.05 + rng.normal(size=1) for c in INPUT_COLS})
    df.insert(0, "cycle", np.arange(1, n + 1))
    df.insert(0, "unit", unit)
    return df


def test_numero_de_variables():
    assert len(FEATURES) == 44 and len(set(FEATURES)) == 44


def test_variables_son_causales():
    """La fila i no puede cambiar si se descartan los ciclos posteriores."""
    df = _engine(60)
    full = build_features(df)
    for i in (0, 5, 9, 10, 25, 59):
        part = build_features(df.iloc[: i + 1])
        np.testing.assert_allclose(part.iloc[-1].to_numpy(), full.iloc[i].to_numpy(), rtol=1e-9, atol=1e-9)


def test_linea_base_usa_solo_los_primeros_ciclos():
    df = _engine(60)
    f = build_features(df)
    base = df["sensor4"].iloc[:BASELINE_CYCLES].mean()
    np.testing.assert_allclose(f["sensor4_rel0"].iloc[BASELINE_CYCLES:].to_numpy(),
                               (df["sensor4"].iloc[BASELINE_CYCLES:] - base).to_numpy(), atol=1e-12)
    # modificar ciclos posteriores a la línea base no la altera
    df2 = df.copy()
    df2.loc[df2.index[30:], "sensor4"] += 100
    np.testing.assert_allclose(build_features(df2)["sensor4_rel0"].iloc[:30].to_numpy(),
                               f["sensor4_rel0"].iloc[:30].to_numpy(), atol=1e-12)


def test_pendiente_de_una_recta():
    df = _engine(60)
    df["sensor7"] = 2.0 * np.arange(60) + 5
    assert build_features(df)["sensor7_slope30"].iloc[-1] == pytest.approx(2.0)


def test_varios_motores_coincide_con_uno_por_uno():
    df = pd.concat([_engine(50, 1, unit=1), _engine(70, 2, unit=2)], ignore_index=True).sample(frac=1, random_state=0)
    multi = build_features_multi(df)
    assert list(multi.index) == list(df.index)
    for u, g in df.groupby("unit"):
        single = build_features(g.sort_values("cycle"))
        np.testing.assert_allclose(multi.loc[single.index].to_numpy(), single.to_numpy(), atol=1e-12)


def test_etiquetas_legibles():
    assert feature_label("sensor11_rel0").startswith("Sensor 11")
    assert feature_label("setting1") == "Ajuste operativo 1"
