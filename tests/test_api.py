"""Pruebas de la API con el modelo guardado y los motores de demostración."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.explain import contributions, xgb_component
from app.features import INPUT_COLS, build_features
from app.main import STATE, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_e_info(client):
    assert client.get("/api/health").json()["status"] == "ok"
    info = client.get("/api/info").json()
    assert info["min_history"] == 30 and info["alarm"]["available"]
    # test_metrics ya no se guarda en el modelo (la prueba se evalúa aparte, en evaluar_final.py)
    assert info["test_metrics"] is None or info["test_metrics"]["RMSE"] > 0


def test_motor_demo_devuelve_rul_alarma_y_real(client):
    r = client.get("/api/demo/engines/1").json()
    n = r["n_cycles"]
    assert len(r["rul"]) == len(r["rul_real"]) == len(r["cycles"]) == len(r["alarm"]["prob"]) == n
    assert 0 <= r["last"]["rul"] <= 125
    assert r["last"]["rul_conservative"] <= r["last"]["rul"]


def test_predict_con_datos_propios_coincide_con_la_demo(client):
    demo = client.get("/api/demo/engines/3").json()
    data = {c: demo["history"][c] for c in INPUT_COLS}
    data["cycle"] = demo["cycles"]
    own = client.post("/api/predict", json={"data": data}).json()
    assert own["rul"] == demo["rul"]


def test_aviso_si_el_historial_no_empieza_en_el_ciclo_1(client):
    demo = client.get("/api/demo/engines/3").json()
    data = {c: demo["history"][c][5:] for c in INPUT_COLS}
    data["cycle"] = demo["cycles"][5:]
    res = client.post("/api/predict", json={"data": data}).json()
    assert any("ciclo 1" in w for w in res["warnings"])


def test_columnas_faltantes_dan_422(client):
    assert client.post("/api/predict", json={"data": {"sensor2": [1.0]}}).status_code == 422


def test_explicacion_suma_la_prediccion_del_componente_xgboost(client):
    """Propiedad de SHAP: valor base + suma de contribuciones = predicción del XGBoost."""
    r = client.get("/api/demo/engines/5/explain").json()
    g = STATE["demo"][5][INPUT_COLS]
    X = build_features(g, STATE["features"]).iloc[[-1]]
    comp = xgb_component(STATE["model"])
    expected = float(comp.predict(X)[0])
    c, base = contributions(STATE["model"], X)
    assert base + c.sum() == pytest.approx(expected, abs=1e-3)
    assert r["xgb_prediction"] == pytest.approx(expected, abs=1e-3)
    assert len(r["drivers"]) == 8
    assert abs(r["drivers"][0]["contribution"]) >= abs(r["drivers"][-1]["contribution"])


def test_explicar_un_ciclo_intermedio(client):
    r = client.get("/api/demo/engines/5/explain?cycle=40").json()
    assert r["cycle_index"] == 39
    assert client.get("/api/demo/engines/5/explain?cycle=99999").status_code == 422
