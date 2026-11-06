"""API y servidor web del predictor de RUL (NASA C-MAPSS FD001, ensamble XGBoost + MLP)."""
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .explain import top_drivers
from .features import INPUT_COLS, MIN_HISTORY, build_features

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE.parent / "models" / "modelo_rul_fd001.joblib"
MAX_CYCLES = 1000

# Referencias de decisión. El margen conservador resta la sobreestimación que el modelo mostró en el 90 % de
# los ciclos de validación cruzada (zona RUL <= 125); lo calcula `entrenar_modelo.py` y viaja dentro del
# modelo guardado. Los umbrales del semáforo son una política ajustable, no un resultado del modelo.
DEFAULT_OVERESTIMATE_P90 = 25.5
CRITICAL_BELOW = 15       # RUL conservadora <= 15 ciclos
INSPECT_BELOW = 50        # RUL conservadora <= 50 ciclos
SATURATION = 120          # el modelo se entrenó con la RUL acotada a 125

STATE = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    bundle = joblib.load(MODEL_PATH)
    STATE["model"] = bundle["model"]
    STATE["features"] = bundle["features"]
    STATE["cap"] = bundle["rul_cap"]
    STATE["name"] = bundle["name"]
    STATE["p90"] = float(bundle.get("overestimate_p90", DEFAULT_OVERESTIMATE_P90))
    STATE["clf"] = bundle.get("classifier")
    STATE["horizon"] = int(bundle.get("alarm_horizon", 30))
    STATE["alarm_th"] = float(bundle.get("alarm_threshold", 0.5))
    STATE["test_metrics"] = bundle.get("test_metrics")
    test = pd.read_csv(BASE / "data" / "demo_test.csv")
    rul = pd.read_csv(BASE / "data" / "demo_rul.csv").set_index("unit")["rul_real"]
    STATE["demo"] = {int(u): g.sort_values("cycle").reset_index(drop=True) for u, g in test.groupby("unit")}
    STATE["demo_rul"] = {int(u): int(v) for u, v in rul.items()}
    yield


app = FastAPI(title="Predictor de RUL de motores turbofán", lifespan=lifespan)


def _status(conservative: float) -> tuple[str, str]:
    if conservative <= CRITICAL_BELOW:
        return "critico", "Programar mantenimiento de inmediato."
    if conservative <= INSPECT_BELOW:
        return "inspeccion", "Programar una inspección en los próximos ciclos."
    return "normal", "Sin acción inmediata; seguir monitoreando."


def _predict(df: pd.DataFrame) -> dict:
    """Predice la RUL en cada ciclo de un motor (df ordenado por ciclo)."""
    X = build_features(df, STATE["features"])
    pred = np.clip(STATE["model"].predict(X), 0, None)
    last = float(pred[-1])
    conservative = max(last - STATE["p90"], 0.0)
    status, advice = _status(conservative)
    warnings = []
    if len(df) < MIN_HISTORY:
        warnings.append(f"Solo hay {len(df)} ciclos; las variables usan ventanas de {MIN_HISTORY}, "
                        "por lo que la estimación es menos fiable.")
    if "cycle" in df and len(df) and int(df["cycle"].iloc[0]) != 1:
        warnings.append("El historial no empieza en el ciclo 1: la línea base del motor se calcula con los "
                        "primeros 10 ciclos recibidos y la estimación puede desviarse.")
    if last >= SATURATION:
        warnings.append(f"El modelo se entrenó con la RUL acotada a {STATE['cap']} ciclos: "
                        f"indica motor saludable, pero no distingue valores mayores.")
    out = {
        "n_cycles": len(df),
        "rul": [round(float(v), 1) for v in pred],
        "last": {"rul": round(last, 1), "rul_conservative": round(conservative, 1),
                 "status": status, "advice": advice},
        "warnings": warnings,
    }
    if STATE.get("clf") is not None:
        prob = STATE["clf"].predict_proba(X)[:, 1]
        out["alarm"] = {"horizon": STATE["horizon"], "threshold": round(STATE["alarm_th"], 2),
                        "prob": [round(float(p), 3) for p in prob],
                        "triggered": bool(prob[-1] >= STATE["alarm_th"])}
    return out


def _explain(df: pd.DataFrame, index: int | None) -> dict:
    """Variables que más pesan en la predicción de un ciclo (por defecto el último)."""
    X = build_features(df, STATE["features"])
    i = len(X) - 1 if index is None else index
    if not 0 <= i < len(X):
        raise HTTPException(422, "El ciclo pedido está fuera del historial.")
    res = top_drivers(STATE["model"], X.iloc[[i]])
    res["cycle_index"] = i
    res["note"] = ("Contribuciones SHAP exactas del componente XGBoost, que aporta la mitad del promedio "
                   "del ensamble. Positivo = aleja la falla (más vida restante); negativo = la acerca."
                   if STATE["name"].startswith("Ensamble") else
                   "Contribuciones SHAP exactas del modelo. Positivo = más vida restante; negativo = menos.")
    return res


class PredictRequest(BaseModel):
    # Formato columnar: {"setting1": [...], "sensor2": [...], ...}; opcionalmente "cycle".
    data: dict[str, list[float]]


class ExplainRequest(PredictRequest):
    index: int | None = None     # posición del ciclo dentro del historial; por defecto, el último


def _frame(data: dict[str, list[float]]) -> pd.DataFrame:
    missing = [c for c in INPUT_COLS if c not in data]
    if missing:
        raise HTTPException(422, f"Faltan columnas: {', '.join(missing)}")
    n = len(data[INPUT_COLS[0]])
    if n == 0:
        raise HTTPException(422, "No hay ciclos para evaluar.")
    if n > MAX_CYCLES:
        raise HTTPException(422, f"Máximo {MAX_CYCLES} ciclos por consulta (recibidos {n}).")
    cols = INPUT_COLS + (["cycle"] if "cycle" in data else [])
    if any(len(data[c]) != n for c in cols):
        raise HTTPException(422, "Todas las columnas deben tener la misma cantidad de valores.")
    df = pd.DataFrame({c: data[c] for c in cols})
    if not np.isfinite(df.to_numpy(float)).all():
        raise HTTPException(422, "Hay valores no numéricos o infinitos.")
    if "cycle" in df:
        df = df.sort_values("cycle", kind="stable").reset_index(drop=True)
    return df


@app.get("/api/health")
def health():
    return {"status": "ok", "model": STATE.get("name"), "features": len(STATE.get("features", []))}


@app.get("/api/info")
def info():
    return {"model": STATE["name"], "rul_cap": STATE["cap"], "input_columns": INPUT_COLS,
            "min_history": MIN_HISTORY, "max_cycles": MAX_CYCLES,
            "overestimate_p90": round(STATE["p90"], 1),
            "thresholds": {"critical": CRITICAL_BELOW, "inspection": INSPECT_BELOW},
            "test_metrics": STATE.get("test_metrics"),
            "alarm": {"horizon": STATE["horizon"], "threshold": round(STATE["alarm_th"], 2),
                      "available": STATE.get("clf") is not None}}


@app.post("/api/predict")
def predict(req: PredictRequest):
    df = _frame(req.data)
    res = _predict(df)
    res["cycles"] = df["cycle"].astype(int).tolist() if "cycle" in df else list(range(1, len(df) + 1))
    return res


@app.post("/api/explain")
def explain(req: ExplainRequest):
    return _explain(_frame(req.data), req.index)


@app.get("/api/demo/engines")
def demo_engines():
    return [{"unit": u, "n_cycles": len(g), "rul_real": STATE["demo_rul"][u]} for u, g in STATE["demo"].items()]


@app.get("/api/demo/engines/{unit}")
def demo_engine(unit: int):
    g = STATE["demo"].get(unit)
    if g is None:
        raise HTTPException(404, "Motor de demostración no encontrado.")
    res = _predict(g[INPUT_COLS])
    n, real_last = len(g), STATE["demo_rul"][unit]
    res["unit"] = unit
    res["cycles"] = g["cycle"].astype(int).tolist()
    # RUL real en cada ciclo: la del último ciclo (dato oficial de NASA) más los ciclos que faltaban.
    res["rul_real"] = [real_last + (n - 1 - i) for i in range(n)]
    res["history"] = {c: g[c].tolist() for c in INPUT_COLS}
    return res


@app.get("/api/demo/engines/{unit}/explain")
def demo_engine_explain(unit: int, cycle: int | None = None):
    g = STATE["demo"].get(unit)
    if g is None:
        raise HTTPException(404, "Motor de demostración no encontrado.")
    index = None
    if cycle is not None:
        hits = np.flatnonzero(g["cycle"].to_numpy() == cycle)
        if not len(hits):
            raise HTTPException(422, "Ese ciclo no existe en el motor.")
        index = int(hits[0])
    return _explain(g[INPUT_COLS], index)


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
