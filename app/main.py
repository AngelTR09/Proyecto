"""API y servidor web del predictor de RUL (NASA C-MAPSS FD001, modelo XGBoost)."""
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .features import INPUT_COLS, MIN_HISTORY, build_features

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE.parent / "models" / "modelo_rul_fd001.joblib"
MAX_CYCLES = 1000

# Referencias de decisión. El informe (sec. 7) muestra que el modelo tiende a sobreestimar la RUL:
# en el 90 % de los motores de prueba, (predicha - real) <= 25,5 ciclos. Restar ese margen da una
# estimación conservadora. Los umbrales del semáforo son una política ajustable, no un resultado del modelo.
OVERESTIMATE_P90 = 25.5
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
    conservative = max(last - OVERESTIMATE_P90, 0.0)
    status, advice = _status(conservative)
    warnings = []
    if len(df) < MIN_HISTORY:
        warnings.append(f"Solo hay {len(df)} ciclos; el modelo usa ventanas de {MIN_HISTORY}, "
                        "por lo que la estimación es menos fiable.")
    if last >= SATURATION:
        warnings.append(f"El modelo se entrenó con la RUL acotada a {STATE['cap']} ciclos: "
                        f"indica motor saludable, pero no distingue valores mayores.")
    return {
        "n_cycles": len(df),
        "rul": [round(float(v), 1) for v in pred],
        "last": {"rul": round(last, 1), "rul_conservative": round(conservative, 1),
                 "status": status, "advice": advice},
        "warnings": warnings,
    }


class PredictRequest(BaseModel):
    # Formato columnar: {"setting1": [...], "sensor2": [...], ...}; opcionalmente "cycle".
    data: dict[str, list[float]]


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
            "overestimate_p90": OVERESTIMATE_P90,
            "thresholds": {"critical": CRITICAL_BELOW, "inspection": INSPECT_BELOW}}


@app.post("/api/predict")
def predict(req: PredictRequest):
    df = _frame(req.data)
    res = _predict(df)
    res["cycles"] = df["cycle"].astype(int).tolist() if "cycle" in df else list(range(1, len(df) + 1))
    return res


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


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
