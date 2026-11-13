# -*- coding: utf-8 -*-
"""Evaluación final del modelo congelado sobre un conjunto de prueba de C-MAPSS.

    python evaluar_final.py --subset FD003        # conjunto virgen: transferencia a otro modo de falla
    python evaluar_final.py --subset FD001        # prueba oficial de FD001 (ya vista durante la exploración)

Es el ÚNICO script que lee archivos de prueba. Reglas (ver docs/PROTOCOLO.md):
  1. El modelo y `app/features.py` deben estar commiteados: el script se niega a correr con cambios sin
     commitear en `models/` o `app/`, y guarda el hash del commit evaluado.
  2. Cada subconjunto se evalúa una sola vez. Si ya existe `outputs/final_<subset>.json` se niega a repetir
     (salvo `--repetir`, que deja constancia en el archivo).
  3. Nada de lo que imprime se usa para elegir, ajustar ni calibrar el modelo.
"""
import argparse
import json
import os
import subprocess
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from app.features import build_features_multi
from ml_utils import EVAL_MAX, SEED, all_metrics, bootstrap_ci, last_rows, load_test

ZONES = [-1, 25, 50, 75, 100, 1000]
ZONE_LABELS = ["0-25", "26-50", "51-75", "76-100", ">100"]


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD003", choices=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--modelo", default="models/modelo_rul.joblib")
    ap.add_argument("--etiqueta", default="", help="prefijo de los archivos de salida, para no pisar evaluaciones de otros modelos")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--repetir", action="store_true", help="vuelve a evaluar un subconjunto ya evaluado (queda registrado)")
    args = ap.parse_args()

    dirty = git("status", "--porcelain", "--", "models", "app", "ml_utils.py", "regimenes.py")
    if dirty:
        raise SystemExit("Hay cambios sin commitear en models/, app/ o ml_utils.py. Congela el modelo con un commit "
                         f"antes de evaluar:\n{dirty}")
    commit = git("rev-parse", "HEAD")
    os.makedirs("outputs", exist_ok=True)
    tag = f"{args.etiqueta}_" if args.etiqueta else ""
    out_json = f"outputs/final_{tag}{args.subset}.json"
    previo = None
    if os.path.exists(out_json):
        if not args.repetir:
            raise SystemExit(f"{out_json} ya existe: {args.subset} ya se evaluó. Cada conjunto se mira una sola vez.")
        previo = json.load(open(out_json, encoding="utf8")).get("commit")

    bundle = joblib.load(args.modelo)
    test, rul = load_test(args.data_dir, args.subset)
    if "normalizer" in bundle:                      # modelo con normalización por régimen (fase 3)
        test = bundle["normalizer"].transform(test)
    X = last_rows(test, build_features_multi(test))[bundle["features"]]
    assert len(X) == len(rul) == test["unit"].nunique()
    pred = np.clip(bundle["model"].predict(X), 0, None)
    cap = bundle["rul_cap"]
    rul_cap = np.minimum(rul, cap)

    res = {
        "subset": args.subset, "n_motores": int(len(rul)), "commit": commit, "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": bundle["name"], "repite_evaluacion_de_commit": previo,
        "contra_RUL_real": {k: float(v) for k, v in all_metrics(rul, pred).items()},
        "contra_RUL_acotada": {k: float(v) for k, v in all_metrics(rul_cap, pred).items()},
        "ic95_contra_RUL_real": {k: [float(x) for x in v] for k, v in bootstrap_ci(rul, pred).items()},
    }
    err = pred - rul
    zona = pd.cut(rul, ZONES, labels=ZONE_LABELS)
    by_zone = pd.DataFrame({"zona": zona, "abs": np.abs(err), "err": err}).groupby("zona", observed=True).agg(
        n=("err", "size"), MAE=("abs", "mean"), sesgo=("err", "mean"))
    # Margen conservador de la API: ¿cubre lo que prometía? (P90 de sobreestimación calculado en validación)
    p90 = bundle.get("overestimate_p90")
    if p90 is not None:
        sobre = err[rul <= EVAL_MAX]
        res["cobertura_margen_p90"] = {"p90_validacion": float(p90), "fraccion_sobreestima_menos_que_p90": float(np.mean(sobre <= p90))}

    json.dump(res, open(out_json, "w", encoding="utf8"), indent=2, ensure_ascii=False)
    by_zone.to_csv(f"outputs/final_{tag}{args.subset}_por_zona.csv")
    pd.DataFrame({"unit": np.arange(1, len(rul) + 1), "RUL_real": rul, "RUL_pred": pred}).to_csv(
        f"outputs/final_{tag}{args.subset}_predicciones.csv", index=False)

    print(f"{args.subset} · {len(rul)} motores · commit {commit[:8]} · {bundle['name']}")
    print("Contra RUL real   :", {k: round(v, 2) for k, v in res["contra_RUL_real"].items()})
    print("Contra RUL acotada:", {k: round(v, 2) for k, v in res["contra_RUL_acotada"].items()})
    print("IC 95 % (bootstrap por motor):", {k: tuple(round(x, 2) for x in v) for k, v in res["ic95_contra_RUL_real"].items()})
    print(by_zone.round(2))
    if p90 is not None:
        print("Margen P90:", {k: round(v, 3) for k, v in res["cobertura_margen_p90"].items()})


if __name__ == "__main__":
    main()
