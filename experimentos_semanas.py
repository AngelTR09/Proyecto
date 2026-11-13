# -*- coding: utf-8 -*-
"""Experimentos por semana del curso aplicados al proyecto de RUL (NASA C-MAPSS FD001).

Cada función mide UNA técnica del sílabo con la misma validación cruzada por motor (GroupKFold, 5 pliegues)
y escribe una tabla en `outputs/` (y figuras en `docs/figuras/`). Sirve para decidir con datos qué técnica entra
al modelo final y cuál solo se usa para analizar. La conclusión de cada una está en `docs/MEJORAS_POR_SEMANA.md`.

    python experimentos_semanas.py --semana 2      # ingeniería de variables (preprocesamiento)
    python experimentos_semanas.py --semana 3      # clasificación binaria (alarma de falla)
    python experimentos_semanas.py --semana 7      # LIME vs. SHAP (requiere models/modelo_rul.joblib)
    python experimentos_semanas.py --semana 9      # clustering
    python experimentos_semanas.py --semana 10     # PCA y t-SNE
    python experimentos_semanas.py --semana 11     # CNN 1D (requiere tensorflow)
    python experimentos_semanas.py --semana todas

Este script nunca toca los archivos de prueba: todo se decide con validación cruzada sobre el entrenamiento.
La evaluación en prueba vive aparte, en `evaluar_final.py` (ver docs/PROTOCOLO.md).
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_mutual_info_score, average_precision_score, precision_recall_fscore_support,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

from app.features import FEATURES, SENSORS, SETTINGS, build_features_multi, feature_label
from ml_utils import (CAP, FIG_DIR, N_SPLITS, SEED, cv_predict, cv_report, load_train)

OUT = "outputs"


def savefig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=130)
    plt.close()


def xgb_default():
    return Pipeline([("sc", StandardScaler()), ("m", XGBRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=5, subsample=0.7, colsample_bytree=0.7,
        random_state=SEED, n_jobs=-1, verbosity=0))])


def context():
    train = load_train()
    X = build_features_multi(train)
    y_raw = train["RUL"].astype(float)
    return train, X, y_raw, y_raw.clip(upper=CAP), train["unit"].to_numpy()


# =====================================================================================================
# Semana 2 · Preprocesamiento: limpieza, normalización y transformación
# =====================================================================================================
def _slope(a):
    n = len(a)
    t = np.arange(n) - (n - 1) / 2.0
    return (t * a).sum() / (t * t).sum()


def _generic_features(df, kinds):
    """Constructor genérico (solo para este experimento) de familias de variables por sensor, todas causales."""
    parts = []
    for _, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        out = {s: g[s].to_numpy(float) for s in SETTINGS}
        for s in SENSORS:
            x = g[s].astype(float).reset_index(drop=True)
            cum = x.expanding(min_periods=1).mean()
            base = cum.copy()
            if len(cum) > 10:
                base.iloc[10:] = cum.iloc[9]
            for k in kinds:
                if k == "raw": v = x
                elif k == "mean5": v = x.rolling(5, min_periods=1).mean()
                elif k == "mean10": v = x.rolling(10, min_periods=1).mean()
                elif k == "mean20": v = x.rolling(20, min_periods=1).mean()
                elif k == "std10": v = x.rolling(10, min_periods=2).std().fillna(0.0)
                elif k == "diff": v = x.diff().fillna(0.0)
                elif k == "slope10": v = x.rolling(10, min_periods=2).apply(_slope, raw=True).fillna(0.0)
                elif k == "slope20": v = x.rolling(20, min_periods=2).apply(_slope, raw=True).fillna(0.0)
                elif k == "slope30": v = x.rolling(30, min_periods=2).apply(_slope, raw=True).fillna(0.0)
                elif k == "ewm10": v = x.ewm(span=10, adjust=False).mean()
                elif k == "ewm30": v = x.ewm(span=30, adjust=False).mean()
                elif k == "rel0": v = x - base
                elif k == "relmean": v = x.rolling(5, min_periods=1).mean() - base
                else: raise ValueError(k)
                out[f"{s}_{k}"] = v.to_numpy()
        parts.append(pd.DataFrame(out, index=g.index))
    return pd.concat(parts).loc[df.index]


def semana2():
    train, _, y_raw, y, groups = context()
    v1 = ["raw", "mean5", "mean10", "std10", "diff", "slope10"]
    sets = {
        "A  sensores originales": ["raw"],
        "B  versión 1 del proyecto (medias, std, diff, pendiente 10)": v1,
        "C  B + media exponencial": v1 + ["ewm10", "ewm30"],
        "D  B + línea base por motor (rel0)": v1 + ["rel0", "relmean"],
        "E  B + ventanas largas (20 y 30)": v1 + ["mean20", "slope20", "slope30"],
        "F  rel0 + ewm + pendientes 10/30 + std": ["raw", "ewm10", "ewm30", "rel0", "slope10", "slope30", "std10"],
        "G  FINAL: rel0 + ewm30 + slope30 (44 variables)": None,
    }
    rows = []
    for name, kinds in sets.items():
        if kinds is None:
            Xa = build_features_multi(train)
        else:
            Xa = _generic_features(train, kinds)
        oof = cv_predict(xgb_default(), Xa, y, groups)
        r = cv_report(oof, y_raw, groups)
        rows.append({"conjunto": name, "n_variables": Xa.shape[1], "CV_RMSE": r["RMSE"], "CV_RMSE_sd": r["RMSE_std"],
                     "CV_MAE": r["MAE"], "CV_NASA": r["NASA"]})
        print(f"{name:62s} CV RMSE {r['RMSE']:.2f} ± {r['RMSE_std']:.2f}")
    tbl = pd.DataFrame(rows)
    tbl.to_csv(f"{OUT}/s2_variables.csv", index=False)
    plt.figure(figsize=(8.5, 4))
    plt.barh([r.split(" (")[0] for r in tbl["conjunto"]], tbl["CV_RMSE"], xerr=tbl["CV_RMSE_sd"], color="#4C72B0")
    plt.gca().invert_yaxis(); plt.xlabel("RMSE en validación cruzada por motor (ciclos)")
    plt.title("Semana 2: efecto de la ingeniería de variables (mismo XGBoost)")
    savefig("s2_variables.png")

    # Evidencia del por qué: la línea base inicial difiere de un motor a otro
    s = "sensor11"
    first = train[train["cycle"] <= 10].groupby("unit")[s].mean()
    last = train.groupby("unit").tail(10).groupby("unit")[s].mean()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(first, bins=20, color="#55A868"); ax[0].set_title(f"{s}: media de los primeros 10 ciclos, por motor")
    ax[0].set_xlabel("valor del sensor")
    ax[1].scatter(first, last, s=14); ax[1].set_xlabel("al inicio"); ax[1].set_ylabel("al final de la vida")
    ax[1].set_title("El valor inicial ya difiere entre motores")
    savefig("s2_linea_base.png")


# =====================================================================================================
# Semana 3 y 5 · Clasificación binaria (regresión logística, SVM) frente a la alarma derivada de la regresión
# =====================================================================================================
def semana3():
    train, X, y_raw, y, groups = context()
    H = 30
    yb = (y_raw.to_numpy() <= H).astype(int)
    P = lambda m: Pipeline([("sc", StandardScaler()), ("m", m)])
    clfs = {
        "Regresión logística": P(LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
        "SVM (kernel RBF)": P(SVC(C=3, gamma="scale", class_weight="balanced")),
        "Random Forest": P(RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_features=0.5,
                                                  class_weight="balanced", n_jobs=-1, random_state=SEED)),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.7,
                                 random_state=SEED, n_jobs=-1, verbosity=0),
    }
    score = lambda m, A: m.predict_proba(A)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(A)

    def best_threshold(s, yy):
        ths = np.quantile(s, np.linspace(0.5, 0.995, 200))
        f2 = [precision_recall_fscore_support(yy, s >= t, beta=2, average="binary", zero_division=0)[2] for t in ths]
        return ths[int(np.argmax(f2))]

    def summarize(name, s_cv, th):
        p, r, f1, _ = precision_recall_fscore_support(yb, s_cv >= th, average="binary", zero_division=0)
        f2 = precision_recall_fscore_support(yb, s_cv >= th, beta=2, average="binary", zero_division=0)[2]
        return {"modelo": name, "CV_AUC": roc_auc_score(yb, s_cv), "CV_AP": average_precision_score(yb, s_cv),
                "CV_precision": p, "CV_recall": r, "CV_F1": f1, "CV_F2": f2}

    rows = []
    for name, c in clfs.items():
        oof = np.zeros(len(yb))
        for a, b in GroupKFold(N_SPLITS).split(X, yb, groups):
            oof[b] = score(clone(c).fit(X.iloc[a], yb[a]), X.iloc[b])
        th = best_threshold(oof, yb)
        rows.append(summarize(name, oof, th))
        print(name, {k: round(v, 3) for k, v in rows[-1].items() if k != "modelo"})
    # Alarma que sale de la regresión (RUL predicha <= umbral): ¿aporta algo el clasificador?
    oof_r = cv_predict(xgb_default(), X, y, groups)
    s_cv = -oof_r
    ths = np.linspace(-60, -5, 111)
    f2 = [precision_recall_fscore_support(yb, s_cv >= t, beta=2, average="binary", zero_division=0)[2] for t in ths]
    rows.append(summarize("Regresión XGBoost (RUL predicha ≤ umbral)", s_cv, ths[int(np.argmax(f2))]))
    print(rows[-1]["modelo"], {k: round(v, 3) for k, v in rows[-1].items() if k != "modelo"})
    pd.DataFrame(rows).to_csv(f"{OUT}/s3_alarma_modelos.csv", index=False)


# =====================================================================================================
# Semana 7 · LIME frente a SHAP sobre una misma predicción
# =====================================================================================================
def semana7():
    import joblib
    from lime.lime_tabular import LimeTabularExplainer
    from app.explain import top_drivers
    train, X, y_raw, y, groups = context()
    bundle = joblib.load("models/modelo_rul.joblib")
    model = bundle["model"]
    order = np.argsort(y_raw.to_numpy())      # filas del entrenamiento, de la más cercana a la falla a la más sana
    picks = {"cerca de la falla": int(order[2]), "en la mitad de la vida": int(order[len(order) // 2]), "motor sano": int(order[-3])}
    explainer = LimeTabularExplainer(X.to_numpy(), feature_names=list(X.columns), mode="regression",
                                     discretize_continuous=False, random_state=SEED)
    rows = []
    for tag, i in picks.items():
        row = X.iloc[[i]]
        shap = top_drivers(model, row, k=6)["drivers"]
        lime = explainer.explain_instance(row.to_numpy()[0], model.predict, num_features=6, num_samples=3000)
        lime_top = [f for f, _ in lime.as_list()]
        shap_top = [d["feature"] for d in shap]
        overlap = len(set(shap_top) & set(lime_top))
        rows.append({"caso": tag, "motor": int(train["unit"].iloc[i]), "RUL_real": float(y_raw.iloc[i]), "RUL_pred": float(model.predict(row)[0]),
                     "coinciden_en_top6": overlap,
                     "SHAP_top3": " | ".join(feature_label(f) for f in shap_top[:3]),
                     "LIME_top3": " | ".join(feature_label(f.split(" ")[0]) for f in lime_top[:3])})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(f"{OUT}/s7_lime_vs_shap.csv", index=False)


# =====================================================================================================
# Semana 9 · Clustering (K-Means, jerárquico, DBSCAN)
# =====================================================================================================
class KMeansDistances(BaseEstimator, TransformerMixin):
    """Distancias a los centroides de K-Means como variables nuevas. Se ajusta dentro de cada pliegue."""
    def __init__(self, k=6):
        self.k = k

    def fit(self, X, y=None):
        self.km_ = KMeans(self.k, n_init=10, random_state=SEED).fit(X)
        return self

    def transform(self, X):
        return self.km_.transform(X)


def semana9():
    train, X, y_raw, y, groups = context()
    rel = X[[c for c in X.columns if c.endswith("_rel0")]].to_numpy()
    Z = StandardScaler().fit_transform(rel)
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(Z), 4000, replace=False)
    stage = pd.cut(y_raw.to_numpy(), [-1, 30, 75, 125, 1000], labels=["0-30", "31-75", "76-125", ">125"])

    rows = []
    sil = {}
    for k in range(2, 9):
        km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
        sil[k] = silhouette_score(Z[sub], km.labels_[sub])
        rows.append({"metodo": f"K-Means k={k}", "silueta": sil[k],
                     "AMI_con_etapa_de_vida": adjusted_mutual_info_score(stage, km.labels_), "grupos": k})
    ag = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(Z[sub])
    rows.append({"metodo": "Jerárquico (Ward) k=4", "silueta": silhouette_score(Z[sub], ag.labels_),
                 "AMI_con_etapa_de_vida": adjusted_mutual_info_score(stage[sub], ag.labels_), "grupos": 4})
    for eps in (1.5, 2.0, 2.5, 3.0):
        db = DBSCAN(eps=eps, min_samples=25).fit(Z[sub])
        n_cl = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        ok = db.labels_ != -1
        sil_db = silhouette_score(Z[sub][ok], db.labels_[ok]) if n_cl > 1 and ok.sum() > n_cl else np.nan
        rows.append({"metodo": f"DBSCAN eps={eps} ({(~ok).mean():.0%} ruido)", "silueta": sil_db,
                     "AMI_con_etapa_de_vida": adjusted_mutual_info_score(stage[sub], db.labels_), "grupos": n_cl})
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/s9_clustering.csv", index=False)
    print(res.round(3).to_string(index=False))

    km = KMeans(4, n_init=10, random_state=SEED).fit(Z)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(list(sil), list(sil.values()), "o-"); ax[0].set_xlabel("k"); ax[0].set_ylabel("silueta"); ax[0].set_title("K-Means: elección de k")
    data = [np.clip(y_raw.to_numpy()[km.labels_ == c], 0, 200) for c in np.argsort([y_raw.to_numpy()[km.labels_ == c].mean() for c in range(4)])[::-1]]
    ax[1].boxplot(data); ax[1].set_xticklabels([f"grupo {i+1}" for i in range(4)]); ax[1].set_ylabel("RUL real"); ax[1].set_title("Los grupos corresponden a etapas de degradación")
    savefig("s9_clustering.png")

    # ¿Las distancias a los centroides mejoran la predicción?
    def with_km(k):
        fu = FeatureUnion([("id", FunctionTransformer()), ("km", KMeansDistances(k))])
        return Pipeline([("sc", StandardScaler()), ("fu", fu), ("m", xgb_default().named_steps["m"])])
    out = []
    for name, m in [("XGBoost (referencia)", xgb_default()), ("XGBoost + distancias K-Means k=4", with_km(4)),
                    ("XGBoost + distancias K-Means k=8", with_km(8))]:
        oof = cv_predict(m, X, y, groups)
        r = cv_report(oof, y_raw, groups)
        out.append({"modelo": name, "CV_RMSE": r["RMSE"], "CV_RMSE_sd": r["RMSE_std"]})
        print(out[-1])
    pd.DataFrame(out).to_csv(f"{OUT}/s9_kmeans_como_variable.csv", index=False)


# =====================================================================================================
# Semana 10 · PCA y t-SNE
# =====================================================================================================
def semana10():
    train, X, y_raw, y, groups = context()
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(random_state=SEED).fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    print("Componentes para 90/95/99 % de varianza:", [int(np.searchsorted(cum, p) + 1) for p in (0.90, 0.95, 0.99)], "de", X.shape[1])

    rel = X[[c for c in X.columns if c.endswith("_rel0")]]
    Zr = StandardScaler().fit_transform(rel)
    pc = PCA(2, random_state=SEED).fit(Zr)
    hi = pc.transform(Zr)[:, 0]
    if np.corrcoef(hi, y_raw)[0, 1] > 0:
        hi = -hi
    tr = train.assign(PC1=hi, RUL=y_raw.to_numpy())
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(range(1, len(cum) + 1), cum, "o-", ms=3); ax[0].axhline(0.95, ls="--", c="gray")
    ax[0].set_xlabel("componentes"); ax[0].set_ylabel("varianza acumulada"); ax[0].set_title("PCA sobre las 44 variables")
    for u in (1, 2, 3, 4, 5):
        g = tr[tr["unit"] == u]
        ax[1].plot(g["RUL"], g["PC1"], lw=1, label=f"motor {u}")
    ax[1].invert_xaxis(); ax[1].set_xlabel("RUL real (hacia la falla →)"); ax[1].set_ylabel("PC1"); ax[1].legend(fontsize=7)
    ax[1].set_title("Primer componente de los sensores normalizados:\nun índice de salud que crece hacia la falla")
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(Zr), 3000, replace=False)
    emb = TSNE(2, perplexity=30, init="pca", random_state=SEED).fit_transform(Zr[sub])
    sc = ax[2].scatter(emb[:, 0], emb[:, 1], c=np.clip(y_raw.to_numpy()[sub], 0, CAP), cmap="viridis_r", s=5)
    plt.colorbar(sc, ax=ax[2], label="RUL (acotada a 125)"); ax[2].set_title("t-SNE: la degradación forma una trayectoria"); ax[2].set_xticks([]); ax[2].set_yticks([])
    savefig("s10_pca_tsne.png")

    mlp = lambda: MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1, early_stopping=True, max_iter=300, random_state=SEED)
    cfg = {
        "Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10))]),
        "Ridge + PCA 95 %": Pipeline([("sc", StandardScaler()), ("p", PCA(0.95, random_state=SEED)), ("m", Ridge(alpha=10))]),
        "MLP": Pipeline([("sc", StandardScaler()), ("m", mlp())]),
        "MLP + PCA 95 %": Pipeline([("sc", StandardScaler()), ("p", PCA(0.95, random_state=SEED)), ("m", mlp())]),
        "XGBoost": xgb_default(),
        "XGBoost + PCA 95 %": Pipeline([("sc", StandardScaler()), ("p", PCA(0.95, random_state=SEED)), ("m", xgb_default().named_steps["m"])]),
    }
    rows = []
    for name, m in cfg.items():
        oof = cv_predict(m, X, y, groups)
        r = cv_report(oof, y_raw, groups)
        rows.append({"modelo": name, "CV_RMSE": r["RMSE"], "CV_RMSE_sd": r["RMSE_std"]})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(f"{OUT}/s10_pca_en_pipeline.csv", index=False)


# =====================================================================================================
# Semana 11 · Red neuronal convolucional 1D sobre ventanas de sensores
# =====================================================================================================
def semana11(window=30):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import keras
        from keras import layers
    except ImportError:
        print("Esta semana necesita TensorFlow/Keras:  pip install tensorflow-cpu")
        return
    train, _, y_raw, y, groups = context()
    ya = y.to_numpy()

    def baseline_rel(df):             # mismo concepto que rel0: sensor menos la media de sus primeros 10 ciclos
        parts = []
        for _, g in df.groupby("unit"):
            g = g.sort_values("cycle")
            x = g[SENSORS].to_numpy(float)
            cum = np.cumsum(x, 0) / np.arange(1, len(x) + 1)[:, None]
            base = cum.copy()
            base[10:] = cum[9]
            parts.append(pd.DataFrame(x - base, index=g.index, columns=SENSORS))
        return pd.concat(parts).loc[df.index]

    def windows(R, df):
        X = np.zeros((len(df), window, R.shape[1]), np.float32)
        for _, g in df.groupby("unit"):
            ix = df.index.get_indexer(g.sort_values("cycle").index)
            a = R.iloc[ix].to_numpy(np.float32)
            pad = np.vstack([np.repeat(a[:1], window - 1, 0), a])
            for i in range(len(a)):
                X[ix[i]] = pad[i:i + window]
        return X

    Rtr = baseline_rel(train)
    sd = Rtr.std()
    Xw = windows(Rtr / sd, train)

    def build(seed):
        keras.utils.set_random_seed(seed)
        m = keras.Sequential([layers.Input((window, len(SENSORS))),
                              layers.Conv1D(32, 5, padding="same", activation="relu"),
                              layers.Conv1D(32, 5, padding="same", activation="relu"),
                              layers.Flatten(), layers.Dropout(0.3),
                              layers.Dense(64, activation="relu"), layers.Dense(1)])
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        return m

    def fit(Xa, ya_, seed):
        m = build(seed)
        m.fit(Xa, ya_ / CAP, validation_split=0.15, epochs=40, batch_size=256, verbose=0, shuffle=True,
              callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
        return m

    oof = np.zeros(len(ya))
    for k, (a, b) in enumerate(GroupKFold(N_SPLITS).split(Xw, ya, groups)):
        oof[b] = fit(Xw[a], ya[a], k).predict(Xw[b], verbose=0).ravel() * CAP
    r = cv_report(oof, y_raw, groups)
    res = pd.DataFrame([{"modelo": f"CNN 1D (ventana {window})", "CV_RMSE": r["RMSE"],
                         "CV_RMSE_sd": r["RMSE_std"], "CV_MAE": r["MAE"]}])
    res.to_csv(f"{OUT}/s11_cnn.csv", index=False)
    print(res.round(2).to_string(index=False))


STEPS = {"2": semana2, "3": semana3, "7": semana7, "9": semana9, "10": semana10, "11": semana11}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--semana", default="todas", choices=list(STEPS) + ["todas"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for key in (STEPS if a.semana == "todas" else [a.semana]):
        t0 = time.time()
        print(f"\n===== Semana {key} =====")
        STEPS[key]()
        print(f"({time.time()-t0:.0f}s)")
