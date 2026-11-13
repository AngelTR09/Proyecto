# Predicción de la vida útil restante (RUL) de motores turbofán

Proyecto de Machine Learning de la Universidad Privada Antenor Orrego (Escuela de Ingeniería de Sistemas e Inteligencia Artificial).
Estima cuántos ciclos de operación le quedan a un motor antes de fallar, a partir de datos tabulares de sensores
(NASA C-MAPSS, subconjunto FD001). Modelo final: ensamble XGBoost + MLP entrenado con la RUL acotada a 110 ciclos. En validación cruzada por motor (zona RUL ≤ 125) tiene RMSE 11,84 ± 0,49 ciclos. En la prueba de FD001, que ya se vio durante la exploración y no es evidencia confirmatoria, da RMSE 10,9 en motores con RUL ≤ 125 y 14,1 en los 100 motores (los 11 motores sanos con RUL > 125 se subestiman a propósito por el recorte). **En FD003, un segundo modo de falla que el modelo no vio, el RMSE es 37,0 (IC 95 %: 30,7 a 43,0): no transfiere.** Entrenando y validando dentro de FD003 el error vuelve a 11,7, así que las variables sirven y el fallo es de transferencia entre modos de falla. Detalle en `docs/PROTOCOLO.md`.

## Contenido

| Ruta | Descripción |
|---|---|
| `entrenar_modelo.py` | Entrenamiento y evaluación del modelo final (flujo principal). Genera `models/`, `outputs/` y `docs/figuras/` |
| `experimentos_semanas.py` | Una prueba por técnica del curso (semanas 2, 3, 7, 9, 10 y 11) con la misma validación; decide qué entra al modelo |
| `evaluar_final.py` | Único script que lee archivos de prueba; evalúa el modelo ya commiteado, una sola vez por conjunto (`docs/PROTOCOLO.md`) |
| `experimentos_recorte.py` | Recorte de la RUL (CAP), calibración isotónica y cota cuantílica, solo con validación cruzada |
| `experimentos_fd003.py` | Diagnóstico de FD003 usando únicamente su conjunto de entrenamiento |
| `ml_utils.py` | Datos (FD001 a FD004), métricas y validación cruzada por motor compartidos por los scripts |
| `rul_cmapss_fd001.py` | Notebook original exportado de Colab (versión 1 del informe); se conserva como línea base |
| `models/modelo_rul.joblib` | Modelo entrenado (regresión, clasificador de alarma y metadatos) que usa la aplicación |
| `app/` | Servicio web: API (FastAPI) y página (HTML/JS) |
| `app/features.py` | Variables del modelo. Fuente única: la usan el entrenamiento y la API |
| `app/explain.py` | Explicaciones SHAP de cada predicción |
| `tests/` | Pruebas de las variables (causalidad, coincidencia entrenamiento/servicio) y de la API |
| `docs/MEJORAS_POR_SEMANA.md` | Qué técnica de cada semana se aplicó, por qué y con qué resultado |
| `app/data/` | Los 100 motores de prueba de NASA, para el modo demostración |
| `Dockerfile`, `render.yaml` | Empaquetado y despliegue en Render |

## Cómo se usa

- **Motor de demostración:** se elige uno de los 100 motores de prueba y se mueve el control de ciclo. La página muestra la
  RUL estimada, una estimación conservadora (resta el margen con que el modelo sobreestimó en el 90 % de los motores de
  prueba), un estado de referencia (normal, inspección, crítico) y la RUL real para comparar.
- **Mis datos:** se pega o sube el historial de un motor (archivo NASA de 26 columnas sin encabezado, o CSV con encabezado).
  Se necesitan `setting1`, `setting2` y los sensores 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20 y 21, un ciclo por fila.

Las variables usan ventanas de 30 ciclos y una línea base calculada con los 10 primeros ciclos del motor, así que la API recibe el
historial completo del motor desde su primer ciclo y no un ciclo aislado. Si el historial no empieza en el ciclo 1, la respuesta incluye un aviso.

## API

| Método y ruta | Función |
|---|---|
| `GET /api/health` | Estado del servicio |
| `GET /api/info` | Columnas requeridas, límites y umbrales |
| `POST /api/predict` | Cuerpo `{"data": {"setting1": [...], "sensor2": [...], ..., "cycle": [...]}}`; devuelve la RUL de cada ciclo |
| `POST /api/explain` | Mismo cuerpo que `/api/predict` más `index` opcional; devuelve las 8 variables que más pesan en esa predicción (SHAP) |
| `GET /api/demo/engines` | Lista de motores de demostración |
| `GET /api/demo/engines/{unit}` | Historial, RUL real y predicción de un motor de demostración |
| `GET /api/demo/engines/{unit}/explain?cycle=N` | Explicación de un ciclo de un motor de demostración |

`/api/predict` y los motores de demostración devuelven además `alarm.prob`: la probabilidad, ciclo a ciclo, de que el motor falle en 30 ciclos o menos.

Documentación interactiva en `/docs`.

## Ejecutar en local

```bash
python -m venv .venv
.venv/Scripts/activate        # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Abrir http://localhost:8000. Con Docker:

```bash
docker build -t rul-motores .
docker run --rm -p 8000:8000 rul-motores
```

## Despliegue en Render

1. En https://render.com crear una cuenta e iniciar sesión con GitHub.
2. **New → Web Service**, elegir este repositorio y, en **Language/Runtime**, seleccionar **Docker**
   (alternativa: **New → Blueprint**, que lee `render.yaml`).
3. Elegir el plan **Free**, dejar la ruta del Dockerfile por defecto y, en **Advanced → Health Check Path**, poner `/api/health`.
4. **Create Web Service**. Al terminar la construcción, la URL pública aparece arriba (`https://<nombre>.onrender.com`).

El servicio escucha en el puerto que Render define en la variable `PORT`. En el plan gratuito el servicio se duerme tras un rato
sin visitas y la primera consulta tarda cerca de un minuto en responder.

## Reentrenar y probar

```bash
pip install -r requirements-train.txt
python entrenar_modelo.py --sin-busqueda   # descarga el dataset a data/, entrena y escribe models/modelo_rul.joblib
python experimentos_semanas.py --semana 2  # o 3, 7, 9, 10, 11, todas
python -m pytest -q tests
```

`entrenar_modelo.py` sin `--sin-busqueda` repite la búsqueda de hiperparámetros (unos 15 minutos). El modelo se guarda directamente
en `models/`; hay que mantener en `requirements.txt` las mismas versiones de scikit-learn y XGBoost con que se entrenó.

## Limitaciones

Los datos son simulados y el entrenamiento tiene un solo modo de falla. En FD003 (otro modo de falla) el modelo sobreestima la vida
restante en unos 54 ciclos en promedio en motores a 25 ciclos o menos de fallar, y el margen conservador cubrió al 57 % de los motores
en lugar del 90 % esperado. El modelo tiende a sobreestimar la RUL cuando al motor le quedan entre 50 y 100 ciclos y se entrenó con la
RUL acotada a 110 ciclos: por encima de ese valor la página muestra «≥ 110». Con solo 100 motores por conjunto de prueba los intervalos
son anchos (en FD001, RMSE de 11,7 a 16,4 ciclos). La explicación de la interfaz corresponde al componente XGBoost del ensamble. Es una herramienta de apoyo a la decisión, no un sistema listo para motores reales.
