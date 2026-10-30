# Predicción de la vida útil restante (RUL) de motores turbofán

Proyecto de Machine Learning de la Universidad Privada Antenor Orrego (Escuela de Ingeniería de Sistemas e Inteligencia Artificial).
Estima cuántos ciclos de operación le quedan a un motor antes de fallar, a partir de datos tabulares de sensores
(NASA C-MAPSS, subconjunto FD001). Modelo final: XGBoost, con MAE 13,05 y RMSE 17,77 ciclos en los 100 motores de prueba.

## Contenido

| Ruta | Descripción |
|---|---|
| `rul_cmapss_fd001.py` | Entrenamiento y evaluación completa (exportado de Colab). Genera `outputs/` |
| `models/modelo_rul_fd001.joblib` | Modelo entrenado que usa la aplicación |
| `app/` | Servicio web: API (FastAPI) y página (HTML/JS) |
| `app/features.py` | Variables de ventana; replica la sección 4.3 del entrenamiento |
| `app/data/` | Los 100 motores de prueba de NASA, para el modo demostración |
| `Dockerfile`, `render.yaml` | Empaquetado y despliegue en Render |

## Cómo se usa

- **Motor de demostración:** se elige uno de los 100 motores de prueba y se mueve el control de ciclo. La página muestra la
  RUL estimada, una estimación conservadora (resta el margen con que el modelo sobreestimó en el 90 % de los motores de
  prueba), un estado de referencia (normal, inspección, crítico) y la RUL real para comparar.
- **Mis datos:** se pega o sube el historial de un motor (archivo NASA de 26 columnas sin encabezado, o CSV con encabezado).
  Se necesitan `setting1`, `setting2` y los sensores 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20 y 21, un ciclo por fila.

El modelo usa ventanas de hasta 10 ciclos, así que la API recibe el historial del motor y no un ciclo aislado.

## API

| Método y ruta | Función |
|---|---|
| `GET /api/health` | Estado del servicio |
| `GET /api/info` | Columnas requeridas, límites y umbrales |
| `POST /api/predict` | Cuerpo `{"data": {"setting1": [...], "sensor2": [...], ..., "cycle": [...]}}`; devuelve la RUL de cada ciclo |
| `GET /api/demo/engines` | Lista de motores de demostración |
| `GET /api/demo/engines/{unit}` | Historial, RUL real y predicción de un motor de demostración |

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

## Reentrenar

`python rul_cmapss_fd001.py` descarga el dataset a `data/`, entrena y guarda los resultados en `outputs/`.
Para actualizar el servicio, copiar `outputs/modelo_rul_fd001.joblib` a `models/` y mantener en `requirements.txt` las
mismas versiones de scikit-learn y XGBoost con que se entrenó.

## Limitaciones

Los datos son simulados y de un solo modo de falla. El modelo tiende a sobreestimar la RUL en la zona intermedia y se
entrenó con la RUL acotada a 125 ciclos. Es una herramienta de apoyo a la decisión, no un sistema listo para motores reales.
