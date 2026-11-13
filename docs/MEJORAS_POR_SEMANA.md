# Mejoras al proyecto de RUL a partir de las semanas 1 a 12 de Machine Learning

Este documento explica qué técnicas del curso se aplicaron al predictor de vida útil restante (NASA C-MAPSS FD001), por qué se eligieron esas y no otras, y qué resultado dio cada una. Todas las cifras salen de `entrenar_modelo.py` y de `experimentos_semanas.py`; las tablas completas quedan en `outputs/` y las figuras en `docs/figuras/`.

> **Actualización (ramas `fix/protocolo-prueba-limpia` y `fix/sesgo-recorte-rul`).** Las cifras de prueba de las tablas de abajo (MAE 9,53, RMSE 12,79, margen de 18,9 ciclos, recorte a 125) corresponden al modelo anterior y a una prueba de FD001 que ya se había visto, así que se conservan solo como historia. El modelo vigente usa recorte a 110 y margen de 15,0 ciclos, y se evaluó de forma confirmatoria en FD003 con resultado desfavorable. Ver `docs/PROTOCOLO.md` y `docs/resultados/`.

## Resultado

| Métrica en los 100 motores de prueba | Versión 1 | Ahora |
|---|---|---|
| MAE (ciclos) | 13,05 | 9,53 |
| RMSE (ciclos) | 17,77 | 12,79 |
| R² | 0,817 | 0,905 |
| Puntaje NASA (menor es mejor) | 720 | 221 |

El RMSE de validación cruzada por motor, medido en la zona de interés (RUL real hasta 125 ciclos), pasó de unos 19,7 a 12,95. Con solo 100 motores de prueba hay incertidumbre: el intervalo de confianza del 95 % del RMSE, obtenido remuestreando motores, va de 10,7 a 14,7 ciclos. Aun así, el límite superior queda por debajo del valor de la versión 1.

La mayor parte de la mejora viene de una sola decisión, tomada en la semana 2 (preprocesamiento), y no de cambiar de algoritmo.

## Cómo se decidió qué entra

Cada técnica se probó con la misma validación: `GroupKFold` de 5 pliegues agrupando por motor, para que ningún ciclo de un motor de validación pueda haber visto a su propio motor en el entrenamiento. El error se mide en la zona RUL ≤ 125, que es donde se decide el mantenimiento. Una técnica entra al modelo si baja el error de forma que supere el ruido (la desviación del RMSE entre pliegues ronda 0,65 ciclos) o si mejora algo que el curso pide cuidar, como la explicabilidad o el mantenimiento del código. Si no cumple ninguna de las dos, se documenta como análisis y no se despliega.

Un cuidado que conviene declarar: durante la exploración se imprimieron también las cifras de la prueba como referencia. Las decisiones se tomaron con la validación cruzada y la prueba coincidió en todas, pero no se puede afirmar que nunca se miró.

## Mapa de semanas

La carpeta de contexto tiene las diapositivas de las semanas 1 a 7, 9 y 10. No hay archivos de las semanas 8, 11 y 12. Para la 11 (CNN) se usó el sílabo y la sección de topologías de la semana 6; la 12 es el proyecto en sí. El sílabo asigna LIME y SHAP a la semana 7, aunque las diapositivas de esa semana tratan solo de evaluación, y se implementaron según el sílabo.

| Semana | Tema | Qué se hizo en el proyecto | Decisión |
|---|---|---|---|
| 1 | Introducción al aprendizaje automático | Se plantea como regresión supervisada y se usa Ridge como referencia clásica (RMSE de validación 17,4) | Se mantiene |
| 2 | Preprocesamiento | Línea base por motor, media exponencial y pendiente de 30 ciclos | Implementada; es la mejora principal |
| 3 | Regresión y clasificación binaria | Alarma «falla en 30 ciclos o menos» con probabilidad, umbral y matriz de confusión | Implementada |
| 4 | Árboles y bosque aleatorio | XGBoost como modelo base y promedio de modelos; Random Forest comparado | Implementada |
| 5 | SVM | SVR y SVC comparados en regresión y en alarma | Evaluada, no desplegada |
| 6 | Aprendizaje profundo | MLP dentro del ensamble | Implementada |
| 7 | Evaluación y validación | Búsqueda de hiperparámetros, curva de aprendizaje, intervalos bootstrap, error por zona, SHAP y LIME | Implementada |
| 9 | Clustering | K-Means, jerárquico y DBSCAN para describir etapas de degradación | Solo análisis |
| 10 | PCA y t-SNE | Índice de salud y mapa de la degradación; PCA dentro de pipelines | Solo análisis |
| 11 | CNN | CNN 1D sobre ventanas de 30 ciclos con Keras | Probada, descartada |
| 12 | Proyecto grupal | Integración de todo lo anterior en el proyecto | Todo lo anterior |

## Lo que se implementó

### Semana 2: línea base por motor y ventanas largas

Cada motor sale con valores de sensor algo distintos de otro, y esa diferencia existe desde los primeros ciclos (figura `s2_linea_base.png`). Un árbol no puede separar «este motor siempre marcó más alto» de «este motor se está degradando». La solución es normalizar por entidad: se resta a cada sensor la media de los 10 primeros ciclos del mismo motor. Queda solo el cambio respecto a su propio estado sano. Esa variable se llama `rel0`.

Se agregaron además la media exponencial de 30 ciclos (`ewm30`), que quita ruido sin perder la tendencia, y la pendiente de una recta ajustada a los últimos 30 ciclos (`slope30`), que mide la velocidad de degradación. Las tres son causales: el valor de un ciclo solo usa ese ciclo y los anteriores, y hay una prueba automática que lo verifica.

| Conjunto de variables (mismo XGBoost) | Variables | RMSE de validación | Prueba |
|---|---|---|---|
| Sensores originales | 16 | 20,65 ± 1,40 | 18,10 |
| Versión 1 (medias, desviación, diferencia, pendiente 10) | 86 | 19,95 ± 1,08 | 18,63 |
| Versión 1 más media exponencial | 114 | 17,98 ± 0,80 | 17,11 |
| Versión 1 más línea base por motor | 114 | 14,05 ± 1,34 | 14,60 |
| Versión 1 más ventanas de 20 y 30 | 128 | 13,99 ± 0,70 | 14,25 |
| Combinación amplia de las anteriores | 100 | 13,43 ± 0,63 | 12,76 |
| Final: `rel0` + `ewm30` + `slope30` | 44 | 13,56 ± 0,66 | 12,65 |

La combinación amplia y la final quedan empatadas dentro del ruido. Se eligió la final porque tiene menos de la mitad de variables, lo que hace más simple el servicio y más legible la explicación.

Un efecto práctico: la API necesita el historial desde el primer ciclo del motor, porque ahí se calcula la línea base. Si el historial empieza más tarde, la respuesta incluye un aviso. El código está en `app/features.py`.

### Semanas 3 a 6: familias de modelos y ensamble

Se compararon cinco familias con las variables nuevas, todas dentro de un `Pipeline` con `StandardScaler` para que el escalado se ajuste solo con el entrenamiento de cada pliegue.

| Modelo | RMSE de validación |
|---|---|
| Ridge (regresión lineal regularizada) | 17,38 ± 0,67 |
| SVR con kernel RBF | 14,25 ± 0,78 |
| XGBoost | 13,56 ± 0,66 |
| MLP | 13,47 ± 0,65 |
| Random Forest | 13,35 ± 0,58 |
| XGBoost con hiperparámetros ajustados | 13,27 ± 0,67 |
| MLP con hiperparámetros ajustados | 13,35 ± 0,78 |
| Promedio de XGBoost y MLP ajustados | 12,95 ± 0,74 |

Ridge queda lejos porque la relación entre sensores y RUL no es lineal. SVR rinde peor y tarda más: su validación cruzada tomó unos 80 segundos contra 4 de XGBoost. Random Forest empata con XGBoost, pero entrena mucho más lento y el modelo pesa más, así que no compensa desplegarlo.

El promedio de XGBoost y MLP gana porque los dos se equivocan de maneras distintas, y promediar reduce la varianza (idea de ensamble de la semana 4). La ventaja es de 0,3 a 0,4 ciclos, menor que la desviación entre pliegues, pero es consistente: el ensamble supera a XGBoost solo y al MLP solo en los cinco pliegues, y baja el puntaje NASA de 8 655 a 8 068. Por eso se eligió. El modelo final se selecciona por validación cruzada dentro de `entrenar_modelo.py`, no a mano.

### Semanas 3 y 5: alarma binaria

El semáforo de la versión 1 usaba umbrales fijos sobre la RUL estimada. Se agregó un clasificador que responde directamente «¿falla en 30 ciclos o menos?» y entrega una probabilidad. Se comparó regresión logística, SVM, Random Forest y XGBoost, y también la alarma que sale de simplemente umbralizar la regresión.

| Alarma de falla en ≤ 30 ciclos | AUC | Precisión | Recall | F2 |
|---|---|---|---|---|
| Regresión logística | 0,993 | 0,79 | 0,96 | 0,92 |
| SVM RBF | 0,996 | 0,84 | 0,98 | 0,95 |
| Random Forest | 0,995 | 0,84 | 0,96 | 0,94 |
| XGBoost | 0,996 | 0,85 | 0,97 | 0,94 |
| Umbral sobre la regresión | 0,996 | 0,85 | 0,97 | 0,94 |

El clasificador no discrimina mejor que la regresión: la RUL estimada ya separa bien los casos cercanos a la falla. Lo que aporta es una probabilidad calibrada (Brier de 0,017) y un umbral elegido con un criterio explícito. Se usó el F2, que pesa el recall el doble que la precisión, porque no detectar una falla cuesta más que una falsa alarma. El umbral resultante es 0,08. En los 100 motores de prueba detecta 24 de 25 fallas cercanas, con 2 falsas alarmas. XGBoost se eligió sobre SVM por estar empatado y devolver probabilidades de forma nativa.

### Semana 7: validación, error por zona e interpretabilidad

La validación cruzada por motor ya existía; se completó con lo que el tema pide para conocer la robustez del modelo.

La curva de aprendizaje (`curva_aprendizaje.png`) muestra el RMSE de validación bajando de 15,8 con 20 motores a 13,3 con 80, con una meseta al final. Agregar más motores parecidos ayudaría poco; el modelo se limita más por la brecha entre entrenamiento (7,2) y validación (13,3) que por falta de capacidad.

El error no es parejo a lo largo de la vida del motor:

| RUL real | Motores | MAE | Sesgo medio |
|---|---|---|---|
| 0 a 25 | 19 | 3,7 | +1,0 |
| 26 a 50 | 14 | 5,3 | +2,9 |
| 51 a 75 | 10 | 7,9 | +1,6 |
| 76 a 100 | 24 | 12,9 | +5,5 |
| más de 100 | 33 | 12,7 | −7,1 |

Cerca de la falla el modelo se equivoca poco, que es lo que importa para decidir mantenimiento. Entre 76 y 100 ciclos sobreestima, y por encima de 100 subestima porque el entrenamiento acota la RUL a 125.

El margen que la API resta para la estimación conservadora (el percentil 90 de la sobreestimación) ahora se calcula con la validación cruzada: 18,9 ciclos. La versión 1 usaba una constante fija (25,5) escrita en el código. En la prueba, el percentil 90 de la sobreestimación es 14,2, así que el margen de validación es conservador.

Para la interpretabilidad se calcularon valores SHAP exactos del componente XGBoost (`shap_global.png`). La variable que más pesa, con diferencia, es el cambio del sensor 11 respecto a su línea base (contribución media de 10 ciclos), seguida del nivel suavizado del sensor 4 y del cambio del sensor 12. Esto concuerda con la semana 2: la mejora principal vino de las variables relativas a la línea base. Las contribuciones se obtienen con `pred_contribs` de XGBoost, sin instalar la librería `shap`, que añadiría dependencias pesadas al contenedor.

LIME se comparó con SHAP sobre tres motores (cerca de la falla, a mitad de vida y sano). Coincidieron en 3, 1 y 2 de las 6 variables principales. LIME ajusta un modelo lineal local con muestras aleatorias, y con variables muy correlacionadas como estas el resultado cambia según la muestra. SHAP en árboles es exacto y determinista, y se verifica con una propiedad que la prueba automática comprueba: valor base más contribuciones es igual a la predicción. Por eso la API usa SHAP.

Un límite de la explicación: corresponde al componente XGBoost, que aporta la mitad del promedio. La interfaz lo indica.

## Lo que se probó y no entró

Estas técnicas no mejoraron la predicción en este problema. Se documentan porque el sílabo las pide y porque el resultado negativo es información.

**Clustering (semana 9).** Sobre las variables `rel0` estandarizadas, K-Means con k = 3 o 4 agrupa los ciclos de forma que coincide con la etapa de vida del motor (información mutua ajustada de 0,45), y la silueta máxima aparece con k = 2 (0,46), que separa «sano» de «degradado». El jerárquico de Ward con k = 4 es menos fiel a las etapas (0,37). DBSCAN no sirve aquí: con cuatro valores de `eps`, o devuelve un solo grupo y mucho ruido, o un solo grupo sin ruido. Los datos son una trayectoria continua sin huecos de densidad. Las distancias a los centroides de K-Means como variables nuevas bajaron el RMSE de validación de 13,56 a 13,38 y 13,35, dentro del ruido, y empeoraron la prueba (12,65 a 12,95 y 13,06). No se incorporaron.

**PCA y t-SNE (semana 10).** Se necesitan 24 de 44 componentes para el 95 % de la varianza. El primer componente de los sensores normalizados funciona como índice de salud que crece hacia la falla, y t-SNE dibuja la degradación como una trayectoria continua (figura `s10_pca_tsne.png`), útiles para visualizar. Como paso previo a los modelos no ayudan: Ridge queda igual (17,38), el MLP pasa de 13,47 a 13,30 (dentro del ruido) y XGBoost empeora de 13,56 a 14,15. Los árboles no necesitan descorrelacionar variables, y PCA descarta señal útil.

**CNN 1D (semana 11).** Con ventanas de 30 ciclos de los 14 sensores normalizados por línea base, la red alcanzó un RMSE de validación de 18,3 ± 2,6 y de 18,6 en la prueba (promedio de cinco semillas), peor que el MLP sobre variables construidas a mano (13,5). La red se sobreajusta, y con solo 80 motores de entrenamiento las ventanas de un mismo motor son casi copias entre sí, así que la red tiende a memorizar el motor en lugar de aprender la degradación (es la explicación más probable; no se probó de forma aislada). Las variables `ewm30` y `slope30` hacen a mano el suavizado que la red tendría que aprender. Varias variantes (otras ventanas y más regularización) tampoco la acercaron. Con más motores de entrenamiento (FD002 y FD004 tienen unos 250) podría cambiar la conclusión. Para desplegarla además haría falta TensorFlow en el contenedor, que es mucho más pesado que el modelo actual.

## Cambios en la aplicación

La API y la página incorporan dos funciones nuevas. `alarm.prob` entrega la probabilidad de falla en 30 ciclos o menos en cada ciclo, y se muestra como una tarjeta junto al estado. `/api/explain` y `/api/demo/engines/{unit}/explain` devuelven las 8 variables que más pesan en una predicción, y la página las dibuja como barras en el panel «¿Por qué esta estimación?», que se actualiza al mover el control de ciclo. Además, `/api/info` informa el margen conservador, el umbral de alarma y las métricas de prueba, para que la página no tenga cifras escritas a mano.

## Viabilidad y mantenimiento

La lógica de variables vivía duplicada en el entrenamiento y en la API, con una nota que advertía que si una cambiaba había que cambiar la otra a mano. Ahora hay una sola fuente, `app/features.py`, y el entrenamiento la importa. Las 13 pruebas de `tests/` cubren que las variables sean causales, que la línea base use solo los primeros ciclos, que el cálculo por lotes coincida con el cálculo motor por motor, y que la API devuelva respuestas consistentes, incluida la propiedad aditiva de SHAP.

El modelo guardado pesa alrededor de 1 MB y no agrega dependencias a `requirements.txt`. El `Dockerfile` y `render.yaml` no necesitan cambios. Las dependencias de entrenamiento, pruebas y experimentos están aparte, en `requirements-train.txt`.

## Límites

- Los datos son simulados y de un solo modo de falla (FD001). Que la técnica funcione aquí no garantiza que funcione en motores reales.
- La línea base por motor exige historial desde el primer ciclo.
- La prueba tiene 100 motores, y por eso los intervalos de confianza son anchos. Una diferencia de 0,3 ciclos entre dos modelos no se distingue de la suerte en la prueba; en validación cruzada con 20 631 filas sí se puede evaluar.
- La RUL se acota a 125 en el entrenamiento, por lo que el modelo no distingue entre 130 y 145 ciclos restantes.
- El modelo guardado se entrenó con Python 3.13 y las versiones de scikit-learn y XGBoost de `requirements.txt`. El `Dockerfile` usa Python 3.12; si al desplegar aparece un error al cargar el archivo, basta con ejecutar `python entrenar_modelo.py --sin-busqueda` y subir el nuevo modelo.

## Archivos y cómo reproducir

Archivos nuevos: `entrenar_modelo.py`, `experimentos_semanas.py`, `ml_utils.py`, `app/explain.py`, `requirements-train.txt`, `tests/test_features.py`, `tests/test_api.py` y este documento con sus figuras. Archivos modificados: `app/features.py`, `app/main.py`, `app/static/index.html`, `app/static/app.js`, `app/static/style.css`, `README.md` y `models/modelo_rul_fd001.joblib`. El notebook original `rul_cmapss_fd001.py` queda intacto como línea base del informe.

```bash
pip install -r requirements-train.txt
python entrenar_modelo.py --sin-busqueda   # regenera el modelo, outputs/ y las figuras
python experimentos_semanas.py --semana todas
python -m pytest -q tests
uvicorn app.main:app --reload
```

Una agrupación posible de commits: primero `app/features.py`, `ml_utils.py` y `tests/test_features.py` (variables nuevas); luego `entrenar_modelo.py`, `app/explain.py` y el modelo (entrenamiento); después `app/main.py` y `app/static/` (API y página); y al final `experimentos_semanas.py`, `docs/`, `README.md` y `requirements-train.txt`. El modelo guardado y `app/features.py` deben ir en el mismo commit: con uno de los dos viejos, el servicio falla al arrancar.
