# Protocolo de evaluación

Escrito **antes** de correr cualquier evaluación nueva. Si un criterio cambia, se corrige aquí con un commit
propio y se explica el motivo; no se ajusta en silencio después de ver resultados.

## 1. Qué se vio y qué no

- **Entrenamiento FD001**: se usa para todo (variables, modelos, hiperparámetros, margen, umbral de alarma),
  siempre con validación cruzada por motor (`GroupKFold`, 5 pliegues).
- **Prueba FD001**: **ya se vio** durante la exploración inicial (versiones anteriores de `entrenar_modelo.py` y
  `experimentos_semanas.py` imprimían métricas sobre ella). No se puede "des-mirar". Sus cifras se reportan como
  referencia, no como evidencia confirmatoria.
- **FD003** (otro modo de falla, mismo tipo de motor): **no se ha usado para decidir nada**. Es el conjunto
  confirmatorio. Se evalúa **una sola vez**, con el modelo congelado.
- FD002 y FD004 quedan fuera por ahora (6 condiciones de operación; requieren otro preprocesamiento).

## 2. Reglas de separación

1. `entrenar_modelo.py` y `experimentos_semanas.py` **no leen archivos de prueba** (usan `load_train`).
2. Solo `evaluar_final.py` lee prueba (`load_test`). Se niega a correr con cambios sin commitear en `models/`,
   `app/` o `ml_utils.py`, y registra el hash del commit evaluado.
3. Cada subconjunto se evalúa una vez; el script se niega a repetirlo salvo `--repetir`, que queda registrado.
4. Ninguna cifra de prueba se usa para elegir, ajustar ni calibrar nada.

## 3. Qué cuenta como mejora (se decide solo con validación cruzada)

Métrica: RMSE por motor en la zona de interés (RUL real ≤ 125), media de los 5 pliegues, como en `cv_report`.
Referencia: el modelo vigente, con su desviación entre pliegues (≈ 0,65 ciclos).

- **Mejora real**: el RMSE de CV baja **más que la desviación entre pliegues** y el intervalo bootstrap pareado
  (remuestreando los mismos motores para ambos modelos) de la diferencia **no incluye 0**.
- **No se afirma** una mejora menor a **0,5 ciclos**, aunque el número baje.
- Toda cifra se reporta con su intervalo, en el README, el informe y las diapositivas.

Criterios por cambio:

| Cambio | Se adopta si… |
|---|---|
| Otro valor de recorte (`CAP`: 100, 110, 125, 140, 150) | se evalúa siempre contra `min(RUL real, CAP)` del propio valor **y** contra la RUL real ≤ 125 común; solo se cambia si mejora el RMSE contra la RUL real ≤ 125 según la regla de mejora real |
| Calibración isotónica (ajustada con predicciones fuera de pliegue) | el sesgo en la zona 76–100 baja y el RMSE **no empeora** más que la desviación entre pliegues; la calibración se valida con validación cruzada anidada, no con los mismos datos con que se ajusta |
| Cota por regresión cuantílica (α = 0,1) | a **igual cobertura** (≈ 90 % de los ciclos con RUL real ≥ cota, medida fuera de pliegue) la cota queda más cerca de la predicción que el margen fijo, o tiene mejor cobertura con la misma distancia |
| Mostrar "≥ 125 ciclos" cuando la predicción está saturada | sin criterio estadístico; es un cambio de comunicación |

## 4. Evaluación confirmatoria

1. Se decide el modelo final solo con validación cruzada y se **commitea** (`models/` + `app/features.py`).
2. Se ejecuta `python evaluar_final.py --subset FD003` **una vez**.
3. Se reporta tal cual, sin ajustar nada después: RMSE, MAE y puntaje NASA contra la RUL real y contra la RUL
   acotada, con IC 95 % por bootstrap sobre motores, sesgo por zona y cobertura del margen conservador.
4. No hay umbral de aprobado/reprobado. Lo que se espera y se declara de antemano: error mayor que en la
   validación de FD001, porque FD003 trae un segundo modo de falla que el modelo no vio. Cuánto mayor es,
   justamente, la medida de fragilidad.
5. En el informe se declara que la prueba de FD001 se vio durante la exploración y que las cifras
   confirmatorias son las del conjunto virgen.

## 5. Límite que no se corrige aquí

Los datos son simulados con un solo modo de falla en entrenamiento. Esta herramienta apoya la decisión de
mantenimiento; no es un sistema para motores reales.

## 6. Resultado de la evaluación confirmatoria (commit `aa11773`)

Ejecutada una sola vez, sin ajustes posteriores. Detalle en `docs/resultados/`.

| Conjunto | Motores | RMSE (IC 95 %) | MAE | Sesgo en 0–25 ciclos |
|---|---|---|---|---|
| FD003, virgen (otro modo de falla) | 100 | 37,0 (30,7 – 43,0) | 26,0 | **+53,6** |
| FD001, prueba ya vista (referencia) | 100 | 14,1 (11,7 – 16,4) | 9,9 | +0,6 |

Lectura: el modelo **no transfiere** a un segundo modo de falla. Cerca de la falla sobreestima la vida restante
en más de 50 ciclos, que es el error más peligroso para mantenimiento. El margen conservador (P90 de FD001) solo
cubrió el 57 % de los motores de FD003, frente al 97 % en FD001. Las cifras de FD001 no son confirmatorias.

### Diagnóstico posterior (solo con `train_FD003`, sin tocar su prueba)

`experimentos_fd003.py`, resultados en `docs/resultados/diagnostico_fd003.csv`:

| Caso | RMSE (RUL ≤ 125) | Sesgo en 0–25 ciclos |
|---|---|---|
| A) modelo FD001 congelado aplicado a `train_FD003` | 35,2 | +37,4 |
| B) entrenar y validar con GroupKFold dentro de FD003 | 11,7 ± 1,0 | +1,0 |

Las variables y las 14 columnas **no** son el problema: con datos de FD003 el error es el mismo que en FD001
(11,7 frente a 11,84). El fallo es de **transferencia entre modos de falla**: un modelo entrenado con un solo
modo no reconoce el otro.

### Aclaración sobre la cifra de FD001

El RMSE de 14,1 en los 100 motores de prueba de FD001 no contradice la mejora en validación. Se reconstruyó el
modelo anterior (CAP = 125; el archivo guardado en git no se puede cargar) y se compararon con la misma definición:

| | RMSE, 100 motores vs RUL real | RMSE, solo RUL ≤ 125 |
|---|---|---|
| CAP = 125 (reconstruido; el original publicó 12,79) | 12,83 | 11,83 |
| CAP = 110 (vigente) | 14,07 | 10,90 |

La diferencia en la zona de elección es −0,92 ciclos con IC 95 % [−2,03; +0,15]: va en el sentido de la
validación, pero en prueba el intervalo incluye 0. En los 11 motores sanos (RUL > 125) el recorte a 110 subestima
más (sesgo −28 frente a −18), lo que no afecta la decisión de mantenimiento pero sube el RMSE global. Esta
comparación usó la prueba de FD001, ya vista, y no se usó para elegir nada.

## 7. Fase 2: entrenar con varios conjuntos (escrita antes de correr el experimento)

Motivo: el modelo de FD001 no transfiere a FD003 (secciones 4 a 6). Se prueba entrenar con ambos modos de falla.

**Estado de las pruebas.** Las de FD001 y FD003 ya se usaron para evaluar un modelo congelado: quedan «gastadas» y
pueden usarse como datos etiquetados (RUL en cada ciclo = RUL final + ciclos que faltaban). Las de **FD002 y FD004
siguen reservadas**: `load_test_labeled` se niega a leerlas y solo `evaluar_final.py` puede hacerlo.

**Comparación (validación cruzada por motor repetida, 3 semillas × 5 pliegues, estratificada por conjunto, zona RUL ≤ 125):**

- M0: entrena solo con `train_FD001` (el modelo vigente).
- M1: entrena con `train_FD001` + `train_FD003`.
- M2: M1 más los motores de prueba etiquetados de FD001 y FD003 (solo como entrenamiento; la evaluación se hace
  únicamente sobre motores de entrenamiento, que tienen la vida completa).

**Se adopta un modelo con varios conjuntos si cumple las cuatro:**

1. En los motores de FD001, el RMSE no empeora más de 0,5 ciclos frente a M0 (límite superior del IC 95 % del
   bootstrap pareado de la diferencia < +0,5).
2. En los motores de FD003, el RMSE es ≤ 13,0 y el sesgo con 0–25 ciclos restantes está entre −5 y +5.
3. El margen P90 cubre al menos el 85 % de los ciclos en cada conjunto.
4. Entre M1 y M2 se prefiere M1; M2 solo si mejora el RMSE en ambos conjuntos más que la desviación entre pliegues
   y con IC pareado que excluya 0.

Se reportan los intervalos por conjunto y sobre el total. No se afirma una mejora menor a 0,5 ciclos.

**Evaluación confirmatoria.** Un modelo de régimen único (FD001 + FD003) **no puede evaluarse sin sesgo en FD002 ni
FD004**: esos conjuntos tienen 6 regímenes de operación y exigen normalizar por régimen. Por eso la reserva virgen
solo se usa con un modelo con preprocesamiento por régimen (Fase 3), entrenado con los `train_*` y evaluado una sola
vez en `test_FD002` y `test_FD004`. Para el modelo de régimen único de esta fase la evidencia es la validación
cruzada repetida; no habrá cifra confirmatoria nueva.

## 8. Fase 3: normalización por régimen (escrita antes de correr)

FD002 y FD004 tienen 6 condiciones de operación. Sin normalizar, los sensores casi no se relacionan con la vida
restante (correlación ≈ 0); normalizando por régimen sube a ≈ −0,6 a −0,8. Se arma un modelo aparte, de
investigación (la API desplegada asume un solo régimen y no lo usa).

1. **Validación cruzada** por motor sobre los cuatro `train_FD00X` juntos (709 motores, 5 pliegues, estratificada por
   conjunto, normalizador reajustado dentro de cada pliegue). Mismas métricas y zona (RUL ≤ 125) de siempre.
2. **Criterio para congelar y evaluar:** en CV, FD002 y FD004 con RMSE ≤ 20 ciclos y |sesgo con 0–25 ciclos| ≤ 10, y
   FD001 y FD003 sin empeorar más de 0,5 ciclos frente a la fase 2. Si no se cumple, **no se evalúa en prueba**, no se
   ajusta nada mirando la prueba, y el resultado queda documentado como límite.
3. **Evaluación confirmatoria:** si se cumple, se commitea el modelo y se ejecuta `evaluar_final.py` una sola vez sobre
   `test_FD002` y `test_FD004` (vírgenes). FD001 y FD003 se reportan solo como referencia (ya gastadas).
4. Sin umbral de aprobado/reprobado en la evaluación; se espera un error mayor que en FD001, porque FD004 combina
   regímenes y dos modos de falla.

## 9. Resultados de las fases 2 y 3

**Fase 2 (validación cruzada repetida, 3 semillas × 5 pliegues; `docs/resultados/fase2_*.csv`).** Se cumplen los cuatro
criterios de la sección 7 y se adopta M1 (entrenar con FD001 + FD003):

| | RMSE en motores FD001 | RMSE en motores FD003 |
|---|---|---|
| M0, solo FD001 | 11,66 | 42,18 |
| M1, FD001 + FD003 | 11,41 | 11,07 |
| M2, M1 + motores de prueba etiquetados | 11,45 | 10,77 |

M1 − M0 en FD001: −0,25 [−0,44; −0,07] (no se afirma como mejora, es < 0,5). M2 no supera la regla de adopción.
Modelo desplegado: `models/modelo_rul.joblib`, ensamble XGBoost + MLP entrenado con los 200 motores, recorte 110,
margen P90 de 13,9 ciclos. Evaluaciones de referencia sobre pruebas ya gastadas (commit `3ac4481`):
FD001 RMSE 14,05 y FD003 RMSE 15,50 contra la RUL real (en FD003 antes era 37,04).

**Fase 3 (normalización por régimen).** En validación cruzada se cumple el criterio de la sección 8 (FD002 13,4 y
FD004 13,6 de RMSE; sesgo con 0–25 ciclos de +1,6 y +2,4). Se congeló el modelo (commit `9783cd7`) y se evaluó **una
sola vez** en las pruebas reservadas (`docs/resultados/final_regimenes_*`):

| Conjunto (virgen) | Motores | RMSE con RUL real ≤ 125 | MAE | RMSE contra RUL real, todos | IC 95 % (todos) | Cobertura margen P90 |
|---|---|---|---|---|---|---|
| FD002 | 259 (202 con RUL ≤ 125) | 11,49 | 8,48 | 28,37 | 24,7 a 31,8 | 92,6 % |
| FD004 | 248 (181 con RUL ≤ 125) | 12,67 | 9,45 | 30,27 | 27,1 a 33,4 | 92,8 % |

Lectura: con normalización por régimen, el modelo generaliza a FD002 y FD004 en la zona que importa, con error
parecido al de validación (13,4 y 13,6) y un margen conservador que cubre más del 90 %. El RMSE contra la RUL real
sin recortar es mucho mayor porque 57 y 67 motores de prueba están sanos, con RUL real de hasta casi 200 ciclos, y el
modelo, por diseño, no distingue más allá de 110 (sesgo −53 y −51 en esos motores; no afecta la decisión de
mantenimiento). Las cifras confirmatorias son estas; las de FD001 y FD003 son solo referencia.

Este modelo (`models/modelo_rul_regimenes.joblib`) es de investigación: la API desplegada usa un solo régimen.
Llevarlo a la API exige guardar y aplicar el normalizador por régimen en el servicio (trabajo pendiente).
