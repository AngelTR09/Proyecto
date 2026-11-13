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
