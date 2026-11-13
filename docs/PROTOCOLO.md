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
