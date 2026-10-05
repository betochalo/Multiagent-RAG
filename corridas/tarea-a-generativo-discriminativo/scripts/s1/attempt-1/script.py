# Subtask s1 — Parte 1: Carga del conjunto Breast Cancer Wisconsin (Diagnostic)
# y división estratificada 70/30 con random_state=42.

import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 1. Carga del conjunto de datos (disponible localmente en sklearn)
# ------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target

# En sklearn: target 0 = 'malignant', 1 = 'benign'
n_total = int(X.shape[0])        # número total de ejemplos
n_atributos = int(X.shape[1])    # número de atributos

n_malignos = int(np.sum(y == 0))
n_benignos = int(np.sum(y == 1))

pct_malignos_total = 100.0 * n_malignos / n_total
pct_benignos_total = 100.0 * n_benignos / n_total

# ------------------------------------------------------------------
# 2. División entrenamiento/prueba: 70% / 30%, estratificada, semilla 42
#    (esta es la única división usada en toda la tarea)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

n_train = int(X_train.shape[0])
n_test = int(X_test.shape[0])

train_malignos = int(np.sum(y_train == 0))
train_benignos = int(np.sum(y_train == 1))
test_malignos = int(np.sum(y_test == 0))
test_benignos = int(np.sum(y_test == 1))

pct_train_malignos = 100.0 * train_malignos / n_train
pct_train_benignos = 100.0 * train_benignos / n_train
pct_test_malignos = 100.0 * test_malignos / n_test
pct_test_benignos = 100.0 * test_benignos / n_test

# ------------------------------------------------------------------
# 3. Verificación de la estratificación (proporciones similares)
# ------------------------------------------------------------------
diff_malignos = abs(pct_train_malignos - pct_test_malignos)
diff_benignos = abs(pct_train_benignos - pct_test_benignos)

# ------------------------------------------------------------------
# 4. Guardar todos los números en resultados.json
# ------------------------------------------------------------------
resultados = {
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "n_ejemplos_total": n_total,
    "n_atributos": n_atributos,
    "clases": {
        "malignos": n_malignos,
        "benignos": n_benignos,
        "pct_malignos": float(pct_malignos_total),
        "pct_benignos": float(pct_benignos_total),
    },
    "parametros_split": {
        "test_size": 0.30,
        "stratify": "y",
        "random_state": 42,
    },
    "n_entrenamiento": n_train,
    "n_prueba": n_test,
    "distribucion_entrenamiento": {
        "malignos": train_malignos,
        "benignos": train_benignos,
        "pct_malignos": float(pct_train_malignos),
        "pct_benignos": float(pct_train_benignos),
    },
    "distribucion_prueba": {
        "malignos": test_malignos,
        "benignos": test_benignos,
        "pct_malignos": float(pct_test_malignos),
        "pct_benignos": float(pct_test_benignos),
    },
    "verificacion_estratificacion": {
        "diff_pct_malignos_train_test": float(diff_malignos),
        "diff_pct_benignos_train_test": float(diff_benignos),
        "estratificacion_ok": bool(diff_malignos < 1.0 and diff_benignos < 1.0),
    },
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# ------------------------------------------------------------------
# 5. Imprimir todas las cifras
# ------------------------------------------------------------------
print("=" * 64)
print("Parte 1 — Datos: Breast Cancer Wisconsin (Diagnostic)")
print("=" * 64)
print(f"Numero total de ejemplos : {n_total}")
print(f"Numero de atributos      : {n_atributos}")
print("-" * 64)
print("Casos por clase (conjunto completo):")
print(f"  Malignos (0): {n_malignos}  ({pct_malignos_total:.2f} %)")
print(f"  Benignos (1): {n_benignos}  ({pct_benignos_total:.2f} %)")
print("-" * 64)
print("Division train_test_split(test_size=0.30, stratify=y, random_state=42):")
print(f"  Tamano entrenamiento: {n_train}  ({100.0 * n_train / n_total:.2f} %)")
print(f"  Tamano prueba       : {n_test}  ({100.0 * n_test / n_total:.2f} %)")
print("-" * 64)
print("Distribucion de clases en ENTRENAMIENTO:")
print(f"  Malignos: {train_malignos}  ({pct_train_malignos:.2f} %)")
print(f"  Benignos: {train_benignos}  ({pct_train_benignos:.2f} %)")
print("Distribucion de clases en PRUEBA:")
print(f"  Malignos: {test_malignos}  ({pct_test_malignos:.2f} %)")
print(f"  Benignos: {test_benignos}  ({pct_test_benignos:.2f} %)")
print("-" * 64)
print("Verificacion de la estratificacion:")
print(f"  Diferencia pct malignos (train vs test): {diff_malignos:.4f} pp")
print(f"  Diferencia pct benignos (train vs test): {diff_benignos:.4f} pp")
print(f"  Estratificacion correcta: {resultados['verificacion_estratificacion']['estratificacion_ok']}")
print("=" * 64)
print("Resultados guardados en resultados.json")
