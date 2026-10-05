import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Subtask s1: carga del dataset y división 70/30 estratificada
# ------------------------------------------------------------------

# 1) Cargar el conjunto Breast Cancer Wisconsin (Diagnostic)
data = load_breast_cancer()
X, y = data.data, data.target
class_names = [str(n) for n in data.target_names]  # ['malignant', 'benign']

n_total = int(X.shape[0])
n_attributes = int(X.shape[1])

# Casos por clase en el dataset completo
class_counts_full = {
    class_names[i]: int(np.sum(y == i)) for i in range(len(class_names))
}
class_props_full = {
    k: float(v / n_total) for k, v in class_counts_full.items()
}

# 2) División 70/30 estratificada por la clase, random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

n_train = int(X_train.shape[0])
n_test = int(X_test.shape[0])

# Casos por clase en cada partición (verificación de la estratificación)
class_counts_train = {
    class_names[i]: int(np.sum(y_train == i)) for i in range(len(class_names))
}
class_counts_test = {
    class_names[i]: int(np.sum(y_test == i)) for i in range(len(class_names))
}
class_props_train = {
    k: float(v / n_train) for k, v in class_counts_train.items()
}
class_props_test = {
    k: float(v / n_test) for k, v in class_counts_test.items()
}

# 3) Guardar todos los números en resultados.json
results = {
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "n_total_examples": n_total,
    "n_attributes": n_attributes,
    "class_counts_full": class_counts_full,
    "class_proportions_full": class_props_full,
    "split_params": {
        "test_size": 0.3,
        "stratify": "y",
        "random_state": 42,
    },
    "n_train": n_train,
    "n_test": n_test,
    "class_counts_train": class_counts_train,
    "class_proportions_train": class_props_train,
    "class_counts_test": class_counts_test,
    "class_proportions_test": class_props_test,
}

with open("resultados.json", "w") as f:
    json.dump(results, f, indent=2)

# 4) Imprimir los mismos números en stdout
print("=== Subtask s1: Breast Cancer Wisconsin (Diagnostic) ===")
print(f"Total de ejemplos: {n_total}")
print(f"Numero de atributos: {n_attributes}")
print("Casos por clase (dataset completo):")
for k in class_names:
    print(f"  {k}: {class_counts_full[k]} ({class_props_full[k]*100:.2f}%)")
print("Division: test_size=0.3, stratify=y, random_state=42")
print(f"Ejemplos en entrenamiento: {n_train}")
print(f"Ejemplos en prueba: {n_test}")
print("Casos por clase (entrenamiento):")
for k in class_names:
    print(f"  {k}: {class_counts_train[k]} ({class_props_train[k]*100:.2f}%)")
print("Casos por clase (prueba):")
for k in class_names:
    print(f"  {k}: {class_counts_test[k]} ({class_props_test[k]*100:.2f}%)")
print("\nContenido de resultados.json:")
print(json.dumps(results, indent=2))
