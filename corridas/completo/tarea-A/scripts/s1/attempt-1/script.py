# Subtask s1 — Parte 1: carga del dataset y división 70/30 estratificada
# Breast Cancer Wisconsin (Diagnostic) via sklearn.datasets.load_breast_cancer

import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def to_plain(obj):
    """Convierte tipos de numpy a tipos nativos de Python para JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


# ------------------------------------------------------------------
# 1. Carga del conjunto de datos
# ------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
target_names = list(data.target_names)  # ['malignant', 'benign']

n_total_examples = int(X.shape[0])
n_attributes = int(X.shape[1])

classes = np.unique(y)
class_counts_full = {
    str(target_names[int(c)]): int(np.sum(y == c)) for c in classes
}
class_fractions_full = {
    str(target_names[int(c)]): float(np.sum(y == c) / n_total_examples)
    for c in classes
}

# ------------------------------------------------------------------
# 2. División 70 % / 30 % estratificada por la clase, random_state=42
#    (esta es la división única que se reutiliza en toda la tarea)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

train_size = int(X_train.shape[0])
test_size = int(X_test.shape[0])

class_counts_train = {
    str(target_names[int(c)]): int(np.sum(y_train == c)) for c in classes
}
class_fractions_train = {
    str(target_names[int(c)]): float(np.sum(y_train == c) / train_size)
    for c in classes
}

class_counts_test = {
    str(target_names[int(c)]): int(np.sum(y_test == c)) for c in classes
}
class_fractions_test = {
    str(target_names[int(c)]): float(np.sum(y_test == c) / test_size)
    for c in classes
}

# ------------------------------------------------------------------
# 3. Resultados
# ------------------------------------------------------------------
results = {
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "loader": "sklearn.datasets.load_breast_cancer",
    "n_total_examples": n_total_examples,
    "n_attributes": n_attributes,
    "class_counts_full": class_counts_full,
    "class_fractions_full": class_fractions_full,
    "split_definition": {
        "test_size": 0.3,
        "train_size_fraction": 0.7,
        "stratify": "y",
        "random_state": 42,
    },
    "train_set": {
        "n_examples": train_size,
        "class_counts": class_counts_train,
        "class_fractions": class_fractions_train,
    },
    "test_set": {
        "n_examples": test_size,
        "class_counts": class_counts_test,
        "class_fractions": class_fractions_test,
    },
}

results = to_plain(results)

# Guardar en el archivo de resultados
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Imprimir los mismos números en stdout
print("=== Parte 1: Datos — Breast Cancer Wisconsin (Diagnostic) ===")
print(f"Numero total de ejemplos: {n_total_examples}")
print(f"Numero de atributos:      {n_attributes}")
print("Casos por clase (dataset completo):")
for name in target_names:
    print(
        f"  {name}: {class_counts_full[name]} "
        f"({class_fractions_full[name] * 100:.2f} %)"
    )
print()
print("Division: test_size=0.3, stratify=y, random_state=42")
print(f"Ejemplos de entrenamiento: {train_size} "
      f"({train_size / n_total_examples * 100:.2f} % del total)")
print(f"Ejemplos de prueba:        {test_size} "
      f"({test_size / n_total_examples * 100:.2f} % del total)")
print("Composicion por clase — entrenamiento:")
for name in target_names:
    print(
        f"  {name}: {class_counts_train[name]} "
        f"({class_fractions_train[name] * 100:.2f} %)"
    )
print("Composicion por clase — prueba:")
for name in target_names:
    print(
        f"  {name}: {class_counts_test[name]} "
        f"({class_fractions_test[name] * 100:.2f} %)"
    )
print()
print("Resultados escritos en resultados.json")
