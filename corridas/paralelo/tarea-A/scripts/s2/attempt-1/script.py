# Subtask s2 — Parte 2: dos clasificadores (GaussianNB vs LogisticRegression)
# Recomputa la división de la Parte 1 (test_size=0.3, stratify=y, random_state=42),
# estandariza con StandardScaler ajustado SOLO con el entrenamiento, entrena ambos
# modelos sobre los datos estandarizados y evalúa ambos sobre el mismo conjunto
# de prueba (nunca usado para ajustar nada). Guarda resultados en resultados.json.

import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

TEST_SIZE = 0.3
RANDOM_STATE = 42

# ---------- Parte 1 (recomputada): carga y división ----------
data = load_breast_cancer()
X, y = data.data, data.target
target_names = [str(t) for t in data.target_names]  # ['malignant', 'benign']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# ---------- Estandarización: el escalador se ajusta únicamente con train ----------
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)   # fit + transform en train
X_test_std = scaler.transform(X_test)         # solo transform en test

# ---------- Parte 2: entrenamiento de los dos clasificadores ----------
models = {
    "GaussianNB": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
}

results = {
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "split_params": {
        "test_size": TEST_SIZE,
        "stratify": "y",
        "random_state": RANDOM_STATE,
    },
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "class_counts_train": {
        target_names[0]: int(np.sum(y_train == 0)),
        target_names[1]: int(np.sum(y_train == 1)),
    },
    "class_counts_test": {
        target_names[0]: int(np.sum(y_test == 0)),
        target_names[1]: int(np.sum(y_test == 1)),
    },
    "scaler": {
        "fit_on": "train_only",
        "train_mean_first3": [float(v) for v in scaler.mean_[:3]],
        "train_std_first3": [float(v) for v in scaler.scale_[:3]],
    },
    "models": {},
    "tabla_parte2": [],
}

# Entrenar en train estandarizado, evaluar en test estandarizado (mismo test para ambos)
for name, model in models.items():
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred)
    results["models"][name] = {
        "accuracy_test": acc,
        "f1_macro_test": f1m,
        "confusion_matrix_test": cm.tolist(),
    }
    results["tabla_parte2"].append(
        {"modelo": name, "accuracy_test": acc, "f1_macro_test": f1m}
    )

# ---------- Tabla en stdout ----------
header = f"{'Modelo':<20}{'Accuracy (test)':>17}{'F1 macro (test)':>18}"
line = "-" * len(header)
print("Parte 2 — Resultados sobre el conjunto de prueba "
      f"(n_train={results['n_train']}, n_test={results['n_test']})")
print(header)
print(line)
for row in results["tabla_parte2"]:
    print(f"{row['modelo']:<20}{row['accuracy_test']:>17.4f}{row['f1_macro_test']:>18.4f}")
print(line)

# ---------- Guardar todos los números calculados ----------
with open("resultados.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nContenido de resultados.json:")
print(json.dumps(results, indent=2))
