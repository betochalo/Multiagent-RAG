# -*- coding: utf-8 -*-
"""
Subtask s3 (Parte 3) — Curva de aprendizaje: generativo vs. discriminativo.

Recomputa la división de la Parte 1 (load_breast_cancer, test_size=0.3,
stratify=y, random_state=42). Para cada fracción de entrenamiento en
[0.05, 0.10, 0.25, 0.50, 1.00] toma una submuestra estratificada del train
(train_test_split sobre el train, stratify=y_train, random_state=42),
estandariza con un StandardScaler ajustado SOLO con esa submuestra, entrena
GaussianNB y LogisticRegression(max_iter=1000) y evalúa la exactitud de ambos
SIEMPRE sobre el mismo conjunto de prueba fijo.
Guarda la tabla en resultados.json y la figura en curva_aprendizaje.png.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
TEST_SIZE = 0.3
FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]

# ----------------------------------------------------------------------
# Parte 1 (recomputada): única división 70/30 estratificada
# ----------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

n_train_total = int(X_train.shape[0])
n_test = int(X_test.shape[0])

# ----------------------------------------------------------------------
# Parte 3: submuestras estratificadas del train + curva de aprendizaje
# ----------------------------------------------------------------------
rows = []
for frac in FRACTIONS:
    if frac >= 1.0:
        # 100 %: la submuestra estratificada es todo el conjunto de entrenamiento
        # (train_test_split no admite train_size=1.0, el complemento sería vacío)
        X_sub, y_sub = X_train, y_train
    else:
        X_sub, _xr, y_sub, _yr = train_test_split(
            X_train, y_train,
            train_size=frac,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )

    # Escalador ajustado SOLO con la submuestra de entrenamiento
    scaler = StandardScaler().fit(X_sub)
    X_sub_std = scaler.transform(X_sub)
    X_test_std = scaler.transform(X_test)  # mismo conjunto de prueba fijo

    nb = GaussianNB().fit(X_sub_std, y_sub)
    lr = LogisticRegression(max_iter=1000).fit(X_sub_std, y_sub)

    acc_nb = float(accuracy_score(y_test, nb.predict(X_test_std)))
    acc_lr = float(accuracy_score(y_test, lr.predict(X_test_std)))

    rows.append({
        "fraction": float(frac),
        "n_train": int(X_sub.shape[0]),
        "accuracy_gaussian_nb": acc_nb,
        "accuracy_logistic_regression": acc_lr,
    })

# ----------------------------------------------------------------------
# Tabla de exactitudes
# ----------------------------------------------------------------------
print("División Parte 1 recomputada: train={} ejemplos, test={} ejemplos "
      "(test_size=0.3, stratify=y, random_state=42)".format(n_train_total, n_test))
print()
header = "{:>10s} {:>12s} {:>18s} {:>22s}".format(
    "Fraccion", "n_train", "Accuracy NB", "Accuracy LR")
print(header)
print("-" * len(header))
for r in rows:
    print("{:>10.2f} {:>12d} {:>18.4f} {:>22.4f}".format(
        r["fraction"], r["n_train"],
        r["accuracy_gaussian_nb"], r["accuracy_logistic_regression"]))

# ----------------------------------------------------------------------
# Figura: exactitud vs. número de ejemplos de entrenamiento
# ----------------------------------------------------------------------
n_values = [r["n_train"] for r in rows]

plt.figure(figsize=(8, 5))
plt.plot(n_values, [r["accuracy_gaussian_nb"] for r in rows],
         marker="o", color="tab:blue",
         label="Naive Bayes gaussiano (generativo)")
plt.plot(n_values, [r["accuracy_logistic_regression"] for r in rows],
         marker="s", color="tab:orange",
         label="Regresión logística (discriminativo)")
plt.xlabel("Número de ejemplos de entrenamiento")
plt.ylabel("Exactitud en el conjunto de prueba")
plt.title("Curva de aprendizaje: NB gaussiano vs. regresión logística\n"
          "(evaluación sobre el mismo conjunto de prueba fijo)")
plt.xticks(n_values)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("curva_aprendizaje.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# resultados.json
# ----------------------------------------------------------------------
results = {
    "subtask": "s3 - Parte 3: curva de aprendizaje",
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "split": {
        "test_size": TEST_SIZE,
        "stratify": True,
        "random_state": RANDOM_STATE,
        "n_train": n_train_total,
        "n_test": n_test,
    },
    "fractions": [r["fraction"] for r in rows],
    "n_train_per_fraction": [r["n_train"] for r in rows],
    "accuracy_gaussian_nb": [r["accuracy_gaussian_nb"] for r in rows],
    "accuracy_logistic_regression": [r["accuracy_logistic_regression"] for r in rows],
    "table": rows,
    "figure": "curva_aprendizaje.png",
}

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2, ensure_ascii=False)

print()
print(json.dumps(results, indent=2, ensure_ascii=False))
print()
print("Figura guardada: curva_aprendizaje.png")
print("Resultados guardados: resultados.json")
