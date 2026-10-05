# -*- coding: utf-8 -*-
"""
Subtarea s3 (Parte 3 — La curva de aprendizaje).

Recomputa la división de la Parte 1 (load_breast_cancer, test_size=0.30,
stratify=y, random_state=42). Para cada fracción del entrenamiento
(5 %, 10 %, 25 %, 50 %, 100 %) toma una submuestra ESTRATIFICADA del conjunto
de entrenamiento con random_state=42 (train_test_split sobre el train),
entrena GaussianNB y LogisticRegression (StandardScaler ajustado SOLO con esa
submuestra) y evalúa la accuracy de ambos SIEMPRE sobre el mismo conjunto de
prueba completo (30 %). Imprime la tabla, dibuja la curva de aprendizaje
(accuracy vs. número de ejemplos de entrenamiento, una curva por modelo) y
guarda la figura como curva_aprendizaje.png y los números en resultados.json.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.30
FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
FRAC_LABELS = ["5%", "10%", "25%", "50%", "100%"]

# ----------------------------------------------------------------------
# Parte 1 (recomputada): única división 70/30 estratificada, random_state=42
# ----------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

n_train_total = int(X_train.shape[0])
n_test_total = int(X_test.shape[0])

clases_train = {
    str(name): int((y_train == i).sum()) for i, name in enumerate(data.target_names)
}
clases_test = {
    str(name): int((y_test == i).sum()) for i, name in enumerate(data.target_names)
}

print("=" * 78)
print(
    "División Parte 1: {} entrenamiento / {} prueba "
    "(test_size=0.30, stratify=y, random_state=42)".format(n_train_total, n_test_total)
)
print("Clases entrenamiento: {}".format(clases_train))
print("Clases prueba:        {}".format(clases_test))
print("=" * 78)

# ----------------------------------------------------------------------
# Parte 3: submuestras estratificadas del entrenamiento + curva de aprendizaje
# ----------------------------------------------------------------------
rows = []
n_by_frac = []
acc_nb_by_frac = []
acc_lr_by_frac = []

for frac, label in zip(FRACTIONS, FRAC_LABELS):
    if frac >= 1.0:
        # 100 %: el conjunto de entrenamiento completo
        X_sub, y_sub = X_train, y_train
    else:
        # Submuestra estratificada del train, con la misma semilla 42
        X_sub, _dx, y_sub, _dy = train_test_split(
            X_train,
            y_train,
            train_size=frac,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )

    n_sub = int(X_sub.shape[0])
    clases_sub = {
        str(name): int((y_sub == i).sum()) for i, name in enumerate(data.target_names)
    }

    # --- Modelo generativo: GaussianNB (atributos crudos, como en la Parte 2) ---
    nb = GaussianNB()
    nb.fit(X_sub, y_sub)
    acc_nb = float(accuracy_score(y_test, nb.predict(X_test)))

    # --- Modelo discriminativo: LogisticRegression con StandardScaler
    #     ajustado SOLO con la submuestra de entrenamiento ---
    scaler = StandardScaler().fit(X_sub)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(scaler.transform(X_sub), y_sub)
    acc_lr = float(accuracy_score(y_test, lr.predict(scaler.transform(X_test))))

    n_by_frac.append(n_sub)
    acc_nb_by_frac.append(acc_nb)
    acc_lr_by_frac.append(acc_lr)
    rows.append(
        {
            "fraccion": label,
            "n_ejemplos_entrenamiento": n_sub,
            "clases_en_submuestra": clases_sub,
            "accuracy_gaussian_nb": acc_nb,
            "accuracy_logistic_regression": acc_lr,
        }
    )

# ----------------------------------------------------------------------
# Tabla de accuracy por fracción y modelo (stdout)
# ----------------------------------------------------------------------
print()
header = "{:<10}{:>12}{:>18}{:>28}".format(
    "Fracción", "n_train", "Acc GaussianNB", "Acc LogisticRegression"
)
print(header)
print("-" * len(header))
for r in rows:
    print(
        "{:<10}{:>12}{:>18.4f}{:>28.4f}".format(
            r["fraccion"],
            r["n_ejemplos_entrenamiento"],
            r["accuracy_gaussian_nb"],
            r["accuracy_logistic_regression"],
        )
    )
print("-" * len(header))
print("Evaluación: siempre sobre el mismo conjunto de prueba completo ({} ejemplos).".format(n_test_total))

# ----------------------------------------------------------------------
# Figura: curva de aprendizaje (accuracy vs. n ejemplos de entrenamiento)
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(
    n_by_frac,
    acc_nb_by_frac,
    marker="o",
    color="tab:blue",
    label="GaussianNB (generativo)",
)
plt.plot(
    n_by_frac,
    acc_lr_by_frac,
    marker="s",
    color="tab:orange",
    label="LogisticRegression (discriminativo)",
)
plt.xlabel("Número de ejemplos de entrenamiento")
plt.ylabel("Accuracy en prueba")
plt.title("Curva de aprendizaje — Breast Cancer (división 70/30, prueba 30 %)")
plt.xticks(n_by_frac)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("curva_aprendizaje.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# resultados.json
# ----------------------------------------------------------------------
results = {
    "subtarea": "s3 - Parte 3: curva de aprendizaje",
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "division_parte1": {
        "test_size": TEST_SIZE,
        "stratify": "y",
        "random_state": RANDOM_STATE,
        "n_entrenamiento": n_train_total,
        "n_prueba": n_test_total,
        "clases_entrenamiento": clases_train,
        "clases_prueba": clases_test,
    },
    "fracciones": FRAC_LABELS,
    "metodo_submuestreo": (
        "train_test_split(X_train, y_train, train_size=fraccion, "
        "stratify=y_train, random_state=42); 100% usa el train completo"
    ),
    "conjunto_evaluacion": "mismo conjunto de prueba completo (30 %) para todos los modelos",
    "curva_aprendizaje": {
        "GaussianNB": {
            r["fraccion"]: {
                "n_ejemplos_entrenamiento": r["n_ejemplos_entrenamiento"],
                "accuracy_prueba": r["accuracy_gaussian_nb"],
            }
            for r in rows
        },
        "LogisticRegression": {
            r["fraccion"]: {
                "n_ejemplos_entrenamiento": r["n_ejemplos_entrenamiento"],
                "accuracy_prueba": r["accuracy_logistic_regression"],
            }
            for r in rows
        },
    },
    "tabla_accuracy": rows,
    "figura": "curva_aprendizaje.png",
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print()
print("Figura guardada en curva_aprendizaje.png; resultados en resultados.json")
print(json.dumps(results, indent=2, ensure_ascii=False))
