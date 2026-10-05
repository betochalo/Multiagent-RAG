import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ------------------------------------------------------------------
# Parte 1 (recomputada en este script): datos y única división
# ------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
class_names = [str(n) for n in data.target_names]  # 0: malignant, 1: benign

n_total, n_attributes = X.shape
class_counts_full = {class_names[c]: int(np.sum(y == c)) for c in np.unique(y)}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

train_class_counts = {class_names[c]: int(np.sum(y_train == c)) for c in np.unique(y_train)}
test_class_counts = {class_names[c]: int(np.sum(y_test == c)) for c in np.unique(y_test)}

# ------------------------------------------------------------------
# Estandarización: el escalador se ajusta SOLO con el entrenamiento
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)   # fit + transform en train
X_test_std = scaler.transform(X_test)         # solo transform en test

# ------------------------------------------------------------------
# Parte 2: GaussianNB y LogisticRegression sobre datos estandarizados
# ------------------------------------------------------------------
models = [
    ("gaussian_nb", "GaussianNB", GaussianNB()),
    ("logistic_regression", "LogisticRegression(max_iter=1000, random_state=42)",
     LogisticRegression(max_iter=1000, random_state=42)),
]

model_results = {}
table_rows = []
for key, label, model in models:
    model.fit(X_train_std, y_train)           # entrenamiento solo con train
    y_pred = model.predict(X_test_std)        # evaluación sobre el mismo test
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred)
    model_results[key] = {
        "model": label,
        "accuracy": acc,
        "f1_macro": f1m,
        "confusion_matrix_test": {"labels": class_names, "matrix": cm.tolist()},
    }
    table_rows.append({"model": label, "accuracy": acc, "f1_macro": f1m})

# ------------------------------------------------------------------
# Resultados -> resultados.json
# ------------------------------------------------------------------
results = {
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "part1_split_recomputed": {
        "test_size_parameter": 0.3,
        "stratify": True,
        "random_state": 42,
        "n_total_examples": int(n_total),
        "n_attributes": int(n_attributes),
        "class_counts_full": class_counts_full,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "train_class_counts": train_class_counts,
        "test_class_counts": test_class_counts,
    },
    "standardization": {
        "scaler": "StandardScaler",
        "fit_on": "train_only",
        "transform_applied_to": ["train", "test"],
        "n_features_scaled": int(X_train_std.shape[1]),
    },
    "models": model_results,
    "table_test": table_rows,
}

with open("resultados.json", "w") as f:
    json.dump(results, f, indent=2)

# ------------------------------------------------------------------
# Figura: tabla con accuracy y F1 macro (informativa)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 2.6))
ax.axis("off")
cell_text = [[r["model"], f"{r['accuracy']:.4f}", f"{r['f1_macro']:.4f}"] for r in table_rows]
tbl = ax.table(cellText=cell_text,
               colLabels=["Modelo", "Accuracy (test)", "F1 macro (test)"],
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.8)
ax.set_title(f"Parte 2 — Desempeño en el conjunto de prueba (n={X_test.shape[0]})", pad=20)
plt.savefig("tabla_parte2.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# Salida por stdout (mismos números que el JSON)
# ------------------------------------------------------------------
print("Parte 1 (recomputada):")
print(f"  Ejemplos totales: {n_total}, atributos: {n_attributes}")
print(f"  Clases (dataset completo): {class_counts_full}")
print(f"  Train: {X_train.shape[0]} ejemplos -> {train_class_counts}")
print(f"  Test : {X_test.shape[0]} ejemplos -> {test_class_counts}")
print()
print("Parte 2 — tabla sobre el conjunto de prueba:")
print("| Modelo | Accuracy | F1 macro |")
print("|---|---|---|")
for r in table_rows:
    print(f"| {r['model']} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} |")
print()
print(json.dumps(results, indent=2))
