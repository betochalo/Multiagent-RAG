# SUBTASK s2 — Parte 2: dos clasificadores (GaussianNB vs LogisticRegression)
# Recomputa la división de la Parte 1, estandariza (escalador ajustado SOLO con
# el entrenamiento), entrena ambos modelos y evalúa accuracy y F1 macro en prueba.

import json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42

# ---------- Parte 1 (recomputada): datos y única división 70/30 estratificada ----------
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=RANDOM_STATE,
)

# ---------- Estandarización: el escalador se ajusta únicamente con el entrenamiento ----------
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)  # fit + transform SOLO en entrenamiento
X_test_std = scaler.transform(X_test)        # la prueba solo se transforma, nunca se ajusta

# ---------- Parte 2: entrenamiento de los dos clasificadores ----------
models = {
    "gaussian_nb": ("Naive Bayes gaussiano", GaussianNB()),
    "logistic_regression": (
        "Regresión logística",
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    ),
}

results = {
    "split": {
        "test_size": 0.3,
        "stratify": "y",
        "random_state": RANDOM_STATE,
        "n_train_examples": int(X_train.shape[0]),
        "n_test_examples": int(X_test.shape[0]),
        "n_attributes": int(X.shape[1]),
    },
    "scaler": {
        "type": "StandardScaler",
        "fit_on": "train_only",
        "train_mean_first3": [float(m) for m in scaler.mean_[:3]],
        "train_std_first3": [float(s) for s in scaler.scale_[:3]],
    },
    "test_metrics": {},
}

rows = []
for key, (pretty_name, model) in models.items():
    model.fit(X_train_std, y_train)          # entrenamiento solo con train estandarizado
    y_pred = model.predict(X_test_std)       # evaluación sobre el conjunto de prueba
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))
    results["test_metrics"][key] = {
        "model": pretty_name,
        "accuracy": acc,
        "f1_macro": f1m,
    }
    rows.append((pretty_name, acc, f1m))

# ---------- Tabla de resultados (Markdown impresa) ----------
lines = []
lines.append("| Modelo | Accuracy (prueba) | F1 macro (prueba) |")
lines.append("|---|---|---|")
for name, acc, f1m in rows:
    lines.append(f"| {name} | {acc:.4f} | {f1m:.4f} |")
md_table = "\n".join(lines)
results["markdown_table"] = md_table

# ---------- Salida: tabla + mismos números en stdout y en resultados.json ----------
print(md_table)
print()
print(json.dumps(results, indent=2, ensure_ascii=False))

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
