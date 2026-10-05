# -*- coding: utf-8 -*-
"""
Subtask s2 (Parte 2): Dos clasificadores sobre Breast Cancer Wisconsin.
- Recomputa la división de la Parte 1: test_size=0.30, stratify=y, random_state=42.
- Estandariza con StandardScaler ajustado SOLO con el entrenamiento.
- Entrena GaussianNB y LogisticRegression(max_iter=1000) sobre datos estandarizados.
- Evalúa accuracy y F1 macro sobre el conjunto de prueba y presenta una tabla.
- Guarda los resultados en resultados.json.
"""

import json

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ----------------------------------------------------------------------
# 1. Datos y división (misma división de la Parte 1, mismos parámetros)
# ----------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# ----------------------------------------------------------------------
# 2. Estandarización: el escalador se ajusta SOLO con el entrenamiento;
#    el mismo escalador transforma el conjunto de prueba.
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 3. Entrenamiento de los dos clasificadores sobre datos estandarizados
# ----------------------------------------------------------------------
modelos = {
    "GaussianNB (generativo)": GaussianNB(),
    "LogisticRegression (discriminativo)": LogisticRegression(max_iter=1000),
}

# ----------------------------------------------------------------------
# 4. Evaluación sobre el conjunto de prueba
# ----------------------------------------------------------------------
resultados = {
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "division": {
        "test_size": 0.30,
        "stratify": "y",
        "random_state": 42,
        "n_entrenamiento": int(X_train.shape[0]),
        "n_prueba": int(X_test.shape[0]),
    },
    "estandarizacion": "StandardScaler ajustado solo con el conjunto de entrenamiento",
    "metricas_prueba": {},
}

filas = []
for nombre, modelo in modelos.items():
    modelo.fit(X_train_std, y_train)
    y_pred = modelo.predict(X_test_std)
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))
    resultados["metricas_prueba"][nombre] = {"accuracy": acc, "f1_macro": f1m}
    filas.append((nombre, acc, f1m))

# ----------------------------------------------------------------------
# 5. Tabla de resultados (texto alineado; pandas no está disponible)
# ----------------------------------------------------------------------
w = max(len(n) for n, _, _ in filas) + 2
encabezado = "{:<{w}}{:>10}{:>10}".format("Modelo", "Accuracy", "F1 macro", w=w)
print("Resultados sobre el conjunto de prueba (n = {})".format(X_test.shape[0]))
print(encabezado)
print("-" * len(encabezado))
for nombre, acc, f1m in filas:
    print("{:<{w}}{:>10.4f}{:>10.4f}".format(nombre, acc, f1m, w=w))
print()

# ----------------------------------------------------------------------
# 6. Guardar todos los números en el archivo de resultados
# ----------------------------------------------------------------------
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# Mismos números por stdout
print(json.dumps(resultados, indent=2, ensure_ascii=False))
