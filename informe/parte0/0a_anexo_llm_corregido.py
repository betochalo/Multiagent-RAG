import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Parte 1: datos y división única 70/30 estratificada
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42)
print("Total:", X.shape[0], "| Malignos:", (y == 0).sum(), "| Benignos:", (y == 1).sum())
print("Train:", len(y_tr), "(", (y_tr == 0).sum(), "malignos ) | Test:", len(y_te))

modelos = {"Naive Bayes gaussiano": GaussianNB(),
           "Regresión logística": LogisticRegression(max_iter=1000)}

# Parte 2: entrenamiento completo, escalador ajustado solo con train
sc = StandardScaler().fit(X_tr)
for nombre, modelo in modelos.items():
    modelo.fit(sc.transform(X_tr), y_tr)
    pred = modelo.predict(sc.transform(X_te))
    print(nombre, "| acc:", round(accuracy_score(y_te, pred), 4),
          "| F1 macro:", round(f1_score(y_te, pred, average="macro"), 4))

# Parte 3: curva de aprendizaje sobre la misma división
fracs = [0.05, 0.10, 0.25, 0.50, 1.00]
tams, curvas = [], {n: [] for n in modelos}
for f in fracs:
    n_k = int(np.ceil(f * len(y_tr)))
    Xs, _, ys, _ = (X_tr, None, y_tr, None) if n_k >= len(y_tr) else train_test_split(X_tr, y_tr, train_size=n_k,
                                    stratify=y_tr, random_state=42)
    sc_k = StandardScaler().fit(Xs)
    tams.append(len(ys))
    for nombre, modelo in modelos.items():
        m = clone(modelo).fit(sc_k.transform(Xs), ys)
        acc = accuracy_score(y_te, m.predict(sc_k.transform(X_te)))
        curvas[nombre].append(acc)
    print("n =", tams[-1], {n: round(curvas[n][-1], 4) for n in curvas})

plt.figure(figsize=(7, 5))
for nombre, accs in curvas.items():
    plt.plot(tams, accs, marker="o", label=nombre)
plt.xlabel("Ejemplos de entrenamiento")
plt.ylabel("Exactitud en prueba")
plt.title("Generativo vs discriminativo: curva de aprendizaje")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("curva_aprendizaje.png", dpi=150, bbox_inches="tight")
