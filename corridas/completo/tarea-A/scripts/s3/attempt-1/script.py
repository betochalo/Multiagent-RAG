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
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]


def class_counts(y):
    values, counts = np.unique(y, return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}


def main():
    # ------------------------------------------------------------------
    # Parte 1 (recomputada localmente): la única división de la tarea
    # 70% / 30%, estratificada por clase, random_state=42
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    results = {
        "subtask": "s3 - curva de aprendizaje (Parte 3)",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "split": {
            "test_size": 0.3,
            "stratify": True,
            "random_state": RANDOM_STATE,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X.shape[1]),
            "train_class_counts": class_counts(y_train),
            "test_class_counts": class_counts(y_test),
        },
        "fractions": [float(f) for f in FRACTIONS],
        "learning_curve": [],
    }

    print("Division Parte 1: train=%d ejemplos, test=%d ejemplos"
          % (X_train.shape[0], X_test.shape[0]))
    print("frac    n_train   acc_GaussianNB   acc_LogReg")
    print("-" * 50)

    # ------------------------------------------------------------------
    # Parte 3: submuestras estratificadas del entrenamiento
    # ------------------------------------------------------------------
    for frac in FRACTIONS:
        if frac >= 1.0:
            # 100% del entrenamiento = el conjunto de entrenamiento completo
            X_sub, y_sub = X_train, y_train
        else:
            # Submuestra estratificada por clase, random_state=42
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train,
                train_size=frac,
                stratify=y_train,
                random_state=RANDOM_STATE,
            )

        n_sub = int(X_sub.shape[0])

        # --- Modelo generativo: GaussianNB (atributos crudos) ---
        nb = GaussianNB()
        nb.fit(X_sub, y_sub)
        acc_nb = float(accuracy_score(y_test, nb.predict(X_test)))

        # --- Modelo discriminativo: regresión logística con el escalador
        #     ajustado SOLO con la submuestra de entrenamiento ---
        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub)
        X_test_scaled = scaler.transform(X_test)  # mismo test completo, sin reajustar
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_sub_scaled, y_sub)
        acc_lr = float(accuracy_score(y_test, lr.predict(X_test_scaled)))

        entry = {
            "fraction": float(frac),
            "n_train_examples": n_sub,
            "subsample_class_counts": class_counts(y_sub),
            "accuracy_gaussian_nb": acc_nb,
            "accuracy_logistic_regression": acc_lr,
        }
        results["learning_curve"].append(entry)
        print(f"{frac:5.2f}  {n_sub:8d}  {acc_nb:14.4f}  {acc_lr:12.4f}")

    # ------------------------------------------------------------------
    # Figura: exactitud vs. número de ejemplos de entrenamiento
    # ------------------------------------------------------------------
    ns = [e["n_train_examples"] for e in results["learning_curve"]]
    accs_nb = [e["accuracy_gaussian_nb"] for e in results["learning_curve"]]
    accs_lr = [e["accuracy_logistic_regression"] for e in results["learning_curve"]]

    plt.figure(figsize=(8, 5))
    plt.plot(ns, accs_nb, marker="o", color="tab:blue",
             label="Naive Bayes gaussiano (generativo)")
    plt.plot(ns, accs_lr, marker="s", color="tab:orange",
             label="Regresión logística (discriminativo)")
    plt.xlabel("Número de ejemplos de entrenamiento")
    plt.ylabel("Exactitud (accuracy) en el conjunto de prueba")
    plt.title("Curva de aprendizaje: generativo vs. discriminativo\n"
              "(Breast Cancer, división 70/30 estratificada, random_state=42)")
    plt.xticks(ns)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("curva_aprendizaje.png", dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Guardar e imprimir resultados
    # ------------------------------------------------------------------
    with open("resultados.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nContenido de resultados.json:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
