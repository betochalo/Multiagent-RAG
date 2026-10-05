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


def main():
    # ---------- Parte 1 (recomputada aquí): división 70/30 estratificada ----------
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])

    clases_tr, conteos_train = np.unique(y_train, return_counts=True)
    clases_te, conteos_test = np.unique(y_test, return_counts=True)

    # ---------- Parte 3: curva de aprendizaje ----------
    fracciones = [0.05, 0.10, 0.25, 0.50, 1.00]

    tamanos = []
    acc_nb_lista = []
    acc_lr_lista = []
    tabla = []

    for f in fracciones:
        if f >= 1.0:
            # 100%: se usa el entrenamiento completo
            X_sub, y_sub = X_train, y_train
        else:
            n_sub_obj = int(round(f * n_train))
            # Submuestra estratificada del entrenamiento, random_state=42
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train,
                train_size=n_sub_obj,
                stratify=y_train,
                random_state=42,
            )
        n_sub = int(X_sub.shape[0])

        # Escalador ajustado SOLO con la submuestra de entrenamiento de esta iteración
        scaler = StandardScaler().fit(X_sub)
        X_sub_esc = scaler.transform(X_sub)
        # El MISMO conjunto de prueba, escalado con el scaler de esta iteración
        X_test_esc = scaler.transform(X_test)

        # Modelo generativo
        nb = GaussianNB()
        nb.fit(X_sub_esc, y_sub)
        acc_nb = float(nb.score(X_test_esc, y_test))

        # Modelo discriminativo
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_sub_esc, y_sub)
        acc_lr = float(lr.score(X_test_esc, y_test))

        tamanos.append(n_sub)
        acc_nb_lista.append(acc_nb)
        acc_lr_lista.append(acc_lr)
        tabla.append({
            "fraccion": float(f),
            "n_entrenamiento": n_sub,
            "exactitud_gaussian_nb": acc_nb,
            "exactitud_regresion_logistica": acc_lr,
        })

    # ---------- Figura: curva de aprendizaje ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tamanos, acc_nb_lista, marker="o", color="tab:blue",
            label="Naive Bayes gaussiano (generativo)")
    ax.plot(tamanos, acc_lr_lista, marker="s", color="tab:red",
            label="Regresión logística (discriminativo)")
    ax.set_xlabel("Número de ejemplos de entrenamiento")
    ax.set_ylabel("Exactitud en el conjunto de prueba")
    ax.set_title("Curva de aprendizaje — Breast Cancer (división 70/30, random_state=42)")
    ax.set_xticks(tamanos)
    ymin = min(min(acc_nb_lista), min(acc_lr_lista)) - 0.03
    ymax = max(max(acc_nb_lista), max(acc_lr_lista)) + 0.03
    ax.set_ylim(max(0.0, ymin), min(1.02, ymax))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("curva_aprendizaje.png", dpi=120)
    plt.close(fig)

    # ---------- Resultados ----------
    resultados = {
        "subtask": "s3_curva_de_aprendizaje",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "split": {
            "test_size": 0.3,
            "stratify": True,
            "random_state": 42,
            "n_train": n_train,
            "n_test": n_test,
            "train_class_counts": {str(c): int(n) for c, n in zip(clases_tr, conteos_train)},
            "test_class_counts": {str(c): int(n) for c, n in zip(clases_te, conteos_test)},
        },
        "fracciones": [float(f) for f in fracciones],
        "tamanos_entrenamiento": tamanos,
        "tabla_exactitudes": tabla,
        "exactitud_gaussian_nb": acc_nb_lista,
        "exactitud_regresion_logistica": acc_lr_lista,
    }

    with open("resultados.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, indent=2, ensure_ascii=False)

    # ---------- Salida estándar (mismos números) ----------
    print("División 70/30 estratificada (random_state=42):")
    print(f"  n_train = {n_train}, n_test = {n_test}")
    print(f"  clases entrenamiento: {dict(zip(clases_tr.tolist(), conteos_train.tolist()))}")
    print(f"  clases prueba:        {dict(zip(clases_te.tolist(), conteos_test.tolist()))}")
    print()
    print("Tabla de exactitudes (mismo conjunto de prueba en todas las filas):")
    encabezado = (f"{'Fraccion':>9} | {'n_train':>7} | "
                  f"{'Exactitud NB':>12} | {'Exactitud RL':>12}")
    print(encabezado)
    print("-" * len(encabezado))
    for fila in tabla:
        print(f"{fila['fraccion'] * 100:8.0f}% | {fila['n_entrenamiento']:7d} | "
              f"{fila['exactitud_gaussian_nb']:12.4f} | "
              f"{fila['exactitud_regresion_logistica']:12.4f}")
    print()
    print(json.dumps(resultados, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
