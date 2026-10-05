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
    acc_nb = []
    acc_lr = []
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
        # El MISMO conjunto de prueba fijo, escalado con el scaler de esta iteración
        X_test_esc = scaler.transform(X_test)

        # Modelo generativo
        modelo_nb = GaussianNB().fit(X_sub_esc, y_sub)
        # Modelo discriminativo
        modelo_lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_sub_esc, y_sub)

        # Evaluación SIEMPRE sobre el conjunto de prueba fijo (nunca sobre entrenamiento)
        a_nb = float(modelo_nb.score(X_test_esc, y_test))
        a_lr = float(modelo_lr.score(X_test_esc, y_test))

        tamanos.append(n_sub)
        acc_nb.append(a_nb)
        acc_lr.append(a_lr)
        tabla.append({
            "fraccion_porcentaje": int(round(f * 100)),
            "fraccion": "{:.0f}%".format(f * 100),
            "n_entrenamiento": n_sub,
            "n_prueba_fijo": n_test,
            "exactitud_gaussian_nb": a_nb,
            "exactitud_regresion_logistica": a_lr,
        })

    # ---------- Figura: curva de aprendizaje ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tamanos, acc_nb, marker="o", color="tab:blue",
            label="Naive Bayes gaussiano (generativo)")
    ax.plot(tamanos, acc_lr, marker="s", color="tab:red",
            label="Regresión logística (discriminativo)")
    ax.set_xlabel("Número de ejemplos de entrenamiento")
    ax.set_ylabel("Exactitud en el conjunto de prueba")
    ax.set_title("Curva de aprendizaje: Breast Cancer (división 70/30, random_state=42)")
    ax.set_xticks(tamanos)
    ymin = max(0.0, min(min(acc_nb), min(acc_lr)) - 0.05)
    ymax = min(1.0, max(max(acc_nb), max(acc_lr)) + 0.02)
    ax.set_ylim(ymin, ymax)
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
            "train_class_counts": {str(int(c)): int(n) for c, n in zip(clases_tr, conteos_train)},
            "test_class_counts": {str(int(c)): int(n) for c, n in zip(clases_te, conteos_test)},
        },
        "nota_evaluacion": (
            "Ambos modelos se evaluan siempre sobre el mismo conjunto de prueba fijo de la "
            "division 70/30 (nunca sobre los datos de entrenamiento). El StandardScaler de "
            "cada iteracion se ajusta solo con la submuestra de entrenamiento correspondiente "
            "y se aplica al conjunto de prueba."
        ),
        "fracciones_porcentaje": [int(round(f * 100)) for f in fracciones],
        "fracciones_etiquetas": ["{:.0f}%".format(f * 100) for f in fracciones],
        "tamanos_entrenamiento": tamanos,
        "n_ejemplos_prueba_fijo": n_test,
        "tabla_exactitudes": tabla,
        "exactitud_gaussian_nb": acc_nb,
        "exactitud_regresion_logistica": acc_lr,
    }

    with open("resultados.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, indent=2, ensure_ascii=False)

    # ---------- Salida estándar (mismos números) ----------
    print("Division 70/30 estratificada (random_state=42):")
    print(f"  n_train = {n_train}, n_test = {n_test}")
    print(f"  clases entrenamiento: {dict(zip(clases_tr.tolist(), conteos_train.tolist()))}")
    print(f"  clases prueba:        {dict(zip(clases_te.tolist(), conteos_test.tolist()))}")
    print()
    print("Tabla de exactitudes (evaluacion siempre sobre el mismo conjunto de prueba fijo):")
    enc = (f"{'Fraccion':>9} | {'n_train':>7} | {'n_test':>6} | "
           f"{'Exactitud NB':>12} | {'Exactitud RL':>12}")
    print(enc)
    print("-" * len(enc))
    for fila in tabla:
        print(f"{fila['fraccion']:>9} | {fila['n_entrenamiento']:7d} | "
              f"{fila['n_prueba_fijo']:6d} | "
              f"{fila['exactitud_gaussian_nb']:12.4f} | "
              f"{fila['exactitud_regresion_logistica']:12.4f}")
    print()
    print(json.dumps(resultados, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
