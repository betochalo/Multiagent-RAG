import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def class_stats(y_arr, target_names):
    """Conteos y proporciones de clase para un vector de etiquetas."""
    unique, counts = np.unique(y_arr, return_counts=True)
    counts_dict = {target_names[int(c)]: int(n) for c, n in zip(unique, counts)}
    for name in target_names:  # asegurar que ambas clases aparezcan
        counts_dict.setdefault(name, 0)
    total = sum(counts_dict.values())
    props_dict = {name: counts_dict[name] / total for name in target_names}
    return counts_dict, props_dict


def main():
    # ------------------------------------------------------------------
    # Parte 1: carga del conjunto Breast Cancer Wisconsin (Diagnostic)
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target
    target_names = list(data.target_names)  # ['malignant', 'benign']

    n_total = int(X.shape[0])
    n_attributes = int(X.shape[1])

    class_counts_full, class_props_full = class_stats(y, target_names)

    # ------------------------------------------------------------------
    # División única de la tarea: 70/30 estratificada, random_state=42
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    train_counts, train_props = class_stats(y_train, target_names)
    test_counts, test_props = class_stats(y_test, target_names)

    # ------------------------------------------------------------------
    # Verificación de la estratificación: las proporciones de cada clase
    # en train y en test deben coincidir con las del conjunto completo.
    # ------------------------------------------------------------------
    max_diff_train = max(
        abs(train_props[name] - class_props_full[name]) for name in target_names
    )
    max_diff_test = max(
        abs(test_props[name] - class_props_full[name]) for name in target_names
    )
    tol = 0.01  # tolerancia de 1 punto porcentual
    stratification_preserved = bool(max_diff_train < tol and max_diff_test < tol)

    results = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "n_total_examples": n_total,
        "n_attributes": n_attributes,
        "class_counts_full": class_counts_full,
        "class_proportions_full": class_props_full,
        "split_parameters": {
            "test_size": 0.3,
            "stratify": True,
            "random_state": 42,
        },
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
        "train_class_proportions": train_props,
        "test_class_proportions": test_props,
        "max_proportion_difference_train": float(max_diff_train),
        "max_proportion_difference_test": float(max_diff_test),
        "stratification_preserved": stratification_preserved,
    }

    # ------------------------------------------------------------------
    # Escritura de resultados y reporte en stdout
    # ------------------------------------------------------------------
    with open("resultados.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

    # Resumen legible
    print("\n--- Resumen ---")
    print(f"Total de ejemplos: {n_total}")
    print(f"Número de atributos: {n_attributes}")
    print(f"Conteo de clases (completo): {class_counts_full}")
    print(f"Proporciones (completo): {class_props_full}")
    print(f"Tamaño entrenamiento: {results['train_size']}")
    print(f"Tamaño prueba: {results['test_size']}")
    print(f"Conteo de clases (entrenamiento): {train_counts}")
    print(f"Conteo de clases (prueba): {test_counts}")
    print(f"Proporciones (entrenamiento): {train_props}")
    print(f"Proporciones (prueba): {test_props}")
    print(f"Estratificación conservada: {stratification_preserved}")


if __name__ == "__main__":
    main()
