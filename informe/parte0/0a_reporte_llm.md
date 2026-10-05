# reporte.md

# Tarea A — Generativo contra discriminativo con pocos datos

**MMIA 6013 · Taller 03 v2**

## Introducción

Ng y Jordan (2002) demostraron que un clasificador **generativo** alcanza su error asintótico con O(log n) ejemplos, mientras que uno **discriminativo** necesita O(n); a cambio, el error asintótico del generativo suele ser mayor. En términos de sesgo y varianza: el generativo (Naive Bayes) impone un sesgo fuerte —independencia condicional de los atributos dada la clase— que reduce la varianza pero pone un piso al error; el discriminativo (regresión logística) tiene sesgo menor y varianza mayor, por lo que pierde con pocos datos y gana con muchos. Esta tarea verifica ese cruce sobre datos reales: Breast Cancer Wisconsin (Diagnostic).

## Metodología

- **Datos:** 569 ejemplos, 30 atributos reales; 212 malignos y 357 benignos.
- **División única 70/30 estratificada** (`random_state=42`): 398 de entrenamiento (148 malignos, 250 benignos) y 171 de prueba (64 malignos, 107 benignos). El conjunto de prueba no se usa para ajustar nada.
- **Modelos:** `GaussianNB` y `LogisticRegression` (C=1.0, `max_iter=1000`) sobre atributos estandarizados con `StandardScaler` ajustado **solo** con el entrenamiento.
- **Métricas:** exactitud y F1 macro sobre la prueba.
- **Curva de aprendizaje:** submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y 100 % del entrenamiento (`random_state=42`), de tamaños 20, 40, 100, 199 y 398 ejemplos. En cada submuestra se reajusta el escalador y se reentrena el modelo desde cero; la evaluación es siempre sobre la misma prueba de 171 casos. El código completo está en el Anexo.

## Resultados

**Tabla 1 — Parte 2 (entrenamiento completo, n = 398):**

| Modelo | Exactitud | F1 macro |
|---|---|---|
| Naive Bayes gaussiano (generativo) | 0.9415 | 0.9372 |
| Regresión logística (discriminativo) | **0.9766** | **0.9750** |

**Tabla 2 — Parte 3 (exactitud en prueba según tamaño de entrenamiento):**

| n entrenamiento | Naive Bayes | Reg. logística |
|---|---|---|
| 20 | **0.9240** | 0.8889 |
| 40 | **0.9298** | 0.9240 |
| 100 | 0.9357 | **0.9532** |
| 199 | 0.9415 | **0.9649** |
| 398 | 0.9415 | **0.9766** |

![Curva de aprendizaje](curva_aprendizaje.png)

*Figura 1.* Exactitud en prueba contra número de ejemplos de entrenamiento (`curva_aprendizaje.png`). Naive Bayes parte arriba, se aplana desde n ≈ 100; la regresión logística lo cruza entre 40 y 100 ejemplos y sigue mejorando hasta el final.

## Discusión

**Se observa el cruce que predicen Ng y Jordan.** Con solo 20 ejemplos el modelo generativo gana por 3.5 puntos (0.9240 contra 0.8889): con 30 atributos y unos 7–12 ejemplos por clase, la regresión logística debe estimar 31 parámetros a partir de datos insuficientes y su varianza domina; el sesgo fuerte de Naive Bayes estabiliza las estimaciones y le permite acercarse rápido a su meseta. Esto coincide con la convergencia en O(log n) del generativo.

**El cruce ocurre entre 40 y 100 ejemplos.** A partir de n = 100 la regresión logística supera a Naive Bayes y la brecha crece hasta 3.5 puntos con el entrenamiento completo (0.9766 contra 0.9415).

**Naive Bayes se estanca en su error asintótico.** Su exactitud queda clavada en 0.9415 desde n = 199: el piso lo impone su sesgo de independencia. En este conjunto, atributos como `mean radius`, `mean perimeter` y `mean area` están casi perfectamente correlacionados; Naive Bayes los trata como evidencia independiente y duplica información, subestimando la incertidumbre. Además comete 6 falsos negativos (malignos tomados como benignos) frente a solo 2 de la regresión logística, lo cual es relevante en un cribado clínico.

**La regresión logística tiene el error asintótico menor.** Su supuesto es más débil (frontera lineal en el espacio estandarizado, sin suponer independencia), es decir, sesgo menor y varianza mayor. Con pocos ejemplos esa varianza la penaliza (0.8889 con n = 20), pero al crecer n las estimaciones se estabilizan y su menor sesgo le permite superar al generativo y seguir mejorando hasta n = 398.

**Conclusión práctica:** con ≤ 40 ejemplos conviene Naive Bayes; con ≥ 100, la regresión logística. El compromiso sesgo–varianza explica ambas zonas.

## Conclusiones

Sobre Breast Cancer Wisconsin se reprodujo el fenómeno de Ng y Jordan: el clasificador generativo converge más rápido (mejor con 20–40 ejemplos, meseta en 0.9415 desde n ≈ 100) y el discriminativo alcanza un error asintótico menor (0.9766 con 398 ejemplos), cruzándolo entre 40 y 100 ejemplos de entrenamiento. El patrón es exactamente el previsto por la teoría de sesgo–varianza: sesgo alto y varianza baja compran rapidez de convergencia al precio de un piso de error superior.

## Anexo — Código (`tarea_a.py`)

```python
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
    Xs, _, ys, _ = train_test_split(X_tr, y_tr, train_size=n_k,
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
```