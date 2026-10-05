import json
import numpy as np

# ------------------------------------------------------------------
# Datos de la tarea (Parte 1): vocabulario de seis tokens y logits
# ------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
z = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)


# ------------------------------------------------------------------
# Softmax con temperatura, numéricamente estable:
# p_i = exp((z_i - max(z))/T) / sum_j exp((z_j - max(z))/T)
# (restar el máximo antes de dividir por T evita overflow/underflow)
# ------------------------------------------------------------------
def softmax_temperature(logits, T):
    logits = np.asarray(logits, dtype=float)
    T = float(T)
    if T <= 0:
        raise ValueError("La temperatura debe ser positiva.")
    z_shift = (logits - np.max(logits)) / T  # estabilidad numérica
    e = np.exp(z_shift)
    return e / np.sum(e)


# ------------------------------------------------------------------
# Entropía de Shannon en bits: H = -sum p_i * log2(p_i)
# ------------------------------------------------------------------
def entropy_bits(p):
    p = np.asarray(p, dtype=float)
    nz = p[p > 0]  # los términos con p=0 no contribuyen
    return float(-np.sum(nz * np.log2(nz)))


# ------------------------------------------------------------------
# Cálculo de las distribuciones y entropías para T = 0.5, 1, 2
# ------------------------------------------------------------------
temperatures = [0.5, 1.0, 2.0]

distributions = {}
entropies = {}
row_sums = {}
for T in temperatures:
    p = softmax_temperature(z, T)
    distributions[T] = p
    entropies[T] = entropy_bits(p)
    row_sums[T] = float(np.sum(p))  # verificación: debe ser 1.0

# ------------------------------------------------------------------
# Tabla única con las tres distribuciones y las tres entropías
# (pandas no está disponible: se formatea manualmente a 4 decimales)
# ------------------------------------------------------------------
header = ["T"] + tokens + ["entropia_bits"]
rows = []
for T in temperatures:
    row = [f"{T:g}"] + [f"{v:.4f}" for v in distributions[T]] + [f"{entropies[T]:.4f}"]
    rows.append(row)

widths = [max(len(header[c]), max(len(r[c]) for r in rows)) for c in range(len(header))]


def fmt_row(r):
    return " | ".join(r[c].rjust(widths[c]) for c in range(len(header)))


sep = "-+-".join("-" * w for w in widths)
table_str = "\n".join([fmt_row(header), sep] + [fmt_row(r) for r in rows])

print("Parte 1 - Softmax con temperatura: distribuciones y entropia (bits)")
print(table_str)

print()
print("Verificacion (suma de cada fila):")
for T in temperatures:
    print(f"  T={T:g}: suma = {row_sums[T]:.4f}")

print()
print("Valores (mismos numeros que en la tabla):")
for T in temperatures:
    dist_str = ", ".join(f"{tok}={distributions[T][i]:.4f}" for i, tok in enumerate(tokens))
    print(f"  T={T:g}: {dist_str} | entropia={entropies[T]:.4f} bits")

# ------------------------------------------------------------------
# Resultados a resultados.json (solo floats/strings de Python)
# ------------------------------------------------------------------
results = {
    "logits": {tok: float(val) for tok, val in zip(tokens, z)},
    "temperatures": [float(T) for T in temperatures],
    "distributions": {
        f"T={T:g}": {tok: float(distributions[T][i]) for i, tok in enumerate(tokens)}
        for T in temperatures
    },
    "entropies_bits": {f"T={T:g}": float(entropies[T]) for T in temperatures},
    "row_sums": {f"T={T:g}": float(row_sums[T]) for T in temperatures},
    "table_4_decimals": table_str,
}

with open("resultados.json", "w") as f:
    json.dump(results, f, indent=2)

print()
print("Resultados escritos en resultados.json")
