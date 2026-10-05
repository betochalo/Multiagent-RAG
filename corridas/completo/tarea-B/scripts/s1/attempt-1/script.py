import json
import numpy as np


def softmax_with_temperature(z, T):
    """Softmax con temperatura, numéricamente estable.

    p_i = exp(z_i / T) / sum_j exp(z_j / T)
    Estabilidad: se resta el máximo de los logits escalados antes de
    exponenciar, evitando desbordamiento de exp.
    """
    z = np.asarray(z, dtype=float)
    scaled = z / float(T)
    scaled = scaled - np.max(scaled)  # resta del máximo (estabilidad numérica)
    exp_scaled = np.exp(scaled)
    return exp_scaled / np.sum(exp_scaled)


def entropy_bits(p):
    """Entropía en bits: H = -sum p_i log2 p_i, con la convención 0*log2(0)=0."""
    p = np.asarray(p, dtype=float)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


# ---- Datos de la tarea: vocabulario de seis tokens y sus logits ----
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = [2.0, 1.0, 0.5, 0.2, -1.0, -3.0]
temperatures = [0.5, 1.0, 2.0]

z = np.array(logits, dtype=float)

distributions = {}
entropies = {}
sums = {}

for T in temperatures:
    p = softmax_with_temperature(z, T)
    distributions[T] = p
    entropies[T] = entropy_bits(p)
    sums[T] = float(np.sum(p))

# ---- Tabla con cuatro decimales (formato de cadena; pandas no está disponible) ----
header = (
    f"{'T':<5} | "
    + " | ".join(f"{t:>7}" for t in tokens)
    + f" | {'H (bits)':>9} | {'suma':>6}"
)
lines = ["Parte 1 — Softmax con temperatura (numéricamente estable)",
         f"Logits: {logits}",
         "",
         header,
         "-" * len(header)]
for T in temperatures:
    row = (
        f"{T:<5.1f} | "
        + " | ".join(f"{p:7.4f}" for p in distributions[T])
        + f" | {entropies[T]:9.4f} | {sums[T]:6.4f}"
    )
    lines.append(row)
table_str = "\n".join(lines)
print(table_str)
print()

# ---- Impresión explícita de las entropías ----
print("Entropías en bits (H = -sum p_i log2 p_i):")
for T in temperatures:
    print(f"  T = {T}: H = {entropies[T]:.4f} bits")
print()

# ---- Verificación de que cada distribución suma 1 ----
print("Verificación (cada fila de la tabla debe sumar 1.0000):")
for T in temperatures:
    print(f"  T = {T}: suma = {sums[T]:.10f}")
print()

# ---- Escritura de resultados a JSON ----
results = {
    "tokens": tokens,
    "logits": [float(v) for v in logits],
    "temperatures": [float(T) for T in temperatures],
    "distributions": {
        f"T={T}": {tok: float(p) for tok, p in zip(tokens, distributions[T])}
        for T in temperatures
    },
    "entropies_bits": {f"T={T}": float(entropies[T]) for T in temperatures},
    "distribution_sums": {f"T={T}": float(sums[T]) for T in temperatures},
}

with open("resultados.json", "w") as f:
    json.dump(results, f, indent=2)

print("Resultados escritos en resultados.json")
