import json
import numpy as np

# ------------------------------------------------------------------
# Datos del problema (Parte 1 — softmax con temperatura)
# Vocabulario de seis tokens t0..t5 con sus logits
# ------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
temperatures = [0.5, 1.0, 2.0]


def softmax_con_temperatura(z, T):
    """
    Softmax con temperatura: p_i = exp(z_i/T) / sum_j exp(z_j/T).

    Implementación numéricamente estable: se resta el máximo de los
    logits antes de exponenciar, de modo que el mayor exponente es 0
    y no ocurre overflow. Matemáticamente el resultado es idéntico.
    """
    z = np.asarray(z, dtype=float)
    z_shift = (z - np.max(z)) / float(T)  # resta del máximo -> estabilidad
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z)


def entropia_bits(p):
    """Entropía en bits: H = -sum p_i log2 p_i (p_i = 0 aporta 0)."""
    p = np.asarray(p, dtype=float)
    p_pos = p[p > 0]
    return float(-np.sum(p_pos * np.log2(p_pos)))


# ------------------------------------------------------------------
# Cálculo de las tres distribuciones, sus entropías y las sumas
# ------------------------------------------------------------------
distribuciones = {}
entropias = {}
sumas = {}

for T in temperatures:
    p = softmax_con_temperatura(logits, T)
    distribuciones[T] = p
    entropias[T] = entropia_bits(p)
    sumas[T] = float(np.sum(p))

# ------------------------------------------------------------------
# Tabla con cuatro decimales
# ------------------------------------------------------------------
col_w = 12
sep = "-" * (8 + col_w * len(temperatures))

lines = []
lines.append("Softmax con temperatura: distribuciones y entropia (bits)")
lines.append(sep)
lines.append("Token".ljust(8) + "".join(f"{'T=' + str(T):<{col_w}}" for T in temperatures))
lines.append(sep)
for i, tok in enumerate(tokens):
    row = tok.ljust(8)
    for T in temperatures:
        row += f"{distribuciones[T][i]:<{col_w}.4f}"
    lines.append(row)
lines.append(sep)
row = "H (bits)".ljust(8)
for T in temperatures:
    row += f"{entropias[T]:<{col_w}.4f}"
lines.append(row)
row = "Suma".ljust(8)
for T in temperatures:
    row += f"{sumas[T]:<{col_w}.4f}"
lines.append(row)
lines.append(sep)

tabla = "\n".join(lines)
print(tabla)

# ------------------------------------------------------------------
# Resultados en JSON (números como floats planos)
# ------------------------------------------------------------------
resultados = {
    "tokens": tokens,
    "logits": [float(x) for x in logits],
    "temperatures": [float(T) for T in temperatures],
    "distributions": {
        f"T={T}": {tok: float(distribuciones[T][i]) for i, tok in enumerate(tokens)}
        for T in temperatures
    },
    "entropies_bits": {f"T={T}": float(entropias[T]) for T in temperatures},
    "distribution_sums": {f"T={T}": float(sumas[T]) for T in temperatures},
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# Impresión de los mismos números calculados (precisión completa)
print("\nValores guardados en resultados.json:")
print(json.dumps(resultados, indent=2))
