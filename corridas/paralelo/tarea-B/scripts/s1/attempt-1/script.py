import json
import numpy as np

# ----- Datos de la tarea (Parte 1): vocabulario de seis tokens y logits -----
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
z = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)


def softmax_con_temperatura(logits, T):
    """Softmax con temperatura, numéricamente estable.

    p_i = exp(z_i/T) / sum_j exp(z_j/T)
    Se resta max(z) de los logits antes de exponenciar para evitar overflow:
    exp((z_i - m)/T) / sum_j exp((z_j - m)/T), con m = max(z), es idéntica
    matemáticamente pero estable numéricamente.
    """
    logits = np.asarray(logits, dtype=float)
    m = np.max(logits)                       # estabilidad numérica
    exp_z = np.exp((logits - m) / T)
    return exp_z / np.sum(exp_z)


def entropia_bits(p):
    """Entropía de Shannon en bits: H = -sum p_i log2(p_i)."""
    p = np.asarray(p, dtype=float)
    p_pos = p[p > 0]                         # convención 0*log2(0) = 0
    return float(-np.sum(p_pos * np.log2(p_pos)))


temperaturas = [0.5, 1.0, 2.0]

distribuciones = {}
entropias = {}
sumas = {}

for T in temperaturas:
    p = softmax_con_temperatura(z, T)
    distribuciones[T] = p
    entropias[T] = entropia_bits(p)
    sumas[T] = float(np.sum(p))
    # Verificación: cada distribución debe sumar 1
    assert abs(sumas[T] - 1.0) < 1e-12, f"La distribución con T={T} no suma 1"

# ----- Tabla única: distribuciones (una fila por temperatura) + entropías -----
col_w = 9
header = f"{'T':>5} |" + "".join(f"{tok:>{col_w}}" for tok in tokens)
header += f" | {'suma':>{col_w}} {'H(bits)':>{col_w}}"
sep = "-" * len(header)
lines = [header, sep]
for T in temperaturas:
    fila = f"{T:>5.1f} |" + "".join(f"{v:>{col_w}.4f}" for v in distribuciones[T])
    fila += f" | {sumas[T]:>{col_w}.4f} {entropias[T]:>{col_w}.4f}"
    lines.append(fila)
tabla = "\n".join(lines)

print("Parte 1 — Softmax con temperatura y entropía en bits")
print(tabla)
print("\nVerificación: cada distribución suma 1 (tolerancia 1e-12): OK")

# ----- Guardar todos los números calculados en resultados.json -----
resultados = {
    "logits": {tok: float(v) for tok, v in zip(tokens, z)},
    "temperaturas": [float(T) for T in temperaturas],
    "distribuciones": {
        f"T_{T}": {tok: float(p) for tok, p in zip(tokens, distribuciones[T])}
        for T in temperaturas
    },
    "suma_distribuciones": {f"T_{T}": sumas[T] for T in temperaturas},
    "entropia_bits": {f"T_{T}": entropias[T] for T in temperaturas},
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

print("\nContenido de resultados.json:")
print(json.dumps(resultados, indent=2))
