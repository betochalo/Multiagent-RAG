import json
import numpy as np

# ------------------------------------------------------------------
# Subtarea s2 (Parte 2): muestreo de núcleo (top-p) con p = 0.9
# sobre la distribución softmax con T = 1 de la Tarea B (MMIA 6013).
# Se recomputa la softmax de T = 1 (Parte 1) a partir de los logits.
# ------------------------------------------------------------------

# Datos literales del enunciado
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
T = 1.0
p_threshold = 0.9

# ------------------------------------------------------------------
# Paso 1: softmax con temperatura, numéricamente estable
# p_i = exp((z_i - max)/T) / sum_j exp((z_j - max)/T)
# ------------------------------------------------------------------
def softmax_with_temperature(z, temperature):
    z = np.asarray(z, dtype=float)
    z_shift = (z - np.max(z)) / temperature
    e = np.exp(z_shift)
    return e / np.sum(e)

probs = softmax_with_temperature(logits, T)

# ------------------------------------------------------------------
# Paso 2: ordenar los tokens por probabilidad descendente
# ------------------------------------------------------------------
order = np.argsort(-probs, kind="stable")
sorted_tokens = [tokens[i] for i in order]
sorted_probs = probs[order]
cumulative = np.cumsum(sorted_probs)

# ------------------------------------------------------------------
# Paso 3: conjunto mínimo de tokens cuya probabilidad acumulada
# alcanza o supera 0.9
# ------------------------------------------------------------------
k = 0
cum_nucleus = 0.0
for cp in cumulative:
    k += 1
    cum_nucleus = float(cp)
    if cum_nucleus >= p_threshold:
        break

nucleus_pos = order[:k]
survivors = [tokens[i] for i in nucleus_pos]
excluded = [tokens[i] for i in order[k:]]

# ------------------------------------------------------------------
# Paso 4: renormalizar las probabilidades del núcleo para que sumen 1
# ------------------------------------------------------------------
renorm = probs[nucleus_pos] / cum_nucleus

# ------------------------------------------------------------------
# Resultados -> resultados.json
# ------------------------------------------------------------------
results = {
    "tokens": tokens,
    "logits": [float(v) for v in logits],
    "temperature": float(T),
    "p_threshold": float(p_threshold),
    "softmax_T1_full": {tok: float(pr) for tok, pr in zip(tokens, probs)},
    "softmax_T1_4dp": {tok: round(float(pr), 4) for tok, pr in zip(tokens, probs)},
    "sorted_order_desc": sorted_tokens,
    "sorted_probs_desc_4dp": [round(float(v), 4) for v in sorted_probs],
    "cumulative_probs_desc_4dp": [round(float(v), 4) for v in cumulative],
    "nucleus_size": int(k),
    "surviving_tokens": survivors,
    "excluded_tokens": excluded,
    "surviving_original_probs_4dp": {
        tok: round(float(probs[tokens.index(tok)]), 4) for tok in survivors
    },
    "nucleus_cumulative_prob": cum_nucleus,
    "nucleus_cumulative_prob_4dp": round(cum_nucleus, 4),
    "renormalized_distribution_full": {
        tok: float(pr) for tok, pr in zip(survivors, renorm)
    },
    "renormalized_distribution_4dp": {
        tok: round(float(pr), 4) for tok, pr in zip(survivors, renorm)
    },
    "renormalized_sum": float(np.sum(renorm)),
}

with open("resultados.json", "w") as f:
    json.dump(results, f, indent=2)

# ------------------------------------------------------------------
# Salida a stdout (los mismos números)
# ------------------------------------------------------------------
print("Logits:", {tok: float(v) for tok, v in zip(tokens, logits)})
print(f"\nSoftmax con T = {T} (numéricamente estable):")
for tok, pr in zip(tokens, probs):
    print(f"  {tok}: {pr:.4f}")
print(f"  suma = {np.sum(probs):.4f}")

print("\nTokens ordenados por probabilidad descendente:")
for tok, pr, cp in zip(sorted_tokens, sorted_probs, cumulative):
    print(f"  {tok}: p = {pr:.4f}  acumulada = {cp:.4f}")

print(f"\nTop-p con p = {p_threshold}:")
print(f"  Tokens sobrevivientes (núcleo, k = {k}): {survivors}")
print(f"  Tokens recortados: {excluded}")
print(f"  Probabilidad acumulada del núcleo: {cum_nucleus:.4f}")

print("\nDistribución renormalizada del núcleo:")
for tok, pr in zip(survivors, renorm):
    orig = probs[tokens.index(tok)]
    print(f"  {tok}: original = {orig:.4f}  ->  renormalizada = {pr:.4f}")
print(f"  suma renormalizada = {np.sum(renorm):.4f}")
