import json
import numpy as np

# ------------------------------------------------------------------
# Subtarea s2 (Parte 2): top-p (nucleus sampling) con p = 0.9
# sobre la distribución softmax con T = 1 de la Parte 1.
# Se recalcula todo desde los logits originales (nada hardcodeado).
# ------------------------------------------------------------------

# Datos de la Parte 1 (mismos parámetros)
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
T = 1.0
p_threshold = 0.9

# ------------------------------------------------------------------
# Softmax con temperatura, numéricamente estable
# ------------------------------------------------------------------
def softmax_with_temperature(z, temperature):
    z = np.asarray(z, dtype=float)
    z_scaled = z / temperature
    z_shifted = z_scaled - np.max(z_scaled)  # resta del máximo para estabilidad
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

probs = softmax_with_temperature(logits, T)

# ------------------------------------------------------------------
# Ordenar tokens por probabilidad descendente
# ------------------------------------------------------------------
order = np.argsort(-probs)                      # índices de mayor a menor prob.
sorted_tokens = [tokens[i] for i in order]
sorted_probs = probs[order]
cumulative = np.cumsum(sorted_probs)

# ------------------------------------------------------------------
# Top-p: conjunto MÍNIMO de tokens cuya acumulada alcanza p = 0.9,
# incluyendo el token que hace cruzar el umbral.
# ------------------------------------------------------------------
k = int(np.searchsorted(cumulative, p_threshold, side="left")) + 1
k = min(k, len(tokens))
survivor_idx = order[:k]
survivors = [tokens[i] for i in survivor_idx]
cum_at_k = float(cumulative[k - 1])

# ------------------------------------------------------------------
# Renormalizar las probabilidades de los supervivientes
# ------------------------------------------------------------------
survivor_probs = probs[survivor_idx]
renorm = survivor_probs / np.sum(survivor_probs)

full_dist = {tok: float(pr) for tok, pr in zip(tokens, probs)}
cum_dict = {sorted_tokens[j]: float(cumulative[j]) for j in range(len(tokens))}
renorm_dist = {tokens[i]: float(r) for i, r in zip(survivor_idx, renorm)}

results = {
    "tokens": tokens,
    "logits": [float(z) for z in logits],
    "temperature": float(T),
    "p_threshold": float(p_threshold),
    "softmax_T1_full_distribution": full_dist,
    "sorted_tokens_desc": sorted_tokens,
    "sorted_probs_desc": [float(x) for x in sorted_probs],
    "cumulative_probs_desc": cum_dict,
    "num_surviving_tokens": int(k),
    "surviving_tokens": survivors,
    "cumulative_prob_of_survivors": cum_at_k,
    "renormalized_distribution": renorm_dist,
    "renormalized_sum": float(np.sum(renorm)),
}

with open("resultados.json", "w") as f:
    json.dump(results, f, indent=2)

# ------------------------------------------------------------------
# Reporte en stdout (mismos números que en resultados.json)
# ------------------------------------------------------------------
print("=== Parte 2: muestreo de núcleo (top-p) con p = 0.9 ===")
print(f"\nLogits: {dict(zip(tokens, [float(z) for z in logits]))}")
print(f"Temperatura: T = {T}")

print("\nSoftmax con T = 1 (distribución completa):")
for tok in tokens:
    print(f"  {tok}: {full_dist[tok]:.6f}")

print("\nTokens ordenados por probabilidad descendente:")
print("  " + " > ".join(sorted_tokens))
print("Probabilidades acumuladas (en ese orden):")
for tok in sorted_tokens:
    print(f"  {tok}: {cum_dict[tok]:.6f}")

print(f"\nUmbral p = {p_threshold}")
print(f"Conjunto mínimo que alcanza 0.9 (incluye el token que cruza el umbral): "
      f"{k} tokens")
print(f"Tokens supervivientes: {survivors}")
print(f"Probabilidad acumulada del núcleo: {cum_at_k:.6f}")

print("\nDistribución renormalizada del núcleo:")
for tok in survivors:
    print(f"  {tok}: {renorm_dist[tok]:.6f}")
print(f"Suma de la distribución renormalizada: {float(np.sum(renorm)):.6f}")
