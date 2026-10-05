import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def softmax_with_temperature(z, temperature):
    """Softmax con temperatura, numéricamente estable (resta el máximo)."""
    z = np.asarray(z, dtype=float)
    z_scaled = z / temperature
    z_shifted = z_scaled - np.max(z_scaled)
    e = np.exp(z_shifted)
    return e / np.sum(e)


# ----- Datos y parámetros (según el enunciado) -----
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0])
T = 1.0
p_threshold = 0.9
n_samples = 10000
seed = 0

# ----- Parte 1 (recomputada): softmax estable en T = 1 -----
probs_T1 = softmax_with_temperature(logits, T)

# ----- Parte 2 (recomputada): top-p con p = 0.9 -----
order_desc = np.argsort(-probs_T1, kind="stable")
sorted_tokens = [tokens[i] for i in order_desc]
sorted_probs = probs_T1[order_desc]
cumulative = np.cumsum(sorted_probs)

# conjunto mínimo cuya probabilidad acumulada alcanza p
n_kept = 0
cum = 0.0
for prob in sorted_probs:
    n_kept += 1
    cum += float(prob)
    if cum >= p_threshold:
        break

surviving_idx = order_desc[:n_kept]
surviving_tokens = [tokens[i] for i in surviving_idx]
surviving_probs = probs_T1[surviving_idx]
renorm = surviving_probs / np.sum(surviving_probs)

# ----- Parte 3: 10 000 muestras con numpy.random.default_rng(0) -----
rng = np.random.default_rng(seed)
sample_idx = rng.choice(len(surviving_tokens), size=n_samples, p=renorm)
counts = np.bincount(sample_idx, minlength=len(surviving_tokens))
empirical = counts / float(n_samples)

# ----- Divergencia KL en bits: D_KL(empírica || teórica), solo tokens supervivientes -----
mask = empirical > 0
kl_bits = float(np.sum(empirical[mask] * np.log2(empirical[mask] / renorm[mask])))
kl_bits_4dec = round(kl_bits, 4)

# ----- Gráfico de barras agrupadas: observado vs teórico -----
x = np.arange(len(surviving_tokens))
width = 0.38
fig, ax = plt.subplots(figsize=(8, 5.5))
b1 = ax.bar(x - width / 2, empirical, width,
            label="Frecuencia observada (10 000 muestras)",
            color="#4C72B0", edgecolor="black")
b2 = ax.bar(x + width / 2, renorm, width,
            label="Probabilidad teórica (top-p renormalizada)",
            color="#DD8452", edgecolor="black")
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.008, f"{h:.4f}",
                ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad / Frecuencia relativa")
ax.set_title("Parte 3 — Top-p (T=1, p=0.9): frecuencias observadas vs probabilidades teóricas\n"
             f"D_KL(empírica || teórica) = {kl_bits_4dec:.4f} bits")
ax.set_xticks(x)
ax.set_xticklabels(surviving_tokens)
ax.set_ylim(0, max(float(np.max(empirical)), float(np.max(renorm))) * 1.18)
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("parte3_top_p_empirico.png", dpi=120)
plt.close(fig)

# ----- Resultados -----
results = {
    "logits": {tokens[i]: float(logits[i]) for i in range(len(tokens))},
    "temperature": float(T),
    "p_threshold": float(p_threshold),
    "n_samples": int(n_samples),
    "rng_seed": int(seed),
    "softmax_T1_full": {tokens[i]: float(probs_T1[i]) for i in range(len(tokens))},
    "sorted_tokens_desc": sorted_tokens,
    "sorted_probs_desc": {sorted_tokens[i]: float(sorted_probs[i]) for i in range(len(sorted_tokens))},
    "cumulative_probs_desc": {sorted_tokens[i]: float(cumulative[i]) for i in range(len(sorted_tokens))},
    "n_tokens_kept": int(n_kept),
    "cut_reached_at_token": sorted_tokens[n_kept - 1],
    "cumulative_at_cut": float(cumulative[n_kept - 1]),
    "surviving_tokens": surviving_tokens,
    "renormalized_distribution_full": {surviving_tokens[i]: float(renorm[i]) for i in range(len(surviving_tokens))},
    "renormalized_distribution_4dec": {surviving_tokens[i]: round(float(renorm[i]), 4) for i in range(len(surviving_tokens))},
    "empirical_counts": {surviving_tokens[i]: int(counts[i]) for i in range(len(surviving_tokens))},
    "empirical_frequencies_full": {surviving_tokens[i]: float(empirical[i]) for i in range(len(surviving_tokens))},
    "empirical_frequencies_4dec": {surviving_tokens[i]: round(float(empirical[i]), 4) for i in range(len(surviving_tokens))},
    "kl_divergence_bits": kl_bits,
    "kl_divergence_bits_4dec": kl_bits_4dec,
}

with open("resultados.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
