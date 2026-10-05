import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================================================================
# Parte 2 (recomputada): softmax estable con T = 1 y top-p con p=0.9
# ==================================================================
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
T = 1.0
p_threshold = 0.9

# Softmax numéricamente estable: restar el máximo antes de exponenciar
z = logits / T
z_shift = z - np.max(z)
exp_z = np.exp(z_shift)
probs_full = exp_z / np.sum(exp_z)

# Top-p: ordenar descendente, conjunto mínimo con acumulada >= p, renormalizar
order = np.argsort(-probs_full)
sorted_tokens = [tokens[i] for i in order]
sorted_probs = probs_full[order]
cumulative = np.cumsum(sorted_probs)
k = int(np.argmax(cumulative >= p_threshold)) + 1  # tamaño mínimo del núcleo

nucleus_tokens = sorted_tokens[:k]
nucleus_probs = sorted_probs[:k]
renorm_sorted = nucleus_probs / np.sum(nucleus_probs)
renorm = {tok: float(pr) for tok, pr in zip(nucleus_tokens, renorm_sorted)}
excluded_tokens = sorted_tokens[k:]

# Probabilidades teóricas del núcleo (alineadas con nucleus_tokens)
p_theory = np.array([renorm[t] for t in nucleus_tokens], dtype=float)

# ==================================================================
# Parte 3: muestreo empírico (10 000 muestras, default_rng(0))
# ==================================================================
rng = np.random.default_rng(0)
n_samples = 10000
draws = rng.choice(len(nucleus_tokens), size=n_samples, p=p_theory)
counts = np.bincount(draws, minlength=len(nucleus_tokens))
q_emp = counts / n_samples

# KL en bits: KL(q || p) = sum_i q_i * log2(q_i / p_i), con 0*log(0) := 0
mask = q_emp > 0
kl_bits = float(np.sum(q_emp[mask] * np.log2(q_emp[mask] / p_theory[mask])))

# ==================================================================
# Gráfico de barras agrupadas: probabilidad teórica vs frecuencia empírica
# ==================================================================
x = np.arange(len(nucleus_tokens))
width = 0.38
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width / 2, p_theory, width,
       label="Probabilidad teórica (top-p renormalizada)", color="#1f77b4", alpha=0.9)
ax.bar(x + width / 2, q_emp, width,
       label=f"Frecuencia empírica ({n_samples} muestras)", color="#ff7f0e", alpha=0.9)
for xi, pt, qe in zip(x, p_theory, q_emp):
    ax.text(xi - width / 2, pt + 0.006, f"{pt:.4f}", ha="center", fontsize=8)
    ax.text(xi + width / 2, qe + 0.006, f"{qe:.4f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(nucleus_tokens)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad / frecuencia")
ax.set_title(f"Top-p (T=1, p={p_threshold}): teórica vs empírica  |  KL = {kl_bits:.6f} bits")
ax.set_ylim(0, max(p_theory.max(), q_emp.max()) * 1.18)
ax.legend(loc="upper right")
ax.text(0.02, 0.90, f"Excluidos del núcleo: {', '.join(excluded_tokens)}",
        transform=ax.transAxes, fontsize=9, va="top", color="dimgray")
plt.tight_layout()
plt.savefig("top_p_teorico_vs_empirico.png", dpi=120)
plt.close(fig)

# ==================================================================
# Resultados
# ==================================================================
results = {
    "tokens": tokens,
    "logits": [float(v) for v in logits],
    "temperature": float(T),
    "p_threshold": float(p_threshold),
    "softmax_T1_full_distribution": {t: float(pr) for t, pr in zip(tokens, probs_full)},
    "sorted_tokens_desc": sorted_tokens,
    "sorted_probs_desc": [float(v) for v in sorted_probs],
    "cumulative_probs_desc": {t: float(c) for t, c in zip(sorted_tokens, cumulative)},
    "num_surviving_tokens": int(k),
    "surviving_tokens": nucleus_tokens,
    "excluded_tokens": excluded_tokens,
    "renormalized_distribution": renorm,
    "rng_seed": 0,
    "n_samples": int(n_samples),
    "observed_counts": {t: int(c) for t, c in zip(nucleus_tokens, counts)},
    "observed_frequencies": {t: float(q) for t, q in zip(nucleus_tokens, q_emp)},
    "theoretical_probs_nucleus": {t: float(pr) for t, pr in zip(nucleus_tokens, p_theory)},
    "kl_empirical_vs_theoretical_bits": kl_bits,
}

with open("resultados.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
