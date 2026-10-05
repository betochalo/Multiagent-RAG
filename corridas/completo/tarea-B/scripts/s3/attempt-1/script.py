# -*- coding: utf-8 -*-
"""
Subtask s3 (Parte 3 — Comprobación empírica).

Recomputa la distribución renormalizada top-p de la Parte 2
(softmax numéricamente estable con T = 1 sobre los logits del enunciado,
top-p con p = 0.9: conjunto mínimo con probabilidad acumulada >= 0.9,
renormalizado), toma exactamente 10 000 muestras con
numpy.random.default_rng(0), compara las frecuencias observadas con las
probabilidades teóricas en un gráfico de barras agrupadas y calcula la
divergencia KL en bits de la distribución empírica respecto de la teórica
sobre los tokens del núcleo.

Resultados -> resultados.json
Figura      -> parte3_comparacion_barras.png
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Datos del enunciado (embedidos literalmente)
# ----------------------------------------------------------------------
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
tokens = np.array(["t0", "t1", "t2", "t3", "t4", "t5"])
T = 1.0            # temperatura de la Parte 1
p_threshold = 0.9  # umbral top-p de la Parte 2
n_samples = 10000  # número exacto de muestras de la Parte 3
seed = 0           # semilla exigida: numpy.random.default_rng(0)


# ----------------------------------------------------------------------
# Parte 1 (recomputada): softmax con temperatura, numéricamente estable
# ----------------------------------------------------------------------
def softmax_with_temperature(z, temp):
    z = np.asarray(z, dtype=float)
    z_shift = (z - np.max(z)) / temp  # resta del máximo -> estabilidad numérica
    e = np.exp(z_shift)
    return e / np.sum(e)


probs_T1 = softmax_with_temperature(logits, T)

# ----------------------------------------------------------------------
# Parte 2 (recomputada): top-p con p = 0.9 y renormalización
# ----------------------------------------------------------------------
order_desc = np.argsort(-probs_T1, kind="stable")
probs_sorted = probs_T1[order_desc]
cumulative = np.cumsum(probs_sorted)

# conjunto MÍNIMO de tokens con probabilidad acumulada >= p
k = int(np.searchsorted(cumulative, p_threshold, side="left")) + 1
k = min(k, len(tokens))

nucleus_idx = np.sort(order_desc[:k])  # núcleo en orden original de tokens
nucleus_tokens = tokens[nucleus_idx]
nucleus_probs = probs_T1[nucleus_idx]
renorm = nucleus_probs / np.sum(nucleus_probs)  # distribución renormalizada

excluded_tokens = np.setdiff1d(tokens, nucleus_tokens)

# ----------------------------------------------------------------------
# Parte 3: 10 000 muestras con numpy.random.default_rng(0)
# ----------------------------------------------------------------------
rng = np.random.default_rng(seed)
samples = rng.choice(nucleus_tokens, size=n_samples, p=renorm)

unique_labels, counts_int = np.unique(samples, return_counts=True)
count_map = {str(t): int(c) for t, c in zip(unique_labels, counts_int)}
counts = np.array([count_map.get(str(t), 0) for t in nucleus_tokens], dtype=float)
obs_freq = counts / float(n_samples)

# ----------------------------------------------------------------------
# Divergencia KL en bits, fórmula del enunciado:
# KL = sum p_teorica * log2(p_teorica / p_observada) sobre el núcleo
# (con 10 000 muestras y p_min ~ 0.094 todos los tokens del núcleo
#  aparecen; se excluyen términos con frecuencia observada nula por seguridad)
# ----------------------------------------------------------------------
mask = obs_freq > 0
kl_bits = float(np.sum(renorm[mask] * np.log2(renorm[mask] / obs_freq[mask])))

# ----------------------------------------------------------------------
# Gráfico de barras agrupadas: observado vs teórico
# ----------------------------------------------------------------------
x = np.arange(len(nucleus_tokens))
width = 0.38
fig, ax = plt.subplots(figsize=(8.5, 5.2))
bars_obs = ax.bar(x - width / 2.0, obs_freq, width,
                  label="Frecuencia relativa observada (10 000 muestras)",
                  color="#4C72B0", edgecolor="black", linewidth=0.5)
bars_teo = ax.bar(x + width / 2.0, renorm, width,
                  label="Probabilidad teórica (top-p renormalizada)",
                  color="#DD8452", edgecolor="black", linewidth=0.5)
for bars in (bars_obs, bars_teo):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.4f}",
                    (b.get_x() + b.get_width() / 2.0, h),
                    ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(nucleus_tokens)
ax.set_xlabel("Token del núcleo (top-p, p = 0.9)")
ax.set_ylabel("Probabilidad")
ax.set_title("Parte 3 — Frecuencias observadas vs. probabilidad teórica\n"
             f"softmax T = 1, top-p p = 0.9, {n_samples} muestras, semilla {seed}")
ax.set_ylim(0.0, 1.15 * float(max(renorm.max(), obs_freq.max())))
ax.text(0.02, 0.97, f"Tokens excluidos por top-p: {', '.join(map(str, excluded_tokens))}",
        transform=ax.transAxes, fontsize=9, va="top", color="0.35")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("parte3_comparacion_barras.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# Resultados -> resultados.json (y stdout)
# ----------------------------------------------------------------------
results = {
    "logits": [float(v) for v in logits],
    "temperature": float(T),
    "p_threshold": float(p_threshold),
    "softmax_T1_full": {str(t): float(p) for t, p in zip(tokens, probs_T1)},
    "sorted_order_desc": [str(t) for t in tokens[order_desc]],
    "sorted_probs_desc": [float(v) for v in probs_sorted],
    "cumulative_probs_desc": [float(v) for v in cumulative],
    "nucleus_size": int(k),
    "surviving_tokens": [str(t) for t in nucleus_tokens],
    "excluded_tokens": [str(t) for t in excluded_tokens],
    "nucleus_cumulative_prob": float(cumulative[k - 1]),
    "renormalized_distribution": {str(t): float(p) for t, p in zip(nucleus_tokens, renorm)},
    "n_samples": int(n_samples),
    "rng_seed": int(seed),
    "observed_counts": {str(t): int(count_map.get(str(t), 0)) for t in nucleus_tokens},
    "observed_relative_frequencies": {str(t): float(f) for t, f in zip(nucleus_tokens, obs_freq)},
    "kl_empirical_vs_theoretical_bits": kl_bits,
    "kl_empirical_vs_theoretical_bits_6dp": float(round(kl_bits, 6)),
    "figure": "parte3_comparacion_barras.png",
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Distribución renormalizada top-p (T = 1, p = 0.9) y comprobación empírica:")
for t, pt, fo, c in zip(nucleus_tokens, renorm, obs_freq, counts):
    print(f"  {t}: p_teorica = {pt:.6f} | frec_observada = {fo:.6f} | conteo = {int(c)}")
print(f"\nTokens supervivientes: {[str(t) for t in nucleus_tokens]}")
print(f"Tokens excluidos: {[str(t) for t in excluded_tokens]}")
print(f"Probabilidad acumulada del núcleo: {cumulative[k - 1]:.6f}")
print(f"KL(empírica || teórica) = {kl_bits:.6f} bits "
      f"(≈ {kl_bits:.2e} bits, pequeña como se espera con {n_samples} muestras)")
print("\nResultados JSON:")
print(json.dumps(results, indent=2))
