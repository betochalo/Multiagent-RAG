import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Subtask s3 (Parte 3) — Comprobación empírica del muestreo top-p
# Recomputa desde cero: softmax estable (T=1), top-p (p=0.9) renormalizado,
# 10,000 muestras con numpy.random.default_rng(0), frecuencias empíricas,
# gráfico de barras comparativo y divergencia KL en bits.
# ----------------------------------------------------------------------

tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
T = 1.0
p_nucleo = 0.9
n_muestras = 10000
semilla = 0

# --- 1) Softmax numéricamente estable a T = 1.0 ------------------------
z = logits / T
z_shift = z - np.max(z)
exp_z = np.exp(z_shift)
probs_softmax = exp_z / np.sum(exp_z)

# --- 2) Top-p: conjunto mínimo que alcanza p=0.9, renormalizado --------
orden = np.argsort(-probs_softmax)            # orden descendente
probs_ordenadas = probs_softmax[orden]
acumulada = np.cumsum(probs_ordenadas)

k = 1
while k < len(acumulada) and acumulada[k - 1] < p_nucleo:
    k += 1
# k = número mínimo de tokens cuya probabilidad acumulada alcanza 0.9

idx_sobrevivientes = np.sort(orden[:k])
probs_sobrevivientes = probs_softmax[idx_sobrevivientes]
probs_renorm = probs_sobrevivientes / np.sum(probs_sobrevivientes)

# Distribución teórica renormalizada sobre el vocabulario completo
teo_full = np.zeros(len(tokens), dtype=float)
teo_full[idx_sobrevivientes] = probs_renorm

# --- 3) Muestreo: 10,000 muestras con default_rng(0) -------------------
rng = np.random.default_rng(semilla)
muestras = rng.choice(idx_sobrevivientes, size=n_muestras, p=probs_renorm)

conteos = np.bincount(muestras, minlength=len(tokens))
emp_full = conteos / float(n_muestras)

# --- 4) KL en bits: D(emp || teo) = sum_i q_i log2(q_i / p_i), q_i > 0 --
mask = emp_full > 0
kl_bits = float(np.sum(emp_full[mask] * np.log2(emp_full[mask] / teo_full[mask])))

# --- 5) Gráfico de barras: observado vs teórico, lado a lado ------------
x = np.arange(len(tokens))
ancho = 0.38
fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - ancho / 2, emp_full, ancho,
            label="Frecuencia observada (empírica)", color="#4C72B0")
b2 = ax.bar(x + ancho / 2, teo_full, ancho,
            label="Probabilidad teórica (top-p renormalizada)", color="#DD8452")
for rect, v in zip(b1, emp_full):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.006, f"{v:.4f}",
            ha="center", fontsize=8)
for rect, v in zip(b2, teo_full):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.006, f"{v:.4f}",
            ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad / Frecuencia")
ax.set_title(
    f"Parte 3 — Observado vs teórico (softmax T={T}, top-p p={p_nucleo}, "
    f"{n_muestras} muestras, semilla {semilla})\n"
    f"KL(emp || teo) = {kl_bits:.6f} bits"
)
ax.legend()
ax.set_ylim(0, max(emp_full.max(), teo_full.max()) * 1.18)
fig.tight_layout()
fig.savefig("parte3_frecuencias_vs_teoricas.png", dpi=120)
plt.close(fig)

# --- 6) Resultados a JSON y stdout --------------------------------------
resultados = {
    "logits": {t: float(v) for t, v in zip(tokens, logits)},
    "temperatura": float(T),
    "p_top_p": float(p_nucleo),
    "softmax_T_1.0": {t: float(v) for t, v in zip(tokens, probs_softmax)},
    "orden_descendente": [tokens[i] for i in orden],
    "probabilidad_acumulada_ordenada": {tokens[i]: float(v)
                                        for i, v in zip(orden, acumulada)},
    "tokens_sobrevivientes": [tokens[i] for i in idx_sobrevivientes],
    "num_tokens_conservados": int(k),
    "probabilidad_acumulada_nucleo": float(acumulada[k - 1]),
    "distribucion_renormalizada": {tokens[i]: float(v)
                                   for i, v in zip(idx_sobrevivientes, probs_renorm)},
    "distribucion_teorica_vocabulario_completo": {t: float(v)
                                                  for t, v in zip(tokens, teo_full)},
    "num_muestras": int(n_muestras),
    "semilla": int(semilla),
    "conteos_observados": {t: int(c) for t, c in zip(tokens, conteos)},
    "frecuencias_observadas": {t: float(v) for t, v in zip(tokens, emp_full)},
    "kl_bits_empirica_vs_teorica": kl_bits,
    "figura": "parte3_frecuencias_vs_teoricas.png",
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

print(json.dumps(resultados, indent=2))
