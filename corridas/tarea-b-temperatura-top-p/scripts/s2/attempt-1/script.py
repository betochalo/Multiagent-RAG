# -*- coding: utf-8 -*-
"""
Subtask s2 (Parte 2) — Muestreo de núcleo (top-p) sobre la distribución de T = 1.

Recomputa de forma independiente (sin depender de archivos de la Parte 1):
  1) la softmax con temperatura numéricamente estable en T = 1 a partir de los
     logits z = [2.0, 1.0, 0.5, 0.2, -1.0, -3.0] para los tokens t0..t5,
  2) el corte top-p con p = 0.9: ordena los tokens por probabilidad descendente,
     conserva el conjunto mínimo cuya probabilidad acumulada alcanza o supera 0.9,
  3) la renormalización de las probabilidades de los tokens sobrevivientes.

Escribe todos los números en resultados.json y los imprime en stdout.
No se requiere figura para este subtask.
"""

import json
import numpy as np

# ----------------------------------------------------------------------
# Datos de la statement (Tarea B), embebidos literalmente
# ----------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
T = 1.0          # temperatura de la Parte 1
p = 0.9          # umbral top-p de la Parte 2

# ----------------------------------------------------------------------
# Softmax con temperatura, numéricamente estable (misma implementación
# que la Parte 1: se resta el máximo de los logits antes de exponenciar)
# ----------------------------------------------------------------------
def softmax_with_temperature(z, temperature):
    z = np.asarray(z, dtype=float)
    z_shift = (z - np.max(z)) / temperature  # shift de estabilidad numérica
    e = np.exp(z_shift)
    return e / np.sum(e)

probs = softmax_with_temperature(logits, T)

# ----------------------------------------------------------------------
# Top-p (nucleus): ordenar por probabilidad descendente y acumular
# ----------------------------------------------------------------------
order = np.argsort(-probs, kind="stable")          # índices de mayor a menor prob.
sorted_tokens = [tokens[i] for i in order]
sorted_probs = probs[order]
cumulative = np.cumsum(sorted_probs)

# Conjunto mínimo cuya probabilidad acumulada alcanza o supera p
n_keep = 0
cum_at_cut = 0.0
for k in range(len(sorted_probs)):
    n_keep += 1
    cum_at_cut = float(cumulative[k])
    if cum_at_cut >= p:
        break

cum_before_cut = float(cumulative[n_keep - 2]) if n_keep >= 2 else 0.0
cut_token = sorted_tokens[n_keep - 1]

surviving_tokens = sorted_tokens[:n_keep]
excluded_tokens = sorted_tokens[n_keep:]
kept_probs = probs[order[:n_keep]]
renorm = kept_probs / kept_probs.sum()             # renormalización a suma 1

# ----------------------------------------------------------------------
# Reporte (floats planos, 4 decimales donde se pide)
# ----------------------------------------------------------------------
def d4(x):
    return float(np.round(float(x), 4))

results = {
    "logits": {tok: float(z) for tok, z in zip(tokens, logits)},
    "temperature": float(T),
    "p_threshold": float(p),
    "softmax_T1_full": {tok: float(pr) for tok, pr in zip(tokens, probs)},
    "softmax_T1_4dec": {tok: d4(pr) for tok, pr in zip(tokens, probs)},
    "sorted_tokens_desc": sorted_tokens,
    "sorted_probs_desc": {tok: float(pr) for tok, pr in zip(sorted_tokens, sorted_probs)},
    "cumulative_probs": {tok: float(c) for tok, c in zip(sorted_tokens, cumulative)},
    "cumulative_probs_4dec": {tok: d4(c) for tok, c in zip(sorted_tokens, cumulative)},
    "n_tokens_kept": int(n_keep),
    "cut_reached_at_token": cut_token,
    "cumulative_at_cut": cum_at_cut,
    "cumulative_before_cut": cum_before_cut,
    "surviving_tokens": surviving_tokens,
    "excluded_tokens": excluded_tokens,
    "renormalized_distribution_full": {tok: float(pr) for tok, pr in zip(surviving_tokens, renorm)},
    "renormalized_distribution_4dec": {tok: d4(pr) for tok, pr in zip(surviving_tokens, renorm)},
    "renormalized_sum_full": float(renorm.sum()),
    "renormalized_sum_4dec": float(sum(d4(pr) for pr in renorm)),
}

with open("resultados.json", "w") as f:
    json.dump(results, f, indent=2)

# ----------------------------------------------------------------------
# Salida por stdout (mismos números)
# ----------------------------------------------------------------------
print("Logits:", results["logits"])
print()
print(f"Softmax con temperatura (numerically stable) en T = {T}:")
for tok in tokens:
    print(f"  {tok}: p = {results['softmax_T1_4dec'][tok]:.4f}   (full: {results['softmax_T1_full'][tok]:.16f})")
print()
print(f"Top-p con p = {p}: tokens ordenados por probabilidad descendente")
print("  token |    p_i |  acumulada")
for tok in sorted_tokens:
    print(f"  {tok:>5} | {results['sorted_probs_desc'][tok]:.4f} | {results['cumulative_probs_4dec'][tok]:.4f}")
print()
print(f"Conjunto mínimo con probabilidad acumulada >= {p}:")
print(f"  acumulada antes del corte ({sorted_tokens[n_keep-2] if n_keep >= 2 else '-'}): "
      f"{cum_before_cut:.4f}  (< {p})")
print(f"  acumulada en el corte   ({cut_token}): {cum_at_cut:.4f}  (>= {p})")
print(f"  -> tokens sobrevivientes: {surviving_tokens}")
print(f"  -> tokens excluidos:      {excluded_tokens}")
print()
print("Distribución renormalizada sobre los tokens sobrevivientes (4 decimales):")
for tok in surviving_tokens:
    print(f"  {tok}: {results['renormalized_distribution_4dec'][tok]:.4f}")
print(f"  suma = {results['renormalized_sum_4dec']:.4f}")
print()
print("Resultados escritos en resultados.json")
