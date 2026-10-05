import json
import numpy as np


def softmax_con_temperatura(z, T):
    """Softmax con temperatura p_i = exp(z_i/T) / sum_j exp(z_j/T),
    numéricamente estable (se resta el máximo antes de exponenciar)."""
    z = np.asarray(z, dtype=float)
    z_shift = (z - np.max(z)) / float(T)
    e = np.exp(z_shift)
    return e / np.sum(e)


# ---- Datos de la Tarea B (embebidos literalmente) ----
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)

# ---- Parte 1 recomputada: softmax estable con T = 1.0 ----
T = 1.0
probs = softmax_con_temperatura(logits, T)

# ---- Parte 2: muestreo de núcleo (top-p) con p = 0.9 ----
p_umbral = 0.9

# Ordenar los tokens por probabilidad en orden descendente
orden = np.argsort(-probs, kind="stable")
probs_ordenadas = probs[orden]
acumulada = np.cumsum(probs_ordenadas)

# Conjunto mínimo cuya probabilidad acumulada alcanza 0.9,
# incluyendo el token que cruza el umbral
alcanza = acumulada >= p_umbral
if np.any(alcanza):
    n_conservados = int(np.argmax(alcanza)) + 1
else:
    n_conservados = len(tokens)  # respaldo: la suma total es 1.0 >= 0.9

idx_nucleo = orden[:n_conservados]
tokens_sobrevivientes = [tokens[i] for i in idx_nucleo]
probs_nucleo = probs[idx_nucleo]
suma_nucleo = float(np.sum(probs_nucleo))

# Renormalizar las probabilidades del núcleo para que sumen 1
q_nucleo = probs_nucleo / suma_nucleo

# Vista sobre el vocabulario completo (los eliminados quedan con 0)
dist_final_vocabulario = {tok: 0.0 for tok in tokens}
for tok, v in zip(tokens_sobrevivientes, q_nucleo):
    dist_final_vocabulario[tok] = round(float(v), 4)

# ---- Empaquetar resultados (floats de Python) ----
resultados = {
    "logits": {tokens[i]: float(logits[i]) for i in range(len(tokens))},
    "temperatura": float(T),
    "p_top_p": float(p_umbral),
    "softmax_T_1.0": {tokens[i]: float(probs[i]) for i in range(len(tokens))},
    "softmax_T_1.0_4dec": {
        tokens[i]: round(float(probs[i]), 4) for i in range(len(tokens))
    },
    "orden_descendente": [tokens[i] for i in orden],
    "probabilidades_ordenadas_4dec": {
        tokens[int(orden[i])]: round(float(probs_ordenadas[i]), 4)
        for i in range(len(tokens))
    },
    "probabilidad_acumulada_4dec": {
        tokens[int(orden[i])]: round(float(acumulada[i]), 4)
        for i in range(len(tokens))
    },
    "tokens_sobrevivientes": tokens_sobrevivientes,
    "num_tokens_conservados": int(n_conservados),
    "probabilidad_acumulada_nucleo": suma_nucleo,
    "probabilidad_acumulada_nucleo_4dec": round(suma_nucleo, 4),
    "distribucion_renormalizada": {
        tok: round(float(v), 4) for tok, v in zip(tokens_sobrevivientes, q_nucleo)
    },
    "distribucion_renormalizada_precisa": {
        tok: float(v) for tok, v in zip(tokens_sobrevivientes, q_nucleo)
    },
    "suma_distribucion_renormalizada": float(np.sum(q_nucleo)),
    "distribucion_tras_top_p_vocabulario_completo": dist_final_vocabulario,
}

# ---- Escribir resultados.json y mostrar los mismos números en stdout ----
with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

print(json.dumps(resultados, indent=2))

print("\nResumen:")
print("Softmax T=1.0 (4 dec):",
      {t: round(float(p), 4) for t, p in zip(tokens, probs)})
print("Tokens sobrevivientes (top-p, p=0.9):", ", ".join(tokens_sobrevivientes))
print("Probabilidad acumulada del núcleo:", round(suma_nucleo, 4))
print("Distribución renormalizada (4 dec):",
      {t: round(float(v), 4) for t, v in zip(tokens_sobrevivientes, q_nucleo)})
