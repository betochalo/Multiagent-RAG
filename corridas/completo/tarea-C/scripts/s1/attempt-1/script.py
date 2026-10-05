# -*- coding: utf-8 -*-
"""
Tarea C — Parte 1: Línea base léxica.
TF-IDF (TfidfVectorizer de scikit-learn, parámetros por defecto) + similitud coseno
sobre el corpus de 10 documentos y las 6 consultas del enunciado.
Escribe resultados en resultados.json e imprime rankings y métricas.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# Corpus d01–d10 (textos exactos del enunciado)
# ----------------------------------------------------------------------
CORPUS = {
    "d01": "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión.",
    "d02": "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización.",
    "d03": "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo.",
    "d04": "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento.",
    "d05": "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación.",
    "d06": "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado.",
    "d07": "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos.",
    "d08": "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales.",
    "d09": "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad.",
    "d10": "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria.",
}

# ----------------------------------------------------------------------
# Consultas q1–q6 y juicios de relevancia (exactos del enunciado)
# ----------------------------------------------------------------------
QUERIES = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

doc_ids = list(CORPUS.keys())
doc_texts = [CORPUS[d] for d in doc_ids]

# ----------------------------------------------------------------------
# Vectorización TF-IDF (parámetros por defecto) y similitud coseno
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, sin ajuste
X_docs = vectorizer.fit_transform(doc_texts)
X_queries = vectorizer.transform([QUERIES[q][0] for q in QUERIES])
S = cosine_similarity(X_queries, X_docs)  # forma (6, 10)

# ----------------------------------------------------------------------
# Evaluación por consulta: ranking completo y posición del relevante
# ----------------------------------------------------------------------
per_query = {}
hits1, hits3, rrs = [], [], []

print("=" * 78)
print("Parte 1 — Línea base léxica: TF-IDF (defecto) + similitud coseno")
print("=" * 78)

for i, qid in enumerate(QUERIES):
    qtext, relevant = QUERIES[qid]
    sims = S[i]
    order = np.argsort(-sims, kind="stable")  # descendente, determinista
    ranking = [(doc_ids[j], float(sims[j])) for j in order]
    rank = [d for d, _ in ranking].index(relevant) + 1  # posición 1-based

    hit1 = 1.0 if rank == 1 else 0.0
    hit3 = 1.0 if rank <= 3 else 0.0
    rr = 1.0 / rank
    hits1.append(hit1)
    hits3.append(hit3)
    rrs.append(rr)

    print(f"\n{qid}: \"{qtext}\"  (relevante: {relevant})")
    for pos, (did, sim) in enumerate(ranking, start=1):
        mark = "  <-- relevante" if did == relevant else ""
        print(f"  {pos:2d}. {did}  sim={sim:.6f}{mark}")
    print(f"  Posición del relevante {relevant}: {rank} | Hit@1={int(hit1)} Hit@3={int(hit3)} RR={rr:.4f}")

    per_query[qid] = {
        "consulta": qtext,
        "relevante": relevant,
        "rank_relevante": int(rank),
        "hit_at_1": float(hit1),
        "hit_at_3": float(hit3),
        "reciprocal_rank": float(rr),
        "ranking": [
            {"pos": int(p), "id": d, "similitud": float(s)}
            for p, (d, s) in enumerate(ranking, start=1)
        ],
    }

# ----------------------------------------------------------------------
# Métricas globales sobre las 6 consultas
# ----------------------------------------------------------------------
hit_at_1 = float(np.mean(hits1))
hit_at_3 = float(np.mean(hits3))
mrr = float(np.mean(rrs))

print("\n" + "=" * 78)
print("Métricas globales de la línea base léxica (6 consultas)")
print("=" * 78)
print(f"Hit@1 = {hit_at_1:.4f}")
print(f"Hit@3 = {hit_at_3:.4f}")
print(f"MRR   = {mrr:.4f}")

results = {
    "metodo": "TF-IDF (TfidfVectorizer con parametros por defecto) + similitud coseno",
    "num_documentos": len(doc_ids),
    "num_consultas": len(QUERIES),
    "metricas_globales": {
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "mrr": mrr,
    },
    "por_consulta": per_query,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nResultados escritos en resultados.json")
