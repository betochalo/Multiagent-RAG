# -*- coding: utf-8 -*-
"""
Subtask s1 — Línea base léxica TF-IDF.
Corpus de 10 documentos y 6 consultas codificados literalmente (sin descargas).
TF-IDF con TfidfVectorizer() por defecto ajustado SOLO sobre los 10 documentos,
consultas transformadas con el mismo vectorizador, similitud coseno,
ranking completo por consulta y métricas Hit@1, Hit@3 y MRR.
Resultados -> resultados.json
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------
# Datos codificados en el script (enunciado, sin descargas)
# ------------------------------------------------------------------
corpus = {
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

# id_consulta: (texto, documento relevante)
queries = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

doc_ids = list(corpus.keys())
doc_texts = [corpus[d] for d in doc_ids]
query_ids = list(queries.keys())
query_texts = [queries[q][0] for q in query_ids]

# ------------------------------------------------------------------
# TF-IDF con parámetros por defecto: fit SOLO sobre los documentos,
# las consultas se transforman con el mismo vectorizador.
# ------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto
X_docs = vectorizer.fit_transform(doc_texts)
X_queries = vectorizer.transform(query_texts)

# Similitud coseno consulta-documento, forma (6, 10)
sims = cosine_similarity(X_queries, X_docs)

# ------------------------------------------------------------------
# Ranking completo por consulta + métricas
# ------------------------------------------------------------------
rankings = {}
per_query = {}
hits1 = 0
hits3 = 0
reciprocal_ranks = []

print("=" * 78)
print("Ranking TF-IDF (similitud coseno) de los 10 documentos por consulta")
print("=" * 78)

for i, qid in enumerate(query_ids):
    relevant = queries[qid][1]
    # Orden descendente por similitud; desempates estables por orden del corpus
    order = np.argsort(-sims[i], kind="stable")
    ranked = [(doc_ids[j], float(sims[i, j])) for j in order]
    rankings[qid] = ranked

    # Rango (1-based) del documento relevante
    rank_of_relevant = next(pos + 1 for pos, (d, _) in enumerate(ranked) if d == relevant)
    rr = 1.0 / rank_of_relevant
    reciprocal_ranks.append(rr)
    if rank_of_relevant == 1:
        hits1 += 1
    if rank_of_relevant <= 3:
        hits3 += 1

    per_query[qid] = {
        "consulta": query_texts[i],
        "relevante": relevant,
        "rank_relevante": int(rank_of_relevant),
        "reciprocal_rank": float(rr),
        "hit_at_1": bool(rank_of_relevant == 1),
        "hit_at_3": bool(rank_of_relevant <= 3),
        "ranking": [{"pos": p + 1, "doc": d, "similitud_coseno": s}
                    for p, (d, s) in enumerate(ranked)],
    }

    print(f"\n{qid}: \"{query_texts[i]}\"  (relevante: {relevant})")
    for p, (d, s) in enumerate(ranked):
        marker = "  <-- relevante" if d == relevant else ""
        print(f"  {p + 1:2d}. {d}  sim={s:.6f}{marker}")
    print(f"  rank del relevante: {rank_of_relevant} | RR={rr:.4f}")

n_queries = len(query_ids)
hit_at_1 = hits1 / n_queries
hit_at_3 = hits3 / n_queries
mrr = float(np.mean(reciprocal_ranks))

print("\n" + "=" * 78)
print("Métricas sobre las 6 consultas (línea base TF-IDF)")
print("=" * 78)
print(f"Hit@1: {hit_at_1:.6f}  ({hits1}/{n_queries})")
print(f"Hit@3: {hit_at_3:.6f}  ({hits3}/{n_queries})")
print(f"MRR  : {mrr:.6f}")

# ------------------------------------------------------------------
# Guardar resultados
# ------------------------------------------------------------------
results = {
    "metodo": "TF-IDF (TfidfVectorizer por defecto) + similitud coseno",
    "n_documentos": len(doc_ids),
    "n_consultas": n_queries,
    "n_vocabulario": int(len(vectorizer.vocabulary_)),
    "hit_at_1": float(hit_at_1),
    "hit_at_3": float(hit_at_3),
    "mrr": float(mrr),
    "aciertos_hit1": int(hits1),
    "aciertos_hit3": int(hits3),
    "por_consulta": per_query,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nResultados escritos en resultados.json")
