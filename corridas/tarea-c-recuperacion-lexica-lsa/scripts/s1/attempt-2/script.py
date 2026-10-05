# -*- coding: utf-8 -*-
"""
Subtask s1 - Lexical TF-IDF baseline (Parte 1 of Tarea C).

Corpus of 10 documents (d01-d10) and 6 queries (q1-q6) with relevance
judgments, all written literally in this script. Fits TfidfVectorizer()
from scikit-learn with ALL default parameters (no stop words, unigrams
only) on the 10 documents, transforms the 6 queries with that same
fitted vectorizer, computes cosine similarity between each query and
each document, ranks the 10 documents per query, and reports Hit@1,
Hit@3 and MRR over the 6 queries.

Results are written to resultados.json (relative path, current folder).
No figure is required for this subtask.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------
# Data (literal, from the statement)
# ------------------------------------------------------------------
docs = {
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

# query id -> (query text, relevant document id)
queries = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

doc_ids = list(docs.keys())
doc_texts = [docs[did] for did in doc_ids]
query_ids = list(queries.keys())
query_texts = [queries[qid][0] for qid in query_ids]
relevant = {qid: queries[qid][1] for qid in query_ids}

# ------------------------------------------------------------------
# TF-IDF with default parameters: fit on the 10 documents only,
# then transform the queries with the same fitted vectorizer.
# ------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # all defaults: no stop words, unigrams only
X_docs = vectorizer.fit_transform(doc_texts)
X_queries = vectorizer.transform(query_texts)

# Cosine similarity between each query and each document
sims = cosine_similarity(X_queries, X_docs)  # shape (6, 10)

# ------------------------------------------------------------------
# Ranking, per-query report and metrics
# ------------------------------------------------------------------
per_query = {}
hits1_count = 0
hits3_count = 0
reciprocal_sum = 0.0

print("=" * 78)
print("TF-IDF baseline (TfidfVectorizer default params) + cosine similarity")
print("Vocabulary size:", len(vectorizer.vocabulary_))
print("=" * 78)

for i, qid in enumerate(query_ids):
    qtext = query_texts[i]
    rel = relevant[qid]

    # Sort the 10 documents by similarity, descending (stable for ties)
    order = np.argsort(-sims[i], kind="stable")
    ranking = [doc_ids[j] for j in order]
    rank = ranking.index(rel) + 1  # 1-based rank of the relevant document

    h1 = 1 if rank == 1 else 0
    h3 = 1 if rank <= 3 else 0
    hits1_count += h1
    hits3_count += h3
    reciprocal_sum += 1.0 / rank

    print()
    print(f'{qid}: "{qtext}"   [relevante: {rel}]')
    print("  Ranking completo (mayor a menor similitud):")
    for pos, j in enumerate(order, start=1):
        marker = "  <-- relevante" if doc_ids[j] == rel else ""
        print(f"    {pos:2d}. {doc_ids[j]}  sim={sims[i][j]:.4f}{marker}")
    print(f"  Rango del relevante ({rel}): {rank}")

    per_query[qid] = {
        "consulta": qtext,
        "relevante": rel,
        "ranking": ranking,
        "rango_relevante": int(rank),
        "similitud_relevante": float(sims[i][doc_ids.index(rel)]),
        "similitudes": {doc_ids[j]: float(sims[i][j]) for j in order},
        "hit_at_1": int(h1),
        "hit_at_3": int(h3),
        "reciprocal_rank": float(1.0 / rank),
    }

n_queries = len(query_ids)
hit_at_1 = float(hits1_count) / n_queries
hit_at_3 = float(hits3_count) / n_queries
mrr = reciprocal_sum / n_queries

print()
print("=" * 78)
print("Hit@1:", hit_at_1, f"({hits1_count} de {n_queries} consultas)")
print("Hit@3:", hit_at_3, f"({hits3_count} de {n_queries} consultas)")
print("MRR  :", mrr)
print("=" * 78)

# ------------------------------------------------------------------
# Write results to resultados.json (current folder)
# ------------------------------------------------------------------
results = {
    "metodo": "TF-IDF (TfidfVectorizer con parametros por defecto) + similitud coseno",
    "num_documentos": len(doc_ids),
    "num_consultas": n_queries,
    "vocabulario_tamano": int(len(vectorizer.vocabulary_)),
    "por_consulta": per_query,
    "hit_at_1": hit_at_1,
    "hit_at_3": hit_at_3,
    "mrr": mrr,
    "hit_at_1_aciertos": int(hits1_count),
    "hit_at_3_aciertos": int(hits3_count),
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Resultados escritos en resultados.json")
