# -*- coding: utf-8 -*-
"""
Subtask s1 — Línea base léxica TF-IDF.
Corpus de 10 documentos, 6 consultas con juicio de relevancia.
TF-IDF (TfidfVectorizer por defecto) + similitud coseno.
Métricas: Hit@1, Hit@3, MRR sobre las 6 consultas.
Resultados -> resultados.json
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# Datos (reproducidos literalmente del enunciado)
# ----------------------------------------------------------------------
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

doc_ids = list(corpus.keys())
doc_texts = [corpus[d] for d in doc_ids]

# Consultas: id -> (texto, documento relevante)
queries = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

query_ids = list(queries.keys())
query_texts = [queries[q][0] for q in query_ids]
relevant = {q: queries[q][1] for q in query_ids}

# ----------------------------------------------------------------------
# Vectorización TF-IDF (parámetros por defecto) y similitud coseno
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer()          # todos los parámetros por defecto
X_docs = vectorizer.fit_transform(doc_texts)      # fit SOLO sobre el corpus
X_queries = vectorizer.transform(query_texts)     # las consultas solo se transforman

S = cosine_similarity(X_queries, X_docs)          # forma (6, 10)

# ----------------------------------------------------------------------
# Ranking y evaluación
# ----------------------------------------------------------------------
hits1_count = 0
hits3_count = 0
rr_sum = 0.0
per_query = {}

print("=" * 78)
print("Ranking completo TF-IDF (similitud coseno) por consulta")
print("=" * 78)

for i, q in enumerate(query_ids):
    sims = S[i]
    # orden descendente por similitud; desempates por orden de documento (estable)
    order = np.argsort(-sims, kind="stable")
    ranked_ids = [doc_ids[j] for j in order]
    rank = ranked_ids.index(relevant[q]) + 1

    h1 = 1.0 if rank == 1 else 0.0
    h3 = 1.0 if rank <= 3 else 0.0
    rr = 1.0 / rank

    hits1_count += h1
    hits3_count += h3
    rr_sum += rr

    per_query[q] = {
        "consulta": queries[q][0],
        "relevante": relevant[q],
        "rank_relevante": float(rank),
        "hit_at_1": float(h1),
        "hit_at_3": float(h3),
        "reciprocal_rank": float(rr),
        "ranking": [
            {"pos": float(p + 1), "doc": ranked_ids[p], "similitud": float(sims[doc_ids.index(ranked_ids[p])])}
            for p in range(len(ranked_ids))
        ],
    }

    print(f"\n{q}: \"{queries[q][0]}\"   [relevante: {relevant[q]}]")
    for p, j in enumerate(order, start=1):
        marker = "  <-- relevante" if doc_ids[j] == relevant[q] else ""
        print(f"   {p:2d}. {doc_ids[j]}  sim={sims[j]:.4f}{marker}")
    print(f"   rank del relevante = {rank} | Hit@1={h1:.0f} | Hit@3={h3:.0f} | RR={rr:.4f}")

n_queries = len(query_ids)
hit_at_1 = hits1_count / n_queries
hit_at_3 = hits3_count / n_queries
mrr = rr_sum / n_queries

print("\n" + "=" * 78)
print("Métricas TF-IDF (línea base léxica) sobre las 6 consultas")
print("=" * 78)
print(f"Hit@1 : {hit_at_1:.4f}  ({int(hits1_count)}/{n_queries})")
print(f"Hit@3 : {hit_at_3:.4f}  ({int(hits3_count)}/{n_queries})")
print(f"MRR   : {mrr:.4f}")

# ----------------------------------------------------------------------
# Guardar resultados
# ----------------------------------------------------------------------
results = {
    "metodo": "TF-IDF (TfidfVectorizer, parámetros por defecto) + similitud coseno",
    "num_documentos": float(len(doc_ids)),
    "num_consultas": float(n_queries),
    "hit_at_1": float(hit_at_1),
    "hit_at_3": float(hit_at_3),
    "mrr": float(mrr),
    "hits1_sobre_6": float(hits1_count),
    "hits3_sobre_6": float(hits3_count),
    "por_consulta": per_query,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nResultados escritos en resultados.json")
