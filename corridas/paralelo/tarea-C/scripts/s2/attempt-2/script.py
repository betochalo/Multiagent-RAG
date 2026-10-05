# -*- coding: utf-8 -*-
"""
Subtask s2 (Parte 2) — Semántica latente (LSA) comparada con la línea base TF-IDF.

Recomputa desde cero la Parte 1 (TfidfVectorizer por defecto + similitud coseno)
sobre el corpus d01-d10 y las consultas q1-q6, ajusta TruncatedSVD(n_components=4,
random_state=0) sobre la matriz TF-IDF de documentos, proyecta las consultas con
el mismo SVD, rankea los 10 documentos por consulta en ambos espacios y evalúa
Hit@1, Hit@3 y MRR sobre las 6 consultas.

Salida: resultados.json + una sola tabla comparativa (Hit@1, Hit@3, MRR de
TF-IDF y LSA) y el ranking por consulta de LSA en stdout.
"""

import json

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# 1. Datos del enunciado (corpus y consultas, embebidos literalmente)
# ----------------------------------------------------------------------------
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

consultas = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

doc_ids = list(corpus.keys())
doc_texts = [corpus[d] for d in doc_ids]
query_ids = list(consultas.keys())
query_texts = [consultas[q][0] for q in query_ids]
relevante = {q: consultas[q][1] for q in query_ids}

# ----------------------------------------------------------------------------
# 2. Línea base TF-IDF (Parte 1): fit sobre los 10 documentos, transform consultas
# ----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto
X_docs = vectorizer.fit_transform(doc_texts)
X_queries = vectorizer.transform(query_texts)

# ----------------------------------------------------------------------------
# 3. LSA (Parte 2): SVD ajustado sobre la matriz de documentos,
#    consultas proyectadas al mismo espacio latente
# ----------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs)      # (10, 4)
X_queries_lsa = svd.transform(X_queries)    # (6, 4)

# ----------------------------------------------------------------------------
# 4. Similitud coseno (matriz DENSA) + ranking de los 10 documentos por consulta
# ----------------------------------------------------------------------------
def matriz_similitud(q_mat, d_mat):
    """Coseno entre cada consulta y cada documento; devuelve ndarray denso (6, 10)."""
    sims = cosine_similarity(q_mat, d_mat)
    return np.asarray(sims, dtype=float)


def rankear(sims):
    """Para cada consulta, lista ordenada (desc, estable) de (doc_id, similitud)."""
    rankings = {}
    for i, qid in enumerate(query_ids):
        fila = np.asarray(sims[i]).ravel()
        orden = np.argsort(-fila, kind="stable")
        rankings[qid] = [(doc_ids[j], float(fila[j])) for j in orden]
    return rankings


sims_tfidf = matriz_similitud(X_queries, X_docs)
sims_lsa = matriz_similitud(X_queries_lsa, X_docs_lsa)
ranking_tfidf = rankear(sims_tfidf)
ranking_lsa = rankear(sims_lsa)

# ----------------------------------------------------------------------------
# 5. Evaluación: Hit@1, Hit@3 y MRR sobre las 6 consultas
# ----------------------------------------------------------------------------
def evaluar(rankings):
    n = len(query_ids)
    hit1 = hit3 = 0
    rr_total = 0.0
    por_consulta = {}
    for qid in query_ids:
        rel = relevante[qid]
        docs_orden = [d for d, _ in rankings[qid]]
        assert len(docs_orden) == len(doc_ids), f"ranking incompleto en {qid}"
        assert rel in docs_orden, f"relevante {rel} ausente en ranking de {qid}"
        rank = docs_orden.index(rel) + 1
        h1 = rank == 1
        h3 = rank <= 3
        rr = 1.0 / rank
        hit1 += int(h1)
        hit3 += int(h3)
        rr_total += rr
        por_consulta[qid] = {
            "consulta": consultas[qid][0],
            "relevante": rel,
            "rank_relevante": int(rank),
            "reciprocal_rank": float(rr),
            "hit_at_1": bool(h1),
            "hit_at_3": bool(h3),
            "ranking": [
                {"pos": k + 1, "doc": d, "similitud_coseno": float(s)}
                for k, (d, s) in enumerate(rankings[qid])
            ],
        }
    metricas = {
        "hit_at_1": float(hit1 / n),
        "hit_at_3": float(hit3 / n),
        "mrr": float(rr_total / n),
    }
    return metricas, por_consulta


metricas_tfidf, por_consulta_tfidf = evaluar(ranking_tfidf)
metricas_lsa, por_consulta_lsa = evaluar(ranking_lsa)

# ----------------------------------------------------------------------------
# 6. UNA sola tabla comparativa + ranking por consulta de LSA (stdout)
# ----------------------------------------------------------------------------
print("Tabla comparativa (6 consultas, corpus d01-d10):")
print(f"{'Metodo':<10}{'Hit@1':>10}{'Hit@3':>10}{'MRR':>10}")
print("-" * 40)
print(
    f"{'TF-IDF':<10}"
    f"{metricas_tfidf['hit_at_1']:>10.4f}"
    f"{metricas_tfidf['hit_at_3']:>10.4f}"
    f"{metricas_tfidf['mrr']:>10.4f}"
)
print(
    f"{'LSA':<10}"
    f"{metricas_lsa['hit_at_1']:>10.4f}"
    f"{metricas_lsa['hit_at_3']:>10.4f}"
    f"{metricas_lsa['mrr']:>10.4f}"
)
print("\nValores exactos TF-IDF:", json.dumps(metricas_tfidf))
print("Valores exactos LSA   :", json.dumps(metricas_lsa))

print("\nRanking LSA por consulta (espacio latente de 4 dimensiones):")
for qid in query_ids:
    rel = relevante[qid]
    rank_rel = por_consulta_lsa[qid]["rank_relevante"]
    seq = " > ".join(f"{d}({s:.3f})" for d, s in ranking_lsa[qid])
    print(f"{qid} [relevante {rel}, rank {rank_rel}]: {seq}")

print("\nRank del relevante por consulta (TF-IDF vs LSA):")
for qid in query_ids:
    print(
        f"{qid}: TF-IDF rank {por_consulta_tfidf[qid]['rank_relevante']} | "
        f"LSA rank {por_consulta_lsa[qid]['rank_relevante']}"
    )

# ----------------------------------------------------------------------------
# 7. Resultados -> resultados.json (todos los números calculados)
# ----------------------------------------------------------------------------
resultados = {
    "metodo_tfidf": "TF-IDF (TfidfVectorizer por defecto) + similitud coseno",
    "metodo_lsa": "LSA (TruncatedSVD n_components=4, random_state=0) + similitud coseno",
    "n_documentos": int(len(doc_ids)),
    "n_consultas": int(len(query_ids)),
    "n_vocabulario": int(X_docs.shape[1]),
    "n_componentes_lsa": int(X_docs_lsa.shape[1]),
    "varianza_explicada_lsa": float(svd.explained_variance_ratio_.sum()),
    "tfidf": {
        "hit_at_1": metricas_tfidf["hit_at_1"],
        "hit_at_3": metricas_tfidf["hit_at_3"],
        "mrr": metricas_tfidf["mrr"],
        "por_consulta": por_consulta_tfidf,
    },
    "lsa": {
        "hit_at_1": metricas_lsa["hit_at_1"],
        "hit_at_3": metricas_lsa["hit_at_3"],
        "mrr": metricas_lsa["mrr"],
        "por_consulta": por_consulta_lsa,
    },
    "tabla_comparativa": {
        "TF-IDF": metricas_tfidf,
        "LSA": metricas_lsa,
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print("\nResultados escritos en resultados.json")
