# -*- coding: utf-8 -*-
"""
Tarea C - Subtask s2 (Parte 2): Semántica latente (LSA) y tabla comparativa.

Recalcula, con los mismos datos y parámetros de la Parte 1:
  * Matriz TF-IDF del corpus d01-d10 y de las 6 consultas (TfidfVectorizer por defecto).
  * Línea base TF-IDF + similitud coseno (Hit@1, Hit@3, MRR).
  * LSA: TruncatedSVD(n_components=4, random_state=0) ajustado sobre la matriz
    TF-IDF del corpus; documentos y consultas proyectados al espacio latente de
    4 dimensiones; similitud coseno y ranking en ese espacio.
Produce UNA tabla comparativa (filas = métodos, columnas = Hit@1, Hit@3, MRR),
imprime el ranking LSA por consulta y escribe resultados.json.
"""

import json

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Corpus (textos literales del enunciado)
# ---------------------------------------------------------------------------
DOC_TEXTS = {
    "d01": ("El mecanismo de atención pondera cada token según su similitud con la "
            "consulta; la atención escalada divide por la raíz de la dimensión."),
    "d02": ("Los transformadores apilan capas de autoatención y redes feed-forward, "
            "con conexiones residuales y normalización."),
    "d03": ("La recuperación aumentada con generación busca fragmentos relevantes y "
            "los añade al prompt del modelo."),
    "d04": ("BM25 es una función de ranking léxica que pondera la frecuencia de "
            "términos y la longitud del documento."),
    "d05": ("Los embeddings densos representan textos como vectores; la similitud "
            "coseno compara su orientación."),
    "d06": ("Un agente con herramientas decide en cada paso qué función llamar y "
            "observa el resultado."),
    "d07": ("ReAct intercala razonamiento y acciones; Reflexion añade una "
            "autocrítica verbal entre intentos."),
    "d08": ("El ajuste fino con LoRA entrena matrices de bajo rango y congela los "
            "pesos originales."),
    "d09": ("La temperatura reescala los logits antes del softmax; valores bajos "
            "concentran la probabilidad."),
    "d10": ("La cuantización reduce la precisión de los pesos a 8 o 4 bits para "
            "ahorrar memoria."),
}
DOC_IDS = ["d%02d" % i for i in range(1, 11)]
DOC_LIST = [DOC_TEXTS[d] for d in DOC_IDS]

# ---------------------------------------------------------------------------
# Consultas y juicios de relevancia (literales del enunciado)
# ---------------------------------------------------------------------------
QUERIES = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}
QUERY_IDS = ["q%d" % i for i in range(1, 7)]
QUERY_LIST = [QUERIES[q][0] for q in QUERY_IDS]
RELEVANT = {q: QUERIES[q][1] for q in QUERY_IDS}

# ---------------------------------------------------------------------------
# Parte 1 (recalculada): TF-IDF por defecto, ajustado SOLO con el corpus
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto
X_docs = vectorizer.fit_transform(DOC_LIST)      # (10, V) — fit sólo sobre el corpus
X_queries = vectorizer.transform(QUERY_LIST)     # (6, V) — sólo transform
print("Vocabulario TF-IDF: %d términos; matriz de documentos: %s"
      % (len(vectorizer.vocabulary_), str(X_docs.shape)))

S_tfidf = cosine_similarity(X_queries, X_docs)   # (6, 10)


def evaluate(S):
    """S[i, j] = similitud de la consulta i con el documento j.
    Devuelve métricas agregadas (Hit@1, Hit@3, MRR) y el detalle por consulta."""
    per_query = {}
    hits1 = 0.0
    hits3 = 0.0
    rr_sum = 0.0
    for i, q in enumerate(QUERY_IDS):
        sims = S[i]
        order = np.argsort(-sims, kind="stable")  # descendente, empates por orden d01..d10
        ranking = [(DOC_IDS[j], float(sims[j])) for j in order]
        rel = RELEVANT[q]
        rank = int([d for d, _ in ranking].index(rel) + 1)
        h1 = 1.0 if rank == 1 else 0.0
        h3 = 1.0 if rank <= 3 else 0.0
        rr = 1.0 / float(rank)
        hits1 += h1
        hits3 += h3
        rr_sum += rr
        per_query[q] = {
            "consulta": QUERIES[q][0],
            "relevante": rel,
            "rank_relevante": float(rank),
            "hit_at_1": float(h1),
            "hit_at_3": float(h3),
            "reciprocal_rank": float(rr),
            "ranking": [
                {"pos": float(p + 1), "doc": d, "similitud": float(s)}
                for p, (d, s) in enumerate(ranking)
            ],
        }
    n = float(len(QUERY_IDS))
    metrics = {
        "hit_at_1": float(hits1 / n),
        "hit_at_3": float(hits3 / n),
        "mrr": float(rr_sum / n),
    }
    return metrics, per_query


metrics_tfidf, per_query_tfidf = evaluate(S_tfidf)

# ---------------------------------------------------------------------------
# Parte 2: LSA — TruncatedSVD(n_components=4, random_state=0) sobre el corpus
# ---------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
L_docs = svd.fit_transform(X_docs)     # ajuste SOLO sobre la TF-IDF del corpus
L_queries = svd.transform(X_queries)   # proyección de las consultas al espacio latente

S_lsa = cosine_similarity(L_queries, L_docs)  # similitud coseno en 4 dimensiones
metrics_lsa, per_query_lsa = evaluate(S_lsa)

# ---------------------------------------------------------------------------
# Tabla comparativa única (filas = métodos, columnas = Hit@1, Hit@3, MRR)
# ---------------------------------------------------------------------------
print("\n" + "=" * 52)
print("Tabla comparativa (6 consultas, 10 documentos)")
print("=" * 52)
print("%-10s%10s%10s%10s" % ("Método", "Hit@1", "Hit@3", "MRR"))
print("-" * 40)
print("%-10s%10.4f%10.4f%10.4f"
      % ("TF-IDF", metrics_tfidf["hit_at_1"], metrics_tfidf["hit_at_3"], metrics_tfidf["mrr"]))
print("%-10s%10.4f%10.4f%10.4f"
      % ("LSA", metrics_lsa["hit_at_1"], metrics_lsa["hit_at_3"], metrics_lsa["mrr"]))
print("-" * 40)

print("\nMétricas agregadas (mismos números de la tabla):")
print("  TF-IDF: Hit@1=%.6f  Hit@3=%.6f  MRR=%.6f"
      % (metrics_tfidf["hit_at_1"], metrics_tfidf["hit_at_3"], metrics_tfidf["mrr"]))
print("  LSA   : Hit@1=%.6f  Hit@3=%.6f  MRR=%.6f"
      % (metrics_lsa["hit_at_1"], metrics_lsa["hit_at_3"], metrics_lsa["mrr"]))

# ---------------------------------------------------------------------------
# Ranking LSA por consulta
# ---------------------------------------------------------------------------
print("\nRanking LSA por consulta (espacio latente de 4 dimensiones):")
for q in QUERY_IDS:
    info = per_query_lsa[q]
    print("\n%s -> relevante: %s | %s" % (q, info["relevante"], info["consulta"]))
    for item in info["ranking"]:
        mark = "   <-- relevante" if item["doc"] == info["relevante"] else ""
        print("  %2d. %s  sim=%+.4f%s" % (int(item["pos"]), item["doc"],
                                          item["similitud"], mark))
    print("  rank del relevante: %d | Hit@1=%d | Hit@3=%d | RR=%.4f"
          % (int(info["rank_relevante"]), int(info["hit_at_1"]),
             int(info["hit_at_3"]), info["reciprocal_rank"]))

# ---------------------------------------------------------------------------
# Resultados -> resultados.json
# ---------------------------------------------------------------------------
results = {
    "subtarea": "s2 - LSA (TruncatedSVD n_components=4, random_state=0) vs linea base TF-IDF",
    "config": {
        "num_documentos": 10.0,
        "num_consultas": 6.0,
        "n_componentes_lsa": 4.0,
        "random_state": 0.0,
        "vectorizer": "TfidfVectorizer (parametros por defecto)",
        "tam_vocabulario": float(len(vectorizer.vocabulary_)),
    },
    "tabla_comparativa": {
        "TF-IDF": {
            "hit_at_1": metrics_tfidf["hit_at_1"],
            "hit_at_3": metrics_tfidf["hit_at_3"],
            "mrr": metrics_tfidf["mrr"],
        },
        "LSA": {
            "hit_at_1": metrics_lsa["hit_at_1"],
            "hit_at_3": metrics_lsa["hit_at_3"],
            "mrr": metrics_lsa["mrr"],
        },
    },
    "tfidf_linea_base": {
        "hit_at_1": metrics_tfidf["hit_at_1"],
        "hit_at_3": metrics_tfidf["hit_at_3"],
        "mrr": metrics_tfidf["mrr"],
        "por_consulta": per_query_tfidf,
    },
    "lsa": {
        "hit_at_1": metrics_lsa["hit_at_1"],
        "hit_at_3": metrics_lsa["hit_at_3"],
        "mrr": metrics_lsa["mrr"],
        "varianza_explicada_por_componente": [float(v) for v in svd.explained_variance_ratio_],
        "varianza_explicada_total": float(np.sum(svd.explained_variance_ratio_)),
        "por_consulta": per_query_lsa,
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nResultados escritos en resultados.json")
