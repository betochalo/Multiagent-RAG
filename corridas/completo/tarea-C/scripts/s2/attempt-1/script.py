# -*- coding: utf-8 -*-
"""
Tarea C - Parte 2 (subtask s2): recuperación léxica (TF-IDF) vs semántica latente (LSA).

Script autocontenido: define el corpus d01-d10 y las consultas q1-q6 con sus juicios
exactamente como en el enunciado, recalcula el baseline TF-IDF (TfidfVectorizer por
defecto + cosine_similarity), aplica LSA con TruncatedSVD(n_components=4,
random_state=0) ajustado (fit) sobre la matriz TF-IDF del corpus y aplicado (transform)
a las consultas, normaliza L2 los vectores latentes y evalúa Hit@1, Hit@3 y MRR con los
mismos juicios. Escribe resultados.json e imprime una única tabla comparativa más el
ranking por consulta de LSA con la posición del documento relevante.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ----------------------------------------------------------------------------
# 1. Datos del enunciado: corpus, consultas y juicios de relevancia
# ----------------------------------------------------------------------------
CORPUS = [
    ("d01", "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión."),
    ("d02", "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización."),
    ("d03", "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo."),
    ("d04", "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento."),
    ("d05", "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación."),
    ("d06", "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado."),
    ("d07", "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos."),
    ("d08", "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales."),
    ("d09", "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad."),
    ("d10", "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria."),
]

CONSULTAS = [
    ("q1", "¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    ("q2", "¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    ("q3", "¿Qué técnica entrena matrices de bajo rango?", "d08"),
    ("q4", "¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    ("q5", "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    ("q6", "¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
]

doc_ids = [d[0] for d in CORPUS]
doc_texts = [d[1] for d in CORPUS]
query_ids = [c[0] for c in CONSULTAS]
query_texts = [c[1] for c in CONSULTAS]
RELEVANTE = {c[0]: c[2] for c in CONSULTAS}

# ----------------------------------------------------------------------------
# 2. Baseline TF-IDF (Parte 1, recalculada aquí con parámetros por defecto)
# ----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto
X_docs = vectorizer.fit_transform(doc_texts)
X_queries = vectorizer.transform(query_texts)
sim_tfidf = cosine_similarity(X_queries, X_docs)  # forma (6, 10)

# ----------------------------------------------------------------------------
# 3. LSA (Parte 2): TruncatedSVD(n_components=4, random_state=0)
#    fit SOLO sobre la matriz TF-IDF del corpus; transform para las consultas.
# ----------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs)
X_queries_lsa = svd.transform(X_queries)

# Normalización L2 de los vectores latentes antes de la similitud coseno
X_docs_lsa_n = normalize(X_docs_lsa, norm="l2")
X_queries_lsa_n = normalize(X_queries_lsa, norm="l2")
sim_lsa = cosine_similarity(X_queries_lsa_n, X_docs_lsa_n)

# ----------------------------------------------------------------------------
# 4. Evaluación: Hit@1, Hit@3, MRR y ranking por consulta
# ----------------------------------------------------------------------------
def evaluar(sim_matrix):
    por_consulta = {}
    hits1, hits3, rrs = [], [], []
    for i, qid in enumerate(query_ids):
        sims = sim_matrix[i]
        # Orden descendente; empates resueltos por el orden original d01..d10
        order = np.argsort(-sims, kind="stable")
        ranking = [
            {"pos": int(p + 1), "id": doc_ids[j], "similitud": float(sims[j])}
            for p, j in enumerate(order)
        ]
        rel = RELEVANTE[qid]
        rank = next(r["pos"] for r in ranking if r["id"] == rel)
        h1 = 1.0 if rank == 1 else 0.0
        h3 = 1.0 if rank <= 3 else 0.0
        rr = 1.0 / float(rank)
        hits1.append(h1)
        hits3.append(h3)
        rrs.append(rr)
        por_consulta[qid] = {
            "consulta": query_texts[i],
            "relevante": rel,
            "rank_relevante": int(rank),
            "hit_at_1": float(h1),
            "hit_at_3": float(h3),
            "reciprocal_rank": float(rr),
            "ranking": ranking,
        }
    metricas = {
        "hit_at_1": float(np.mean(hits1)),
        "hit_at_3": float(np.mean(hits3)),
        "mrr": float(np.mean(rrs)),
    }
    return metricas, por_consulta

met_tfidf, por_tfidf = evaluar(sim_tfidf)
met_lsa, por_lsa = evaluar(sim_lsa)

# ----------------------------------------------------------------------------
# 5. Salida en pantalla: tabla comparativa única + ranking por consulta de LSA
# ----------------------------------------------------------------------------
print("=" * 62)
print("Tarea C - Parte 2: TF-IDF vs LSA (TruncatedSVD, 4 dimensiones)")
print("=" * 62)
print()
print("Tabla comparativa (6 consultas, 10 documentos)")
header = f"{'Método':<14}{'Hit@1':>10}{'Hit@3':>10}{'MRR':>10}"
print(header)
print("-" * len(header))
print(f"{'TF-IDF':<14}{met_tfidf['hit_at_1']:>10.4f}{met_tfidf['hit_at_3']:>10.4f}{met_tfidf['mrr']:>10.4f}")
print(f"{'LSA (SVD 4)':<14}{met_lsa['hit_at_1']:>10.4f}{met_lsa['hit_at_3']:>10.4f}{met_lsa['mrr']:>10.4f}")

print()
print("Posición del documento relevante por consulta")
print(f"{'consulta':<10}{'relevante':<12}{'rank TF-IDF':>13}{'rank LSA':>10}")
for qid in query_ids:
    print(f"{qid:<10}{RELEVANTE[qid]:<12}{por_tfidf[qid]['rank_relevante']:>13}"
          f"{por_lsa[qid]['rank_relevante']:>10}")

print()
print("Ranking por consulta - LSA (SVD 4) ('*' marca el documento relevante)")
for qid in query_ids:
    info = por_lsa[qid]
    print()
    print(f"{qid}: {info['consulta']}")
    print(f"    relevante: {info['relevante']} -> posición {info['rank_relevante']} "
          f"(Hit@1={info['hit_at_1']:.0f}, Hit@3={info['hit_at_3']:.0f}, "
          f"RR={info['reciprocal_rank']:.4f})")
    for r in info["ranking"]:
        marca = " *" if r["id"] == info["relevante"] else ""
        print(f"    {r['pos']:>2}. {r['id']}  sim={r['similitud']:+.4f}{marca}")

var_ratio = svd.explained_variance_ratio_
print()
print("Varianza explicada por las 4 componentes SVD: "
      + ", ".join(f"{v:.4f}" for v in var_ratio)
      + f" (total={float(var_ratio.sum()):.4f})")

# ----------------------------------------------------------------------------
# 6. Resultados -> resultados.json
# ----------------------------------------------------------------------------
resultados = {
    "tarea": "Tarea C - Parte 2: recuperación léxica (TF-IDF) vs semántica latente (LSA)",
    "config": {
        "num_documentos": len(CORPUS),
        "num_consultas": len(CONSULTAS),
        "tfidf": "TfidfVectorizer() con parámetros por defecto + cosine_similarity",
        "lsa": ("TruncatedSVD(n_components=4, random_state=0) fit sobre la matriz "
                "TF-IDF del corpus y transform sobre las consultas; normalize(norm='l2') "
                "de documentos y consultas antes de la similitud coseno"),
    },
    "corpus": [{"id": d[0], "texto": d[1]} for d in CORPUS],
    "consultas": [{"id": c[0], "texto": c[1], "relevante": c[2]} for c in CONSULTAS],
    "tabla_comparativa": [
        {"metodo": "TF-IDF", "hit_at_1": met_tfidf["hit_at_1"],
         "hit_at_3": met_tfidf["hit_at_3"], "mrr": met_tfidf["mrr"]},
        {"metodo": "LSA (SVD 4)", "hit_at_1": met_lsa["hit_at_1"],
         "hit_at_3": met_lsa["hit_at_3"], "mrr": met_lsa["mrr"]},
    ],
    "tfidf": {"metricas": met_tfidf, "por_consulta": por_tfidf},
    "lsa": {"metricas": met_lsa, "por_consulta": por_lsa},
    "svd": {
        "n_components": 4,
        "random_state": 0,
        "varianza_explicada_ratio": [float(v) for v in var_ratio],
        "varianza_explicada_total": float(var_ratio.sum()),
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print()
print("Resultados escritos en resultados.json")
