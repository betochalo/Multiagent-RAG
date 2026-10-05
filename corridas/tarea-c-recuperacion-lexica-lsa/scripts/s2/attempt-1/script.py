# -*- coding: utf-8 -*-
"""
Tarea C - Parte 2: Semantica latente (LSA) y tabla comparativa.

Recomputa la linea base TF-IDF (Parte 1: TfidfVectorizer por defecto +
similitud coseno) y construye la variante LSA:
  - TruncatedSVD(n_components=4, random_state=0) ajustado SOLO sobre la
    matriz TF-IDF de los 10 documentos,
  - documentos y consultas proyectados a 4 dimensiones con ese mismo SVD,
  - vectores latentes normalizados con norma L2 (Normalizer),
  - similitud coseno en el espacio latente.
Evalua Hit@1, Hit@3 y MRR sobre las 6 consultas con los mismos juicios,
imprime UNA sola tabla comparativa (metricas + rango del relevante por
consulta para ambos metodos) y guarda todo en resultados.json.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# 1. Datos literales del enunciado
# ----------------------------------------------------------------------
doc_ids = ["d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10"]
docs = [
    "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión.",
    "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización.",
    "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo.",
    "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento.",
    "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación.",
    "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado.",
    "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos.",
    "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales.",
    "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad.",
    "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria.",
]

# (id, consulta, documento relevante)
queries = [
    ("q1", "¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    ("q2", "¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    ("q3", "¿Qué técnica entrena matrices de bajo rango?", "d08"),
    ("q4", "¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    ("q5", "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    ("q6", "¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
]

q_ids = [q[0] for q in queries]
q_texts = [q[1] for q in queries]

# ----------------------------------------------------------------------
# 2. Linea base TF-IDF (Parte 1, recomputada aqui)
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parametros por defecto
X_docs = vectorizer.fit_transform(docs)        # ajuste solo sobre documentos
X_queries = vectorizer.transform(q_texts)      # consultas con el mismo vocabulario

sim_tfidf = cosine_similarity(X_queries, X_docs)  # shape (6, 10)

# ----------------------------------------------------------------------
# 3. LSA: TruncatedSVD(n_components=4, random_state=0) + Normalizer(L2)
# ----------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
U_docs = svd.fit_transform(X_docs)        # SVD ajustado sobre los documentos
U_queries = svd.transform(X_queries)      # consultas proyectadas con el MISMO SVD

normalizer = Normalizer(norm="l2")        # normalizacion L2 de los vectores latentes
L_docs = normalizer.fit_transform(U_docs)
L_queries = normalizer.transform(U_queries)

sim_lsa = cosine_similarity(L_queries, L_docs)

# ----------------------------------------------------------------------
# 4. Evaluacion: Hit@1, Hit@3, MRR y rango del relevante por consulta
# ----------------------------------------------------------------------
def evaluate(sim_matrix):
    """Rankea los 10 documentos por similitud y evalua contra el juicio relevante."""
    hit1, hit3, rrs, ranks = [], [], [], {}
    for row, (qid, _, rel) in zip(sim_matrix, queries):
        order = np.argsort(-np.asarray(row).ravel(), kind="stable")
        ranking = [doc_ids[i] for i in order]
        rank = ranking.index(rel) + 1  # rango 1-based del documento relevante
        ranks[qid] = float(rank)
        hit1.append(1.0 if rank == 1 else 0.0)
        hit3.append(1.0 if rank <= 3 else 0.0)
        rrs.append(1.0 / rank)
    return {
        "hit_at_1": float(np.mean(hit1)),
        "hit_at_3": float(np.mean(hit3)),
        "mrr": float(np.mean(rrs)),
        "rango_del_relevante_por_consulta": ranks,
    }

res_tfidf = evaluate(sim_tfidf)
res_lsa = evaluate(sim_lsa)

results = {
    "tfidf_baseline": res_tfidf,
    "lsa": res_lsa,
    "lsa_config": {
        "n_components": 4,
        "random_state": 0,
        "normalizacion_latente": "l2",
    },
    "lsa_varianza_explicada_por_componente": [float(v) for v in svd.explained_variance_ratio_],
    "lsa_varianza_explicada_total": float(np.sum(svd.explained_variance_ratio_)),
}

# ----------------------------------------------------------------------
# 5. Tabla unica comparativa
# ----------------------------------------------------------------------
def fmt_row(name, res):
    ranks = res["rango_del_relevante_por_consulta"]
    rank_str = " ".join("{:>4d}".format(int(ranks[q])) for q in q_ids)
    return "{:<9} {:>6.3f} {:>7.3f} {:>7.3f} | {}".format(
        name, res["hit_at_1"], res["hit_at_3"], res["mrr"], rank_str
    )

print("Tabla comparativa TF-IDF vs LSA "
      "(r_qi = rango 1-based del documento relevante de la consulta qi)")
print("{:<9} {:>6} {:>7} {:>7} | {}".format(
    "Metodo", "Hit@1", "Hit@3", "MRR", " ".join("{:>4s}".format(q) for q in q_ids)))
print(fmt_row("TF-IDF", res_tfidf))
print(fmt_row("LSA", res_lsa))

# ----------------------------------------------------------------------
# 6. Guardar resultados (mismos numeros) en resultados.json
# ----------------------------------------------------------------------
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nContenido de resultados.json:")
print(json.dumps(results, indent=2, ensure_ascii=False))
print("\nResultados guardados en resultados.json")
