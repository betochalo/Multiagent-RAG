# -*- coding: utf-8 -*-
"""
Subtask s3 (Parte 3) — Tarea C: análisis por consulta.

Recalcula aquí ambos métodos con los mismos datos y parámetros (no lee archivos
de otros subtasks):
  - TF-IDF: TfidfVectorizer() con parámetros por defecto + cosine_similarity
  - LSA:    TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF
            del corpus + normalización L2 de documentos y consultas + coseno

Para cada consulta y método determina si el documento relevante queda fuera del
top 3 (fallo), imprime una tabla por consulta con el rank del relevante bajo
cada método, lista los términos compartidos entre la consulta y su documento
relevante (intersección de tokens en minúsculas con el tokenizador por defecto
de TfidfVectorizer) con su frecuencia documental en el corpus, y cierra
indicando explícitamente en qué consultas falla TF-IDF y en cuáles LSA.

Escribe todos los números en resultados.json y los imprime en stdout.
"""

import sys
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TOP_K = 3

# ----------------------------------------------------------------------------
# Corpus d01–d10 y consultas q1–q6 con juicios (textos exactos del enunciado)
# ----------------------------------------------------------------------------
corpus = [
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

consultas = [
    ("q1", "¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    ("q2", "¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    ("q3", "¿Qué técnica entrena matrices de bajo rango?", "d08"),
    ("q4", "¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    ("q5", "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    ("q6", "¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
]

doc_ids = [c[0] for c in corpus]
doc_texts = [c[1] for c in corpus]
idx_of = {cid: i for i, cid in enumerate(doc_ids)}

# ----------------------------------------------------------------------------
# Método 1: TF-IDF (parámetros por defecto) + similitud coseno
# (fit solo sobre el corpus; las consultas solo se transforman)
# ----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()
X_docs = vectorizer.fit_transform(doc_texts)
X_queries = vectorizer.transform([q[1] for q in consultas])
sim_tfidf = cosine_similarity(X_queries, X_docs)

# ----------------------------------------------------------------------------
# Método 2: LSA = TruncatedSVD(n_components=4, random_state=0) + normalización
# L2 de documentos y consultas + similitud coseno
# (fit del SVD solo sobre el corpus)
# ----------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
lsa_docs = svd.fit_transform(X_docs)
lsa_queries = svd.transform(X_queries)
lsa_docs_n = normalize(lsa_docs, norm="l2")
lsa_queries_n = normalize(lsa_queries, norm="l2")
sim_lsa = cosine_similarity(lsa_queries_n, lsa_docs_n)

# ----------------------------------------------------------------------------
# Tokens en minúsculas con el tokenizador por defecto de TfidfVectorizer y
# frecuencia documental (en cuántos documentos del corpus aparece cada término)
# ----------------------------------------------------------------------------
analyzer = vectorizer.build_analyzer()
doc_token_sets = [set(analyzer(t)) for t in doc_texts]
doc_freq = {}
for tset in doc_token_sets:
    for tok in tset:
        doc_freq[tok] = doc_freq.get(tok, 0) + 1


def orden_ranking(sim_row):
    """Ids de documentos ordenados por similitud descendente
    (empates resueltos por el orden d01..d10 del corpus)."""
    order = np.argsort(-np.asarray(sim_row), kind="stable")
    return [doc_ids[j] for j in order]


# ----------------------------------------------------------------------------
# Análisis por consulta: rank del relevante, fallo (fuera del top 3) y
# términos compartidos consulta-relevante con su frecuencia documental
# ----------------------------------------------------------------------------
por_consulta = {}
filas_tabla = []

for i, (qid, qtext, rel) in enumerate(consultas):
    ranking_t = orden_ranking(sim_tfidf[i])
    ranking_l = orden_ranking(sim_lsa[i])
    rank_tfidf = ranking_t.index(rel) + 1
    rank_lsa = ranking_l.index(rel) + 1
    fallo_tfidf = bool(rank_tfidf > TOP_K)
    fallo_lsa = bool(rank_lsa > TOP_K)
    top3_tfidf = ranking_t[:TOP_K]
    top3_lsa = ranking_l[:TOP_K]
    sim_rel_tfidf = float(sim_tfidf[i, idx_of[rel]])
    sim_rel_lsa = float(sim_lsa[i, idx_of[rel]])

    compartidos = sorted(set(analyzer(qtext)) & doc_token_sets[idx_of[rel]])
    terminos = [{"termino": tok, "df": int(doc_freq[tok])} for tok in compartidos]

    por_consulta[qid] = {
        "consulta": qtext,
        "relevante": rel,
        "rank_tfidf": int(rank_tfidf),
        "rank_lsa": int(rank_lsa),
        "fallo_tfidf": fallo_tfidf,
        "fallo_lsa": fallo_lsa,
        "sim_tfidf_relevante": sim_rel_tfidf,
        "sim_lsa_relevante": sim_rel_lsa,
        "top3_tfidf": top3_tfidf,
        "top3_lsa": top3_lsa,
        "num_terminos_compartidos": int(len(terminos)),
        "terminos_compartidos": terminos,
    }
    filas_tabla.append((qid, rel, int(rank_tfidf), int(rank_lsa), fallo_tfidf, fallo_lsa))

falla_tfidf = [f[0] for f in filas_tabla if f[4]]
falla_lsa = [f[0] for f in filas_tabla if f[5]]

# ----------------------------------------------------------------------------
# Impresión: tabla por consulta
# ----------------------------------------------------------------------------
print("=" * 86)
print("Parte 3 — Análisis por consulta (fallo = documento relevante fuera del top 3)")
print("=" * 86)
header = (f"{'consulta':<10}{'relevante':<11}{'rank_TF-IDF':>13}{'rank_LSA':>10}"
          f"{'fallo_TF-IDF':>14}{'fallo_LSA':>11}")
print(header)
print("-" * len(header))
for qid, rel, rt, rl, ft, fl in filas_tabla:
    print(f"{qid:<10}{rel:<11}{rt:>13}{rl:>10}"
          f"{('SI' if ft else 'no'):>14}{('SI' if fl else 'no'):>11}")

# ----------------------------------------------------------------------------
# Impresión: detalle por consulta con términos compartidos y su df
# ----------------------------------------------------------------------------
print()
for qid, qtext, rel in consultas:
    info = por_consulta[qid]
    print("-" * 86)
    print(f"{qid}: {qtext}")
    print(f"  relevante: {rel} | rank TF-IDF = {info['rank_tfidf']} | rank LSA = {info['rank_lsa']} | "
          f"fallo TF-IDF = {'SI' if info['fallo_tfidf'] else 'no'} | fallo LSA = {'SI' if info['fallo_lsa'] else 'no'}")
    print(f"  sim(relevante) TF-IDF = {info['sim_tfidf_relevante']} | sim(relevante) LSA = {info['sim_lsa_relevante']}")
    print(f"  top 3 TF-IDF: {', '.join(info['top3_tfidf'])} | top 3 LSA: {', '.join(info['top3_lsa'])}")
    if info["terminos_compartidos"]:
        print("  términos compartidos consulta-relevante (token: frecuencia documental en el corpus de 10 docs):")
        print("    " + ", ".join(f"{t['termino']} (df={t['df']})" for t in info["terminos_compartidos"]))
    else:
        print("  términos compartidos consulta-relevante: ninguno")

# ----------------------------------------------------------------------------
# Cierre: listado explícito de fallos por método
# ----------------------------------------------------------------------------
print("=" * 86)
print("Resumen — consultas donde el relevante queda fuera del top 3")
print("=" * 86)
print(f"TF-IDF falla en: {', '.join(falla_tfidf) if falla_tfidf else 'ninguna'} "
      f"({len(falla_tfidf)}/{len(consultas)} consultas)")
print(f"LSA    falla en: {', '.join(falla_lsa) if falla_lsa else 'ninguna'} "
      f"({len(falla_lsa)}/{len(consultas)} consultas)")

# ----------------------------------------------------------------------------
# Resultados -> resultados.json
# ----------------------------------------------------------------------------
resultados = {
    "tarea": "Tarea C - Parte 3: análisis por consulta (fallo = relevante fuera del top 3)",
    "config": {
        "num_documentos": len(corpus),
        "num_consultas": len(consultas),
        "tfidf": "TfidfVectorizer() con parámetros por defecto + cosine_similarity",
        "lsa": "TruncatedSVD(n_components=4, random_state=0) fit sobre el corpus + normalize(norm='l2') + coseno",
        "criterio_fallo": "rank del documento relevante > 3",
    },
    "tabla_por_consulta": [
        {
            "consulta": qid,
            "relevante": rel,
            "rank_tfidf": rt,
            "rank_lsa": rl,
            "fallo_tfidf": ft,
            "fallo_lsa": fl,
        }
        for (qid, rel, rt, rl, ft, fl) in filas_tabla
    ],
    "por_consulta": por_consulta,
    "resumen": {
        "consultas_falla_tfidf": falla_tfidf,
        "consultas_falla_lsa": falla_lsa,
        "num_fallos_tfidf": int(len(falla_tfidf)),
        "num_fallos_lsa": int(len(falla_lsa)),
        "hit_at_3_tfidf": float((len(consultas) - len(falla_tfidf)) / len(consultas)),
        "hit_at_3_lsa": float((len(consultas) - len(falla_lsa)) / len(consultas)),
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print()
print("Resultados escritos en resultados.json")
print(json.dumps(resultados, indent=2, ensure_ascii=False))
