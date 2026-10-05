# -*- coding: utf-8 -*-
"""
Subtask s3 (Parte 3) — Análisis por consulta.

Recomputa ambos pipelines del enunciado con los mismos datos y parámetros:
  * TF-IDF: TfidfVectorizer de scikit-learn con parámetros por defecto,
    ajustado SOLO con el corpus d01–d10, + similitud coseno.
  * LSA: TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF,
    + similitud coseno.

Para cada método y cada consulta (q1–q6, relevantes d04, d03, d08, d05, d09, d02):
  * top 3 de documentos,
  * rango exacto del documento relevante,
  * fallo = relevante fuera del top 3,
  * términos compartidos consulta–documento relevante según la tokenización
    del vectorizador, indicando cuáles están en el vocabulario TF-IDF.

Escribe resultados.json e imprime la tabla pedida y las consultas falladas
por cada método. No se requiere ninguna figura.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# Datos del enunciado (embebidos literalmente)
# ----------------------------------------------------------------------------

DOCS = {
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

QUERIES = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

DOC_IDS = list(DOCS.keys())      # d01..d10 en orden fijo
QUERY_IDS = list(QUERIES.keys()) # q1..q6 en orden fijo
TOP_K = 3

# ----------------------------------------------------------------------------
# Pipelines: fit SOLO con el corpus; las consultas solo se transforman
# ----------------------------------------------------------------------------

doc_texts = [DOCS[d] for d in DOC_IDS]
query_texts = [QUERIES[q][0] for q in QUERY_IDS]

vectorizer = TfidfVectorizer()  # parámetros por defecto
X_docs = vectorizer.fit_transform(doc_texts)
X_queries = vectorizer.transform(query_texts)
vocab = vectorizer.vocabulary_
analyzer = vectorizer.build_analyzer()

svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs)
X_queries_lsa = svd.transform(X_queries)

sims_tfidf = cosine_similarity(X_queries, X_docs)        # 6 x 10
sims_lsa = cosine_similarity(X_queries_lsa, X_docs_lsa)  # 6 x 10

# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------

def rank_row(sims_row):
    """Orden descendente estable: los empates conservan el orden d01..d10."""
    row = np.asarray(sims_row).ravel()
    order = np.argsort(-row, kind="stable")
    docs = [DOC_IDS[i] for i in order]
    vals = [float(row[i]) for i in order]
    return docs, vals


def fmt_list(terms):
    return ", ".join(terms) if terms else "(ninguno)"


def fmt_docs(top3):
    return ", ".join("{} ({:.4f})".format(e["doc"], e["similitud_coseno"]) for e in top3)


def bool_es(b):
    return "sí" if b else "no"

# ----------------------------------------------------------------------------
# Análisis por consulta
# ----------------------------------------------------------------------------

por_metodo = {"tfidf": {}, "lsa": {}}
tabla = []

for qi, qid in enumerate(QUERY_IDS):
    q_text, rel = QUERIES[qid]

    # Términos según la tokenización del vectorizador
    q_terms = set(analyzer(q_text))
    d_terms = set(analyzer(DOCS[rel]))
    shared = sorted(q_terms & d_terms)
    shared_in_vocab = [t for t in shared if t in vocab]
    shared_out_vocab = [t for t in shared if t not in vocab]
    q_oov = sorted(t for t in q_terms if t not in vocab)
    d_oov = sorted(t for t in d_terms if t not in vocab)

    entry_common = {
        "consulta": q_text,
        "relevante": rel,
        "terminos_compartidos": shared,
        "terminos_compartidos_en_vocabulario_tfidf": shared_in_vocab,
        "terminos_compartidos_fuera_de_vocabulario": shared_out_vocab,
        "terminos_consulta_fuera_de_vocabulario": q_oov,
        "terminos_documento_relevante_fuera_de_vocabulario": d_oov,
    }

    for metodo, sims in (("tfidf", sims_tfidf), ("lsa", sims_lsa)):
        docs, vals = rank_row(sims[qi])
        rank_rel = docs.index(rel) + 1
        por_metodo[metodo][qid] = dict(entry_common)
        por_metodo[metodo][qid].update({
            "ranking_completo": [
                {"pos": p + 1, "doc": docs[p], "similitud_coseno": vals[p]}
                for p in range(len(docs))
            ],
            "top3": [
                {"pos": p + 1, "doc": docs[p], "similitud_coseno": vals[p]}
                for p in range(min(TOP_K, len(docs)))
            ],
            "rank_relevante": int(rank_rel),
            "fallo_top3": bool(rank_rel > TOP_K),
        })

    t = por_metodo["tfidf"][qid]
    l = por_metodo["lsa"][qid]
    tabla.append({
        "consulta": qid,
        "texto_consulta": q_text,
        "relevante": rel,
        "terminos_compartidos": shared,
        "terminos_compartidos_en_vocabulario": shared_in_vocab,
        "terminos_compartidos_fuera_de_vocabulario": shared_out_vocab,
        "terminos_consulta_fuera_de_vocabulario": q_oov,
        "rank_tfidf": t["rank_relevante"],
        "rank_lsa": l["rank_relevante"],
        "fallo_tfidf": t["fallo_top3"],
        "fallo_lsa": l["fallo_top3"],
    })

falladas = {
    metodo: [qid for qid in QUERY_IDS if por_metodo[metodo][qid]["fallo_top3"]]
    for metodo in ("tfidf", "lsa")
}

# ----------------------------------------------------------------------------
# Resultados (JSON) — todos los números calculados
# ----------------------------------------------------------------------------

var_explicada = float(np.sum(svd.explained_variance_ratio_))

resultados = {
    "subtarea": "s3 - Parte 3: análisis por consulta",
    "config": {
        "corpus": DOC_IDS,
        "consultas": {qid: QUERIES[qid][0] for qid in QUERY_IDS},
        "relevantes": {qid: QUERIES[qid][1] for qid in QUERY_IDS},
        "vectorizador": "TfidfVectorizer (parámetros por defecto), ajustado solo con el corpus",
        "lsa": "TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF",
        "n_documentos": len(DOC_IDS),
        "n_consultas": len(QUERY_IDS),
        "n_vocabulario_tfidf": int(len(vocab)),
        "n_componentes_lsa": 4,
        "varianza_explicada_lsa": var_explicada,
        "criterio_fallo": "el documento relevante queda fuera del top 3",
    },
    "consultas_falladas": {
        "tfidf": falladas["tfidf"],
        "lsa": falladas["lsa"],
    },
    "tfidf": {
        "consultas_falladas": falladas["tfidf"],
        "por_consulta": por_metodo["tfidf"],
    },
    "lsa": {
        "consultas_falladas": falladas["lsa"],
        "por_consulta": por_metodo["lsa"],
    },
    "tabla_por_consulta": tabla,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------------
# Salida por stdout (los mismos números)
# ----------------------------------------------------------------------------

print("=" * 112)
print("Parte 3 — Análisis por consulta (TF-IDF por defecto vs LSA TruncatedSVD(4, random_state=0))")
print("=" * 112)
print("Vocabulario TF-IDF: {} términos | Varianza explicada LSA (4 componentes): {:.4f}".format(
    len(vocab), var_explicada))

print("\nTabla resumen")
print("-" * 112)
print("{:<9}{:<11}{:<70}{:>12}{:>10}".format(
    "consulta", "relevante", "términos compartidos (tokenización del vectorizador)",
    "rank TF-IDF", "rank LSA"))
print("-" * 112)
for row in tabla:
    print("{:<9}{:<11}{:<70}{:>12}{:>10}".format(
        row["consulta"], row["relevante"],
        fmt_list(row["terminos_compartidos"]),
        row["rank_tfidf"], row["rank_lsa"]))

print("\nConsultas falladas (relevante fuera del top 3):")
print("  TF-IDF: {}".format(", ".join(falladas["tfidf"]) if falladas["tfidf"] else "ninguna"))
print("  LSA:    {}".format(", ".join(falladas["lsa"]) if falladas["lsa"] else "ninguna"))

print("\nDetalle por consulta")
print("-" * 112)
for qid in QUERY_IDS:
    t = por_metodo["tfidf"][qid]
    l = por_metodo["lsa"][qid]
    print("\n{}: {}".format(qid, t["consulta"]))
    print("  Relevante: {}".format(t["relevante"]))
    print("  TF-IDF top3: {}".format(fmt_docs(t["top3"])))
    print("  LSA    top3: {}".format(fmt_docs(l["top3"])))
    print("  Rango del relevante -> TF-IDF: {} | LSA: {}".format(t["rank_relevante"], l["rank_relevante"]))
    print("  Fallo (fuera del top 3) -> TF-IDF: {} | LSA: {}".format(
        bool_es(t["fallo_top3"]), bool_es(l["fallo_top3"])))
    print("  Términos compartidos consulta-relevante: {}".format(fmt_list(t["terminos_compartidos"])))
    print("    en vocabulario TF-IDF: {}".format(fmt_list(t["terminos_compartidos_en_vocabulario_tfidf"])))
    print("    fuera de vocabulario:  {}".format(fmt_list(t["terminos_compartidos_fuera_de_vocabulario"])))
    print("  Términos de la consulta fuera de vocabulario: {}".format(
        fmt_list(t["terminos_consulta_fuera_de_vocabulario"])))

print("\nResultados escritos en resultados.json")
