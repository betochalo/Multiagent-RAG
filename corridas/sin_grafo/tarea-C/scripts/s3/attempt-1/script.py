# -*- coding: utf-8 -*-
"""
Subtarea s3 (Parte 3) - Análisis de fallos por consulta.

Recomputa ambas tuberías de las Partes 1 y 2 con los mismos datos y parámetros:
  - corpus d01-d10 (textos literales del enunciado)
  - consultas q1-q6 con relevantes q1->d04, q2->d03, q3->d08, q4->d05,
    q5->d09, q6->d02
  - TF-IDF: TfidfVectorizer (parámetros por defecto) + similitud coseno
  - LSA: TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF
    del corpus (las consultas se proyectan con el mismo SVD)

Para cada método y cada consulta obtiene el rank del documento relevante,
marca como FALLO el caso "relevante fuera del top 3" (rank > 3) y extrae los
términos compartidos entre la consulta y su documento relevante (tokens en
minúsculas con el patrón de tokens por defecto de TfidfVectorizer: palabras
de 2+ caracteres), para explicar los fallos por solapamiento léxico.

Escribe resultados.json y muestra los mismos números por stdout.
"""

import json

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. Datos literales del enunciado
# ---------------------------------------------------------------------------
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

doc_ids = [f"d{i:02d}" for i in range(1, 11)]
doc_texts = [corpus[d] for d in doc_ids]
query_ids = [f"q{i}" for i in range(1, 7)]
query_texts = [consultas[q][0] for q in query_ids]
relevante = {q: consultas[q][1] for q in query_ids}

# ---------------------------------------------------------------------------
# 2. Tubería 1: TF-IDF (parámetros por defecto), ajustado SOLO con el corpus
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()
X_docs = vectorizer.fit_transform(doc_texts)
X_queries = vectorizer.transform(query_texts)
S_tfidf = cosine_similarity(X_queries, X_docs)

# ---------------------------------------------------------------------------
# 3. Tubería 2: LSA, SVD ajustado SOLO con la matriz TF-IDF del corpus
# ---------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs)
X_queries_lsa = svd.transform(X_queries)
S_lsa = cosine_similarity(X_queries_lsa, X_docs_lsa)

# ---------------------------------------------------------------------------
# 4. Evaluación por consulta: rank del relevante y fallos (rank > 3)
# ---------------------------------------------------------------------------
def ranking_de_fila(sim_fila):
    """Ids de documentos ordenados por similitud descendente (empates: d01..d10)."""
    orden = np.argsort(-sim_fila, kind="stable")
    return [doc_ids[i] for i in orden]


def evaluar_metodo(S, nombre):
    por_consulta = {}
    falladas = []
    hits1 = 0
    hits3 = 0
    suma_rr = 0.0
    for qi, q in enumerate(query_ids):
        sim_fila = S[qi]
        ranking = ranking_de_fila(sim_fila)
        rel = relevante[q]
        rank = int(ranking.index(rel)) + 1
        sim_rel = float(sim_fila[doc_ids.index(rel)])
        hit1 = rank == 1
        hit3 = rank <= 3
        rr = 1.0 / rank
        hits1 += int(hit1)
        hits3 += int(hit3)
        suma_rr += rr
        por_consulta[q] = {
            "consulta": consultas[q][0],
            "relevante": rel,
            "rank_relevante": rank,
            "similitud_relevante": sim_rel,
            "hit_at_1": bool(hit1),
            "hit_at_3": bool(hit3),
            "reciprocal_rank": float(rr),
            "falla_top3": bool(rank > 3),
            "ranking": [
                {
                    "pos": p + 1,
                    "doc": ranking[p],
                    "similitud": float(sim_fila[doc_ids.index(ranking[p])]),
                }
                for p in range(len(doc_ids))
            ],
        }
        if rank > 3:
            falladas.append({"consulta": q, "relevante": rel, "rank_relevante": rank})
    metricas = {
        "hit_at_1": float(hits1) / len(query_ids),
        "hit_at_3": float(hits3) / len(query_ids),
        "mrr": float(suma_rr) / len(query_ids),
    }
    return {
        "metodo": nombre,
        "metricas": metricas,
        "por_consulta": por_consulta,
        "consultas_falladas": falladas,
        "num_consultas_falladas": len(falladas),
    }


res_tfidf = evaluar_metodo(S_tfidf, "TF-IDF")
res_lsa = evaluar_metodo(S_lsa, "LSA")

# ---------------------------------------------------------------------------
# 5. (b) Términos compartidos consulta - documento relevante
#     tokens en minúsculas, patrón por defecto: palabras de 2+ caracteres
# ---------------------------------------------------------------------------
analyzer = vectorizer.build_analyzer()
vocabulario = set(vectorizer.vocabulary_.keys())

terminos_compartidos = {}
for q in query_ids:
    rel = relevante[q]
    toks_q = sorted(set(analyzer(consultas[q][0])))
    toks_d = sorted(set(analyzer(corpus[rel])))
    compartidos = sorted(set(toks_q) & set(toks_d))
    terminos_compartidos[q] = {
        "relevante": rel,
        "tokens_consulta": toks_q,
        "tokens_documento": toks_d,
        "terminos_compartidos": compartidos,
        "num_terminos_compartidos": len(compartidos),
        "compartidos_presentes_en_vocabulario_tfidf": sorted(
            t for t in compartidos if t in vocabulario
        ),
    }

# ---------------------------------------------------------------------------
# 6. Salida: resultados.json
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "s3 - Analisis de fallos por consulta (Parte 3)",
    "config": {
        "corpus": "d01-d10 (textos literales del enunciado)",
        "num_documentos": len(doc_ids),
        "num_consultas": len(query_ids),
        "juicios_de_relevancia": relevante,
        "vectorizer": "TfidfVectorizer (parametros por defecto)",
        "tam_vocabulario": int(len(vocabulario)),
        "lsa": "TruncatedSVD(n_components=4, random_state=0)",
        "varianza_explicada_lsa": float(svd.explained_variance_ratio_.sum()),
        "definicion_de_fallo": "rank del relevante > 3 (fuera del top 3)",
    },
    "rank_relevante_por_consulta": {
        "TF-IDF": {q: res_tfidf["por_consulta"][q]["rank_relevante"] for q in query_ids},
        "LSA": {q: res_lsa["por_consulta"][q]["rank_relevante"] for q in query_ids},
    },
    "consultas_falladas": {
        "TF-IDF": res_tfidf["consultas_falladas"],
        "LSA": res_lsa["consultas_falladas"],
    },
    "metricas_recomputadas": {
        "TF-IDF": res_tfidf["metricas"],
        "LSA": res_lsa["metricas"],
    },
    "terminos_compartidos_consulta_relevante": terminos_compartidos,
    "detalle_por_metodo": {"TF-IDF": res_tfidf, "LSA": res_lsa},
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 7. Impresión de los mismos números en stdout
# ---------------------------------------------------------------------------
lin = "=" * 74
print(lin)
print("s3 - Análisis de fallos por consulta (Parte 3)")
print(lin)
print(
    "Corpus d01-d10 | %d consultas | TfidfVectorizer por defecto (vocabulario=%d)"
    % (len(query_ids), len(vocabulario))
)
print(
    "LSA: TruncatedSVD(n_components=4, random_state=0), varianza explicada=%.4f"
    % resultados["config"]["varianza_explicada_lsa"]
)
print("Fallo = relevante fuera del top 3 (rank > 3)")
print()

print("(a) Rank del documento relevante por método")
print("-" * 74)
print("%-9s %-10s %14s %14s" % ("consulta", "relevante", "rank TF-IDF", "rank LSA"))
for q in query_ids:
    print(
        "%-9s %-10s %14d %14d"
        % (
            q,
            relevante[q],
            res_tfidf["por_consulta"][q]["rank_relevante"],
            res_lsa["por_consulta"][q]["rank_relevante"],
        )
    )
print()

for nombre, res in (("TF-IDF", res_tfidf), ("LSA", res_lsa)):
    m = res["metricas"]
    print(
        "Consultas falladas por %s (relevante fuera del top 3): %d"
        % (nombre, res["num_consultas_falladas"])
    )
    if res["consultas_falladas"]:
        for fallo in res["consultas_falladas"]:
            print(
                "   %s -> relevante %s en rank %d"
                % (fallo["consulta"], fallo["relevante"], fallo["rank_relevante"])
            )
    else:
        print("   ninguna")
    print(
        "   Métricas recomputadas: Hit@1=%.4f  Hit@3=%.4f  MRR=%.4f"
        % (m["hit_at_1"], m["hit_at_3"], m["mrr"])
    )
    print()

print("(b) Términos compartidos entre cada consulta y su documento relevante")
print("-" * 74)
for q in query_ids:
    info = terminos_compartidos[q]
    print(
        "%s (relevante %s) comparte %d término(s): %s"
        % (
            q,
            info["relevante"],
            info["num_terminos_compartidos"],
            info["terminos_compartidos"],
        )
    )
    print("    tokens consulta  : %s" % (info["tokens_consulta"],))
    print("    tokens documento : %s" % (info["tokens_documento"],))
print()
print("Resultados escritos en resultados.json")
