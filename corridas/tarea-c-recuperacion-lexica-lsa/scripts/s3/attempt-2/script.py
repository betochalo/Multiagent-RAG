# -*- coding: utf-8 -*-
"""
Tarea C - Parte 3 (subtask s3): análisis de fallos por consulta.

Recomputa los dos rankings de las Partes 1-2 con los mismos parámetros:
  * TF-IDF: TfidfVectorizer() por defecto + similitud coseno.
  * LSA:    TruncatedSVD(n_components=4, random_state=0) sobre la matriz
            TF-IDF del corpus + normalización L2 + similitud coseno.

Para cada método y consulta determina si el documento relevante queda en el
top-3, lista las consultas fallidas de cada método y, para cada fallo,
tokeniza consulta y documento relevante con la tokenización por defecto del
vectorizador (minúsculas, tokens de 2+ caracteres), lista los términos
compartidos con sus pesos TF-IDF, señala explícitamente si el solape es
vacío o está formado solo por stop words, y muestra el top-3 recuperado
en su lugar.

Todos los números se escriben en resultados.json y se imprimen por stdout.
"""

import json
import sys

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# 1. Datos literales del enunciado (corpus, consultas y juicios de relevancia)
# ---------------------------------------------------------------------------
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

DOC_IDS = sorted(DOCS)
QUERY_IDS = sorted(QUERIES)
TEXTO_Q = {q: QUERIES[q][0] for q in QUERY_IDS}
RELEVANTE = {q: QUERIES[q][1] for q in QUERY_IDS}

# Stop words de referencia en español. El TfidfVectorizer por defecto NO las
# elimina (stop_words=None); esta lista se usa SOLO para clasificar el solape
# consulta-documento como vacío / solo stop words / con términos de contenido.
STOP_WORDS_ES = set("""
a al algo alguna algunas alguno algunos ante antes aqui asi aun bien cada como
con contra cual cuales cuando de del desde donde dos el ella ellas ello ellos
en entre era eran es esa esas ese eso esta estaba estan estar estas este esto
estos fue fueron ha habia han hasta hay la las le les lo los mas me mi mis
mucho muy nada ni no nos nuestra nuestro o otra otras otro otros para pero
poco por porque que quien quienes se segun ser si sin sobre solo son su sus
tal tambien tan tanto te tiene tienen toda todas todo todos tu tus un una
unas unos usted ustedes va van ya yo qué más está están así también según aquí
""".split())

# ---------------------------------------------------------------------------
# 2. Recomputación de los rankings (mismos parámetros que las Partes 1 y 2)
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # todos los parámetros por defecto
X_docs = vectorizer.fit_transform([DOCS[d] for d in DOC_IDS])
X_queries = vectorizer.transform([TEXTO_Q[q] for q in QUERY_IDS])

# --- Método 1: TF-IDF + coseno (los vectores TF-IDF ya salen L2-normalizados,
#     por lo que el producto escalar es la similitud coseno)
sims_tfidf = (X_queries @ X_docs.T).toarray()

# --- Método 2: LSA = TruncatedSVD(4, random_state=0) + normalización L2 + coseno
svd = TruncatedSVD(n_components=4, random_state=0)
normalizer = Normalizer()
docs_lsa = normalizer.fit_transform(svd.fit_transform(X_docs))
queries_lsa = normalizer.transform(svd.transform(X_queries))
sims_lsa = queries_lsa @ docs_lsa.T


def evaluar_metodo(sims, nombre):
    ranking, rango, sims_json = {}, {}, {}
    for i, q in enumerate(QUERY_IDS):
        orden = np.argsort(-sims[i], kind="stable")
        ranking[q] = [DOC_IDS[j] for j in orden]
        rango[q] = int(ranking[q].index(RELEVANTE[q]) + 1)
        sims_json[q] = {DOC_IDS[j]: float(sims[i, j]) for j in range(len(DOC_IDS))}
    return {
        "metodo": nombre,
        "similitudes_coseno": sims_json,
        "ranking": ranking,
        "rango_del_relevante": rango,
        "hit_at_1": float(np.mean([rango[q] == 1 for q in QUERY_IDS])),
        "hit_at_3": float(np.mean([rango[q] <= 3 for q in QUERY_IDS])),
        "mrr": float(np.mean([1.0 / rango[q] for q in QUERY_IDS])),
        "consultas_fallidas": [q for q in QUERY_IDS if rango[q] > 3],
    }


ev_tfidf = evaluar_metodo(sims_tfidf, "tfidf")
ev_lsa = evaluar_metodo(sims_lsa, "lsa")

# ---------------------------------------------------------------------------
# 3. Tokenización por defecto del vectorizador y solape consulta-relevante
# ---------------------------------------------------------------------------
analizador = vectorizer.build_analyzer()  # minúsculas + tokens de 2+ caracteres
tokens_consulta = {q: analizador(TEXTO_Q[q]) for q in QUERY_IDS}
tokens_docs = {d: analizador(DOCS[d]) for d in DOC_IDS}

X_docs_dense = X_docs.toarray()
X_queries_dense = X_queries.toarray()
vocab = vectorizer.vocabulary_


def peso_tfidf(matriz, fila, termino):
    j = vocab.get(termino)
    return None if j is None else float(matriz[fila, j])


terminos_compartidos = {}
for q in QUERY_IDS:
    d = RELEVANTE[q]
    comunes = sorted(set(tokens_consulta[q]) & set(tokens_docs[d]))
    i, k = QUERY_IDS.index(q), DOC_IDS.index(d)
    terminos_compartidos[q] = [
        {
            "termino": t,
            "es_stop_word": bool(t in STOP_WORDS_ES),
            "peso_tfidf_consulta": peso_tfidf(X_queries_dense, i, t),
            "peso_tfidf_documento": peso_tfidf(X_docs_dense, k, t),
        }
        for t in comunes
    ]


def estado_solape(q):
    lista = terminos_compartidos[q]
    if not lista:
        return "vacio"
    if all(t["es_stop_word"] for t in lista):
        return "solo_stop_words"
    return "con_terminos_de_contenido"


# ---------------------------------------------------------------------------
# 4. Análisis de fallos (relevante fuera del top-3)
# ---------------------------------------------------------------------------
def analizar_fallos(ev, sims, nombre_metodo):
    analisis = {}
    for q in ev["consultas_fallidas"]:
        d = RELEVANTE[q]
        i, k = QUERY_IDS.index(q), DOC_IDS.index(d)
        estado = estado_solape(q)
        sw = [t["termino"] for t in terminos_compartidos[q] if t["es_stop_word"]]
        cont = [t["termino"] for t in terminos_compartidos[q] if not t["es_stop_word"]]
        if estado == "vacio":
            razon = ("El solape es vacío: la consulta no comparte ningún token con su "
                     "documento relevante, de modo que el método no tiene señales "
                     "directas para recuperarlo.")
        elif estado == "solo_stop_words":
            razon = ("El solape está formado solo por stop words (" + ", ".join(sw) +
                     "), términos muy frecuentes en el corpus y con poco poder "
                     "discriminativo; sin términos de contenido compartidos, el "
                     "relevante no alcanza el top-3.")
        else:
            razon = ("La consulta y el relevante comparten términos de contenido (" +
                     ", ".join(cont) + "), pero otros documentos obtienen mayor "
                     "similitud y desplazan al relevante fuera del top-3.")
        if nombre_metodo == "lsa":
            razon += (" La proyección latente a 4 dimensiones funde temas distintos y "
                      "no llega a situar el relevante entre los tres primeros.")
        analisis[q] = {
            "consulta": q,
            "texto_consulta": TEXTO_Q[q],
            "documento_relevante": d,
            "texto_documento_relevante": DOCS[d],
            "rango_del_relevante": ev["rango_del_relevante"][q],
            "top3_recuperado_en_su_lugar": ev["ranking"][q][:3],
            "tokens_consulta": tokens_consulta[q],
            "tokens_documento_relevante": tokens_docs[d],
            "terminos_compartidos": terminos_compartidos[q],
            "estado_del_solape": estado,
            "similitud_coseno_consulta_relevante": float(sims[i, k]),
            "explicacion": razon,
        }
    return analisis


fallos_tfidf = analizar_fallos(ev_tfidf, sims_tfidf, "tfidf")
fallos_lsa = analizar_fallos(ev_lsa, sims_lsa, "lsa")

# ---------------------------------------------------------------------------
# 5. Tabla resumen (consulta, método, rango, top-3, términos compartidos)
# ---------------------------------------------------------------------------
tabla_resumen = []
for q in QUERY_IDS:
    for metodo, ev in (("tfidf", ev_tfidf), ("lsa", ev_lsa)):
        tabla_resumen.append({
            "consulta": q,
            "metodo": metodo,
            "rango_del_relevante": ev["rango_del_relevante"][q],
            "relevante_en_top3": bool(ev["rango_del_relevante"][q] <= 3),
            "top3": ev["ranking"][q][:3],
            "terminos_compartidos": [t["termino"] for t in terminos_compartidos[q]],
            "estado_del_solape": estado_solape(q),
        })


def terminos_como_cadena(q):
    if not terminos_compartidos[q]:
        return "(vacío)"
    return ", ".join(
        (t["termino"] + "*" if t["es_stop_word"] else t["termino"])
        for t in terminos_compartidos[q]
    )


# ---------------------------------------------------------------------------
# 6. Salida por stdout
# ---------------------------------------------------------------------------
print("=" * 100)
print("Tarea C - Parte 3: análisis de fallos por consulta (TF-IDF frente a LSA)")
print("=" * 100)
print("Parámetros: TfidfVectorizer() por defecto + coseno | "
      "TruncatedSVD(n_components=4, random_state=0) + L2 + coseno")

for ev, sims in ((ev_tfidf, sims_tfidf), (ev_lsa, sims_lsa)):
    print("\n" + "-" * 100)
    print("Método: " + ev["metodo"])
    print("-" * 100)
    print("Similitudes coseno consulta-documento:")
    print("        " + " ".join(f"{d:>8}" for d in DOC_IDS))
    for i, q in enumerate(QUERY_IDS):
        print(f"{q:<8}" + " ".join(f"{sims[i, j]:8.4f}" for j in range(len(DOC_IDS))))
    print("Rankings (rango del relevante entre corchetes):")
    for q in QUERY_IDS:
        print(f"  {q}: " + " > ".join(ev["ranking"][q]) +
              f"  [relevante {RELEVANTE[q]}: rango {ev['rango_del_relevante'][q]}]")
    print(f"Métricas agregadas: Hit@1={ev['hit_at_1']:.4f}  "
          f"Hit@3={ev['hit_at_3']:.4f}  MRR={ev['mrr']:.4f}")
    fallidas = ev["consultas_fallidas"]
    print("Consultas fallidas (relevante fuera del top-3): " +
          (", ".join(fallidas) if fallidas else "ninguna"))

print("\nVarianza explicada por las 4 componentes LSA: " +
      ", ".join(f"{v:.4f}" for v in svd.explained_variance_ratio_))

for nombre, fallos in (("tfidf", fallos_tfidf), ("lsa", fallos_lsa)):
    print("\n" + "=" * 100)
    print(f"Análisis detallado de fallos ({nombre})")
    print("=" * 100)
    if not fallos:
        print("  El relevante está en el top-3 para todas las consultas: sin fallos.")
    for q, a in fallos.items():
        print(f"\n  {q}: \"{a['texto_consulta']}\"  ->  relevante {a['documento_relevante']} "
              f"en rango {a['rango_del_relevante']} (fuera del top-3)")
        print("  Top-3 recuperado en su lugar: " + ", ".join(a["top3_recuperado_en_su_lugar"]))
        print("  Tokens consulta: " + ", ".join(a["tokens_consulta"]))
        print(f"  Tokens {a['documento_relevante']}: " + ", ".join(a["tokens_documento_relevante"]))
        if not a["terminos_compartidos"]:
            print("  Términos compartidos: NINGUNO -> solape VACÍO")
        else:
            print("  Términos compartidos (peso TF-IDF en consulta | en documento):")
            for t in a["terminos_compartidos"]:
                marca = "  [stop word]" if t["es_stop_word"] else ""
                pc = t["peso_tfidf_consulta"]
                pd_ = t["peso_tfidf_documento"]
                pc_s = "n/a" if pc is None else f"{pc:.4f}"
                pd_s = "n/a" if pd_ is None else f"{pd_:.4f}"
                print(f"    - {t['termino']}{marca}: {pc_s} | {pd_s}")
        print("  Estado del solape: " + a["estado_del_solape"])
        print("  Explicación: " + a["explicacion"])

print("\n" + "=" * 100)
print("Tabla resumen (* = stop word; solape entre la consulta y su documento relevante)")
print("=" * 100)
print(f"{'consulta':<9}{'método':<8}{'rango':<6}{'¿top3?':<7}{'top-3':<28}términos compartidos")
print("-" * 100)
for fila in tabla_resumen:
    print(f"{fila['consulta']:<9}{fila['metodo']:<8}{fila['rango_del_relevante']:<6}"
          f"{'SÍ' if fila['relevante_en_top3'] else 'NO':<7}"
          f"{', '.join(fila['top3']):<28}{terminos_como_cadena(fila['consulta'])}")

# ---------------------------------------------------------------------------
# 7. Resultados -> resultados.json
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "s3 - Parte 3: análisis de fallos por consulta",
    "parametros": {
        "tfidf": ("TfidfVectorizer() por defecto (lowercase=True, "
                  "token_pattern='(?u)\\\\b\\\\w\\\\w+\\\\b', norm='l2') + similitud coseno"),
        "lsa": ("TruncatedSVD(n_components=4, random_state=0) ajustado sobre la matriz "
                "TF-IDF del corpus + Normalizer(L2) + similitud coseno"),
        "criterio_de_fallo": "el documento relevante queda fuera del top-3",
        "nota_stop_words": ("El TfidfVectorizer por defecto NO elimina stop words; la lista "
                            "stop_words_de_referencia se usa solo para clasificar el solape "
                            "como vacío, solo stop words o con términos de contenido."),
        "stop_words_de_referencia": sorted(STOP_WORDS_ES),
    },
    "tfidf": ev_tfidf,
    "lsa": dict(ev_lsa, varianza_explicada=[float(v) for v in svd.explained_variance_ratio_]),
    "tokens": {"consultas": tokens_consulta, "documentos": tokens_docs},
    "terminos_compartidos_consulta_relevante": terminos_compartidos,
    "analisis_de_fallos": {"tfidf": fallos_tfidf, "lsa": fallos_lsa},
    "tabla_resumen": tabla_resumen,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print("\nResultados escritos en resultados.json")
