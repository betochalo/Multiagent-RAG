"""Las tablas de la Parte 2.b, derivadas de los CSV crudos y de las trazas de cada corrida.

    uv run python evaluacion/tablas.py            # → resultados/tablas.md

Por tarea y variante: comprobaciones aprobadas, procedencia, subtareas fallidas u omitidas,
intentos de código, tokens y duración (de resumen_solver.csv y resultados_solver.csv). Los
tokens por agente no están en el resumen del kit: salen del evento `finish` de cada traza.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
VARIANTES = {v: nombre for v, nombre in [("completo", "Completo"), ("sin_grafo", "Sin grafo"),
                                         ("paralelo", "Paralelo (ext. A)")]
             if (RAIZ / "resultados" / v / "resumen_solver.csv").exists()}
AGENTES = ["indexer", "indexer_community", "indexer_course", "planner", "programmer", "critic",
           "writer"]


def leer(variante: str) -> tuple[list[dict], list[dict]]:
    carpeta = RAIZ / "resultados" / variante
    with open(carpeta / "resultados_solver.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    with open(carpeta / "resumen_solver.csv", encoding="utf-8") as f:
        resumen = list(csv.DictReader(f))
    return filas, resumen


def traza(variante: str, tarea: str) -> list[dict]:
    p = RAIZ / "corridas" / variante / f"tarea-{tarea}" / "traza.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def miles(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def main() -> None:
    out = ["# Tablas de la Parte 2.b", "",
           "Generado por `evaluacion/tablas.py` desde `resultados/*/resultados_solver.csv`, "
           "`resultados/*/resumen_solver.csv` y `corridas/*/tarea-*/traza.jsonl`.", ""]

    # ---------------------------------------------------------------- por tarea
    out += ["## Por tarea", "",
            "| Tarea | Variante | Comprobaciones | Procedencia | Subtareas no resueltas | "
            "Intentos de código | Tokens entrada | Tokens salida | Duración (s) | Status |",
            "|---|---|---|---|---|---:|---:|---:|---:|---|"]
    totales = {}
    for variante, nombre in VARIANTES.items():
        filas, resumen = leer(variante)
        t = defaultdict(int)
        for r in resumen:
            proc = next(f["detalle"] for f in filas if f["tarea"] == r["tarea"]
                        and f["tipo"] == "procedencia")
            fin = next(e for e in reversed(traza(variante, r["tarea"])) if e["kind"] == "finish")
            malas = [f"{i} ({s})" for i, s in fin["subtasks"].items() if s in {"failed", "skipped"}]
            out.append(f"| {r['tarea']} | {nombre} | {r['aprobadas']}/{r['total']} | {proc} | "
                       f"{', '.join(malas) or '—'} | {r['intentos_codigo']} | "
                       f"{miles(int(r['tokens_entrada']))} | {miles(int(r['tokens_salida']))} | "
                       f"{float(r['duracion_s']):.0f} | {r['status']} |")
            for k in ("aprobadas", "total", "intentos_codigo", "tokens_entrada", "tokens_salida"):
                t[k] += int(r[k])
            t["duracion_s"] += float(r["duracion_s"])
            t["no_resueltas"] += len(malas)
        totales[variante] = t
    out += ["", "| Variante | Comprobaciones | Subtareas no resueltas | Intentos de código | "
            "Tokens (entrada + salida) | Duración total (min) |", "|---|---|---:|---:|---:|---:|"]
    for variante, nombre in VARIANTES.items():
        t = totales[variante]
        out.append(f"| {nombre} | {t['aprobadas']}/{t['total']} | {t['no_resueltas']} | "
                   f"{t['intentos_codigo']} | {miles(t['tokens_entrada'] + t['tokens_salida'])} | "
                   f"{t['duracion_s'] / 60:.1f} |")

    # ---------------------------------------------------------------- por agente
    out += ["", "## Tokens por agente (entrada + salida, suma de las 5 tareas)", "",
            "| Agente | " + " | ".join(VARIANTES.values()) + " |",
            "|---|" + "---:|" * len(VARIANTES)]
    por_agente = {v: defaultdict(lambda: [0, 0, 0]) for v in VARIANTES}
    for variante in VARIANTES:
        _, resumen = leer(variante)
        for r in resumen:
            fin = next(e for e in reversed(traza(variante, r["tarea"])) if e["kind"] == "finish")
            for agente, uso in fin["usage_by_agent"].items():
                acc = por_agente[variante][agente]
                acc[0] += uso["calls"]
                acc[1] += uso["tokens_in"]
                acc[2] += uso["tokens_out"]
    for agente in AGENTES:
        celdas = []
        for variante in VARIANTES:
            calls, tin, tout = por_agente[variante].get(agente, [0, 0, 0])
            celdas.append(f"{miles(tin + tout)} ({calls} llamadas)" if calls else "—")
        out.append(f"| {agente} | " + " | ".join(celdas) + " |")

    out += ["", "## Tokens por agente y tarea (entrada + salida)", ""]
    for variante, nombre in VARIANTES.items():
        _, resumen = leer(variante)
        out += [f"**{nombre}**", "", "| Tarea | " + " | ".join(AGENTES) + " |",
                "|---|" + "---:|" * len(AGENTES)]
        for r in resumen:
            fin = next(e for e in reversed(traza(variante, r["tarea"])) if e["kind"] == "finish")
            uso = fin["usage_by_agent"]
            out.append(f"| {r['tarea']} | " + " | ".join(
                miles(uso[a]["tokens_in"] + uso[a]["tokens_out"]) if a in uso else "—"
                for a in AGENTES) + " |")
        out.append("")

    destino = RAIZ / "resultados" / "tablas.md"
    destino.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
