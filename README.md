# Solver multiagente con GraphRAG

Taller 03 v2 de MMIA 6013 — IA Generativa y Agentes (USFQ). El solver recibe el enunciado de una
tarea en PDF, lo descompone en subtareas, recupera el contexto con un grafo de conocimiento, escribe
y ejecuta código en un sandbox, lo critica y redacta el entregable (Markdown, PDF o notebook
ejecutado). El informe está en [`informe/`](informe/).

## Requisitos

- Python 3.13 y [uv](https://docs.astral.sh/uv/)
- VPN GlobalProtect de la USFQ (LLM en el vLLM de la H200, embeddings `bge-m3` en su Ollama)
- Qdrant local en Docker

```bash
uv sync
docker run -d -p 6333:6333 qdrant/qdrant:v1.19.1
```

No hace falta ninguna clave. La configuración está en
[`config/settings.py`](src/investigation_agent/config/settings.py) y se puede cambiar con
variables de entorno.

## Uso

```bash
# Resolver un enunciado
uv run python -m investigation_agent <enunciado.pdf> [carpeta_salida]

# Evaluar con el golden set (variante completa)
uv run python evaluacion/evaluar_solver.py --solver investigation_agent.graph.orchestrator:Solver \
    --corridas corridas/completo --salida resultados/completo/resultados_solver.csv

# Ablación sin grafo y extensión en paralelo: misma orden con
#   ABLATION_NO_GRAPH=true   (→ corridas/sin_grafo, resultados/sin_grafo)
#   PARALLEL_SUBTASKS=true   (→ corridas/paralelo,  resultados/paralelo)

# Tablas del informe a partir de los CSV y las trazas
uv run python evaluacion/tablas.py

# Frenos forzados con un modelo de guion, sin H200 (Parte 3)
echo n | uv run python -m investigation_agent.brakes

# Pruebas, sin H200
uv run pytest
```

## Estructura

| Carpeta | Contenido |
|---|---|
| `src/investigation_agent/` | `graph/` (orquestador LangGraph y nodos), `services/` (lector de PDF, grafo, RAG, sandbox, comprobaciones, traza), `brakes.py` |
| `tests/` | pruebas sin modelo |
| `evaluacion/` | golden set (`golden_tareas.json`), evaluador del kit con cambios marcados, enunciados de las tareas reales |
| `corridas/` | por variante y tarea: `traza.jsonl`, `plan.json`, `grafo.json`, scripts y entregable; `frenos/` |
| `resultados/` | `resultados_solver.csv` y `resumen_solver.csv` crudos de cada variante, `tablas.md` |
| `informe/` | informe y salidas de la Parte 0 |
| `taller-03-v2-solver-multiagente/` | kit del curso: `solver-v2/` (enunciados, scripts de la Parte 0, evaluador original) y `notas-teoricas/` |
