"""Draw the statement graph saved by a run: its sections and their `depende_de` edges.

    uv run python -m investigation_agent.draw_graph corridas/<task>/grafo.json informe/grafo.png

Reference edges (extracted by rule from "Parte N" mentions) are drawn solid with the text
that created them; preamble edges are dotted. Each section shows how many entities it
mentions and how many of those also appear in the course notes.
"""

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from investigation_agent.services.knowledge_graph import DEPENDS_ON, KnowledgeGraphService  # noqa: E402


def draw(graph_path: str | Path, out_path: str | Path) -> Path:
    kg = KnowledgeGraphService()
    graph = kg.load(graph_path)
    source = graph.graph.get("source", "")
    sections = kg.sections(graph)
    pos = {n: (0.0, -i * 1.6) for i, n in enumerate(sections)}

    fig, ax = plt.subplots(figsize=(12, 1.6 * len(sections) + 1))
    ax.set_xlim(-10, 12)
    ax.set_ylim(-1.6 * len(sections) + 0.6, 0.9)
    boxes = {}
    for n in sections:
        d = graph.nodes[n]
        entities = [v for _, v, e in graph.out_edges(n, data=True) if e.get("kind") == "menciona"]
        shared = [e for e in entities if any(s != source for s in graph.nodes[e]["sources"])]
        title = textwrap.shorten(d["title"] or n, 48)
        label = f"{title}\n[{n}] · {len(entities)} entidades, {len(shared)} también en las notas"
        x, y = pos[n]
        boxes[n] = ax.text(x, y, label, ha="center", va="center", fontsize=9,
                           bbox={"boxstyle": "round,pad=0.5", "fc": "#eef2ff", "ec": "#4c51bf"})

    # Box edges in data coordinates, to draw each edge as a bracket outside the boxes.
    fig.canvas.draw()
    to_data = ax.transData.inverted()
    edges = {}
    for n, t in boxes.items():
        (x0, _), (x1, _) = to_data.transform(t.get_bbox_patch().get_window_extent())
        edges[n] = (x0, x1)
    right = max(x1 for _, x1 in edges.values())
    left = min(x0 for x0, _ in edges.values())

    lanes = {"reference": 0, "preamble": 0}
    for u, v, e in sorted(graph.edges(data=True), key=lambda x: abs(pos[x[0]][1] - pos[x[1]][1])
                          if x[0] in pos and x[1] in pos else 0):
        if e.get("kind") != DEPENDS_ON or u not in pos or v not in pos:
            continue
        reference = e.get("rule") == "reference"
        kind = "reference" if reference else "preamble"
        lanes[kind] += 1
        y1, y2 = pos[u][1] + 0.12, pos[v][1] - 0.12
        if reference:  # right side, one lane per edge: u depends on v, the arrow points at v
            xa, xb, xl = edges[u][1], edges[v][1], right + 0.55 * lanes[kind]
        else:
            xa, xb, xl = edges[u][0], edges[v][0], left - 0.35 * lanes[kind]
        path = MplPath([(xa, y1), (xl, y1), (xl, y2), (xb, y2)])
        ax.add_patch(FancyArrowPatch(
            path=path, arrowstyle="-|>", mutation_scale=14, zorder=3,
            lw=1.8 if reference else 0.8, color="#c53030" if reference else "#a0aec0",
            ls="-" if reference else ":"))
        if reference:
            ax.text(right + 0.55 * 3 + 0.4, pos[v][1] - 0.45,
                    f"{u} → {v}\n" + textwrap.fill(f"«…{_words(e.get('evidence', ''))}…»", 34),
                    fontsize=7.5, color="#c53030", va="center")

    ax.plot([], [], color="#c53030", lw=1.8, label="depende_de · regla: referencia «Parte N»")
    ax.plot([], [], color="#a0aec0", lw=0.8, ls=":", label="depende_de · regla: preámbulo")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title(f"Grafo del enunciado: {source}", fontsize=11)
    ax.axis("off")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _words(snippet: str) -> str:
    """The evidence snippet without the words its fixed-width window cut in half."""
    words = snippet.split()
    return " ".join(words[1:-1]) if len(words) > 2 else snippet.strip()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m investigation_agent.draw_graph <grafo.json> <out.png>")
    print(draw(sys.argv[1], sys.argv[2]))
