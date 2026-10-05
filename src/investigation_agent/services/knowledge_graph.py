"""Deterministic skeleton of the task's knowledge graph: sections as nodes, cross-references
as `depende_de` edges. Rules, not the LLM, build it.

A flat retriever ranks fragments by similarity to the question, and the section that
holds the answer often does not resemble the question: "Parte 3" says "la misma división
de la Parte 1", and only Parte 1 says what that split is (Parte 0.b). The edge is literal
text, so a regex finds it every time, for free, and cannot hallucinate one.

Two rules create `depende_de` edges:

- **reference**: a section mentions another ("la Parte 1", "las Partes 2 y 3",
  "la pregunta anterior");
- **preamble**: every section depends on the text before the first section, where task
  statements put shared data (a corpus, a table of logits) that no section cites by name.

The LLM layer (entities, relations, communities) goes on top of this skeleton.
"""

import json
import re
import unicodedata
from pathlib import Path

import networkx as nx

from investigation_agent.services.pdf_reader import ParsedDocument

DEPENDS_ON = "depende_de"
PREAMBLE_ID = "preambulo"

_KINDS = r"parte|pregunta|ejercicio|secci[oó]n|punto|actividad"
# "Parte 3 — La curva de aprendizaje" → kind "parte", number "3"
_NUMBERED_TITLE = re.compile(rf"^\s*({_KINDS})\s+(\d+)\b", re.I)
# "la Parte 1", "las Partes 2 y 3", "Partes 1, 2 y 4"
_REFERENCE = re.compile(rf"\b({_KINDS})s?\s+(\d+(?:\s*(?:,|y|e|o)\s*\d+)*)", re.I)
# "la parte anterior", "la pregunta previa"
_PREVIOUS = re.compile(rf"\b(?:la|el)\s+({_KINDS})\s+(?:anterior|previa|previo)\b", re.I)


class KnowledgeGraphService:
    @staticmethod
    def sections(graph: nx.DiGraph) -> list[str]:
        """Section node ids in document order."""
        return sorted((n for n, d in graph.nodes(data=True) if d.get("kind") == "section"),
                      key=lambda n: graph.nodes[n]["order"])

    def build_skeleton(self, doc: ParsedDocument) -> nx.DiGraph:
        """Correction 2: the deterministic skeleton. Sections are nodes; every cross
        reference ("Parte 1", "la pregunta anterior") is a `depende_de` edge extracted by a
        rule, never by the model."""
        graph = nx.DiGraph(source=doc.source)
        order: list[str] = []
        for i, section in enumerate(doc.sections):
            node_id, kind, number = _identify(section.title, is_first=i == 0)
            if node_id in graph:  # two sections with the same title
                node_id = f"{node_id}-{i}"
            graph.add_node(node_id, kind="section", section_kind=kind, number=number,
                           title=section.title, level=section.level, page=section.page,
                           text=section.text, order=i)
            order.append(node_id)

        by_label = {(d["section_kind"], d["number"]): n for n, d in graph.nodes(data=True)
                    if d["number"] is not None}
        for position, node_id in enumerate(order):
            data = graph.nodes[node_id]
            for target, evidence in _references(data["text"], by_label, order, position, graph):
                if target != node_id:
                    graph.add_edge(node_id, target, kind=DEPENDS_ON, rule="reference",
                                   evidence=evidence)
            if PREAMBLE_ID in graph and node_id != PREAMBLE_ID \
                    and not graph.has_edge(node_id, PREAMBLE_ID):
                graph.add_edge(node_id, PREAMBLE_ID, kind=DEPENDS_ON, rule="preamble",
                               evidence="shared data before the first section")
        return graph

    # ---------------------------------------------------------------- queries
    @staticmethod
    def dependencies(graph: nx.DiGraph, node_id: str) -> list[str]:
        """Everything `node_id` depends on, transitively, ordered as in the document."""
        deps_graph = nx.subgraph_view(
            graph, filter_edge=lambda u, v: graph.edges[u, v].get("kind") == DEPENDS_ON)
        deps = nx.descendants(deps_graph, node_id)
        return sorted(deps, key=lambda n: graph.nodes[n].get("order", 0))

    def section_context(self, graph: nx.DiGraph, node_id: str) -> list[dict]:
        """The literal section plus the closure of its dependencies, each with its citation.
        The researcher always hands this to a subtask, before any similarity search."""
        out = []
        for n in self.dependencies(graph, node_id) + [node_id]:
            d = graph.nodes[n]
            out.append({"id": n, "title": d["title"], "page": d["page"], "text": d["text"],
                        "source": graph.graph.get("source", ""),
                        "role": "section" if n == node_id else "dependency"})
        return out

    # ---------------------------------------------------------------- entity layer
    @staticmethod
    def entity_id(name: str) -> str:
        return f"ent:{_slug(name)}"

    def add_entities(self, graph: nx.DiGraph, section_id: str, source: str,
                     entities: list[dict], relations: list[dict]) -> None:
        """Merge LLM-extracted entities into the graph by normalized name: "Naive Bayes" in
        the statement and "naive bayes" in the course notes are one node, and it remembers
        every source that mentions it. `section_id` gets a `menciona` edge to each entity."""
        for e in entities:
            name = str(e.get("name", "")).strip()
            if not name:
                continue
            eid = self.entity_id(name)
            if eid not in graph:
                graph.add_node(eid, kind="entity", name=name, type=e.get("type", "concepto"),
                               description=e.get("description", ""), sources=[])
            node = graph.nodes[eid]
            if source not in node["sources"]:
                node["sources"].append(source)
            if e.get("description") and len(e["description"]) > len(node["description"]):
                node["description"] = e["description"]
            if section_id in graph:
                graph.add_edge(section_id, eid, kind="menciona")
        for r in relations:
            a, b = self.entity_id(str(r.get("source", ""))), self.entity_id(str(r.get("target", "")))
            if a in graph and b in graph and a != b:
                graph.add_edge(a, b, kind="relacion", label=r.get("type", "relacionado_con"),
                               source=source)

    def merge(self, graph: nx.DiGraph, other: nx.DiGraph) -> None:
        """Merge another graph (the course notes') into `graph`, fusing entities by id."""
        for n, d in other.nodes(data=True):
            if n not in graph:
                graph.add_node(n, **{k: (list(v) if isinstance(v, list) else v)
                                     for k, v in d.items()})
            elif d.get("kind") == "entity":
                node = graph.nodes[n]
                node["sources"] = sorted(set(node.get("sources", [])) | set(d.get("sources", [])))
                if len(d.get("description", "")) > len(node.get("description", "")):
                    node["description"] = d["description"]
        for u, v, d in other.edges(data=True):
            if not graph.has_edge(u, v):
                graph.add_edge(u, v, **d)

    @staticmethod
    def communities(graph: nx.DiGraph, min_size: int) -> list[list[str]]:
        """Louvain communities of the entity subgraph (undirected, fixed seed)."""
        entities = [n for n, d in graph.nodes(data=True) if d.get("kind") == "entity"]
        sub = graph.subgraph(entities).to_undirected()
        if sub.number_of_edges() == 0:
            return []
        found = nx.community.louvain_communities(sub, seed=0)
        return [sorted(c) for c in found if len(c) >= min_size]

    @staticmethod
    def add_community(graph: nx.DiGraph, index: int, members: list[str], summary: str) -> None:
        cid = f"com:{index}"
        graph.add_node(cid, kind="community", summary=summary, size=len(members))
        for m in members:
            graph.add_edge(m, cid, kind="pertenece_a")

    @staticmethod
    def entity_context(graph: nx.DiGraph, section_id: str, max_entities: int = 15) -> dict:
        """Entities the section mentions, their related entities one hop away, and the
        summaries of the communities they belong to (local + global search)."""
        mentioned = [v for _, v, d in graph.out_edges(section_id, data=True)
                     if d.get("kind") == "menciona"]
        related: list[str] = []
        for e in mentioned:
            for nb in list(graph.successors(e)) + list(graph.predecessors(e)):
                if graph.nodes[nb].get("kind") == "entity" and nb not in mentioned \
                        and nb not in related:
                    related.append(nb)
        chosen = (mentioned + related)[:max_entities]
        communities = sorted({c for e in chosen for c in graph.successors(e)
                              if graph.nodes[c].get("kind") == "community"})
        return {
            "entities": [{"name": graph.nodes[e]["name"], "type": graph.nodes[e]["type"],
                          "description": graph.nodes[e]["description"],
                          "sources": graph.nodes[e]["sources"]} for e in chosen],
            "communities": [graph.nodes[c]["summary"] for c in communities],
        }

    # ---------------------------------------------------------------- persistence
    @staticmethod
    def save(graph: nx.DiGraph, path: str | Path) -> None:
        data = nx.node_link_data(graph, edges="edges")
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> nx.DiGraph:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return nx.node_link_graph(data, edges="edges")


# -------------------------------------------------------------------- rules
def _identify(title: str, is_first: bool) -> tuple[str, str | None, str | None]:
    """(node id, section kind, number). "Parte 3 — ..." → ("parte-3", "parte", "3").
    The first section is the preamble: the title plus whatever precedes the first part."""
    m = _NUMBERED_TITLE.match(title)
    if m:
        kind = _slug(m.group(1))
        return f"{kind}-{m.group(2)}", kind, m.group(2)
    if is_first:
        return PREAMBLE_ID, None, None
    return _slug(title) or "seccion", None, None


def _references(text: str, by_label: dict[tuple[str, str], str], order: list[str],
                position: int, graph: nx.DiGraph):
    """(target node, evidence) for every cross-reference in a section's text."""
    for m in _REFERENCE.finditer(text):
        kind = _slug(m.group(1))
        for number in re.findall(r"\d+", m.group(2)):
            target = by_label.get((kind, number))
            if target:
                yield target, _snippet(text, m)
    for m in _PREVIOUS.finditer(text):
        kind = _slug(m.group(1))
        earlier = [n for n in order[:position] if graph.nodes[n]["section_kind"] == kind]
        if earlier:
            yield earlier[-1], _snippet(text, m)


def _snippet(text: str, m: re.Match, width: int = 40) -> str:
    return " ".join(text[max(0, m.start() - width):m.end() + width].split())


def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")
