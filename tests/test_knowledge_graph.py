"""Correction 2: the deterministic skeleton (sections + `depende_de` by rule) and the
entity layer merged by normalized name. This is the Parte 0.b failure, fixed."""

import networkx as nx

from investigation_agent.services.knowledge_graph import KnowledgeGraphService
from investigation_agent.services.pdf_reader import PdfReaderService


def _skeleton(settings, pdf):
    return KnowledgeGraphService().build_skeleton(PdfReaderService(settings).read(pdf))


def test_reference_edge_parte3_to_parte1(settings, task_a):
    g = _skeleton(settings, task_a)
    edge = g.edges["parte-3", "parte-1"]
    assert edge["kind"] == "depende_de" and edge["rule"] == "reference"
    assert "Parte 1" in edge["evidence"]


def test_every_section_depends_on_preamble(settings, task_a):
    g = _skeleton(settings, task_a)
    for n in KnowledgeGraphService.sections(g):
        if n != "preambulo":
            assert g.has_edge(n, "preambulo")


def test_section_context_carries_the_data_of_parte1(settings, task_a):
    """The question of 0.b: with which data and split is the learning curve computed?
    The answer lives in Parte 1, which flat similarity never retrieves."""
    kg = KnowledgeGraphService()
    g = _skeleton(settings, task_a)
    ctx = kg.section_context(g, "parte-3")
    assert [c["id"] for c in ctx][-1] == "parte-3"
    assert "parte-1" in [c["id"] for c in ctx]
    text = "\n".join(c["text"] for c in ctx)
    for fact in ["load_breast_cancer", "random_state", "estratific"]:
        assert fact in text


def test_entities_merge_by_normalized_name():
    kg = KnowledgeGraphService()
    g = nx.DiGraph()
    g.add_node("parte-2", kind="section")
    kg.add_entities(g, "parte-2", "tarea.pdf",
                    [{"name": "Naive Bayes", "type": "metodo", "description": "generativo"}], [])
    other = nx.DiGraph()
    kg.add_entities(other, "nota", "s1.md",
                    [{"name": "naive  bayes", "type": "metodo",
                      "description": "modelo generativo con independencia condicional"}], [])
    kg.merge(g, other)
    entities = [n for n, d in g.nodes(data=True) if d["kind"] == "entity"]
    assert entities == ["ent:naive-bayes"]
    assert g.nodes["ent:naive-bayes"]["sources"] == ["s1.md", "tarea.pdf"]
    assert "independencia" in g.nodes["ent:naive-bayes"]["description"]
