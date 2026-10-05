"""The orchestrator: routes decided by code, the contract on every path (correction 6)."""

import json

from investigation_agent.brakes import NullRag, ScriptedChatModel
from investigation_agent.graph.orchestrator import (
    Solver, diagram, route_after_critic, route_after_guard, route_after_validation)


def test_route_after_validation(settings):
    route = route_after_validation(settings)
    assert route({"plan_errors": []}) == "next_subtask"
    assert route({"plan_errors": ["x"], "plan_attempts": 1}) == "plan"
    assert route({"plan_errors": ["x"], "plan_attempts": settings.max_plan_attempts}) == "finish"


def test_route_after_guard_and_critic():
    state = {"current": "s1", "guard": {"allowed": False},
             "tasks": {"s1": {"status": "pending"}}}
    assert route_after_guard(state) == "program"
    state["tasks"]["s1"]["status"] = "failed"
    assert route_after_guard(state) == "next_subtask"
    state["guard"]["allowed"] = True
    assert route_after_guard(state) == "execute"
    assert route_after_critic({"current": "s1", "tasks": {"s1": {"status": "approved"}}}) \
        == "next_subtask"
    assert route_after_critic({"current": "s1", "tasks": {"s1": {"status": "pending"}}}) \
        == "program"


def test_diagram_has_the_conditional_edges():
    mermaid = diagram()
    for node in ["validate_plan", "guard", "critic", "check_deliverable", "finish"]:
        assert node in mermaid
    assert "-.->" in mermaid  # conditional edges are dashed


def test_failed_run_still_returns_the_contract_and_the_trace(settings, tmp_path):
    solver = Solver(settings, llm=ScriptedChatModel(scripts={}, calls={}), rag=NullRag())
    result = solver.solve(str(tmp_path / "missing.pdf"), str(tmp_path / "out"))
    assert result["status"] == "fallido"
    assert set(result) >= {"status", "entregables", "subtareas", "usage", "model", "trace"}
    assert result["usage"] == {"tokens_entrada": 0, "tokens_salida": 0}
    events = [json.loads(line) for line in open(result["trace"])]
    assert events[0]["kind"] == "start"
    assert events[-1]["kind"] == "error" and "PdfReadError" in events[-1]["error"]


def test_invalid_plans_end_in_failure_after_the_cap(settings, tmp_path, task_a):
    """A planner that never covers the sections: three plans, three rejections, no work."""
    from investigation_agent.brakes import Reply, _no_entities

    bad = json.dumps({"deliverable": {"format": "md", "filename": "reporte.md"},
                      "subtasks": [{"id": "s1", "type": "code", "section": "parte-1",
                                    "description": "x"}]})
    model = ScriptedChatModel(scripts={"indexer_entities": _no_entities,
                                       "planner": lambda u, n: Reply(bad)}, calls={})
    result = Solver(settings, llm=model, rag=NullRag()).solve(str(task_a), str(tmp_path))
    assert result["status"] == "fallido" and result["entregables"] == []
    events = [json.loads(line) for line in open(result["trace"])]
    rejected = [e for e in events if e.get("node") == "plan_validator" and not e["valid"]]
    assert len(rejected) == settings.max_plan_attempts
    assert "not covered" in rejected[0]["errors"][0]
