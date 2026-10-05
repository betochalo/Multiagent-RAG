"""Parte 4, option A: ready subtasks run at once, round-robin over the replicas."""

import json

from investigation_agent.brakes import (
    GOOD, TASK_A, NullRag, Reply, ScriptedChatModel, _approve, _no_entities, _plan, _writer)
from investigation_agent.graph.orchestrator import Solver


def _plan_with_dependency(user: str, n: int) -> Reply:
    """Four code subtasks; s4 depends on s1, so it must wait for a second wave."""
    reply = _plan(user, n, code_subtasks=4)
    plan = json.loads(reply.content)
    plan["subtasks"][3]["depends_on"] = ["s1"]
    return Reply(json.dumps(plan), reply.tokens_in, reply.tokens_out)


def _run(settings, tmp_path, parallel: bool):
    s = settings.model_copy(update={"parallel_subtasks": parallel})
    model = ScriptedChatModel(scripts={
        "indexer_entities": _no_entities, "planner": _plan_with_dependency,
        "programmer": lambda u, n: Reply(GOOD, 300, 400), "critic": _approve,
        "writer_report": _writer}, calls={})
    result = Solver(s, llm=model, rag=NullRag()).solve(str(TASK_A), str(tmp_path / str(parallel)))
    events = [json.loads(line) for line in open(result["trace"])]
    return result, events


def test_waves_follow_the_dependencies(settings, tmp_path):
    result, events = _run(settings, tmp_path, parallel=True)
    waves = [e for e in events if e.get("node") == "dispatch"]
    assert waves[0]["wave"] == ["s1", "s2", "s3"]
    assert waves[1]["wave"] == ["s4"]  # waited for s1
    assert waves[-1]["wave"] == []      # → writer
    # round-robin over the two replicas, and every call says which one served it
    assert set(waves[0]["replicas"].values()) == {0, 1}
    starts = {e["subtask"]: e["replica"] for e in events if e.get("node") == "subtask_start"}
    assert starts == {"s1": 0, "s2": 1, "s3": 0, "s4": 0}
    assert all("replica" in e for e in events if e["kind"] == "llm_call")
    assert result["status"] == "completado"


def test_same_outcome_as_sequential(settings, tmp_path):
    parallel, _ = _run(settings, tmp_path, parallel=True)
    sequential, events = _run(settings, tmp_path, parallel=False)
    assert parallel["subtareas"] == sequential["subtareas"]
    assert parallel["status"] == sequential["status"] == "completado"
    assert not any(e.get("node") == "dispatch" for e in events)  # baseline untouched
