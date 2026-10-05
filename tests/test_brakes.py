"""Parte 3: each brake forced by a real run with the scripted model, asserted on the trace."""

import json

import pytest

from investigation_agent.brakes import run_brake


def _run(name, settings, tmp_path):
    result = run_brake(name, tmp_path / name, settings)
    events = [json.loads(line) for line in open(result["trace"])]
    return result, events


def _status(result):
    return {t["id"]: (t["status"], t["intentos"]) for t in result["subtareas"]}


def test_budget(settings, tmp_path):
    result, events = _run("budget", settings, tmp_path)
    status = _status(result)
    assert status["s1"][0] == "approved" and status["s2"][0] == "approved"
    assert all(s == "skipped" for i, (s, _) in status.items() if i not in {"s1", "s2"})
    assert any(e["kind"] == "budget" and e["node"] == "queue" for e in events)
    # The writer still delivers and says what was not done.
    assert result["status"] == "parcial" and result["entregables"]
    report = open(result["entregables"][0], encoding="utf-8").read()
    assert "No se completó" in report and "token budget" in report


def test_attempts_and_repetition(settings, tmp_path):
    result, events = _run("attempts", settings, tmp_path)
    assert _status(result)["s1"] == ("failed", settings.max_code_attempts)
    assert _status(result)["s2"][0] == "approved"  # the queue went on
    runs = [e for e in events if e["kind"] == "execution" and e["subtask"] == "s1"]
    assert len(runs) == 1  # the identical copies were never executed
    critic = next(e for e in events if e.get("node") == "critic" and e["subtask"] == "s1")
    assert not critic["approved"] and critic["decided_by"] == "code checks"
    repeated = [e for e in events if e.get("node") == "guard" and e["subtask"] == "s1"
                and any("identical" in p for p in e["problems"])]
    assert len(repeated) == settings.max_code_attempts - 1


def test_timeout(settings, tmp_path):
    result, events = _run("timeout", settings, tmp_path)
    first = next(e for e in events if e["kind"] == "execution" and e["subtask"] == "s1")
    assert first["timed_out"] and first["returncode"] is None
    assert "killed after 3 s (process group" in first["stderr_tail"]
    assert _status(result)["s1"] == ("approved", 2)


def test_network_needs_a_human(settings, tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "n")
    result, events = _run("network", settings, tmp_path)
    assert len(asked) == 1 and "fetch_openml" in asked[0]
    request = next(e for e in events if e["kind"] == "network_request")
    assert request["asked"] and not request["approved"]
    runs = [e for e in events if e["kind"] == "execution" and e["subtask"] == "s1"]
    assert [r["attempt"] for r in runs] == [2]  # the download never ran
    assert _status(result)["s1"] == ("approved", 2)


def test_network_without_confirmation_is_refused_without_asking(settings, tmp_path,
                                                                 monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("must not ask"))
    from investigation_agent import brakes
    original = brakes.scenarios

    def no_confirmation():
        found = original()
        found["network"].overrides = {"network_confirmation": False}
        return found

    monkeypatch.setattr(brakes, "scenarios", no_confirmation)
    _, events = _run("network", settings, tmp_path)
    request = next(e for e in events if e["kind"] == "network_request")
    assert not request["asked"] and not request["approved"]
