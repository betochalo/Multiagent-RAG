"""Correction 1: the plan is a graph validated by code."""

import copy

import pytest

from investigation_agent.graph.nodes import validate_plan_structure

SECTIONS = [{"id": "preambulo", "numbered": False}, {"id": "parte-1", "numbered": True},
            {"id": "parte-2", "numbered": True}]
VALID = {
    "deliverable": {"format": "md", "filename": "reporte.md", "sections": [],
                    "max_words": None, "max_pages": None},
    "subtasks": [
        {"id": "s1", "type": "code", "section": "parte-1", "depends_on": []},
        {"id": "s2", "type": "text", "section": "parte-2", "depends_on": ["s1"]},
    ],
}


def _plan(**changes):
    plan = copy.deepcopy(VALID)
    for path, value in changes.items():
        if path == "filename":
            plan["deliverable"]["filename"] = value
        else:
            i, key = path.split("__")
            plan["subtasks"][int(i)][key] = value
    return plan


def test_valid_plan():
    assert validate_plan_structure(VALID, SECTIONS) == []


@pytest.mark.parametrize("changes, message", [
    ({"1__id": "s1"}, "duplicated subtask ids"),
    ({"1__depends_on": ["s9"]}, "does not exist"),
    ({"1__depends_on": ["s2"]}, "depends on itself"),
    ({"0__depends_on": ["s2"]}, "dependency cycle"),
    ({"1__section": "parte-7"}, "section 'parte-7' does not exist"),
    ({"1__section": "parte-1"}, "not covered by any subtask: ['parte-2']"),
    ({"filename": "reporte.pdf"}, "does not match format"),
])
def test_invalid_plans(changes, message):
    errors = validate_plan_structure(_plan(**changes), SECTIONS)
    assert any(message in e for e in errors), errors


def test_empty_plan():
    plan = copy.deepcopy(VALID)
    plan["subtasks"] = []
    errors = validate_plan_structure(plan, SECTIONS)
    assert "the plan has no subtasks" in errors
