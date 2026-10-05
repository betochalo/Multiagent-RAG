"""Shared fixtures. No test calls the H200 or Qdrant: the LLM is scripted (see brakes.py)."""

from pathlib import Path

import pytest

from investigation_agent.config.settings import PROJECT_ROOT, Settings, get_settings

STATEMENTS = PROJECT_ROOT / "taller-03-v2-solver-multiagente" / "solver-v2" / "enunciados"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return get_settings().model_copy(update={"cache_dir": str(tmp_path / ".cache")})


@pytest.fixture
def task_a() -> Path:
    return STATEMENTS / "tarea-a-generativo-discriminativo.pdf"


@pytest.fixture
def task_c() -> Path:
    return STATEMENTS / "tarea-c-recuperacion-lexica-lsa.pdf"
