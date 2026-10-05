"""Parte 3: the four brakes, each forced by a real run of the solver with a scripted model.

    uv run python -m investigation_agent.brakes                  # all four → corridas/frenos/
    uv run python -m investigation_agent.brakes timeout network  # some of them

The brakes are code, so they are forced without spending tokens: the graph, the sandbox,
the guard, the critic's checks and the trace are the real ones; only the LLM is replaced
by `ScriptedChatModel`, which answers each agent from a script, and the retriever by
`NullRag`. Each scenario writes its own `corridas/frenos/<brake>/` with the trace.

- budget:   the run hits the writer's reserve; the pending subtasks are skipped and the
            writer still delivers, saying what was not done.
- attempts: the programmer returns the same leaky script; the critic rejects it, the
            repetition detector refuses to run it again, and at the cap the subtask fails
            while the queue moves on.
- timeout:  an infinite loop is killed with its process group; the next attempt passes.
- network:  a script that calls fetch_openml waits for a human; the human says no (read
            from stdin), the script never runs, and the next attempt uses local data.
"""

import json
import re
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from investigation_agent.config.settings import PROJECT_ROOT, Settings, get_settings
from investigation_agent.prompts import load_prompt

TASK_A = (PROJECT_ROOT / "taller-03-v2-solver-multiagente" / "solver-v2" / "enunciados"
          / "tarea-a-generativo-discriminativo.pdf")
AGENTS = ["indexer_entities", "indexer_community", "planner", "programmer", "critic",
          "writer_report", "writer_notebook"]


# -------------------------------------------------------------------- scripted model
@dataclass
class Reply:
    content: str
    tokens_in: int = 50
    tokens_out: int = 50
    finish_reason: str = "stop"


Script = Callable[[str, int], Reply]  # (user message, call number for this agent) → reply


class ScriptedChatModel(BaseChatModel):
    """A chat model that answers from a script, one per agent. The agent is recognized by
    its system prompt; each reply carries the token usage the script declares, so the
    budget brake sees real numbers."""

    scripts: dict[str, Script]
    model_name: str = "scripted"
    calls: dict[str, int] = {}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None,
                  **kwargs) -> ChatResult:
        system, user = str(messages[0].content), str(messages[-1].content)
        agent = next((a for a in AGENTS if load_prompt(a) == system), "unknown")
        n = self.calls.get(agent, 0) + 1
        self.calls[agent] = n
        reply = self.scripts[agent](user, n) if agent in self.scripts else Reply("{}")
        msg = AIMessage(content=reply.content,
                        usage_metadata={"input_tokens": reply.tokens_in,
                                        "output_tokens": reply.tokens_out,
                                        "total_tokens": reply.tokens_in + reply.tokens_out},
                        response_metadata={"finish_reason": reply.finish_reason})
        return ChatResult(generations=[ChatGeneration(message=msg)])


class NullRag:
    """A retriever with nothing indexed: the brakes need no Qdrant and no embeddings."""

    def ingest(self, path) -> int:
        return 0

    def has_source(self, source: str) -> bool:
        return True

    def retrieve(self, query: str, k=None, where=None) -> list[dict]:
        return []


# -------------------------------------------------------------------- scripts per agent
def _no_entities(user: str, n: int) -> Reply:
    return Reply(json.dumps({"entities": [], "relations": []}), 20, 20)


def _plan(user: str, n: int, code_subtasks: int = 2) -> Reply:
    """One code subtask per required section of the statement, `s1`, `s2`, ..."""
    required = re.search(r"REQUIRED SECTIONS[^:]*: (.+)", user).group(1).split(", ")
    subtasks = [{"id": f"s{i + 1}", "type": "code" if i < code_subtasks else "text",
                 "section": sid, "depends_on": [], "description": f"work for {sid}",
                 "criteria": ["writes resultados.json"], "expects_figure": False}
                for i, sid in enumerate(required)]
    plan = {"deliverable": {"format": "md", "filename": "reporte.md",
                            "sections": ["Introducción", "Resultados", "Conclusiones"],
                            "max_words": 1200, "max_pages": None},
            "subtasks": subtasks}
    return Reply(json.dumps(plan), 400, 300)


def _approve(user: str, n: int) -> Reply:
    return Reply(json.dumps({"approved": True, "feedback": ""}), 200, 50)


def _writer(user: str, n: int) -> Reply:
    """A report with the required sections and the list of what was not done."""
    not_done = re.search(r"NOT COMPLETED:\n(.+?)(?:\n\n|\Z)", user, flags=re.S)
    text = ("# Reporte\n\n## Introducción\n\nReporte del modelo de guion.\n\n"
            "## Resultados\n\nSe resumen los resultados de las subtareas aprobadas.\n\n")
    if not_done:
        text += "**No se completó:**\n\n" + not_done.group(1) + "\n\n"
    text += "## Conclusiones\n\nVer la traza.\n"
    return Reply(text, 300, 200)


def _code(body: str) -> str:
    return f"```python\n{body.strip()}\n```"


GOOD = _code("""
import json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
acc = GaussianNB().fit(Xtr, ytr).score(Xte, yte)
json.dump({"accuracy_nb": acc}, open("resultados.json", "w"), indent=2)
print(acc)
""")

# The Parte 0.c script: a tree evaluated on its training data.
LEAKY = _code("""
import json
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
X, y = load_breast_cancer(return_X_y=True)
model = DecisionTreeClassifier(random_state=0).fit(X, y)
acc = (model.predict(X) == y).mean()
json.dump({"accuracy": float(acc)}, open("resultados.json", "w"), indent=2)
print(acc)
""")

INFINITE = _code("""
import json
total = 0
while True:
    total += 1
""")

DOWNLOAD = _code("""
import json
from sklearn.datasets import fetch_openml
X, y = fetch_openml("breast-cancer", version=1, return_X_y=True, as_frame=False)
json.dump({"rows": len(X)}, open("resultados.json", "w"), indent=2)
""")


def _sequence(*replies: str, tokens: tuple[int, int] = (300, 400)) -> Script:
    """The n-th call gets the n-th reply; later calls repeat the last one."""
    return lambda user, n: Reply(replies[min(n, len(replies)) - 1], *tokens)


# -------------------------------------------------------------------- scenarios
@dataclass
class Scenario:
    name: str
    description: str
    scripts: dict[str, Script]
    overrides: dict


def scenarios() -> dict[str, Scenario]:
    base = {"indexer_entities": _no_entities, "planner": _plan, "critic": _approve,
            "writer_report": _writer}
    return {s.name: s for s in [
        Scenario(
            "budget",
            "Each programmer call costs 6 000 tokens against a budget of 20 000 with 8 000 "
            "reserved for the writer: after s2 the queue stops, the rest is skipped and the "
            "writer still delivers.",
            {**base, "programmer": _sequence(GOOD, tokens=(3000, 3000))},
            {"token_budget": 20_000, "writer_reserve_tokens": 8_000}),
        Scenario(
            "attempts",
            "The programmer returns the same leaky script for s1 every time: the critic "
            "rejects it once, the repetition detector refuses the identical copies, and at "
            "3 attempts s1 fails while s2 still runs.",
            {**base, "programmer": lambda user, n: Reply(
                LEAKY if "SUBTASK s1 " in user else GOOD, 300, 400)},
            {}),
        Scenario(
            "timeout",
            "The first script of s1 never ends: the sandbox kills its process group after "
            "3 s and the critic sends it back; the second attempt passes.",
            {**base, "programmer": _sequence(INFINITE, GOOD)},
            {"sandbox_timeout_s": 3}),
        Scenario(
            "network",
            "The first script of s1 calls fetch_openml: the guard flags the download and "
            "asks a human on stdin; the answer is no, the script never runs, and the second "
            "attempt uses the local dataset.",
            {**base, "programmer": _sequence(DOWNLOAD, GOOD)},
            {"network_confirmation": True}),
    ]}


def run_brake(name: str, output_dir: str | Path, settings: Settings | None = None) -> dict:
    """Run one scenario on the Tarea A statement and return the solver's contract."""
    from investigation_agent.graph.orchestrator import Solver

    scenario = scenarios()[name]
    out = Path(output_dir)
    # The scripted indexer extracts nothing: it must never touch the real course cache.
    cache = Path(tempfile.mkdtemp(prefix="brakes-cache-"))
    s = (settings or get_settings()).model_copy(
        update={"cache_dir": str(cache), **scenario.overrides})
    model = ScriptedChatModel(scripts=scenario.scripts, calls={})
    result = Solver(s, llm=model, rag=NullRag()).solve(str(TASK_A), str(out))
    (out / "escenario.json").write_text(json.dumps(
        {"brake": name, "description": scenario.description, "overrides": scenario.overrides,
         "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    names = sys.argv[1:] or list(scenarios())
    for name in names:
        result = run_brake(name, PROJECT_ROOT / "corridas" / "frenos" / name)
        subtasks = ", ".join(f"{t['id']}={t['status']}({t['intentos']})"
                             for t in result["subtareas"])
        print(f"{name:<9} status={result['status']:<10} {subtasks}  trace={result['trace']}")


if __name__ == "__main__":
    main()
