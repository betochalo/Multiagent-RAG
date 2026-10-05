"""LangGraph orchestrator: the state graph wiring the nodes with their conditional edges.

    read → index → plan ⇄ validate_plan → next_subtask ─┬─→ write ⇄ check_deliverable → finish
                                             ▲            └─→ research → program → guard → execute → critic
                                             └────────────────────────────────────────────────────────┘

Every branch is decided by a function of the state, never by the LLM:

- validate_plan → plan again with the list of problems, or finish when attempts run out;
- next_subtask → write when the queue is empty or the budget reached the writer's reserve;
- research → next_subtask for a conceptual subtask (the writer answers it);
- program → next_subtask when the budget ran out mid-subtask;
- guard → execute, back to program with the problems, or next_subtask at the attempt cap;
- critic → next_subtask on approval or at the attempt cap, else back to program;
- check_deliverable → write again with the unsupported numbers, or finish.

`Solver().solve(pdf, output)` is the contract `evaluar_solver.py` reads.
"""

import copy
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from investigation_agent.config.h200 import get_h200_client
from investigation_agent.config.llm import get_chat_model
from investigation_agent.config.qdrant import get_qdrant_client
from investigation_agent.config.settings import Settings, get_settings
from investigation_agent.graph.nodes import FINAL, SolverNodes, SolverState
from investigation_agent.services.knowledge_graph import KnowledgeGraphService
from investigation_agent.services.pdf_reader import PdfReaderService
from investigation_agent.services.rag import RagService
from investigation_agent.services.sandbox import Sandbox
from investigation_agent.services.trace import Tracer

# Each code attempt is ~5 steps; LangGraph's default limit (25) stops a real run.
RECURSION_LIMIT = 1000


# -------------------------------------------------------------------- routes
def route_after_validation(settings: Settings):
    def route(state: SolverState) -> str:
        if not state.get("plan_errors"):
            return "next_subtask"
        if state.get("plan_attempts", 0) < settings.max_plan_attempts:
            return "plan"
        return "finish"
    return route


def route_after_queue(state: SolverState) -> str:
    return "research" if state.get("current") else "write"


def route_after_research(state: SolverState) -> str:
    t = state["tasks"][state["current"]]
    return "next_subtask" if t["status"] in FINAL else "program"


def route_after_program(state: SolverState) -> str:
    t = state["tasks"][state["current"]]
    return "next_subtask" if t["status"] in FINAL else "guard"


def route_after_guard(state: SolverState) -> str:
    if state["guard"]["allowed"]:
        return "execute"
    t = state["tasks"][state["current"]]
    return "next_subtask" if t["status"] in FINAL else "program"


def route_after_critic(state: SolverState) -> str:
    t = state["tasks"][state["current"]]
    return "next_subtask" if t["status"] in FINAL else "program"


def route_after_publish_check(settings: Settings):
    def route(state: SolverState) -> str:
        if state.get("deliverable_ok") or state.get("write_attempts", 0) >= settings.max_write_attempts:
            return "finish"
        return "write"
    return route


# -------------------------------------------------------------------- Parte 4: parallel
def subtask_runner(nodes: SolverNodes):
    """One branch of a parallel wave: the whole loop of one subtask, research → program →
    guard → execute → critic with its retries, through the same node methods and the same
    route functions as the sequential graph. It runs on its own copy of the nodes bound to
    one H200 replica, and returns only its own entry of `tasks` (merged by `merge_tasks`)."""
    def run(payload: dict) -> dict:
        worker = copy.copy(nodes)
        worker.replica = payload["replica"]
        worker.llm = nodes.replicas[worker.replica]
        sid = payload["id"]
        st: dict = {"tasks": copy.deepcopy(payload["tasks"]), "current": sid,
                    "source": payload["source"], "statement": payload["statement"],
                    "rejected": []}
        worker.tracer.event("node", node="subtask_start", subtask=sid, replica=worker.replica)
        st.update(worker.research(st))
        step = route_after_research(st)
        while step != "next_subtask":
            if step == "program":
                st.update(worker.program(st))
                step = route_after_program(st)
            elif step == "guard":
                st.update(worker.guard(st))
                step = route_after_guard(st)
            elif step == "execute":
                st.update(worker.execute(st))
                step = "critic"
            else:  # critic
                st.update(worker.critic(st))
                step = route_after_critic(st)
        return {"tasks": {sid: st["tasks"][sid]}}
    return run


def route_dispatch(nodes: SolverNodes):
    """Every subtask of the wave goes to its own branch, round-robin over the replicas; an
    empty wave (queue done, or budget at the writer's reserve) goes to the writer."""
    def route(state: SolverState):
        wave = state.get("wave") or []
        if not wave:
            return "write"
        tasks = state["tasks"]
        return [Send("subtask", {
            "id": sid, "replica": n % len(nodes.replicas),
            "source": state["source"], "statement": state["statement"],
            # the subtask and the results of its dependencies, nothing else
            "tasks": {i: tasks[i] for i in [sid, *tasks[sid]["depends_on"]]},
        }) for n, sid in enumerate(wave)]
    return route


# -------------------------------------------------------------------- graph
def build_graph(nodes: SolverNodes, settings: Settings):
    if settings.parallel_subtasks:
        return build_parallel_graph(nodes, settings)
    g = StateGraph(SolverState)
    for name in ["read", "index", "plan", "validate_plan", "next_subtask", "research",
                 "program", "guard", "execute", "critic", "write", "check_deliverable",
                 "finish"]:
        g.add_node(name, getattr(nodes, name))

    g.add_edge(START, "read")
    g.add_edge("read", "index")
    g.add_edge("index", "plan")
    g.add_edge("plan", "validate_plan")
    g.add_conditional_edges("validate_plan", route_after_validation(settings),
                            ["plan", "next_subtask", "finish"])
    g.add_conditional_edges("next_subtask", route_after_queue, ["research", "write"])
    g.add_conditional_edges("research", route_after_research, ["program", "next_subtask"])
    g.add_conditional_edges("program", route_after_program, ["guard", "next_subtask"])
    g.add_conditional_edges("guard", route_after_guard, ["execute", "program", "next_subtask"])
    g.add_edge("execute", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["program", "next_subtask"])
    g.add_edge("write", "check_deliverable")
    g.add_conditional_edges("check_deliverable", route_after_publish_check(settings),
                            ["write", "finish"])
    g.add_edge("finish", END)
    return g.compile()


def build_parallel_graph(nodes: SolverNodes, settings: Settings):
    """Parte 4, option A. The queue becomes `dispatch`, which sends every ready subtask to a
    `subtask` branch at once; when the wave finishes, `dispatch` runs again for the
    subtasks that were waiting on it. Everything before and after is the baseline graph.

        … validate_plan → dispatch ─┬─ Send → subtask (×n, in parallel) ─→ dispatch
                                    └─→ write ⇄ check_deliverable → finish
    """
    g = StateGraph(SolverState)
    for name in ["read", "index", "plan", "validate_plan", "dispatch", "write",
                 "check_deliverable", "finish"]:
        g.add_node(name, getattr(nodes, name))
    g.add_node("subtask", subtask_runner(nodes))

    g.add_edge(START, "read")
    g.add_edge("read", "index")
    g.add_edge("index", "plan")
    g.add_edge("plan", "validate_plan")
    g.add_conditional_edges("validate_plan", route_after_validation(settings),
                            {"plan": "plan", "next_subtask": "dispatch", "finish": "finish"})
    g.add_conditional_edges("dispatch", route_dispatch(nodes), ["subtask", "write"])
    g.add_edge("subtask", "dispatch")
    g.add_edge("write", "check_deliverable")
    g.add_conditional_edges("check_deliverable", route_after_publish_check(settings),
                            ["write", "finish"])
    g.add_edge("finish", END)
    return g.compile()


# -------------------------------------------------------------------- solver
class Solver:
    """The Taller 03 v2 contract: `Solver().solve(ruta_pdf, carpeta_salida) -> dict`."""

    def __init__(self, settings: Settings | None = None, llm=None, rag=None):
        """`llm` and `rag` default to the H200 and Qdrant; the brakes (Parte 3) and the
        tests inject a scripted model and an empty retriever instead."""
        self.settings = settings or get_settings()
        self.llm = llm
        self.rag = rag

    def build(self, output_dir: Path, tracer: Tracer):
        s = self.settings
        llm = self.llm or get_chat_model(s)
        replicas = [llm]
        if s.parallel_subtasks:  # Parte 4: the second replica takes every other branch
            replicas.append(self.llm or get_chat_model(s, port=s.h200_replica_port))
        nodes = SolverNodes(
            settings=s,
            llm=llm,
            replicas=replicas,
            reader=PdfReaderService(s),
            kg=KnowledgeGraphService(),
            rag=self.rag or RagService(get_h200_client(s), get_qdrant_client(s), s),
            sandbox=Sandbox(s),
            tracer=tracer,
            output_dir=output_dir,
        )
        return nodes, build_graph(nodes, s)

    def solve(self, ruta_pdf: str, carpeta_salida: str) -> dict:
        out = Path(carpeta_salida).resolve()
        out.mkdir(parents=True, exist_ok=True)
        tracer = Tracer(out / "traza.jsonl")
        tracer.event("start", pdf=str(ruta_pdf), output=str(out),
                     ablations={"no_graph": self.settings.ablation_no_graph,
                                "no_critic": self.settings.ablation_no_critic},
                     parallel_subtasks=self.settings.parallel_subtasks)
        state: dict = {}
        model = ""
        try:
            nodes, graph = self.build(out, tracer)
            model = nodes.model_name
            state = graph.invoke({"pdf_path": str(Path(ruta_pdf).resolve()), "output_dir": str(out)},
                                 {"recursion_limit": RECURSION_LIMIT})
        except Exception as err:  # correction 6: the trace says why; the contract still holds
            tracer.event("error", error=f"{type(err).__name__}: {err}")
            state = {**state, "status": "fallido", "error": f"{type(err).__name__}: {err}"}
        return self._contract(state, tracer, model)

    def _contract(self, state: dict, tracer: Tracer, model: str) -> dict:
        deliverable = state.get("deliverable")
        tasks = state.get("tasks") or {}
        result = {
            "status": state.get("status", "fallido"),
            "entregables": [deliverable] if deliverable and Path(deliverable).exists() else [],
            "subtareas": [{"id": t["id"], "tipo": t["type"], "status": t["status"],
                           "intentos": t["attempts"]} for t in tasks.values()],
            "usage": {"tokens_entrada": tracer.tokens_in, "tokens_salida": tracer.tokens_out},
            "model": model,
            "trace": str(tracer.path),
        }
        if state.get("error"):
            result["error"] = state["error"]
        return result

    def run(self, pregunta: str) -> dict:
        """The Taller 4 contract: the question is the path to a task statement."""
        pdf = Path(pregunta)
        result = self.solve(str(pdf), str(Path("corridas") / pdf.stem))
        answer = ""
        if result["entregables"] and Path(result["entregables"][0]).suffix in {".md", ".txt"}:
            answer = Path(result["entregables"][0]).read_text(encoding="utf-8")
        elif result["entregables"]:
            answer = result["entregables"][0]
        return {"answer": answer, "trace": result["trace"], "status": result["status"],
                "model": result["model"], "usage": result["usage"]}


def diagram(parallel: bool = False) -> str:
    """Mermaid diagram of the orchestrator, built from the compiled graph itself."""
    from unittest.mock import MagicMock

    nodes = MagicMock(spec=SolverNodes)
    nodes.replicas = [None, None]
    settings = get_settings().model_copy(update={"parallel_subtasks": parallel})
    return build_graph(nodes, settings).get_graph().draw_mermaid()
