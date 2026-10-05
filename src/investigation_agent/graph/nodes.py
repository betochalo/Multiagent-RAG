"""LangGraph nodes: one method per role, each receiving its services and clients.

Deterministic nodes (reader, executor, validators) never call the LLM. LLM nodes propose,
and code decides: the plan is validated by code, scripts pass a static guard before they
run, the critic runs code checks before asking the LLM, and every number of the
deliverable is checked for provenance before it is published.

Every LLM call goes through `_ask`, which enforces the token budget (with a reserve for
the writer), retries when the model spends `max_tokens` reasoning and returns an empty
`content`, and records the call in the trace.
"""

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import networkx as nx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from openai import LengthFinishReasonError
from pydantic import BaseModel, Field, ValidationError

from investigation_agent.config.settings import Settings
from investigation_agent.prompts import load_prompt
from investigation_agent.services.checks import (
    check_execution, check_format, deliverable_text, unsupported_numbers)
from investigation_agent.services.deliverable import (
    copy_figures, write_markdown, write_notebook, write_pdf)
from investigation_agent.services.knowledge_graph import KnowledgeGraphService
from investigation_agent.services.pdf_reader import ParsedDocument, PdfReaderService, Section
from investigation_agent.services.rag import RagService
from investigation_agent.services.sandbox import Sandbox
from investigation_agent.services.trace import Tracer

FINAL = {"approved", "done", "failed", "skipped"}


class BudgetExhaustedError(RuntimeError):
    """The token budget does not allow another call for this agent."""


class LLMError(RuntimeError):
    """The LLM call failed or returned nothing usable."""


# -------------------------------------------------------------------- plan schema
class Subtask(BaseModel):
    id: str
    type: Literal["code", "text"]
    section: str
    depends_on: list[str] = Field(default_factory=list)
    description: str
    criteria: list[str] = Field(default_factory=list)
    expects_figure: bool = False


class Deliverable(BaseModel):
    format: Literal["md", "pdf", "ipynb"]
    filename: str
    sections: list[str] = Field(default_factory=list)
    max_words: int | None = None
    max_pages: int | None = None


class Plan(BaseModel):
    deliverable: Deliverable
    subtasks: list[Subtask]


def validate_plan_structure(plan: dict, sections: list[dict]) -> list[str]:
    """Correction 1: the plan is a graph validated by code. Unique ids, dependencies that
    exist and form no cycle, sections that exist, and every work section covered."""
    errors = []
    subtasks = plan["subtasks"]
    ids = [s["id"] for s in subtasks]
    if not subtasks:
        errors.append("the plan has no subtasks")
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        errors.append(f"duplicated subtask ids: {duplicated}")
    section_ids = {s["id"] for s in sections}
    dag = nx.DiGraph()
    dag.add_nodes_from(ids)
    for s in subtasks:
        if s["section"] not in section_ids:
            errors.append(f"subtask {s['id']}: section {s['section']!r} does not exist; "
                          f"valid ids: {sorted(section_ids)}")
        for d in s["depends_on"]:
            if d not in ids:
                errors.append(f"subtask {s['id']} depends on {d!r}, which does not exist")
            elif d == s["id"]:
                errors.append(f"subtask {s['id']} depends on itself")
            else:
                dag.add_edge(d, s["id"])
    if not nx.is_directed_acyclic_graph(dag):
        errors.append(f"dependency cycle: {nx.find_cycle(dag)}")
    covered = {s["section"] for s in subtasks}
    missing = [s["id"] for s in sections if s["numbered"] and s["id"] not in covered]
    if missing:
        errors.append(f"work sections not covered by any subtask: {missing}")
    deliverable = plan["deliverable"]
    if not deliverable["filename"].endswith(f".{deliverable['format']}"):
        errors.append(f"deliverable filename {deliverable['filename']!r} does not match "
                      f"format {deliverable['format']!r}")
    return errors


# -------------------------------------------------------------------- state
def merge_tasks(old: dict | None, new: dict | None) -> dict:
    """Reducer for `tasks`: parallel subtasks (Parte 4) each return only their own entry, and
    the entries are merged by id. A sequential node returns the whole dict, merged the same."""
    return {**(old or {}), **(new or {})}


class SolverState(TypedDict, total=False):
    pdf_path: str
    output_dir: str
    source: str  # the statement's file name
    statement: str  # the statement's full text
    document: dict  # ParsedDocument as a dict
    sections: list[dict]  # [{id, title, numbered, text}] in document order
    plan: dict | None
    plan_errors: list[str]
    plan_attempts: int
    # subtask id → subtask + status, attempts, feedback, results...
    tasks: Annotated[dict[str, dict], merge_tasks]
    wave: list[str]  # Parte 4: the subtasks dispatched together
    order: list[str]  # subtask ids in plan order
    current: str | None
    context: str
    script: str
    guard: dict
    execution: dict
    review: dict
    rejected: list[str]  # hashes of rejected scripts (repetition detector)
    deliverable: str
    write_feedback: list[str]
    write_attempts: int
    deliverable_ok: bool
    budget_exhausted: bool
    status: str


# -------------------------------------------------------------------- nodes
class SolverNodes:
    def __init__(self, settings: Settings, llm: BaseChatModel, reader: PdfReaderService,
                 kg: KnowledgeGraphService, rag: RagService, sandbox: Sandbox, tracer: Tracer,
                 output_dir: str | Path, replicas: list[BaseChatModel] | None = None):
        self.settings = settings
        self.llm = llm
        # Parte 4: one chat model per H200 replica; a parallel subtask runs on one of them.
        self.replicas = replicas or [llm]
        self.replica = 0  # index of the replica this instance calls (recorded per call)
        self.reader = reader
        self.kg = kg
        self.rag = rag
        self.sandbox = sandbox
        self.tracer = tracer
        self.out = Path(output_dir)
        self.model_name = getattr(llm, "model_name", "") or ""
        self._graph: nx.DiGraph | None = None
        # Tokens spent building the course-notes cache: a one-time cost, not this task's.
        self._budget_offset = 0

    @property
    def max_code_attempts(self) -> int:
        return 1 if self.settings.ablation_no_critic else self.settings.max_code_attempts

    # ================================================================ reader (no LLM)
    def read(self, state: SolverState) -> dict:
        doc = self.reader.read(state["pdf_path"])
        self.tracer.event("node", node="reader", source=doc.source, pages=len(doc.pages),
                          sections=[s.title for s in doc.sections], tables=doc.tables)
        return {"source": doc.source, "statement": doc.text, "document": asdict(doc)}

    # ================================================================ indexer
    def index(self, state: SolverState) -> dict:
        d = state["document"]
        doc = ParsedDocument(source=d["source"], pages=d["pages"], tables=d["tables"],
                             sections=[Section(**s) for s in d["sections"]])
        graph = self.kg.build_skeleton(doc)

        self.rag.ingest(state["pdf_path"])
        notes = self._course_notes()
        for note in notes:
            if not self.rag.has_source(note.name):
                self.rag.ingest(note)

        if not self.settings.ablation_no_graph:
            try:
                self._index_statement_entities(graph, doc.source)
                self.kg.merge(graph, self._course_graph(notes))
                self._summarize_communities(graph)
            except BudgetExhaustedError as err:
                self.tracer.event("budget", node="indexer", detail=str(err))

        self._graph = graph
        self.kg.save(graph, self.out / "grafo.json")
        sections = [{"id": n, "title": graph.nodes[n]["title"],
                     "numbered": graph.nodes[n]["number"] is not None,
                     "text": graph.nodes[n]["text"]} for n in self.kg.sections(graph)]
        kinds = [d.get("kind") for _, d in graph.nodes(data=True)]
        self.tracer.event("node", node="indexer", sections=len(sections),
                          entities=kinds.count("entity"), communities=kinds.count("community"),
                          depends_on=[(u, v) for u, v, e in graph.edges(data=True)
                                      if e.get("kind") == "depende_de" and e.get("rule") == "reference"])
        return {"sections": sections}

    def _course_notes(self) -> list[Path]:
        return sorted(Path(self.settings.course_notes_dir).glob("*.md"))

    def _extract(self, agent: str, title: str, text: str, charge: bool = True) -> dict:
        data = self._ask_json(agent, load_prompt("indexer_entities"),
                              f"SECTION: {title}\n\n{text[:8000]}", charge=charge)
        return {"entities": data.get("entities") or [], "relations": data.get("relations") or []}

    def _index_statement_entities(self, graph: nx.DiGraph, source: str) -> None:
        sections = [n for n in self.kg.sections(graph) if graph.nodes[n]["text"].strip()]
        found = self._parallel(lambda n: self._extract(
            "indexer", graph.nodes[n]["title"] or n, graph.nodes[n]["text"]), sections)
        for n, result in zip(sections, found):
            if isinstance(result, dict):
                self.kg.add_entities(graph, n, source, result["entities"], result["relations"])

    def _course_graph(self, notes: list[Path]) -> nx.DiGraph:
        """Entities of the course notes. Each note section is extracted once and cached by
        its content and the prompt: a failed extraction is retried on the next run alone."""
        prompt = load_prompt("indexer_entities")
        cache_path = Path(self.settings.cache_dir) / "course_extractions.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

        graph = nx.DiGraph()
        jobs = []
        for note in notes:
            for i, (title, text) in enumerate(RagService.sections(note)):
                if len(text.strip()) < 80:
                    continue
                nid = f"nota:{note.name}#{i}"
                graph.add_node(nid, kind="note_section", title=title, source=note.name)
                key = hashlib.sha256(f"{prompt}\0{title}\0{text}".encode()).hexdigest()[:24]
                jobs.append((nid, note.name, title, text, key))

        todo = [j for j in jobs if j[4] not in cache]
        spent_before = self.tracer.tokens_total
        # Outside the task budget: the cache is built once and reused by every task.
        found = self._parallel(lambda j: self._extract("indexer_course", j[2], j[3],
                                                       charge=False), todo)
        self._budget_offset += self.tracer.tokens_total - spent_before
        for job, result in zip(todo, found):
            if isinstance(result, dict):
                cache[job[4]] = result
        if todo:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

        missing = 0
        for nid, source, _, _, key in jobs:
            if key in cache:
                self.kg.add_entities(graph, nid, source, cache[key]["entities"],
                                     cache[key]["relations"])
            else:
                missing += 1
        self.tracer.event("course_cache", sections=len(jobs), extracted_now=len(todo),
                          missing=missing)
        return graph

    def _summarize_communities(self, graph: nx.DiGraph) -> None:
        cache_path = Path(self.settings.cache_dir) / "communities.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        communities = self.kg.communities(graph, self.settings.community_min_size)
        keys = [hashlib.sha256("|".join(c).encode()).hexdigest()[:16] for c in communities]
        todo = [(k, c) for k, c in zip(keys, communities) if k not in cache]

        def describe(members: list[str]) -> str:
            lines = [f"- {graph.nodes[m]['name']} ({graph.nodes[m]['type']}; "
                     f"{', '.join(graph.nodes[m]['sources'])}): {graph.nodes[m]['description']}"
                     for m in members]
            rels = [f"- {graph.nodes[u]['name']} {d.get('label', '')} {graph.nodes[v]['name']}"
                    for u, v, d in graph.subgraph(members).edges(data=True)
                    if d.get("kind") == "relacion"]
            user = "ENTITIES:\n" + "\n".join(lines) + "\n\nRELATIONS:\n" + "\n".join(rels)
            return self._ask("indexer_community", load_prompt("indexer_community"), user)

        found = self._parallel(lambda kc: describe(kc[1]), todo)
        for (k, _), summary in zip(todo, found):
            if isinstance(summary, str) and summary.strip():
                cache[k] = summary.strip()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        for i, (k, members) in enumerate(zip(keys, communities)):
            if k in cache:
                self.kg.add_community(graph, i, members, cache[k])

    # ================================================================ planner
    def plan(self, state: SolverState) -> dict:
        attempts = state.get("plan_attempts", 0) + 1
        sections = state["sections"]
        user = "STATEMENT SECTIONS:\n\n" + "\n\n".join(
            f"### [{s['id']}] {s['title']}\n{s['text']}" for s in sections)
        user += "\n\nREQUIRED SECTIONS (each must be covered by a subtask): " + \
            ", ".join(s["id"] for s in sections if s["numbered"])
        if state.get("plan_errors"):
            user += "\n\nYOUR PREVIOUS PLAN WAS INVALID. Fix these problems:\n- " + \
                "\n- ".join(state["plan_errors"])
        try:
            plan = Plan.model_validate(
                self._ask_json("planner", load_prompt("planner"), user)).model_dump()
            errors = []
            (self.out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
        except BudgetExhaustedError as err:
            plan, errors, attempts = None, [str(err)], self.settings.max_plan_attempts
        except (LLMError, ValidationError) as err:
            plan, errors = None, [f"the plan is not valid JSON for the schema: {err}"[:1500]]
        self.tracer.event("node", node="planner", attempt=attempts, errors=errors)
        return {"plan": plan, "plan_errors": errors, "plan_attempts": attempts}

    def validate_plan(self, state: SolverState) -> dict:
        plan = state.get("plan")
        if plan is None:
            return {}
        errors = validate_plan_structure(plan, state["sections"])
        self.tracer.event("node", node="plan_validator", valid=not errors, errors=errors)
        if errors:
            return {"plan_errors": errors}
        tasks = {s["id"]: {**s, "status": "pending", "attempts": 0, "feedback": "",
                           "script": "", "workdir": None, "results": None, "stdout": "",
                           "context": ""} for s in plan["subtasks"]}
        return {"plan_errors": [], "tasks": tasks, "order": [s["id"] for s in plan["subtasks"]],
                "rejected": []}

    # ================================================================ queue (no LLM)
    def next_subtask(self, state: SolverState) -> dict:
        tasks = state["tasks"]
        pending = [i for i in state["order"] if tasks[i]["status"] == "pending"]
        if pending and self._remaining() <= self.settings.writer_reserve_tokens:
            for i in pending:
                tasks[i].update(status="skipped", feedback="token budget reached its writer reserve")
            self.tracer.event("budget", node="queue", skipped=pending,
                              remaining=self._remaining())
            return {"current": None, "tasks": tasks, "budget_exhausted": True}
        ready = [i for i in pending
                 if all(tasks[d]["status"] in FINAL for d in tasks[i]["depends_on"])]
        current = ready[0] if ready else None
        self.tracer.event("node", node="queue", current=current, pending=len(pending))
        return {"current": current, "tasks": tasks, "script": "", "context": ""}

    def dispatch(self, state: SolverState) -> dict:
        """Parte 4: the queue, for every ready subtask at once. Same budget rule as
        `next_subtask`; the orchestrator sends each subtask of the wave to its own branch."""
        tasks = state["tasks"]
        pending = [i for i in state["order"] if tasks[i]["status"] == "pending"]
        if pending and self._remaining() <= self.settings.writer_reserve_tokens:
            for i in pending:
                tasks[i].update(status="skipped", feedback="token budget reached its writer reserve")
            self.tracer.event("budget", node="dispatch", skipped=pending,
                              remaining=self._remaining())
            return {"wave": [], "tasks": tasks, "budget_exhausted": True}
        wave = [i for i in pending if all(tasks[d]["status"] in FINAL for d in tasks[i]["depends_on"])]
        self.tracer.event("node", node="dispatch", wave=wave, pending=len(pending),
                          replicas={i: n % len(self.replicas) for n, i in enumerate(wave)})
        return {"wave": wave}

    # ================================================================ researcher
    def research(self, state: SolverState) -> dict:
        tasks = state["tasks"]
        t = tasks[state["current"]]
        blocks, citations = [], []
        if not self.settings.ablation_no_graph:
            graph = self._load_graph()
            # The literal section and the closure of its dependencies always go first.
            for c in self.kg.section_context(graph, t["section"]):
                cite = f"{c['source']} · {c['title'] or c['id']} · p. {c['page']}"
                blocks.append(f"[{cite}] ({c['role']})\n{c['text']}")
                citations.append(cite)
            ents = self.kg.entity_context(graph, t["section"])
            if ents["entities"]:
                blocks.append("[knowledge graph · entities]\n" + "\n".join(
                    f"- {e['name']} ({e['type']}; {', '.join(e['sources'])}): {e['description']}"
                    for e in ents["entities"]))
            if ents["communities"]:
                blocks.append("[knowledge graph · community summaries]\n" +
                              "\n".join(f"- {c}" for c in ents["communities"]))
        sources = [state["source"]] + [n.name for n in self._course_notes()]
        for h in self.rag.retrieve(t["description"], where={"source": sources}):
            cite = f"{h['source']} · {h['section']}"
            blocks.append(f"[{cite}] (similar, score {h['score']:.3f})\n{h['text']}")
            citations.append(cite)
        for d in t["depends_on"]:
            if tasks[d]["results"]:
                blocks.append(f"[results of subtask {d}, for reference only]\n"
                              + json.dumps(tasks[d]["results"], ensure_ascii=False)[:3000])
        context = "\n\n".join(blocks)
        if t["type"] == "text":
            t.update(status="done", context=context)
        self.tracer.event("node", node="researcher", subtask=t["id"], citations=citations,
                          chars=len(context))
        return {"context": context, "tasks": tasks}

    # ================================================================ programmer
    def program(self, state: SolverState) -> dict:
        tasks = state["tasks"]
        t = tasks[state["current"]]
        t["attempts"] += 1
        user = (f"SUBTASK {t['id']} (section {t['section']})\n{t['description']}\n\n"
                "CRITERIA:\n" + "\n".join(f"- {c}" for c in t["criteria"]) +
                f"\n\nRESULTS FILE: {self.settings.results_file}"
                f"\nFIGURE REQUIRED: {'yes' if t['expects_figure'] else 'no'}"
                f"\n\nCONTEXT:\n{state.get('context', '')}")
        if t["feedback"]:
            user += (f"\n\nYOUR PREVIOUS SCRIPT WAS REJECTED. Correct it:\n{t['feedback']}"
                     f"\n\nPREVIOUS SCRIPT:\n```python\n{t['script']}\n```")
        try:
            code = _extract_code(self._ask("programmer", load_prompt("programmer"), user))
        except BudgetExhaustedError as err:
            t.update(status="failed", feedback=str(err))
            self.tracer.event("budget", node="programmer", subtask=t["id"], detail=str(err))
            return {"tasks": tasks, "script": "", "budget_exhausted": True}
        except LLMError as err:
            code = ""
            t["feedback"] = f"the previous call failed: {err}"
        self.tracer.event("node", node="programmer", subtask=t["id"], attempt=t["attempts"],
                          lines=code.count("\n") + 1 if code else 0)
        return {"tasks": tasks, "script": code}

    # ================================================================ guard (no LLM)
    def guard(self, state: SolverState) -> dict:
        tasks = state["tasks"]
        t = tasks[state["current"]]
        code = state.get("script", "")
        rejected = list(state.get("rejected", []))
        digest = _script_hash(code)
        allow_network, downloads, problems = False, [], []
        if not code.strip():
            problems = ["the programmer returned no code"]
        elif digest in rejected:
            problems = ["this script is identical to one already rejected: it is not run "
                        "again; change the approach"]
        else:
            verdict = self.sandbox.guard(code)
            problems, downloads = verdict.problems, verdict.downloads
            if verdict.needs_network and not problems:
                allow_network = self._confirm_network(t["id"], downloads)
                if not allow_network:
                    problems = [f"the script downloads data ({', '.join(downloads)}) and no "
                                "human approved network access; use data available locally "
                                "or written in the statement"]
        if problems:
            rejected.append(digest)
            t.update(feedback="GUARD:\n- " + "\n- ".join(problems), script=code)
            if t["attempts"] >= self.max_code_attempts:
                t["status"] = "failed"
        self.tracer.event("node", node="guard", subtask=t["id"], attempt=t["attempts"],
                          allowed=not problems, problems=problems, downloads=downloads,
                          allow_network=allow_network)
        return {"guard": {"allowed": not problems, "allow_network": allow_network,
                          "problems": problems}, "tasks": tasks, "rejected": rejected}

    def _confirm_network(self, subtask: str, downloads: list[str]) -> bool:
        approved = False
        if self.settings.network_confirmation:
            answer = input(f"\n[network] subtask {subtask} wants to download: "
                           f"{', '.join(downloads)}. Allow? [y/N] ")
            approved = answer.strip().lower() in {"y", "yes", "s", "si", "sí"}
        self.tracer.event("network_request", subtask=subtask, downloads=downloads,
                          asked=self.settings.network_confirmation, approved=approved)
        return approved

    # ================================================================ executor (no LLM)
    def execute(self, state: SolverState) -> dict:
        t = state["tasks"][state["current"]]
        workdir = self.out / "scripts" / t["id"] / f"attempt-{t['attempts']}"
        r = self.sandbox.run_script(state["script"], workdir,
                                    allow_network=state["guard"]["allow_network"])
        self.tracer.execution(t["id"], t["attempts"], r.returncode, r.duration_s, r.files,
                              timed_out=r.timed_out, stderr_tail=r.stderr[-500:])
        return {"execution": {"workdir": str(workdir), "returncode": r.returncode,
                              "stdout": r.stdout[-4000:], "stderr": r.stderr[-4000:],
                              "duration_s": r.duration_s, "timed_out": r.timed_out,
                              "files": r.files}}

    # ================================================================ critic
    def critic(self, state: SolverState) -> dict:
        """Correction 4: the results contract is checked by code first (Parte 0.c); only a
        script that passes is shown to the LLM critic."""
        tasks = state["tasks"]
        t = tasks[state["current"]]
        ex, code = state["execution"], state["script"]
        workdir = Path(ex["workdir"])
        rejected = list(state.get("rejected", []))
        if self.settings.ablation_no_critic:
            approved, problems, by = True, [], "ablation: no critic"
        else:
            report = check_execution(code, ex["returncode"], ex["stderr"], workdir,
                                     self.settings.results_file, t["expects_figure"],
                                     ex["timed_out"])
            approved, problems, by = report.passed, report.problems, "code checks"
            if approved:  # the LLM only sees what the code checks let through
                approved, problems, by = self._llm_review(t, code, ex, workdir)

        if approved:
            results_path = workdir / self.settings.results_file
            results = json.loads(results_path.read_text(encoding="utf-8")) \
                if results_path.exists() else {}
            t.update(status="approved", results=results, workdir=str(workdir), script=code,
                     stdout=ex["stdout"], feedback="")
        else:
            rejected.append(_script_hash(code))
            t.update(feedback=f"CRITIC ({by}):\n- " + "\n- ".join(problems), script=code)
            if t["attempts"] >= self.max_code_attempts:
                t["status"] = "failed"
        self.tracer.event("node", node="critic", subtask=t["id"], attempt=t["attempts"],
                          approved=approved, decided_by=by, problems=problems)
        return {"review": {"approved": approved, "decided_by": by, "problems": problems},
                "tasks": tasks, "rejected": rejected}

    def _llm_review(self, t: dict, code: str, ex: dict,
                    workdir: Path) -> tuple[bool, list[str], str]:
        results_path = workdir / self.settings.results_file
        user = (f"SUBTASK {t['id']}: {t['description']}\n\nCRITERIA:\n" +
                "\n".join(f"- {c}" for c in t["criteria"]) +
                f"\n\nSCRIPT:\n```python\n{code}\n```\n\nSTDOUT:\n{ex['stdout'][-3000:]}"
                f"\n\nRESULTS ({self.settings.results_file}):\n"
                f"{results_path.read_text(encoding='utf-8')[:4000]}")
        try:
            data = self._ask_json("critic", load_prompt("critic"), user)
        except (BudgetExhaustedError, LLMError) as err:
            # The code checks passed; without an LLM opinion they decide.
            return True, [f"LLM review unavailable ({err}); approved on code checks"], "code checks"
        approved = bool(data.get("approved"))
        feedback = str(data.get("feedback", "")).strip()
        return approved, [] if approved else [feedback or "rejected without feedback"], "LLM critic"

    # ================================================================ writer
    def write(self, state: SolverState) -> dict:
        attempts = state.get("write_attempts", 0) + 1
        spec = state["plan"]["deliverable"]
        tasks, order = state["tasks"], state["order"]
        path = self.out / spec["filename"]
        approved = [tasks[i] for i in order if tasks[i]["status"] == "approved"]
        figures = copy_figures([Path(t["workdir"]) for t in approved], self.out)
        incomplete = [f"{i} ({tasks[i]['section']}): {tasks[i]['status']} — "
                      f"{tasks[i]['feedback'][:300]}"
                      for i in order if tasks[i]["status"] in {"failed", "skipped", "pending"}]
        if spec["format"] == "ipynb":
            self._write_notebook(state, path, approved, incomplete)
        else:
            markdown = self._compose_report(state, approved, figures, incomplete)
            if spec["format"] == "pdf":
                pages = write_pdf(markdown, path, assets_dir=self.out)
                self.tracer.event("node", node="writer", attempt=attempts, file=path.name,
                                  pages=pages)
            else:
                write_markdown(markdown, path)
                self.tracer.event("node", node="writer", attempt=attempts, file=path.name,
                                  words=len(markdown.split()))
        return {"deliverable": str(path), "write_attempts": attempts}

    def _writer_material(self, state: SolverState, approved: list[dict],
                         incomplete: list[str]) -> str:
        tasks, order = state["tasks"], state["order"]
        spec = state["plan"]["deliverable"]
        parts = [f"STATEMENT:\n{state['statement']}",
                 f"DELIVERABLE: {spec['filename']} — required sections in order: "
                 f"{spec['sections']}; max words: {spec['max_words']}; max pages: {spec['max_pages']}"]
        for t in approved:
            parts.append(f"### APPROVED {t['id']} ({t['section']}): {t['description']}\n"
                         f"RESULTS:\n{json.dumps(t['results'], ensure_ascii=False, indent=1)[:6000]}"
                         f"\nSTDOUT:\n{t['stdout'][-2500:]}")
        for i in order:
            if tasks[i]["status"] == "done":
                parts.append(f"### CONCEPTUAL {i} ({tasks[i]['section']}): "
                             f"{tasks[i]['description']}\nCONTEXT:\n{tasks[i]['context'][:6000]}")
        if incomplete:
            parts.append("NOT COMPLETED:\n- " + "\n- ".join(incomplete))
        if state.get("write_feedback"):
            parts.append("YOUR PREVIOUS VERSION WAS REJECTED. Fix:\n- " +
                         "\n- ".join(state["write_feedback"]))
        return "\n\n".join(parts)

    def _compose_report(self, state: SolverState, approved: list[dict], figures: list[str],
                        incomplete: list[str]) -> str:
        user = self._writer_material(state, approved, incomplete)
        user += f"\n\nFIGURES AVAILABLE: {figures or 'none'}"
        try:
            text = self._ask("writer", load_prompt("writer_report"), user, writer=True)
            return re.sub(r"^```(?:markdown|md)?\s*\n|\n```\s*$", "", text.strip())
        except (BudgetExhaustedError, LLMError) as err:
            self.tracer.event("budget", node="writer", detail=f"fallback report: {err}")
            return _fallback_report(state["plan"]["deliverable"], approved, figures, incomplete,
                                    str(err))

    def _write_notebook(self, state: SolverState, path: Path, approved: list[dict],
                        incomplete: list[str]) -> None:
        by_id = {t["id"]: t for t in approved}
        user = self._writer_material(state, approved, incomplete)
        try:
            data = self._ask_json("writer", load_prompt("writer_notebook"), user, writer=True)
            cells = [c for c in data.get("cells", []) if isinstance(c, dict)]
        except (BudgetExhaustedError, LLMError) as err:
            # Without the writer, the notebook is the approved scripts with their headings.
            self.tracer.event("budget", node="writer", detail=f"fallback notebook: {err}")
            cells = []
        out, used = [], set()
        for c in cells:
            if c.get("type") == "code":
                sid = c.get("subtask")
                if sid in by_id and sid not in used:
                    out.append({"type": "code", "source": by_id[sid]["script"]})
                    used.add(sid)
            elif c.get("source"):
                out.append({"type": "markdown", "source": str(c["source"])})
        for t in approved:  # every approved script runs in the notebook, even if omitted
            if t["id"] not in used:
                out += [{"type": "markdown", "source": f"### {t['section']}: {t['description']}"},
                        {"type": "code", "source": t["script"]}]
        if incomplete:
            out.append({"type": "markdown",
                        "source": "## No completado\n\n" + "\n".join(f"- {i}" for i in incomplete)})
        r = write_notebook(out, path, self.sandbox)
        self.tracer.execution("notebook", state.get("write_attempts", 0) + 1, r.returncode,
                              r.duration_s, r.files, timed_out=r.timed_out,
                              stderr_tail=r.stderr[-800:])

    # ================================================================ publisher checks (no LLM)
    def check_deliverable(self, state: SolverState) -> dict:
        spec = state["plan"]["deliverable"]
        path = Path(state["deliverable"])
        report = check_format(path, spec["sections"], spec["max_words"], spec["max_pages"])
        problems = list(report.problems)
        if path.exists():
            # Correction 5: every number comes from an execution or from the statement.
            unsupported = unsupported_numbers(deliverable_text(path), state["statement"],
                                              self.out, exclude={path})
            if unsupported:
                problems.append("these numbers appear in no results file and not in the "
                                f"statement; copy them from the results or remove them: {unsupported}")
        self.tracer.event("node", node="publisher", attempt=state["write_attempts"],
                          ok=not problems, problems=problems)
        return {"write_feedback": problems, "deliverable_ok": not problems}

    def finish(self, state: SolverState) -> dict:
        tasks = state.get("tasks") or {}
        delivered = bool(state.get("deliverable")) and Path(state["deliverable"]).exists()
        if not delivered:
            status = "fallido"
        elif state.get("deliverable_ok") and all(t["status"] in {"approved", "done"}
                                                 for t in tasks.values()):
            status = "completado"
        else:
            status = "parcial"
        self.tracer.event("finish", status=status, plan_errors=state.get("plan_errors", []),
                          subtasks={i: t["status"] for i, t in tasks.items()},
                          usage_by_agent=dict(self.tracer.by_agent),
                          tokens_in=self.tracer.tokens_in, tokens_out=self.tracer.tokens_out)
        return {"status": status}

    # ================================================================ LLM plumbing
    def _remaining(self) -> int:
        return self.settings.token_budget - (self.tracer.tokens_total - self._budget_offset)

    def _ask(self, agent: str, system: str, user: str, json_mode: bool = False,
             writer: bool = False, charge: bool = True) -> str:
        """One LLM call with the budget check, the empty-content retry and the trace."""
        if charge:
            floor = 0 if writer else self.settings.writer_reserve_tokens
            if self._remaining() <= floor:
                raise BudgetExhaustedError(
                    f"token budget: {self._remaining()} tokens left, {floor} reserved for the writer")
        max_tokens = self.settings.h200_max_tokens
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        while True:
            kwargs = {"max_tokens": max_tokens}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            t0 = time.perf_counter()
            try:
                msg = self.llm.bind(**kwargs).invoke(messages)
            except LengthFinishReasonError as err:
                # In JSON mode the client raises instead of returning an empty content; the
                # tokens were spent all the same, and the same retry applies.
                usage = err.completion.usage
                self.tracer.llm_call(agent, self.model_name, usage.prompt_tokens if usage else 0,
                                     usage.completion_tokens if usage else 0,
                                     round(time.perf_counter() - t0, 2), finish_reason="length",
                                     max_tokens=max_tokens, replica=self.replica)
                if max_tokens < self.settings.h200_max_tokens_cap:
                    max_tokens = min(max_tokens * 2, self.settings.h200_max_tokens_cap)
                    continue
                raise LLMError(f"empty content (finish_reason=length, max_tokens={max_tokens})") \
                    from err
            except Exception as err:  # network, HTTP, timeout: recorded, then raised
                self.tracer.llm_call(agent, self.model_name, 0, 0,
                                     round(time.perf_counter() - t0, 2), error=repr(err)[:500],
                                     replica=self.replica)
                raise LLMError(f"{type(err).__name__}: {err}") from err
            usage = msg.usage_metadata or {}
            finish = msg.response_metadata.get("finish_reason")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            self.tracer.llm_call(agent, self.model_name, usage.get("input_tokens", 0),
                                 usage.get("output_tokens", 0), round(time.perf_counter() - t0, 2),
                                 finish_reason=finish, max_tokens=max_tokens, replica=self.replica)
            if content.strip():
                return content
            if finish == "length" and max_tokens < self.settings.h200_max_tokens_cap:
                # The model spent the budget reasoning: retry with more room, not "".
                max_tokens = min(max_tokens * 2, self.settings.h200_max_tokens_cap)
                continue
            raise LLMError(f"empty content (finish_reason={finish}, max_tokens={max_tokens})")

    def _ask_json(self, agent: str, system: str, user: str, writer: bool = False,
                  charge: bool = True) -> dict:
        text = self._ask(agent, system, user, json_mode=True, writer=writer, charge=charge)
        try:
            return _parse_json(text)
        except ValueError as err:
            raise LLMError(f"the reply is not a JSON object: {err}") from err

    def _parallel(self, fn, items: list) -> list:
        """Run `fn` over `items` concurrently. A failed item yields its exception; a budget
        error is re-raised once all calls finish."""
        if not items:
            return []

        def safe(item):
            try:
                return fn(item)
            except Exception as err:  # noqa: BLE001 — each failure is already in the trace
                return err

        with ThreadPoolExecutor(max_workers=self.settings.index_workers) as pool:
            results = list(pool.map(safe, items))
        budget = next((r for r in results if isinstance(r, BudgetExhaustedError)), None)
        if budget:
            raise budget
        return results

    def _load_graph(self) -> nx.DiGraph:
        if self._graph is None:
            self._graph = self.kg.load(self.out / "grafo.json")
        return self._graph


# -------------------------------------------------------------------- helpers
def _extract_code(reply: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, flags=re.S)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return reply.strip() + "\n" if "import " in reply else ""


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("the JSON is not an object")
    return data


def _script_hash(code: str) -> str:
    normalized = "\n".join(line.rstrip() for line in code.strip().splitlines() if line.strip())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _fallback_report(spec: dict, approved: list[dict], figures: list[str],
                     incomplete: list[str], reason: str) -> str:
    """A report built by code, without the LLM, when the writer cannot be called. It copies
    the numbers verbatim from the results, so it always passes the provenance check."""
    lines = ["# Reporte (generado sin redactor)", "",
             f"> El redactor no pudo ejecutarse ({reason}). Este documento reúne, sin "
             "interpretación, los resultados aprobados.", ""]
    sections = spec["sections"] or ["Resultados"]
    # The results go under the section that names them, or the last one.
    target = next((s for s in sections if "result" in s.lower()), sections[-1])
    for section in sections:
        lines += [f"## {section}", ""]
        if section == target:
            for t in approved:
                lines += [f"**{t['id']} — {t['description']}**", ""]
                lines += [f"- {k}: {v}" for k, v in _flat(t["results"])] + [""]
            lines += [f"![{f}]({f})" for f in figures] + [""]
            if incomplete:
                lines += ["No se completó:", ""] + [f"- {x}" for x in incomplete] + [""]
        else:
            lines += ["(sin redactar)", ""]
    return "\n".join(lines)


def _flat(obj, prefix: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flat(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list) and all(not isinstance(v, (dict, list)) for v in obj):
        yield prefix, obj
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flat(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj
