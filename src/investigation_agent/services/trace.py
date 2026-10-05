"""Correction 6 — run trace: one JSON line per event, appended as it happens.

Written on every path: each event is flushed to `traza.jsonl` the moment it occurs, so a
run that crashes or exhausts its budget still leaves the trace of what it did. Per LLM
call it records the agent, model, input and output tokens, latency and error; per
execution, the exit code, duration and the files it created.
"""

import json
import threading
import time
from collections import defaultdict
from pathlib import Path


class Tracer:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.tokens_in = 0
        self.tokens_out = 0
        self.by_agent: dict[str, dict[str, int]] = defaultdict(
            lambda: {"calls": 0, "tokens_in": 0, "tokens_out": 0})
        # The indexer calls the LLM from several threads.
        self._lock = threading.Lock()

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out

    def event(self, kind: str, **data) -> None:
        record = {"ts": round(time.time(), 3), "kind": kind, **data}
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)

    def llm_call(self, agent: str, model: str, tokens_in: int, tokens_out: int,
                 latency_s: float, error: str | None = None, **extra) -> None:
        with self._lock:
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out
            stats = self.by_agent[agent]
            stats["calls"] += 1
            stats["tokens_in"] += tokens_in
            stats["tokens_out"] += tokens_out
        self.event("llm_call", agent=agent, model=model, tokens_in=tokens_in,
                   tokens_out=tokens_out, latency_s=latency_s, error=error, **extra)

    def execution(self, subtask: str, attempt: int, returncode: int | None, duration_s: float,
                  files: list[str], timed_out: bool = False, **extra) -> None:
        self.event("execution", subtask=subtask, attempt=attempt, returncode=returncode,
                   duration_s=duration_s, files=files, timed_out=timed_out, **extra)
