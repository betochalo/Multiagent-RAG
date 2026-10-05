"""Sandbox for generated code: a static guard on the syntax tree, then an isolated process.

The guard runs before any process exists. It rejects network and process modules, file
deletion, `eval`/`exec`, and literal paths that leave the working folder, and it flags
dataset downloads (`fetch_openml`, `load_dataset`, `download=True`) as needing a human.

The process gets an empty environment (no variable from `.env` reaches the script, so a
generated script cannot print an API key into a log), its own working folder, a timeout
that kills the whole process group, and no network: it runs inside `unshare -rn` (a new
network namespace with no interfaces) unless a human approved the download.
"""

import ast
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from investigation_agent.config.settings import Settings

_FORBIDDEN_MODULES = {
    "socket", "requests", "urllib", "urllib3", "http", "httpx", "aiohttp", "ftplib",
    "smtplib", "telnetlib", "subprocess", "multiprocessing", "ctypes", "importlib", "pty",
}
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "breakpoint", "input"}
_FORBIDDEN_ATTRS = {
    "system", "popen", "remove", "unlink", "rmdir", "removedirs", "rmtree", "kill",
    "killpg", "fork", "execv", "execve", "execvp", "spawnl", "spawnv", "chmod", "chown",
    "putenv", "environ",
}
_DOWNLOAD_CALLS = {"fetch_openml", "load_dataset", "urlretrieve", "hf_hub_download",
                   "snapshot_download", "fetch_20newsgroups", "fetch_california_housing",
                   "fetch_covtype", "fetch_kddcup99", "fetch_lfw_people", "fetch_olivetti_faces",
                   "fetch_rcv1", "fetch_species_distributions"}


@dataclass
class GuardVerdict:
    allowed: bool
    needs_network: bool = False
    problems: list[str] = field(default_factory=list)
    downloads: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    returncode: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool
    files: list[str]  # created or modified, relative to the working folder


class Sandbox:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.unshare = shutil.which("unshare")

    # ---------------------------------------------------------------- static guard
    def guard(self, code: str) -> GuardVerdict:
        """Correction 3, before any process exists: a static guard on the syntax tree."""
        try:
            tree = ast.parse(code)
        except SyntaxError as err:
            return GuardVerdict(False, problems=[f"SyntaxError line {err.lineno}: {err.msg}"])
        problems, downloads = [], []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) \
                    else [node.module or ""]
                for name in names:
                    if name.split(".")[0] in _FORBIDDEN_MODULES:
                        problems.append(f"line {node.lineno}: import of {name!r} is not allowed")
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name in _DOWNLOAD_CALLS:
                            downloads.append(f"line {node.lineno}: {alias.name}")
            elif isinstance(node, ast.Call):
                name = _call_name(node)
                if name in _FORBIDDEN_CALLS:
                    problems.append(f"line {node.lineno}: call to {name}() is not allowed")
                if name in _DOWNLOAD_CALLS:
                    downloads.append(f"line {node.lineno}: {name}()")
                for kw in node.keywords:
                    if kw.arg == "download" and isinstance(kw.value, ast.Constant) \
                            and kw.value.value is True:
                        downloads.append(f"line {node.lineno}: download=True")
            elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
                problems.append(f"line {node.lineno}: .{node.attr} is not allowed")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if _leaves_folder(node.value):
                    problems.append(f"line {node.lineno}: path {node.value!r} leaves the "
                                    "working folder")
        downloads = sorted(set(downloads))
        return GuardVerdict(allowed=not problems, needs_network=bool(downloads),
                            problems=sorted(set(problems)), downloads=downloads)

    # ---------------------------------------------------------------- process
    def run(self, args: list[str], workdir: str | Path, allow_network: bool = False,
            timeout_s: int | None = None) -> RunResult:
        """Correction 3, the process: run `args` in `workdir` with an empty environment, no
        network, no terminal on stdin and a timeout that kills the process group."""
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        timeout_s = timeout_s or self.settings.sandbox_timeout_s
        before = _snapshot(workdir)
        env = {
            "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
            "HOME": str(workdir),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(workdir / ".mpl"),
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            # A crash by signal (SIGSEGV, SIGILL) dumps the Python traceback to stderr.
            "PYTHONFAULTHANDLER": "1",
        }
        cmd = list(args)
        if not allow_network and self.unshare:
            # A new network namespace has no interfaces but a down loopback; bringing `lo`
            # up lets a notebook kernel talk to nbconvert over localhost, and nothing else.
            cmd = [self.unshare, "-rn", "sh", "-c", 'ip link set lo up 2>/dev/null; exec "$@"',
                   "sandbox", *cmd]
        t0 = time.perf_counter()
        # stdin is closed: a script must never read from (or block on) the user's terminal.
        proc = subprocess.Popen(cmd, cwd=workdir, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)  # the whole group, not just the parent
            stdout, stderr = proc.communicate()
            stderr += f"\n[sandbox] killed after {timeout_s} s (process group {proc.pid})"
        duration = round(time.perf_counter() - t0, 2)
        after = _snapshot(workdir)
        files = sorted(p for p, mtime in after.items() if before.get(p) != mtime)
        return RunResult(None if timed_out else proc.returncode, stdout, stderr, duration,
                         timed_out, files)

    def run_script(self, code: str, workdir: str | Path, allow_network: bool = False) -> RunResult:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "script.py").write_text(code, encoding="utf-8")
        return self.run([sys.executable, "script.py"], workdir, allow_network)


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def _leaves_folder(value: str) -> bool:
    # A bare "/" is a separator (f"{hits}/{n}", " / ".join), not a path.
    if "\n" in value or len(value) > 300 or value.strip() in {"/", "//"}:
        return False
    return value.startswith(("/", "~")) or "../" in value or value == ".."


def _snapshot(folder: Path) -> dict[str, float]:
    return {str(p.relative_to(folder)): p.stat().st_mtime for p in folder.rglob("*")
            if p.is_file() and ".mpl" not in p.parts}
