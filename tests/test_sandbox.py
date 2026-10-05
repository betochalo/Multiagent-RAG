"""Correction 3: the static guard before any process exists, and the isolated process:
empty environment, no network, own folder, a timeout that kills the process group."""

import os
import shutil
import sys
import time

import pytest

from investigation_agent.services.sandbox import Sandbox

needs_unshare = pytest.mark.skipif(shutil.which("unshare") is None, reason="needs unshare")


@pytest.mark.parametrize("code, problem", [
    ("import socket", "import of 'socket'"),
    ("import subprocess", "import of 'subprocess'"),
    ("from urllib.request import urlopen", "import of 'urllib.request'"),
    ("eval('1+1')", "call to eval()"),
    ("import os\nos.remove('x')", ".remove is not allowed"),
    ("import os\nprint(os.environ)", ".environ is not allowed"),
    ("open('/etc/passwd').read()", "leaves the working folder"),
    ("open('../secret.txt')", "leaves the working folder"),
    ("x = input()", "call to input()"),
    ("def f(:\n  pass", "SyntaxError"),
])
def test_guard_rejects(settings, code, problem):
    verdict = Sandbox(settings).guard(code)
    assert not verdict.allowed
    assert any(problem in p for p in verdict.problems), verdict.problems


@pytest.mark.parametrize("code", [
    "from sklearn.datasets import fetch_openml\nfetch_openml('iris')",
    "from datasets import load_dataset\nload_dataset('imdb')",
    "import torchvision\ntorchvision.datasets.MNIST('.', download=True)",
])
def test_guard_flags_downloads_for_a_human(settings, code):
    verdict = Sandbox(settings).guard(code)
    assert verdict.allowed and verdict.needs_network and verdict.downloads


def test_guard_allows_slash_as_a_separator(settings):
    """Found in the Tarea C run: f"{hits}/{n}" has a bare "/" constant, which is no path."""
    verdict = Sandbox(settings).guard('hits, n = 5, 6\nprint(f"{hits}/{n}", " / ".join("ab"))')
    assert verdict.allowed, verdict.problems
    assert not Sandbox(settings).guard('open(f"/home/{user}/x")').allowed


def test_guard_allows_a_normal_script(settings):
    code = ("import json\nfrom sklearn.datasets import load_iris\n"
            "X, y = load_iris(return_X_y=True)\njson.dump({'n': len(X)}, open('r.json', 'w'))")
    verdict = Sandbox(settings).guard(code)
    assert verdict.allowed and not verdict.needs_network


def test_environment_is_empty(settings, tmp_path, monkeypatch):
    """No variable from .env reaches a script, so it cannot print an API key into a log.
    `run` is called directly: the guard would already reject `os.environ`."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-leak")
    r = Sandbox(settings).run_script(
        "import os\nprint(os.environ.get('OPENAI_API_KEY'), sorted(os.environ))", tmp_path)
    assert r.returncode == 0
    assert "sk-should-never-leak" not in r.stdout
    assert r.stdout.startswith("None")


@needs_unshare
def test_no_network(settings, tmp_path):
    r = Sandbox(settings).run_script(
        "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=3)", tmp_path)
    assert r.returncode != 0
    assert "unreachable" in r.stderr.lower()


def test_timeout_kills_the_process_group(settings, tmp_path):
    """The child the script starts in the background dies with it."""
    r = Sandbox(settings).run(
        ["sh", "-c", "sleep 60 & echo $! > child.pid; wait"], tmp_path, timeout_s=1)
    assert r.timed_out and r.returncode is None
    assert r.duration_s < 10
    child = int((tmp_path / "child.pid").read_text())
    time.sleep(0.3)
    assert not os.path.exists(f"/proc/{child}") or \
        open(f"/proc/{child}/stat").read().split()[2] == "Z"


def test_reports_created_files(settings, tmp_path):
    r = Sandbox(settings).run_script(
        "import json\njson.dump({'a': 1}, open('resultados.json', 'w'))", tmp_path)
    assert r.returncode == 0
    # script.py existed before the run: only what the execution wrote counts.
    assert r.files == ["resultados.json"]


def test_stdin_is_closed(settings, tmp_path):
    """A script that reads stdin gets EOF at once instead of blocking on the terminal."""
    r = Sandbox(settings).run([sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"],
                              tmp_path, timeout_s=5)
    assert not r.timed_out and r.stdout.strip() == "''"
