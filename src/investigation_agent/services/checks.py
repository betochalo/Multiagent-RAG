"""Code checks that decide before any LLM is asked: the critic's checks on an execution,
the provenance of every number in the deliverable, and the deliverable's format.

They are code, so a well-written explanation cannot talk them out of a verdict (Parte 0.c).
"""

import ast
import json
import math
import re
import signal
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# A number, with decimal point or comma, optionally in scientific notation (1.0003e-08) or a
# percentage. Without the exponent, "1.0003e-08" would be read as "1" and escape provenance.
_NUMBER = re.compile(r"(?<![\w.,])(-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)(\s?%)?(?![\w])")
_METRIC_KEYS = ("acc", "f1", "exact", "precision", "recall", "auc")
# Files that hold numbers no execution produced: the trace, the graph (its descriptions copy
# numbers from the statement and the notes), the plan, caches.
_NOT_ARTIFACTS = re.compile(r"traza|grafo|graph|plan|resumen|cache|enunciado|\.mpl")


@dataclass
class CheckReport:
    passed: bool
    problems: list[str] = field(default_factory=list)


# -------------------------------------------------------------------- critic checks
def check_execution(code: str, returncode: int | None, stderr: str, workdir: Path,
                    results_file: str, expects_figure: bool, timed_out: bool) -> CheckReport:
    """Exit code, results contract, NaN, implausible metrics, leakage, figures."""
    problems: list[str] = []
    if timed_out:
        problems.append("the script exceeded the sandbox timeout and was killed")
    elif returncode != 0:
        tail = "\n".join(stderr.strip().splitlines()[-12:])
        killed = f" (killed by {signal.Signals(-returncode).name})" \
            if returncode is not None and returncode < 0 else ""
        problems.append(f"exit code {returncode}{killed}; stderr:\n{tail}")

    results_path = workdir / results_file
    results = None
    if not results_path.exists():
        problems.append(f"no {results_file}: every script must write its numbers there")
    else:
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            problems.append(f"{results_file} is not valid JSON: {err}")
    if results is not None:
        if not results:
            problems.append(f"{results_file} is empty")
        bad = [k for k, v in _flatten(results) if isinstance(v, float) and not math.isfinite(v)]
        if bad:
            problems.append(f"NaN or infinite values in {results_file}: {bad[:5]}")
        implausible = [f"{k}={v}" for k, v in _flatten(results)
                       if isinstance(v, (int, float)) and not isinstance(v, bool)
                       and any(m in k.lower().split(".")[-1] for m in _METRIC_KEYS)
                       and 0.999 <= v <= 1.0]
        if implausible:
            problems.append(f"implausible metrics (>= 0.999 on a noisy problem): {implausible[:5]}"
                            " — is the model evaluated on its training data?")

    problems += [f"leakage: {p}" for p in leakage(code)]

    if expects_figure and not any(workdir.rglob("*.png")):
        problems.append("the subtask asks for a figure and the script saved no .png")
    return CheckReport(not problems, problems)


def leakage(code: str) -> list[str]:
    """Two static leakage patterns:
    - predicting on the same variable a model was fitted on (Parte 0.c);
    - fitting a transformer on a variable that is split into train/test afterwards (a
      scaler fitted on all the data leaks the test statistics into training)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    fitted: dict[str, int] = {}  # variable → first line it was fitted on
    evaluated: list[tuple[str, int]] = []
    derived: dict[str, str] = {}  # X_scaled = scaler.fit_transform(X) → X_scaled: X
    splits: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "fit_transform" \
                    and call.args and isinstance(call.args[0], ast.Name):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        derived[target.id] = call.args[0].id
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else \
            node.func.id if isinstance(node.func, ast.Name) else ""
        first = node.args[0].id if node.args and isinstance(node.args[0], ast.Name) else None
        if first is None:
            continue
        if name in {"fit", "fit_transform"}:
            fitted.setdefault(first, node.lineno)
        elif name in {"predict", "predict_proba", "score", "decision_function"}:
            evaluated.append((first, node.lineno))
        elif name == "train_test_split":
            splits.append((first, node.lineno))
    out = [f"line {line}: predicts on {var!r}, the same data the model was fitted on"
           for var, line in evaluated if var in fitted]
    for var, line in splits:
        origin = derived.get(var, var)
        fit_line = fitted.get(var) or fitted.get(origin)
        if fit_line is not None and fit_line < line:
            out.append(f"line {fit_line}: a transformer is fitted on {origin!r} before it is "
                       f"split at line {line}: test statistics leak into training")
    return sorted(set(out))


def _flatten(obj, prefix: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


# -------------------------------------------------------------------- provenance
def numbers(text: str) -> list[tuple[float, int, str]]:
    """(value, decimals, raw) of every number; a percentage also yields its fraction."""
    out = []
    for m in _NUMBER.finditer(text):
        raw = m.group(1).replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            continue
        decimals = _decimals(raw)
        out.append((value, decimals, m.group(0).strip()))
        if m.group(2):
            out.append((value / 100, decimals + 2, m.group(0).strip()))
    return out


def _decimals(raw: str) -> int:
    """Decimal places the number is written to: 0.9883 → 4, 1.0003e-08 → 12, 1.5e+03 → -2.
    The provenance tolerance is half a unit in that place."""
    mantissa, _, exponent = raw.lower().partition("e")
    places = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return places - int(exponent or 0)


def artifact_numbers(output_dir: Path, exclude: set[Path]) -> list[float]:
    """Every number written by an execution: JSON, CSV, text and logs, notebook outputs."""
    values: list[float] = []
    for ext in ("json", "csv", "txt", "log", "out"):
        for f in output_dir.rglob(f"*.{ext}"):
            rel = str(f.relative_to(output_dir))
            if f in exclude or f.stat().st_size > 5_000_000 or _NOT_ARTIFACTS.search(rel):
                continue
            values += [v for v, _, _ in numbers(f.read_text(encoding="utf-8", errors="replace"))]
    for nb in output_dir.rglob("*.ipynb"):
        for cell in json.loads(nb.read_text(encoding="utf-8")).get("cells", []):
            for o in cell.get("outputs", []):
                text = "".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", ""))
                values += [v for v, _, _ in numbers(text)]
    return values


def unsupported_numbers(deliverable_text: str, statement_text: str, output_dir: Path,
                        exclude: set[Path]) -> list[str]:
    """Numbers with two or more decimals in the deliverable that appear neither in anything
    an execution wrote nor in the statement. They go back to the writer by name."""
    given = {v for v, _, _ in numbers(statement_text)}
    measured = artifact_numbers(output_dir, exclude)

    def supported(value: float, decimals: int) -> bool:
        # Half a unit in the last written place; the float slack is relative, or an absolute
        # 1e-9 would swallow every value around 1e-08.
        tolerance = 0.5 * 10 ** -decimals + 1e-12 * abs(value)
        return value in given or any(abs(a - value) <= tolerance for a in measured)

    out = []
    for m in _NUMBER.finditer(strip_code(deliverable_text)):
        # "98.83 %" is backed by 98.83 or by 0.9883: each form of the number is checked.
        forms = numbers(m.group(0))
        if all(d < 2 for _, d, _ in forms):
            continue
        if not any(supported(v, d) for v, d, _ in forms):
            out.append(m.group(0).strip())
    return sorted(set(out), key=out.index)


def strip_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


# -------------------------------------------------------------------- format
def deliverable_text(path: Path, with_outputs: bool = False) -> str:
    """The prose of a deliverable: Markdown as is, a PDF's text, a notebook's Markdown cells
    (plus outputs on request)."""
    if path.suffix == ".pdf":
        with pymupdf.open(path) as pdf:
            return unicodedata.normalize("NFKC", "\n".join(p.get_text() for p in pdf))
    if path.suffix == ".ipynb":
        parts = []
        for cell in json.loads(path.read_text(encoding="utf-8")).get("cells", []):
            if cell["cell_type"] == "markdown":
                parts.append("".join(cell.get("source", "")))
            elif with_outputs:
                for o in cell.get("outputs", []):
                    parts.append("".join(o.get("text", "")))
                    parts.append("".join(o.get("data", {}).get("text/plain", "")))
        return "\n".join(parts)
    return path.read_text(encoding="utf-8", errors="replace")


def check_format(path: Path, sections: list[str], max_words: int | None,
                 max_pages: int | None) -> CheckReport:
    if not path.exists():
        return CheckReport(False, [f"the deliverable {path.name} was not written"])
    text = deliverable_text(path)
    problems = []
    positions = [_heading_position(text, s) for s in sections]
    missing = [s for s, p in zip(sections, positions) if p < 0]
    if missing:
        problems.append(f"missing sections: {missing}")
    elif positions != sorted(positions):
        problems.append(f"sections out of order; required order: {sections}")
    if max_words:
        n = len(strip_code(text).split())
        if n > max_words:
            problems.append(f"{n} words, the limit is {max_words}")
    if max_pages and path.suffix == ".pdf":
        with pymupdf.open(path) as pdf:
            if pdf.page_count > max_pages:
                problems.append(f"{pdf.page_count} pages, the limit is {max_pages}")
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        not_run = sum(c.get("execution_count") is None for c in code)
        errors = sum(o.get("output_type") == "error" for c in code for o in c.get("outputs", []))
        if not code or not_run or errors:
            problems.append(f"notebook not fully executed: {len(code)} code cells, "
                            f"{not_run} not run, {errors} with errors")
    return CheckReport(not problems, problems)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


def _heading_position(text: str, section: str) -> int:
    """Offset of the first line that IS the heading, with or without `#` or a number."""
    target, pos = _norm(section), 0
    for line in text.splitlines(keepends=True):
        h = re.sub(r"^\d+ ", "", _norm(line.lstrip("# ")))
        if h and len(h) <= len(target) + 40 and (h == target or h.startswith(target + " ")):
            return pos
        pos += len(line)
    return -1
