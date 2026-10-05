"""Corrections 4 and 5: the critic's code checks (Parte 0.c) and number provenance (0.a)."""

import json

from investigation_agent.services.checks import (
    check_execution, check_format, leakage, unsupported_numbers)

TREE_ON_TRAIN = """
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
X, y = load_breast_cancer(return_X_y=True)
model = DecisionTreeClassifier().fit(X, y)
acc = (model.predict(X) == y).mean()
"""

SCALER_BEFORE_SPLIT = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_scaled = StandardScaler().fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, random_state=42)
"""

CLEAN = """
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)
m = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr, ytr)
acc = m.score(Xte, yte)
"""


def _workdir(tmp_path, results):
    (tmp_path / "resultados.json").write_text(json.dumps(results))
    return tmp_path


def test_exit_zero_tree_is_rejected(tmp_path):
    """The 0.c script: exit code 0 and a results file, still rejected by code."""
    report = check_execution(TREE_ON_TRAIN, 0, "", _workdir(tmp_path, {"accuracy": 1.0}),
                             "resultados.json", expects_figure=False, timed_out=False)
    assert not report.passed
    assert any("implausible" in p for p in report.problems)
    assert any("leakage" in p for p in report.problems)


def test_scaler_fitted_before_split_is_leakage():
    assert any("before it is split" in p for p in leakage(SCALER_BEFORE_SPLIT))


def test_clean_script_has_no_leakage():
    assert leakage(CLEAN) == []


def test_missing_results_nan_and_figure(tmp_path):
    report = check_execution(CLEAN, 0, "", tmp_path, "resultados.json", True, False)
    assert any("no resultados.json" in p for p in report.problems)
    assert any("no .png" in p for p in report.problems)
    (tmp_path / "resultados.json").write_text('{"f1": NaN}')
    report = check_execution(CLEAN, 0, "", tmp_path, "resultados.json", False, False)
    assert any("NaN" in p for p in report.problems)


def test_crash_and_timeout(tmp_path):
    work = _workdir(tmp_path, {"acc": 0.9})
    crash = check_execution(CLEAN, 1, "Traceback\nValueError: boom", work, "resultados.json",
                            False, False)
    assert any("ValueError: boom" in p for p in crash.problems)
    killed = check_execution(CLEAN, None, "", work, "resultados.json", False, True)
    assert any("timeout" in p for p in killed.problems)


def test_provenance_flags_invented_numbers(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "resultados.json").write_text('{"acc_lr": 0.98830409}')
    report = "LR: 0.9883 (98.83 %). NB: 0.9415. Datos: 569 filas, test 0.3."
    out = unsupported_numbers(report, "test_size=0.3", tmp_path, exclude=set())
    assert out == ["0.9415"]


def test_provenance_reads_scientific_notation(tmp_path):
    """Control de Lectura 2: the numbers live around 1e-08. Before, "1.0003e-08" was read as
    "1" and an invented value in scientific notation escaped the check."""
    (tmp_path / "resultados.json").write_text('{"min_s2_B": 1.0003319062211514e-08}')
    report = "B float32: 1.0003319e-08 (SVD). B float64: 9.9999999e-09. Separación 1e-4."
    out = unsupported_numbers(report, "separacion 1e-4", tmp_path, exclude=set())
    assert out == ["9.9999999e-09"]


def test_provenance_ignores_trace_and_graph(tmp_path):
    (tmp_path / "traza.jsonl").write_text('{"x": 0.9415}')
    (tmp_path / "grafo.json").write_text('{"x": 0.9415}')
    assert unsupported_numbers("NB: 0.9415", "", tmp_path, exclude=set()) == ["0.9415"]


def test_format_sections_order_and_words(tmp_path):
    md = tmp_path / "reporte.md"
    md.write_text("# R\n\n## Resultados\n\nuno\n\n## Introducción\n\ndos\n")
    report = check_format(md, ["Introducción", "Resultados"], None, None)
    assert any("out of order" in p for p in report.problems)
    md.write_text("## Introducción\n\n" + "palabra " * 50 + "\n\n## Resultados\n")
    report = check_format(md, ["Introducción", "Resultados"], 20, None)
    assert report.problems == ["54 words, the limit is 20"]  # "##" counts, as in the evaluator
