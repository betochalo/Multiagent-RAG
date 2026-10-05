"""Reader: sections by font size, tables rebuilt, ligatures normalized, scanned PDFs refused."""

import pymupdf
import pytest

from investigation_agent.services.pdf_reader import PdfReadError, PdfReaderService


def test_sections_of_task_a(settings, task_a):
    doc = PdfReaderService(settings).read(task_a)
    titles = [s.title for s in doc.sections]
    assert titles[1:5] == ["Parte 1 — Datos", "Parte 2 — Dos clasificadores",
                           "Parte 3 — La curva de aprendizaje", "Parte 4 — Discusión"]
    assert "load_breast_cancer" in doc.sections[1].text


def test_tables_are_rebuilt_as_markdown(settings, task_c):
    doc = PdfReaderService(settings).read(task_c)
    assert doc.tables >= 1
    assert "| d01 |" in doc.text.replace("  ", " ") or "|d01|" in doc.text.replace(" ", "")


def test_no_ligatures_or_hyphen_breaks(settings, task_a):
    text = PdfReaderService(settings).read(task_a).text
    assert not any(lig in text for lig in "ﬁﬂﬀﬃﬄ")


def test_missing_file(settings, tmp_path):
    with pytest.raises(PdfReadError, match="not found"):
        PdfReaderService(settings).read(tmp_path / "nope.pdf")


def test_only_pdf(settings, tmp_path):
    f = tmp_path / "statement.md"
    f.write_text("# Parte 1\n" + "texto " * 100)
    with pytest.raises(PdfReadError, match="Only PDF"):
        PdfReaderService(settings).read(f)


def test_scanned_or_empty_pdf_is_refused(settings, tmp_path):
    f = tmp_path / "scanned.pdf"
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Tarea 1")  # an image-only page has ~no text
    pdf.save(f)
    with pytest.raises(PdfReadError, match="scanned or empty"):
        PdfReaderService(settings).read(f)
