"""Deliverable writers: Markdown as is, Markdown printed to PDF, and an executed notebook.

The PDF is printed with PyMuPDF (`Story`), the same way the kit prints the task statements,
with no browser or LaTeX. The notebook is executed by nbconvert inside the sandbox: the
kernel inherits the sandbox's empty environment and has no network.
"""

import re
import shutil
import sys
from pathlib import Path

import nbformat
import pymupdf
from markdown_it import MarkdownIt

from investigation_agent.services.sandbox import RunResult, Sandbox

_CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.3; }
h1 { font-size: 15pt; margin-bottom: 4pt; }
h2 { font-size: 12pt; margin-top: 8pt; margin-bottom: 3pt; }
p { margin-top: 2pt; margin-bottom: 4pt; text-align: justify; }
table { border-collapse: collapse; margin: 4pt 0; }
th, td { border: 0.6pt solid #888; padding: 2pt 4pt; font-size: 9pt; }
th { background-color: #eeeeee; }
code { font-family: monospace; font-size: 9pt; }
img { width: 300; }
"""


def write_markdown(text: str, path: Path) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def write_pdf(markdown: str, path: Path, assets_dir: Path) -> int:
    """Print Markdown to an A4 PDF. Images are resolved against `assets_dir`. Returns the
    page count, which the caller checks against the statement's limit."""
    markdown = re.sub(r"\$([^$\n]+)\$", r"\1", markdown)
    html = MarkdownIt("commonmark").enable("table").render(markdown)
    story = pymupdf.Story(html=html, user_css=_CSS, archive=pymupdf.Archive(str(assets_dir)))
    writer = pymupdf.DocumentWriter(str(path))
    a4 = pymupdf.paper_rect("a4")
    box = a4 + (50, 50, -50, -50)
    more = True
    while more:
        device = writer.begin_page(a4)
        more, _ = story.place(box)
        story.draw(device)
        writer.end_page()
    writer.close()
    with pymupdf.open(path) as pdf:
        return pdf.page_count


def write_notebook(cells: list[dict], path: Path, sandbox: Sandbox) -> RunResult:
    """Build a notebook from [{"type": "markdown"|"code", "source": str}] and execute it in
    place, in the sandbox, so every number in it is a cell output."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    for cell in cells:
        make = nbformat.v4.new_code_cell if cell["type"] == "code" else nbformat.v4.new_markdown_cell
        nb.cells.append(make(cell["source"]))
    nbformat.write(nb, path)
    return sandbox.run(
        [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
         "--ExecutePreprocessor.timeout=600", path.name],
        workdir=path.parent)


def copy_figures(src_dirs: list[Path], dest: Path) -> list[str]:
    """Copy the figures of approved scripts next to the deliverable. Returns their names."""
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    for src in src_dirs:
        for png in sorted(src.glob("*.png")):
            shutil.copy2(png, dest / png.name)
            names.append(png.name)
    return names
