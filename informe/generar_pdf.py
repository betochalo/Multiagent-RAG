"""Render informe.md to informe.pdf: Markdown -> HTML (markdown-it) -> PDF (headless Chromium).

    uv run python informe/generar_pdf.py

Relative links to the repository (../src/..., ../corridas/...) become GitHub links so they
work inside the PDF. The Mermaid diagram is drawn by mermaid.js, loaded from a CDN.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/betochalo/Multiagent-RAG/blob/main"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 10pt; line-height: 1.45;
       color: #1a1a1a; }
h1 { font-size: 17pt; margin: 0 0 8pt; }
h2 { font-size: 14pt; margin-top: 20pt; border-bottom: 1px solid #ccc; padding-bottom: 3pt;
     break-after: avoid; }
h3 { font-size: 12pt; margin-top: 14pt; break-after: avoid; }
h4 { font-size: 10.5pt; margin-top: 12pt; break-after: avoid; }
a { color: #0b57d0; text-decoration: none; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.5pt; background: #f3f3f3;
       padding: 0 2px; border-radius: 2px; }
pre { background: #f6f6f6; border: 1px solid #e2e2e2; padding: 6pt 8pt; font-size: 8pt;
      white-space: pre-wrap; word-break: break-word; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 8.5pt;
        break-inside: avoid; }
th, td { border: 1px solid #d0d0d0; padding: 3pt 5pt; vertical-align: top; }
/* Long identifiers (test names, paths) must wrap or they squeeze the other columns. */
td code { overflow-wrap: anywhere; }
th { background: #f0f0f0; }
img { max-width: 100%; display: block; margin: 8pt auto; }
.mermaid { text-align: center; break-inside: avoid; }
hr { border: none; border-top: 1px solid #ccc; }
/* A bold one-line caption ("**Por tarea**") stays with the table below it. */
p:has(> strong:only-child) { break-after: avoid; }
/* Key-value tables written as "| | |" have an empty header row. */
thead:not(:has(th:not(:empty))) { display: none; }
"""


def to_html(md_text: str) -> str:
    md = MarkdownIt("commonmark").enable("table")
    default_fence = md.renderer.rules.get("fence")

    def fence(tokens, idx, options, env):
        tok = tokens[idx]
        if tok.info.strip() == "mermaid":
            return f'<div class="mermaid">\n{tok.content}</div>\n'
        return default_fence(tokens, idx, options, env)

    md.renderer.rules["fence"] = fence
    body = md.render(md_text)
    # Repository links: ../path -> GitHub (line anchors #L123 keep working there).
    body = re.sub(r'href="\.\./([^"]*)"', lambda m: f'href="{REPO_URL}/{m.group(1)}"', body)
    # Links inside informe/ (parte0/..., orquestador.mmd) also go to GitHub.
    body = re.sub(r'href="(?!https?:|#)([^"]+)"',
                  lambda m: f'href="{REPO_URL}/informe/{m.group(1)}"', body)
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Informe Taller 03 v2</title>
<style>{CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});</script>
</head><body>
{body}
</body></html>"""


def main() -> None:
    chromium = shutil.which("chromium") or shutil.which("google-chrome-stable")
    if not chromium:
        sys.exit("chromium not found")
    html_path = HERE / "informe.html"
    pdf_path = HERE / "informe.pdf"
    html_path.write_text(to_html((HERE / "informe.md").read_text(encoding="utf-8")),
                         encoding="utf-8")
    subprocess.run([chromium, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=15000", f"--print-to-pdf={pdf_path}",
                    html_path.as_uri()], check=True, capture_output=True)
    html_path.unlink()
    print(pdf_path)


if __name__ == "__main__":
    main()
