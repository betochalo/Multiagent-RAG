"""PDF reader: text per page with tables rebuilt as Markdown, split into sections by heading.

What the code checks, because each of these fails silently downstream:

- **Ligatures.** "clasiﬁcadores" carries U+FB01; a regex or BM25 term "clasificadores" never
  matches it. Text is NFKC-normalized.
- **Justified text.** The PDF breaks lines wherever it wants. Lines of a paragraph are
  joined, and a word hyphenated across lines is rejoined.
- **Tables.** Extracted as text, a table is a soup of cells; the reader rebuilds it as a
  Markdown table and drops its cells from the running text so nothing appears twice.
- **No text.** Scanned PDFs are not supported: a document with fewer than `pdf_min_chars`
  extracted characters is rejected with `PdfReadError` instead of yielding an empty task.

Headings are found by font size relative to the body font, so the reader needs no
knowledge of a given document's template. Single-column layout is assumed.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from investigation_agent.config.settings import Settings

_BOLD = 16  # PyMuPDF span flags
_MONO = 8


class PdfReadError(ValueError):
    """The file is not a readable PDF with enough text to work on."""


@dataclass
class Section:
    title: str
    level: int  # 0 = document title, 1 = top-level section, ...
    page: int  # 1-based page where the section starts
    text: str = ""  # body as Markdown, tables included


@dataclass
class ParsedDocument:
    source: str
    pages: list[str]  # Markdown per page: page n is pages[n - 1]
    sections: list[Section]
    tables: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(self.pages)


class PdfReaderService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def read(self, path: str | Path) -> ParsedDocument:
        path = Path(path)
        if not path.is_file():
            raise PdfReadError(f"File not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise PdfReadError(f"Only PDF files are supported, got {path.name!r}")
        try:
            pdf = pymupdf.open(path)
        except (pymupdf.FileDataError, RuntimeError) as err:
            raise PdfReadError(f"Cannot open {path.name!r} as a PDF: {err}") from err

        with pdf:
            doc = ParsedDocument(source=path.name, pages=[], sections=[])
            body_size, heading_levels = self._font_profile(pdf)
            for page in pdf:
                items = self._page_items(page, doc, heading_levels)
                doc.pages.append(_normalize("\n\n".join(text for _, text in items)))
                self._add_to_sections(doc, items, page.number + 1)

        chars = len(re.sub(r"\s", "", doc.text))
        if chars < self.settings.pdf_min_chars:
            raise PdfReadError(
                f"{path.name!r} has only {chars} characters of text (minimum "
                f"{self.settings.pdf_min_chars}): it may be scanned or empty, which is not supported")
        for s in doc.sections:
            s.text = _normalize(s.text)
        return doc

    # ---------------------------------------------------------------- fonts
    def _font_profile(self, pdf: pymupdf.Document) -> tuple[float, dict[float, int]]:
        """The body font size (the one carrying most characters) and a level per heading
        size: the largest size is level 0, the next level 1, and so on."""
        chars: Counter[float] = Counter()
        for page in pdf:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        chars[round(span["size"], 1)] += len(span["text"].strip())
        if not chars:
            return 0.0, {}
        body = chars.most_common(1)[0][0]
        bigger = sorted((s for s in chars if s >= body * self.settings.pdf_heading_ratio),
                        reverse=True)
        return body, {size: level for level, size in enumerate(bigger)}

    # ---------------------------------------------------------------- one page
    def _page_items(self, page: pymupdf.Page, doc: ParsedDocument,
                    heading_levels: dict[float, int]) -> list[tuple[int | None, str]]:
        """The page as an ordered list of (heading level or None, Markdown text)."""
        # A 1-row "table" is a framed note, not data.
        tables = [t for t in page.find_tables().tables if t.row_count >= 2 and t.col_count >= 2]
        doc.tables += len(tables)
        boxes = [pymupdf.Rect(t.bbox) for t in tables]
        positioned: list[tuple[float, int | None, str]] = [
            (t.bbox[1], None, _markdown_table(t.extract())) for t in tables]

        for block in page.get_text("dict", sort=True)["blocks"]:
            if block.get("type") != 0:
                continue
            # Lines inside a table are already in its Markdown version.
            lines = [l for l in block["lines"]
                     if not any(pymupdf.Rect(l["bbox"]).intersects(b) for b in boxes)]
            positioned += self._block_items(lines, heading_levels)

        positioned.sort(key=lambda item: item[0])
        return [(level, text) for _, level, text in positioned]

    @staticmethod
    def _block_items(lines: list[dict],
                     heading_levels: dict[float, int]) -> list[tuple[float, int | None, str]]:
        """Group a block's lines into headings and paragraphs. Consecutive heading lines of
        the same level are one heading (a long title wraps)."""
        out: list[tuple[float, int | None, str]] = []
        paragraph: list[str] = []
        mono = False
        y0 = 0.0

        def flush():
            if paragraph:
                text = _join_lines(paragraph, "\n" if mono else " ")
                out.append((y0, None, f"```\n{text}\n```" if mono else text))
                paragraph.clear()

        for line in lines:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"]).strip()
            size = round(max(s["size"] for s in spans), 1)
            level = heading_levels.get(size) if all(s["flags"] & _BOLD for s in spans) else None
            if level is not None:
                flush()
                if out and out[-1][1] == level:
                    prev_y, _, prev_text = out[-1]
                    out[-1] = (prev_y, level, f"{prev_text} {text}")
                else:
                    out.append((line["bbox"][1], level, text))
                continue
            line_mono = all(s["flags"] & _MONO for s in spans)
            if paragraph and line_mono != mono:
                flush()
            if not paragraph:
                y0, mono = line["bbox"][1], line_mono
            paragraph.append(text)
        flush()
        return out

    # ---------------------------------------------------------------- sections
    @staticmethod
    def _add_to_sections(doc: ParsedDocument, items: list[tuple[int | None, str]],
                         page: int) -> None:
        for level, text in items:
            if level is not None:
                doc.sections.append(Section(title=_normalize(text), level=level, page=page))
                continue
            if not doc.sections:  # text before any heading
                doc.sections.append(Section(title="", level=0, page=page))
            current = doc.sections[-1]
            current.text = f"{current.text}\n\n{text}" if current.text else text


# -------------------------------------------------------------------- helpers
def _normalize(text: str) -> str:
    """NFKC turns ligatures (U+FB01 'ﬁ') and other compatibility forms into plain letters."""
    return unicodedata.normalize("NFKC", text).strip()


def _join_lines(lines: list[str], joiner: str) -> str:
    """Join the visual lines of a paragraph, rejoining words hyphenated at a line end
    ("experi-" + "mento"). A hyphen before an uppercase letter or digit is kept."""
    out = ""
    for line in lines:
        line = line.strip() if joiner == " " else line.rstrip()
        if not out:
            out = line
        elif joiner == " " and re.search(r"\w-$", out) and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out}{joiner}{line}"
    return out


def _markdown_table(rows: list[list[str | None]]) -> str:
    cells = [[_join_lines((c or "").splitlines(), " ").replace("|", "\\|") for c in row]
             for row in rows]
    header, *body = cells
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)
