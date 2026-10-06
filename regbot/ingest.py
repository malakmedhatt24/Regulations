"""Turn the regulation PDFs into citation-ready chunks.

The regulations are written as numbered clauses ("1.6", "4.27", ...) grouped under
numbered sections, so we chunk on those boundaries rather than on a fixed window:
one chunk ~= one regulation, tagged with its document, section, sub-heading and page.
Appendices (programme structures, module outlines, marking criteria) have no clause
numbers and are split on programme / module headings instead.

Run directly to (re)build the index:  python -m regbot.ingest
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pdfplumber

from . import config

# Running headers / footers printed on every page.
_BOILERPLATE = [
    re.compile(r"^University of London General Regulations \d{4}-\d{4}$"),
    re.compile(r"^Programme Regulations \d{4}-\d{4} Computer Science and specialisms$"),
    re.compile(r"^\(BSc, Graduate Diploma, Graduate Certificate\)$"),
    re.compile(r"^University of London( Page \d+ of \d+| \d+)$"),
]
_SECTION = re.compile(r"^(\d{1,2}) ([A-Z][^.]{2,80})$")
_REG = re.compile(r"^(\d{1,2})\.(\d{1,2})$")
_APPENDIX = re.compile(r"^Appendix ([A-Z]) [–-] (.+)$")
_PROGRAMME = re.compile(r"^(BSc Computer Science.*|Graduate (Diploma|Certificate) in .+)$")
_MODULE = re.compile(r"^([A-Z][^•\[]{2,90}) \[(CM\d{4})\]$")
_LEVEL = re.compile(r"^Level [4-6]$")

SKIP_PAGES = {1, 2}  # cover page and table of contents


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    doc_short: str
    section: str
    subheading: str
    reg: str
    page_start: int
    page_end: int
    text: str

    @property
    def citation(self) -> str:
        where = f"Reg {self.reg}" if self.reg else (self.subheading or self.section)
        pages = (
            f"p.{self.page_start}"
            if self.page_start == self.page_end
            else f"pp.{self.page_start}-{self.page_end}"
        )
        return f"{self.doc_short}, {where}, {pages}"

    @property
    def header(self) -> str:
        parts = [self.doc_title, self.section]
        if self.subheading:
            parts.append(self.subheading)
        if self.reg:
            parts.append(f"Regulation {self.reg}")
        return " > ".join(p for p in parts if p)


@dataclass
class _Builder:
    doc: config.SourceDoc
    section: str = "About these regulations: definitions and summary of changes"
    subheading: str = ""
    reg: str = ""
    lines: list[tuple[int, str]] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)

    def flush(self) -> None:
        body = [ln for ln in self.lines if ln[1].strip()]
        self.lines = []
        if not body:
            return
        for piece in _split_long(body, config.MAX_CHUNK_CHARS):
            text = "\n".join(t for _, t in piece).strip()
            if len(text) < 25:  # stray heading fragments
                continue
            self.chunks.append(
                Chunk(
                    id=f"{self.doc.doc_id}-{len(self.chunks):04d}",
                    doc_id=self.doc.doc_id,
                    doc_title=self.doc.title,
                    doc_short=self.doc.short,
                    section=self.section,
                    subheading=self.subheading,
                    reg=self.reg,
                    page_start=piece[0][0],
                    page_end=piece[-1][0],
                    text=text,
                )
            )


def _split_long(lines: list[tuple[int, str]], limit: int) -> list[list[tuple[int, str]]]:
    """Split an over-long block on line boundaries, overlapping by two lines."""
    total = sum(len(t) + 1 for _, t in lines)
    if total <= limit:
        return [lines]
    pieces, cur, size = [], [], 0
    for ln in lines:
        if cur and size + len(ln[1]) > limit:
            pieces.append(cur)
            cur = cur[-2:]
            size = sum(len(t) + 1 for _, t in cur)
        cur.append(ln)
        size += len(ln[1]) + 1
    if cur:
        pieces.append(cur)
    return pieces


def _is_subheading(line: str) -> bool:
    s = line.strip()
    return (
        3 <= len(s) <= 70
        and s[0].isupper()
        and not s.endswith((".", ",", ";", ":"))
        and not s.startswith(("•", "o "))
        and "%" not in s
        and not _REG.match(s)
    )


def _page_lines(path: Path) -> list[tuple[int, str, bool]]:
    """(page number, line text, is the whole line bold) for every body line."""
    out: list[tuple[int, str, bool]] = []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            if pno in SKIP_PAGES:
                continue
            for ln in page.extract_text_lines(return_chars=True):
                line = ln["text"].strip()
                if not line or any(p.match(line) for p in _BOILERPLATE):
                    continue
                chars = [c for c in ln["chars"] if c["text"].strip()]
                bold = bool(chars) and all("Bold" in c["fontname"] for c in chars)
                out.append((pno, line, bold))
    return out


def _bold_text(lines: list[tuple[int, str, bool]], i: int) -> bool:
    """Is line i bold running text, i.e. not a clause number or section heading?"""
    if not 0 <= i < len(lines):
        return False
    _, text, bold = lines[i]
    return bold and not _REG.match(text) and not _SECTION.match(text)


def chunk_document(doc: config.SourceDoc) -> list[Chunk]:
    b = _Builder(doc)
    section_no = 0
    in_appendix = ""

    lines = _page_lines(doc.path)
    for idx, (pno, line, bold) in enumerate(lines):
        # Long programme names wrap onto a second line: "... (Machine Learning and Artificial" / "Intelligence)".
        if in_appendix == "A" and b.subheading.count("(") > b.subheading.count(")") and not b.lines:
            b.subheading = f"{b.subheading} {line}"
            continue
        if m := _APPENDIX.match(line):
            b.flush()
            in_appendix = m.group(1)
            b.section, b.subheading, b.reg = f"Appendix {m.group(1)} – {m.group(2)}", "", ""
            continue

        if not in_appendix:
            # Section headings are numbered 1, 2, 3 ... in order; requiring the next
            # number avoids mistaking a wrapped line like "30 September ..." for one.
            m = _SECTION.match(line)
            if m and int(m.group(1)) == section_no + 1:
                b.flush()
                section_no += 1
                b.section, b.subheading, b.reg = f"Section {section_no}: {m.group(2)}", "", ""
                continue
            # Sub-headings ("Interruption of studies") are bold lines on their own.
            # Bold table-header rows either come in runs (a bold neighbour that is
            # not itself a heading or clause number) or follow "...as follows:".
            if (
                section_no > 0
                and bold
                and _is_subheading(line)
                and not (_bold_text(lines, idx - 1) or _bold_text(lines, idx + 1))
                and not lines[idx - 1][1].endswith(":")
            ):
                b.flush()
                b.subheading, b.reg = line, ""
                continue
            if m := _REG.match(line):
                b.flush()
                b.reg = f"{m.group(1)}.{m.group(2)}"
                continue
        elif in_appendix == "A" and _PROGRAMME.match(line) and "•" not in line:
            b.flush()
            b.subheading = line
            continue
        elif in_appendix == "B":
            if _LEVEL.match(line):
                b.flush()
                b.subheading = line
                continue
            if m := _MODULE.match(line):
                b.flush()
                level = b.subheading.split(" – ")[0] if b.subheading else ""
                b.subheading = f"{level} – {line}" if level else line
                b.lines.append((pno, line))
                continue

        b.lines.append((pno, line))

    b.flush()
    return b.chunks


def build_index(out_dir: Path = config.INDEX_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in config.SOURCES:
        chunks.extend(chunk_document(doc))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks.json").write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=1)
    )
    return chunks


def load_chunks(index_dir: Path = config.INDEX_DIR) -> list[Chunk]:
    path = index_dir / "chunks.json"
    if not path.exists():
        return build_index(index_dir)
    return [Chunk(**d) for d in json.loads(path.read_text())]


if __name__ == "__main__":
    built = build_index()
    for doc in config.SOURCES:
        n = sum(c.doc_id == doc.doc_id for c in built)
        print(f"{doc.short}: {n} chunks")
    print(f"Wrote {config.INDEX_DIR / 'chunks.json'}")
