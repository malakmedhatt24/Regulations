import pytest

from regbot.generator import ExtractiveGenerator, build_messages, format_context
from regbot.ingest import load_chunks
from regbot.retriever import HybridRetriever, tokenize


@pytest.fixture(scope="module")
def chunks():
    return load_chunks()


@pytest.fixture(scope="module")
def retriever(chunks):
    return HybridRetriever(chunks)


def find(chunks, doc_id, reg):
    return [c for c in chunks if c.doc_id == doc_id and c.reg == reg]


# --- ingestion ---------------------------------------------------------------


def test_both_documents_indexed(chunks):
    assert {c.doc_id for c in chunks} == {"general", "cs"}


def test_regulations_are_split_one_per_chunk(chunks):
    (pass_mark,) = find(chunks, "cs", "4.3")
    assert pass_mark.text == "The pass mark for any element of assessment is 40%."
    assert pass_mark.page_start == 12
    assert pass_mark.section == "Section 4: Assessment for the programme"


def test_subheadings_and_boilerplate(chunks):
    (interrupt,) = find(chunks, "general", "1.6")
    assert interrupt.subheading == "Interruption of studies"
    assert all("Page " not in c.text.splitlines()[-1] for c in chunks)
    assert {c.subheading for c in find(chunks, "cs", "4.34")} == {"Deferring an examination"}


def test_tables_stay_with_their_regulation(chunks):
    (scale,) = find(chunks, "cs", "7.2")
    assert "70% or above First Class Honours" in scale.text


def test_module_outlines_are_labelled(chunks):
    assert any("[CM1005]" in c.subheading and c.section.startswith("Appendix B") for c in chunks)


# --- retrieval ---------------------------------------------------------------

# (student question, document, regulation that answers it)
RETRIEVAL_CASES = [
    ("What is the pass mark for a module?", "cs", "4.3"),
    ("Can I pause my studies for a year because I am sick?", "general", "1.6"),
    ("Am I allowed to use ChatGPT in my coursework?", "general", "7.9"),
    ("How is my final degree classification calculated?", "cs", "7.5"),
    ("What marks do I need for a first class degree?", "cs", "7.2"),
    ("What is condonement?", "cs", "4.29"),
    ("What happens if I miss the continuing registration deadline?", "general", "1.20"),
    ("Can I defer my final exam?", "cs", "4.34"),
    ("What is the minimum period of registration for the BSc?", "cs", "2.2"),
    ("How do I get my marks rechecked and is there a fee?", "general", "11.2"),
    ("What does regulation 4.27 say?", "cs", "4.27"),
    ("How many attempts do I get at a module?", "cs", "4.18"),
    ("How many times can I resit a failed module?", "cs", "5.4"),
]


@pytest.mark.parametrize("question,doc_id,reg", RETRIEVAL_CASES)
def test_retrieval_finds_answering_regulation(retriever, question, doc_id, reg):
    top = retriever.search(question, k=5)
    assert any(h.chunk.doc_id == doc_id and h.chunk.reg == reg for h in top), [
        h.chunk.citation for h in top
    ]


def test_module_question_hits_appendix(retriever):
    top = retriever.search("What is the Data Science Graduate Certificate made up of?", k=3)
    assert any("Graduate Certificate in Data Science" in h.chunk.subheading for h in top)


def test_tokenizer_keeps_codes_and_numbers():
    toks = tokenize("Module CM1005 and regulation 4.27, pass at 40%")
    assert {"cm1005", "4.27", "40%"} <= set(toks)


# --- generation plumbing (no API calls) -----------------------------------


def test_prompt_contains_numbered_excerpts_and_history(retriever):
    hits = retriever.search("pass mark", k=2)
    msgs = build_messages(
        "And for the final project?",
        hits,
        [{"role": "user", "content": "What is the pass mark?"}, {"role": "assistant", "content": "40% [1]"}],
    )
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert '<excerpt n="1"' in msgs[-1]["content"] and '<excerpt n="2"' in msgs[-1]["content"]
    assert msgs[-1]["content"].endswith("Student question: And for the final project?")
    assert hits[0].chunk.citation in format_context(hits)


def test_extractive_fallback_cites_sources(retriever):
    hits = retriever.search("interrupt studies", k=3)
    out = "".join(ExtractiveGenerator().stream("q", hits, []))
    assert "General Regs, Reg 1.6" in out
