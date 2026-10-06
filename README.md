# University Regulations Assistant (RAG chatbot)

A retrieval-augmented chatbot that answers student questions using **only** two documents:

| Document | File |
|---|---|
| University of London General Regulations 2026-2027 | `data/general-regs-2026-27.pdf` |
| Programme Regulations 2026-2027: Computer Science (BSc, Graduate Diploma, Graduate Certificate) | `data/prog-regs-computer-science-bsc-2026-27.pdf` |

Every answer cites the regulation it relies on (for example *CS Programme Regs, Reg 4.3, p.12*). If the answer isn't in the documents, the bot says so instead of guessing.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # then put your Anthropic API key in .env
streamlit run app.py            # web chat at http://localhost:8501
# or
python cli.py                   # terminal chat
python cli.py "Can I defer an exam?"
```

The search index is built automatically on first run. To rebuild it after you replace a PDF, run `python -m regbot.ingest`.

Without an API key the bot still works, but in **retrieval-only mode**: it shows the most relevant passages verbatim instead of writing an answer.

## How it works

```
PDFs ──► ingest.py ──► chunks (one per regulation) ──► retriever.py ──► top-k excerpts ──► generator.py (Claude) ──► cited answer
```

1. **Ingestion (`regbot/ingest.py`).** `pdfplumber` extracts the text and drops running headers, footers, the cover and the contents page. The text is then split along the documents' own structure instead of fixed-size windows:
   - numbered sections (`4 Assessment for the programme`) and clauses (`4.3`), so one chunk ≈ one regulation;
   - bold sub-headings (`Interruption of studies`, `Deferring an examination`), detected from the font;
   - appendices split per programme (Appendix A) and per module (Appendix B, e.g. `[CM1005]`).

   Each chunk records its document, section, sub-heading, regulation number and pages, which become its citation. Tables (pass criteria, classification bands) stay with the regulation they belong to.
2. **Retrieval (`regbot/retriever.py`).** This is a hybrid search, so no embedding model has to be downloaded:
   - **BM25** over stemmed words;
   - **TF-IDF cosine** over word unigrams and bigrams;
   - the two rankings merged with **Reciprocal Rank Fusion**;
   - a small synonym map from student phrasing to the regulations' wording (*ChatGPT → artificial intelligence*, *pause → interruption*, *resit → attempt*…);
   - an exact boost when the question cites a regulation number ("what does 4.27 say?");
   - for follow-up questions, a second search that includes the previous question.

   Optional: install `sentence-transformers` and set `REGBOT_EMBED_MODEL` (for example `BAAI/bge-small-en-v1.5`) to add dense semantic search as a third ranker.
3. **Generation (`regbot/generator.py`).** The top excerpts are numbered and passed to Claude with a system prompt that:
   - limits answers to the excerpts;
   - requires `[n]` citations;
   - explains how the General and Programme regulations relate (programme rules apply where the General Regulations defer to them);
   - declines off-topic questions.

   Responses stream into the UI. The system prompt is prompt-cached, and server-side refusal fallback is enabled.

## Configuration

Set these in `.env` or as environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Required for generated answers |
| `REGBOT_MODEL` | `claude-opus-5-5` | Claude model |
| `REGBOT_EFFORT` | `medium` | `low` is faster and cheaper; `high` is more thorough |
| `REGBOT_TOP_K` | `8` | Excerpts sent per question |
| `REGBOT_EMBED_MODEL` | none | Optional dense retriever |

## Tests

```bash
pytest
```

The tests check the chunking (regulation boundaries, sub-headings, tables, appendices) and run a retrieval benchmark of student-style questions. Each question must return the regulation that answers it in the top 5. Add a case to `RETRIEVAL_CASES` in `tests/test_regbot.py` whenever you find a question the bot answers badly.

## Project layout

```
app.py              Streamlit chat UI
cli.py              terminal chat
regbot/config.py    document list and settings
regbot/ingest.py    PDF → structured chunks
regbot/retriever.py hybrid BM25 + TF-IDF (+ optional dense) retriever
regbot/generator.py Claude prompt and streaming; offline extractive fallback
regbot/pipeline.py  retrieve → generate
data/               the two regulation PDFs
tests/              chunking and retrieval tests
```

To add another programme's regulations, put the PDF in `data/`, add a `SourceDoc` to `regbot/config.py`, and run `python -m regbot.ingest`.
