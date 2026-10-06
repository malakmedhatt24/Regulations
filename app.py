"""Streamlit chat UI.  Run with:  streamlit run app.py"""

import streamlit as st

from regbot import config
from regbot.generator import ClaudeGenerator
from regbot.pipeline import RegBot

st.set_page_config(page_title="Regulations Assistant", page_icon="🎓")


@st.cache_resource
def get_bot() -> RegBot:
    return RegBot()


bot = get_bot()

with st.sidebar:
    st.header("About")
    st.write("Answers come only from these two documents:")
    for doc in config.SOURCES:
        st.markdown(f"- {doc.title}")
    if isinstance(bot.generator, ClaudeGenerator):
        st.caption(f"Model: `{bot.generator.model}` · effort `{bot.generator.effort}`")
    else:
        st.warning("No ANTHROPIC_API_KEY found: showing matching passages only.")
    st.caption(
        "This assistant can make mistakes. The University's interpretation of its "
        "regulations is final, so confirm anything important with the University."
    )
    if st.button("New conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("🎓 Regulations Assistant")
st.caption("Ask about registration, assessment, resits, progression, plagiarism, degree classification…")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(hits) -> None:
    with st.expander(f"Sources ({len(hits)} excerpts)"):
        for n, h in enumerate(hits, start=1):
            st.markdown(f"**[{n}] {h.chunk.citation}**  \n_{h.chunk.header}_")
            st.text(h.chunk.text)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits"):
            show_sources(msg["hits"])

if question := st.chat_input("e.g. How many times can I resit a module?"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the regulations…"):
            hits, chunks = bot.answer(question, history)
        reply = st.write_stream(chunks)
        show_sources(hits)
    st.session_state.messages.append({"role": "assistant", "content": reply, "hits": hits})
