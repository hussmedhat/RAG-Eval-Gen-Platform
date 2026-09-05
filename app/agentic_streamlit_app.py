"""
Streamlit front end for the 3-agent pipeline (Retriever -> Analyst -> Answer),
talking to app/agentic_main.py.

Two tabs:
  - Ingest Knowledge: same as Part 1 (file / URL / Wikipedia)
  - Ask: a lightweight chat interface. Unlike Part 1, this passes running
    conversation history to the backend so the Retriever's Query Rewriter
    tool can resolve follow-up questions ("what about its downsides?")
    into standalone queries.

"""
import time

import requests
import streamlit as st

st.set_page_config(page_title="Agentic RAG Platform", layout="wide")

with st.sidebar:
    st.header("Settings")
    api_base = st.text_input("Backend URL", value="http://localhost:8000")
    if st.button("Clear conversation"):
        st.session_state.pop("chat_history", None)
        st.rerun()

st.title("Agentic RAG Platform")
st.caption("Retriever \u2192 Analyst \u2192 Answer")

tab_ingest, tab_ask = st.tabs(["\U0001F4E5 Ingest Knowledge", "\U0001F4AC Ask"])

# =========================================================
# INGEST TAB
# =========================================================
with tab_ingest:
    st.subheader("Upload a file")
    uploaded = st.file_uploader(
        "PDF, DOCX, TXT, PPTX, WAV, or source code",
        type=["pdf", "docx", "txt", "pptx", "ppt", "wav", "py", "js", "ts",
              "java", "cpp", "c", "go", "rb", "rs", "cs"],
    )
    if uploaded is not None and st.button("Ingest file"):
        with st.spinner("Ingesting..."):
            try:
                resp = requests.post(
                    f"{api_base}/ingest/file",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    timeout=180,
                )
                if resp.ok:
                    st.success(f"Ingested {resp.json()['chunks_ingested']} chunks from {uploaded.name}")
                else:
                    st.error(resp.json().get("detail", resp.text))
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

    st.divider()
    st.subheader("Add a web page")
    url = st.text_input("URL", key="url_input")
    if st.button("Ingest URL") and url:
        with st.spinner("Fetching and ingesting..."):
            try:
                resp = requests.post(f"{api_base}/ingest/url", json={"url": url}, timeout=120)
                if resp.ok:
                    st.success(f"Ingested {resp.json()['chunks_ingested']} chunks from {url}")
                else:
                    st.error(resp.json().get("detail", resp.text))
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

    st.divider()
    st.subheader("Add a Wikipedia page")
    wiki_query = st.text_input("Wikipedia topic (title, not a URL)", key="wiki_input")
    wiki_lang = st.text_input("Language code", value="en", key="wiki_lang")
    if st.button("Ingest Wikipedia") and wiki_query:
        with st.spinner("Fetching and ingesting..."):
            try:
                resp = requests.post(
                    f"{api_base}/ingest/wikipedia",
                    json={"query": wiki_query, "lang": wiki_lang},
                    timeout=120,
                )
                if resp.ok:
                    st.success(f"Ingested {resp.json()['chunks_ingested']} chunks")
                else:
                    st.error(resp.json().get("detail", resp.text))
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

# =========================================================
# ASK TAB
# =========================================================
with tab_ask:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {"question": ..., "answer": ...}

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])

    question = st.chat_input("Ask a question about your ingested documents")

    if question:
        with st.chat_message("user"):
            st.write(question)

        history_text = "\n".join(
            f"Q: {t['question']}\nA: {t['answer']}" for t in st.session_state.chat_history
        )

        with st.chat_message("assistant"):
            with st.spinner("Retrieving \u2192 analyzing \u2192 answering..."):
                start = time.time()
                try:
                    resp = requests.post(
                        f"{api_base}/ask",
                        json={"question": question, "history": history_text},
                        timeout=180,
                    )
                except requests.RequestException as e:
                    st.error(f"Request failed: {e}")
                    resp = None

            if resp is not None:
                elapsed = time.time() - start
                if resp.ok:
                    data = resp.json()
                    answer_text = data["answer"]
                    st.write(answer_text)

                    if data.get("disclaimer"):
                        st.warning(data["disclaimer"])

                    st.caption(
                        f"{data['num_citations']} citation(s) \u2022 "
                        f"{data['loops_used']} loop(s) \u2022 {elapsed:.1f}s"
                    )

                    with st.expander("Agent trace", expanded=False):
                        st.write(f"**Evidence judged sufficient:** {data['sufficient']}")
                        st.write(f"**Retrieval/analysis loops used:** {data['loops_used']}")
                        if data["max_loops_reached"]:
                            st.warning(
                                "Max feedback loops reached before evidence was "
                                "confirmed sufficient \u2014 answer may be incomplete."
                            )

                        trace = data.get("trace", [])
                        if trace:
                            st.markdown("**Pipeline log:**")
                            log_lines = "\n".join(
                                f"{entry['level']:<7} {entry['logger'].split('.')[-1]}: {entry['message']}"
                                for entry in trace
                            )
                            st.code(log_lines, language="log")

                        if data["key_facts"]:
                            st.markdown("**Key facts noted by the Analyst:**")
                            for fact in data["key_facts"]:
                                st.markdown(f"- {fact}")

                        if data["tables"]:
                            st.markdown("**Tables extracted:**")
                            for t in data["tables"]:
                                headers = t.get("headers", [])
                                rows = t.get("rows", [])
                                if headers and rows:
                                    st.table([dict(zip(headers, row)) for row in rows])

                        comparison = data["comparison"]
                        if comparison:
                            agreements = comparison.get("agreements", [])
                            contradictions = comparison.get("contradictions", [])
                            source_points = comparison.get("source_specific_points", {})

                            if agreements:
                                st.markdown("**Agreements across sources:**")
                                for a in agreements:
                                    st.markdown(f"- {a}")

                            if contradictions:
                                st.markdown("**Contradictions across sources:**")
                                for c in contradictions:
                                    st.markdown(f"- {c}")

                            if source_points:
                                st.markdown("**Source-specific points:**")
                                for source, points in source_points.items():
                                    st.markdown(f"*{source}*")
                                    for p in points:
                                        st.markdown(f"  - {p}")

                        if data["data_analysis"]:
                            st.markdown("**Numeric analysis:**")
                            da = data["data_analysis"]
                            if "ranking" in da:
                                st.markdown("*Ranking:*")
                                st.table(da["ranking"])
                            if "stats" in da:
                                st.markdown("*Stats:*")
                                st.json(da["stats"])

                        st.markdown("**Raw response:**")
                        st.json(data)
                    st.session_state.chat_history.append({"question": question, "answer": answer_text})
                else:
                    st.error(resp.json().get("detail", resp.text))
