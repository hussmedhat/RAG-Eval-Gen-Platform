# app/ui/streamlit_app.py
import time

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Evaluator-Generator RAG", layout="wide")
st.title("Evaluator–Generator RAG Platform")

tab_ingest, tab_ask = st.tabs(["📥 Ingest Knowledge", "💬 Ask a Question"])

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
                    f"{API_BASE}/ingest/file",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    timeout=120,
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
                resp = requests.post(f"{API_BASE}/ingest/url", json={"url": url}, timeout=200)
                if resp.ok:
                    st.success(f"Ingested {resp.json()['chunks_ingested']} chunks from {url}")
                else:
                    st.error(resp.json().get("detail", resp.text))
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

    st.divider()
    st.subheader("Add a Wikipedia page")
    wiki_query = st.text_input("Wikipedia topic", key="wiki_input")
    wiki_lang = st.text_input("Language code", value="en", key="wiki_lang")
    if st.button("Ingest Wikipedia") and wiki_query:
        with st.spinner("Fetching and ingesting..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/ingest/wikipedia",
                    json={"query": wiki_query, "lang": wiki_lang},
                    timeout=60,
                )
                if resp.ok:
                    st.success(f"Ingested {resp.json()['chunks_ingested']} chunks")
                else:
                    st.error(resp.json().get("detail", resp.text))
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

with tab_ask:
    st.subheader("Ask a question")
    question = st.text_area("Your question", height=100)

    if st.button("Ask") and question.strip():
        with st.spinner("Running Generator ↔ Evaluator loop..."):
            start = time.time()
            try:
                resp = requests.post(f"{API_BASE}/ask", json={"question": question}, timeout=180)
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
                resp = None

        if resp is not None:
            elapsed = time.time() - start
            if resp.ok:
                data = resp.json()
                st.markdown("### Final Answer")
                st.write(data["answer"])

                if data.get("disclaimer"):
                    st.warning(data["disclaimer"])
                else:
                    st.success(f"Accepted after {data['loops_used']} loop(s) in {elapsed:.1f}s")

                with st.expander("Evaluation trace"):
                    for i, ev in enumerate(data["evaluation_history"], start=1):
                        st.markdown(f"**Loop {i}** — accepted: `{ev['accepted']}`")
                        st.write({
                            "accuracy": ev["accuracy_score"],
                            "relevance": ev["relevance_score"],
                            "completeness": ev["completeness_score"],
                            "grounding": ev["grounding_score"],
                        })
                        if ev["feedback"]:
                            st.caption(f"Feedback: {ev['feedback']}")
                        st.divider()
            else:
                st.error(resp.json().get("detail", resp.text))
