from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.resrag import HybridIndex, extract_pdf

load_dotenv()

st.set_page_config(page_title="ResRAG", page_icon="R", layout="wide", initial_sidebar_state="expanded")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")

st.markdown(
    """
<style>
#MainMenu, footer {visibility:hidden;}
header {background:transparent!important;}
.stApp {background:#ffffff; color:#2f2f2f;}
section[data-testid="stSidebar"] {background:#f7f7f8; border-right:1px solid #e5e5e5;}
section[data-testid="stSidebar"] > div {padding-top:1rem;}
.block-container {max-width:1000px; padding-top:1.2rem; padding-bottom:7rem;}
.topbar {display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem;gap:16px;}
.brand {font-weight:650;font-size:17px;letter-spacing:-.2px;}
.brand-sub {color:#777;font-size:12px;margin-top:2px;}
.doc-pill {border:1px solid #e5e5e5;background:#fafafa;border-radius:999px;padding:7px 11px;font-size:12px;color:#666;max-width:360px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.empty {min-height:52vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;}
.empty h1 {font-size:32px;letter-spacing:-1.1px;margin:0 0 8px;font-weight:650;color:#2f2f2f;}
.empty p {color:#737373;margin:0;max-width:500px;}
.answer {line-height:1.62;}
.source-card {border:1px solid #e7e7e7;border-radius:12px;padding:12px 14px;background:#fbfbfb;margin-bottom:10px;}
.source-meta {font-size:11px;color:#777;margin-bottom:5px;text-transform:uppercase;letter-spacing:.06em;}
.source-text {font-size:13px;line-height:1.55;color:#414141;white-space:pre-wrap;}
.status-chip {display:inline-block;font-size:11px;color:#666;background:#f3f3f3;border-radius:999px;padding:3px 7px;margin-left:5px;}
.small-muted {font-size:12px;color:#7b7b7b;}
.stChatMessage {padding-top:.6rem;padding-bottom:.6rem;}
.stChatMessage[data-testid="user-message"] {background:transparent;}
.stChatInputContainer {border-top:0!important;}
button[kind="primary"] {border-radius:10px;}
[data-testid="stFileUploader"] section {border:1px dashed #cfcfcf;background:#fff;border-radius:12px;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_embedder(name: str):
    return SentenceTransformer(name)


@st.cache_resource(show_spinner=False)
def get_reranker(name: str):
    return CrossEncoder(name) if name else None


def build_index(chunks):
    return HybridIndex(
        EMBEDDING_MODEL,
        RERANKER_MODEL or None,
        embedder=get_embedder(EMBEDDING_MODEL),
        reranker=get_reranker(RERANKER_MODEL),
    )


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Add OPENAI_API_KEY to your .env file.")
    kwargs = {"api_key": api_key}
    if os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    return OpenAI(**kwargs)


def generate_answer(question: str, retrieved: list[dict]) -> str:
    if not OPENAI_MODEL:
        raise RuntimeError("Add OPENAI_MODEL to your .env file.")
    context = "\n\n---\n\n".join(
        f"[Page {x['chunk'].page} · {x['chunk'].kind}]\n{x['chunk'].text}" for x in retrieved
    )
    system = """You answer questions using only the supplied PDF excerpts.
Do not use outside knowledge to fill gaps. If the excerpts do not support an answer, say so.
Every factual statement should include a page citation in the form [Page N].
When a table excerpt supports a number or comparison, preserve the table's meaning and do not invent values.
Do not invent citations, facts, numbers, quotations, or conclusions.
Write naturally and directly, without mentioning the retrieval process unless useful."""
    response = get_client().chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"PDF excerpts:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content or "I couldn't generate an answer from the document."


def render_sources(sources: list[dict]):
    with st.expander(f"Sources · {len(sources)} passages"):
        for item in sources:
            chunk = item["chunk"]
            score = item.get("rerank_score", item.get("hybrid_score", 0))
            label = chunk.kind.capitalize()
            if chunk.kind == "table":
                label = "Table"
            st.markdown(
                f"<div class='source-card'><div class='source-meta'>Page {chunk.page} · {label} · {score:.3f}</div><div class='source-text'>{chunk.text}</div></div>",
                unsafe_allow_html=True,
            )


with st.sidebar:
    st.markdown("<div class='brand'>ResRAG</div><div class='brand-sub'>PDF knowledge, kept grounded.</div>", unsafe_allow_html=True)
    st.write("")
    uploaded = st.file_uploader("Document", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        st.caption(uploaded.name)
        if st.button("Add to chat", type="primary", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getbuffer())
                path = tmp.name
            try:
                with st.spinner("Reading document…"):
                    chunks = extract_pdf(path)
                    index = build_index(chunks)
                    index.build(chunks)
                st.session_state.index = index
                st.session_state.doc_name = uploaded.name
                st.session_state.messages = []
                table_count = sum(c.kind == "table" for c in chunks)
                ocr_count = sum(c.kind == "ocr" for c in chunks)
                st.success(f"Ready · {len(chunks)} passages")
                if table_count:
                    st.caption(f"Detected {table_count} native table passage{'s' if table_count != 1 else ''}.")
                if ocr_count:
                    st.caption(f"OCR fallback produced {ocr_count} passage{'s' if ocr_count != 1 else ''}.")
                elif os.getenv("OCR_ENABLED", "0") != "1":
                    st.caption("Scanned pages: enable OCR_ENABLED=1 to add an OCR fallback.")
            except Exception as exc:
                st.error(f"Couldn't read this PDF: {exc}")
            finally:
                Path(path).unlink(missing_ok=True)
    st.divider()
    st.markdown("**About**")
    st.markdown("Hybrid BM25 + dense retrieval, RRF fusion, and cross-encoder reranking, with page-aware text and native tables.")
    st.caption(f"Embedding · {EMBEDDING_MODEL}\nReranker · {RERANKER_MODEL}")
    if "index" in st.session_state and st.button("Clear document", use_container_width=True):
        for key in ("index", "doc_name", "messages"):
            st.session_state.pop(key, None)
        st.rerun()

st.markdown(
    f"<div class='topbar'><div><div class='brand'>ResRAG</div><div class='brand-sub'>Ask questions about your document</div></div>"
    + (f"<div class='doc-pill'>{st.session_state.doc_name}</div>" if "doc_name" in st.session_state else "")
    + "</div>",
    unsafe_allow_html=True,
)

if "index" not in st.session_state:
    st.markdown(
        "<div class='empty'><h1>What would you like to know?</h1><p>Upload a PDF from the sidebar, then ask questions in a normal conversation. Answers stay tied to the document and include page citations.</p></div>",
        unsafe_allow_html=True,
    )
else:
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            if message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Message ResRAG…")
    if question:
        st.session_state.setdefault("messages", []).append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking…"):
                    sources = st.session_state.index.retrieve(question, dense_k=16, sparse_k=16, final_k=6)
                    answer = generate_answer(question, sources)
                st.markdown(answer, unsafe_allow_html=True)
                render_sources(sources)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as exc:
                message = f"I couldn't answer that yet: {exc}"
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
