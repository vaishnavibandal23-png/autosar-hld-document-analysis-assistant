import streamlit as st
from pathlib import Path
from rag_pipeline import RAGEngine
from config import DOCUMENTS_DIR, INDEX_DIR

st.set_page_config(page_title="AUTOSAR HLD Assistant", page_icon="🚗", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.answer-box {padding: 1rem; border-radius: 10px; background: #f5f8ff; border: 1px solid #dbe5f5;}
.source-box {padding: .7rem; border-left: 4px solid #2b78e4; background: #f8fafc; margin-bottom: .5rem;}
.small {color:#64748b; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_engine():
    return RAGEngine()

engine = get_engine()

st.title("🚗 AUTOSAR HLD Document Analysis Assistant")
st.caption("RAG-based architecture search with page/section citations")

with st.sidebar:
    st.header("Document Management")
    uploaded = st.file_uploader("Upload AUTOSAR HLD PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        saved = 0
        for f in uploaded:
            target = DOCUMENTS_DIR / Path(f.name).name
            target.write_bytes(f.getbuffer())
            saved += 1
        st.success(f"{saved} PDF(s) saved.")
        if st.button("Build / Rebuild Knowledge Base", type="primary"):
            with st.spinner("Extracting, chunking and indexing documents..."):
                result = engine.build_index()
            st.success(f"Indexed {result['chunks']} chunks from {result['documents']} document(s).")
            st.cache_resource.clear()
            st.rerun()

    if st.button("Rebuild Index"):
        with st.spinner("Rebuilding..."):
            result = engine.build_index()
        st.success(f"Indexed {result['chunks']} chunks.")
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("System")
    st.write("Vector store: FAISS")
    st.write("Embeddings: Sentence Transformers")
    st.write("LLM: Ollama (optional)")
    st.write(f"Documents: {len(list(DOCUMENTS_DIR.glob('*.pdf'))) if DOCUMENTS_DIR.exists() else 0}")

st.subheader("Ask Questions About Your AUTOSAR HLD")

if not INDEX_DIR.exists() or not (INDEX_DIR / "index.faiss").exists():
    st.info("Upload one or more HLD PDFs, then click **Build / Rebuild Knowledge Base**.")
else:
    question = st.text_input(
        "Question",
        placeholder="Example: Which software components are defined in the HLD?"
    )
    top_k = st.slider("Number of retrieved sections", 2, 8, 5)

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving evidence and generating answer..."):
            result = engine.ask(question.strip(), top_k=top_k)

        st.markdown("### Answer")
        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

        if result["warnings"]:
            st.warning(" | ".join(result["warnings"]))

        st.markdown("### Sources")
        for i, src in enumerate(result["sources"], 1):
            st.markdown(
                f"""<div class="source-box">
                <b>[{i}] {src["document"]}</b><br>
                Section: {src["section"] or "Not identified"}<br>
                Page: {src["page"]}<br>
                Similarity: {src["score"]:.3f}
                </div>""",
                unsafe_allow_html=True
            )

        with st.expander("Retrieved evidence"):
            for i, src in enumerate(result["sources"], 1):
                st.markdown(f"**[{i}] {src['document']} — Page {src['page']}**")
                st.write(src["text"])
                st.divider()

        st.caption(f"Latency: {result['latency']:.2f} seconds")
