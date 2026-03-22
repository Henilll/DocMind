import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="DocMind — PDF Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ─── Root Variables ─── */
:root {
    --bg:          #0D0F14;
    --surface:     #13161E;
    --surface2:    #1A1E2A;
    --border:      #252A38;
    --accent:      #6EE7B7;
    --accent2:     #818CF8;
    --accent3:     #F472B6;
    --text:        #E8EAF0;
    --muted:       #6B7280;
    --danger:      #F87171;
}

/* ─── Global Reset ─── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.main .block-container {
    padding: 2rem 2.5rem 4rem;
    max-width: 1100px;
}

/* ─── Hide Streamlit Branding ─── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0;
}

/* ─── Sidebar Logo / Header ─── */
.sidebar-brand {
    background: linear-gradient(135deg, #1e2235 0%, #13161e 100%);
    border-bottom: 1px solid var(--border);
    padding: 2rem 1.5rem 1.5rem;
    margin-bottom: 1.5rem;
}

.sidebar-brand h1 {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 0.2rem;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sidebar-brand p {
    font-size: 0.75rem;
    color: var(--muted);
    margin: 0;
    letter-spacing: 0.5px;
}

/* ─── Sidebar Section Labels ─── */
.sidebar-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0 1.5rem;
    margin-bottom: 0.5rem;
}

/* ─── File Uploader ─── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

[data-testid="stFileUploader"] label {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
}

/* ─── Buttons ─── */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.3px !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.2s ease !important;
    border: none !important;
    width: 100% !important;
}

/* Primary button */
.stButton:first-of-type > button,
.btn-primary > button {
    background: linear-gradient(135deg, var(--accent) 0%, #34d399 100%) !important;
    color: #0a0f0a !important;
    box-shadow: 0 4px 20px rgba(110, 231, 183, 0.2) !important;
}
.stButton:first-of-type > button:hover {
    box-shadow: 0 4px 28px rgba(110, 231, 183, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Secondary / danger button */
.btn-danger > button {
    background: transparent !important;
    color: var(--danger) !important;
    border: 1.5px solid rgba(248, 113, 113, 0.3) !important;
}
.btn-danger > button:hover {
    background: rgba(248, 113, 113, 0.08) !important;
    border-color: var(--danger) !important;
}

/* Ask button */
.ask-btn > button {
    background: linear-gradient(135deg, var(--accent2) 0%, #6366f1 100%) !important;
    color: white !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(129, 140, 248, 0.25) !important;
}
.ask-btn > button:hover {
    box-shadow: 0 4px 28px rgba(129, 140, 248, 0.45) !important;
    transform: translateY(-1px) !important;
}

/* ─── Text Input ─── */
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent2) !important;
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.12) !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--muted) !important;
}

/* ─── Page Title ─── */
.page-hero {
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}

.page-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0 0 0.5rem;
    line-height: 1.1;
}

.page-hero h1 span.g1 {
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.page-hero p {
    color: var(--muted);
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}

/* ─── Status Badge ─── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    margin-bottom: 1.5rem;
}
.status-ready {
    background: rgba(110, 231, 183, 0.1);
    border: 1px solid rgba(110, 231, 183, 0.3);
    color: var(--accent);
}
.status-idle {
    background: rgba(107, 114, 128, 0.1);
    border: 1px solid var(--border);
    color: var(--muted);
}
.dot { width: 7px; height: 7px; border-radius: 50%; }
.dot-green { background: var(--accent); box-shadow: 0 0 6px var(--accent); }
.dot-gray  { background: var(--muted); }

/* ─── Query Box ─── */
.query-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 2rem;
}

/* ─── Chat Bubbles ─── */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.chat-item {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    transition: border-color 0.2s;
}
.chat-item:hover { border-color: #2e3448; }

.chat-q {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    padding: 1.1rem 1.4rem;
    border-bottom: 1px solid var(--border);
    background: rgba(129, 140, 248, 0.04);
}

.chat-a {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    padding: 1.1rem 1.4rem;
}

.avatar {
    width: 30px;
    height: 30px;
    border-radius: 8px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
}

.avatar-user {
    background: linear-gradient(135deg, var(--accent2), #6366f1);
    color: white;
}

.avatar-ai {
    background: linear-gradient(135deg, var(--accent), #34d399);
    color: #0a0f0a;
}

.chat-text {
    font-size: 0.9rem;
    line-height: 1.65;
    padding-top: 0.25rem;
    flex: 1;
}

.chat-q .chat-text { color: var(--muted); font-weight: 400; }
.chat-a .chat-text { color: var(--text); }

/* ─── Expander ─── */
[data-testid="stExpander"] {
    background: transparent !important;
    border: none !important;
    border-top: 1px solid var(--border) !important;
    border-radius: 0 !important;
}

[data-testid="stExpander"] summary {
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    padding: 0.6rem 1.4rem !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.3px;
}

[data-testid="stExpander"] summary:hover { color: var(--text) !important; }

[data-testid="stExpander"] > div > div {
    background: transparent !important;
    padding: 0.5rem 1.4rem 1rem !important;
}

/* ─── Chunk Card ─── */
.chunk-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
    font-size: 0.82rem;
    line-height: 1.6;
    color: #9ca3af;
}

.chunk-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}

.chunk-tag {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 0.15rem 0.6rem;
    border-radius: 4px;
    background: rgba(110, 231, 183, 0.1);
    color: var(--accent);
    border: 1px solid rgba(110, 231, 183, 0.2);
}

/* ─── Stat Cards ─── */
.stat-row {
    display: flex;
    gap: 0.75rem;
    margin-top: 1rem;
}

.stat-card {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}

.stat-card .val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
}

.stat-card .lbl {
    font-size: 0.68rem;
    color: var(--muted);
    margin-top: 0.3rem;
    letter-spacing: 0.5px;
}

/* ─── Spinner / Alert Colors ─── */
.stSpinner > div { border-top-color: var(--accent) !important; }

.stAlert {
    border-radius: 10px !important;
    font-size: 0.85rem !important;
}

/* ─── Section Header ─── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* ─── Empty State ─── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--muted);
    border: 1.5px dashed var(--border);
    border-radius: 16px;
    background: var(--surface);
}

.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-state h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #3a3f54;
    margin-bottom: 0.5rem;
}
.empty-state p { font-size: 0.85rem; color: #2e3448; }

/* ─── Divider ─── */
hr { border-color: var(--border) !important; }

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3a3f54; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Constants
# ---------------------------
CHROMA_DIR = "chroma-db"
UPLOAD_DIR = "uploaded_files"
Path(UPLOAD_DIR).mkdir(exist_ok=True)
Path(CHROMA_DIR).mkdir(exist_ok=True)

# ---------------------------
# Session State
# ---------------------------
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {"pages": 0, "chunks": 0, "name": ""}

# ---------------------------
# Helper Functions
# ---------------------------
def clear_chroma_db():
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    Path(CHROMA_DIR).mkdir(exist_ok=True)

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def create_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embedding_model, persist_directory=CHROMA_DIR
    )
    return vectorstore, len(docs), len(chunks)

def load_vectorstore():
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)

def get_answer(query):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )
    llm = ChatMistralAI(model="mistral-small-2506")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant.\n\nUse ONLY the provided context to answer the question.\n\nIf the answer is not present in the context, say: "I could not find the answer in the document." """),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = prompt.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)
    return response.content, docs

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h1>DocMind</h1>
        <p>PDF Intelligence Engine</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload a PDF to begin"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col_build, _ = st.columns([1, 0.01])
    with col_build:
        build_btn = st.button("⚡ Build Knowledge Base", key="build")

    if build_btn:
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
        else:
            try:
                with st.spinner("Embedding document…"):
                    clear_chroma_db()
                    pdf_path = save_uploaded_file(uploaded_file)
                    _, total_pages, total_chunks = create_vectorstore(pdf_path)
                    st.session_state.vectorstore_ready = True
                    st.session_state.doc_stats = {
                        "pages": total_pages,
                        "chunks": total_chunks,
                        "name": uploaded_file.name,
                    }
                st.success("Knowledge base ready!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Stats
    if st.session_state.vectorstore_ready:
        stats = st.session_state.doc_stats
        st.markdown(f"""
        <div style="margin-top:1.25rem">
            <div style="font-size:0.72rem;color:var(--muted);margin-bottom:0.6rem;
                        font-family:'Syne',sans-serif;letter-spacing:1.5px;
                        text-transform:uppercase;font-weight:700;">
                Loaded Document
            </div>
            <div style="font-size:0.82rem;color:var(--text);background:var(--surface2);
                        border:1px solid var(--border);border-radius:10px;
                        padding:0.7rem 0.9rem;margin-bottom:0.75rem;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                📄 {stats["name"]}
            </div>
            <div class="stat-row">
                <div class="stat-card">
                    <div class="val">{stats["pages"]}</div>
                    <div class="lbl">Pages</div>
                </div>
                <div class="stat-card">
                    <div class="val">{stats["chunks"]}</div>
                    <div class="lbl">Chunks</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Session</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
        if st.button("🗑 Clear Chat History", key="clear"):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# MAIN PAGE
# ---------------------------
st.markdown("""
<div class="page-hero">
    <h1>Ask your <span class="g1">document</span> anything.</h1>
    <p>Semantic search · Context-aware answers · Source tracing</p>
</div>
""", unsafe_allow_html=True)

# Status badge
if st.session_state.vectorstore_ready:
    st.markdown("""
    <div class="status-badge status-ready">
        <span class="dot dot-green"></span> Knowledge base active
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-badge status-idle">
        <span class="dot dot-gray"></span> No document loaded — upload a PDF in the sidebar
    </div>
    """, unsafe_allow_html=True)

# Query area
st.markdown('<div class="query-box">', unsafe_allow_html=True)
query = st.text_input(
    "Question",
    placeholder="What is this document about? Summarize section 3. Who are the key authors?",
    label_visibility="collapsed",
    key="query_input",
)

col_ask, col_space = st.columns([1, 5])
with col_ask:
    st.markdown('<div class="ask-btn">', unsafe_allow_html=True)
    ask_button = st.button("Ask →", key="ask")
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Handle ask
if ask_button:
    if not st.session_state.vectorstore_ready and not os.path.exists(CHROMA_DIR):
        st.warning("Upload a PDF and build the knowledge base first.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching & synthesizing…"):
                answer, retrieved_docs = get_answer(query)
            st.session_state.chat_history.append({
                "question": query,
                "answer": answer,
                "sources": retrieved_docs,
            })
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ---------------------------
# Chat History
# ---------------------------
if st.session_state.chat_history:
    st.markdown('<div class="section-header">Conversation</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"""
        <div class="chat-item">
            <div class="chat-q">
                <div class="avatar avatar-user">You</div>
                <div class="chat-text">{chat['question']}</div>
            </div>
            <div class="chat-a">
                <div class="avatar avatar-ai">AI</div>
                <div class="chat-text">{chat['answer']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"📎 {len(chat['sources'])} source chunks retrieved"):
            for j, doc in enumerate(chat["sources"], 1):
                page_no = doc.metadata.get("page", "?")
                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-label">
                        <span class="chunk-tag">Chunk {j}</span>
                        <span style="font-size:0.72rem;color:var(--muted);">Page {page_no}</span>
                    </div>
                    {doc.page_content}
                </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🧠</div>
        <h3>No conversations yet</h3>
        <p>Upload a PDF, build your knowledge base, and start asking questions.</p>
    </div>
    """, unsafe_allow_html=True)