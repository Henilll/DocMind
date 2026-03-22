from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os, tempfile, shutil

load_dotenv()

# Always resolve the folder where THIS script lives
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

# ── Global RAG state ──
vectorstore = None
retriever   = None
pdf_name    = None

embedding_model = MistralAIEmbeddings(model="mistral-embed")
llm = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant.\n"
     "Use ONLY the provided context to answer the question.\n"
     'If the answer is not present in the context, say: "I could not find the answer in the document."'),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])


# ── Routes ──

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():
    global vectorstore, retriever, pdf_name

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)
    file.save(tmp_path)

    try:
        # Load & split
        docs   = PyPDFLoader(tmp_path).load()
        chunks = RecursiveCharacterTextSplitter(
                     chunk_size=1000, chunk_overlap=200
                 ).split_documents(docs)

        # Build in-memory Chroma store
        chroma_dir  = os.path.join(tmp_dir, "chroma")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=chroma_dir
        )
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
        )
        pdf_name = file.filename
        return jsonify({"message": f"'{file.filename}' ready — {len(chunks)} chunks indexed."})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route("/chat", methods=["POST"])
def chat():
    if retriever is None:
        return jsonify({"error": "Please upload a PDF first."}), 400

    body  = request.get_json(silent=True) or {}
    query = body.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        docs    = retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs)
        fp      = prompt.invoke({"context": context, "question": query})
        resp    = llm.invoke(fp)
        return jsonify({"answer": resp.content})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def status():
    return jsonify({"loaded": retriever is not None, "pdf": pdf_name})


if __name__ == "__main__":
    print(f"\n  DocMind running  →  http://localhost:5000\n")
    # use_reloader=False avoids double-init of heavy models
    port = int(os.environ.get("PORT", 5010))
    app.run(host="0.0.0.0", port=port)
