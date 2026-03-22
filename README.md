# DocMind 🧠

> Chat with any PDF using AI — powered by MistralAI, LangChain, and ChromaDB.

![Python](https://img.shields.io/badge/Python-3.10+-f0c040?style=flat-square&logo=python&logoColor=black)
![Flask](https://img.shields.io/badge/Flask-3.x-white?style=flat-square&logo=flask&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square&logo=langchain)
![MistralAI](https://img.shields.io/badge/MistralAI-mistral--small-FF7000?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--store-9B59B6?style=flat-square)

---

## What is DocMind?

DocMind is a **Retrieval-Augmented Generation (RAG)** application that lets you upload any PDF and have a natural conversation with it. Ask questions, extract insights, or summarize content — the AI answers using only what's in your document.

No hallucinations. No guessing. Just your document, intelligently searched.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        DocMind Flow                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   📄 PDF Upload                                                 │
│        │                                                        │
│        ▼                                                        │
│   PyPDFLoader → loads all pages as documents                    │
│        │                                                        │
│        ▼                                                        │
│   RecursiveCharacterTextSplitter                                │
│   (chunk_size=1000, overlap=200)                                │
│        │                                                        │
│        ▼                                                        │
│   MistralAI Embeddings (mistral-embed)                          │
│   → converts each chunk into a vector                           │
│        │                                                        │
│        ▼                                                        │
│   ChromaDB → stores all vectors locally                         │
│                                                                 │
│   ─────────────────────────────────────────                     │
│                                                                 │
│   💬 User Question                                              │
│        │                                                        │
│        ▼                                                        │
│   MMR Retriever (k=4, fetch_k=10, lambda=0.5)                  │
│   → finds the 4 most relevant & diverse chunks                  │
│        │                                                        │
│        ▼                                                        │
│   ChatPromptTemplate                                            │
│   → injects context + question into prompt                      │
│        │                                                        │
│        ▼                                                        │
│   MistralAI LLM (mistral-small-2506)                            │
│   → generates answer grounded in context                        │
│        │                                                        │
│        ▼                                                        │
│   ✅ Answer displayed in chat UI                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | HTML, CSS, Vanilla JS | Chat UI with PDF upload |
| **Backend** | Flask (Python) | REST API server |
| **LLM** | MistralAI `mistral-small-2506` | Answer generation |
| **Embeddings** | MistralAI `mistral-embed` | Text → vector conversion |
| **Vector Store** | ChromaDB | Storing & searching embeddings |
| **RAG Framework** | LangChain | Orchestrating the entire pipeline |
| **PDF Loader** | PyPDFLoader | Parsing PDF pages |
| **Text Splitter** | RecursiveCharacterTextSplitter | Chunking documents |
| **Retrieval** | MMR (Maximal Marginal Relevance) | Diverse, relevant chunk retrieval |

---

## Project Structure

```
DocMind/
├── app.py                  # Flask backend — API routes
├── index.html              # Frontend UI
├── create_database.py      # Standalone script to pre-index a PDF
├── main.py                 # CLI version of the RAG chat
├── requirements.txt        # Python dependencies
├── .env                    # API keys (never committed)
├── .gitignore
└── chroma-db/              # Local vector store (auto-generated)
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/docmind.git
cd docmind
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root folder:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

Get your free API key at [console.mistral.ai](https://console.mistral.ai)

### 5. Run the app

```bash
python app.py
```

Open your browser at **http://localhost:5010**

---

## Usage

1. Open the app in your browser
2. Click **"Click or drag a PDF here"** in the sidebar
3. Wait for the document to be indexed (you'll see a confirmation message)
4. Type your question in the chat input and press **Enter**
5. DocMind retrieves the most relevant chunks and answers using only your document

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend UI |
| `POST` | `/upload` | Upload and index a PDF file |
| `POST` | `/chat` | Send a question, get an answer |
| `GET` | `/status` | Check if a document is loaded |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `MISTRAL_API_KEY` | ✅ Yes | Your MistralAI API key |
| `PORT` | Auto (Railway) | Port override for deployment |

---

## Live Demo

🚀 https://docmind-etyl.onrender.com/

---

## Author

Built by **Henil Bhavsar**

---

## License

MIT License — free to use, modify, and distribute.
