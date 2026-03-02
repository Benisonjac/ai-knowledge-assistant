# 🧠 AI Knowledge Assistant


> **Production-grade RAG application** with LangGraph Agentic Retrieval, Cross-Encoder Reranking, Hallucination Checking, and LangSmith Observability — powered entirely by **free, open-source models** via the HuggingFace Inference API (zero local GPU required).

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?logo=chainlink)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_RAG-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal?logo=fastapi)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple)

---

## ✨ Key Features & Skills Demonstrated

| Feature | Technology | Resume Skill |
|---|---|---|
| **Agentic RAG Pipeline** | LangGraph (state machine) | LLM Application Architecture |
| **Two-Stage Retrieval** | ChromaDB + FlashRank Reranking | Information Retrieval, Vector Search |
| **Free Cloud LLM Inference** | HuggingFace Inference API (Mistral 7B) | Open-Source LLM Integration |
| **Hallucination Detection** | LLM-as-judge self-correction | AI Safety & Reliability |
| **Conversational Memory** | Query rewriting with chat history | Conversational AI |
| **Document Processing** | PDF, DOCX, PPTX, TXT, Markdown | Unstructured Data Processing |
| **Remote Embeddings** | HuggingFace Inference API (MiniLM) | Embedding Models |
| **REST API** | FastAPI + Pydantic v2 | API Development |
| **Interactive UI** | Streamlit chat interface | Full-Stack AI Development |
| **Observability** | LangSmith tracing | MLOps, Monitoring |
| **RAG Evaluation** | RAGAS (Faithfulness, Relevancy, Precision, Recall) | AI/ML Evaluation |
| **Containerization** | Docker + Docker Compose | DevOps |
| **Type Safety** | Pydantic-settings, mypy strict | Software Engineering Best Practices |

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────┐
│  Streamlit   │────▶│   FastAPI Backend     │────▶│  ChromaDB    │
│  Chat UI     │◀────│                       │◀────│  Vector Store│
└──────────────┘     │  ┌────────────────┐   │     └──────────────┘
                     │  │  LangGraph     │   │
                     │  │  Agentic RAG   │   │     ┌──────────────┐
                     │  │                │   │────▶│  HuggingFace │
                     │  │  1. Rewrite    │   │◀────│  Inference   │
                     │  │  2. Retrieve   │   │     │  API (free)  │
                     │  │  3. Rerank     │   │     └──────────────┘
                     │  │  3. Rerank     │   │
                     │  │  4. Generate   │   │     ┌──────────────┐
                     │  │  5. Validate   │   │────▶│  LangSmith   │
                     │  └────────────────┘   │     │  (Tracing)   │
                     └──────────────────────┘     └──────────────┘
```

### LangGraph RAG Flow

```
Start → Rewrite Query → Retrieve Documents → Has Docs?
                                                  │
                              ┌────────────────── yes ──────────────────┐
                              ▼                                         ▼
                      Generate Answer                           Handle No Docs
                              │                                         │
                              ▼                                         ▼
                    Check Hallucination                               END
                              │
                     ┌──── grounded? ────┐
                     ▼                   ▼
                    END            Retry (rewrite)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Free [HuggingFace account & API token](https://huggingface.co/settings/tokens)

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/ai-knowledge-assistant.git
cd ai-knowledge-assistant

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your free HUGGINGFACEHUB_API_TOKEN
```

### 3. Add Documents

Place your PDF, DOCX, TXT, MD, or PPTX files in `data/documents/`.

### 4. Run the Application

```bash
# Terminal 1: Start the API server
uvicorn src.api.app:app --reload --port 8000

# Terminal 2: Start the Streamlit UI
streamlit run src/ui/chat_app.py
```

Open **http://localhost:8501** in your browser.

### Docker (Alternative)

```bash
docker compose up --build
```

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

This project includes a standalone [streamlit_app.py](streamlit_app.py) that runs the full RAG pipeline
without needing a separate FastAPI server — perfect for Streamlit Cloud's single-process model.

### Steps

1. **Push your repo to GitHub** (public or private).

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. **Deploy a new app:**
   - **Repository:** select your repo
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`

4. **Add secrets** (in the Streamlit Cloud dashboard → **Settings → Secrets**):
   ```toml
   HUGGINGFACEHUB_API_TOKEN = "hf_your-token-here"

   # Optional – LangSmith observability
   # LANGCHAIN_API_KEY = "ls-your-key-here"
   ```

5. **Click Deploy!** Streamlit Cloud will install `requirements.txt` + `packages.txt` automatically.

> **Tip:** The sample document in `data/documents/` is included in the repo, so the app
> can ingest it on first run. Upload more files via the sidebar.

---

## 📁 Project Structure

```
ai-knowledge-assistant/
├── streamlit_app.py     # ⭐ Streamlit Cloud entry-point (standalone)
├── requirements.txt     # Dependencies for Streamlit Cloud
├── packages.txt         # System packages for Streamlit Cloud
├── src/
│   ├── config/          # Settings, logging (pydantic-settings)
│   ├── ingestion/       # Document loading, chunking, embedding, vector store
│   │   ├── document_loader.py   # Multi-format document loading
│   │   ├── chunker.py           # Recursive text splitting
│   │   ├── embeddings.py        # HuggingFace Inference API embeddings
│   │   ├── vector_store.py      # ChromaDB operations
│   │   └── pipeline.py          # End-to-end ingestion orchestration
│   ├── retrieval/       # RAG engine
│   │   ├── retriever.py         # Two-stage retrieval (vector + rerank)
│   │   ├── reranker.py          # FlashRank cross-encoder reranking
│   │   ├── prompts.py           # Prompt templates (RAG, rewrite, grading)
│   │   ├── llm.py               # LLM factory with retry logic
│   │   └── rag_chain.py         # ⭐ LangGraph agentic RAG workflow
│   ├── api/             # FastAPI REST API
│   │   ├── app.py               # Endpoints (query, ingest, upload, stats)
│   │   └── schemas.py           # Pydantic request/response models
│   ├── ui/              # Streamlit chat interface
│   │   └── chat_app.py          # Interactive chat with source display
│   └── evaluation/      # RAG quality evaluation
│       └── ragas_eval.py        # RAGAS metrics (faithfulness, relevancy, etc.)
├── tests/               # Pytest test suite
├── data/
│   └── documents/       # Place your documents here
├── pyproject.toml       # Modern Python packaging (local/Docker)
├── Dockerfile
├── docker-compose.yml
├── .streamlit/          # Streamlit config & secrets template
└── .env.example
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📊 Evaluation

```python
from src.evaluation import evaluate_rag_pipeline, EvalSample

samples = [
    EvalSample(
        question="What is RAG?",
        answer="RAG retrieves documents and uses them for LLM generation.",
        contexts=["RAG combines retrieval with generation..."],
        ground_truth="RAG is retrieval augmented generation.",
    )
]

scores = await evaluate_rag_pipeline(samples)
# {'faithfulness': 0.95, 'answer_relevancy': 0.92, 'context_precision': 0.88, 'context_recall': 0.90}
```

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Ask a question (RAG) |
| `POST` | `/ingest/upload` | Upload & ingest a file |
| `POST` | `/ingest/directory` | Ingest a directory of files |
| `GET` | `/store/stats` | Vector store statistics |

Interactive docs: **http://localhost:8000/docs**

---

## 🎯 Resume Bullet Points

Use these when describing this project on your resume:

- **Built a production-grade RAG application** using LangChain, LangGraph, and ChromaDB with agentic retrieval, cross-encoder reranking, and self-corrective hallucination checking — powered entirely by free, open-source models via HuggingFace Inference API
- **Implemented a LangGraph state machine** for multi-step reasoning with query rewriting, two-stage retrieval (vector search + FlashRank reranking), and LLM-as-judge validation
- **Designed a full-stack AI system** with FastAPI REST API, Streamlit chat UI, HuggingFace Inference API for LLM & embeddings, and LangSmith observability tracing
- **Deployed a zero-cost AI stack** using HuggingFace free-tier Inference API (Mistral 7B), remote embeddings, and FlashRank reranking — no local GPU or paid API keys required

---

## 📝 License

MIT
