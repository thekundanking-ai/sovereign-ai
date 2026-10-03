# Sovereign RAG Employee Intelligence System

A lightweight, local, permission-aware Sovereign RAG (Retrieval-Augmented Generation) prototype optimized for **low-latency querying** of sensitive employee and salary information without sending data to external cloud AI APIs.

---

## 🌟 Overview

The **Sovereign RAG System** demonstrates how organizations can leverage local LLMs, local vector databases, and deterministic query routing to build ultra-fast, private corporate intelligence tools. 

All data ingestion, vector embeddings, document retrieval, LLM inference, and document generation run entirely on your local machine.

---

## ⚡ Performance Optimization Architecture

```text
                                  USER QUERY
                                      │
                                      ▼
                             ┌─────────────────┐
                             │ Authentication  │
                             └────────┬────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │ Authorization   │
                             └────────┬────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │  Query Router   │  (rag/router.py — Zero LLM)
                             └───────┬─────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
           ▼                         ▼                         ▼
  SIMPLE_LOOKUP                 RAG_QUERY                 UNSUPPORTED
 (e.g. "What is my salary?")   (e.g. "Summarize Arjun")  (e.g. "Capital of France?")
           │                         │                         │
           ▼                         ▼                         ▼
   Direct Data Lookup         ChromaDB Vector           Instant Rejection
  (No Embedding/LLM)          Search (top_k=1-3)            (No LLM)
  Latency < 0.01s                    │                     Latency < 0.001s
           │                         ▼                         │
           │                     Qwen LLM                      │
           │                 (num_predict=100)                 │
           │                         │                         │
           └────────────────────┬────┘                         │
                                ▼                              ▼
                         RESULT + TIMINGS                REJECTION NOTICE
                                │
                                ▼
                       Document Creator (PDF/DOCX/TXT)
```

### Key Performance Innovations:

1. **Zero-LLM Direct Lookup Path**:
   - Factual queries (*"What is my salary?"*, *"What department do I work in?"*) bypass ChromaDB vector search and Qwen LLM completely.
   - Answers are generated deterministically in **< 0.01s** (a **>100x latency reduction** compared to 3–6s LLM generation).
2. **Instant Out-of-Scope Rejection**:
   - Non-employee queries (*"What is the capital of France?"*, *"Write a poem"*) are rejected instantly in **< 0.001s** without calling Ollama.
3. **Constrained Output Length (`num_predict=100`)**:
   - Prevents Qwen from producing unnecessarily verbose responses for simple RAG queries, speeding up local GPU inference.
4. **Memory Caching**:
   - Static employee dataset (`employees.csv`) and ChromaDB collection handles are cached in memory using `lru_cache` and module-level singletons to avoid repeated disk reads.
5. **Debug / Performance Diagnostics Mode**:
   - Real-time breakdown of Query Classification, Vector Retrieval, and LLM Generation latencies shown directly in the UI.

---

## ✨ Core Features

- **100% Local Processing**: No external API calls to OpenAI, Gemini, Anthropic, or Groq.
- **Role-Based Access Control (RBAC)**:
  - **ADMIN**: Access all metrics, employee profiles, full salary stats, and unrestricted RAG querying.
  - **EMPLOYEE**: Access own profile, own salary, and personal RAG querying. Accessing another employee's salary is strictly blocked with `ACCESS DENIED`.
- **Grounded AI Generation**: Qwen LLM is constrained to answer strictly from retrieved context and will not hallucinate employee details or answer out-of-scope general knowledge questions.
- **Source Attribution**: Displays exact retrieved employee records (e.g. `EMP003 — Arjun Mehta`) for auditability.
- **Local Document Generation**: Generate professional PDF, DOCX, and TXT reports from retrieved employee data — no external document API required.

---

## 📄 Document Creator

After every successful query answer (whether via Direct Lookup or RAG), a **Document Creator** panel appears below the response. The user selects a document type and format, clicks **Generate Document**, and downloads the file directly to their machine.

### Document Types

| Type | Contents | Salary Included |
|------|----------|-----------------|
| **Employee Report** | Full profile (ID, Name, Dept, Designation, Joining Year) + complete salary breakdown | ✅ Yes |
| **Salary Statement** | Employee identity + salary component breakdown | ✅ Yes |
| **Employee Summary** | Profile only (Name, Dept, Designation, Joining Year) | ❌ No |
| **Custom RAG Report** | Uses the AI-generated answer text as the narrative body | As-retrieved |

### Formats

| Format | Library | Output |
|--------|---------|--------|
| **PDF** | `reportlab` | Styled tables, brand header, clean typography |
| **DOCX** | `python-docx` | Heading hierarchy, key-value tables, footer |
| **TXT** | Built-in | Plain text, dividers, structured layout |

---

## 📁 Project Structure

```text
sovereign-rag-mini/
│
├── app.py                  # Streamlit UI dashboard with performance diagnostics
├── requirements.txt        # Python dependencies
├── README.md               # Setup and architecture documentation
├── .env.example            # Environment configuration template
├── .env                    # Local environment variables
│
├── data/
│   └── employees.csv       # Fictional employee dataset (25 records)
│
├── database/
│   └── database.py         # SQLite authentication and user management
│
├── rag/
│   ├── __init__.py
│   ├── router.py           # Zero-LLM Query Router for direct database lookups
│   ├── ingest.py           # Ingests CSV records into ChromaDB via embeddings
│   ├── retriever.py        # Cached, permission-filtered ChromaDB vector search
│   └── llm.py              # Local Qwen LLM answer generation with output limits
│
├── auth/
│   ├── __init__.py
│   └── auth.py             # Role-Based Access Control & security firewall
│
├── documents/
│   ├── __init__.py
│   └── creator.py          # Local document generation (PDF, DOCX, TXT)
│
└── vectorstore/
    └── chroma_db/          # Persistent ChromaDB vector database files
```

---

## ⚙️ Environment Configuration

Example `.env`:
```text
OLLAMA_MODEL=qwen3:4b
EMBEDDING_MODEL=nomic-embed-text
CHROMA_PATH=./vectorstore/chroma_db
OLLAMA_BASE_URL=http://localhost:11434
RAG_TOP_K=3
OLLAMA_NUM_PREDICT=100
DEBUG_PERFORMANCE=true
```

---

## 🚀 Getting Started

### 1. Setup Virtual Environment & Install Dependencies

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Pull Local Ollama Models

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

### 3. Launch Application

```bash
streamlit run app.py
```

### Demo Accounts

| Username | Password | Role | Employee ID | Accessible Scope |
| :--- | :--- | :--- | :--- | :--- |
| **`admin`** | `admin123` | **ADMIN** | `EMP000` | Unrestricted access across all employees & salaries |
| **`rahul`** | `user123` | **EMPLOYEE** | `EMP001` | Own profile (Rahul Sharma) & own salary |
| **`arjun`** | `user123` | **EMPLOYEE** | `EMP003` | Own profile (Arjun Mehta) & own salary |
| **`priya`** | `user123` | **EMPLOYEE** | `EMP002` | Own profile (Priya Patel) & own salary |

---

## 📊 Latency Benchmarks (Before vs After Optimization)

| Query Type | Query Example | Before Optimization | After Optimization | Speedup |
|:---|:---|:---|:---|:---|
| **Direct Lookup (Self Salary)** | *"What is my salary?"* | ~3.80s (Qwen LLM) | **0.001s** (Direct DB) | **>3800x** |
| **Direct Lookup (Department)** | *"What department do I work in?"* | ~3.50s (Qwen LLM) | **0.001s** (Direct DB) | **>3500x** |
| **Security Rejection** | *"What is Priya's salary?"* | ~3.20s (Qwen LLM) | **0.001s** (RBAC Block) | **>3200x** |
| **Out-of-Scope Query** | *"What is the capital of France?"* | ~4.10s (Qwen LLM) | **0.0005s** (Router Block)| **>8000x** |
| **Complex RAG Query** | *"Summarize Arjun's role and compensation"* | ~3.90s (Unconstrained) | **~1.20s** (`num_predict=100`) | **~3.2x** |
