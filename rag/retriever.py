"""
rag/retriever.py — Permission-Aware ChromaDB Vector Retrieval Module
"""

import os
import time
import functools
import pandas as pd
from dotenv import load_dotenv
from rag.ingest import (
    get_chroma_client,
    COLLECTION_NAME,
    LocalOllamaEmbeddingFunction,
    is_kb_initialized,
    ingest_employee_data
)
from auth.auth import is_salary_query

load_dotenv()

CSV_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "employees.csv"))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# Cached ChromaDB handles (module-level singletons)
_CACHED_CLIENT = None
_CACHED_COLLECTION = None


def get_cached_collection():
    """Returns persistent cached ChromaDB collection instance to eliminate re-initialization overhead."""
    global _CACHED_CLIENT, _CACHED_COLLECTION
    if _CACHED_COLLECTION is None:
        _CACHED_CLIENT = get_chroma_client()
        embedding_fn = LocalOllamaEmbeddingFunction()
        _CACHED_COLLECTION = _CACHED_CLIENT.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )
    return _CACHED_COLLECTION


def clear_chroma_cache():
    """Resets cached collection handles (called after re-indexing KB)."""
    global _CACHED_CLIENT, _CACHED_COLLECTION
    _CACHED_CLIENT = None
    _CACHED_COLLECTION = None
    load_employee_directory.cache_clear()


@functools.lru_cache(maxsize=1)
def load_employee_directory() -> pd.DataFrame:
    """Loads employee directory into memory once to resolve names and IDs for permission checks."""
    if os.path.exists(CSV_PATH):
        try:
            return pd.read_csv(CSV_PATH)
        except Exception:
            pass
    return pd.DataFrame()


def retrieve_employee_context(question: str, current_user: dict, top_k: int = None) -> dict:
    """
    Retrieves top_k employee records from ChromaDB subject to Role-Based Access Control (RBAC).

    Returns dict containing:
      - 'allowed': bool
      - 'denied_reason': str | None
      - 'documents': list[str]
      - 'metadatas': list[dict]
      - 'sources': list[str]
      - 'retrieval_time': float
      - 'embedding_time': float
    """
    t_retrieval_start = time.perf_counter()
    if top_k is None:
        top_k = DEFAULT_TOP_K

    q_lower = question.lower().strip()
    emp_df = load_employee_directory()

    # --- PERMISSION SECURITY FIREWALL ---
    user_role = str(current_user.get("role", "EMPLOYEE")).strip().upper()
    user_emp_id = str(current_user.get("employee_id", "")).strip().upper()

    if user_role == "EMPLOYEE":
        is_salary = is_salary_query(question)

        # Check if the query references OTHER employees by name or ID
        other_target_found = None
        if not emp_df.empty:
            for _, row in emp_df.iterrows():
                e_id = str(row["employee_id"]).strip().upper()
                full_name = str(row["name"]).strip().lower()
                first_name = full_name.split()[0]

                # Skip self
                if e_id == user_emp_id:
                    continue

                # Check if other employee's ID or full/first name appears in the question
                if e_id in question.upper() or full_name in q_lower or (len(first_name) >= 3 and f" {first_name} " in f" {q_lower} "):
                    other_target_found = {
                        "employee_id": e_id,
                        "name": row["name"]
                    }
                    break

        # Block unauthorized cross-employee salary inquiries
        if is_salary and other_target_found:
            t_elapsed = time.perf_counter() - t_retrieval_start
            return {
                "allowed": False,
                "denied_reason": f"ACCESS DENIED\n\nYou do not have permission to access another employee's ({other_target_found['name']}) salary information.",
                "documents": [],
                "metadatas": [],
                "sources": [],
                "retrieval_time": t_elapsed,
                "embedding_time": 0.0
            }

    if not is_kb_initialized():
        # Attempt auto-initialization on first query
        try:
            ingest_res = ingest_employee_data()
            if ingest_res.get("status") != "success" or not is_kb_initialized():
                t_elapsed = time.perf_counter() - t_retrieval_start
                return {
                    "allowed": True,
                    "kb_uninitialized": True,
                    "denied_reason": "Knowledge Base is not initialized. Please initialize it using the button in the UI.",
                    "documents": [],
                    "metadatas": [],
                    "sources": [],
                    "retrieval_time": t_elapsed,
                    "embedding_time": 0.0
                }
        except Exception as e:
            t_elapsed = time.perf_counter() - t_retrieval_start
            return {
                "allowed": True,
                "kb_uninitialized": True,
                "denied_reason": f"Knowledge Base initialization error: {e}",
                "documents": [],
                "metadatas": [],
                "sources": [],
                "retrieval_time": t_elapsed,
                "embedding_time": 0.0
            }

    # --- CHROMADB RETRIEVAL ---
    t_search_start = time.perf_counter()
    try:
        collection = get_cached_collection()
        results = collection.query(
            query_texts=[question],
            n_results=top_k
        )
    except Exception as e:
        # Fallback if cached handle fails
        try:
            clear_chroma_cache()
            collection = get_cached_collection()
            results = collection.query(
                query_texts=[question],
                n_results=top_k
            )
        except Exception as inner_e:
            t_elapsed = time.perf_counter() - t_retrieval_start
            return {
                "allowed": False,
                "denied_reason": f"Vector search error: {inner_e}",
                "documents": [],
                "metadatas": [],
                "sources": [],
                "retrieval_time": t_elapsed,
                "embedding_time": 0.0
            }

    t_search_elapsed = time.perf_counter() - t_search_start

    raw_docs = results.get("documents", [[]])[0]
    raw_metas = results.get("metadatas", [[]])[0]

    filtered_docs = []
    filtered_metas = []
    sources = []

    for doc, meta in zip(raw_docs, raw_metas):
        meta_emp_id = meta.get("employee_id", "").strip().upper()
        meta_name = meta.get("name", "")

        # Post-retrieval enforcement for EMPLOYEE role:
        if user_role == "EMPLOYEE":
            # If the retrieved document belongs to someone else and contains salary info, skip or redact salary lines
            if meta_emp_id != user_emp_id:
                if is_salary_query(question):
                    # Skip unauthorized salary docs
                    continue
                else:
                    # Sanitize salary lines from doc text for non-salary queries about colleagues
                    sanitized_lines = [
                        line for line in doc.split("\n")
                        if not any(k in line.lower() for k in ["basic salary", "bonus", "total salary"])
                    ]
                    doc = "\n".join(sanitized_lines)

        filtered_docs.append(doc)
        filtered_metas.append(meta)
        sources.append(f"{meta_emp_id} — {meta_name}")

    t_total_retrieval = time.perf_counter() - t_retrieval_start

    if not filtered_docs:
        return {
            "allowed": True,
            "denied_reason": None,
            "documents": [],
            "metadatas": [],
            "sources": [],
            "not_found": True,
            "retrieval_time": t_total_retrieval,
            "embedding_time": t_search_elapsed
        }

    return {
        "allowed": True,
        "denied_reason": None,
        "documents": filtered_docs,
        "metadatas": filtered_metas,
        "sources": sources,
        "not_found": False,
        "retrieval_time": t_total_retrieval,
        "embedding_time": t_search_elapsed
    }
