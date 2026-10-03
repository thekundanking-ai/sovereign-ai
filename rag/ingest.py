import os
import pandas as pd
import chromadb
import ollama
from dotenv import load_dotenv

load_dotenv()

import math
import re

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./vectorstore/chroma_db")
COLLECTION_NAME = "employee_kb"


def get_chroma_client():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)


def fallback_local_embedding(text: str, dim: int = 384) -> list[float]:
    """Generates a deterministic normalized term feature vector as ultimate fallback."""
    vec = [0.0] * dim
    words = re.findall(r'\w+', text.lower())
    for w in words:
        idx = abs(hash(w)) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def generate_embedding(text: str) -> list[float]:
    """Generates embedding vector for text using local Ollama embedding models with fallback."""
    models_to_try = [EMBEDDING_MODEL, "nomic-embed-text", "all-minilm", OLLAMA_MODEL]
    seen = set()
    models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

    for model in models_to_try:
        try:
            res = ollama.embed(model=model, input=text)
            if "embeddings" in res and res["embeddings"]:
                return res["embeddings"][0]
        except Exception:
            pass
        
        try:
            res = ollama.embeddings(model=model, prompt=text)
            if "embedding" in res and res["embedding"]:
                return res["embedding"]
        except Exception:
            pass

    # Deterministic local fallback vector
    return fallback_local_embedding(text)


from chromadb import EmbeddingFunction


class LocalOllamaEmbeddingFunction(EmbeddingFunction):
    """ChromaDB compatible embedding function using Ollama with batch & fallback support."""
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model_name = model_name

    def name(self) -> str:
        return f"ollama_{self.model_name.replace(':', '_')}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        for model_to_try in [self.model_name, "nomic-embed-text", "all-minilm"]:
            try:
                res = ollama.embed(model=model_to_try, input=input)
                if "embeddings" in res and res["embeddings"] and len(res["embeddings"]) == len(input):
                    return res["embeddings"]
            except Exception:
                pass

        embeddings = []
        for text in input:
            embeddings.append(generate_embedding(text))
        return embeddings


def format_employee_document(row: pd.Series) -> str:
    """Converts a pandas DataFrame row into a structured document text."""
    return (
        f"Employee ID: {row['employee_id']}\n"
        f"Name: {row['name']}\n"
        f"Department: {row['department']}\n"
        f"Designation: {row['designation']}\n"
        f"Joining Year: {row['joining_year']}\n"
        f"Basic Salary: ₹{row['basic_salary']:,}\n"
        f"Bonus: ₹{row['bonus']:,}\n"
        f"Total Salary: ₹{row['total_salary']:,}"
    )


def ingest_employee_data(csv_path: str = None) -> dict:
    """
    Loads employees.csv, generates embeddings locally via Ollama,
    and stores employee records in ChromaDB.
    """
    if csv_path is None:
        csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "employees.csv"))

    if not os.path.exists(csv_path):
        return {"status": "error", "message": f"CSV file not found at {csv_path}"}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to read CSV file: {e}"}

    client = get_chroma_client()
    embedding_fn = LocalOllamaEmbeddingFunction()

    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    documents = []
    metadatas = []
    ids = []

    for _, row in df.iterrows():
        emp_id = str(row["employee_id"]).strip()
        doc_text = format_employee_document(row)

        documents.append(doc_text)
        ids.append(emp_id)
        metadatas.append({
            "employee_id": emp_id,
            "name": str(row["name"]).strip(),
            "department": str(row["department"]).strip(),
            "designation": str(row["designation"]).strip(),
            "joining_year": int(row["joining_year"]),
            "basic_salary": int(row["basic_salary"]),
            "bonus": int(row["bonus"]),
            "total_salary": int(row["total_salary"])
        })

    # Ingest / Upsert into ChromaDB
    try:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        return {
            "status": "success",
            "count": len(documents),
            "collection": COLLECTION_NAME,
            "message": f"Successfully indexed {len(documents)} employee records into ChromaDB."
        }
    except Exception as e:
        return {"status": "error", "message": f"Error indexing data into ChromaDB: {e}"}


def is_kb_initialized() -> bool:
    """Checks if the ChromaDB vector database exists and contains employee records."""
    try:
        client = get_chroma_client()
        collections = [c.name for c in client.list_collections()]
        if COLLECTION_NAME not in collections:
            return False
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count() > 0
    except Exception:
        return False


if __name__ == "__main__":
    print("[Ingest] Ingesting employee data...")
    res = ingest_employee_data()
    print(res)
