"""
rag/llm.py — Grounded Local LLM Generation (Ollama / Qwen)
"""

import os
import re
import time
import ollama
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "100"))

# Optimized, concise system prompt for faster prefill
SYSTEM_PROMPT = """You are a private employee information assistant.
Answer ONLY using the supplied context.
If information is missing, state: "Information not found in the employee database."
Do not invent employee information. Be concise and cite Employee IDs (e.g. EMP001)."""


def strip_think_tags(text: str) -> str:
    """Removes <think>...</think> reasoning blocks produced by Qwen3 thinking models."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def check_ollama_status() -> dict:
    """Checks if Ollama daemon is running and lists available models."""
    try:
        models_response = ollama.list()
        model_names = []
        if isinstance(models_response, dict) and "models" in models_response:
            model_names = [m.get("model", m.get("name", "")) for m in models_response["models"]]
        elif hasattr(models_response, "models"):
            model_names = [
                m.model if hasattr(m, "model") else getattr(m, "name", "")
                for m in models_response.models
            ]
        model_names = [m for m in model_names if m]

        model_installed = any(OLLAMA_MODEL in m for m in model_names)
        return {
            "status": "ok",
            "available_models": model_names,
            "target_model": OLLAMA_MODEL,
            "target_model_installed": model_installed
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Ollama service is not running or unreachable at {OLLAMA_BASE_URL}.\n\nPlease start Ollama and run: `ollama pull {OLLAMA_MODEL}`"
        }


def generate_answer(question: str, context: str, user_info: dict = None) -> dict:
    """
    Generates a grounded RAG response using Ollama and Qwen LLM.
    Returns dict with answer and exact llm_generation_time timing.
    """
    t_start = time.perf_counter()

    ollama_check = check_ollama_status()
    if ollama_check["status"] == "error":
        t_elapsed = time.perf_counter() - t_start
        return {
            "status": "error",
            "answer": f"⚠️ **Ollama Offline Error**\n\n{ollama_check['message']}",
            "llm_generation_time": t_elapsed
        }

    target_model = OLLAMA_MODEL
    if not ollama_check["target_model_installed"] and ollama_check["available_models"]:
        installed_models = ollama_check["available_models"]
        qwen_fallbacks = [m for m in installed_models if "qwen" in m.lower()]
        target_model = qwen_fallbacks[0] if qwen_fallbacks else installed_models[0]

    user_name = user_info.get("name", "User") if user_info else "User"
    user_role = user_info.get("role", "EMPLOYEE") if user_info else "EMPLOYEE"

    prompt_content = f"""User: {user_name} ({user_role})

Question: {question}

Context:
{context if context.strip() else '[No records found]'}"""

    try:
        response = ollama.chat(
            model=target_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_content}
            ],
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": OLLAMA_NUM_PREDICT,
                "think": False
            }
        )

        t_elapsed = time.perf_counter() - t_start
        raw_answer = response['message']['content']
        answer_text = strip_think_tags(raw_answer)

        return {
            "status": "success",
            "answer": answer_text,
            "model_used": target_model,
            "llm_generation_time": t_elapsed
        }

    except Exception as e:
        t_elapsed = time.perf_counter() - t_start
        error_msg = str(e)
        if "not found" in error_msg.lower():
            return {
                "status": "error",
                "answer": f"⚠️ **Model Not Found**: `{OLLAMA_MODEL}` is not pulled in Ollama.\n\nRun: `ollama pull {OLLAMA_MODEL}`",
                "llm_generation_time": t_elapsed
            }
        return {
            "status": "error",
            "answer": f"⚠️ Error generating AI response: {error_msg}",
            "llm_generation_time": t_elapsed
        }
