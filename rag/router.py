"""
rag/router.py — Lightweight Zero-LLM Query Router

Classifies user questions into:
  1. SIMPLE_LOOKUP: Direct structured database answer (No LLM, No Vector Search, Latency <0.01s)
  2. RAG_QUERY:     Complex/narrative search requiring ChromaDB + Qwen LLM
  3. UNSUPPORTED:   Out-of-scope non-employee queries rejected instantly

Security Rules:
  - Enforces identical RBAC permissions as RAG.
  - Blocks unauthorized cross-employee salary inquiries before lookup.
"""

import time
import re
import pandas as pd
from auth.auth import is_salary_query
from rag.retriever import load_employee_directory


def route_query(question: str, current_user: dict) -> dict:
    """
    Routes user question using fast regex & keyword pattern matching.

    Returns dict:
    {
        "query_type": "SIMPLE_LOOKUP" | "RAG_QUERY" | "UNSUPPORTED",
        "allowed": bool,
        "denied_reason": str | None,
        "answer": str | None,
        "documents": list[str],
        "metadatas": list[dict],
        "sources": list[str],
        "classification_time": float,
        "llm_bypassed": bool
    }
    """
    t_start = time.perf_counter()
    q_clean = question.strip()
    q_lower = q_clean.lower()

    user_role = str(current_user.get("role", "EMPLOYEE")).strip().upper()
    user_emp_id = str(current_user.get("employee_id", "")).strip().upper()

    emp_df = load_employee_directory()

    # ── 1. UNSUPPORTED QUERY DETECTION ─────────────────────────────────────────
    # Non-employee topics that shouldn't consume LLM compute
    unsupported_triggers = [
        "capital of", "write a poem", "quantum physics", "tell me a joke",
        "python code", "translate ", "weather in", "meaning of life",
        "recipe for", "who is the president", "solve equation", "convert units",
        "general knowledge"
    ]
    
    # Check if query matches unsupported keywords and contains no employee terms
    is_employee_related = any(k in q_lower for k in [
        "employee", "emp", "salary", "department", "designation", "bonus",
        "joining", "role", "work", "paid", "earning", "staff", "team", "who is"
    ]) or (not emp_df.empty and any(str(row["employee_id"]).lower() in q_lower for _, row in emp_df.iterrows()))

    if any(trig in q_lower for trig in unsupported_triggers) and not is_employee_related:
        t_elapsed = time.perf_counter() - t_start
        return {
            "query_type": "UNSUPPORTED",
            "allowed": True,
            "denied_reason": None,
            "answer": "This question is outside the Employee Intelligence knowledge base. I can only answer queries regarding internal employee records.",
            "documents": [],
            "metadatas": [],
            "sources": [],
            "classification_time": t_elapsed,
            "llm_bypassed": True
        }

    # ── 2. COMPLEX RAG QUERY DETECTION ─────────────────────────────────────────
    # Summaries, comparisons, or multi-employee department queries require RAG + Qwen narrative
    rag_triggers = [
        "summarize", "summary", "give me a report", "compare", "overview of all",
        "list all employees", "describe the department", "breakdown of"
    ]
    if any(trig in q_lower for trig in rag_triggers):
        t_elapsed = time.perf_counter() - t_start
        return {
            "query_type": "RAG_QUERY",
            "allowed": True,
            "denied_reason": None,
            "answer": None,
            "documents": [],
            "metadatas": [],
            "sources": [],
            "classification_time": t_elapsed,
            "llm_bypassed": False
        }

    # ── 3. SIMPLE LOOKUP DETECTION ─────────────────────────────────────────────
    # Identify target employee (Self or specific named/ID employee)
    target_emp = None
    is_self = False

    # Check for self references
    self_keywords = [
        "my salary", "my department", "my designation", "when did i join",
        "my employee id", "my emp id", "my details", "my bonus",
        "my total salary", "my basic salary", "my package", "what is my",
        "where do i work", "department do i work", "what department",
        "which department", "what is my role", "my role", "my position"
    ]

    if any(k in q_lower for k in self_keywords):
        is_self = True
        if not emp_df.empty:
            matched = emp_df[emp_df["employee_id"].str.strip().str.upper() == user_emp_id]
            if not matched.empty:
                target_emp = matched.iloc[0].to_dict()

    # If not self, search for referenced employee by ID or Name
    if target_emp is None and not emp_df.empty:
        for _, row in emp_df.iterrows():
            e_id = str(row["employee_id"]).strip().upper()
            f_name = str(row["name"]).strip().lower()
            first_n = f_name.split()[0]

            if e_id in q_clean.upper() or f_name in q_lower or (len(first_n) >= 3 and f" {first_n} " in f" {q_lower} "):
                target_emp = row.to_dict()
                if e_id == user_emp_id:
                    is_self = True
                break

    # If target employee resolved, check if it's a simple factual attribute lookup
    if target_emp:
        t_emp_id = str(target_emp["employee_id"]).strip().upper()
        t_name = str(target_emp["name"]).strip()

        # RBAC Check for simple lookups
        is_salary_q = is_salary_query(q_clean)
        if user_role == "EMPLOYEE" and not is_self and t_emp_id != user_emp_id and is_salary_q:
            t_elapsed = time.perf_counter() - t_start
            return {
                "query_type": "SIMPLE_LOOKUP",
                "allowed": False,
                "denied_reason": f"ACCESS DENIED\n\nYou do not have permission to access another employee's ({t_name}) salary information.",
                "answer": None,
                "documents": [],
                "metadatas": [],
                "sources": [],
                "classification_time": t_elapsed,
                "llm_bypassed": True
            }

        # Build clean employee metadata dictionary
        try:
            joining_yr = int(target_emp["joining_year"])
            basic_sal = int(target_emp["basic_salary"])
            bonus_sal = int(target_emp["bonus"])
            total_sal = int(target_emp["total_salary"])
        except (ValueError, TypeError):
            joining_yr = target_emp.get("joining_year", 0)
            basic_sal = target_emp.get("basic_salary", 0)
            bonus_sal = target_emp.get("bonus", 0)
            total_sal = target_emp.get("total_salary", 0)

        meta = {
            "employee_id": t_emp_id,
            "name": t_name,
            "department": str(target_emp["department"]).strip(),
            "designation": str(target_emp["designation"]).strip(),
            "joining_year": joining_yr,
            "basic_salary": basic_sal,
            "bonus": bonus_sal,
            "total_salary": total_sal
        }

        doc_text = (
            f"Employee ID: {meta['employee_id']}\n"
            f"Name: {meta['name']}\n"
            f"Department: {meta['department']}\n"
            f"Designation: {meta['designation']}\n"
            f"Joining Year: {meta['joining_year']}\n"
            f"Basic Salary: ₹{meta['basic_salary']:,}\n"
            f"Bonus: ₹{meta['bonus']:,}\n"
            f"Total Salary: ₹{meta['total_salary']:,}"
        )
        sources = [f"{meta['employee_id']} — {meta['name']} (Local Employee Database)"]

        # Deterministic Answer Construction
        answer = None
        if any(k in q_lower for k in ["total salary", "salary", "package", "compensation", "basic salary", "bonus"]):
            if is_self:
                answer = f"Your total salary package is ₹{meta['total_salary']:,} (Basic Salary: ₹{meta['basic_salary']:,}, Annual Bonus: ₹{meta['bonus']:,})."
            else:
                answer = f"{meta['name']} ({meta['employee_id']}) has a total salary package of ₹{meta['total_salary']:,} (Basic Salary: ₹{meta['basic_salary']:,}, Annual Bonus: ₹{meta['bonus']:,})."

        elif any(k in q_lower for k in ["department", "dept", "team"]):
            if is_self:
                answer = f"You work in the {meta['department']} department as a {meta['designation']}."
            else:
                answer = f"{meta['name']} ({meta['employee_id']}) works in the {meta['department']} department as a {meta['designation']}."

        elif any(k in q_lower for k in ["designation", "role", "title", "position"]):
            if is_self:
                answer = f"Your designation is {meta['designation']} in the {meta['department']} department."
            else:
                answer = f"{meta['name']} ({meta['employee_id']}) holds the designation of {meta['designation']} in the {meta['department']} department."

        elif any(k in q_lower for k in ["join", "joined", "hired", "start year"]):
            if is_self:
                answer = f"You joined the company in {meta['joining_year']}."
            else:
                answer = f"{meta['name']} ({meta['employee_id']}) joined the company in {meta['joining_year']}."

        elif any(k in q_lower for k in ["who is", "employee id", "emp id", "details"]):
            if is_self or user_role == "ADMIN":
                answer = (
                    f"{meta['name']} (ID: {meta['employee_id']}) is a {meta['designation']} in the {meta['department']} department. "
                    f"Joined in {meta['joining_year']} with a total salary of ₹{meta['total_salary']:,}."
                )
            else:
                answer = (
                    f"{meta['name']} (ID: {meta['employee_id']}) is a {meta['designation']} in the {meta['department']} department who joined in {meta['joining_year']}. "
                    f"(Salary details are confidential)."
                )

        if answer:
            t_elapsed = time.perf_counter() - t_start
            return {
                "query_type": "SIMPLE_LOOKUP",
                "allowed": True,
                "denied_reason": None,
                "answer": answer,
                "documents": [doc_text],
                "metadatas": [meta],
                "sources": sources,
                "classification_time": t_elapsed,
                "llm_bypassed": True
            }

    # Default fallback: Treat as RAG Query if pattern not explicitly matched as simple
    t_elapsed = time.perf_counter() - t_start
    return {
        "query_type": "RAG_QUERY",
        "allowed": True,
        "denied_reason": None,
        "answer": None,
        "documents": [],
        "metadatas": [],
        "sources": [],
        "classification_time": t_elapsed,
        "llm_bypassed": False
    }
