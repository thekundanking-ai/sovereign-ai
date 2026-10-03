import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure working directory imports work smoothly
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.database import init_db
from auth.auth import authenticate_user
from rag.ingest import ingest_employee_data, is_kb_initialized
from rag.retriever import retrieve_employee_context, load_employee_directory
from rag.router import route_query
from rag.llm import generate_answer, check_ollama_status
from documents.creator import (
    extract_employee_from_context,
    generate_document,
    DOC_TYPES,
    DOC_FORMATS,
)

load_dotenv()

# --- STREAMLIT CONFIGURATION ---
st.set_page_config(
    page_title="Sovereign AI — Private Employee Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark/light enterprise dashboard styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 20px;
    }
    .badge-container {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 25px;
    }
    .status-badge {
        background-color: rgba(40, 167, 69, 0.1);
        color: #28a745;
        border: 1px solid rgba(40, 167, 69, 0.3);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
    }
    .source-box {
        background-color: rgba(108, 117, 125, 0.08);
        border-left: 4px solid #0d6efd;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 15px;
        font-size: 0.9rem;
    }
    .denied-box {
        background-color: rgba(220, 53, 69, 0.1);
        border: 1px solid rgba(220, 53, 69, 0.3);
        color: #dc3545;
        padding: 15px;
        border-radius: 8px;
        font-weight: 600;
        margin-top: 15px;
    }
    .perf-box {
        background-color: rgba(255, 193, 7, 0.08);
        border: 1px solid rgba(255, 193, 7, 0.3);
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 16px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #d97706;
    }
    .doc-creator-box {
        background-color: rgba(13, 110, 253, 0.04);
        border: 1px solid rgba(13, 110, 253, 0.2);
        border-radius: 10px;
        padding: 20px 24px;
        margin-top: 24px;
    }
    .doc-creator-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0d6efd;
        margin-bottom: 4px;
    }
    .doc-creator-subtitle {
        font-size: 0.85rem;
        color: #6c757d;
        margin-bottom: 16px;
    }
    .footer-text {
        text-align: center;
        font-size: 0.8rem;
        color: #888;
        margin-top: 40px;
        border-top: 1px solid #333;
        padding-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize SQLite database schema
init_db()

# --- SESSION STATE INITIALIZATION ---
if "user" not in st.session_state:
    st.session_state.user = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Document Creator session state
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_context" not in st.session_state:
    st.session_state.last_context = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "last_metadatas" not in st.session_state:
    st.session_state.last_metadatas = []
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "doc_bytes" not in st.session_state:
    st.session_state.doc_bytes = None
if "doc_filename" not in st.session_state:
    st.session_state.doc_filename = None
if "doc_format" not in st.session_state:
    st.session_state.doc_format = None


# --- AUTHENTICATION LOGIN PAGE ---
def render_login():
    st.markdown("<div style='text-align: center; padding-top: 40px;'>", unsafe_allow_html=True)
    st.title("🛡️ Sovereign AI")
    st.markdown("### Private Employee Intelligence — Local RAG Prototype")
    st.markdown("<p style='color: #888;'>Designed for local/private processing without cloud APIs.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        st.markdown("---")
        with st.form("login_form"):
            st.subheader("Account Login")
            username_input = st.text_input("Username", placeholder="e.g. admin or rahul").strip()
            password_input = st.text_input("Password", type="password", placeholder="e.g. admin123 or user123").strip()
            submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                if not username_input or not password_input:
                    st.error("Please enter both username and password.")
                else:
                    authenticated_user = authenticate_user(username_input, password_input)
                    if authenticated_user:
                        st.session_state.user = authenticated_user
                        st.success(f"Welcome back, {authenticated_user['name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        st.info("""
        💡 **Demo Accounts:**
        - **Admin**: `admin` / `admin123` (Full access to all employees & salaries)
        - **Employee 1**: `rahul` / `user123` (Rahul Sharma - EMP001)
        - **Employee 2**: `arjun` / `user123` (Arjun Mehta - EMP003)
        - **Employee 3**: `priya` / `user123` (Priya Patel - EMP002)

        *Note: This is a prototype local authentication system.*
        """)


# If user is not logged in, render login page
if not st.session_state.user:
    render_login()
    st.stop()


# --- LOGGED IN USER INTERFACE ---
user = st.session_state.user
emp_df = load_employee_directory()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🛡️ SOVEREIGN AI")
    st.markdown("<span style='color: #28a745; font-weight: bold;'>● LOCAL / SECURE</span>", unsafe_allow_html=True)
    st.markdown("---")

    # User Profile Card
    st.markdown(f"**Logged User:** {user['name']}")
    role_color = "🔴 Admin" if user["role"] == "ADMIN" else "🔵 Employee"
    st.markdown(f"**Role:** {role_color}")
    if user["employee_id"] != "EMP000":
        st.markdown(f"**Employee ID:** `{user['employee_id']}`")
    
    st.markdown("---")

    # Navigation Menu
    navigation = st.radio(
        "Navigation",
        ["📊 Dashboard", "👥 Employees", "💰 Salary Overview", "🤖 AI Assistant", "⚙️ System & KB Status"],
        index=0
    )

    st.markdown("---")
    # Debug / Performance Mode Toggle
    show_perf_metrics = st.checkbox("⚡ Debug / Performance Mode", value=True)

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.chat_history = []
        # Clear Document Creator state on logout
        st.session_state.last_answer = None
        st.session_state.last_context = None
        st.session_state.last_sources = []
        st.session_state.last_metadatas = []
        st.session_state.last_query = None
        st.session_state.doc_bytes = None
        st.session_state.doc_filename = None
        st.session_state.doc_format = None
        st.rerun()


# --- HEADER & SOVEREIGNTY BADGES ---
st.markdown(f"<div class='main-header'>Sovereign RAG Intelligence System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Local Vector RAG & Grounded Qwen LLM for Sensitive HR Data</div>", unsafe_allow_html=True)

st.markdown("""
<div class='badge-container'>
    <span class='status-badge'>🟢 LOCAL AI</span>
    <span class='status-badge'>🟢 LOCAL DATA</span>
    <span class='status-badge'>🟢 LOCAL VECTOR DB</span>
    <span class='status-badge'>🟢 LOCAL DOCUMENT GENERATION</span>
    <span class='status-badge'>🟢 NO EXTERNAL AI API</span>
</div>
""", unsafe_allow_html=True)


# --- PAGE 1: DASHBOARD ---
if navigation == "📊 Dashboard":
    st.subheader("📊 Company Overview")

    m1, m2, m3, m4 = st.columns(4)
    total_emps = len(emp_df) if not emp_df.empty else 0
    depts_count = emp_df["department"].nunique() if not emp_df.empty else 0

    m1.metric("Total Employees", total_emps)
    m2.metric("Departments", depts_count)

    if user["role"] == "ADMIN" and not emp_df.empty:
        avg_sal = emp_df["total_salary"].mean()
        m3.metric("Average Salary", f"₹{avg_sal:,.0f}")
        max_sal = emp_df["total_salary"].max()
        m4.metric("Highest Salary", f"₹{max_sal:,.0f}")
    else:
        m3.metric("Average Salary", "🔒 Admin Only")
        m4.metric("Your Employee ID", user.get("employee_id", "N/A"))

    st.markdown("---")

    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.subheader("Department Breakdown")
        if not emp_df.empty:
            dept_counts = emp_df["department"].value_counts().reset_index()
            dept_counts.columns = ["Department", "Count"]
            st.dataframe(dept_counts, use_container_width=True, hide_index=True)
        else:
            st.info("No employee data loaded.")

    with col_right:
        st.subheader("Your Profile Summary")
        if user["role"] == "EMPLOYEE" and not emp_df.empty:
            user_row = emp_df[emp_df["employee_id"] == user["employee_id"]]
            if not user_row.empty:
                r = user_row.iloc[0]
                st.write(f"**Name:** {r['name']}")
                st.write(f"**Department:** {r['department']}")
                st.write(f"**Designation:** {r['designation']}")
                st.write(f"**Joining Year:** {r['joining_year']}")
                st.write(f"**Total Salary:** ₹{r['total_salary']:,}")
            else:
                st.write("Profile metadata not found in database.")
        else:
            st.write(f"**Logged in as Administrator:** {user['name']}")
            st.write("You have unrestricted access to search, view, and query all employee records.")


# --- PAGE 2: EMPLOYEES DIRECTORY ---
elif navigation == "👥 Employees":
    st.subheader("👥 Employee Directory")

    if emp_df.empty:
        st.warning("No employee data available in CSV.")
    else:
        # Search & Filter
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            search_query = st.text_input("🔍 Search by Name or ID", "").strip().lower()
        with col_s2:
            dept_filter = st.selectbox("Filter Department", ["All"] + list(emp_df["department"].unique()))

        filtered_df = emp_df.copy()

        if dept_filter != "All":
            filtered_df = filtered_df[filtered_df["department"] == dept_filter]

        if search_query:
            filtered_df = filtered_df[
                filtered_df["name"].str.lower().str.contains(search_query) |
                filtered_df["employee_id"].str.lower().str.contains(search_query)
            ]

        # Apply Salary Confidentiality formatting based on role
        display_df = filtered_df.copy()
        display_df["basic_salary"] = display_df["basic_salary"].astype(object)
        display_df["bonus"] = display_df["bonus"].astype(object)
        display_df["total_salary"] = display_df["total_salary"].astype(object)

        if user["role"] == "EMPLOYEE":
            user_emp_id = user["employee_id"]
            # Redact salary columns for all employees except logged-in user
            for idx, row in display_df.iterrows():
                if str(row["employee_id"]).strip() != str(user_emp_id).strip():
                    display_df.at[idx, "basic_salary"] = "🔒 Confidential"
                    display_df.at[idx, "bonus"] = "🔒 Confidential"
                    display_df.at[idx, "total_salary"] = "🔒 Confidential"
                else:
                    display_df.at[idx, "basic_salary"] = f"₹{int(row['basic_salary']):,}"
                    display_df.at[idx, "bonus"] = f"₹{int(row['bonus']):,}"
                    display_df.at[idx, "total_salary"] = f"₹{int(row['total_salary']):,}"
        else:
            # Admin sees formatted salaries
            display_df["basic_salary"] = display_df["basic_salary"].apply(lambda x: f"₹{int(x):,}")
            display_df["bonus"] = display_df["bonus"].apply(lambda x: f"₹{int(x):,}")
            display_df["total_salary"] = display_df["total_salary"].apply(lambda x: f"₹{int(x):,}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)


# --- PAGE 3: SALARY OVERVIEW ---
elif navigation == "💰 Salary Overview":
    st.subheader("💰 Salary & Compensation Information")

    if user["role"] == "ADMIN":
        st.success("🔑 Admin Privilege: Viewing company-wide salary statistics.")
        if not emp_df.empty:
            st.dataframe(emp_df.sort_values(by="total_salary", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No employee data loaded.")
    else:
        st.info("ℹ️ As an Employee, you can only view your own salary record.")
        user_row = emp_df[emp_df["employee_id"] == user["employee_id"]] if not emp_df.empty else pd.DataFrame()

        if not user_row.empty:
            r = user_row.iloc[0]
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Basic Salary", f"₹{r['basic_salary']:,}")
            sc2.metric("Annual Bonus", f"₹{r['bonus']:,}")
            sc3.metric("Total Package", f"₹{r['total_salary']:,}")

            st.markdown("---")
            st.json({
                "Employee ID": r["employee_id"],
                "Full Name": r["name"],
                "Department": r["department"],
                "Designation": r["designation"],
                "Basic Salary": f"₹{r['basic_salary']:,}",
                "Bonus": f"₹{r['bonus']:,}",
                "Total Salary": f"₹{r['total_salary']:,}"
            })
        else:
            st.warning("Your employee record was not found.")


# --- PAGE 4: AI ASSISTANT (OPTIMIZED RAG) ---
elif navigation == "🤖 AI Assistant":
    st.subheader("🤖 Sovereign Employee AI Assistant")
    st.markdown("Ask natural language questions regarding employee records. Simple queries use direct database lookups for instant answers.")

    # KB Initialized check
    if not is_kb_initialized():
        st.warning("⚠️ **Knowledge Base Not Initialized**: Please click the button below or go to 'System & KB Status' to index ChromaDB.")
        if st.button("🚀 Initialize Employee Knowledge Base Now"):
            with st.spinner("Indexing employee CSV into local ChromaDB..."):
                res = ingest_employee_data()
                if res["status"] == "success":
                    st.success(f"✓ {res['count']} employee records indexed locally into ChromaDB!")
                    st.rerun()
                else:
                    st.error(res["message"])

    # Quick prompt buttons
    st.markdown("**Suggested Quick Questions:**")
    if user["role"] == "EMPLOYEE":
        q_cols = st.columns(3)
        if q_cols[0].button("What is my salary?"):
            st.session_state.active_query = "What is my salary?"
        if q_cols[1].button("What department do I work in?"):
            st.session_state.active_query = "What department do I work in?"
        if q_cols[2].button("What is Priya Patel's salary?"):
            st.session_state.active_query = "What is Priya Patel's salary?"
    else:
        q_cols = st.columns(3)
        if q_cols[0].button("Who works in Engineering?"):
            st.session_state.active_query = "Who works in Engineering?"
        if q_cols[1].button("What is Arjun Mehta's total salary?"):
            st.session_state.active_query = "What is Arjun Mehta's total salary?"
        if q_cols[2].button("Who is the Engineering Director?"):
            st.session_state.active_query = "Who is the Engineering Director?"

    active_query = st.session_state.get("active_query", "")

    # Query Input Form
    with st.form("rag_form"):
        user_query = st.text_input("Ask a question about employee data:", value=active_query, placeholder="e.g. What is Arjun Mehta's total salary?").strip()
        submit_btn = st.form_submit_button("Ask Sovereign AI", use_container_width=True)

    query_to_run = user_query if submit_btn else active_query
    should_run = bool(submit_btn or active_query)

    # Clear active query state after reading
    if "active_query" in st.session_state:
        del st.session_state["active_query"]

    if should_run and query_to_run:
        st.markdown(f"**Question:** {query_to_run}")

        # ── STEP 1: QUERY ROUTER (Zero-LLM Fast Path) ──
        route_res = route_query(query_to_run, user)

        # Case 1: UNSUPPORTED QUERY
        if route_res["query_type"] == "UNSUPPORTED":
            st.session_state.last_answer = None
            st.session_state.last_metadatas = []
            st.session_state.last_sources = []
            st.session_state.doc_bytes = None

            st.info(f"ℹ️ {route_res['answer']}")

            if show_perf_metrics:
                st.markdown(f"""
                <div class='perf-box'>
                    <strong>⚡ Performance Metrics</strong><br>
                    • Query Classification: {route_res['classification_time']*1000:.2f} ms<br>
                    • Embedding & Search: 0.00 ms (Bypassed)<br>
                    • LLM Generation: 0.00 ms (Bypassed)<br>
                    • <strong>Total Response Time: {route_res['classification_time']*1000:.2f} ms</strong> (Out-of-Scope Rejected)
                </div>
                """, unsafe_allow_html=True)

        # Case 2: SIMPLE LOOKUP (Direct Database Path — Zero LLM)
        elif route_res["query_type"] == "SIMPLE_LOOKUP":
            if not route_res["allowed"]:
                st.session_state.last_answer = None
                st.session_state.last_metadatas = []
                st.session_state.last_sources = []
                st.session_state.doc_bytes = None

                st.markdown(f"""
                <div class='denied-box'>
                    🛑 ACCESS DENIED<br><br>
                    {route_res['denied_reason']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("### AI Answer")
                st.success(f"⚡ {route_res['answer']}")

                if route_res["sources"]:
                    st.markdown("""
                    <div class='source-box'>
                        <strong>Retrieved Sources:</strong><br>
                    """ + "<br>".join([f"• {src}" for src in route_res["sources"]]) + """
                    </div>
                    """, unsafe_allow_html=True)

                if show_perf_metrics:
                    st.markdown(f"""
                    <div class='perf-box'>
                        <strong>⚡ Performance Metrics</strong><br>
                        • Query Classification: {route_res['classification_time']*1000:.2f} ms<br>
                        • Embedding & Search: 0.00 ms (Bypassed)<br>
                        • LLM Generation: 0.00 ms (Bypassed)<br>
                        • <strong>Total Response Time: {route_res['classification_time']*1000:.2f} ms</strong> (Direct Database Lookup)
                    </div>
                    """, unsafe_allow_html=True)

                # Store result in session state for Document Creator
                st.session_state.last_answer = route_res["answer"]
                st.session_state.last_context = "\n\n".join(route_res["documents"])
                st.session_state.last_sources = route_res["sources"]
                st.session_state.last_metadatas = route_res["metadatas"]
                st.session_state.last_query = query_to_run
                st.session_state.doc_bytes = None
                st.session_state.doc_filename = None
                st.session_state.doc_format = None

        # Case 3: RAG QUERY (ChromaDB Vector Search + Qwen LLM)
        else:
            with st.spinner("Searching local ChromaDB vector store..."):
                retrieval_res = retrieve_employee_context(query_to_run, user)

            if not retrieval_res["allowed"]:
                st.session_state.last_answer = None
                st.session_state.last_metadatas = []
                st.session_state.last_sources = []
                st.session_state.doc_bytes = None

                st.markdown(f"""
                <div class='denied-box'>
                    🛑 ACCESS DENIED<br><br>
                    {retrieval_res['denied_reason']}
                </div>
                """, unsafe_allow_html=True)

            elif retrieval_res.get("kb_uninitialized"):
                st.session_state.last_answer = None
                st.warning(f"⚠️ {retrieval_res.get('denied_reason', 'Knowledge Base is not initialized.')}")
                if st.button("🚀 Initialize Employee Knowledge Base Now"):
                    with st.spinner("Indexing employee CSV into local ChromaDB..."):
                        res = ingest_employee_data()
                        if res["status"] == "success":
                            st.success(f"✓ {res['count']} employee records indexed into ChromaDB!")
                            st.rerun()
                        else:
                            st.error(res["message"])

            elif retrieval_res.get("not_found"):
                st.session_state.last_answer = None
                st.session_state.last_metadatas = []
                st.session_state.doc_bytes = None
                st.warning("I couldn't find any relevant employee records matching your query in the local employee database.")

            else:
                # Grounded LLM Generation
                context_text = "\n\n".join(retrieval_res["documents"])
                with st.spinner("Generating answer locally via Ollama Qwen LLM..."):
                    llm_res = generate_answer(query_to_run, context_text, user)

                if llm_res["status"] == "success":
                    st.markdown("### AI Answer")
                    st.write(llm_res["answer"])

                    if retrieval_res["sources"]:
                        st.markdown("""
                        <div class='source-box'>
                            <strong>Retrieved Sources:</strong><br>
                        """ + "<br>".join([f"• {src}" for src in retrieval_res["sources"]]) + """
                        </div>
                        """, unsafe_allow_html=True)

                    with st.expander("🔍 View Raw Vector Context"):
                        for idx, doc in enumerate(retrieval_res["documents"]):
                            st.code(doc, language="yaml")

                    t_class = route_res.get("classification_time", 0.0)
                    t_ret = retrieval_res.get("retrieval_time", 0.0)
                    t_llm = llm_res.get("llm_generation_time", 0.0)
                    t_tot = t_class + t_ret + t_llm

                    if show_perf_metrics:
                        st.markdown(f"""
                        <div class='perf-box'>
                            <strong>⚡ Performance Metrics</strong><br>
                            • Query Classification: {t_class*1000:.2f} ms<br>
                            • Embedding & Search: {t_ret*1000:.2f} ms ({t_ret:.3f}s)<br>
                            • LLM Generation: {t_llm*1000:.2f} ms ({t_llm:.3f}s)<br>
                            • <strong>Total Response Time: {t_tot:.3f}s</strong> (Full RAG Pipeline)
                        </div>
                        """, unsafe_allow_html=True)

                    # Store RAG result in session state for Document Creator
                    st.session_state.last_answer = llm_res["answer"]
                    st.session_state.last_context = context_text
                    st.session_state.last_sources = retrieval_res["sources"]
                    st.session_state.last_metadatas = retrieval_res["metadatas"]
                    st.session_state.last_query = query_to_run
                    st.session_state.doc_bytes = None
                    st.session_state.doc_filename = None
                    st.session_state.doc_format = None

                else:
                    st.session_state.last_answer = None
                    st.error(llm_res["answer"])

    # ── DOCUMENT CREATOR UI ──────────────────────────────────────────────────
    # Only displayed when a valid answer exists in session state
    if st.session_state.get("last_answer"):
        st.markdown("---")
        st.markdown("""
        <div class='doc-creator-box'>
            <div class='doc-creator-title'>📄 Document Creator</div>
            <div class='doc-creator-subtitle'>
                Generate an official document from the retrieved employee data.
                All generation happens locally — no cloud APIs.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")  # spacing after the styled box

        dc_col1, dc_col2 = st.columns(2)
        with dc_col1:
            selected_doc_type = st.selectbox(
                "Document Type",
                DOC_TYPES,
                key="dc_doc_type",
                help=(
                    "Employee Report: Full profile + salary\n"
                    "Salary Statement: Salary-focused\n"
                    "Employee Summary: Profile only, no salary\n"
                    "Custom RAG Report: Uses the AI answer as narrative body"
                )
            )
        with dc_col2:
            selected_format = st.selectbox(
                "Format",
                DOC_FORMATS,
                key="dc_format"
            )

        if st.button("📄 Generate Document", use_container_width=True, type="primary"):
            with st.spinner("Generating document locally..."):
                try:
                    # Extract validated employee data from RAG metadata
                    employee_data = extract_employee_from_context(
                        st.session_state.last_metadatas, user
                    )

                    # For non-Custom types, employee_data must be present
                    if selected_doc_type != "Custom RAG Report" and employee_data is None:
                        st.error(
                            "⚠️ Could not extract authorized employee data from the result.\n\n"
                            "This may happen if:\n"
                            "- The query result does not contain your employee record\n"
                            "- You do not have permission to generate a document for this record\n\n"
                            "Try asking a question about your own employee details first."
                        )
                    else:
                        doc_bytes, doc_filename = generate_document(
                            doc_type=selected_doc_type,
                            doc_format=selected_format,
                            employee=employee_data,
                            rag_answer=st.session_state.last_answer,
                            sources=st.session_state.last_sources,
                        )
                        st.session_state.doc_bytes = doc_bytes
                        st.session_state.doc_filename = doc_filename
                        st.session_state.doc_format = selected_format

                except ValueError as e:
                    st.error(f"⚠️ Document generation error: {e}")
                except RuntimeError as e:
                    st.error(f"⚠️ {e}")
                except Exception as e:
                    st.error(f"⚠️ Unexpected error during document generation: {e}")

        # ── Download Button (shown after successful generation) ──
        if st.session_state.get("doc_bytes") and st.session_state.get("doc_filename"):
            fmt = st.session_state.doc_format
            mime_map = {
                "PDF": "application/pdf",
                "DOCX": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "TXT": "text/plain",
            }
            mime = mime_map.get(fmt, "application/octet-stream")

            st.success(f"✅ Document generated successfully — **{st.session_state.doc_filename}**")
            st.download_button(
                label=f"⬇️ Download {fmt}",
                data=st.session_state.doc_bytes,
                file_name=st.session_state.doc_filename,
                mime=mime,
                use_container_width=True,
            )

    elif st.session_state.get("last_answer") is None and not (should_run and query_to_run):
        # First load — show a helpful hint
        st.markdown("---")
        st.info(
            "💡 **Document Creator** will appear here after a successful AI answer.\n\n"
            "Ask a question above to retrieve employee data, then generate a PDF, DOCX, or TXT report."
        )


# --- PAGE 5: SYSTEM & KB STATUS ---
elif navigation == "⚙️ System & KB Status":
    st.subheader("⚙️ Local System Diagnostics & Vector Store")

    kb_status = is_kb_initialized()
    if kb_status:
        st.success("🟢 **Vector Database:** Active & Initialized (ChromaDB)")
    else:
        st.error("🔴 **Vector Database:** Not Initialized")

    st.markdown("---")
    st.markdown("### Knowledge Base Ingestion")
    if st.button("🔄 Re-index Employee Knowledge Base", use_container_width=True):
        with st.spinner("Indexing employees.csv into ChromaDB via qwen3:4b..."):
            res = ingest_employee_data()
            if res["status"] == "success":
                st.success(f"✓ Successfully indexed {res['count']} employee records!")
                st.rerun()
            else:
                st.error(f"Ingestion failed: {res['message']}")

    st.markdown("---")
    st.markdown("### Ollama & Local LLM Diagnostics")
    ollama_diag = check_ollama_status()
    if ollama_diag["status"] == "ok":
        st.success("🟢 **Ollama Service:** Running locally")
        st.write(f"**Configured LLM Model:** `{ollama_diag['target_model']}`")
        st.write(f"**Target Model Installed:** {'Yes' if ollama_diag['target_model_installed'] else 'No (Will use fallback)'}")
        st.write(f"**Installed Local Models:** {', '.join(ollama_diag['available_models']) if ollama_diag['available_models'] else 'None'}")
    else:
        st.warning(ollama_diag["message"])

    st.markdown("---")
    st.markdown("### Document Generation Status")
    doc_libs = {}
    try:
        import reportlab
        doc_libs["ReportLab (PDF)"] = f"✅ Installed (v{reportlab.Version})"
    except ImportError:
        doc_libs["ReportLab (PDF)"] = "❌ Not installed — run: `pip install reportlab`"
    try:
        import docx
        doc_libs["python-docx (DOCX)"] = "✅ Installed"
    except ImportError:
        doc_libs["python-docx (DOCX)"] = "❌ Not installed — run: `pip install python-docx`"

    for lib, status in doc_libs.items():
        st.write(f"**{lib}:** {status}")


# --- FOOTER ---
st.markdown("""
<div class='footer-text'>
    Local LLM: Qwen (qwen3:4b) | Embeddings: nomic-embed-text | Vector DB: ChromaDB | Data Source: Local Employee Database<br>
    Document Generation: ReportLab (PDF) · python-docx (DOCX) · Built-in (TXT)<br>
    <em>Sovereign RAG Architecture — Optimized for Fast Local Response Latency</em>
</div>
""", unsafe_allow_html=True)
