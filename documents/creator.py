"""
documents/creator.py — Local Document Generation Module

Receives validated, permission-filtered employee data from the application layer
and generates TXT, PDF, or DOCX documents without independently querying the LLM
or database.

Architecture:
    RAG Retrieval
        ↓
    Validated Context + Structured Employee Data (passed in by app.py)
        ↓
    Document Creator  (this module)
        ↓
    bytes  →  Streamlit Download Button
"""

import io
import html
from datetime import datetime

# ─── Supported Constants ────────────────────────────────────────────────────

DOC_TYPES = ["Employee Report", "Salary Statement", "Employee Summary", "Custom RAG Report"]
DOC_FORMATS = ["PDF", "DOCX", "TXT"]

BRAND_NAME = "SOVEREIGN AI"
BRAND_SUBTITLE = "Local RAG Prototype"


# ─── Helper: Safe Integer Parsing ─────────────────────────────────────────────

def _safe_int(val, default: int = 0) -> int:
    """Safely converts string/float/int to integer, handling currency symbols and commas."""
    if val is None:
        return default
    try:
        if isinstance(val, (int, float)):
            return int(val)
        cleaned = str(val).replace("₹", "").replace("$", "").replace(",", "").strip()
        return int(float(cleaned))
    except (ValueError, TypeError):
        return default


# ─── Helper: Format Indian Rupee ─────────────────────────────────────────────

def _fmt_inr(amount) -> str:
    """Returns a formatted Indian Rupee string, e.g. ₹11,00,000"""
    try:
        return f"₹{_safe_int(amount):,}"
    except Exception:
        return str(amount)


# ─── Helper: Build report timestamp ─────────────────────────────────────────

def _timestamp() -> str:
    return datetime.now().strftime("%d %B %Y, %I:%M %p")


# ─── Security: Extract & validate employee from RAG metadata ─────────────────

def extract_employee_from_context(metadatas: list[dict], user: dict) -> dict | None:
    """
    Extracts a structured employee dict from the RAG retrieval metadata.

    Security Rules:
    - ADMIN: May use the first relevant metadata record.
    - EMPLOYEE: May ONLY use a record that matches their own employee_id.

    Returns a structured employee dict or None if authorization fails / data missing.
    """
    if not metadatas:
        return None

    user_role = str(user.get("role", "EMPLOYEE")).strip().upper()
    user_emp_id = str(user.get("employee_id", "")).strip().upper()

    if user_role == "ADMIN":
        # Admin: use the first (highest-similarity) retrieved metadata record
        meta = metadatas[0]
    else:
        # Employee: scan all returned metadatas for their own record only
        meta = None
        for m in metadatas:
            if str(m.get("employee_id", "")).strip().upper() == user_emp_id:
                meta = m
                break
        if meta is None:
            return None

    # Build a clean, typed employee dict from the metadata
    try:
        employee = {
            "employee_id": str(meta.get("employee_id", "N/A")).strip(),
            "name": str(meta.get("name", "N/A")).strip(),
            "department": str(meta.get("department", "N/A")).strip(),
            "designation": str(meta.get("designation", "N/A")).strip(),
            "joining_year": _safe_int(meta.get("joining_year"), 0),
            "basic_salary": _safe_int(meta.get("basic_salary"), 0),
            "bonus": _safe_int(meta.get("bonus"), 0),
            "total_salary": _safe_int(meta.get("total_salary"), 0),
        }
        return employee
    except Exception:
        return None


# ─── TXT Generator ───────────────────────────────────────────────────────────

def create_txt_document(
    doc_type: str,
    employee: dict | None,
    rag_answer: str,
    sources: list[str],
) -> bytes:
    """
    Generates a plain-text document and returns it as UTF-8 bytes.
    """
    lines = []
    divider = "-" * 56

    lines.append(divider)
    lines.append(f"          {BRAND_NAME}")

    if doc_type == "Employee Report":
        lines.append("          EMPLOYEE REPORT")
    elif doc_type == "Salary Statement":
        lines.append("          SALARY STATEMENT")
    elif doc_type == "Employee Summary":
        lines.append("          EMPLOYEE SUMMARY")
    else:
        lines.append("          CUSTOM RAG REPORT")

    lines.append(divider)
    lines.append(f"Generated: {_timestamp()}")
    lines.append("")

    if doc_type == "Employee Report" and employee:
        lines.append("Employee Information")
        lines.append("")
        lines.append(f"  Employee ID    {employee['employee_id']}")
        lines.append(f"  Name           {employee['name']}")
        lines.append(f"  Department     {employee['department']}")
        lines.append(f"  Designation    {employee['designation']}")
        lines.append(f"  Joining Year   {employee['joining_year']}")
        lines.append("")
        lines.append("Salary Information")
        lines.append("")
        lines.append(f"  Basic Salary   {_fmt_inr(employee['basic_salary'])}")
        lines.append(f"  Bonus          {_fmt_inr(employee['bonus'])}")
        lines.append(f"  Total Salary   {_fmt_inr(employee['total_salary'])}")

    elif doc_type == "Salary Statement" and employee:
        lines.append("Salary Statement")
        lines.append("")
        lines.append(f"  Employee       {employee['name']}")
        lines.append(f"  Employee ID    {employee['employee_id']}")
        lines.append("")
        lines.append(f"  Basic Salary   {_fmt_inr(employee['basic_salary'])}")
        lines.append(f"  Bonus          {_fmt_inr(employee['bonus'])}")
        lines.append(f"  Total Salary   {_fmt_inr(employee['total_salary'])}")

    elif doc_type == "Employee Summary" and employee:
        lines.append("Employee Summary")
        lines.append("")
        lines.append(f"  Name           {employee['name']}")
        lines.append(f"  Department     {employee['department']}")
        lines.append(f"  Designation    {employee['designation']}")
        lines.append(f"  Joining Year   {employee['joining_year']}")

    else:
        # Custom RAG Report — use AI-generated narrative
        lines.append("AI-Generated Report")
        lines.append("")
        lines.append(rag_answer if rag_answer else "[No content available]")

    lines.append("")
    lines.append(divider)
    lines.append("Source:")
    if sources:
        for src in sources:
            lines.append(f"  Local Employee Database — {src}")
    else:
        lines.append("  Sovereign RAG Knowledge Base")
    lines.append("")
    lines.append(f"Generated by {BRAND_NAME}")
    lines.append(BRAND_SUBTITLE)
    lines.append(divider)

    return "\n".join(lines).encode("utf-8")


# ─── PDF Generator ────────────────────────────────────────────────────────────

def create_pdf_document(
    doc_type: str,
    employee: dict | None,
    rag_answer: str,
    sources: list[str],
) -> bytes:
    """
    Generates a PDF document using ReportLab and returns it as bytes.
    Uses HTML escaping on dynamic strings to prevent XML parsing errors in Paragraphs.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        raise ImportError(
            "reportlab is required for PDF generation. "
            "Install it with: pip install reportlab"
        )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    # ── Styles ──
    styles = getSampleStyleSheet()

    style_brand = ParagraphStyle(
        "Brand",
        parent=styles["Heading1"],
        fontSize=16,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=2,
    )
    style_doc_title = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading2"],
        fontSize=13,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
        textColor=colors.HexColor("#16213e"),
        spaceAfter=4,
    )
    style_section = ParagraphStyle(
        "Section",
        parent=styles["Heading3"],
        fontSize=11,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0f3460"),
        spaceBefore=14,
        spaceAfter=6,
    )
    style_body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        fontName="Helvetica",
        leading=15,
        textColor=colors.HexColor("#222222"),
    )
    style_footer = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=8,
        fontName="Helvetica",
        alignment=TA_CENTER,
        textColor=colors.HexColor("#888888"),
        spaceAfter=0,
    )
    style_meta = ParagraphStyle(
        "Meta",
        parent=styles["Normal"],
        fontSize=8,
        fontName="Helvetica",
        textColor=colors.HexColor("#888888"),
        alignment=TA_CENTER,
    )

    # ── Table Style ──
    table_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])

    # ── Build Content ──
    story = []

    # Header
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(BRAND_NAME, style_brand))

    doc_title_map = {
        "Employee Report": "EMPLOYEE REPORT",
        "Salary Statement": "SALARY STATEMENT",
        "Employee Summary": "EMPLOYEE SUMMARY",
        "Custom RAG Report": "CUSTOM RAG REPORT",
    }
    story.append(Paragraph(doc_title_map.get(doc_type, "REPORT"), style_doc_title))
    story.append(Paragraph(f"Generated: {_timestamp()}", style_meta))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dee2e6")))
    story.append(Spacer(1, 10))

    # Document Body
    if doc_type == "Employee Report" and employee:
        story.append(Paragraph("Employee Information", style_section))
        emp_data = [
            ["Field", "Value"],
            ["Employee ID", str(employee["employee_id"])],
            ["Name", str(employee["name"])],
            ["Department", str(employee["department"])],
            ["Designation", str(employee["designation"])],
            ["Joining Year", str(employee["joining_year"])],
        ]
        t = Table(emp_data, colWidths=[5 * cm, 11 * cm])
        t.setStyle(table_style)
        story.append(t)

        story.append(Paragraph("Salary Information", style_section))
        sal_data = [
            ["Component", "Amount"],
            ["Basic Salary", _fmt_inr(employee["basic_salary"])],
            ["Bonus", _fmt_inr(employee["bonus"])],
            ["Total Salary", _fmt_inr(employee["total_salary"])],
        ]
        t2 = Table(sal_data, colWidths=[5 * cm, 11 * cm])
        t2.setStyle(table_style)
        story.append(t2)

    elif doc_type == "Salary Statement" and employee:
        story.append(Paragraph("Salary Statement", style_section))
        sal_data = [
            ["Field", "Value"],
            ["Employee", str(employee["name"])],
            ["Employee ID", str(employee["employee_id"])],
        ]
        t = Table(sal_data, colWidths=[5 * cm, 11 * cm])
        t.setStyle(table_style)
        story.append(t)
        story.append(Spacer(1, 8))

        comp_data = [
            ["Salary Component", "Amount"],
            ["Basic Salary", _fmt_inr(employee["basic_salary"])],
            ["Annual Bonus", _fmt_inr(employee["bonus"])],
            ["Total Package", _fmt_inr(employee["total_salary"])],
        ]
        t2 = Table(comp_data, colWidths=[5 * cm, 11 * cm])
        t2.setStyle(table_style)
        story.append(t2)

    elif doc_type == "Employee Summary" and employee:
        story.append(Paragraph("Employee Summary", style_section))
        sum_data = [
            ["Field", "Value"],
            ["Name", str(employee["name"])],
            ["Department", str(employee["department"])],
            ["Designation", str(employee["designation"])],
            ["Joining Year", str(employee["joining_year"])],
        ]
        t = Table(sum_data, colWidths=[5 * cm, 11 * cm])
        t.setStyle(table_style)
        story.append(t)

    else:
        # Custom RAG Report — escape special chars and split by newlines for clean ReportLab flow
        story.append(Paragraph("AI-Generated Report", style_section))
        story.append(Spacer(1, 6))

        answer_text = rag_answer or "[No content available]"
        paragraphs = [p.strip() for p in answer_text.split("\n") if p.strip()]
        if not paragraphs:
            paragraphs = ["[No content available]"]

        for p_text in paragraphs:
            safe_p = html.escape(p_text)
            story.append(Paragraph(safe_p, style_body))
            story.append(Spacer(1, 6))

    # Source section
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dee2e6")))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Source:", style_section))
    if sources:
        for src in sources:
            safe_src = html.escape(str(src))
            story.append(Paragraph(f"• Local Employee Database — {safe_src}", style_body))
    else:
        story.append(Paragraph("• Sovereign RAG Knowledge Base", style_body))

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated by {BRAND_NAME} — {BRAND_SUBTITLE}", style_footer))
    story.append(Paragraph("Designed for local/private processing", style_footer))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ─── DOCX Generator ──────────────────────────────────────────────────────────

def create_docx_document(
    doc_type: str,
    employee: dict | None,
    rag_answer: str,
    sources: list[str],
) -> bytes:
    """
    Generates a DOCX document using python-docx and returns it as bytes.
    Includes robust fallback for styles, run colors, and XML elements.
    """
    try:
        from docx import Document as DocxDocument
        from docx.shared import Pt, RGBColor, Cm
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
    except ImportError:
        raise ImportError(
            "python-docx is required for DOCX generation. "
            "Install it with: pip install python-docx"
        )

    doc = DocxDocument()

    # ── Page margins ──
    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    # ── Helper functions ──
    def add_heading(text: str, level: int = 1, color_hex: str = "1a1a2e"):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(text)
        run.bold = True
        if level == 1:
            run.font.size = Pt(16)
        elif level == 2:
            run.font.size = Pt(13)
        elif level == 3:
            run.font.size = Pt(11)
        r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
        run.font.color.rgb = RGBColor(r, g, b)
        return p

    def add_bullet(text: str):
        """Safely adds a bullet point, falling back to unicode bullet if style missing."""
        try:
            p = doc.add_paragraph(style="List Bullet")
            run = p.add_run(text)
            run.font.size = Pt(9)
        except KeyError:
            p = doc.add_paragraph()
            r1 = p.add_run("• ")
            r1.font.size = Pt(9)
            r2 = p.add_run(text)
            r2.font.size = Pt(9)
        return p

    def add_kv_table(rows: list[tuple]):
        """Adds a clean key-value table."""
        table = doc.add_table(rows=len(rows), cols=2)
        table.style = "Table Grid"
        for i, (key, val) in enumerate(rows):
            cells = table.rows[i].cells
            cells[0].text = str(key)
            cells[1].text = str(val)
            for para in cells[0].paragraphs:
                for run in para.runs:
                    run.bold = True
                    run.font.size = Pt(10)
            for para in cells[1].paragraphs:
                for run in para.runs:
                    run.font.size = Pt(10)
        return table

    def add_divider():
        try:
            p = doc.add_paragraph()
            pPr = p._p.get_or_add_pPr()
            pBdr = OxmlElement("w:pBdr")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "6")
            bottom.set(qn("w:space"), "1")
            bottom.set(qn("w:color"), "1a1a2e")
            pBdr.append(bottom)
            pPr.append(pBdr)
        except Exception:
            pass

    # ── Document Header ──
    add_divider()
    add_heading(BRAND_NAME, level=1)

    doc_title_map = {
        "Employee Report": "EMPLOYEE REPORT",
        "Salary Statement": "SALARY STATEMENT",
        "Employee Summary": "EMPLOYEE SUMMARY",
        "Custom RAG Report": "CUSTOM RAG REPORT",
    }
    add_heading(doc_title_map.get(doc_type, "REPORT"), level=2, color_hex="16213e")

    meta_p = doc.add_paragraph(f"Generated: {_timestamp()}")
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in meta_p.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    add_divider()
    doc.add_paragraph()

    # ── Document Body ──
    if doc_type == "Employee Report" and employee:
        add_heading("Employee Information", level=3, color_hex="0f3460")
        add_kv_table([
            ("Employee ID", employee["employee_id"]),
            ("Name", employee["name"]),
            ("Department", employee["department"]),
            ("Designation", employee["designation"]),
            ("Joining Year", str(employee["joining_year"])),
        ])
        doc.add_paragraph()
        add_heading("Salary Information", level=3, color_hex="0f3460")
        add_kv_table([
            ("Basic Salary", _fmt_inr(employee["basic_salary"])),
            ("Bonus", _fmt_inr(employee["bonus"])),
            ("Total Salary", _fmt_inr(employee["total_salary"])),
        ])

    elif doc_type == "Salary Statement" and employee:
        add_heading("Salary Statement", level=3, color_hex="0f3460")
        add_kv_table([
            ("Employee", employee["name"]),
            ("Employee ID", employee["employee_id"]),
        ])
        doc.add_paragraph()
        add_heading("Compensation Breakdown", level=3, color_hex="0f3460")
        add_kv_table([
            ("Basic Salary", _fmt_inr(employee["basic_salary"])),
            ("Annual Bonus", _fmt_inr(employee["bonus"])),
            ("Total Package", _fmt_inr(employee["total_salary"])),
        ])

    elif doc_type == "Employee Summary" and employee:
        add_heading("Employee Summary", level=3, color_hex="0f3460")
        add_kv_table([
            ("Name", employee["name"]),
            ("Department", employee["department"]),
            ("Designation", employee["designation"]),
            ("Joining Year", str(employee["joining_year"])),
        ])

    else:
        # Custom RAG Report — AI narrative
        add_heading("AI-Generated Report", level=3, color_hex="0f3460")
        doc.add_paragraph()
        answer_text = rag_answer or "[No content available]"
        paragraphs = [p.strip() for p in answer_text.split("\n") if p.strip()]
        if not paragraphs:
            paragraphs = ["[No content available]"]

        for p_text in paragraphs:
            body_para = doc.add_paragraph()
            run = body_para.add_run(p_text)
            run.font.size = Pt(10)

    # ── Source Section ──
    doc.add_paragraph()
    add_divider()
    add_heading("Source", level=3, color_hex="555555")
    if sources:
        for src in sources:
            add_bullet(f"Local Employee Database — {src}")
    else:
        add_bullet("Sovereign RAG Knowledge Base")

    # ── Footer ──
    doc.add_paragraph()
    add_divider()
    footer_p = doc.add_paragraph(f"Generated by {BRAND_NAME} — {BRAND_SUBTITLE}")
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in footer_p.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    footer_p2 = doc.add_paragraph("Designed for local/private processing")
    footer_p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in footer_p2.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()


# ─── Unified Entry Point ──────────────────────────────────────────────────────

def generate_document(
    doc_type: str,
    doc_format: str,
    employee: dict | None,
    rag_answer: str,
    sources: list[str],
) -> tuple[bytes, str]:
    """
    Master entry point called by app.py.

    Args:
        doc_type:    One of DOC_TYPES
        doc_format:  One of DOC_FORMATS ("PDF", "DOCX", "TXT")
        employee:    Structured employee dict or None (for Custom RAG Report)
        rag_answer:  LLM-generated answer (used in Custom RAG Report)
        sources:     Retrieved source strings

    Returns:
        (bytes, filename) tuple

    Raises:
        ValueError: If doc_type or doc_format is invalid
        RuntimeError: On generation failure
    """
    if doc_type not in DOC_TYPES:
        raise ValueError(f"Invalid document type: {doc_type}. Must be one of {DOC_TYPES}")
    if doc_format not in DOC_FORMATS:
        raise ValueError(f"Invalid format: {doc_format}. Must be one of {DOC_FORMATS}")

    # For non-Custom types, employee data is required
    if doc_type != "Custom RAG Report" and employee is None:
        raise ValueError(
            f"Employee data is required for '{doc_type}'. "
            "Ensure a valid employee record was retrieved."
        )

    # Generate filename
    emp_id_part = employee["employee_id"] if employee else "rag"
    doc_slug = doc_type.lower().replace(" ", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext_map = {"PDF": "pdf", "DOCX": "docx", "TXT": "txt"}
    filename = f"{emp_id_part}_{doc_slug}_{ts}.{ext_map[doc_format]}"

    try:
        if doc_format == "TXT":
            doc_bytes = create_txt_document(doc_type, employee, rag_answer, sources)
        elif doc_format == "PDF":
            doc_bytes = create_pdf_document(doc_type, employee, rag_answer, sources)
        elif doc_format == "DOCX":
            doc_bytes = create_docx_document(doc_type, employee, rag_answer, sources)
        else:
            raise ValueError(f"Unsupported format: {doc_format}")
    except ImportError as e:
        raise RuntimeError(f"Missing library: {e}")
    except Exception as e:
        raise RuntimeError(f"Document generation failed: {e}")

    return doc_bytes, filename
