"""
Authentication and Role-Based Access Control (RBAC) Module.

NOTE: This is a prototype local authentication system designed for demonstration and educational purposes.
Do not use this directly in enterprise production without HTTPS, JWT/session management, and rate limiting.
"""

from database.database import get_user_by_username, verify_password, init_db

# Salary related keywords to flag salary queries
SALARY_KEYWORDS = [
    "salary", "salaries", "pay", "payout", "bonus", "basic_salary",
    "total_salary", "compensation", "package", "earnings", "earned", "remuneration"
]


def authenticate_user(username: str, password: str) -> dict | None:
    """Authenticates username and password against SQLite database."""
    init_db()
    user = get_user_by_username(username)
    if not user:
        return None
    
    if verify_password(password, user["password_hash"], user["salt"]):
        return {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "employee_id": user["employee_id"],
            "name": user["name"]
        }
    return None


def is_salary_query(query_text: str) -> bool:
    """Detects if query is asking about salary or financial compensation."""
    q_lower = query_text.lower()
    return any(keyword in q_lower for keyword in SALARY_KEYWORDS)


def can_access_employee_salary(current_user: dict, target_emp_id: str = None, target_name: str = None) -> bool:
    """
    Role Access Policy:
    - ADMIN: Can view any employee's salary and metrics.
    - EMPLOYEE: Can ONLY view their own salary. Accessing any other employee's salary is DENIED.
    """
    if not current_user:
        return False
    
    # Admin can access everything
    if current_user.get("role") == "ADMIN":
        return True
    
    # Employee role checks
    user_emp_id = current_user.get("employee_id", "").strip().upper()
    user_name = current_user.get("name", "").strip().lower()

    if target_emp_id and target_emp_id.strip().upper() == user_emp_id:
        return True

    if target_name and user_name in target_name.strip().lower():
        return True

    if not target_emp_id and not target_name:
        # Requesting general self-info
        return True

    return False
