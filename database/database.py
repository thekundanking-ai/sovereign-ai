import os
import sqlite3
import hashlib
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "sovereign.db")
CSV_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "employees.csv"))


def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    if not salt:
        salt = os.urandom(16).hex()
    salted = f"{password}{salt}".encode('utf-8')
    pw_hash = hashlib.sha256(salted).hexdigest()
    return pw_hash, salt


def verify_password(password: str, pw_hash: str, salt: str) -> bool:
    new_hash, _ = hash_password(password, salt)
    return new_hash == pw_hash


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite database schema and seeds initial users from CSV."""
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL,
            employee_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()

    # Seed Admin User if not exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        pw_hash, salt = hash_password("admin123")
        cursor.execute(
            "INSERT INTO users (username, password_hash, salt, role, employee_id, name) VALUES (?, ?, ?, ?, ?, ?)",
            ("admin", pw_hash, salt, "ADMIN", "EMP000", "System Administrator")
        )

    # Seed Employee Users from employees.csv if available
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            for _, row in df.iterrows():
                emp_id = str(row['employee_id']).strip()
                full_name = str(row['name']).strip()
                first_name = full_name.split()[0].lower()
                
                cursor.execute("SELECT * FROM users WHERE employee_id = ?", (emp_id,))
                if not cursor.fetchone():
                    pw_hash, salt = hash_password("user123")
                    cursor.execute(
                        "INSERT INTO users (username, password_hash, salt, role, employee_id, name) VALUES (?, ?, ?, ?, ?, ?)",
                        (first_name, pw_hash, salt, "EMPLOYEE", emp_id, full_name)
                    )
        except Exception as e:
            print(f"[Database] Error seeding employees from CSV: {e}")

    conn.commit()
    conn.close()


def get_user_by_username(username: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None


def get_user_by_emp_id(employee_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE employee_id = ?", (employee_id.strip(),))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None


def get_all_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role, employee_id, name, created_at FROM users")
    users = cursor.fetchall()
    conn.close()
    return [dict(u) for u in users]


if __name__ == "__main__":
    init_db()
    print("[Database] Database initialized and seeded successfully.")
