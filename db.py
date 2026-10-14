"""SQLite persistence: users (hashed passwords), assessments, audit trail, policy settings.

Demo-grade security: salted PBKDF2 password hashes, lock-out after repeated failures and an
audit log. For production use a real identity provider (SSO) and a managed database.
"""
import hashlib
import hmac
import json
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime

from config import BASE

DB_PATH = BASE / "data" / "bank.db"
ROLES = ["Loan Officer", "Branch Manager", "Credit Analyst", "Administrator"]
MAX_FAILED = 5

DEMO_USERS = [  # username, password, role, full name, branch
    ("officer1", "Officer@123", "Loan Officer", "Alex Carter", "Main Branch"),
    ("officer2", "Officer@123", "Loan Officer", "Maya Singh", "Main Branch"),
    ("manager1", "Manager@123", "Branch Manager", "Daniel Moore", "Main Branch"),
    ("analyst1", "Analyst@123", "Credit Analyst", "Priya Nair", "Head Office"),
    ("admin", "Admin@123", "Administrator", "System Admin", "Head Office"),
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run(sql, params=(), fetch=None):
    DB_PATH.parent.mkdir(exist_ok=True)
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as c:
        c.row_factory = sqlite3.Row
        cur = c.execute(sql, params)
        c.commit()
        if fetch == "one":
            r = cur.fetchone()
            return dict(r) if r else None
        if fetch == "all":
            return [dict(r) for r in cur.fetchall()]
        return cur.lastrowid


def hash_pw(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 150_000).hex()


def init_db():
    run("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, full_name TEXT,
        role TEXT NOT NULL, branch TEXT, salt TEXT NOT NULL, pw_hash TEXT NOT NULL,
        active INTEGER DEFAULT 1, failed_attempts INTEGER DEFAULT 0, last_login TEXT, created_at TEXT)""")
    run("""CREATE TABLE IF NOT EXISTS assessments(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, officer TEXT, officer_name TEXT,
        branch TEXT, applicant_name TEXT, purpose TEXT, loan_amount REAL, probability REAL,
        rf_prob REAL, gb_prob REAL, recommendation TEXT, risk_level TEXT, mode TEXT,
        applicant_json TEXT, status TEXT, final_decision TEXT, reviewer TEXT,
        review_comment TEXT, reviewed_at TEXT, override INTEGER DEFAULT 0)""")
    run("""CREATE TABLE IF NOT EXISTS audit(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, username TEXT, action TEXT, detail TEXT)""")
    run("CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT)")
    if not run("SELECT COUNT(*) AS n FROM users", fetch="one")["n"]:
        for u, pw, role, name, branch in DEMO_USERS:
            create_user(u, pw, role, name, branch, who="system")
    if not get_setting("low_t"):
        set_setting("low_t", "0.25")
        set_setting("high_t", "0.50")


# ---------- settings ----------
def get_setting(key):
    r = run("SELECT value FROM settings WHERE key=?", (key,), "one")
    return r["value"] if r else None


def set_setting(key, value):
    run("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)))


def get_thresholds():
    return float(get_setting("low_t")), float(get_setting("high_t"))


# ---------- audit ----------
def audit(username, action, detail=""):
    run("INSERT INTO audit(ts,username,action,detail) VALUES(?,?,?,?)", (now(), username, action, detail))


def list_audit(limit=500):
    return run("SELECT ts AS Time, username AS User, action AS Action, detail AS Detail "
               "FROM audit ORDER BY id DESC LIMIT ?", (limit,), "all")


# ---------- users ----------
def create_user(username, password, role, full_name, branch, who="admin"):
    salt = secrets.token_hex(16)
    run("INSERT INTO users(username,full_name,role,branch,salt,pw_hash,created_at) VALUES(?,?,?,?,?,?,?)",
        (username.strip().lower(), full_name, role, branch, salt, hash_pw(password, salt), now()))
    audit(who, "USER_CREATED", f"{username} ({role})")


def list_users():
    return run("SELECT username AS Username, full_name AS Name, role AS Role, branch AS Branch, "
               "CASE active WHEN 1 THEN 'Active' ELSE 'Deactivated' END AS Status, "
               "failed_attempts AS 'Failed attempts', last_login AS 'Last login' "
               "FROM users ORDER BY role, username", fetch="all")


def set_password(username, new_password, who):
    salt = secrets.token_hex(16)
    run("UPDATE users SET salt=?, pw_hash=?, failed_attempts=0 WHERE username=?",
        (salt, hash_pw(new_password, salt), username))
    audit(who, "PASSWORD_CHANGED", username)


def set_active(username, active, who):
    run("UPDATE users SET active=?, failed_attempts=0 WHERE username=?", (int(active), username))
    audit(who, "USER_ACTIVATED" if active else "USER_DEACTIVATED", username)


def unlock(username, who):
    run("UPDATE users SET failed_attempts=0 WHERE username=?", (username,))
    audit(who, "USER_UNLOCKED", username)


def authenticate(username, password, role):
    """Return (user_dict, None) on success or (None, error_message)."""
    uname = username.strip().lower()
    u = run("SELECT * FROM users WHERE username=?", (uname,), "one")
    if not u:
        audit(uname or "-", "LOGIN_FAILED", f"unknown user, {role} portal")
        return None, "Invalid username or password."
    if not u["active"]:
        return None, "This account is deactivated. Please contact your administrator."
    if u["failed_attempts"] >= MAX_FAILED:
        return None, "Account locked after too many failed attempts. Ask an administrator to unlock it."
    if not hmac.compare_digest(hash_pw(password, u["salt"]), u["pw_hash"]):
        run("UPDATE users SET failed_attempts=failed_attempts+1 WHERE id=?", (u["id"],))
        audit(uname, "LOGIN_FAILED", f"bad password, {role} portal")
        left = MAX_FAILED - (u["failed_attempts"] + 1)
        return None, "Invalid username or password." + (f" {left} attempt(s) left." if left <= 2 else "")
    if u["role"] != role:
        audit(uname, "LOGIN_DENIED", f"{u['role']} tried the {role} portal")
        return None, f"This account does not have access to the {role} portal."
    run("UPDATE users SET failed_attempts=0, last_login=? WHERE id=?", (now(), u["id"]))
    audit(uname, "LOGIN", f"{role} portal")
    return u, None


# ---------- assessments ----------
def save_assessment(rec: dict) -> int:
    return run("""INSERT INTO assessments(created_at,officer,officer_name,branch,applicant_name,purpose,
        loan_amount,probability,rf_prob,gb_prob,recommendation,risk_level,mode,applicant_json,status,
        final_decision) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
               (rec.get("created_at", now()), rec["officer"], rec["officer_name"], rec["branch"],
                rec["name"], rec["purpose"], rec["loan_amount"], rec["p"], rec["rf"], rec["gb"],
                rec["rec"], rec["level"], rec["mode"], json.dumps(rec["applicant"]), rec["status"],
                rec.get("final_decision", "")))


def _decorate(r):
    c = r["created_at"]
    r["ref"] = f"LA-{c[2:4]}{c[5:7]}{c[8:10]}-{r['id']:04d}"
    return r


def get_assessments(officer=None, branch=None, status=None):
    sql, params = "SELECT * FROM assessments WHERE 1=1", []
    for col, val in (("officer", officer), ("branch", branch), ("status", status)):
        if val:
            sql += f" AND {col}=?"
            params.append(val)
    return [_decorate(r) for r in run(sql + " ORDER BY id DESC", tuple(params), "all")]


def review(aid, decision, comment, reviewer, override):
    status = "Approved by manager" if decision == "Approve" else "Rejected by manager"
    run("UPDATE assessments SET status=?, final_decision=?, reviewer=?, review_comment=?, reviewed_at=?, "
        "override=? WHERE id=?", (status, decision, reviewer, comment, now(), int(override), aid))
    audit(reviewer, "REVIEW_" + decision.upper(), f"assessment {aid}" + (" (override)" if override else ""))
