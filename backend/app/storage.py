"""SQLite-хранилище: пользователи, threat-артефакты, оценки MITRE ATT&CK."""
from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.config import DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_password(password: str) -> str:
    salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return f"pbkdf2${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt, digest = stored.split("$")
    except ValueError:
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return check.hex() == digest


def get_conn() -> sqlite3.Connection:
    # DB_PATH перечитывается, чтобы меняться через переменные окружения/тесты
    from app.config import DB_PATH as _cfg_path
    db = os.getenv("DB_PATH", _cfg_path)
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'viewer',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS threats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            source TEXT,
            severity INTEGER DEFAULT 5,          -- 1..10
            confidence REAL DEFAULT 0.5,         -- 0..1
            first_seen TEXT,
            description TEXT,
            tags TEXT,                           -- JSON
            indicators TEXT,                     -- JSON: [{type,value}]
            techniques TEXT,                     -- JSON: ["T1059.001", ...]
            quantum_score REAL,                  -- квантовая оценка уязвимости модели CPU
            created_at TEXT NOT NULL
        );
        """
    )
    # Демо-администратор (меняется переменными окружения при инициализации)
    cur = conn.execute("SELECT COUNT(*) c FROM users")
    if cur.fetchone()["c"] == 0:
        admin = os.getenv("ADMIN_USERNAME", "admin")
        passwd = os.getenv("ADMIN_PASSWORD", "admin12345")
        conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
            (admin, hash_password(passwd), "admin", _now()),
        )
        analyst = os.getenv("ANALYST_USERNAME", "")
        if analyst:
            conn.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
                (analyst, hash_password(os.getenv("ANALYST_PASSWORD", "analyst12345")),
                 "analyst", _now()),
            )
        conn.commit()
    conn.close()
