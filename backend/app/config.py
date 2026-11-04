"""Настройки бэкенда TI/ATT&CK/Quantum-Security платформы."""
from __future__ import annotations

import os
from pathlib import Path

# JWT
JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-secret-please-change-me-0123456789abcdef")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_TTL_MIN = int(os.getenv("ACCESS_TOKEN_TTL_MIN", "120"))

# Хранилище
DB_PATH = os.getenv("DB_PATH", "/data/platform.db")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# CORS (адрес фронтенда)
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://frontend:3000"
).split(",") if o.strip()]

# Роли
ROLE_ADMIN = "admin"
ROLE_ANALYST = "analyst"
ROLE_VIEWER = "viewer"
