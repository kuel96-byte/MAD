"""FastAPI: Threat Intelligence + MITRE ATT&CK + квантовые модели CPU/безопасности.

Запуск (dev): uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Docker:       см. docker-compose.yml (сервис backend)
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import mitre_attack, quantum_models, storage
from app.config import ACCESS_TOKEN_TTL_MIN, CORS_ORIGINS, JWT_ALGORITHM, JWT_SECRET
from app.schemas import (LoginRequest, NoCloningRequest, QuantumProfileRequest,
                         RegisterRequest, ThreatCreate, TokenResponse)

app = FastAPI(
    title="TI / ATT&CK / Quantum-Security Platform",
    version="1.0.0",
    description=(
        "Платформа threat-разведки: управление индикаторами компрометации, карта MITRE ATT&CK, "
        "описание компьютерной модели через квантовые модели CPU и оценка компьютерной безопасности."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer = HTTPBearer(auto_error=False)


@app.on_event("startup")
def _startup() -> None:
    init_app()


def init_app() -> None:
    """Инициализация БД и демо-данных. Вызывается на старте и в тестах явно."""
    storage.init_db()
    seed_demo_threats()


# ------------------------------------------------------------------ авторизация
def create_token(username: str, role: str) -> tuple[str, int]:
    expires = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_TTL_MIN)
    payload = {"sub": username, "role": role, "exp": expires}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM), ACCESS_TOKEN_TTL_MIN * 60


def current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> dict:
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Не авторизован",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        data = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Срок токена истёк")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Некорректный токен")
    return {"username": data["sub"], "role": data.get("role", "viewer")}


def require_roles(*roles: str):
    def dep(user: dict = Depends(current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Недостаточно прав")
        return user
    return dep


@app.post("/api/auth/login", response_model=TokenResponse)
def login(body: LoginRequest):
    conn = storage.get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (body.username,)).fetchone()
    conn.close()
    if row is None or not storage.verify_password(body.password, row["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Неверный логин или пароль")
    token, ttl = create_token(row["username"], row["role"])
    return TokenResponse(access_token=token, expires_in=ttl,
                         username=row["username"], role=row["role"])


@app.post("/api/auth/register", status_code=201)
def register(body: RegisterRequest, user: dict = Depends(require_roles("admin"))):
    conn = storage.get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
            (body.username, storage.hash_password(body.password), body.role, storage._now()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status.HTTP_409_CONFLICT, "Пользователь уже существует")
    conn.close()
    return {"ok": True, "username": body.username, "role": body.role}


@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)):
    return user


# ------------------------------------------------------------------ threat intelligence
def _row_to_threat(r: sqlite3.Row) -> dict:
    d = dict(r)
    for key in ("tags", "indicators", "techniques"):
        d[key] = json.loads(d.get(key) or "[]")
    return d


@app.get("/api/threats")
def list_threats(user: dict = Depends(current_user)):
    conn = storage.get_conn()
    rows = conn.execute("SELECT * FROM threats ORDER BY severity DESC, id DESC").fetchall()
    conn.close()
    return [_row_to_threat(r) for r in rows]


@app.get("/api/threats/{threat_id}")
def get_threat(threat_id: int, user: dict = Depends(current_user)):
    conn = storage.get_conn()
    row = conn.execute("SELECT * FROM threats WHERE id=?", (threat_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Артефакт не найден")
    threat = _row_to_threat(row)
    profile = quantum_models.compute_quantum_profile(threat)
    threat["quantum_profile"] = profile
    threat["attack_techniques"] = mitre_attack.techniques_for(threat)
    return threat


@app.post("/api/threats", status_code=201)
def create_threat(body: ThreatCreate,
                  user: dict = Depends(require_roles("admin", "analyst"))):
    data = body.model_dump()
    profile = quantum_models.compute_quantum_profile({**data, "id": 1})
    conn = storage.get_conn()
    cur = conn.execute(
        """INSERT INTO threats (name, source, severity, confidence, first_seen, description,
                                tags, indicators, techniques, quantum_score, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (data.pop("name"), data.pop("source"), data.pop("severity"), data.pop("confidence"),
         data.pop("first_seen"), data.pop("description"),
         json.dumps(data.pop("tags"), ensure_ascii=False),
         json.dumps([i.model_dump() for i in data.pop("indicators")], ensure_ascii=False),
         json.dumps(data.pop("techniques"), ensure_ascii=False),
         profile["composite_score"], storage._now()),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return {"ok": True, "id": new_id, "quantum_profile": profile}


@app.delete("/api/threats/{threat_id}", status_code=204)
def delete_threat(threat_id: int, user: dict = Depends(require_roles("admin"))):
    conn = storage.get_conn()
    conn.execute("DELETE FROM threats WHERE id=?", (threat_id,))
    conn.commit()
    conn.close()


# ------------------------------------------------------------------ MITRE ATT&CK и модель CPU
@app.get("/api/mitre/matrix")
def mitre_matrix(user: dict = Depends(current_user)):
    return mitre_attack.ATTACK_KNOWLEDGE


@app.get("/api/computer-model")
def computer_model(cpu_name: str = "Intel Core i7 (Skylake-class)",
                   user: dict = Depends(current_user)):
    """Описание компьютерной модели через квантовые модели CPU."""
    model = mitre_attack.computer_model(cpu_name)
    model["quantum_pipeline_walk"] = quantum_models.hadamard_diffusion(depth=14, seed=42)
    return model


# ------------------------------------------------------------------ квантовые модели безопасности
@app.post("/api/quantum/profile")
def quantum_profile(body: QuantumProfileRequest, user: dict = Depends(current_user)):
    threat = {"id": 1, "severity": body.severity, "confidence": body.confidence,
              "open_ports": body.open_ports, "unpatched_cves": body.unpatched_cves,
              "encryption_coverage": body.encryption_coverage,
              "audit_frequency": body.audit_frequency}
    profile = quantum_models.compute_quantum_profile(threat)
    profile["computer_model"] = mitre_attack.computer_model(body.cpu_name)
    return profile


@app.post("/api/quantum/no-cloning")
def no_cloning(body: NoCloningRequest, user: dict = Depends(current_user)):
    return quantum_models.no_cloning_audit(body.image_hash, body.reference_hash)


@app.get("/api/stats")
def stats(user: dict = Depends(current_user)):
    conn = storage.get_conn()
    total = conn.execute("SELECT COUNT(*) c FROM threats").fetchone()["c"]
    high = conn.execute("SELECT COUNT(*) c FROM threats WHERE severity >= 8").fetchone()["c"]
    avg = conn.execute("SELECT AVG(quantum_score) a FROM threats").fetchone()["a"]
    conn.close()
    return {"total_threats": total, "high_severity": high,
            "avg_quantum_score": round(avg or 0.0, 4)}


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ------------------------------------------------------------------ демо-данные
def seed_demo_threats() -> None:
    conn = storage.get_conn()
    if conn.execute("SELECT COUNT(*) c FROM threats").fetchone()["c"] > 0:
        conn.close()
        return
    demos = [
        dict(name="Lazarus — APT37 loader", source="MISP", severity=9, confidence=0.85,
             first_seen="2026-08-14", description="Вредоносный загрузчик с DLL-инъекцией",
             tags=["dll injection", "powershell"],
             indicators=[{"type": "sha256", "value": "9f2b…c1"}, {"type": "ipv4", "value": "185.220.101.4"}],
             techniques=["T1055.002", "T1059.001"]),
        dict(name="Spectre-var CPU side-channel", source="internal audit", severity=8, confidence=0.7,
             first_seen="2026-09-02", description="Эксплуатация спекулятивного исполнения (кэш-тайминг)",
             tags=["cpu", "side channel"],
             indicators=[{"type": "domain", "value": "probe.example"}],
             techniques=["hardware.cpu.sidechannel"]),
        dict(name="Container breakout attempt", source="Falco", severity=7, confidence=0.6,
             first_seen="2026-10-11", description="Попытка побега из контейнера в docker-сети платформы",
             tags=["container breakout", "privilege escalation"],
             indicators=[{"type": "url", "value": "http://c2.example/beacon"}],
             techniques=["T1611", "T1068"]),
        dict(name="Credential theft — LSASS dump", source="EDR", severity=9, confidence=0.9,
             first_seen="2026-10-28", description="Дамп памяти lsass.exe для кражи хэшей",
             tags=["lsass memory"],
             indicators=[{"type": "sha256", "value": "aa11…99"}],
             techniques=["T1003.001", "T1562.001"]),
    ]
    for d in demos:
        prof = quantum_models.compute_quantum_profile({**d, "id": 1})
        conn.execute(
            """INSERT INTO threats (name, source, severity, confidence, first_seen, description,
                                    tags, indicators, techniques, quantum_score, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (d["name"], d["source"], d["severity"], d["confidence"], d["first_seen"],
             d["description"], json.dumps(d["tags"], ensure_ascii=False),
             json.dumps(d["indicators"], ensure_ascii=False),
             json.dumps(d["techniques"], ensure_ascii=False),
             prof["composite_score"], storage._now()),
        )
    conn.commit()
    conn.close()
