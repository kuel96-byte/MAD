"""Pydantic-схемы запросов/ответов API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LoginRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=4, max_length=128)


class RegisterRequest(LoginRequest):
    role: str = "viewer"

    @field_validator("role")
    @classmethod
    def _role(cls, v: str) -> str:
        if v not in ("viewer", "analyst", "admin"):
            raise ValueError("role must be viewer|analyst|admin")
        return v


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    username: str
    role: str


class Indicator(BaseModel):
    type: str = "ipv4"     # ipv4/domain/url/sha256/email
    value: str


class ThreatCreate(BaseModel):
    name: str = Field(min_length=2, max_length=200)
    source: str = "manual"
    severity: int = Field(5, ge=1, le=10)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    first_seen: Optional[str] = None
    description: str = ""
    tags: list[str] = []
    indicators: list[Indicator] = []
    techniques: list[str] = []
    open_ports: int = Field(4, ge=0, le=100)
    unpatched_cves: int = Field(8, ge=0, le=1000)
    encryption_coverage: float = Field(0.5, ge=0.0, le=1.0)
    audit_frequency: int = Field(2, ge=0, le=100)


class ThreatOut(ThreatCreate):
    id: int
    quantum_score: Optional[float] = None
    created_at: str = ""

    class Config:
        from_attributes = True


class QuantumProfileRequest(BaseModel):
    severity: int = Field(5, ge=1, le=10)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    open_ports: int = 4
    unpatched_cves: int = 8
    encryption_coverage: float = 0.5
    audit_frequency: int = 2
    cpu_name: str = "Intel Core i7 (Skylake-class)"


class NoCloningRequest(BaseModel):
    image_hash: str
    reference_hash: str
