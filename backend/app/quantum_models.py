"""Квантовые модели оценки компьютерной (CPU) модели и кибербезопасности.

Здесь реализованы «квантово-вдохновлённые» модели, описывающие вычислительную
модель компьютера и безопасность через квантовые аналогии:

1. QubitThreatState — состояние угрозы как суперпозиция |safe⟩ / |threat⟩;
   вероятность обнаружения |β|² вычисляется из амплитуд, нормированных по Борну.
2. HadamardDiffusion — диффузия атакующего кода по ветвлениям предсказания
   (аналог квантового блуждания на графе состояний процессора).
3. QuantumAnnealingScorer — оценка уязвимости CPU-модели как поиск минимума
   энергии гамильтониана H = A·E_security + B·E_performance (QUBO-формулировка).
4. NoCloningAudit — проверка целостности образа системы: теорема о запрете
   клонирования как модель недопустимого копирования защищённых данных.
5. ZenoEffectMitigation — «квантовый зенов эффект»: частые аудиты замедляют
   эскалацию привилегий (переходы между состояниями модели «замораживаются»).

Эти функции детерминированы (seed), поэтому результаты воспроизводимы.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# --------------------------------------------------------------------- базовая qubit-модель
@dataclass
class QubitThreatState:
    """Суперпозиция состояний безопасности машины: |ψ⟩ = α|safe⟩ + β|threat⟩."""

    alpha: float = 1.0
    beta: float = 0.0

    def normalized(self) -> "QubitThreatState":
        norm = math.hypot(self.alpha, self.beta) or 1.0
        return QubitThreatState(self.alpha / norm, self.beta / norm)

    @property
    def p_threat(self) -> float:
        """Вероятность коллапса в |threat⟩ по правилу Борна."""
        s = self.normalized()
        return s.beta ** 2

    def rotate(self, theta_rad: float) -> "QubitThreatState":
        """Поворот вектора состояния (аналог гейта Ry)."""
        c, s = math.cos(theta_rad / 2), math.sin(theta_rad / 2)
        return QubitThreatState(c * self.alpha - s * self.beta, s * self.alpha + c * self.beta)


def threat_state_from_severity(severity: int, confidence: float) -> QubitThreatState:
    """Карта: severity(1..10) + confidence(0..1) → амплитуды суперпозиции."""
    sev = max(1, min(10, int(severity))) / 10.0
    conf = max(0.0, min(1.0, float(confidence)))
    beta = math.sqrt(sev * conf)
    alpha = math.sqrt(max(0.0, 1.0 - beta ** 2))
    return QubitThreatState(alpha, beta)


# --------------------------------------------------------------------- квантовое блуждание
def hadamard_diffusion(depth: int = 6, seed: int = 42) -> dict:
    """Квантовое блуждание по конвейеру CPU с H-гейтом на каждом такте.

    Моделирует распространение вредоносной нагрузки по стадиям конвейера:
    амплитуда «заражения» растёт как числоPaths*C( depth )/2^depth.
    """
    rng = random.Random(seed)
    amplitudes = [1.0] + [0.0] * depth
    for step in range(depth):
        nxt = [0.0] * (depth + 1)
        for i, a in enumerate(amplitudes):
            if a == 0.0 and i > step:
                continue
            noise = rng.uniform(-0.05, 0.05)
            nxt[i] += a / math.sqrt(2) + noise
            nxt[min(i + 1, depth)] += a / math.sqrt(2) - noise
        amplitudes = nxt
    probs = [a * a for a in amplitudes]
    total = sum(probs) or 1.0
    probs = [p / total for p in probs]
    spread = sum(p * i for i, p in enumerate(probs)) / depth
    return {"probabilities": [round(p, 4) for p in probs],
            "pipeline_spread": round(spread, 4)}


# --------------------------------------------------------------------- QUBO-оценка уязвимости
def quantum_annealing_score(features: dict, seed: int = 7, sweeps: int = 200) -> dict:
    """Минимизация псевдо-гамильтониана методом имитации отжига.

    features: {"open_ports": n, "unpatched_cves": n, "cpu_microarch_risk": 0..1,
               "encryption_coverage": 0..1, "audit_frequency": n}
    Возвращает энергию основного состояния и вектор конфигурации защиты.
    """
    rng = random.Random(seed)
    ports = float(features.get("open_ports", 5))
    cves = float(features.get("unpatched_cves", 10))
    uarch = float(features.get("cpu_microarch_risk", 0.5))
    enc = float(features.get("encryption_coverage", 0.5))
    audit = float(features.get("audit_frequency", 1))

    # квадратичная форма (QUBO): x = [close_ports, patch, harden_uarch, encrypt, audit_up]
    def energy(x: list[int]) -> float:
        e = 0.0
        e += 3.0 * ports * (1 - x[0])          # штраф за открытые порты
        e += 4.0 * cves * (1 - x[1])           # штраф незакрытых CVE
        e += 5.0 * uarch * (1 - x[2])          # риск микроархитектуры (Spectre/Meltdown-класс)
        e -= 2.5 * enc * x[3]                  # выигрыш от шифрования
        e -= 1.5 * min(audit, 10) / 10 * x[4]  # выигрыш от частых аудитов (зенов эффект)
        e += 1.2 * x[0] * x[1]                 # синергия патчей и сегментации
        e -= 0.8 * x[2] * x[3]                 # усиление + шифрование работают вместе
        return e

    best_x = [0] * 5
    best_e = energy(best_x)
    temp = 10.0
    cur = best_x[:]
    for it in range(sweeps):
        temp *= 0.97
        j = rng.randrange(5)
        trial = cur[:]
        trial[j] ^= 1
        e = energy(trial)
        if e <= energy(cur) or rng.random() < math.exp(-(e - energy(cur)) / max(temp, 1e-6)):
            cur = trial
        if e < best_e:
            best_e, best_x = e, trial[:]
    vulnerability = max(0.0, min(1.0, (best_e + 12.0) / 25.0))
    return {
        "ground_energy": round(best_e, 3),
        "configuration": {
            "segment_network": bool(best_x[0]),
            "apply_patches": bool(best_x[1]),
            "microcode_hardening": bool(best_x[2]),
            "full_disk_encryption": bool(best_x[3]),
            "continuous_audit": bool(best_x[4]),
        },
        "quantum_vulnerability": round(vulnerability, 4),
    }


# --------------------------------------------------------------------- запрет клонирования
def no_cloning_audit(image_hash: str, reference_hash: str) -> dict:
    """Модель «no-cloning theorem»: точная копия защищённого образа невозможна.

    Сходство хэшей оценивается побитово — любое несанкционированное клонирование
    или подмена образа обнаруживаются как разрушение «квантового» состояния данных.
    """
    try:
        a = int(image_hash, 16)
        b = int(reference_hash, 16)
        diff_bits = bin(a ^ b).count("1")
        bits = max(len(bin(a)), len(bin(b)), 1)
        fidelity = 1.0 - diff_bits / bits
    except (ValueError, TypeError):
        fidelity = 0.0
    return {"fidelity": round(fidelity, 4),
            "clone_detected": fidelity < 0.999,
            "verdict": "integrity_ok" if fidelity >= 0.999 else "possible_clone_or_tamper"}


# --------------------------------------------------------------------- зенов эффект
def zeno_effect_mitigation(base_escalation_rate: float, audits_per_hour: int) -> dict:
    """Частые измерения (аудиты) подавляют переход системы в скомпрометированное состояние.

    P(no escalation | t, n измерений) = (1 - λ(t/n)²)^n → exp(-λ t²/n) → 1 при n→∞.
    """
    lam = max(float(base_escalation_rate), 1e-9)
    n = max(int(audits_per_hour), 0)
    t = 1.0  # час
    if n == 0:
        survival = math.exp(-lam * t)
    else:
        step = t / n
        survival = (1.0 - lam * step ** 2) ** n
    return {"survival_probability": round(max(0.0, min(1.0, survival)), 4),
            "mitigation_gain": round(max(0.0, min(1.0, survival - math.exp(-lam * t))), 4)}


# --------------------------------------------------------------------- сводный скоринг
def compute_quantum_profile(threat: dict) -> dict:
    """Полный квантовый профиль артефакта угрозы для конкретной CPU-модели."""
    state = threat_state_from_severity(threat.get("severity", 5),
                                       threat.get("confidence", 0.5))
    walk = hadamard_diffusion(depth=6, seed=int(threat.get("id", 1) or 1))
    anneal = quantum_annealing_score({
        "open_ports": threat.get("open_ports", 4),
        "unpatched_cves": threat.get("unpatched_cves", 8),
        "cpu_microarch_risk": min(1.0, threat.get("severity", 5) / 10.0),
        "encryption_coverage": threat.get("encryption_coverage", 0.5),
        "audit_frequency": threat.get("audit_frequency", 2),
    }, seed=int(threat.get("id", 1) or 1))
    zeno = zeno_effect_mitigation(threat.get("severity", 5) / 20.0,
                                  threat.get("audit_frequency", 2))
    return {
        "qubit_state": {"alpha": round(state.alpha, 4), "beta": round(state.beta, 4),
                        "p_threat": round(state.p_threat, 4)},
        "diffusion": walk,
        "annealing": anneal,
        "zeno": zeno,
        "composite_score": round(
            0.45 * state.p_threat
            + 0.35 * anneal["quantum_vulnerability"]
            + 0.20 * (1.0 - zeno["survival_probability"]), 4),
    }
