"""Справочник MITRE ATT&CK и описание вычислительной (компьютерной) модели CPU.

computer_model() — формальное описание машины Тьюринга-класса, реализуемой
процессором (конвейер, состояния, микроархитектурные каналы атак).
ATTACK_KNOWLEDGE — сокращённая база тактик/техник MITRE ATT&CK Enterprise,
связанных с уровнями модели компьютера.
"""
from __future__ import annotations


# --------------------------------------------------------------------- компьютерная модель
def computer_model(cpu_name: str = "Intel Core i7 (Skylake-class)") -> dict:
    """Описание компьютерной модели через квантовые аналогии.

    Классическая вычислительная модель (машина с конвейером состояний)
    дополняется квантовым представлением: регистры процессора трактуются как
    кубиты в базисном состоянии, такты конвейера — как унитарные гейты,
    а побочные каналы (Spectre/Meltdown) — как «измерения», нарушающие
    суперпозицию защищённых состояний.
    """
    return {
        "cpu": cpu_name,
        "classical_model": {
            "type": "Конвейерная машина фон Неймана с предсказанием переходов",
            "stages": ["fetch", "decode", "execute", "memory", "writeback"],
            "pipeline_depth": 14,
            "word_size_bits": 64,
            "address_space": "2^48 (4-уровневая пагинация)",
            "states": "Каждый такт — детерминированный переход S_t = U(S_{t-1})",
            "isa": "x86-64",
        },
        "quantum_analogy": {
            "register_as_qubit": "64-битный регистр ↔ система из 64 кубитов в базисном состоянии |b0…b63⟩",
            "clock_cycle_as_gate": "Такт конвейера ≈ унитарный гейт U_t; программа = последовательность U_T…U_1|ψ0⟩",
            "branch_prediction_as_superposition":
                "Предсказатель ветвей поддерживает суперпозицию путей исполнения до «коллапса» "
                "(подтверждения/опровержения условия перехода)",
            "speculative_execution":
                "Спекулятивное исполнение вычисляет амплитуды обоих исходов if-ветвления; "
                "откат (squash) аналогичен обратному унитарному преобразованию",
            "side_channel_as_measurement":
                "Атаки Spectre/Meltdown используют «измерение» (тайминг кэша), которое "
                "необратимо раскрывает амплитуды защищённых состояний — нарушение модели "
                "изолированных подпространств (user/kernel)",
            "no_cloning": "Копирование секретных регистровых состояний ядром запрещено политикой "
                          "изоляции — аналог теоремы о запрете клонирования",
        },
        "security_layers": [
            {"layer": "Микроархитектура", "risks": ["Spectre (T1595/T1068 side-channel)",
                                                     "Meltdown", "MDS/ZombieLoad", "Downfall"],
             "mitigations": ["microcode updates", "KPTI", "SSBD", "Retpoline"]},
            {"layer": "ОС / изоляция", "risks": ["эскалация привилегий", "инъекции в адресное пространство"],
             "mitigations": ["ASLR", "SMEP/SMAP", "CET", "VBS/HVCI"]},
            {"layer": "Программы", "risks": ["RCE", "live-атаки на память процесса"],
             "mitigations": ["DEP/NX", "stack canaries", "sandboxing"]},
        ],
    }


# --------------------------------------------------------------------- MITRE ATT&CK (сокращённо)
ATTACK_KNOWLEDGE = {
    "matrix_version": "Enterprise ATT&CK v14 (сокращённый справочник)",
    "tactics": [
        {"id": "TA0001", "name": "Initial Access", "ru": "Начальный доступ"},
        {"id": "TA0002", "name": "Execution", "ru": "Исполнение"},
        {"id": "TA0003", "name": "Persistence", "ru": "Закрепление"},
        {"id": "TA0004", "name": "Privilege Escalation", "ru": "Повышение привилегий"},
        {"id": "TA0005", "name": "Defense Evasion", "ru": "Обход защиты"},
        {"id": "TA0006", "name": "Credential Access", "ru": "Доступ к учётным данным"},
        {"id": "TA0007", "name": "Discovery", "ru": "Разведка"},
        {"id": "TA0008", "name": "Lateral Movement", "ru": "Боковое перемещение"},
        {"id": "TA0009", "name": "Collection", "ru": "Сбор"},
        {"id": "TA0010", "name": "Exfiltration", "ru": "Экфильтрация"},
        {"id": "TA0040", "name": "Impact", "ru": "Воздействие"},
    ],
    "techniques": [
        {"id": "T1059.001", "tactic": "TA0002", "name": "PowerShell",
         "cpu_level": "Программы", "quantum_note": "исполнение нового «гейта» в пользовательском подпространстве"},
        {"id": "T1055.002", "tactic": "TA0005", "name": "NTDLL DLL Injection",
         "cpu_level": "ОС / изоляция", "quantum_note": "внедрение амплитуд в чужое состояние процесса"},
        {"id": "T1068", "tactic": "TA0004", "name": "Exploitation for Privilege Escalation",
         "cpu_level": "ОС / изоляция", "quantum_note": "переход |user⟩ → |kernel⟩ мимо барьера измерения"},
        {"id": "T1595.002", "tactic": "TA0001", "name": "Vulnerability Scanning",
         "cpu_level": "Микроархитектура", "quantum_note": "зондирование состояний конвейера"},
        {"id": "T1611", "tactic": "TA0005", "name": "Escape to Host / Container Breakout",
         "cpu_level": "Микроархитектура", "quantum_note": "нарушение изоляции виртуальных подпространств"},
        {"id": "T1610", "tactic": "TA0001", "name": "Deploy Container",
         "cpu_level": "Программы", "quantum_note": "подготовка среды для «квантового блуждания» кода"},
        {"id": "T1053.005", "tactic": "TA0003", "name": "Scheduled Task",
         "cpu_level": "Программы", "quantum_note": "периодическое «измерение» планировщиком"},
        {"id": "T1562.001", "tactic": "TA0005", "name": "Disable or Modify Tools",
         "cpu_level": "ОС / изоляция", "quantum_note": "подавление аудит-измерений (снятие зенов эффекта)"},
        {"id": "T1003.001", "tactic": "TA0006", "name": "LSASS Memory",
         "cpu_level": "Программы", "quantum_note": "чтение защищённых «регистровых» состояний"},
        {"id": "T1041", "tactic": "TA0010", "name": "Exfiltration Over C2 Channel",
         "cpu_level": "Программы", "quantum_note": "перенос амплитуд вовне по управляемому каналу"},
        {"id": "T1489", "tactic": "TA0040", "name": "Service Stop",
         "cpu_level": "ОС / изоляция", "quantum_note": "принудительный коллапс рабочих состояний"},
        {"id": "hardware.cpu.sidechannel", "tactic": "TA0007", "name": "CPU Side-Channel (Spectre-class)",
         "cpu_level": "Микроархитектура",
         "quantum_note": "тайминговое измерение кэша раскрывает спекулятивные амплитуды"},
    ],
}


def techniques_for(threat: dict) -> list[dict]:
    """Подбор техник ATT&CK по тегам артефакта."""
    tags = {t.lower() for t in threat.get("tags", [])}
    out = []
    for tech in ATTACK_KNOWLEDGE["techniques"]:
        blob = f"{tech['name']} {tech['id']} {tech['cpu_level']}".lower()
        if any(tag in blob for tag in tags):
            out.append(tech)
    if not out and any(x in blob_all(threat) for x in ("cpu", "spectre", "meltdown")):
        out = [t for t in ATTACK_KNOWLEDGE["techniques"] if t["cpu_level"] == "Микроархитектура"]
    return out


def blob_all(threat: dict) -> str:
    return f"{threat.get('name','')} {threat.get('description','')} {' '.join(threat.get('tags', []))}".lower()
