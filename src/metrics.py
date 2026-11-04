"""Метрики качества аннотаций: ROUGE-1, ROUGE-2, ROUGE-L и F1 классификации предложений.

ROUGE реализован самостоятельно (Lin, 2004) и работает с любыми токенами,
в том числе с кириллицей и с целочисленными идентификаторами слов. Стандартный
токенизатор пакета rouge-score отбрасывает символы вне [a-z0-9] и для русского
текста даёт неверные нулевые значения, поэтому здесь он не используется.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def ngram_counts(tokens: Sequence, n: int) -> Counter:
    """Мультимножество n-грамм последовательности токенов."""
    if n == 1:
        return Counter(tokens)
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def prf(overlap: float, cand_total: float, ref_total: float) -> tuple[float, float, float]:
    """Точность P, полнота R и F-мера (β = 1) по числу совпадений."""
    p = overlap / cand_total if cand_total > 0 else 0.0
    r = overlap / ref_total if ref_total > 0 else 0.0
    f = 2.0 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def rouge_n(cand: Sequence, ref: Sequence, n: int = 1) -> tuple[float, float, float]:
    """ROUGE-N: совпадения n-грамм с отсечением по числу вхождений в эталон."""
    c, r = ngram_counts(cand, n), ngram_counts(ref, n)
    overlap = sum((c & r).values())
    return prf(overlap, sum(c.values()), sum(r.values()))


def rouge_n_from_counts(c: Counter, r: Counter) -> float:
    """F-мера ROUGE-N по готовым мультимножествам n-грамм."""
    return prf(sum((c & r).values()), sum(c.values()), sum(r.values()))[2]


def lcs_length_dp(a: Sequence, b: Sequence) -> int:
    """Длина наибольшей общей подпоследовательности (классическое динамическое программирование)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def lcs_length(a: Sequence, b: Sequence) -> int:
    """Длина LCS бит-параллельным алгоритмом (Allison, Dix, 1986; Hyyrö, 2004).

    Столбец таблицы динамического программирования по b хранится битами целого
    числа V; переход по символу x: V ← ((V + U) | (V − U)), U = V & M[x], где M[x] —
    маска позиций x в b. Длина LCS равна числу нулевых битов V. Результат совпадает
    с lcs_length_dp, но вычисляется в десятки раз быстрее.
    """
    if not a or not b:
        return 0
    masks: dict = {}
    for i, y in enumerate(b):
        masks[y] = masks.get(y, 0) | (1 << i)
    full = (1 << len(b)) - 1
    V = full
    for x in a:
        U = V & masks.get(x, 0)
        V = ((V + U) | (V - U)) & full
    return len(b) - bin(V).count("1")


def rouge_l(cand: Sequence, ref: Sequence) -> tuple[float, float, float]:
    """ROUGE-L: P, R, F по длине LCS аннотации и эталона (как rougeL в rouge-score)."""
    return prf(lcs_length(cand, ref), len(cand), len(ref))


def greedy_oracle(sent_tokens: list[Sequence], ref: Sequence, max_sents: int) -> list[int]:
    """Эталонная разметка предложений жадным отбором (Nallapati et al., 2017).

    На каждом шаге добавляется предложение, сильнее всего увеличивающее
    ROUGE-1 F + ROUGE-2 F аннотации (предложения склеиваются в исходном порядке);
    отбор прекращается, когда прироста нет или набрано max_sents предложений.
    Возвращает отсортированные индексы выбранных предложений.
    """
    ref1, ref2 = ngram_counts(ref, 1), ngram_counts(ref, 2)
    selected: list[int] = []
    best = 0.0
    while len(selected) < max_sents:
        best_i, best_score = -1, best
        for i in range(len(sent_tokens)):
            if i in selected:
                continue
            cand = [t for j in sorted(selected + [i]) for t in sent_tokens[j]]
            score = (rouge_n_from_counts(ngram_counts(cand, 1), ref1)
                     + rouge_n_from_counts(ngram_counts(cand, 2), ref2))
            if score > best_score:
                best_i, best_score = i, score
        if best_i < 0:
            break
        selected.append(best_i)
        best = best_score
    return sorted(selected)


def select_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Индексы k предложений с наибольшими оценками, в исходном порядке.

    При равных оценках предпочтение отдаётся более раннему предложению.
    """
    k = max(1, min(k, len(scores)))
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores, dtype=float)))
    return np.sort(order[:k])


def summary_length(n_sentences: int, r: float) -> int:
    """Число предложений аннотации k = ⌈r·n⌉ (не менее одного)."""
    return max(1, int(np.ceil(r * n_sentences - 1e-9)))
