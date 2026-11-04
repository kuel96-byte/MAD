"""Проверка реализации ROUGE и вспомогательных функций на примерах, посчитанных вручную.

Запуск: python -m pytest tests  (или python tests/test_metrics.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from metrics import greedy_oracle, lcs_length, rouge_l, rouge_n, select_top_k, summary_length  # noqa: E402


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_rouge1_cyrillic():
    cand = "кошка сидит на ковре".split()
    ref = "кошка лежит на ковре у окна".split()
    p, r, f = rouge_n(cand, ref, 1)          # совпадения: кошка, на, ковре = 3
    assert close(p, 3 / 4) and close(r, 3 / 6) and close(f, 2 * 0.75 * 0.5 / 1.25)


def test_rouge1_clipping():
    p, r, f = rouge_n("да да да".split(), "да нет".split(), 1)   # «да» засчитывается 1 раз
    assert close(p, 1 / 3) and close(r, 1 / 2)


def test_rouge2():
    cand = "кошка сидит на ковре".split()
    ref = "кошка сидит на полу".split()
    p, r, f = rouge_n(cand, ref, 2)          # биграммы: (кошка сидит), (сидит на) = 2 из 3
    assert close(p, 2 / 3) and close(r, 2 / 3) and close(f, 2 / 3)


def test_lcs_and_rouge_l():
    assert lcs_length("a b c d e".split(), "a c e".split()) == 3
    assert lcs_length([], [1, 2]) == 0
    p, r, f = rouge_l("мама мыла раму".split(), "мама раму мыла".split())   # LCS = 2
    assert close(p, 2 / 3) and close(r, 2 / 3)


def test_integer_tokens_equal_strings():
    a, b = "в лесу родилась ёлочка".split(), "в лесу она росла".split()
    vocab = {w: i for i, w in enumerate(sorted(set(a + b)))}
    assert rouge_n(a, b, 1) == rouge_n([vocab[w] for w in a], [vocab[w] for w in b], 1)


def test_zero_overlap():
    assert rouge_n(["а"], ["б"], 1) == (0.0, 0.0, 0.0)


def test_select_top_k_order_and_ties():
    s = np.array([0.1, 0.9, 0.5, 0.9])
    assert select_top_k(s, 2).tolist() == [1, 3]       # исходный порядок
    assert select_top_k(np.zeros(4), 2).tolist() == [0, 1]   # при равенстве — ранние


def test_summary_length():
    assert summary_length(30, 0.1) == 3                 # без ошибки округления 0,1·30
    assert summary_length(35, 0.1) == 4
    assert summary_length(5, 0.05) == 1


def test_greedy_oracle():
    sents = ["солнце светит ярко".split(), "дождь идёт".split(), "ветер дует сильно".split()]
    ref = "солнце светит и ветер дует".split()
    assert greedy_oracle(sents, ref, 3) == [0, 2]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
