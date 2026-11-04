"""Модели оценки значимости предложений f(x; θ) и базовые методы.

Обучаемые классификаторы: логистическая регрессия, наивный байесовский
классификатор (мультиномиальный), линейный SVM, случайный лес.
Базовые методы без обучения: Lead-k и TextRank. Дополнительно — «оракул»
(идеальный классификатор, знающий эталонные метки) как верхняя граница.

Все методы возвращают оценку s_i, монотонно связанную с p_i = P(y_i = 1 | x_i);
аннотация составляется из k предложений с наибольшими s_i, поэтому
монотонное преобразование оценки (например, калибровка Платта для SVM)
не меняет результата.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC

log = logging.getLogger(__name__)

MODEL_NAMES = {
    "logreg": "Логистическая регрессия",
    "nb": "Наивный байесовский классификатор",
    "svm": "Линейный SVM",
    "rf": "Случайный лес",
    "lead": "Lead-k",
    "textrank": "TextRank",
    "oracle": "Оракул (верхняя граница)",
}


class SentenceScorer:
    """Обёртка классификатора: масштабирование плотных признаков, обучение, оценка s_i."""

    def __init__(self, name: str, C: float = 1.0, min_samples_leaf: int = 3,
                 n_estimators: int = 200, seed: int = 0, n_jobs: int = 1) -> None:
        self.name = name
        self.C = C
        self.seed = seed
        self.constant = False
        if name == "logreg":
            self.model = LogisticRegression(C=C, solver="liblinear", class_weight="balanced",
                                            max_iter=2000)
            self.scaler = StandardScaler()
        elif name == "svm":
            self.model = LinearSVC(C=C, class_weight="balanced", dual="auto", max_iter=20000,
                                   random_state=seed)
            self.scaler = StandardScaler()
        elif name == "nb":
            # α = 1/C: большее C — слабее сглаживание, т. е. слабее регуляризация
            self.model = MultinomialNB(alpha=1.0 / C)
            self.scaler = MinMaxScaler(clip=True)     # признаки NB должны быть неотрицательны
        elif name == "rf":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators, max_features="sqrt", min_samples_leaf=min_samples_leaf,
                class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed)
            self.scaler = None                        # деревьям масштабирование не нужно
        else:
            raise ValueError(f"неизвестная модель {name}")

    def _design(self, X: sparse.csr_matrix, D: np.ndarray, fit: bool) -> sparse.csr_matrix:
        if self.scaler is not None:
            D = self.scaler.fit_transform(D) if fit else self.scaler.transform(D)
        return sparse.hstack([X, sparse.csr_matrix(D.astype(np.float32))], format="csr")

    def fit(self, X: sparse.csr_matrix, D: np.ndarray, y: np.ndarray) -> "SentenceScorer":
        Z = self._design(X, D, fit=True)
        if len(np.unique(y)) < 2:
            # вырожденная выборка: все оценки равны, отбор сводится к Lead-k
            log.warning("%s: в обучающей выборке один класс, оценки постоянны", self.name)
            self.constant = True
            return self
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.model.fit(Z, y)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            log.warning("%s (C=%g): оптимизатор не сошёлся за max_iter", self.name, self.C)
        return self

    def score(self, X: sparse.csr_matrix, D: np.ndarray) -> np.ndarray:
        if self.constant:
            return np.zeros(X.shape[0])
        Z = self._design(X, D, fit=False)
        if self.name == "logreg":
            return self.model.predict_proba(Z)[:, 1]
        if self.name == "svm":
            return self.model.decision_function(Z)
        if self.name == "nb":
            # логарифм отношения правдоподобий классов: не насыщается, в отличие от p_i
            jll = self.model.predict_joint_log_proba(Z)
            return jll[:, 1] - jll[:, 0]
        return self.model.predict_proba(Z)[:, 1]


def lead_scores(n: int) -> np.ndarray:
    """Lead-k: чем раньше предложение, тем выше оценка."""
    return -np.arange(n, dtype=float)


def textrank_scores(counts: sparse.csr_matrix, damping: float = 0.85,
                    tol: float = 1e-6, max_iter: int = 200) -> np.ndarray:
    """TextRank (Mihalcea, Tarau, 2004) для предложений одного документа.

    Вес ребра w_ij = |S_i ∩ S_j| / (log|S_i| + log|S_j|), где S_i — множество значимых
    слов предложения, |S_i| — число значимых слов в нём; оценки — неподвижная точка
    WS(V_i) = (1 − d) + d · Σ_j w_ji / Σ_k w_jk · WS(V_j).
    """
    n = counts.shape[0]
    if n == 1:
        return np.ones(1)
    B = (counts > 0).astype(np.float64)
    overlap = (B @ B.T).toarray()
    np.fill_diagonal(overlap, 0.0)
    length = np.asarray(counts.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        loglen = np.where(length > 0, np.log(np.maximum(length, 1)), 0.0)
    denom = loglen[:, None] + loglen[None, :]
    W = np.where(denom > 0, overlap / np.where(denom > 0, denom, 1.0), 0.0)
    out = W.sum(axis=1)
    M = np.divide(W, out[:, None], out=np.zeros_like(W), where=out[:, None] > 0)
    ws = np.ones(n)
    for _ in range(max_iter):
        new = (1.0 - damping) + damping * (M.T @ ws)
        if np.max(np.abs(new - ws)) < tol:
            ws = new
            break
        ws = new
    return ws
