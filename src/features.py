"""Построение вектора признаков x_i предложения.

x_i = (TF-IDF-вектор предложения размерности m;
       относительная позиция, обратная позиция, признак первого предложения;
       логарифм длины; косинусная близость к центроиду документа и к заголовку).

TF-IDF вычисляется так же, как sklearn.TfidfVectorizer(max_features=m,
sublinear_tf=True, smooth_idf=True, norm="l2"), но по заранее посчитанной
матрице частот: словарь из m наиболее частых в обучающей выборке значимых слов,
документом для IDF служит предложение.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from data import Corpus

DENSE_NAMES = ["pos_rel", "pos_inv", "is_first", "len_log", "sim_centroid", "sim_title"]


def _rowwise_cosine(X: sparse.csr_matrix, doc: np.ndarray, Y: sparse.csr_matrix) -> np.ndarray:
    """cos(X_i, Y_doc(i)) для каждой строки i; строки X уже нормированы по L2."""
    Y = normalize(Y.tocsr())
    Y.sort_indices()
    Xc = X.tocoo()
    yvals = np.asarray(Y[doc[Xc.row], Xc.col]).ravel()
    return np.bincount(Xc.row, weights=Xc.data * yvals, minlength=X.shape[0])


class FeatureBuilder:
    """Признаки предложений; словарь и IDF настраиваются только по обучающей выборке."""

    def __init__(self, m: int) -> None:
        self.m = m
        self.cols: np.ndarray | None = None
        self.tfidf: TfidfTransformer | None = None

    def fit(self, corpus: Corpus) -> tuple[sparse.csr_matrix, np.ndarray]:
        tf = np.asarray(corpus.counts.sum(axis=0)).ravel()
        nz = np.flatnonzero(tf)
        top = nz[np.lexsort((nz, -tf[nz]))][: self.m]     # m самых частых, при равенстве — меньший id
        self.cols = np.sort(top)
        self.tfidf = TfidfTransformer(sublinear_tf=True, smooth_idf=True, norm="l2")
        self.tfidf.fit(corpus.counts[:, self.cols])
        return self.transform(corpus)

    def transform(self, corpus: Corpus) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Возвращает разреженную TF-IDF-часть и плотную часть (6 признаков)."""
        X = self.tfidf.transform(corpus.counts[:, self.cols]).astype(np.float32).tocsr()
        doc = corpus.sent_doc_index()
        n = corpus.n_sents
        # центроид документа — сумма TF-IDF-векторов его предложений (масштаб не влияет на косинус)
        A = sparse.csr_matrix((np.ones(n, dtype=np.float32), (doc, np.arange(n))), shape=(corpus.n_docs, n))
        sim_centroid = _rowwise_cosine(X, doc, A @ X)
        title = self.tfidf.transform(corpus.title_counts[:, self.cols])
        sim_title = _rowwise_cosine(X, doc, title)
        pos = corpus.sent_position()
        n_doc = np.diff(corpus.sent_offsets)[doc]
        dense = np.column_stack([
            pos / np.maximum(n_doc - 1, 1),        # относительная позиция ∈ [0; 1]
            1.0 / (pos + 1.0),                     # обратная позиция
            (pos == 0).astype(float),              # первое предложение
            np.log1p(corpus.sent_length()),        # логарифм длины в словах
            sim_centroid,
            sim_title,
        ]).astype(np.float32)
        return X, dense
