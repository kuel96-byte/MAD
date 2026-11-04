"""Загрузка корпуса Gazeta, сегментация на предложения и токены (razdel),
построение эталонной (oracle) разметки и компактное хранение корпуса.

Корпус хранится в виде целочисленных идентификаторов слов: это позволяет
параллельным процессам эксперимента держать в памяти весь пул документов.
"""
from __future__ import annotations

import logging
import pickle
import re
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import razdel
from scipy import sparse

import config
from metrics import greedy_oracle

log = logging.getLogger(__name__)

WORD_RE = re.compile(r"\w")

# Служебные слова русского языка: исключаются из признаков TF-IDF и из графа
# TextRank, но не из подсчёта ROUGE (стандартная процедура ROUGE их сохраняет).
STOPWORDS = frozenset("""
а без более бы был была были было быть в вам вас весь во вот все всего всех вы
где да даже для до его ее ей ему если есть еще же за здесь и из или им их к как
когда кто ли либо мне может мы на над надо наш не него нее нет ни них но ну о об
однако он она они оно от очень по под при с со так также такой там те тем то того
тоже той только том ты у уже хотя чего чей чем что чтобы чье чья эта эти это этого
этой этом этот я ведь вдруг вновь всё её ещё затем зато кроме между нам нами ним
ними нему ней нею своих свой свою своя своё свое себе себя сам сама сами само
самый тот та тех теми тому чём чем-то который которая которое которые которого
которой котором которых которым которыми потому поэтому почему про сейчас сюда
туда тут через после перед около вокруг среди пока раз лишь уж бы будет будут
буду будем будете будешь стал стала стали стало можно нельзя нужно должен должна
должны является являются был ли этих этим этими всем всеми всей всю вся всякий
каждый каждая каждое какой какая какое какие один одна одно одни два две три
""".split())


def normalize_token(tok: str) -> str:
    """Приведение токена к нижнему регистру с заменой «ё» на «е»."""
    return tok.lower().replace("ё", "е")


def tokenize(text: str) -> list[str]:
    """Словные токены razdel (знаки препинания отбрасываются)."""
    return [normalize_token(t.text) for t in razdel.tokenize(text) if WORD_RE.search(t.text)]


def sentenize(text: str) -> list[str]:
    """Сегментация текста на предложения (razdel)."""
    return [s.text for s in razdel.sentenize(text)]


def download_split(split: str) -> Path:
    """Скачивает parquet-файл раздела корпуса Gazeta, если его ещё нет."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / f"{split}_0000.parquet"
    if not path.exists():
        url = config.GAZETA_URL.format(split=split)
        log.info("Загрузка %s", url)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def _process_doc(args: tuple[str, str, str]) -> dict | None:
    """Предобработка одного документа (выполняется в отдельном процессе)."""
    text, summary, title = args
    sents = [tokenize(s) for s in sentenize(text)]
    sents = [s for s in sents if s]                       # пустые «предложения» отбрасываются
    if len(sents) < config.MIN_SENTENCES:
        return None
    ref = tokenize(summary)
    if not ref:
        return None
    oracle = greedy_oracle(sents, ref, config.ORACLE_MAX_SENTENCES)
    return {"sents": sents, "ref": ref, "title": tokenize(title or ""), "oracle": oracle}


@dataclass
class Corpus:
    """Предобработанная часть корпуса в «плоском» виде.

    Предложение s документа d занимает строки sent_offsets[d] … sent_offsets[d+1]−1;
    его токены — tok[tok_offsets[s]:tok_offsets[s+1]].
    """
    split: str
    sent_offsets: np.ndarray      # (n_docs + 1,) границы документов в массиве предложений
    tok: np.ndarray               # (n_tokens,) идентификаторы всех словных токенов
    tok_offsets: np.ndarray       # (n_sents + 1,) границы предложений в массиве tok
    ref: np.ndarray               # токены эталонных аннотаций
    ref_offsets: np.ndarray       # (n_docs + 1,)
    title_counts: sparse.csr_matrix   # (n_docs, V) частоты значимых слов заголовка
    counts: sparse.csr_matrix     # (n_sents, V) частоты значимых слов предложений
    labels: np.ndarray            # (n_sents,) эталонные метки y_i ∈ {0, 1}
    doc_ids: np.ndarray           # номера документов в исходном разделе

    @property
    def n_docs(self) -> int:
        return len(self.sent_offsets) - 1

    @property
    def n_sents(self) -> int:
        return len(self.labels)

    def doc_slice(self, d: int) -> slice:
        return slice(self.sent_offsets[d], self.sent_offsets[d + 1])

    def sent_tokens(self, s: int) -> np.ndarray:
        return self.tok[self.tok_offsets[s]:self.tok_offsets[s + 1]]

    def ref_tokens(self, d: int) -> np.ndarray:
        return self.ref[self.ref_offsets[d]:self.ref_offsets[d + 1]]

    def sent_doc_index(self) -> np.ndarray:
        """Номер документа (внутри корпуса) для каждого предложения."""
        return np.repeat(np.arange(self.n_docs), np.diff(self.sent_offsets))

    def sent_position(self) -> np.ndarray:
        """Порядковый номер предложения в документе (с нуля)."""
        starts = np.repeat(self.sent_offsets[:-1], np.diff(self.sent_offsets))
        return np.arange(self.n_sents) - starts

    def sent_length(self) -> np.ndarray:
        return np.diff(self.tok_offsets)

    def subset(self, docs: np.ndarray) -> "Corpus":
        """Подкорпус из документов с указанными номерами (в заданном порядке)."""
        docs = np.asarray(docs)
        sent_idx = np.concatenate([np.arange(self.sent_offsets[d], self.sent_offsets[d + 1]) for d in docs])
        n_per_doc = np.diff(self.sent_offsets)[docs]
        sent_offsets = np.concatenate([[0], np.cumsum(n_per_doc)])
        tok_len = np.diff(self.tok_offsets)[sent_idx]
        tok = np.concatenate([self.tok[self.tok_offsets[s]:self.tok_offsets[s + 1]] for s in sent_idx])
        ref_len = np.diff(self.ref_offsets)[docs]
        ref = np.concatenate([self.ref[self.ref_offsets[d]:self.ref_offsets[d + 1]] for d in docs])
        return Corpus(self.split, sent_offsets, tok, np.concatenate([[0], np.cumsum(tok_len)]),
                      ref, np.concatenate([[0], np.cumsum(ref_len)]),
                      self.title_counts[docs], self.counts[sent_idx], self.labels[sent_idx],
                      self.doc_ids[docs])


class Vocabulary:
    """Глобальный словарь: слово ↔ целочисленный идентификатор."""

    def __init__(self) -> None:
        self.index: dict[str, int] = {}
        self.words: list[str] = []

    def ids(self, tokens: list[str]) -> list[int]:
        out = []
        for t in tokens:
            i = self.index.get(t)
            if i is None:
                i = self.index[t] = len(self.words)
                self.words.append(t)
            out.append(i)
        return out

    def stop_mask(self) -> np.ndarray:
        return np.array([w in STOPWORDS or w.isdigit() for w in self.words], dtype=bool)


def _count_matrix(rows: list[list[int]], vocab_size: int, stop: np.ndarray) -> sparse.csr_matrix:
    """Матрица частот значимых слов (служебные слова и числа исключаются)."""
    indptr, indices = [0], []
    for r in rows:
        indices.extend(i for i in r if not stop[i])
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    m = sparse.csr_matrix((data, np.array(indices, dtype=np.int32), np.array(indptr)),
                          shape=(len(rows), vocab_size))
    m.sum_duplicates()
    return m


def prepare_corpora(n_jobs: int = config.N_JOBS) -> dict[str, Corpus]:
    """Загрузка, предобработка и кэширование пула обучения, валидации и теста."""
    cache = config.CACHE_DIR / f"corpus_{config.TRAIN_POOL_SIZE}_{config.VAL_SIZE}_{config.TEST_SIZE}.pkl"
    if cache.exists():
        log.info("Корпус загружается из кэша %s", cache.name)
        with open(cache, "rb") as f:
            return pickle.load(f)

    sizes = {"train": config.TRAIN_POOL_SIZE, "validation": config.VAL_SIZE, "test": config.TEST_SIZE}
    processed = {split: process_split(split, size, n_jobs) for split, size in sizes.items()}
    vocab = Vocabulary()
    corpora = build_corpora(processed, vocab)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(corpora, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(cache.with_name(cache.stem.replace("corpus", "vocabulary") + ".pkl"), "wb") as f:
        pickle.dump(vocab.words, f, protocol=pickle.HIGHEST_PROTOCOL)
    return corpora


def process_split(split: str, size: int, n_jobs: int = config.N_JOBS, seed: int = config.SEED
                  ) -> tuple[list[dict], np.ndarray]:
    """Случайная выборка size документов раздела, сегментация и эталонная разметка."""
    df = pd.read_parquet(download_split(split), columns=["text", "summary", "title"])
    rng = np.random.default_rng(seed)
    # берётся с запасом 10 %: часть документов может быть отброшена фильтрами
    idx = rng.permutation(len(df))[: int(size * 1.1) + 10]
    jobs = list(zip(df.text.iloc[idx], df.summary.iloc[idx], df.title.iloc[idx]))
    log.info("Предобработка %s: %d документов", split, len(jobs))
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        docs = list(ex.map(_process_doc, jobs, chunksize=32))
    keep = [i for i, d in enumerate(docs) if d is not None][:size]
    log.info("%s: оставлено %d документов", split, len(keep))
    return [docs[i] for i in keep], idx[keep]


def build_corpora(processed: dict[str, tuple[list[dict], np.ndarray]], vocab: Vocabulary) -> dict[str, Corpus]:
    """Перевод обработанных документов в «плоские» корпуса; словарь vocab пополняется."""
    raw: dict[str, dict] = {}
    for split, (docs, ids) in processed.items():
        sents = [vocab.ids(s) for d in docs for s in d["sents"]]
        raw[split] = {"docs": docs, "ids": ids, "sents": sents,
                      "refs": [vocab.ids(d["ref"]) for d in docs],
                      "titles": [vocab.ids(d["title"]) for d in docs]}
    stop = vocab.stop_mask()
    V = len(vocab.words)

    corpora = {}
    for split, r in raw.items():
        docs = r["docs"]
        n_per_doc = np.array([len(d["sents"]) for d in docs])
        labels = np.zeros(int(n_per_doc.sum()), dtype=np.int8)
        sent_offsets = np.concatenate([[0], np.cumsum(n_per_doc)])
        for d, doc in enumerate(docs):
            labels[sent_offsets[d] + np.array(doc["oracle"], dtype=int)] = 1
        tok_len = np.array([len(s) for s in r["sents"]])
        ref_len = np.array([len(x) for x in r["refs"]])
        corpora[split] = Corpus(
            split=split,
            sent_offsets=sent_offsets,
            tok=np.fromiter((t for s in r["sents"] for t in s), dtype=np.int32, count=int(tok_len.sum())),
            tok_offsets=np.concatenate([[0], np.cumsum(tok_len)]),
            ref=np.fromiter((t for x in r["refs"] for t in x), dtype=np.int32, count=int(ref_len.sum())),
            ref_offsets=np.concatenate([[0], np.cumsum(ref_len)]),
            title_counts=_count_matrix(r["titles"], V, stop),
            counts=_count_matrix(r["sents"], V, stop),
            labels=labels,
            doc_ids=np.asarray(r["ids"]),
        )
    return corpora


def corpus_stats(c: Corpus) -> dict:
    """Описательная статистика части корпуса (для раздела 2 записки)."""
    n_sent = np.diff(c.sent_offsets)
    ref_len = np.diff(c.ref_offsets)
    doc_tok = np.add.reduceat(c.sent_length(), c.sent_offsets[:-1])
    pos_per_doc = np.add.reduceat(c.labels.astype(int), c.sent_offsets[:-1])
    return {
        "split": c.split, "docs": c.n_docs, "sentences": c.n_sents,
        "sent_per_doc_mean": n_sent.mean(), "sent_per_doc_median": float(np.median(n_sent)),
        "sent_per_doc_min": int(n_sent.min()), "sent_per_doc_max": int(n_sent.max()),
        "words_per_doc_mean": doc_tok.mean(), "words_per_summary_mean": ref_len.mean(),
        "oracle_sent_per_doc_mean": pos_per_doc.mean(),
        "oracle_share_of_sentences": c.labels.mean(),
    }
