"""Проведение вычислительного эксперимента.

Схема: для каждой конфигурации (модель, N, m, C, ε, повтор) формируется
обучающая выборка из пула, строятся признаки, обучается модель, и на
фиксированной тестовой (или валидационной) выборке вычисляется ROUGE для
всех значений коэффициента сжатия r (r влияет только на этап отбора, поэтому
переобучения не требует).

Метод общих случайных чисел: в пределах повтора rep обучающие выборки разных
объёмов вложены (первые N документов одной перестановки пула), а множества
искажённых меток для разных ε вложены (метка i искажается, если u_i < ε при
фиксированных u_i ~ R[0; 1)). Это уменьшает дисперсию разностей W между
соседними значениями параметров и делает оценки производных устойчивее.
"""
from __future__ import annotations

import hashlib
import itertools
import logging
import os
import pickle
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from data import Corpus
from features import FeatureBuilder
from metrics import prf, rouge_l, rouge_n, select_top_k, summary_length
from models import SentenceScorer, lead_scores, textrank_scores

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ кэш процесса-исполнителя
_CORPORA: dict[str, dict[str, Corpus]] = {}
_EVALUATORS: dict[tuple, "Evaluator"] = {}


def load_corpora(path: str) -> dict[str, Corpus]:
    """Загрузка корпуса из файла кэша (один раз на процесс)."""
    if path not in _CORPORA:
        with open(path, "rb") as f:
            _CORPORA[path] = pickle.load(f)
    return _CORPORA[path]


# ------------------------------------------------------------------ оценивание аннотаций
class Evaluator:
    """Быстрый расчёт ROUGE-1 для набора документов при разных k.

    Для каждого документа заранее строится матрица частот слов его предложений
    (по объединённому словарю документа и эталона) и вектор частот эталона.
    """

    def __init__(self, corpus: Corpus) -> None:
        self.c = corpus
        self.docs = []
        for d in range(corpus.n_docs):
            sl = corpus.doc_slice(d)
            toks = [corpus.sent_tokens(s) for s in range(sl.start, sl.stop)]
            ref = corpus.ref_tokens(d)
            lens = np.array([len(t) for t in toks])
            uniq, inv = np.unique(np.concatenate(toks + [ref]), return_inverse=True)
            n_tok = int(lens.sum())
            M = np.zeros((len(toks), len(uniq)), dtype=np.int16)
            np.add.at(M, (np.repeat(np.arange(len(toks)), lens), inv[:n_tok]), 1)
            ref_counts = np.bincount(inv[n_tok:], minlength=len(uniq))
            self.docs.append((M, ref_counts, lens, len(ref)))

    def evaluate(self, scores: np.ndarray, r_grid: list[float], full: bool = False) -> dict:
        """ROUGE-1 F (а при full — ещё ROUGE-2, ROUGE-L, P, R и TP/FP/FN) по документам."""
        c = self.c
        n_docs, n_r = c.n_docs, len(r_grid)
        out = {"r1f": np.zeros((n_docs, n_r), dtype=np.float32)}
        if full:
            for name in ("r1p", "r1r", "r2f", "rlf", "tp", "fp", "fn", "k"):
                out[name] = np.zeros((n_docs, n_r), dtype=np.float32)
        for d, (M, ref_counts, lens, ref_len) in enumerate(self.docs):
            sl = c.doc_slice(d)
            s = scores[sl]
            n = len(s)
            for j, r in enumerate(r_grid):
                sel = select_top_k(s, summary_length(n, r))
                overlap = np.minimum(M[sel].sum(axis=0), ref_counts).sum()
                p, rec, f = prf(float(overlap), float(lens[sel].sum()), float(ref_len))
                out["r1f"][d, j] = f
                if full:
                    cand = np.concatenate([c.sent_tokens(sl.start + i) for i in sel]).tolist()
                    ref = c.ref_tokens(d).tolist()
                    y = c.labels[sl]
                    tp = int(y[sel].sum())
                    out["r1p"][d, j], out["r1r"][d, j] = p, rec
                    out["r2f"][d, j] = rouge_n(cand, ref, 2)[2]
                    out["rlf"][d, j] = rouge_l(cand, ref)[2]
                    out["tp"][d, j], out["fp"][d, j] = tp, len(sel) - tp
                    out["fn"][d, j], out["k"][d, j] = int(y.sum()) - tp, len(sel)
        return out


def get_evaluator(path: str, split: str, docs: tuple | None = None) -> Evaluator:
    """Объект оценивания для раздела корпуса (создаётся один раз на процесс)."""
    key = (path, split, docs)
    if key not in _EVALUATORS:
        corpus = load_corpora(path)[split]
        if docs is not None:
            corpus = corpus.subset(np.array(docs))
        _EVALUATORS[key] = Evaluator(corpus)
    return _EVALUATORS[key]


# ------------------------------------------------------------------ задачи
@dataclass(frozen=True)
class Spec:
    """Группа конфигураций с общей обучающей выборкой и общими признаками.

    pairs — список пар (C, ε), для каждой из которых обучается своя модель.
    """
    corpus_path: str
    model: str
    N: int
    m: int
    rep: int
    pairs: tuple            # ((C, eps), ...); для случайного леса C = None
    r_grid: tuple
    seed: int
    eval_split: str = "test"
    eval_docs: tuple | None = None      # подмножество документов eval_split (кросс-валидация)
    train_docs: tuple | None = None     # явный список обучающих документов (кросс-валидация)
    msl: int = 3
    n_estimators: int = 200
    full: bool = False
    tag: str = ""

    def key(self, C, eps) -> str:
        cs = "na" if C is None else f"{C:g}"
        msl = self.msl if self.model == "rf" else 0
        ntr = self.n_estimators if self.model == "rf" else 0
        extra = ""
        if self.train_docs is not None:
            extra = "|tr=" + hashlib.md5(np.array(self.train_docs).tobytes()).hexdigest()[:10]
        if self.eval_docs is not None:
            extra += "|ev=" + hashlib.md5(np.array(self.eval_docs).tobytes()).hexdigest()[:10]
        return (f"{self.eval_split}|{self.model}|N={self.N}|m={self.m}|C={cs}|eps={eps:g}"
                f"|rep={self.rep}|msl={msl}|T={ntr}|seed={self.seed}|r={len(self.r_grid)}{extra}")


def train_permutation(n_pool: int, rep: int, seed: int) -> np.ndarray:
    """Перестановка пула для повтора rep; обучающая выборка объёма N — её первые N документов."""
    return np.random.default_rng([seed, rep, 1]).permutation(n_pool)


def noise_uniforms(n_sents: int, rep: int, seed: int) -> np.ndarray:
    """Случайные числа u_i ~ R[0; 1) для искажения меток (метка i меняется, если u_i < ε)."""
    return np.random.default_rng([seed, rep, 2]).random(n_sents)


def run_spec(spec: Spec) -> list[tuple[str, dict]]:
    """Выполняет группу конфигураций; возвращает [(ключ, результаты оценивания)]."""
    t0 = time.perf_counter()
    corpora = load_corpora(spec.corpus_path)
    pool = corpora["train"]
    if spec.train_docs is not None:
        train_idx = np.array(spec.train_docs)
    else:
        train_idx = train_permutation(pool.n_docs, spec.rep, spec.seed)[: spec.N]
    train = pool.subset(train_idx)
    # равномерные числа берутся для предложений пула, чтобы множества искажений были вложены
    u_pool = noise_uniforms(pool.n_sents, spec.rep, spec.seed)
    sent_idx = np.concatenate([np.arange(pool.sent_offsets[d], pool.sent_offsets[d + 1]) for d in train_idx])
    u = u_pool[sent_idx]

    fb = FeatureBuilder(spec.m)
    X, D = fb.fit(train)
    ev = get_evaluator(spec.corpus_path, spec.eval_split, spec.eval_docs)
    Xe, De = fb.transform(ev.c)

    out = []
    for C, eps in spec.pairs:
        y = train.labels.copy()
        flip = u < eps
        y[flip] = 1 - y[flip]
        model = SentenceScorer(spec.model, C=1.0 if C is None else C, min_samples_leaf=spec.msl,
                               n_estimators=spec.n_estimators, seed=spec.seed + spec.rep, n_jobs=1)
        model.fit(X, D, y)
        res = ev.evaluate(model.score(Xe, De), list(spec.r_grid), full=spec.full)
        res["n_train_sents"] = np.array(train.n_sents)
        res["n_flipped"] = np.array(int(flip.sum()))
        out.append((spec.key(C, eps), res))
    log.debug("%s N=%d m=%d rep=%d: %.1f c", spec.model, spec.N, spec.m, spec.rep, time.perf_counter() - t0)
    return out


def run_baselines(corpus_path: str, split: str, r_grid: list[float], damping: float,
                  tol: float, max_iter: int) -> dict[str, dict]:
    """Lead-k, TextRank и оракул на всех документах раздела split (полный набор метрик)."""
    ev = get_evaluator(corpus_path, split)
    c = ev.c
    scores = {"lead": np.zeros(c.n_sents), "textrank": np.zeros(c.n_sents),
              "oracle": c.labels.astype(float)}
    for d in range(c.n_docs):
        sl = c.doc_slice(d)
        scores["lead"][sl] = lead_scores(sl.stop - sl.start)
        scores["textrank"][sl] = textrank_scores(c.counts[sl], damping, tol, max_iter)
    return {name: ev.evaluate(s, r_grid, full=True) for name, s in scores.items()}


# ------------------------------------------------------------------ хранилище результатов
class TaskStore:
    """Результаты конфигураций на диске (один .npz на конфигурацию) — позволяет
    продолжить прерванный расчёт и повторно использовать совпадающие конфигурации."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / (hashlib.md5(key.encode()).hexdigest() + ".npz")

    def has(self, key: str, full: bool = False) -> bool:
        """Есть ли результат (при full — с полным набором метрик)."""
        path = self._path(key)
        if not path.exists():
            return False
        if not full:
            return True
        with np.load(path) as z:
            return "r2f" in z.files

    def put(self, key: str, res: dict) -> None:
        tmp = self._path(key).with_suffix(".tmp.npz")
        np.savez_compressed(tmp, __key__=np.array(key), **res)
        os.replace(tmp, self._path(key))

    def get(self, key: str) -> dict:
        with np.load(self._path(key)) as z:
            return {k: z[k] for k in z.files if k != "__key__"}


def execute(specs: list[Spec], store: TaskStore, n_jobs: int, label: str) -> None:
    """Параллельно выполняет недостающие конфигурации и сохраняет их результаты."""
    todo = []
    for s in specs:
        pairs = tuple(p for p in s.pairs if not store.has(s.key(*p), full=s.full))
        if pairs:
            todo.append(replace(s, pairs=pairs))
    n_cfg = sum(len(s.pairs) for s in todo)
    if not todo:
        log.info("[%s] все конфигурации уже рассчитаны", label)
        return
    # сначала — самые трудоёмкие группы (большие N, случайный лес), для равномерной загрузки
    todo.sort(key=lambda s: -(s.N * (20 if s.model == "rf" else 1) * len(s.pairs)))
    log.info("[%s] к расчёту: %d групп, %d конфигураций, процессов: %d", label, len(todo), n_cfg, n_jobs)
    t0, done = time.perf_counter(), 0
    results = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered",
                       batch_size=1)(delayed(run_spec)(s) for s in todo)
    for group in results:
        for key, res in group:
            store.put(key, res)
            done += 1
        if done % 10 == 0 or done == n_cfg:
            el = time.perf_counter() - t0
            log.info("[%s] %d/%d конфигураций, %.0f c (осталось ≈ %.0f c)", label, done, n_cfg, el,
                     el / done * (n_cfg - done))


# ------------------------------------------------------------------ планы экспериментов
@dataclass(frozen=True)
class Plan:
    """Общие для всех задач параметры: путь к корпусу, сетка r, зерно, базовая точка."""
    corpus_path: str
    r_grid: tuple
    seed: int
    base: dict                 # {"N", "m", "r", "eps"}
    hyper: dict                # {модель: {"C": C0} или {"msl": ...}}
    n_estimators: int

    def C0(self, model: str):
        return None if model == "rf" else self.hyper[model]["C"]

    def msl(self, model: str) -> int:
        return self.hyper.get("rf", {}).get("msl", 3) if model == "rf" else 3

    def spec(self, model: str, N: int, m: int, rep: int, pairs, **kw) -> Spec:
        kw.setdefault("msl", self.msl(model))
        return Spec(self.corpus_path, model, int(N), int(m), int(rep), tuple(pairs), self.r_grid,
                    self.seed, n_estimators=self.n_estimators, **kw)

    def key(self, model: str, N: int, m: int, C, eps: float, rep: int, **kw) -> str:
        if model == "rf":
            C = None
        return self.spec(model, N, m, rep, (), **kw).key(C, eps)


def _grouped_specs(plan: Plan, configs: list[tuple], **kw) -> list[Spec]:
    """configs: (model, N, m, C, eps, rep) → группы с общей обучающей выборкой."""
    groups: dict[tuple, list] = {}
    for model, N, m, C, eps, rep in configs:
        pair = (None if model == "rf" else C, eps)
        g = groups.setdefault((model, N, m, rep), [])
        if pair not in g:
            g.append(pair)
    return [plan.spec(model, N, m, rep, pairs, **kw) for (model, N, m, rep), pairs in groups.items()]


def tune_specs(plan: Plan, grid: dict, reps=(0, 1)) -> list[Spec]:
    """Выбор гиперпараметров в базовой точке по валидационной выборке."""
    b = plan.base
    specs = []
    for model, g in grid.items():
        for rep in reps:
            if model == "rf":
                specs += [plan.spec("rf", b["N"], b["m"], rep, [(None, b["eps"])], msl=msl,
                                    eval_split="validation") for msl in g["min_samples_leaf"]]
            else:
                specs.append(plan.spec(model, b["N"], b["m"], rep, [(C, b["eps"]) for C in g["C"]],
                                       eval_split="validation"))
    return specs


def base_configs(plan: Plan, models, reps) -> list[tuple]:
    """Конфигурации базовой точки для всех моделей и повторов."""
    b = plan.base
    return [(mdl, b["N"], b["m"], plan.C0(mdl), b["eps"], rep) for mdl in models for rep in reps]


def ofat_configs(plan: Plan, models, grids: dict, reps) -> list[tuple]:
    """Однофакторный эксперимент: каждый параметр меняется при остальных в базовой точке."""
    b = plan.base
    cfg = []
    for mdl in models:
        C0 = plan.C0(mdl)
        for rep in reps:
            cfg += [(mdl, N, b["m"], C0, b["eps"], rep) for N in grids["N"]]
            cfg += [(mdl, b["N"], m, C0, b["eps"], rep) for m in grids["m"]]
            if mdl != "rf":
                cfg += [(mdl, b["N"], b["m"], C, b["eps"], rep) for C in grids["C"]]
            cfg += [(mdl, b["N"], b["m"], C0, e, rep) for e in grids["eps"]]
    return cfg


def grid2_configs(plan: Plan, model: str, p: str, q: str, grids: dict, reps) -> list[tuple]:
    """Узлы двумерной сетки (p, q) для изоэфф; r на обучение не влияет."""
    cfg = []
    for vp, vq, rep in itertools.product(grids[p], grids[q], reps):
        v = {"N": plan.base["N"], "m": plan.base["m"], "C": plan.C0(model), "eps": plan.base["eps"]}
        for name, val in ((p, vp), (q, vq)):
            if name != "r":
                v[name] = val
        cfg.append((model, v["N"], v["m"], v["C"], v["eps"], rep))
    return cfg


def factorial_runs(model: str, levels: dict) -> tuple[list[str], list[dict]]:
    """Опыты полного факторного эксперимента 2^k и центральная точка.

    Возвращает список факторов и список опытов {"x": нормированные уровни, "v": значения}.
    Для случайного леса фактор C не определён и в план не входит.
    """
    factors = [f for f in ("N", "m", "C", "r", "eps") if not (model == "rf" and f == "C")]
    runs = []
    for signs in itertools.product((-1, 1), repeat=len(factors)):
        v = {f: levels[f][0] if s < 0 else levels[f][1] for f, s in zip(factors, signs)}
        runs.append({"x": dict(zip(factors, signs)), "v": v})
    center = {}
    for f in factors:
        lo, hi = levels[f]
        center[f] = float(np.sqrt(lo * hi)) if f in ("N", "m", "C") else (lo + hi) / 2
    center["N"] = int(round(center["N"]))
    center["m"] = int(round(center["m"]))
    runs.append({"x": {f: 0 for f in factors}, "v": center, "center": True})
    return factors, runs


def factorial_configs(plan: Plan, model: str, levels: dict, reps) -> list[tuple]:
    """Конфигурации обучения для опытов полного факторного эксперимента."""
    _, runs = factorial_runs(model, levels)
    cfg = []
    for run, rep in itertools.product(runs, reps):
        v = run["v"]
        cfg.append((model, v["N"], v["m"], v.get("C", plan.C0(model)), v["eps"], rep))
    return list(dict.fromkeys(cfg))


def cv_specs(plan: Plan, models, folds: int) -> tuple[list[Spec], list[tuple]]:
    """Кросс-валидация (GroupKFold по документам) на обучающей выборке базовой точки."""
    from sklearn.model_selection import KFold
    pool = load_corpora(plan.corpus_path)["train"]
    docs = train_permutation(pool.n_docs, 0, plan.seed)[: plan.base["N"]]
    # разбиение по документам: все предложения документа попадают в один блок
    kf = KFold(n_splits=folds, shuffle=True, random_state=plan.seed)
    specs, meta = [], []
    for k, (tr, te) in enumerate(kf.split(docs)):
        for mdl in models:
            s = plan.spec(mdl, len(tr), plan.base["m"], 0, [(plan.C0(mdl), plan.base["eps"])],
                          eval_split="train", train_docs=tuple(docs[tr].tolist()),
                          eval_docs=tuple(docs[te].tolist()))
            specs.append(s)
            meta.append((mdl, k, s.key(plan.C0(mdl), plan.base["eps"])))
    return specs, meta
