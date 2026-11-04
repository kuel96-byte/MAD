"""Исследование эффективности моделей машинного обучения для аннотирования текста.

Запуск всех этапов:            python main.py
Отдельный этап:                python main.py --stage ofat
Быстрая проверка (smoke-тест): python main.py --quick
Быстрая проверка с нуля:       python main.py --quick --fresh

Этапы: prepare → tune → baselines → cv → base → ofat → isoeff → factorial → analyze → plots.
Каждый этап использует результаты предыдущих; рассчитанные конфигурации
хранятся в results/store, поэтому прерванный расчёт продолжается с места остановки.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # параллелизм — на уровне процессов, не потоков

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analysis as A  # noqa: E402
import config  # noqa: E402
import data  # noqa: E402
import experiment as X  # noqa: E402
import plots  # noqa: E402
from models import MODEL_NAMES  # noqa: E402

log = logging.getLogger("main")
STAGES = ["prepare", "tune", "baselines", "cv", "base", "ofat", "isoeff", "factorial", "analyze", "plots"]


# ------------------------------------------------------------------ служебное
def setup_logging() -> None:
    """Журналирование в консоль и в файл logs/run_*.log."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[
        logging.FileHandler(config.LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)])


def enable_quick_mode() -> None:
    """Уменьшенные объёмы для проверки работоспособности всей цепочки за несколько минут."""
    config.RESULTS_DIR = config.ROOT / "results_quick"
    config.FIG_DIR = config.ROOT / "figures_quick"
    config.TRAIN_POOL_SIZE, config.VAL_SIZE, config.TEST_SIZE = 400, 80, 120
    config.N_GRID = [50, 100, 200]
    config.M_GRID = [300, 1000, 3000]
    config.C_GRID = [0.01, 0.1, 1.0, 10.0]
    config.EPS_GRID = [0.0, 0.15, 0.3]
    config.PARAM_GRIDS = {"N": config.N_GRID, "m": config.M_GRID, "C": config.C_GRID,
                          "r": config.R_GRID, "eps": config.EPS_GRID}
    config.BASE = {"N": 100, "m": 1000, "C": 1.0, "r": 0.10, "eps": 0.0}
    config.N_REPEATS, config.N_REPEATS_GRID, config.CV_FOLDS = 2, 2, 3
    config.N_BOOTSTRAP, config.N_BOOTSTRAP_DIST = 60, 300
    config.RF_N_ESTIMATORS = 20
    config.TUNE_GRID = {"logreg": {"C": [0.1, 1.0]}, "svm": {"C": [0.01, 0.1]},
                        "nb": {"C": [0.3, 3.0]}, "rf": {"min_samples_leaf": [1, 10]}}
    config.ISOEFF_GRID = {"N": [50, 100, 200], "m": [300, 1000, 3000], "C": [0.01, 0.1, 1.0, 10.0],
                          "r": [0.05, 0.1, 0.2, 0.3], "eps": [0.0, 0.15, 0.3]}
    config.FACTORIAL_LEVELS = {"N": (50, 200), "m": (300, 3000), "C": (1e-2, 10.0),
                               "r": (0.10, 0.50), "eps": (0.0, 0.30)}


def corpus_path() -> str:
    """Путь к файлу предобработанного корпуса."""
    return str(config.CACHE_DIR / f"corpus_{config.TRAIN_POOL_SIZE}_{config.VAL_SIZE}_{config.TEST_SIZE}.pkl")


def store() -> X.TaskStore:
    """Хранилище результатов; запоминает, для какого корпуса оно рассчитано, и не даёт
    смешать результаты разных объёмов выборок (ключи конфигураций корпус не содержат)."""
    st = X.TaskStore(config.RESULTS_DIR / "store")
    stamp = st.root / "_corpus.txt"
    name = os.path.basename(corpus_path())
    if stamp.exists() and stamp.read_text(encoding="utf-8").strip() != name:
        raise RuntimeError(f"Хранилище {st.root} рассчитано для корпуса {stamp.read_text(encoding='utf-8').strip()}, "
                           f"а сейчас задан {name} (изменены объёмы выборок в config.py). "
                           "Удалите каталог results/store или укажите другой RESULTS_DIR.")
    stamp.write_text(name, encoding="utf-8")
    return st


def save_csv(df: pd.DataFrame, name: str) -> None:
    """Сохранение таблицы результатов в CSV."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / name, index=False, encoding="utf-8-sig", float_format="%.6g")


def save_json(obj, name: str) -> None:
    """Сохранение результатов в JSON."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


def load_json(name: str):
    """Чтение результатов из JSON."""
    with open(config.RESULTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def make_plan(hyper: dict | None = None, base: dict | None = None) -> X.Plan:
    """Общие параметры задач: базовая точка и гиперпараметры моделей."""
    if hyper is None or base is None:
        bp = load_json("base_point.json")
        hyper, base = bp["hyper"], bp["base"]
    return X.Plan(corpus_path(), tuple(config.R_GRID), config.SEED, base, hyper, config.RF_N_ESTIMATORS)


def r_index(r: float) -> int:
    """Номер значения коэффициента сжатия в сетке R_GRID."""
    return int(np.argmin(np.abs(np.array(config.R_GRID) - r)))


def reps_main() -> list[int]:
    """Номера повторов однофакторного эксперимента."""
    return list(range(config.N_REPEATS))


def reps_grid() -> list[int]:
    """Номера повторов для изоэфф и факторного эксперимента."""
    return list(range(config.N_REPEATS_GRID))


def perdoc(st: X.TaskStore, plan: X.Plan, model: str, N, m, C, eps, reps, metric: str = "r1f",
           split: str = "test") -> np.ndarray:
    """Массив (повторы × документы × значения r) метрики для конфигурации."""
    return np.stack([st.get(plan.key(model, N, m, C, eps, rep, eval_split=split))[metric] for rep in reps])


def baseline_key(method: str, split: str) -> str:
    """Ключ хранилища для результатов базового метода."""
    return f"baseline|{split}|{method}|r={len(config.R_GRID)}"


# ------------------------------------------------------------------ этапы
def stage_prepare() -> None:
    """Этап prepare: загрузка и предобработка корпуса."""
    corpora = data.prepare_corpora()
    save_csv(pd.DataFrame([data.corpus_stats(c) for c in corpora.values()]), "corpus_stats.csv")
    log.info("Статистика корпуса сохранена")


def stage_tune(jobs: int) -> None:
    """Выбор C0 / α0 / min_samples_leaf и r0 по валидационной выборке."""
    b = config.BASE
    hyper0 = {m: {"C": 1.0} for m in config.C_MODELS} | {"rf": {"msl": config.RF_MIN_SAMPLES_LEAF}}
    plan = make_plan(hyper0, {"N": b["N"], "m": b["m"], "r": b["r"], "eps": b["eps"]})
    st = store()
    X.execute(X.tune_specs(plan, config.TUNE_GRID), st, jobs, "tune")
    rows = []
    for model, g in config.TUNE_GRID.items():
        values = g["min_samples_leaf"] if model == "rf" else g["C"]
        for v in values:
            for rep in (0, 1):
                if model == "rf":
                    key = plan.key("rf", b["N"], b["m"], None, b["eps"], rep, msl=v, eval_split="validation")
                else:
                    key = plan.key(model, b["N"], b["m"], v, b["eps"], rep, eval_split="validation")
                w = st.get(key)["r1f"].mean(axis=0)
                for r, wr in zip(config.R_GRID, w):
                    rows.append({"model": model, "hyper": "min_samples_leaf" if model == "rf" else "C",
                                 "value": v, "rep": rep, "r": r, "W_val": wr})
    df = pd.DataFrame(rows)
    save_csv(df, "tune.csv")
    agg = df.groupby(["model", "value", "r"], as_index=False).W_val.mean()
    lo, hi = config.R_TASK_RANGE
    in_range = agg[(agg.r >= lo - 1e-9) & (agg.r <= hi + 1e-9)]
    best_by_r = in_range.groupby(["model", "r"]).W_val.max().groupby("r").mean()
    r0 = float(best_by_r.idxmax())
    hyper = {}
    for model in config.TUNE_GRID:
        d = agg[(agg.model == model) & (np.isclose(agg.r, r0))]
        v = float(d.loc[d.W_val.idxmax(), "value"])
        hyper[model] = {"msl": int(v)} if model == "rf" else {"C": v}
    base = {"N": b["N"], "m": b["m"], "r": r0, "eps": b["eps"]}
    save_json({"base": base, "hyper": hyper, "criterion": "максимум W на валидационной выборке"},
              "base_point.json")
    log.info("Базовая точка: %s; гиперпараметры: %s", base, hyper)


def stage_baselines() -> None:
    """Этап baselines: Lead-k, TextRank и оракул."""
    st = store()
    for split in ("test", "validation"):
        if all(st.has(baseline_key(m, split)) for m in config.BASELINES + config.REFERENCE):
            continue
        res = X.run_baselines(corpus_path(), split, list(config.R_GRID), config.TEXTRANK_DAMPING,
                              config.TEXTRANK_TOL, config.TEXTRANK_MAX_ITER)
        for method, r in res.items():
            st.put(baseline_key(method, split), r)
    log.info("Базовые методы рассчитаны")


def stage_cv(jobs: int) -> None:
    """Этап cv: кросс-валидация в базовой точке."""
    plan = make_plan()
    specs, meta = X.cv_specs(plan, config.MODELS, config.CV_FOLDS)
    st = store()
    X.execute(specs, st, jobs, "cv")
    j = r_index(plan.base["r"])
    rows = [{"model": m, "fold": k, "W_cv": float(st.get(key)["r1f"][:, j].mean())} for m, k, key in meta]
    save_csv(pd.DataFrame(rows), "cv_folds.csv")


def stage_base(jobs: int) -> None:
    """Этап base: все модели в базовой точке с полным набором метрик."""
    plan = make_plan()
    specs = X._grouped_specs(plan, X.base_configs(plan, config.MODELS, reps_main()), full=True)
    X.execute(specs, store(), jobs, "base")


def stage_ofat(jobs: int) -> None:
    """Этап ofat: однофакторные зависимости по N, m, C и ε."""
    plan = make_plan()
    grids = {"N": config.N_GRID, "m": config.M_GRID, "C": config.C_GRID, "eps": config.EPS_GRID}
    specs = X._grouped_specs(plan, X.ofat_configs(plan, config.MODELS, grids, reps_main()))
    X.execute(specs, store(), jobs, "ofat")


def ofat_values(param: str) -> list:
    """Сетка значений исследуемого параметра."""
    return config.R_GRID if param == "r" else config.PARAM_GRIDS[param]


def ofat_perdoc(st, plan, model: str, param: str, reps) -> tuple[list, np.ndarray]:
    """Значения параметра и матрица (значения × документы) W, усреднённых по повторам."""
    b = plan.base
    j0 = r_index(b["r"])
    vals, mats = [], []
    if param == "r":
        arr = perdoc(st, plan, model, b["N"], b["m"], plan.C0(model), b["eps"], reps).mean(axis=0)
        return list(config.R_GRID), arr.T
    for v in ofat_values(param):
        cfg = {"N": b["N"], "m": b["m"], "C": plan.C0(model), "eps": b["eps"]} | {param: v}
        arr = perdoc(st, plan, model, cfg["N"], cfg["m"], cfg["C"], cfg["eps"], reps)
        vals.append(v)
        mats.append(arr[:, :, j0].mean(axis=0))
    return vals, np.stack(mats)


def influence_ranking(st, plan) -> pd.DataFrame:
    """Размах ΔW_j = max W − min W по исследованному диапазону параметра (без сглаживания)."""
    rows = []
    for model in config.MODELS:
        for param in ("N", "m", "C", "r", "eps"):
            if param == "C" and model == "rf":
                continue
            vals, mat = ofat_perdoc(st, plan, model, param, reps_main())
            w = mat.mean(axis=1)
            if param == "r":   # для r — только заданный диапазон [0,1; 0,5]
                keep = [i for i, v in enumerate(vals) if config.R_TASK_RANGE[0] - 1e-9 <= v]
                w = w[keep]
            rows.append({"model": model, "param": param, "W_min": w.min(), "W_max": w.max(),
                         "delta_W": w.max() - w.min()})
    return pd.DataFrame(rows)


def best_model(st, plan) -> str:
    """Модель с наибольшим W в базовой точке."""
    j0 = r_index(plan.base["r"])
    b = plan.base
    w = {m: perdoc(st, plan, m, b["N"], b["m"], plan.C0(m), b["eps"], reps_main())[:, :, j0].mean()
         for m in config.MODELS}
    return max(w, key=w.get)


def isoeff_pair(st, plan) -> tuple[str, str]:
    """Два параметра с наибольшим размахом W для построения изоэфф."""
    rank = influence_ranking(st, plan)
    bm = best_model(st, plan)
    top = rank[rank.model == bm].sort_values("delta_W", ascending=False).param.tolist()[:2]
    order = ["N", "m", "C", "r", "eps"]
    return tuple(sorted(top, key=order.index))


def stage_isoeff(jobs: int) -> None:
    """Этап isoeff: расчёт W на двумерной сетке."""
    plan = make_plan()
    st = store()
    p, q = isoeff_pair(st, plan)
    models = [m for m in config.MODELS if not (m == "rf" and "C" in (p, q))]
    log.info("Изоэффы в плоскости (%s, %s) для моделей %s", p, q, models)
    cfg = []
    for model in models:
        cfg += X.grid2_configs(plan, model, p, q, config.ISOEFF_GRID, reps_grid())
    X.execute(X._grouped_specs(plan, cfg), st, jobs, "isoeff")
    save_json({"p": p, "q": q, "models": models}, "isoeff_pair.json")


def stage_factorial(jobs: int) -> None:
    """Этап factorial: полный факторный эксперимент."""
    plan = make_plan()
    cfg = []
    for model in config.MODELS:
        cfg += X.factorial_configs(plan, model, config.FACTORIAL_LEVELS, reps_grid())
    X.execute(X._grouped_specs(plan, cfg), store(), jobs, "factorial")


# ------------------------------------------------------------------ анализ
def micro_f1(tp, fp, fn) -> float:
    """F1-мера классификации предложений по суммарным TP, FP, FN."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def stage_analyze() -> None:
    """Этап analyze: статистическая обработка и таблицы результатов."""
    plan = make_plan()
    st = store()
    rng = np.random.default_rng(config.SEED)
    b = plan.base
    j0 = r_index(b["r"])
    B, conf, alpha = config.N_BOOTSTRAP, config.CONF_LEVEL, config.ALPHA
    reps = reps_main()

    # --- 1. метрики в базовой точке ------------------------------------------
    base_full, rows = {}, []
    for model in config.MODELS:
        res = [st.get(plan.key(model, b["N"], b["m"], plan.C0(model), b["eps"], rep)) for rep in reps]
        base_full[model] = {k: np.stack([r[k] for r in res]) for k in ("r1f", "r2f", "rlf", "r1p", "r1r", "tp", "fp", "fn")}
    for method in config.BASELINES + config.REFERENCE:
        r = st.get(baseline_key(method, "test"))
        base_full[method] = {k: r[k][None] for k in ("r1f", "r2f", "rlf", "r1p", "r1r", "tp", "fp", "fn")}
    per_doc_base = {}
    for method, res in base_full.items():
        for metric in ("r1f", "r2f", "rlf", "r1p", "r1r"):
            x = res[metric][:, :, j0].mean(axis=0)      # усреднение по повторам
            m, lo, hi = A.bootstrap_ci(x, B, conf, rng)
            _, slo, shi = A.student_ci(x, conf)
            rep_means = res[metric][:, :, j0].mean(axis=1)
            rows.append({"method": method, "metric": metric, "mean": m, "lo": lo, "hi": hi,
                         "student_lo": slo, "student_hi": shi,
                         "sd_reps": float(rep_means.std(ddof=1)) if len(rep_means) > 1 else 0.0,
                         "n_docs": len(x), "n_reps": res[metric].shape[0]})
            if metric == "r1f":
                per_doc_base[method] = x
        tp, fp, fn = (res[k][:, :, j0] for k in ("tp", "fp", "fn"))
        f1 = np.mean([micro_f1(tp[i].sum(), fp[i].sum(), fn[i].sum()) for i in range(tp.shape[0])])
        n_docs = tp.shape[1]
        boots = []
        for _ in range(B):
            idx = rng.integers(0, n_docs, n_docs)
            boots.append(np.mean([micro_f1(tp[i, idx].sum(), fp[i, idx].sum(), fn[i, idx].sum())
                                  for i in range(tp.shape[0])]))
        rows.append({"method": method, "metric": "f1_cls", "mean": f1, "lo": np.quantile(boots, alpha / 2),
                     "hi": np.quantile(boots, 1 - alpha / 2), "n_docs": n_docs, "n_reps": tp.shape[0]})
    base_metrics = pd.DataFrame(rows)
    save_csv(base_metrics, "base_metrics.csv")
    save_csv(pd.DataFrame(per_doc_base), "base_perdoc_W.csv")

    # --- 2. сравнение методов ---------------------------------------------------
    methods_cmp = {k: v for k, v in per_doc_base.items() if k != "oracle"}
    save_csv(A.paired_comparisons(methods_cmp, alpha, B, rng), "comparisons.csv")

    # --- 3. кросс-валидация -----------------------------------------------------
    if (config.RESULTS_DIR / "cv_folds.csv").exists():
        cv = pd.read_csv(config.RESULTS_DIR / "cv_folds.csv")
        summ = cv.groupby("model").W_cv.agg(["mean", "std", "count"]).reset_index()
        test_w = base_metrics[(base_metrics.metric == "r1f")].set_index("method")["mean"]
        summ["W_test"] = summ.model.map(test_w)
        save_csv(summ, "cv_summary.csv")

    # --- 4. однофакторные зависимости и чувствительность ------------------------
    local_rows = []
    for param in ("N", "m", "C", "r", "eps"):
        rows = []
        curves = {}
        for model in config.MODELS:
            if param == "C" and model not in config.C_MODELS:
                continue
            vals, mat = ofat_perdoc(st, plan, model, param, reps)
            for v, x in zip(vals, mat):
                m, lo, hi = A.bootstrap_ci(x, B, conf, rng)
                rows.append({"method": model, "value": v, "W": m, "lo": lo, "hi": hi})
            log_scale = param in config.LOG_PARAMS
            vv = np.array(vals, dtype=float)
            sel = slice(None)
            if param == "r":    # производные — только в заданном диапазоне
                sel = vv >= config.R_TASK_RANGE[0] - 1e-9
            c = A.sensitivity_curves(vv[sel], mat[sel], log_scale, B=min(B, 300), rng=rng, conf=conf)
            c.insert(0, "method", model)
            curves[model] = c
            x0 = b[param] if param != "C" else plan.C0(model)
            loc = A.local_sensitivity(c.drop(columns="method"), x0)
            local_rows.append({"method": model, "param": param, "x0": x0, **loc,
                               "W_range_min": float(mat.mean(axis=1)[sel].min()),
                               "W_range_max": float(mat.mean(axis=1)[sel].max())})
        for method in config.BASELINES + (config.REFERENCE if param == "r" else []):
            w_all = st.get(baseline_key(method, "test"))["r1f"]
            for i, v in enumerate(ofat_values(param)):
                x = w_all[:, i] if param == "r" else w_all[:, j0]
                m, lo, hi = A.bootstrap_ci(x, B, conf, rng)
                rows.append({"method": method, "value": v, "W": m, "lo": lo, "hi": hi})
        save_csv(pd.DataFrame(rows), f"ofat_{param}.csv")
        save_csv(pd.concat(curves.values(), ignore_index=True), f"sensitivity_{param}.csv")
    local = pd.DataFrame(local_rows)
    local["delta_W"] = local.W_range_max - local.W_range_min
    save_csv(local, "sensitivity_local.csv")
    save_csv(influence_ranking(st, plan), "influence_ranking.csv")

    # фактическая доля искажённых меток
    noise_rows = []
    for e in config.EPS_GRID:
        r = st.get(plan.key("logreg", b["N"], b["m"], plan.C0("logreg"), e, 0))
        noise_rows.append({"eps": e, "flipped": int(r["n_flipped"]), "n_sents": int(r["n_train_sents"]),
                           "share": int(r["n_flipped"]) / int(r["n_train_sents"])})
    save_csv(pd.DataFrame(noise_rows), "noise_check.csv")

    # --- 5. изоэффы ------------------------------------------------------------------
    if (config.RESULTS_DIR / "isoeff_pair.json").exists():
        info = load_json("isoeff_pair.json")
        p, q = info["p"], info["q"]
        summary = {"p": p, "q": q, "models": {}}
        for model in info["models"]:
            Wg = np.zeros((len(config.ISOEFF_GRID[p]), len(config.ISOEFF_GRID[q])))
            SE = np.zeros_like(Wg)
            rows = []
            for i, vp in enumerate(config.ISOEFF_GRID[p]):
                for k, vq in enumerate(config.ISOEFF_GRID[q]):
                    cfg = {"N": b["N"], "m": b["m"], "C": plan.C0(model), "eps": b["eps"], "r": b["r"]}
                    cfg[p], cfg[q] = vp, vq
                    arr = perdoc(st, plan, model, cfg["N"], cfg["m"], cfg["C"], cfg["eps"], reps_grid())
                    x = arr[:, :, r_index(cfg["r"])].mean(axis=0)
                    Wg[i, k] = x.mean()
                    SE[i, k] = x.std(ddof=1) / np.sqrt(len(x))
                    rows.append({"p": p, "q": q, "vp": vp, "vq": vq, "W": Wg[i, k], "se": SE[i, k]})
            save_csv(pd.DataFrame(rows), f"isoeff_{model}.csv")
            up, uq, Z = A.isoeff_surface(A.to_u(config.ISOEFF_GRID[p], p in config.LOG_PARAMS),
                                         A.to_u(config.ISOEFF_GRID[q], q in config.LOG_PARAMS),
                                         Wg, float(SE.mean()))
            reach, unreach = A.reachable_levels(Z, config.ISOEFF_LEVELS)
            np.savez_compressed(config.RESULTS_DIR / f"isoeff_surface_{model}.npz", up=up, uq=uq, Z=Z)
            summary["models"][model] = {"W_min": float(Z.min()), "W_max": float(Z.max()),
                                        "W_raw_min": float(Wg.min()), "W_raw_max": float(Wg.max()),
                                        "reachable": reach, "unreachable": unreach}
        oracle_w = st.get(baseline_key("oracle", "test"))["r1f"].mean(axis=0)
        summary["oracle_by_r"] = dict(zip(map(str, config.R_GRID), map(float, oracle_w)))
        save_json(summary, "isoeff_summary.json")

    # --- 6. полный факторный эксперимент: коэффициенты чувствительности ---------------
    fac_all = []
    centers = {}
    for model in config.MODELS:
        factors, runs = X.factorial_runs(model, config.FACTORIAL_LEVELS)
        corner = [r for r in runs if not r.get("center")]
        try:
            y = []
            for run in corner:
                v = run["v"]
                arr = perdoc(st, plan, model, v["N"], v["m"], v.get("C", plan.C0(model)), v["eps"], reps_grid())
                y.append(arr[:, :, r_index(v["r"])].mean(axis=1))
            center = [r for r in runs if r.get("center")][0]["v"]
            arr = perdoc(st, plan, model, center["N"], center["m"], center.get("C", plan.C0(model)),
                         center["eps"], reps_grid())
            yc = arr[:, :, r_index(center["r"])].mean(axis=1)
        except FileNotFoundError:
            log.warning("ПФЭ для %s не рассчитан", model)
            continue
        df = A.factorial_coefficients(factors, corner, np.array(y), yc, alpha)
        df.insert(0, "method", model)
        fac_all.append(df)
        centers[model] = df.attrs.get("center", {}) | {"s2": df.attrs["s2"], "R": df.attrs["R"],
                                                       "n_runs": df.attrs["n_runs"], "factors": factors}
    if fac_all:
        save_csv(pd.concat(fac_all, ignore_index=True), "factorial_coefficients.csv")
        save_json(centers, "factorial_center.json")

    # --- 7. идентификация законов распределения -----------------------------------------
    bm = best_model(st, plan)
    x = per_doc_base[bm]
    boot = A.bootstrap_means(x, config.N_BOOTSTRAP_DIST, rng)
    np.save(config.RESULTS_DIR / "bootstrap_W_best.npy", boot)
    ident = {"best_model": bm,
             "bootstrap_normal": A.identify(boot, "normal", alpha, rng),
             "perdoc_beta": A.identify(x, "beta", alpha, rng),
             "perdoc_normal": A.identify(x, "normal", alpha, rng)}
    save_json(ident, "distribution_tests.json")

    # --- 8. пример аннотации -----------------------------------------------------------------
    save_json(example_summary(plan, st), "example_summary.json")
    log.info("Анализ завершён; таблицы в %s", config.RESULTS_DIR)


def example_summary(plan: X.Plan, st: X.TaskStore) -> dict:
    """Пример: эталон, аннотации лучшей модели (повтор 0), Lead-k и TextRank для одного документа."""
    corp = X.load_corpora(plan.corpus_path)["test"]
    n_sent = np.diff(corp.sent_offsets)
    d = int(np.argmin(np.abs(n_sent - np.median(n_sent))))
    raw = pd.read_parquet(data.download_split("test"), columns=["text", "summary", "title"]).iloc[int(corp.doc_ids[d])]
    sents = [s for s in data.sentenize(raw.text) if data.tokenize(s)]
    k = int(np.ceil(plan.base["r"] * len(sents) - 1e-9))
    # перерасчёт оценок для этого документа: обучение лучшей модели на повторе 0
    bm = best_model(st, plan)
    pool = X.load_corpora(plan.corpus_path)["train"]
    tr = pool.subset(X.train_permutation(pool.n_docs, 0, plan.seed)[: plan.base["N"]])
    from features import FeatureBuilder
    from models import SentenceScorer, textrank_scores
    from metrics import select_top_k, rouge_n
    fb = FeatureBuilder(plan.base["m"])
    Xtr, Dtr = fb.fit(tr)
    model = SentenceScorer(bm, C=plan.C0(bm) or 1.0, min_samples_leaf=plan.msl(bm),
                           n_estimators=plan.n_estimators, seed=plan.seed).fit(Xtr, Dtr, tr.labels)
    one = corp.subset(np.array([d]))
    Xe, De = fb.transform(one)
    sel = {"model": select_top_k(model.score(Xe, De), k),
           "lead": np.arange(k),
           "textrank": select_top_k(textrank_scores(one.counts), k),
           "oracle": np.flatnonzero(one.labels)}
    ref = data.tokenize(raw.summary)
    out = {"title": raw.title, "reference": raw.summary, "n_sentences": len(sents), "k": k, "best_model": bm}
    for name, idx in sel.items():
        text = " ".join(sents[i] for i in idx)
        out[name] = {"sentences": [int(i) + 1 for i in idx], "text": text,
                     "rouge1_f": rouge_n(data.tokenize(text), ref, 1)[2]}
    return out


# ------------------------------------------------------------------ графики
def stage_plots() -> None:
    """Этап plots: построение всех графиков."""
    plots.setup()
    F = config.FIG_DIR
    R = config.RESULTS_DIR
    bp = load_json("base_point.json")
    base, hyper = bp["base"], bp["hyper"]
    for param in ("N", "m", "C", "r", "eps"):
        df = pd.read_csv(R / f"ofat_{param}.csv")
        x0 = base[param] if param != "C" else None      # у моделей разные C0 — общей вертикали нет
        plots.param_curves(df, param, F / f"W_vs_{param}.png", base_value=x0,
                           task_range=config.R_TASK_RANGE if param == "r" else None)
        sens = pd.read_csv(R / f"sensitivity_{param}.csv")
        curves = {m: g.drop(columns="method").reset_index(drop=True) for m, g in sens.groupby("method", sort=False)}
        plots.sensitivity(curves, param, F / f"sensitivity_{param}.png", base_value=x0)
    if (R / "isoeff_summary.json").exists():
        info = load_json("isoeff_summary.json")
        p, q = info["p"], info["q"]
        for model in info["models"]:
            z = np.load(R / f"isoeff_surface_{model}.npz")
            xp = np.exp(z["up"]) if p in config.LOG_PARAMS else z["up"]
            xq = np.exp(z["uq"]) if q in config.LOG_PARAMS else z["uq"]
            raw = pd.read_csv(R / f"isoeff_{model}.csv")
            base_pt = (base[p] if p != "C" else hyper[model]["C"], base[q] if q != "C" else hyper[model]["C"])
            plots.isoeffs(p, q, xp, xq, z["Z"], config.ISOEFF_GRID[p], config.ISOEFF_GRID[q], raw.W.values,
                          config.ISOEFF_LEVELS, 0.01, F / f"isoeff_{model}.png", MODEL_NAMES[model], base_pt)
    bmets = pd.read_csv(R / "base_metrics.csv")
    plots.base_bars(bmets[bmets.metric.isin(["r1f", "r2f", "rlf"])], F / "base_methods.png")
    ident = load_json("distribution_tests.json")
    from scipy import stats
    boot = np.load(R / "bootstrap_W_best.npy")
    bn = ident["bootstrap_normal"]
    plots.distribution(boot, {"нормальный закон": stats.norm(bn["M"], bn["sigma"])}, F / "dist_bootstrap.png",
                       "Бутстреп-оценка W*, отн. ед.", bins=A.n_bins(len(boot)) * 2)
    x = pd.read_csv(R / "base_perdoc_W.csv")[ident["best_model"]].values
    pb, pn = ident["perdoc_beta"], ident["perdoc_normal"]
    plots.distribution(x, {"бета-распределение": stats.beta(pb["alpha"] + 1, pb["beta"] + 1),
                           "нормальный закон": stats.norm(pn["M"], pn["sigma"])},
                       F / "dist_perdoc.png", "W документа, отн. ед.", bins=A.n_bins(len(x)))
    if (R / "factorial_coefficients.csv").exists():
        fac = pd.read_csv(R / "factorial_coefficients.csv")
        for model, g in fac.groupby("method"):
            plots.factorial_pareto(g, F / f"factorial_{model}.png", MODEL_NAMES[model])
    plots.flowchart(FLOW_BLOCKS, F / "scheme_algorithm.png")
    plots.program_scheme(F / "scheme_program.png")
    log.info("Графики сохранены в %s", F)


FLOW_BLOCKS = [
    (1, "terminal", "Начало"),
    (2, "data", "Ввод: корпус Gazeta, сетки x1–x5,\nR, B, α, базовая точка"),
    (3, "predefined", "Предобработка: razdel,\noracle-разметка $y_i$"),
    (4, "predefined", "Выбор C0 и r0 по\nвалидационной выборке"),
    (5, "loop_start", "A: модель (ЛР, НБК, SVM, СЛ)"),
    (6, "loop_start", "B: параметр $x_j$, значение, повтор"),
    (7, "process", "Выборка объёма N, искажение\nдоли ε меток"),
    (8, "predefined", "Признаки $x_i$: TF-IDF (m),\nпозиция, длина, близость"),
    (9, "process", "Обучение f(x; θ),\nоценки $p_i$ на тесте"),
    (10, "process", "Аннотация: $k=\\lceil r\\cdot n\\rceil$ предложений\nс наибольшими $p_i$; ROUGE"),
    (11, "loop_end", "B"),
    (12, "loop_end", "A"),
    (13, "predefined", "Статистическая обработка: ДИ,\n$S_j$, $E_j$, изоэффы, КЧ, гипотезы"),
    (14, "data", "Вывод: таблицы CSV,\nграфики PNG"),
    (15, "terminal", "Конец"),
]


# ------------------------------------------------------------------ точка входа
def clear_quick_results() -> None:
    """Удаление результатов, графиков и кэша корпуса быстрого режима (ключ --fresh)."""
    for d in (config.RESULTS_DIR, config.FIG_DIR):
        shutil.rmtree(d, ignore_errors=True)
    cache = Path(corpus_path())
    for f in (cache, cache.with_name(cache.stem.replace("corpus", "vocabulary") + ".pkl")):
        f.unlink(missing_ok=True)


def main() -> None:
    """Разбор параметров командной строки и выполнение этапов."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    ap.add_argument("--from-stage", choices=STAGES, help="выполнить этапы, начиная с указанного")
    ap.add_argument("--quick", action="store_true", help="быстрый режим с малыми объёмами (проверка)")
    ap.add_argument("--jobs", type=int, default=config.N_JOBS, help="число параллельных процессов")
    ap.add_argument("--fresh", action="store_true",
                    help="только вместе с --quick: удалить прежние результаты быстрого режима и посчитать всё заново")
    args = ap.parse_args()
    if args.fresh and not args.quick:
        ap.error("ключ --fresh используется только вместе с --quick "
                 "(в полном режиме он удалил бы результаты многочасового расчёта)")
    if args.quick:
        enable_quick_mode()
    if args.fresh:
        clear_quick_results()
    setup_logging()
    stages = STAGES if args.stage == "all" else [args.stage]
    if args.from_stage:
        stages = STAGES[STAGES.index(args.from_stage):]
    log.info("Этапы: %s; процессов: %d; режим: %s", stages, args.jobs, "быстрый" if args.quick else "полный")
    t0 = time.perf_counter()
    for s in stages:
        t = time.perf_counter()
        log.info("=== Этап %s ===", s)
        fn = globals()[f"stage_{s}"]
        fn(args.jobs) if "jobs" in fn.__code__.co_varnames else fn()
        log.info("Этап %s завершён за %.0f c", s, time.perf_counter() - t)
    log.info("Всего %.0f c", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
