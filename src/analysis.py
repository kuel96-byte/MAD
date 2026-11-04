"""Статистическая обработка результатов эксперимента.

Обозначения и процедуры проверки гипотез согласованы с учебником
Петухов Г.Б., Девяткин А.М., Якунин В.И. Методы статистической обработки
гидрометеорологической информации. – СПб.: ВКА, 2014 (далее — [1]):
критерий Пирсона — §7.2.1, формула (7.2.12); критерий Колмогорова — §7.2.3,
(7.2.21); метод стохастической индикации — §7.2.4 и §7.5.2, (7.5.28)–(7.5.29);
оценки параметров бета-распределения — §7.5.1, (7.5.14)–(7.5.16);
коэффициенты чувствительности полного факторного эксперимента — §16.4–16.5,
(16.5.5), (16.5.8), (16.5.9).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import interpolate, stats


# ------------------------------------------------------------------ доверительные интервалы
def bootstrap_means(x: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """B бутстреп-оценок среднего (ресэмплинг документов с возвращением)."""
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    return x[idx].mean(axis=1)


def bootstrap_ci(x: np.ndarray, B: int, conf: float, rng: np.random.Generator) -> tuple[float, float, float]:
    """Среднее и процентильный бутстреп-интервал с доверительной вероятностью conf."""
    bm = bootstrap_means(x, B, rng)
    a = (1 - conf) / 2
    return float(np.mean(x)), float(np.quantile(bm, a)), float(np.quantile(bm, 1 - a))


def student_ci(x: np.ndarray, conf: float) -> tuple[float, float, float]:
    """Доверительный интервал МатОж при неизвестном СКО: x̄ ∓ t_{β,n−1}·s/√n ([1], §6.8.2)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m, s = x.mean(), x.std(ddof=1)
    t = stats.t.ppf((1 + conf) / 2, n - 1)
    return float(m), float(m - t * s / math.sqrt(n)), float(m + t * s / math.sqrt(n))


# ------------------------------------------------------------------ сравнение моделей
def holm(pvals: np.ndarray) -> np.ndarray:
    """Поправка Холма на множественные сравнения (скорректированные p-значения)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_comparisons(per_doc: dict[str, np.ndarray], alpha: float, B: int,
                       rng: np.random.Generator) -> pd.DataFrame:
    """Попарное сравнение методов по W на одних и тех же документах.

    Парный t-критерий Стьюдента и критерий знаковых рангов Уилкоксона;
    p-значения корректируются по Холму внутри каждого критерия.
    """
    rows = []
    for a, b in itertools.combinations(per_doc, 2):
        d = per_doc[a] - per_doc[b]
        t, pt = stats.ttest_rel(per_doc[a], per_doc[b])
        w, pw = stats.wilcoxon(per_doc[a], per_doc[b], zero_method="wilcox")
        bm = bootstrap_means(d, B, rng)
        rows.append({"A": a, "B": b, "mean_diff": d.mean(), "diff_lo": np.quantile(bm, alpha / 2),
                     "diff_hi": np.quantile(bm, 1 - alpha / 2), "sd_diff": d.std(ddof=1),
                     "t": t, "p_t": pt, "wilcoxon_W": w, "p_w": pw, "n": len(d)})
    df = pd.DataFrame(rows)
    df["p_t_holm"] = holm(df.p_t.values)
    df["p_w_holm"] = holm(df.p_w.values)
    df["significant"] = (df.p_t_holm < alpha) & (df.p_w_holm < alpha)
    return df


# ------------------------------------------------------------------ идентификация законов распределения
def asymmetry_excess(x: np.ndarray) -> tuple[float, float]:
    """Оценки коэффициентов асимметрии и эксцесса ([1], (7.2.4′), (7.2.5′))."""
    x = np.asarray(x, dtype=float)
    mu = x - x.mean()
    s2 = np.mean(mu ** 2)
    return float(np.mean(mu ** 3) / s2 ** 1.5), float(np.mean(mu ** 4) / s2 ** 2 - 3.0)


def n_bins(n: int) -> int:
    """Число разрядов L ≈ 5·lg n, но в пределах [5; 25] ([1], зам. 6.2.2)."""
    return int(np.clip(round(5 * math.log10(n)), 5, 25))


def pearson_test(x: np.ndarray, cdf, n_params: int, zeta: float, L: int | None = None) -> dict:
    """Критерий Пирсона: u = Σ m_l²/(nP_l) − n ≈ χ²(L − 1 − s) ([1], (7.2.12), зам. 7.2.2).

    Разряды равной ширины на [x'_1; x'_n]; крайние разряды продлены до границ
    носителя гипотетического закона, чтобы ΣP_l = 1. Соседние разряды
    объединяются, пока в каждом не будет m_l ≥ 5 и nP_l ≥ 5.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    L0 = L or n_bins(n)
    inner = np.linspace(x[0], x[-1], L0 + 1)[1:-1]          # внутренние границы разрядов
    m = list(np.bincount(np.searchsorted(inner, x, side="right"), minlength=L0).astype(float))
    P = list(np.diff(cdf(np.r_[-np.inf, inner, np.inf])))
    i = 0
    while i < len(m):
        if (m[i] < 5 or n * P[i] < 5) and len(m) > 1:
            j = i + 1 if i + 1 < len(m) else i - 1     # объединяем с соседом справа (или слева)
            lo, hi = min(i, j), max(i, j)
            m[lo] += m[hi]
            P[lo] += P[hi]
            del m[hi], P[hi]
            i = max(lo - 1, 0)
            continue
        i += 1
    m, P = np.array(m), np.array(P)
    L_eff = len(m)
    u = float(np.sum(m ** 2 / (n * P)) - n)
    k = L_eff - 1 - n_params
    crit = float(stats.chi2.ppf(1 - zeta, k)) if k > 0 else float("nan")
    p = float(stats.chi2.sf(u, k)) if k > 0 else float("nan")
    return {"test": "Пирсон χ²", "u": u, "df": k, "L": L_eff, "u_crit": crit, "p": p,
            "accept": bool(u < crit) if k > 0 else None}


def kolmogorov_test(x: np.ndarray, cdf, zeta: float) -> dict:
    """Критерий Колмогорова: λ = √n·max|F*(x) − F^г(x)| ([1], (7.2.21), табл. 7.2.4)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    F = cdf(x)
    D = max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n))
    lam = math.sqrt(n) * D
    crit = float(stats.kstwobign.isf(zeta))
    return {"test": "Колмогоров λ", "u": lam, "u_crit": crit, "p": float(stats.kstwobign.sf(lam)),
            "accept": bool(lam < crit)}


def kolmogorov_mc_pvalue(x: np.ndarray, fit, sampler, n_sim: int, rng: np.random.Generator) -> float:
    """p-значение λ методом Монте-Карло при параметрах, оценённых по той же выборке.

    Классическое распределение Колмогорова (7.2.22) выведено для полностью
    известного гипотетического закона; при оценённых параметрах критерий
    консервативен ([1], зам. 7.2.4), поэтому распределение λ моделируется:
    выборка из подобранного закона → повторная оценка параметров → λ.
    """
    def lam(sample):
        dist = fit(sample)
        s = np.sort(sample)
        F = dist.cdf(s)
        k = len(s)
        return math.sqrt(k) * max(np.max(np.arange(1, k + 1) / k - F), np.max(F - np.arange(k) / k))
    lam0 = lam(np.asarray(x, dtype=float))
    dist0 = fit(np.asarray(x, dtype=float))
    sims = np.array([lam(sampler(dist0, len(x), rng)) for _ in range(n_sim)])
    return float((1 + np.sum(sims >= lam0)) / (n_sim + 1))


def stochastic_indication(x: np.ndarray, cdf, zeta: float) -> dict:
    """Метод стохастической индикации (Геоборина–Петухова), [1], §7.2.4, (7.5.28)–(7.5.29).

    Если гипотеза верна, супериндикатор ω = F^г(x) равномерен на (0; 1], т. е.
    M[ω] = 1/2, M[ω²] = 1/3. Обе оценки должны попасть в интервалы
    1/2 ∓ t_β/(2√(3n)) и 1/3 ∓ 2t_β/(3√(5n)), где β = √(1 − ζ) (две проверки).
    """
    w = cdf(np.asarray(x, dtype=float))
    n = len(w)
    beta = math.sqrt(1 - zeta)
    t = float(stats.norm.ppf((1 + beta) / 2))
    m1, m2 = float(w.mean()), float(np.mean(w ** 2))
    e1, e2 = t / (2 * math.sqrt(3 * n)), 2 * t / (3 * math.sqrt(5 * n))
    ok = abs(m1 - 0.5) <= e1 and abs(m2 - 1 / 3) <= e2
    return {"test": "Стохастическая индикация", "M_omega": m1, "M_omega2": m2, "eps1": e1, "eps2": e2,
            "t_beta": t, "accept": bool(ok)}


def beta_moments(v: np.ndarray, a: float = 0.0, b: float = 1.0) -> tuple[float, float]:
    """Оценки параметров формы α, β бета-распределения по моментам ([1], (7.5.14)–(7.5.16)).

    Плотность ∝ (v − a)^α (b − v)^β; в обозначениях scipy это beta(α + 1, β + 1, loc=a, scale=b − a).
    """
    v = np.asarray(v, dtype=float)
    m1 = (v.mean() - a) / (b - a)
    m2 = (np.mean(v ** 2) - 2 * a * v.mean() + a ** 2) / (b - a) ** 2
    k = m1 * (1 - m1) / (m2 - m1 ** 2) - 1
    return float(m1 * k - 1), float((1 - m1) * k - 1)


def identify(x: np.ndarray, family: str, zeta: float, rng: np.random.Generator,
             n_sim: int = 500) -> dict:
    """Проверка гипотезы о законе распределения выборки x тремя методами."""
    x = np.asarray(x, dtype=float)
    if family == "normal":
        mu, sd = float(x.mean()), float(x.std(ddof=1))
        dist = stats.norm(mu, sd)
        params = {"M": mu, "sigma": sd}
        fit = lambda s: stats.norm(s.mean(), s.std(ddof=1))
        sampler = lambda d, n, g: g.normal(d.mean(), d.std(), n)
    elif family == "beta":
        al, be = beta_moments(x)
        dist = stats.beta(al + 1, be + 1)
        params = {"alpha": al, "beta": be}
        def fit(s):
            a1, b1 = beta_moments(s)
            return stats.beta(a1 + 1, b1 + 1)
        sampler = lambda d, n, g: d.rvs(size=n, random_state=g)
    else:
        raise ValueError(family)
    res = {"family": family, "n": len(x), **params}
    pear = pearson_test(x, dist.cdf, n_params=2, zeta=zeta)
    kol = kolmogorov_test(x, dist.cdf, zeta)
    kol["p_mc"] = kolmogorov_mc_pvalue(x, fit, sampler, n_sim, rng)
    si = stochastic_indication(x, dist.cdf, zeta)
    a, e = asymmetry_excess(x)
    res.update({"asymmetry": a, "excess": e, "pearson": pear, "kolmogorov": kol, "indication": si})
    return res


# ------------------------------------------------------------------ функции чувствительности и влияния
def to_u(x: np.ndarray, log: bool) -> np.ndarray:
    """Аргумент сглаживания: ln x для логарифмических параметров, иначе x."""
    return np.log(np.asarray(x, dtype=float)) if log else np.asarray(x, dtype=float)


def fit_smooth(u: np.ndarray, w: np.ndarray, se: np.ndarray):
    """Сглаживающий кубический сплайн W(u) с весами 1/σ и s = n (критерий Райнша).

    При малом числе узлов (< 5) используется полином второй степени.
    """
    se = np.maximum(np.asarray(se, dtype=float), 1e-6)
    if len(u) >= 5:
        return interpolate.UnivariateSpline(u, w, w=1.0 / se, k=3, s=len(u))
    coef = np.polyfit(u, w, deg=min(2, len(u) - 1), w=1.0 / se)
    p = np.poly1d(coef)

    class _Poly:
        def __call__(self, z, nu=0):
            return np.polyder(p, nu)(z) if nu else p(z)
    return _Poly()


def sensitivity_curves(x: np.ndarray, per_doc: np.ndarray, log: bool, B: int,
                       rng: np.random.Generator, n_fine: int = 200, conf: float = 0.95) -> pd.DataFrame:
    """Сглаженная зависимость W(x), абсолютная чувствительность S = ∂W/∂x и
    относительная функция влияния E = (∂W/∂x)·(x/W) с бутстреп-полосами.

    per_doc: матрица (число значений x) × (число документов) значений W, усреднённых
    по повторам. Бутстреп ресэмплирует документы одинаково для всех x, сохраняя
    корреляцию между точками кривой.
    """
    x = np.asarray(x, dtype=float)
    u = to_u(x, log)
    u_f = np.linspace(u.min(), u.max(), n_fine)
    x_f = np.exp(u_f) if log else u_f

    def curves(mat):
        w = mat.mean(axis=1)
        se = mat.std(axis=1, ddof=1) / math.sqrt(mat.shape[1])
        sp = fit_smooth(u, w, se)
        W = sp(u_f)
        dWdu = sp(u_f, 1)
        S = dWdu / x_f if log else dWdu        # ∂W/∂x = (∂W/∂u)·(du/dx)
        with np.errstate(divide="ignore", invalid="ignore"):
            E = S * x_f / W
        return W, S, E

    W, S, E = curves(per_doc)
    n_docs = per_doc.shape[1]
    boots = [curves(per_doc[:, rng.integers(0, n_docs, n_docs)]) for _ in range(B)]
    a = (1 - conf) / 2
    out = {"x": x_f, "W": W, "S": S, "E": E}
    for name, i in (("W", 0), ("S", 1), ("E", 2)):
        arr = np.array([b[i] for b in boots])
        out[f"{name}_lo"] = np.nanquantile(arr, a, axis=0)
        out[f"{name}_hi"] = np.nanquantile(arr, 1 - a, axis=0)
    return pd.DataFrame(out)


def local_sensitivity(curve: pd.DataFrame, x0: float) -> dict:
    """Значения S и E (с полосами) в базовой точке x0 (линейная интерполяция по сетке)."""
    return {c: float(np.interp(x0, curve.x, curve[c])) for c in curve.columns if c != "x"}


# ------------------------------------------------------------------ изоэффы
def isoeff_surface(u_p: np.ndarray, u_q: np.ndarray, W: np.ndarray, se: float, n_fine: int = 150):
    """Сглаженная поверхность W(u_p, u_q) бикубическим сплайном.

    Параметр сглаживания s = n·σ² (n — число узлов, σ — средняя стандартная
    ошибка W в узле), т. е. сплайн проходит в пределах случайной погрешности.
    """
    kx, ky = min(3, len(u_p) - 1), min(3, len(u_q) - 1)
    sp = interpolate.RectBivariateSpline(u_p, u_q, W, kx=kx, ky=ky, s=W.size * se ** 2)
    up = np.linspace(u_p.min(), u_p.max(), n_fine)
    uq = np.linspace(u_q.min(), u_q.max(), n_fine)
    return up, uq, sp(up, uq)


def reachable_levels(Z: np.ndarray, levels: list[float]) -> tuple[list[float], list[float]]:
    """Разделение уровней изоэфф на достижимые и недостижимые."""
    lo, hi = float(np.nanmin(Z)), float(np.nanmax(Z))
    reach = [l for l in levels if lo <= l <= hi]
    return reach, [l for l in levels if l not in reach]


# ------------------------------------------------------------------ полный факторный эксперимент
def normalize_factor(v: float, lo: float, hi: float, log: bool) -> float:
    """Нормирование фактора к [−1; 1] ([1], (16.5.5)); для N, m, C — по ln x."""
    if log:
        v, lo, hi = math.log(v), math.log(lo), math.log(hi)
    return (2 * v - hi - lo) / (hi - lo)


def factorial_coefficients(factors: list[str], runs: list[dict], y_reps: np.ndarray,
                           y_center: np.ndarray | None, alpha: float) -> pd.DataFrame:
    """Оценки коэффициентов чувствительности (КЧ) по ПФЭ 2^k ([1], (16.5.8), (16.5.9)).

    Модель: y = a0 + Σ a_i x_i + Σ_{i<j} a_ij x_i x_j. y_reps — матрица
    (опыты × повторы) значений W. Дисперсия воспроизводимости σ² оценивается
    по разбросу между повторами; D[ã] = σ²/(n·R) для среднего по R повторам.
    Значимость КЧ — по t-критерию Стьюдента с n(R − 1) степенями свободы.
    """
    X = np.array([[r["x"][f] for f in factors] for r in runs], dtype=float)
    n, R = y_reps.shape
    y = y_reps.mean(axis=1)
    s2 = float(np.mean(y_reps.var(axis=1, ddof=1))) if R > 1 else float("nan")
    D = s2 / (n * R)
    dof = n * (R - 1)
    tcrit = float(stats.t.ppf(1 - alpha / 2, dof)) if dof > 0 else float("nan")
    cols = [("a0", np.ones(n))]
    cols += [(f"a_{f}", X[:, i]) for i, f in enumerate(factors)]
    cols += [(f"a_{f}·{g}", X[:, i] * X[:, j]) for (i, f), (j, g) in itertools.combinations(enumerate(factors), 2)]
    rows = []
    for name, xcol in cols:
        a = float(np.sum(xcol * y) / np.sum(xcol ** 2))
        t = a / math.sqrt(D) if D > 0 else float("nan")
        kind = "свободный член" if name == "a0" else ("билинейный" if "·" in name else "линейный")
        rows.append({"coef": name, "kind": kind, "value": a, "se": math.sqrt(D), "t": t,
                     "t_crit": tcrit, "significant": bool(abs(t) > tcrit) if name != "a0" else None})
    df = pd.DataFrame(rows)
    if y_center is not None and len(y_center):
        # проверка кривизны: разность среднего в центре плана и a0
        yc = float(np.mean(y_center))
        se_c = math.sqrt(s2 / len(y_center) + D)
        df.attrs["center"] = {"y_center": yc, "a0": float(y.mean()), "diff": yc - float(y.mean()),
                              "t": (yc - float(y.mean())) / se_c, "t_crit": tcrit}
    df.attrs["s2"] = s2
    df.attrs["R"] = R
    df.attrs["n_runs"] = n
    return df
