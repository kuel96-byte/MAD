"""Графики результатов (PNG, 300 dpi, подписи на русском, десятичная запятая).

Цвета методов закреплены (не зависят от числа кривых на рисунке); кроме цвета
каждый метод имеет свой маркер и тип линии, чтобы рисунки читались и при
чёрно-белой печати.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from models import MODEL_NAMES  # noqa: E402

STYLE = {
    "logreg": {"color": "#2a78d6", "marker": "o", "ls": "-"},
    "nb": {"color": "#eb6834", "marker": "s", "ls": "-"},
    "svm": {"color": "#1baf7a", "marker": "^", "ls": "-"},
    "rf": {"color": "#c98500", "marker": "D", "ls": "-"},
    "lead": {"color": "#d55181", "marker": None, "ls": "--"},
    "textrank": {"color": "#008300", "marker": None, "ls": "-."},
    "oracle": {"color": "#52514e", "marker": None, "ls": ":"},
}

PARAM_LABEL = {
    "N": "Объём обучающей выборки N, документов",
    "m": "Размер словаря TF-IDF m, признаков",
    "C": "Коэффициент регуляризации C, отн. ед. (для НБК α = 1/C)",
    "r": "Коэффициент сжатия r, отн. ед.",
    "eps": "Доля ошибочных меток ε, отн. ед.",
}
PARAM_SYMBOL = {"N": "N", "m": "m", "C": "C", "r": "r", "eps": "ε"}
S_UNITS = {"N": "1/документ", "m": "1/признак", "C": "отн. ед.", "r": "отн. ед.", "eps": "отн. ед."}
W_LABEL = "Показатель эффективности W (ROUGE-1 F), отн. ед."
LOG_PARAMS = {"N", "m", "C"}


def _comma(x, _pos=None) -> str:
    """Подпись деления с десятичной запятой."""
    if x == 0:
        return "0"
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", " ")
    if abs(x) >= 1 and float(x).is_integer():
        return f"{x:.0f}"
    s = f"{x:.6g}"
    if "e" in s:
        mant, exp = s.split("e")
        return f"{mant.replace('.', ',')}·10$^{{{int(exp)}}}$"
    return s.replace(".", ",")


def setup() -> None:
    """Общие настройки оформления графиков."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11, "axes.titlesize": 11,
        "axes.labelsize": 11, "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.grid": True, "grid.color": "#d9d8d4", "grid.linewidth": 0.6,
        "axes.edgecolor": "#52514e", "axes.linewidth": 0.8, "lines.linewidth": 1.6,
        "lines.markersize": 5, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 100,
        "axes.formatter.use_mathtext": True,
    })


def _format_axes(ax, xlog: bool = False) -> None:
    """Логарифмическая шкала (при необходимости) и подписи делений с запятой."""
    if xlog:
        ax.set_xscale("log")
        lo, hi = ax.get_xlim()
        wide = np.log10(hi / lo) > 3.5          # широкий диапазон (C): подписи только у степеней 10
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,) if wide else (1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(_comma))
    ax.yaxis.set_major_formatter(FuncFormatter(_comma))


def _save(fig, path: Path) -> None:
    """Сохранение рисунка в файл PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def param_curves(df: pd.DataFrame, param: str, path: Path, base_value: float | None = None,
                 task_range: tuple | None = None) -> None:
    """W(x_j) для всех методов с 95 % бутстреп-интервалами.

    df: столбцы method, value, W, lo, hi.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for method in [m for m in STYLE if m in set(df.method)]:
        d = df[df.method == method].sort_values("value")
        st = STYLE[method]
        ax.plot(d.value, d.W, color=st["color"], ls=st["ls"], marker=st["marker"],
                label=MODEL_NAMES[method])
        ax.fill_between(d.value, d.lo, d.hi, color=st["color"], alpha=0.15, lw=0)
    if base_value is not None:
        ax.axvline(base_value, color="#8a8984", lw=0.8, ls=(0, (2, 2)))
        ax.annotate("базовая точка", (base_value, ax.get_ylim()[0]), xytext=(3, 3),
                    textcoords="offset points", fontsize=8, color="#52514e")
    if task_range is not None:
        lo, hi = task_range
        xmin = df.value.min()
        if xmin < lo:
            ax.axvspan(xmin, lo, color="#ecebe8", zorder=0)
    _format_axes(ax, param in LOG_PARAMS)
    ax.set_xlabel(PARAM_LABEL[param])
    ax.set_ylabel(W_LABEL)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    _save(fig, path)


def sensitivity(curves: dict[str, pd.DataFrame], param: str, path: Path, base_value: float | None) -> None:
    """S_j(x) = ∂W/∂x_j и E_j(x) = (∂W/∂x_j)·(x_j/W) с 95 % бутстреп-полосами."""
    sym = PARAM_SYMBOL[param]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for method, c in curves.items():
        st = STYLE[method]
        for ax, col in zip(axes, ("S", "E")):
            ax.plot(c.x, c[col], color=st["color"], ls=st["ls"], label=MODEL_NAMES[method])
            ax.fill_between(c.x, c[f"{col}_lo"], c[f"{col}_hi"], color=st["color"], alpha=0.13, lw=0)
            mk = np.linspace(0, len(c) - 1, 7).astype(int)
            ax.plot(c.x.iloc[mk], c[col].iloc[mk], ls="none", marker=st["marker"], color=st["color"])
    for ax in axes:
        ax.axhline(0, color="#52514e", lw=0.8)
        if base_value is not None:
            ax.axvline(base_value, color="#8a8984", lw=0.8, ls=(0, (2, 2)))
        _format_axes(ax, param in LOG_PARAMS)
        ax.set_xlabel(PARAM_LABEL[param])
    if param in LOG_PARAMS:
        axes[0].set_yscale("symlog", linthresh=_linthresh(curves))
        axes[0].yaxis.set_major_formatter(FuncFormatter(_comma))
    axes[0].set_ylabel(f"Чувствительность $S_{{{sym}}}$ = ∂W/∂{sym}, {S_UNITS[param]}")
    axes[1].set_ylabel(f"Функция влияния $E_{{{sym}}}$, отн. ед.")
    axes[0].set_title("а) абсолютная чувствительность", loc="left")
    axes[1].set_title("б) относительная функция влияния", loc="left")
    handles = [plt.Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], label=MODEL_NAMES[m])
               for m in curves]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    _save(fig, path)


def _linthresh(curves) -> float:
    """Порог линейного участка симметричной логарифмической шкалы."""
    vals = np.concatenate([np.abs(c.S.values) for c in curves.values()])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(np.quantile(vals, 0.1)) if len(vals) else 1e-6


def isoeffs(p: str, q: str, xp: np.ndarray, xq: np.ndarray, Z: np.ndarray, raw_p, raw_q, raw_W,
            levels: list[float], aux_step: float, path: Path, title: str, base: tuple | None) -> None:
    """Изоэффы W = const в плоскости (x_p, x_q): основные уровни 0,1 (0,1) 0,9 —
    сплошные линии, вспомогательные — тонкие штриховые."""
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    P, Q = np.meshgrid(xp, xq, indexing="ij")
    zmin, zmax = np.nanmin(Z), np.nanmax(Z)
    im = ax.pcolormesh(P, Q, Z, cmap="Blues", shading="auto", alpha=0.85)
    aux = np.arange(np.floor(zmin / aux_step) * aux_step, zmax + aux_step, aux_step)
    aux = [a for a in aux if zmin < a < zmax and not any(abs(a - l) < 1e-9 for l in levels)]
    if aux:
        cs = ax.contour(P, Q, Z, levels=aux, colors="#52514e", linewidths=0.6, linestyles="--")
        ax.clabel(cs, fmt=lambda v: _comma(round(v, 3)), fontsize=7)
    main = [l for l in levels if zmin < l < zmax]
    if main:
        cs = ax.contour(P, Q, Z, levels=main, colors="#0b0b0b", linewidths=1.6)
        ax.clabel(cs, fmt=lambda v: "W = " + _comma(round(v, 2)), fontsize=9)
    RP, RQ = np.meshgrid(raw_p, raw_q, indexing="ij")
    ax.plot(RP.ravel(), RQ.ravel(), ls="none", marker=".", color="#52514e", ms=3)
    if base is not None:
        ax.plot(*base, marker="*", color="#e34948", ms=12, ls="none", label="базовая точка")
        ax.legend(loc="upper right", fontsize=8)
    if p in LOG_PARAMS:
        ax.set_xscale("log")
    if q in LOG_PARAMS:
        ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_comma))
    ax.yaxis.set_major_formatter(FuncFormatter(_comma))
    ax.set_xlabel(PARAM_LABEL[p])
    ax.set_ylabel(PARAM_LABEL[q])
    ax.set_title(title, loc="left")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(W_LABEL)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(_comma))
    ax.grid(False)
    _save(fig, path)


def base_bars(df: pd.DataFrame, path: Path) -> None:
    """Сравнение методов в базовой точке по ROUGE-1/2/L с 95 % интервалами."""
    metrics = [("r1f", "ROUGE-1 F"), ("r2f", "ROUGE-2 F"), ("rlf", "ROUGE-L F")]
    methods = [m for m in STYLE if m in set(df.method)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / len(methods)
    hatches = {"lead": "//", "textrank": "\\\\", "oracle": ".."}
    for i, m in enumerate(methods):
        d = df[df.method == m].set_index("metric")
        xs = np.arange(len(metrics)) + (i - (len(methods) - 1) / 2) * width
        vals = [d.loc[k, "mean"] for k, _ in metrics]
        err = [[d.loc[k, "mean"] - d.loc[k, "lo"] for k, _ in metrics],
               [d.loc[k, "hi"] - d.loc[k, "mean"] for k, _ in metrics]]
        ax.bar(xs, vals, width * 0.92, color=STYLE[m]["color"], label=MODEL_NAMES[m],
               hatch=hatches.get(m), edgecolor="white", linewidth=0.5, yerr=err,
               error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#0b0b0b"})
    ax.set_xticks(np.arange(len(metrics)), [lbl for _, lbl in metrics])
    ax.yaxis.set_major_formatter(FuncFormatter(_comma))
    ax.set_ylabel("Значение метрики, отн. ед.")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    _save(fig, path)


def distribution(x: np.ndarray, dists: dict, path: Path, xlabel: str, bins: int) -> None:
    """Гистограмма и эмпирическая функция распределения с гипотетическими законами."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.hist(x, bins=bins, density=True, color="#b7d3f6", edgecolor="white", label="гистограмма")
    grid = np.linspace(x.min(), x.max(), 400)
    styles = [("#0b0b0b", "-"), ("#e34948", "--")]
    for (name, d), (c, ls) in zip(dists.items(), styles):
        ax.plot(grid, d.pdf(grid), color=c, ls=ls, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Плотность распределения, 1/отн. ед.")
    ax.set_title("а) гистограмма и гипотетические плотности", loc="left")
    ax.legend(fontsize=8)
    ax = axes[1]
    xs = np.sort(x)
    ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color="#2a78d6",
            label="статистическая ФР $F^*(x)$")
    for (name, d), (c, ls) in zip(dists.items(), styles):
        ax.plot(grid, d.cdf(grid), color=c, ls=ls, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Функция распределения, отн. ед.")
    ax.set_title("б) статистическая и гипотетические ФР", loc="left")
    ax.legend(fontsize=8, loc="lower right")
    for a in axes:
        a.xaxis.set_major_formatter(FuncFormatter(_comma))
        a.yaxis.set_major_formatter(FuncFormatter(_comma))
    fig.tight_layout()
    _save(fig, path)


def factorial_pareto(df: pd.DataFrame, path: Path, title: str) -> None:
    """Диаграмма Парето модулей коэффициентов чувствительности ПФЭ."""
    d = df[df.coef != "a0"].copy()
    d["abs"] = d.value.abs()
    d = d.sort_values("abs")
    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.27 * len(d) + 1)))
    colors = ["#2a78d6" if v > 0 else "#e34948" for v in d.value]
    ax.barh(np.arange(len(d)), d["abs"], color=colors, edgecolor="white")
    thr = float(d.t_crit.iloc[0] * d.se.iloc[0])
    ax.axvline(thr, color="#0b0b0b", ls="--", lw=1)
    ax.annotate("порог значимости", (thr, len(d) - 0.6), xytext=(4, 0), textcoords="offset points",
                fontsize=8)
    labels = [c.replace("a_", "").replace("eps", "ε") for c in d.coef]
    ax.set_yticks(np.arange(len(d)), [f"$a_{{{lab}}}$" for lab in labels])
    ax.xaxis.set_major_formatter(FuncFormatter(_comma))
    ax.set_xlabel("Модуль коэффициента чувствительности |a|, отн. ед.")
    ax.set_title(title, loc="left")
    ax.legend(handles=[patches.Patch(color="#2a78d6", label="a > 0"),
                       patches.Patch(color="#e34948", label="a < 0")], fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


# ------------------------------------------------------------------ схема алгоритма (ГОСТ 19.701-90)
def _block(ax, kind: str, x: float, y: float, text: str, w: float = 4.0, h: float = 0.62) -> None:
    """Символ схемы по ГОСТ 19.701-90 с текстом."""
    kw = {"fill": False, "lw": 1.0, "edgecolor": "black"}
    if kind == "terminal":                        # пуск-останов: прямоугольник со скруглёнными сторонами
        ax.add_patch(patches.FancyBboxPatch((x - w / 2 + h / 2, y - h / 2), w - h, h,
                                            boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                            mutation_aspect=1, **kw))
    elif kind == "process":                       # процесс
        ax.add_patch(patches.Rectangle((x - w / 2, y - h / 2), w, h, **kw))
    elif kind == "predefined":                    # предопределённый процесс
        ax.add_patch(patches.Rectangle((x - w / 2, y - h / 2), w, h, **kw))
        for dx in (0.18, w - 0.18):
            ax.plot([x - w / 2 + dx] * 2, [y - h / 2, y + h / 2], color="black", lw=1.0)
    elif kind == "data":                          # данные: параллелограмм
        s = 0.25
        ax.add_patch(patches.Polygon([(x - w / 2 + s, y + h / 2), (x + w / 2, y + h / 2),
                                      (x + w / 2 - s, y - h / 2), (x - w / 2, y - h / 2)], closed=True, **kw))
    elif kind == "decision":                      # решение: ромб
        ax.add_patch(patches.Polygon([(x, y + h * 0.9), (x + w / 2, y), (x, y - h * 0.9), (x - w / 2, y)],
                                     closed=True, **kw))
    elif kind == "loop_start":                    # граница цикла (начало)
        c = 0.18
        ax.add_patch(patches.Polygon([(x - w / 2 + c, y + h / 2), (x + w / 2 - c, y + h / 2),
                                      (x + w / 2, y + h / 2 - c), (x + w / 2, y - h / 2),
                                      (x - w / 2, y - h / 2), (x - w / 2, y + h / 2 - c)], closed=True, **kw))
    elif kind == "loop_end":                      # граница цикла (конец)
        c = 0.18
        ax.add_patch(patches.Polygon([(x - w / 2, y + h / 2), (x + w / 2, y + h / 2),
                                      (x + w / 2, y - h / 2 + c), (x + w / 2 - c, y - h / 2),
                                      (x - w / 2 + c, y - h / 2), (x - w / 2, y - h / 2 + c)], closed=True, **kw))
    ax.text(x, y, text, ha="center", va="center", fontsize=7.4, linespacing=1.1)


def flowchart(blocks: list[tuple], path: Path) -> None:
    """Схема алгоритма: блоки (номер, вид, текст) сверху вниз, линии потока без стрелок
    (направление сверху вниз — основное по ГОСТ 19.701-90), номера блоков слева сверху."""
    n = len(blocks)
    step, h = 0.76, 0.6                          # шаг между блоками и высота блока, усл. ед.
    scale = 0.62                                 # дюймов на условную единицу: размер рисунка по данным
    fig, ax = plt.subplots(figsize=(4.8 * scale, ((n - 1) * step + 1.4) * scale))
    ys = [-(i * step) for i in range(n)]
    for i, (num, kind, text) in enumerate(blocks):
        y = ys[i]
        _block(ax, kind, 0.0, y, text, h=h)
        ax.text(-2.3, y + h / 2 + 0.02, str(num), fontsize=7, ha="left", va="bottom")
        if i < n - 1:
            top = y - (h * 0.9 if kind == "decision" else h / 2)
            nxt = ys[i + 1] + (h * 0.9 if blocks[i + 1][1] == "decision" else h / 2)
            ax.plot([0, 0], [top, nxt], color="black", lw=1.0)
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(ys[-1] - 0.7, 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, path)


def program_scheme(path: Path) -> None:
    """Структурная схема программы: модули (символ «предопределённый процесс») и связи вызова."""
    boxes = {
        "main.py": (7.0, 3.0, "управление этапами\nэксперимента"),
        "config.py": (14.2, 3.0, "параметры (используются\nвсеми модулями)"),
        "plots.py": (1.6, 1.6, "графики, схемы"),
        "analysis.py": (5.2, 1.6, "статистическая\nобработка"),
        "experiment.py": (8.8, 1.6, "планы эксперимента,\nрасчёт конфигураций"),
        "data.py": (12.4, 1.6, "корпус, сегментация,\nэталонная разметка"),
        "annotate.py": (16.0, 1.6, "аннотирование статьи\nпо ссылке"),
        "models.py": (7.0, 0.2, "модели и базовые\nметоды"),
        "features.py": (10.6, 0.2, "признаки\nпредложений"),
        "metrics.py": (14.2, 0.2, "ROUGE, отбор k\nпредложений"),
    }
    edges = [("main.py", "plots.py"), ("main.py", "analysis.py"), ("main.py", "experiment.py"),
             ("main.py", "data.py"), ("experiment.py", "models.py"), ("experiment.py", "features.py"),
             ("experiment.py", "metrics.py"), ("data.py", "metrics.py"), ("annotate.py", "data.py"),
             ("annotate.py", "features.py"), ("annotate.py", "models.py"), ("annotate.py", "metrics.py")]
    w, h = 3.2, 0.95
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for a, b in edges:
        (x1, y1, _), (x2, y2, _) = boxes[a], boxes[b]
        same_row = abs(y1 - y2) < 1e-9          # связь внутри ряда — горизонтальная
        start = (x1 - w / 2, y1) if same_row else (x1, y1 - h / 2)
        end = (x2 + w / 2, y2) if same_row else (x2, y2 + h / 2)
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "-|>", "color": "#52514e", "lw": 0.8, "shrinkA": 0, "shrinkB": 0})
    for name, (x, y, text) in boxes.items():
        ls = "--" if name == "config.py" else "-"
        ax.add_patch(patches.Rectangle((x - w / 2, y - h / 2), w, h, fill=True, facecolor="white",
                                       edgecolor="black", lw=1.0, ls=ls, zorder=3))
        for dx in (0.15, w - 0.15):
            ax.plot([x - w / 2 + dx] * 2, [y - h / 2, y + h / 2], color="black", lw=1.0, ls=ls, zorder=4)
        ax.text(x, y + 0.18, name, ha="center", va="center", fontsize=9, fontweight="bold", zorder=5)
        ax.text(x, y - 0.17, text, ha="center", va="center", fontsize=7.5, zorder=5, linespacing=1.0)
    ax.set_xlim(-0.2, 17.8)
    ax.set_ylim(-0.5, 3.6)
    ax.axis("off")
    _save(fig, path)
