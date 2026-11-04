"""Аннотирование статьи по ссылке, из файла или из текста обученной моделью.

Запуск:  python annotate.py https://example.ru/news/123 [--r 0.1 | --k 4] [--model logreg] [--all]
         python annotate.py --file статья.txt
         python annotate.py                      (программа сама спросит ссылку)

Модель один раз обучается на статьях корпуса Gazeta с гиперпараметрами базовой
точки (results/base_point.json) и сохраняется в data/cache; повторные запуски
используют сохранённую модель. Аннотации записываются в results/annotations.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
from scipy import sparse

import config
import data
from data import Corpus
from features import FeatureBuilder
from metrics import select_top_k, summary_length
from models import MODEL_NAMES, SentenceScorer

log = logging.getLogger("annotate")

ANNOTATOR_DOCS = 3000          # статей корпуса для обучения модели-аннотатора
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


# ------------------------------------------------------------------ извлечение текста из HTML
class ArticleParser(HTMLParser):
    """Собирает заголовок и текстовые блоки страницы, пропуская меню, скрипты и подвалы."""

    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg",
            "button", "select", "iframe"}
    BLOCK = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "section", "article", "td", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.in_title = self.in_h1 = self.in_p = False
        self.title, self.h1 = "", ""
        self.buf: list[str] = []
        self.blocks: list[tuple[str, bool]] = []      # (текст блока, был ли это <p>)

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        if text:
            self.blocks.append((text, self.in_p))
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        elif tag in self.BLOCK:
            self._flush()
            if tag == "p":
                self.in_p = True
        if tag == "title":
            self.in_title = True
        if tag == "h1":
            self.in_h1 = True

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag in self.BLOCK:
            self._flush()
            if tag == "p":
                self.in_p = False
        if tag == "title":
            self.in_title = False
        if tag == "h1":
            self.in_h1 = False

    def handle_data(self, d):
        if self.in_title:
            self.title += d
        if self.in_h1 and not self.skip_depth:
            self.h1 += d
        if not self.skip_depth:
            self.buf.append(d)


def extract_article(html: str) -> tuple[str, str]:
    """Заголовок и текст статьи: абзацы <p>, а если их мало — длинные текстовые блоки страницы."""
    p = ArticleParser()
    p.feed(html)
    p._flush()
    title = re.sub(r"\s+", " ", p.h1 or p.title).strip()
    seen, paras, blocks = set(), [], []
    for text, is_p in p.blocks:
        if text in seen:
            continue
        seen.add(text)
        if is_p and len(text) >= 40:
            paras.append(text)
        if len(text) >= 80 and len(text.split()) >= 10:
            blocks.append(text)
    body = paras if sum(map(len, paras)) >= 400 else blocks
    return title, clean("\n".join(body))


def clean(text: str) -> str:
    """Удаляет сноски вида [1], [вд], [источник не указан] и знаки ударения (как в Википедии)."""
    text = re.sub(r"\[(?:\d+|[а-яa-z ]{1,25})\]", "", text, flags=re.I)
    return text.replace("́", "")


def normalize_url(url: str) -> str:
    """Добавляет схему и переводит кириллицу в адресе в допустимый вид (IDN, %-кодирование)."""
    url = url.strip().strip('"\'<>')
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    p = urllib.parse.urlsplit(url)
    host = p.hostname.encode("idna").decode("ascii") if p.hostname else ""
    netloc = host + (f":{p.port}" if p.port else "")
    safe = "/%:@!$&'()*+,;=~-._"
    return urllib.parse.urlunsplit((p.scheme, netloc, urllib.parse.quote(p.path, safe=safe),
                                    urllib.parse.quote(p.query, safe=safe + "?"), ""))


def fetch(url: str, timeout: float = 60, attempts: int = 2) -> str:
    """Загрузка страницы; кодировка — из заголовков HTTP, затем из <meta charset>, иначе UTF-8."""
    req = urllib.request.Request(normalize_url(url),
                                 headers={"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.8"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                charset = resp.headers.get_content_charset()
            break
        except (TimeoutError, urllib.error.URLError) as e:
            if attempt == attempts - 1:
                raise ConnectionError(f"не удалось загрузить страницу: {e}") from e
            time.sleep(2)
    if not charset:
        m = re.search(rb"charset=[\"']?([\w-]+)", raw[:4096], re.I)
        charset = m.group(1).decode() if m else "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


# ------------------------------------------------------------------ модель-аннотатор
def base_hyper() -> tuple[dict, dict]:
    """Базовая точка и гиперпараметры, выбранные в эксперименте (или значения по умолчанию)."""
    import json
    bp = config.RESULTS_DIR / "base_point.json"
    if bp.exists():
        d = json.loads(bp.read_text(encoding="utf-8"))
        return d["base"], d["hyper"]
    return ({"N": config.BASE["N"], "m": config.BASE["m"], "r": config.BASE["r"], "eps": 0.0},
            {"logreg": {"C": 0.03}, "svm": {"C": 0.003}, "nb": {"C": 1.0}, "rf": {"msl": 3}})


def load_annotator(model: str, n_docs: int = ANNOTATOR_DOCS) -> dict:
    """Загружает сохранённую модель или обучает новую на n_docs статьях корпуса Gazeta."""
    path = config.CACHE_DIR / f"annotator_{model}_{n_docs}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    base, hyper = base_hyper()
    print(f"Первый запуск: обучение модели «{MODEL_NAMES[model]}» на {n_docs} статьях корпуса Gazeta "
          "(1–2 мин)...", flush=True)
    t0 = time.perf_counter()
    docs, ids = data.process_split("train", n_docs)
    vocab = data.Vocabulary()
    corpus = data.build_corpora({"train": (docs, ids)}, vocab)["train"]
    fb = FeatureBuilder(int(base["m"]))
    X, D = fb.fit(corpus)
    h = hyper[model]
    scorer = SentenceScorer(model, C=h.get("C", 1.0), min_samples_leaf=h.get("msl", 3),
                            n_estimators=config.RF_N_ESTIMATORS, seed=config.SEED, n_jobs=-1)
    scorer.fit(X, D, corpus.labels)
    ann = {"model": model, "index": vocab.index, "stop": vocab.stop_mask(), "fb": fb, "scorer": scorer,
           "r": float(base["r"]), "m": int(base["m"]), "hyper": h, "n_docs": corpus.n_docs}
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ann, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Модель обучена за {time.perf_counter() - t0:.0f} с и сохранена: {path}", flush=True)
    return ann


def document_corpus(title: str, sentences: list[list[str]], ann: dict) -> Corpus:
    """Один документ в формате Corpus; слова вне словаря обучающей выборки в признаки не входят."""
    index, stop = ann["index"], ann["stop"]
    V = len(stop)
    known = [[index[t] for t in s if t in index] for s in sentences]
    title_ids = [index[t] for t in data.tokenize(title) if t in index]
    lens = np.array([len(s) for s in sentences])
    n = len(sentences)
    return Corpus(split="article", sent_offsets=np.array([0, n]),
                  tok=np.full(int(lens.sum()), -1, dtype=np.int32),
                  tok_offsets=np.concatenate([[0], np.cumsum(lens)]),
                  ref=np.zeros(0, dtype=np.int32), ref_offsets=np.array([0, 0]),
                  title_counts=data._count_matrix([title_ids], V, stop),
                  counts=data._count_matrix(known, V, stop) if known else sparse.csr_matrix((0, V)),
                  labels=np.zeros(n, dtype=np.int8), doc_ids=np.array([0]))


def annotate(title: str, text: str, ann: dict, r: float | None = None, k: int | None = None) -> dict:
    """Аннотация: k = ⌈r·n⌉ предложений с наибольшими оценками в исходном порядке."""
    # абзацы делятся на предложения по отдельности: строка без точки (подзаголовок) не склеится со следующей
    raw = [s for para in text.split("\n") for s in data.sentenize(para) if data.tokenize(s)]
    if len(raw) < 2:
        raise ValueError("в тексте меньше двух предложений — аннотировать нечего")
    corpus = document_corpus(title, [data.tokenize(s) for s in raw], ann)
    X, D = ann["fb"].transform(corpus)
    scores = ann["scorer"].score(X, D)
    k = k or summary_length(len(raw), r if r is not None else ann["r"])
    sel = select_top_k(scores, min(k, len(raw)))
    return {"title": title, "sentences": raw, "scores": scores, "selected": sel, "k": len(sel)}


# ------------------------------------------------------------------ вывод
def report(res: dict, source: str, ann: dict, show_all: bool) -> str:
    """Текст результата: аннотация и (по запросу) оценки всех предложений."""
    lines = [f"Заголовок: {res['title'] or '—'}", f"Источник: {source}",
             f"Модель: {MODEL_NAMES[ann['model']]}; предложений в статье: {len(res['sentences'])}; "
             f"в аннотации: {res['k']}", "", "АННОТАЦИЯ:"]
    for i in res["selected"]:
        lines.append(f"  [{i + 1}] {res['sentences'][i]}")
    if show_all:
        prob = ann["model"] in ("logreg", "rf")       # оценка — вероятность p_i; у SVM и НБК — «счёт»
        lines += ["", "Все предложения по убыванию оценки " + ("p_i" if prob else "s_i") +
                  " (★ — вошли в аннотацию):"]
        order = np.argsort(-res["scores"], kind="stable")
        chosen = set(res["selected"].tolist())
        for rank, i in enumerate(order, 1):
            mark = "★" if i in chosen else " "
            s = f"{res['scores'][i]:.3f}" if prob else f"{res['scores'][i]:+.3f}"
            lines.append(f"{mark} {rank:3d}. [{i + 1:3d}] {s}  {res['sentences'][i][:110]}")
    return "\n".join(lines)


def save(text: str) -> Path:
    """Сохранение аннотации в каталог results/annotations."""
    out = config.RESULTS_DIR / "annotations" / time.strftime("annotation_%Y%m%d_%H%M%S.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def run_once(source: str, ann: dict, args, kind: str = "url") -> None:
    """Аннотирование одной статьи (ссылка, файл или текст)."""
    if kind == "url":
        title, body = extract_article(fetch(source))
    else:
        content = Path(source).read_text(encoding="utf-8", errors="replace") if kind == "file" else source
        if kind == "file" and Path(source).suffix.lower() in {".html", ".htm"}:
            title, body = extract_article(content)
        else:                                   # обычный текст: первая строка — заголовок
            first, _, rest = content.strip().partition("\n")
            title, body = (first, rest) if rest.strip() else ("", first)
    if len(body) < 200:
        print("Не удалось извлечь текст статьи (страница пустая, требует входа или подгружает текст "
              "скриптом). Сохраните текст статьи в файл .txt и запустите: annotate.py --file файл.txt")
        return
    res = annotate(title, body, ann, r=args.r, k=args.k)
    out = report(res, source if kind != "text" else "текст из командной строки", ann, args.all)
    print("\n" + out)
    print(f"\nСохранено: {save(out)}")


def main() -> None:
    """Разбор параметров командной строки; диалоговый режим без параметров."""
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description="Аннотирование статьи по ссылке обученной моделью")
    ap.add_argument("url", nargs="?", help="ссылка на статью (если не указана — программа спросит)")
    ap.add_argument("--file", help="текстовый (.txt) или HTML-файл со статьёй; в .txt первая строка — заголовок")
    ap.add_argument("--text", help="текст статьи прямо в командной строке")
    ap.add_argument("--r", type=float, default=None, help="коэффициент сжатия (по умолчанию — r0 эксперимента)")
    ap.add_argument("--k", type=int, default=None, help="число предложений аннотации (вместо --r)")
    ap.add_argument("--model", choices=["logreg", "svm", "nb", "rf"], default="logreg")
    ap.add_argument("--all", action="store_true", help="показать оценки всех предложений")
    args = ap.parse_args()
    ann = load_annotator(args.model)
    try:
        if args.file:
            run_once(args.file, ann, args, "file")
        elif args.text:
            run_once(args.text, ann, args, "text")
        elif args.url:
            run_once(args.url, ann, args)
        else:                                   # диалоговый режим
            while True:
                url = input("\nВставьте ссылку на статью (пустая строка — выход): ").strip()
                if not url:
                    break
                try:
                    run_once(url, ann, args)
                except Exception as e:          # noqa: BLE001 — сообщаем и ждём следующую ссылку
                    print(f"Ошибка: {e}")
    except Exception as e:                      # noqa: BLE001
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
