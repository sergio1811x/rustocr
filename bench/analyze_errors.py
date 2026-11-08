#!/usr/bin/env python3
"""Глубокий разбор ошибок OCR против текстового слоя PDF.

Для каждого числа из эталона (pdftotext), которого нет в OCR:
 1) ищем ближайший кандидат среди «лишних» чисел OCR той же страницы
    (edit distance по цифрам) и классифицируем:
       замена цифры / потеря-лишняя цифра / запятая↔точка / склейка чисел;
 2) если кандидата нет — смотрим строку эталона, в которой число жило:
       все числа строки пропали → «строка потеряна»,
       часть чисел строки на месте → «ячейка потеряна».

Использование:
    python3 analyze_errors.py [summary.json] [--pages]

    --pages  печатать ещё и разбивку потерь по страницам.
Выход: текстовый отчёт + errors_detail.json рядом со сводкой.
"""
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REAL = REPO / "tests/data/real"
SUMMARY = Path(
    sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
    else "/tmp/rustocr_bench/out/summary_200dpi.json"
)
SHOW_PAGES = "--pages" in sys.argv


from compare_nums import NUM_RE, canon, show  # noqa: E402  (тот же каталог)


def canon_lines(text: str):
    """Числа по строкам — [(строка, Counter), ...] для «живости» строки."""
    out = []
    for line in text.splitlines():
        c = canon(line)
        if c:
            out.append((line.strip(), c))
    return out


def digits(s) -> str:
    return re.sub(r"\D", "", s[0]) + re.sub(r"\D", "", s[1])


def lev(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def classify(miss, extra):
    """Классифицировать пропажу по лучшему кандидату из extra.

    Возвращает (тип, кандидат, пояснение) или (None, None, None).
    """
    md, best = digits(miss), None
    best_d = 10**9
    for cand in extra:
        if extra[cand] <= 0:
            continue
        cd = digits(cand)
        d = lev(md, cd)
        # штраф за сильную разницу длины, чтобы «3» не примерялось к «3744»
        d += abs(len(md) - len(cd)) * 2
        if d < best_d:
            best_d, best = d, cand
    if best is None or best_d > 4:
        return None, None, None
    bd = digits(best)
    if md == bd:
        return "запятая/точка", best, f"{show(miss)} → {show(best)}"
    if len(md) == len(bd) and best_d <= 1:
        diff = [(a, b) for a, b in zip(md, bd) if a != b]
        return "замена цифры", best, f"{show(miss)} → {show(best)} ({'|'.join(a+'↔'+b for a, b in diff)})"
    if abs(len(md) - len(bd)) == 1 and best_d <= 1:
        return "потеря/лишняя цифра", best, f"{show(miss)} → {show(best)}"
    if md in bd and len(bd) > len(md):
        return "склейка чисел", best, f"{show(miss)} влилось в {show(best)}"
    if bd in md and len(md) > len(bd):
        return "разрыв числа", best, f"{show(miss)} развалилось, часть = {show(best)}"
    return "прочая замена", best, f"{show(miss)} → {show(best)}"


def ref_page_text(pdf: Path, page: int) -> str:
    r = subprocess.run(
        ["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(pdf), "-"],
        capture_output=True,
        text=True,
    )
    return r.stdout


def main():
    data = json.loads(SUMMARY.read_text())
    types = Counter()
    per_doc = {}
    examples = defaultdict(list)
    detail = []

    for doc in data:
        pdf = REAL / doc["file"]
        doc_types = Counter()
        for pi, page in enumerate(doc["pages"], 1):
            if "error" in page or "text" not in page:
                continue
            ref = ref_page_text(pdf, pi)
            ref_c = canon(ref)
            ocr_c = canon(page["text"])
            # лишние числа OCR (кандидаты на замену)
            extra = ocr_c - ref_c
            missing = ref_c - ocr_c
            if not missing:
                continue
            ref_lines = canon_lines(ref)
            for miss, cnt in missing.items():
                for _ in range(cnt):
                    typ, cand, note = classify(miss, extra)
                    if typ:
                        doc_types[typ] += 1
                        types[typ] += 1
                        if cand:
                            extra[cand] -= 1
                        if len(examples[typ]) < 12:
                            examples[typ].append(f"{doc['file']} стр.{pi}: {note}")
                        detail.append(
                            {"file": doc["file"], "page": pi, "type": typ, "note": note}
                        )
                        continue
                    # кандидата нет: строка потеряна целиком или ячейка?
                    # ищем строку с числом с ХУДШИМ покрытием OCR —
                    # число могло встречаться в нескольких строках
                    line_txt, alive, total = "", 0, 0
                    for txt, c in ref_lines:
                        if c[miss] > 0:
                            a = sum(
                                min(k, ocr_c.get(n, 0)) for n, k in c.items()
                            )
                            t_ = sum(c.values())
                            if not line_txt or a / t_ < alive / total:
                                line_txt, alive, total = txt, a, t_
                            if a == 0:
                                break  # хуже уже не будет
                    if total and alive == 0:
                        typ = "строка потеряна"
                    elif total and alive < total:
                        typ = "ячейка потеряна"
                    else:
                        typ = "не найдено в эталоне-строках"
                    doc_types[typ] += 1
                    types[typ] += 1
                    if len(examples[typ]) < 12:
                        short = line_txt[:90]
                        examples[typ].append(
                            f"{doc['file']} стр.{pi}: {miss} · строка: {short}"
                        )
                    detail.append(
                        {
                            "file": doc["file"],
                            "page": pi,
                            "type": typ,
                            "number": miss,
                            "line": line_txt[:200],
                        }
                    )
        per_doc[doc["file"]] = doc_types

    print("=== ТИПЫ ОШИБОК (все документы) ===")
    for t, c in types.most_common():
        print(f"  {t}: {c}")
    print("\n=== ПО ДОКУМЕНТАМ ===")
    for f, c in per_doc.items():
        tot = sum(c.values())
        if tot:
            print(f"  {f}: {tot} :: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))
    print("\n=== ПРИМЕРЫ ===")
    for t, c in types.most_common():
        print(f"\n[{t}]")
        for e in examples[t][:8]:
            print("   ", e)
    if SHOW_PAGES:
        print("\n=== ПО СТРАНИЦАМ (топ-20) ===")
        pc = Counter((d["file"], d["page"]) for d in detail)
        for (f, p), c in pc.most_common(20):
            print(f"  {f} стр.{p}: {c}")
    out = SUMMARY.parent / "errors_detail.json"
    out.write_text(json.dumps(detail, ensure_ascii=False, indent=1))
    print(f"\nдетали → {out}")


if __name__ == "__main__":
    main()
