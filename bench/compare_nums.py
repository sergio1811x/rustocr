#!/usr/bin/env python3
"""Сверка чисел OCR против текстового слоя PDF (эталон).

Канонизация: юниты м2/м3 заменяем; пробелы между цифрами склеиваем
ТОЛЬКО как разряды тысяч — следующая группа обязана быть из ровно 3 цифр
(«1 082 006,66» → «1082006,66», но «54 10 115,56» не склеивается в одно:
pdftotext печатает соседние колонки через один пробел, это разные числа);
число нормализуем в (целая, дробная): последний разделитель — десятичный.
Так «21,2» == «21.2» (та же величина), но «8000,0» != «80000».
Покрытие по мультимножеству вхождений.
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REAL = REPO / "tests/data/real"
SUMMARY = Path(
    sys.argv[1] if len(sys.argv) > 1 else "/tmp/rustocr_bench/out/summary_200dpi.json"
)


NUM_RE = re.compile(r"\d+(?:[.,]\d+)+|\d+")


def join_thousands(t: str) -> str:
    """Склеить пробелы-разряды: «1 082 006,66» → «1082006,66».

    Склеиваем пробел между цифрами только если следующая группа —
    ровно 3 цифры, за которыми конец/разделитель/ещё разряд: это похоже
    на разряды тысяч. «54 10 115,56» — «10» из 2 цифр, не склеивается
    с «54»: так соседние колонки pdftotext не превращаются в одно число.
    """
    return re.sub(
        r"(?<=\d)[ \u00a0](?=\d{3}(?:[.,]|\s|\D|$))",
        "",
        t,
    )


def norm_num(tok: str) -> tuple:
    """('целая','дробная') — без учёта вида десятичного разделителя."""
    parts = re.split(r"[.,]", tok)
    if len(parts) == 1:
        return (tok, "")
    return ("".join(parts[:-1]), parts[-1])


def canon(text: str) -> Counter:
    t = re.sub(r"[мм][23]\b", "мX", text)  # юниты не должны склеиваться с числами
    t = join_thousands(t)
    return Counter(norm_num(tok) for tok in NUM_RE.findall(t))


def show(n: tuple) -> str:
    return n[0] + ("," + n[1] if n[1] else "")


def main():
    data = json.loads(SUMMARY.read_text())
    tot_ref = tot_found = 0
    misses = {}
    for doc in data:
        pdf = REAL / doc["file"]
        ref = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            capture_output=True,
            text=True,
        ).stdout
        ocr_text = "\n".join(p.get("text", "") for p in doc["pages"] if "text" in p)
        ref_c = canon(ref)
        ocr_c = canon(ocr_text)
        found = sum(min(cnt, ocr_c.get(num, 0)) for num, cnt in ref_c.items())
        tot_ref += sum(ref_c.values())
        tot_found += found
        cov = 100.0 * found / max(1, sum(ref_c.values()))
        print(f"{doc['file']}: чисел {sum(ref_c.values())}, найдено {found} ({cov:.1f}%)")
        miss = {n: c - min(c, ocr_c.get(n, 0)) for n, c in ref_c.items() if c > ocr_c.get(n, 0)}
        if miss:
            top = sorted(miss.items(), key=lambda kv: -kv[1])[:15]
            misses[doc["file"]] = miss
            print("   не хватает:", ", ".join(f"{show(n)}×{c}" for n, c in top))
    print(
        f"\nИТОГО: {tot_found}/{tot_ref} = "
        f"{100.0 * tot_found / max(1, tot_ref):.1f}% чисел распознано верно"
    )
    SUMMARY.parent.joinpath("misses.json").write_text(
        json.dumps(misses, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
