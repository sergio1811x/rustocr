#!/usr/bin/env python3
"""Точный диф потерянных чисел: строка эталона vs строки OCR страницы.

Для каждого пропавшего числа берём строку эталона (pdftotext -layout),
где оно жило, и показываем ближайшую по числам строку OCR — видно,
что именно произошло (числа нет / прочитано иначе / строка пропала).

    python3 bench/diff_misses.py <summary.json> <фрагмент-имени-файла>
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REAL = REPO / "tests/data/real"
SUMMARY = Path(sys.argv[1] if len(sys.argv) > 1 else "")
ONLY = sys.argv[2] if len(sys.argv) > 2 else ""

from compare_nums import NUM_RE, canon, show  # noqa: E402  (тот же каталог)


def ref_page(pdf: Path, page: int) -> str:
    r = subprocess.run(
        ["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(pdf), "-"],
        capture_output=True,
        text=True,
    )
    return r.stdout


def main():
    data = json.loads(SUMMARY.read_text())
    for doc in data:
        if ONLY and ONLY not in doc["file"]:
            continue
        pdf = REAL / doc["file"]
        n_miss = 0
        for pi, page in enumerate(doc["pages"], 1):
            if "error" in page or "text" not in page:
                continue
            ref = ref_page(pdf, pi)
            ref_c, ocr_c = canon(ref), canon(page["text"])
            missing = ref_c - ocr_c
            if not missing:
                continue
            ocr_lines = [l.strip() for l in page["text"].splitlines() if l.strip()]
            for miss, cnt in missing.items():
                for _ in range(min(cnt, 2)):  # максимум 2 показа на число
                    # строка эталона с этим числом
                    rline = ""
                    for l in ref.splitlines():
                        if miss in canon(l):
                            rline = l.strip()
                            break
                    # ближайшая строка OCR: максимум общих чисел с эталоном
                    best, best_common = "", -1
                    rc = canon(rline) if rline else Counter()
                    for ol in ocr_lines:
                        oc = canon(ol)
                        common = sum((oc & rc).values())
                        if common > best_common:
                            best_common, best = common, ol
                    n_miss += 1
                    print(f"— {doc['file']} стр.{pi} ЧИСЛО {show(miss)}")
                    print(f"   эталон : {rline[:110]}")
                    print(f"   OCR    : {best[:110]}")
        print(f"### {doc['file']}: показано {n_miss} потерь")


if __name__ == "__main__":
    main()
