#!/usr/bin/env python3
"""Верификация корпуса: сверка OCR-текста с текстовым слоем PDF (эталон).

Прогоняет каждый PDF из tests/data/real через CLI `rustocr <pdf> --json`
(без поднятия сервера), собирает текст страниц и сверяет числа
против pdftotext по канонизации bench/compare_nums.py (разряды тысяч,
десятичный разделитель, юниты м²/м³). Итог — покрытие чисел по каждому
документу и суммарное; код возврата 1, если покрытие ниже порога.

Использование:
    python3 scripts/verify_corpus.py [порог-процентов]   # по умолчанию 99
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REAL = REPO / "tests" / "data" / "real"
OUT = Path("/tmp/rustocr_verify")
THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 99.0

NUM_RE = re.compile(r"\d+(?:[.,]\d+)+|\d+")


def join_thousands(t: str) -> str:
    """«1 082 006,66» → «1082006,66»; соседние числа не склеиваем.

    Отличие от bench/compare_nums.py: дробные части маскируются перед
    склейкой, иначе «26,00 312,00» (одиночный пробел между колонками в
    OCR-тексте) превращается в «26,00312,00» и теряет ОБА числа из
    сравнения. В эталоне pdftotext -layout колонки разведены большими
    прогалами, поэтому там это не стреляло.
    """
    fracs: list[str] = []

    def mask(m: re.Match) -> str:
        fracs.append(m.group(0))
        return "\u0001" * len(m.group(0))

    masked = re.sub(r"[.,]\d+", mask, t)
    joined = re.sub(r"(?<=\d)[ \u00a0](?=\d{3}(?!\d))", "", masked)
    return re.sub("\u0001+", lambda m: fracs.pop(0), joined)


def norm_num(tok: str) -> tuple:
    parts = re.split(r"[.,]", tok)
    return ("".join(parts[:-1]), parts[-1]) if len(parts) > 1 else (tok, "")


def canon(text: str) -> Counter:
    t = re.sub(r"[мм][23]\b", "мX", text)
    t = join_thousands(t)
    return Counter(norm_num(tok) for tok in NUM_RE.findall(t))


def show(n: tuple) -> str:
    return n[0] + ("," + n[1] if n[1] else "")


def find_binary() -> Path:
    for cand in (REPO / "target/release/rustocr", REPO / "bin/rustocr"):
        if cand.exists():
            return cand
    sys.exit("не найден бинарник rustocr (target/release/ или bin/)")


def main() -> int:
    bin_path = find_binary()
    OUT.mkdir(parents=True, exist_ok=True)
    tot_ref = tot_found = 0
    below = []
    for pdf in sorted(REAL.glob("*.pdf")):
        env = {
            "ORT_DYLIB_PATH": str(REPO / "runtime/libonnxruntime.so.1.20.0"),
            "RUSTOCR_MODELS": str(REPO / "models"),
            "PATH": "/usr/bin:/bin",
        }
        res = subprocess.run(
            [str(bin_path), str(pdf), "--json"],
            env=env, capture_output=True, text=True, timeout=1200,
        )
        if res.returncode != 0:
            print(f"[FAIL] {pdf.name}: код {res.returncode}: {res.stderr.strip()[:200]}")
            below.append(pdf.name)
            continue
        try:
            pages = json.loads(res.stdout).get("pages", [])
        except json.JSONDecodeError as e:
            print(f"[FAIL] {pdf.name}: stdout не JSON: {e}")
            below.append(pdf.name)
            continue
        ocr_text = "\n".join(p.get("text", "") for p in pages if "text" in p)
        ref = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            capture_output=True, text=True,
        ).stdout
        ref_c, ocr_c = canon(ref), canon(ocr_text)
        n_ref = sum(ref_c.values())
        if n_ref == 0:
            # скан без текстового слоя — эталона нет, сверять нечего
            print(f"[skip] {pdf.name}: текстовый слой пуст (скан)")
            continue
        found = sum(min(c, ocr_c.get(n, 0)) for n, c in ref_c.items())
        tot_ref += n_ref
        tot_found += found
        cov = 100.0 * found / max(1, n_ref)
        mark = "ok " if cov >= THRESHOLD else "НИЖЕ"
        print(f"[{mark}] {pdf.name}: чисел {n_ref}, найдено {found} ({cov:.1f}%)")
        if cov < THRESHOLD:
            below.append(pdf.name)
            miss = {n: c - min(c, ocr_c.get(n, 0)) for n, c in ref_c.items() if c > ocr_c.get(n, 0)}
            top = sorted(miss.items(), key=lambda kv: -kv[1])[:10]
            print("       не хватает:", ", ".join(f"{show(n)}×{c}" for n, c in top))
    total_cov = 100.0 * tot_found / max(1, tot_ref)
    print(f"\nИТОГО: {tot_found}/{tot_ref} = {total_cov:.1f}% чисел распознано верно")
    if below:
        print("ниже порога:", ", ".join(below))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
