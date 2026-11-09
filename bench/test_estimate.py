#!/usr/bin/env python3
"""Регрессионный тест сметного конвейера (--estimate) на корпусе реальных смет.

Прогоняет каждый PDF из tests/data/real через `rustocr <pdf> --estimate`
и сравнивает стабильную статистику (число разделов/позиций/ресурсов,
итоги, виды ошибок валидации) с золотым снимком tests/estimate/golden/.

    python3 bench/test_estimate.py            # сравнение с golden
    python3 bench/test_estimate.py --update   # пересоздать golden
    python3 bench/test_estimate.py имя_файла  # один документ (фильтр)

Код возврата: 0 — совпадение, 2 — расхождение/сбой. Снимки обновлять
осознанно: изменение статистики = изменение поведения парсера.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REAL = REPO / "tests/data/real"
GOLDEN = REPO / "tests/estimate/golden"
BIN = REPO / "target/release/rustocr"
OUT = Path(os.environ.get("ESTIMATE_OUT", "/tmp/rustocr_estimate"))
UPDATE = "--update" in sys.argv
ONLY = next((a for a in sys.argv[1:] if not a.startswith("-")), "")


def run_doc(pdf: Path) -> dict:
    """Прогнать одну смету, вернуть стабильную статистику."""
    out_json = OUT / (pdf.stem + ".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["ORT_DYLIB_PATH"] = str(REPO / "runtime/libonnxruntime.so.1.20.0")
    env["RUSTOCR_MODELS"] = str(REPO / "models")
    res = subprocess.run(
        [str(BIN), str(pdf), "--estimate", str(out_json)],
        env=env, capture_output=True, text=True, timeout=1200,
    )
    if res.returncode != 0:
        return {"error": f"код {res.returncode}: {res.stderr.strip()[:200]}"}
    try:
        doc = json.loads(out_json.read_text())
    except Exception as e:
        return {"error": f"JSON: {e}"}
    est = doc
    stats = est["validation"]["stats"]
    secs = []
    for s in est["sections"]:
        pos_n = len(s.get("positions", [])) + sum(len(u.get("positions", [])) for u in s.get("subsections", []))
        secs.append({"title": s["title"][:60], "positions": pos_n,
                     "total": s.get("total", {}).get("value") if s.get("total") else None})
    return {
        "pages": est["pages"],
        "sections": stats["sections"],
        "positions": stats["positions"],
        "resources": stats["resources"],
        "table_rows": stats["table_rows"],
        "unclassified_rows": stats["unclassified_rows"],
        "qty_price_checked": stats["qty_price_checked"],
        "section_totals_checked": stats["section_totals_checked"],
        "total": est["totals"].get("total", {}).get("value") if est["totals"].get("total") else None,
        "vat": est["totals"].get("vat", {}).get("value") if est["totals"].get("vat") else None,
        "total_with_vat": (est["totals"].get("total_with_vat", {}).get("value")
                           if est["totals"].get("total_with_vat") else None),
        "errors": sorted(i["kind"] for i in est["validation"]["errors"]),
        "warnings": sorted(i["kind"] for i in est["validation"]["warnings"]),
        "sections_detail": secs,
    }


def main() -> int:
    pdfs = sorted(REAL.glob("*.pdf"))
    if ONLY:
        pdfs = [p for p in pdfs if ONLY in p.name]
    if not pdfs:
        print("нет PDF в", REAL)
        return 2
    if UPDATE:
        GOLDEN.mkdir(parents=True, exist_ok=True)
    print(f"{'документ':<44} {'разд':>4} {'поз':>4} {'рес':>4} {'итог':>13} {'ош':>3} {'пре':>3}")
    print("-" * 84)
    bad = []
    for pdf in pdfs:
        snap = run_doc(pdf)
        if "error" in snap:
            print(f"{pdf.name:<44} СБОЙ: {snap['error']}")
            bad.append(pdf.name)
            continue
        total = snap["total"]
        total_s = f"{total:,.0f}".replace(",", " ") if total else "—"
        print(f"{pdf.name:<44} {snap['sections']:>4} {snap['positions']:>4} "
              f"{snap['resources']:>4} {total_s:>13} {len(snap['errors']):>3} {len(snap['warnings']):>3}")
        gold = GOLDEN / (pdf.stem + ".json")
        if UPDATE:
            gold.write_text(json.dumps(snap, ensure_ascii=False, indent=1))
        elif not gold.exists():
            print(f"    ⚠ нет золотого снимка (запусти с --update): {gold.name}")
            bad.append(pdf.name + ":no_golden")
        else:
            was = json.loads(gold.read_text())
            if was != snap:
                print(f"    ✗ РАСХОЖДЕНИЕ с золотым снимком:")
                for k in sorted(set(was) | set(snap)):
                    if was.get(k) != snap.get(k):
                        print(f"      {k}: {was.get(k)} → {snap.get(k)}")
                bad.append(pdf.name + ":diff")
    print("-" * 84)
    if UPDATE:
        print(f"золотые снимки записаны: {GOLDEN} ({len(pdfs)} шт.)")
        return 0
    if bad:
        print("РАСХОЖДЕНИЯ/СБОИ:", ", ".join(bad))
        return 2
    print(f"OK: {len(pdfs)} документов совпадают с золотыми снимками")
    return 0


if __name__ == "__main__":
    sys.exit(main())
