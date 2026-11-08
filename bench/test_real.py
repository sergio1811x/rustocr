#!/usr/bin/env python3
"""Прогон реальных PDF-смет через RustOCR-сервер.

1) растеризация pdftoppm (как скан),
2) POST /ocr на каждую страницу,
3) сводка + сохранение JSON-результатов.

Пути рассчитываются от корня репозитория, вывод — в /tmp/rustocr_bench
(переопределяется переменными PNG_DIR / OUT_DIR). Возобновляемо: документы,
уже обработанные ранее, пропускаются (FORCE=1 — прогнать заново).
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REAL = REPO / "tests/data/real"
MODELS = Path(os.environ.get("RUSTOCR_MODELS", str(REPO / "models")))
PNG_DIR = Path(os.environ.get("PNG_DIR", "/tmp/rustocr_bench/png"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/tmp/rustocr_bench/out"))
DOCS_DIR = OUT_DIR / "docs"
DPI = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ONLY = sys.argv[2] if len(sys.argv) > 2 else ""  # фильтр по имени файла
PORT = int(os.environ.get("PORT", "8080"))
FORCE = os.environ.get("FORCE") == "1"


def start_server() -> subprocess.Popen:
    env = dict(os.environ)
    env["ORT_DYLIB_PATH"] = str(REPO / "runtime/libonnxruntime.so.1.20.0")
    env["RUSTOCR_MODELS"] = str(MODELS)
    env["RUSTOCR_PORT"] = str(PORT)
    proc = subprocess.Popen(
        [str(REPO / "bin/rustocr"), "serve", "--port", str(PORT)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/health", timeout=2
            ) as r:
                if r.status == 200:
                    return proc
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("сервер не поднялся")


def rasterize(pdf: Path) -> list:
    stem = pdf.stem
    target = PNG_DIR / stem
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pdftoppm", "-r", str(DPI), "-png", str(pdf), str(target / "p")],
        check=True,
    )
    return sorted(target.glob("p-*.png"))


def ocr(png: Path) -> dict:
    data = png.read_bytes()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/ocr",
        data=data,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = r.read()
    return json.loads(body), time.time() - t0


def merge_summaries() -> list:
    """Собрать сводку из всех сохранённых документов."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    docs = []
    for f in sorted(DOCS_DIR.glob("doc_*.json")):
        docs.append(json.loads(f.read_text()))
    return docs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(REAL.glob("*.pdf"))
    if ONLY:
        keys = [k.strip() for k in ONLY.split(",")]
        pdfs = [p for p in pdfs if any(k in p.name for k in keys)]
    # уже сделанные (возобновление после обрыва)
    done = {f.stem for f in DOCS_DIR.glob("doc_*.json")}
    todo = [p for p in pdfs if FORCE or p.stem not in done]
    print(f"PDF всего: {len(pdfs)}, осталось: {len(todo)}, DPI: {DPI}", flush=True)
    if not todo:
        summary = merge_summaries()
        (OUT_DIR / f"summary_{DPI}dpi.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1)
        )
        print(f"Готово, ничего не осталось. JSON → {OUT_DIR / f'summary_{DPI}dpi.json'}")
        return
    server = start_server()
    print("сервер поднялся, PID", server.pid, flush=True)
    summary = []
    try:
        for pdf in todo:
            pages = rasterize(pdf)
            pages.sort(key=lambda p: int(p.stem.split("-")[-1]))
            doc = {"file": pdf.name, "pages": []}
            t_all = 0.0
            n_blocks = 0
            confs = []
            for png in pages:
                try:
                    res, dt = ocr(png)
                except Exception as e:  # noqa: BLE001
                    print(f"  !! {png.name}: {e}")
                    doc["pages"].append({"page": png.name, "error": str(e)})
                    continue
                doc["pages"].append(
                    {
                        "page": png.name,
                        "blocks": len(res.get("blocks", [])),
                        "conf": round(
                            sum(b["confidence"] for b in res["blocks"])
                            / max(1, len(res["blocks"])),
                            3,
                        ),
                        "timing_ms": res["timing_ms"],
                        "http_s": round(dt, 2),
                        "text": res.get("text", ""),
                        "raw": res,
                    }
                )
                t_all += dt
                n_blocks += len(res.get("blocks", []))
                confs.extend(b["confidence"] for b in res.get("blocks", []))
            doc["total_s"] = round(t_all, 1)
            doc["n_blocks"] = n_blocks
            doc["avg_conf"] = round(sum(confs) / max(1, len(confs)), 3)
            (DOCS_DIR / f"doc_{pdf.stem}.json").write_text(
                json.dumps(doc, ensure_ascii=False)
            )
            summary.append(doc)
            print(
                f"{pdf.name}: {len(pages)} стр., {n_blocks} блоков, "
                f"увер. {doc['avg_conf']}, {t_all:.1f} c HTTP",
                flush=True,
            )
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    summary = merge_summaries()
    (OUT_DIR / f"summary_{DPI}dpi.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1)
    )
    print("JSON →", OUT_DIR / f"summary_{DPI}dpi.json")


if __name__ == "__main__":
    main()
