#!/usr/bin/env python3
"""Параллельная загрузка двух разных PDF в один сервер.

Раньше временная папка растеризации называлась по PID процесса —
два одновременных запроса писали в одну папку и портили друг друга.
Теперь у каждого запроса своё уникальное имя (PID + наносекунды + счётчик).

Проверяем: два самых маленьких PDF, сначала по очереди (эталон),
потом 3 раунда одновременно — текст и число страниц должны совпадать.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REAL = REPO / "tests/data/real"
PORT = 8123


def start_server() -> subprocess.Popen:
    env = dict(os.environ)
    env["ORT_DYLIB_PATH"] = str(REPO / "runtime/libonnxruntime.so.1.20.0")
    env["RUSTOCR_MODELS"] = str(REPO / "models")
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


def post(pdf: Path):
    data = pdf.read_bytes()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/ocr",
        data=data,
        headers={"Content-Type": "application/pdf"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read())
    return pdf.name, body, time.time() - t0


def main():
    pdfs = sorted(REAL.glob("*.pdf"), key=lambda p: p.stat().st_size)
    assert len(pdfs) >= 2, "нужно минимум два PDF"
    a, b = pdfs[0], pdfs[1]
    print(f"PDF для теста: {a.name} ({a.stat().st_size//1024} КБ), "
          f"{b.name} ({b.stat().st_size//1024} КБ)")
    server = start_server()
    ok = True
    try:
        # эталон: по очереди
        ref = {}
        for p in (a, b):
            name, body, dt = post(p)
            assert "error" not in body, f"{name}: {body.get('error')}"
            ref[name] = body
            print(f"эталон {name}: страниц {len(body.get('pages', []))}, {dt:.1f} с")

        # параллельно, несколько раундов
        for rnd in range(3):
            with ThreadPoolExecutor(max_workers=2) as ex:
                futs = [ex.submit(post, p) for p in (a, b)]
                for f in futs:
                    name, body, dt = f.result()
                    if "error" in body:
                        print(f"раунд {rnd}: {name}: ОШИБКА: {body['error']}")
                        ok = False
                        continue
                    same = (
                        json.dumps(body.get("text"), ensure_ascii=False)
                        == json.dumps(ref[name].get("text"), ensure_ascii=False)
                    )
                    npg = len(body.get("pages", []))
                    print(f"раунд {rnd}: {name}: страниц {npg}, "
                          f"текст совпадает: {same}, {dt:.1f} с")
                    if not same or npg != len(ref[name].get("pages", [])):
                        ok = False
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    print("ИТОГ:", "OK — параллельные PDF не мешают друг другу" if ok else "ПРОВАЛ")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
