#!/usr/bin/env python3
# Стресс-тест: «плохие сканы» реальных смет.
# Берём PNG-страницы, отрисованные из реальных PDF (уже лежат в
# /tmp/rustocr_bench/png/<документ>/p-NNN.png, 200 dpi), создаём
# деградированные варианты и прогоняем через RustOCR-сервер.
#
# Деградации:
#   lowdpi    — уменьшение до ~120 dpi и обратно (пикселизация)
#   blur      — гауссово размытие 1.5 px (не в фокусе)
#   noisejpg  — шум сигма 15 + JPEG q=50 (фото телефона)
#   skew      — поворот на 1.8° + обрезка по холсту (криво положили)
#   shadow    — градиентная тень через страницу + лёгкий шум (фото на столе)
#   faded     — выцветший принт: грей + контраст до 35%
#
# Метрика: те же канонизированные числа (bench/compare_nums.py),
# эталон — pdftotext -layout. Сравнение против 200-dpi «чистого»
# прогона не делаем: считаем по каждому варианту напрямую.
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parents[1]
PNG_DIR = Path(os.environ.get("PNG_DIR", "/tmp/rustocr_bench/png"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/tmp/rustocr_bench/stress"))
PORT = int(os.environ.get("PORT", "8081"))
MAX_PAGES_PER_DOC = int(os.environ.get("MAX_PAGES_PER_DOC", "2"))  # скорость
REAL = REPO / "tests/data/real"

sys.path.insert(0, str(REPO / "bench"))
from compare_nums import canon  # noqa: E402


def start_server() -> subprocess.Popen:
    env = dict(os.environ)
    env["ORT_DYLIB_PATH"] = str(REPO / "runtime/libonnxruntime.so.1.20.0")
    env["RUSTOCR_MODELS"] = str(REPO / "models")
    env["RUSTOCR_PORT"] = str(PORT)
    proc = subprocess.Popen(
        [str(REPO / "bin/rustocr"), "serve", "--port", str(PORT)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("сервер не поднялся")


def degrade(kind: str, img: Image.Image) -> Image.Image:
    if kind == "lowdpi":
        w, h = img.size
        k = 0.6  # 200dpi -> ~120dpi
        small = img.resize((int(w * k), int(h * k)), Image.LANCZOS)
        return small.resize((w, h), Image.LANCZOS)
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(1.5))
    if kind == "noisejpg":
        arr = np.asarray(img.convert("RGB")).astype(np.int16)
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 15, arr.shape[:2])[:, :, None]
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr)
        buf = Path("/tmp/_stress.jpg")
        im.save(buf, quality=50)
        return Image.open(buf).convert("RGB")
    if kind == "skew":
        ang = 1.8
        rot = img.rotate(ang, expand=False, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
        return rot
    if kind == "shadow":
        arr = np.asarray(img.convert("RGB")).astype(np.int16)
        h, w = arr.shape[:2]
        # диагональная тень слева-сверху
        xs = np.linspace(90, 0, w)[None, :]
        ys = np.linspace(45, 0, h)[:, None]
        grad = (xs + ys).clip(0, 110)  # до ~43% затемнения
        arr = np.clip(arr - grad[:, :, None], 0, 255).astype(np.uint8)
        arr = arr.astype(np.int16)
        rng = np.random.default_rng(11)
        arr = np.clip(arr + rng.normal(0, 8, arr.shape[:2])[:, :, None], 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if kind == "faded":
        g = img.convert("L")
        arr = np.asarray(g).astype(np.float32)
        lo, hi = np.percentile(arr, (1, 99))
        arr = 200 + (arr - lo) / max(1.0, hi - lo) * 55  # 200..255 — бледный
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    raise ValueError(kind)


def ocr(png: Path) -> dict:
    data = png.read_bytes()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/ocr", data=data,
        headers={"Content-Type": "application/octet-stream"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def ref_page(pdf: Path, page: int) -> str:
    r = subprocess.run(
        ["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(pdf), "-"],
        capture_output=True, text=True,
    )
    return r.stdout


def main():
    kinds = sys.argv[1].split(",") if len(sys.argv) > 1 else [
        "lowdpi", "blur", "noisejpg", "skew", "shadow", "faded",
    ]
    doc_filter = os.environ.get("DOCS", "")  # подстроки через запятую
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    server = start_server()
    print("сервер поднялся, PID", server.pid, flush=True)
    results = []  # {kind, file, page, ref_n, got_n, matched, extra, time_s}
    try:
        docs = sorted(p for p in PNG_DIR.iterdir() if p.is_dir())
        if doc_filter:
            keys = [k.strip() for k in doc_filter.split(",")]
            docs = [d for d in docs if any(k in d.name for k in keys)]
        for d in docs:
            pdf = REAL / f"{d.name}.pdf"
            if not pdf.exists():
                continue
            pages = sorted(d.glob("p-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
            pages = pages[:MAX_PAGES_PER_DOC] + pages[-MAX_PAGES_PER_DOC:]
            seen = set()
            pages = [p for p in pages if not (p in seen or seen.add(p))]
            for png in pages:
                pageno = int(png.stem.split("-")[-1])
                ref = ref_page(pdf, pageno)
                ref_nums = canon(ref)
                base = Image.open(png).convert("RGB")
                for kind in kinds:
                    img = degrade(kind, base)
                    tmp = OUT_DIR / f"{d.name}_p{pageno:03d}_{kind}.png"
                    img.save(tmp)
                    t0 = time.time()
                    try:
                        res = ocr(tmp)
                    except Exception as e:  # noqa: BLE001
                        print(f"  !! {tmp.name}: {e}", flush=True)
                        results.append({"kind": kind, "file": d.name, "page": pageno,
                                        "error": str(e)})
                        continue
                    dt = time.time() - t0
                    got_nums = canon(res.get("text", ""))
                    from collections import Counter
                    matched = sum((ref_nums & got_nums).values())
                    results.append({
                        "kind": kind, "file": d.name, "page": pageno,
                        "ref_n": sum(ref_nums.values()),
                        "got_n": sum(got_nums.values()),
                        "matched": matched,
                        "time_s": round(dt, 2),
                    })
                    tmp.unlink()  # не копить гигабайты
                    r = results[-1]
                    if "ref_n" in r:
                        print(f"{kind:8s} {d.name[:28]:28s} p{pageno:02d} "
                              f"{r['matched']}/{r['ref_n']}", flush=True)
    finally:
        server.terminate()
    (OUT_DIR / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

    # сводка по видам деградации
    print("\n=== СВОДКА (числа: точные/эталон) ===")
    for kind in kinds:
        rs = [r for r in results if r["kind"] == kind and "ref_n" in r]
        if not rs:
            continue
        ref = sum(r["ref_n"] for r in rs)
        got = sum(r["matched"] for r in rs)
        acc = got / max(1, ref) * 100
        worst = min(rs, key=lambda r: r["matched"] / max(1, r["ref_n"]))
        print(f"{kind:8s}: {got:5d}/{ref:5d} = {acc:5.2f}%   "
              f"худший: {worst['file']} p{worst['page']} "
              f"({worst['matched']}/{worst['ref_n']})")


if __name__ == "__main__":
    main()
