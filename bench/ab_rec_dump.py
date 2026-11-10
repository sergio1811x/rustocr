#!/usr/bin/env python3
"""Полный дамп argmax-таймстепов v5 и v6 на одной картинке, рядом."""
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import sys

sys.path.insert(0, "/home/z/my-project/scripts")
from ab_rec_direct import render, load_table, decode  # noqa: E402

TEXT = "Смета 6000 100 50 аа 000"
MODELS = [
    ("v5", "/home/z/my-project/models/rec_cyrillic.onnx",
     "/home/z/my-project/models/cyrillic_dict.txt"),
    ("v6", "/tmp/models_v6/models/rec_cyrillic.onnx",
     "/tmp/models_v6/models/cyrillic_dict.txt"),
]


def prep(img):
    ratio = img.width / img.height
    w = max(1, int(round(48 * ratio)))
    img = img.resize((w, 48))
    a = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
    a = (a / 255.0 - 0.5) / 0.5
    return a[None].astype(np.float32)


def main():
    img = render(TEXT)
    x = prep(img)
    print(f"text={TEXT!r} input={x.shape}")
    dumps = {}
    for name, path, dp in MODELS:
        table = load_table(dp)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"x": x})[0][0]
        seq = []
        for row in out:
            bi = int(np.argmax(row))
            seq.append((bi, float(row[bi])))
        dumps[name] = seq
        s, conf = decode(out, table)
        print(f"[{name}] decoded={s!r} conf={conf:.3f}")
    t = max(len(d) for d in dumps.values())
    print(f"\n{'t':>4} | {'v5 idx char p':>22} | {'v6 idx char p':>22}")
    for i in range(t):
        cells = []
        for name, _, _ in MODELS:
            if i < len(dumps[name]):
                bi, p = dumps[name][i]
                cells.append(f"{bi:4d} {bi:>4} p={p:.2f}")
            else:
                cells.append(" " * 22)
        # печатаем с символами
        row = []
        for name, _, dp in MODELS:
            if i < len(dumps[name]):
                bi, p = dumps[name][i]
                table = load_table(dp)
                ch = table[bi] if bi < len(table) else "?"
                row.append(f"{bi:4d} {ch!r:>7} {p:.2f}")
            else:
                row.append(" " * 16)
        print(f"{i:>4} | {row[0]:>16} | {row[1]:>16}")


if __name__ == "__main__":
    main()
