#!/usr/bin/env python3
"""Перебор вариантов препроцессинга для v6 rec: нормализация × порядок каналов."""
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import sys

sys.path.insert(0, "/home/z/my-project/scripts")
from ab_rec_direct import render, load_table, decode  # noqa: E402

TEXT = "Смета к проекту Каркасный дом 25х100х6000мм"
V6 = "/tmp/models_v6/models/rec_cyrillic.onnx"
DICT = "/tmp/models_v6/models/cyrillic_dict.txt"


def prep(img, mode):
    ratio = img.width / img.height
    w = max(1, int(round(48 * ratio)))
    img = img.resize((w, 48))
    a = np.asarray(img, dtype=np.float32)  # HWC RGB
    if "bgr" in mode:
        a = a[:, :, ::-1]
    x = a.transpose(2, 0, 1)  # CHW
    if "pm1" in mode:  # (v/255-0.5)/0.5 -> [-1,1]
        x = (x / 255.0 - 0.5) / 0.5
    elif "01" in mode:  # v/255 -> [0,1]
        x = x / 255.0
    elif "std" in mode:  # ImageNet mean/std
        m = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        s = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        x = (x / 255.0 - m) / s
    elif "127" in mode:  # (v-127.5)/127.5 == pm1
        x = (x - 127.5) / 127.5
    return x[None].astype(np.float32)


def main():
    table = load_table(DICT)
    img = render(TEXT)
    sess = ort.InferenceSession(V6, providers=["CPUExecutionProvider"])
    for mode in ["pm1_rgb", "pm1_bgr", "01_rgb", "01_bgr", "std_rgb", "std_bgr"]:
        x = prep(img, mode)
        out = sess.run(None, {"x": x})[0]
        s, conf = decode(out[0], table)
        ok = "OK " if s == TEXT else "   "
        print(f"[{mode:>8}] conf={conf:.3f} {ok}{s!r}")


if __name__ == "__main__":
    main()
