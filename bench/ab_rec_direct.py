#!/usr/bin/env python3
"""Прямое сравнение v5 и v6 rec-моделей на рендеренной строке.

Рендерим текст, прогоняем через обе ONNX-модели с нашим препроцессингом
(высота 48, нормализация (v/255-0.5)/0.5), CTC-декодируем и печатаем:
- число таймстепов T на данную ширину,
- декодированный текст,
- для v6 — дамп argmax по таймстепам (индекс, символ, вероятность).
"""
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
V5 = "/home/z/my-project/models/rec_cyrillic.onnx"
V6 = "/tmp/models_v6/models/rec_cyrillic.onnx"


def load_table(path):
    table = [""]  # blank
    with open(path, encoding="utf-8") as f:
        for line in f:
            table.append(line.rstrip("\n"))
    table.append(" ")
    return table


def render(text, height=32):
    font = ImageFont.truetype(FONT, height)
    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    w = d.textlength(text, font=font)
    img = Image.new("RGB", (int(w) + 16, height + 16), "white")
    ImageDraw.Draw(img).text((8, 8), text, fill="black", font=font)
    return img


def prep(img, rec_h=48):
    ratio = img.width / img.height
    w = max(1, int(round(rec_h * ratio)))
    img = img.resize((w, rec_h))
    x = np.zeros((1, 3, rec_h, w), dtype=np.float32)
    a = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)  # HWC->CHW
    a = (a / 255.0 - 0.5) / 0.5
    x[0] = a
    return x


def decode(probs, table):
    """CTC-декод: argmax, схлопывание повторов, пропуск blank."""
    out, confs, prev = [], [], -1
    for row in probs:
        bi = int(np.argmax(row))
        if bi != prev and bi != 0:
            out.append(table[bi] if bi < len(table) else "<?%d>" % bi)
            confs.append(float(row[bi]))
        prev = bi
    return "".join(out), (sum(confs) / len(confs) if confs else 0.0)


def main():
    text = "Смета к проекту Каркасный дом 25х100х6000мм"
    img = render(text)
    x = prep(img)
    print(f"текст: {text!r}\nкроп: {img.width}x{img.height} -> вход {x.shape}\n")

    for name, path, dict_path in [
        ("v5", V5, "/home/z/my-project/models/cyrillic_dict.txt"),
        ("v6", V6, "/tmp/models_v6/models/cyrillic_dict.txt"),
    ]:
        table = load_table(dict_path)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"x": x})[0]
        probs = out[0]
        t, c = probs.shape
        s, conf = decode(probs, table)
        print(f"[{name}] T={t} C={c} table={len(table)} conf={conf:.3f}")
        print(f"[{name}] decoded: {s!r}\n")
        if name == "v6":
            print("дамп v6 argmax (t, idx, char, prob):")
            prev = -1
            for ti, row in enumerate(probs):
                bi = int(np.argmax(row))
                ch = table[bi] if bi < len(table) else "<?>"
                if ti < 60 or bi != 0:
                    print(f"  t={ti:3d} idx={bi:3d} {ch!r:>8} p={row[bi]:.3f}")
                prev = bi


if __name__ == "__main__":
    main()
