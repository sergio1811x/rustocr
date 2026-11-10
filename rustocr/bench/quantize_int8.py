#!/usr/bin/env python3
"""Динамическая INT8-квантизация rec-моделей PP-OCRv5 — ОТВЕРГНУТО.

Результат эксперимента (2 ядра CPU, libonnxruntime 1.20, opset-экспорт
Paddle 7): чтение стало В 2 РАЗА МЕДЛЕННЕЕ (newizba_5644: 113,9 с против
54,6 с на float32), средняя уверенность упала (0,888 против 0,951).
Причина: модель почти целиком конволюционная — dynamic quantization
переводит в int8 только MatMul/LSTM, а конволюции остаются во float32,
плюс конверсия opset 7→11 добавляет накладные расходы.

Скрипт оставлен как протокол отрицательного результата: не запускать
без нового обоснования. Если вернуться к идее — нужна статическая
квантизация с калибровкой по реальным кропам.

    python3 bench/quantize_int8.py   # создаст models_int8/ (НЕ подключать)
"""
import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "models"
DST = REPO / "models_int8"


def main():
    DST.mkdir(exist_ok=True)
    for name in ("rec_cyrillic.onnx", "rec_latin.onnx"):
        print(f"квантизация {name} …")
        quantize_dynamic(
            str(SRC / name),
            str(DST / name),
            weight_type=QuantType.QInt8,
        )
    # детектор оставляем как есть (6% времени — смысла нет),
    # словари нужны движку рядом с моделями
    shutil.copy(SRC / "det.onnx", DST / "det.onnx")
    for d in ("cyrillic_dict.txt", "latin_dict.txt"):
        shutil.copy(SRC / d, DST / d)
    for f in sorted(DST.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
