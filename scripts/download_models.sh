#!/usr/bin/env bash
# Скачивает тяжёлые onnx-модели, не влезающие в git (лимит 100 МБ/файл):
#   - models/table/slanext_wired.onnx    — структура таблиц (SLANeXt-wired, ~350 МБ)
#   - models/layout/pp_doclayoutv3.onnx  — layout-анализ (PP-DocLayoutV3, ~12 МБ)
# Лёгкие артефакты (yml, словарь структуры, hunspell-словари, slanet-plus.onnx)
# уже лежат в git. Конвейер деградирует мягко: без этих файлов всё работает,
# только без нейросетевой структуры таблиц и layout-разметки.
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() { # fetch <url> <dest> <expected-min-bytes>
    local url="$1" dest="$2" min="$3"
    if [ -s "$dest" ] && [ "$(stat -c%s "$dest")" -ge "$min" ]; then
        echo "[ok] $dest уже на месте ($(stat -c%s "$dest") байт)"
        return 0
    fi
    echo "[..] качаю $url"
    curl -fL --retry 3 --connect-timeout 30 -o "$dest.part" "$url"
    local size
    size="$(stat -c%s "$dest.part")"
    if [ "$size" -lt "$min" ]; then
        rm -f "$dest.part"
        echo "[err] $url: размер $size < $min — битый файл, удалил" >&2
        return 1
    fi
    mv "$dest.part" "$dest"
    echo "[ok] $dest ($size байт)"
}

# SLANeXt-wired (структура таблиц) — официальный onnx-экспорт PaddlePaddle
fetch "https://huggingface.co/PaddlePaddle/SLANeXt_wired_onnx/resolve/main/inference.onnx" \
      models/table/slanext_wired.onnx 100000000

# PP-DocLayoutV3 (layout) — официальный onnx-экспорт PaddlePaddle
fetch "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx/resolve/main/inference.onnx" \
      models/layout/pp_doclayoutv3.onnx 1000000

echo "готово."
