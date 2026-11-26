#!/usr/bin/env bash
# Запуск RustOCR-сервера. При необходимости — сборка.
set -euo pipefail
cd "$(dirname "$0")"

# приоритет: собранный локально таргет → готовый бинарник из архива → сборка
if [ -f target/release/rustocr ]; then
    BIN=target/release/rustocr
elif [ -f bin/rustocr ]; then
    BIN=bin/rustocr
else
    echo "[rustocr] собираю (первый запуск займёт пару минут)..."
    cargo build --release
    BIN=target/release/rustocr
fi

export ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-$PWD/runtime/libonnxruntime.so.1.20.0}"
export RUSTOCR_MODELS="${RUSTOCR_MODELS:-$PWD/models}"
export RUSTOCR_PORT="${RUSTOCR_PORT:-8080}"
# RUSTOCR_LANG=ru|en|auto — можно задать снаружи

# Гибрид «текстовый слой вместо OCR» опирается на pdftotext/pdfinfo из
# poppler-utils. Без них цифровые PDF читаются через OCR — заголовки
# разделов ломаются («Стропил ьна я») и точность чисел падает. При пустом
# RUSTOCR_TEXT_LAYER=0 это предупреждение можно игнорировать.
for tool in pdftotext pdfinfo; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "[rustocr] ВНИМАНИЕ: '$tool' не найден (нужен пакет poppler-utils)." >&2
        echo "[rustocr] Без него born-digital PDF идут через OCR вместо текстового слоя." >&2
        if [ -x /usr/bin/apt-get ]; then
            echo "[rustocr] Установить: sudo apt-get install -y poppler-utils" >&2
        elif [ -x /usr/bin/dnf ]; then
            echo "[rustocr] Установить: sudo dnf install -y poppler-utils" >&2
        elif [ -x /usr/bin/apk ]; then
            echo "[rustocr] Установить: apk add poppler-utils" >&2
        fi
        break
    fi
done

echo "[rustocr] сервер на http://0.0.0.0:${RUSTOCR_PORT} (демо-страница: /)"
exec "$BIN" serve --port "$RUSTOCR_PORT"
