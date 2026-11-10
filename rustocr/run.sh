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

echo "[rustocr] сервер на http://0.0.0.0:${RUSTOCR_PORT} (демо-страница: /)"
exec "$BIN" serve --port "$RUSTOCR_PORT"
