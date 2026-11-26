# Многоступенчатая сборка: статический бинарник со встроенными шрифтами
FROM rust:1.83-slim AS build
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY fonts/ fonts/
RUN cargo build --release --bin rustocr

FROM debian:bookworm-slim
# poppler-utils: pdftotext/pdfinfo — гибрид «текстовый слой вместо OCR»
# для born-digital PDF (без poppler такие сметы читаются через OCR и
# заголовки разделов ломаются). timeout — из coreutils, уже в базовом образе.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates poppler-utils && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/target/release/rustocr /usr/local/bin/rustocr
EXPOSE 8080
# движок прогревается при старте, первый запрос не платит за шаблоны
CMD ["rustocr", "serve", "--port", "8080", "--lang", "both"]
