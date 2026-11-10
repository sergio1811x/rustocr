# Многоступенчатая сборка: статический бинарник со встроенными шрифтами
FROM rust:1.83-slim AS build
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY fonts/ fonts/
RUN cargo build --release --bin rustocr

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/target/release/rustocr /usr/local/bin/rustocr
EXPOSE 8080
# движок прогревается при старте, первый запрос не платит за шаблоны
CMD ["rustocr", "serve", "--port", "8080", "--lang", "both"]
