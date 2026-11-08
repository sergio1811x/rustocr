//! RustOCR — быстрый серверный OCR (RU+EN).
//! Режимы: `rustocr serve [--port N]` — HTTP-сервис; `rustocr <картинка> [--json]` — CLI.

mod det;
mod geom;
mod pipeline;
mod rec;
mod server;

use anyhow::Result;
use pipeline::OcrEngine;
use std::path::PathBuf;
use std::sync::Arc;

fn setup_env() {
    // где искать рантайм ONNX Runtime, если не указан явно
    if std::env::var("ORT_DYLIB_PATH").is_err() {
        for cand in [
            "runtime/libonnxruntime.so.1.20.0",
            "runtime/libonnxruntime.so",
        ] {
            if std::path::Path::new(cand).exists() {
                std::env::set_var("ORT_DYLIB_PATH", cand);
                break;
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_env();
    let args: Vec<String> = std::env::args().skip(1).collect();

    let models = pipeline::models_dir();
    let engine = match OcrEngine::load(&models) {
        Ok(e) => Arc::new(e),
        Err(e) => {
            eprintln!(
                "Не удалось загрузить модели из «{}».\n{}\nПодсказки:\n  \
                 RUSTOCR_MODELS=/путь/к/models ORT_DYLIB_PATH=/путь/к/libonnxruntime.so cargo run --release -- serve",
                models.display(),
                e
            );
            std::process::exit(2);
        }
    };
    eprintln!(
        "Модели загружены: детектор + кириллица + латиница (PP-OCRv5). Словарь RU: {} символов.",
        engine.table_size()
    );

    // режим сервера
    let mut mode_serve = args.is_empty();
    let mut port: u16 = std::env::var("RUSTOCR_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8080);
    let mut files: Vec<PathBuf> = Vec::new();
    let mut json_out = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "serve" => mode_serve = true,
            "--port" | "-p" => {
                if let Some(v) = args.get(i + 1) {
                    port = v.parse().unwrap_or(8080);
                }
                i += 1;
            }
            "--json" => json_out = true,
            _ => {
                mode_serve = false;
                files.push(PathBuf::from(&args[i]));
            }
        }
        i += 1;
    }

    if mode_serve {
        server::serve(engine, port).await
    } else {
        cli(engine, &files, json_out)
    }
}

fn cli(engine: Arc<OcrEngine>, files: &[PathBuf], json_out: bool) -> Result<()> {
    for path in files {
        let img = image::open(path)?;
        let res = engine.recognize(&img.to_rgb8())?;
        if json_out {
            println!(
                "{}",
                serde_json::json!({
                    "file": path.display().to_string(),
                    "text": res.text,
                    "width": res.width,
                    "height": res.height,
                    "blocks": res.blocks.iter().map(|b| serde_json::json!({
                        "text": b.text, "confidence": b.confidence, "lang": b.lang,
                        "quad": [b.quad[0].x, b.quad[0].y, b.quad[1].x, b.quad[1].y,
                                 b.quad[2].x, b.quad[2].y, b.quad[3].x, b.quad[3].y],
                    })).collect::<Vec<_>>(),
                    "timing_ms": {"det": res.det_ms, "rec": res.rec_ms, "total": res.total_ms},
                })
            );
        } else {
            println!("{}", res.text);
            eprintln!(
                "-- {}×{} · детекция {} мс · чтение {} мс · всего {} мс",
                res.width, res.height, res.det_ms, res.rec_ms, res.total_ms
            );
        }
    }
    Ok(())
}
