//! RustOCR — быстрый серверный OCR (RU+EN).
//! Режимы: `rustocr serve [--port N]` — HTTP-сервис;
//! `rustocr <картинка|pdf> [--json] [--xlsx out.xlsx]` — CLI;
//! `rustocr <pdf> --estimate [out.json]` — сметный конвейер: иерархия
//! раздел → позиция → ресурсы + валидация; `--pazl [out.json]` — DTO ПАЗЛ.

mod det;
mod estimate;
mod geom;
mod imgio;
mod pdf;
mod pipeline;
mod rec;
mod server;
mod table;

use anyhow::Result;
use pipeline::{EngineOptions, OcrEngine, OcrResult};
use std::path::{Path, PathBuf};
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

struct CliCfg {
    json_out: bool,
    xlsx: Option<PathBuf>,
    /// Some(None) — вывести в stdout; Some(Some(p)) — записать в файл
    estimate: Option<Option<PathBuf>>,
    /// DTO ПАЗЛ (LLM-парсер, если настроен, иначе правила)
    pazl: Option<Option<PathBuf>>,
    dpi: u32,
    jobs: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_env();
    let args: Vec<String> = std::env::args().skip(1).collect();

    let opts = EngineOptions::from_env();
    // одно окружение ONNX Runtime на весь процесс (требование ort)
    let ort_env = match ort::init().build() {
        Ok(e) => e,
        Err(e) => {
            eprintln!(
                "Не удалось инициализировать ONNX Runtime.\n{}\nПодсказка: ORT_DYLIB_PATH=/путь/к/libonnxruntime.so",
                e
            );
            std::process::exit(2);
        }
    };
    let engine = match OcrEngine::load(&opts, &ort_env) {
        Ok(e) => Arc::new(e),
        Err(e) => {
            eprintln!(
                "Не удалось загрузить модели из «{}».\n{}\nПодсказки:\n  \
                 RUSTOCR_MODELS=/путь/к/models ORT_DYLIB_PATH=/путь/к/libonnxruntime.so cargo run --release -- serve",
                opts.models_dir.display(),
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
    let mut cfg = CliCfg {
        json_out: false,
        xlsx: None,
        estimate: None,
        pazl: None,
        dpi: pdf::dpi_from_env(),
        jobs: std::env::var("RUSTOCR_JOBS")
            .ok()
            .and_then(|v| v.parse().ok()),
    };
    let is_flag = |s: &str| {
        s.starts_with('-')
            || matches!(s, "serve")
    };
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
            "--json" => cfg.json_out = true,
            "--estimate" => {
                cfg.estimate = Some(match args.get(i + 1) {
                    Some(v) if !is_flag(v) => {
                        i += 1;
                        Some(PathBuf::from(v))
                    }
                    _ => None,
                });
            }
            "--pazl" => {
                cfg.pazl = Some(match args.get(i + 1) {
                    Some(v) if !is_flag(v) => {
                        i += 1;
                        Some(PathBuf::from(v))
                    }
                    _ => None,
                });
            }
            "--xlsx" => {
                if let Some(v) = args.get(i + 1) {
                    cfg.xlsx = Some(PathBuf::from(v));
                }
                i += 1;
            }
            "--dpi" => {
                if let Some(v) = args.get(i + 1) {
                    cfg.dpi = pdf::clamp_dpi(v.parse().unwrap_or(pdf::DEFAULT_DPI));
                }
                i += 1;
            }
            "--jobs" | "-j" => {
                if let Some(v) = args.get(i + 1) {
                    cfg.jobs = v.parse().ok();
                }
                i += 1;
            }
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
        cli(engine, &ort_env, &files, &cfg, &opts)
    }
}

fn cli(
    engine: Arc<OcrEngine>,
    ort_env: &ort::environment::Environment,
    files: &[PathBuf],
    cfg: &CliCfg,
    base_opts: &EngineOptions,
) -> Result<()> {
    let mut xlsx_pages: Vec<(std::sync::Arc<Vec<image::RgbImage>>, Vec<OcrResult>)> =
        Vec::new();

    for path in files {
        let is_pdf = pdf::is_pdf_path(path) || {
            // странный формат? заглянем в первые байты
            std::fs::File::open(path)
                .and_then(|mut f| {
                    use std::io::Read;
                    let mut magic = [0u8; 5];
                    f.read_exact(&mut magic)?;
                    Ok(magic)
                })
                .map(|m| pdf::is_pdf_bytes(&m))
                .unwrap_or(false)
        };

        if is_pdf {
            let t0 = std::time::Instant::now();
            let images = std::sync::Arc::new(pdf::render_pdf(path, cfg.dpi)?);
            let render_ms = t0.elapsed().as_millis() as u64;
            eprintln!(
                "== {} · страниц: {} · рендер {} мс",
                path.display(),
                images.len(),
                render_ms
            );

            let t1 = std::time::Instant::now();
            let results = run_pages(engine.clone(), ort_env, images.clone(), cfg.jobs, base_opts)?;
            let ocr_ms = t1.elapsed().as_millis() as u64;

            if cfg.json_out {
                let pages: Vec<serde_json::Value> = results
                    .iter()
                    .enumerate()
                    .map(|(n, r)| {
                        let mut v = r.to_json();
                        v["page"] = serde_json::json!(n + 1);
                        v
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::json!({
                        "file": path.display().to_string(),
                        "pages": pages,
                        "timing_ms": {
                            "render": render_ms,
                            "ocr": ocr_ms,
                            "total": render_ms + ocr_ms,
                        },
                    })
                );
            } else {
                for (n, r) in results.iter().enumerate() {
                    println!("=== Страница {} ===", n + 1);
                    println!("{}", r.text);
                }
            }
            eprintln!(
                "-- PDF: {} стр · OCR {} мс · всего {} мс ({:.1} с/стр)",
                results.len(),
                ocr_ms,
                render_ms + ocr_ms,
                ocr_ms as f64 / results.len().max(1) as f64 / 1000.0
            );
            xlsx_pages.push((images, results));
        } else {
            // лимиты декодирования — как в HTTP-сервере
            let mut img = crate::imgio::open_file(path)?.to_rgb8();
            crate::imgio::flatten_if_faded(&mut img);
            let res = engine.recognize(&img)?;
            if cfg.json_out {
                let mut v = res.to_json();
                v["file"] = serde_json::json!(path.display().to_string());
                println!("{}", v);
            } else {
                println!("{}", res.text);
                eprintln!(
                    "-- {}×{} · детекция {} мс · чтение {} мс · всего {} мс{}",
                    res.width, res.height, res.det_ms, res.rec_ms, res.total_ms,
                    if res.skipped > 0 { format!(" · ПОТЕРЯНО ФРАГМЕНТОВ: {}", res.skipped) } else { String::new() }
                );
            }
            xlsx_pages.push((std::sync::Arc::new(vec![img]), vec![res]));
        }
    }

    if let Some(xlsx_path) = &cfg.xlsx {
        let mut imgs: Vec<image::RgbImage> = Vec::new();
        let mut ress: Vec<OcrResult> = Vec::new();
        for (i, r) in xlsx_pages.iter() {
            imgs.extend(i.iter().cloned());
            ress.extend(r.iter().cloned());
        }
        let n = table::write_xlsx(&engine, &imgs, &ress, xlsx_path)?;
        eprintln!("-- Excel: {} лист(ов) → {}", n, xlsx_path.display());
    }

    // сметный конвейер: иерархия + валидация (+ DTO ПАЗЛ)
    if cfg.estimate.is_some() || cfg.pazl.is_some() {
        for (path, (imgs, ress)) in files.iter().zip(xlsx_pages.iter()) {
            let t0 = std::time::Instant::now();
            let pages: Vec<estimate::PageInput> = ress
                .iter()
                .enumerate()
                .map(|(n, r)| estimate::PageInput {
                    page: n + 1,
                    img: imgs.get(n),
                    result: r,
                })
                .collect();
            let doc = estimate::parse_estimate(Some(&engine), &pages);
            eprintln!(
                "-- Смета {}: разделов {}, позиций {}, ресурсов {}, ошибок валидации {}, предупреждений {} ({:?})",
                path.display(),
                doc.validation.stats.sections,
                doc.validation.stats.positions,
                doc.validation.stats.resources,
                doc.validation.errors.len(),
                doc.validation.warnings.len(),
                t0.elapsed()
            );
            if let Some(out) = &cfg.estimate {
                let mut v = serde_json::to_value(&doc)?;
                v["file"] = serde_json::json!(path.display().to_string());
                emit_json(&v, out.as_deref(), "estimate")?;
            }
            if let Some(out) = &cfg.pazl {
                let (dto, parser, llm_err) = estimate::pazl::parse_pazl(&doc);
                let mut v = serde_json::to_value(&dto)?;
                v["file"] = serde_json::json!(path.display().to_string());
                v["parser"] = serde_json::json!(parser);
                if let Some(e) = llm_err {
                    v["llm_error"] = serde_json::json!(e);
                    eprintln!("-- LLM-парсер не сработал ({}), использованы правила", e);
                }
                emit_json(&v, out.as_deref(), "pazl")?;
            }
        }
    }
    Ok(())
}

/// Записать JSON в файл или вывести в stdout.
fn emit_json(v: &serde_json::Value, out: Option<&Path>, tag: &str) -> Result<()> {
    let s = serde_json::to_string_pretty(v)?;
    match out {
        Some(p) => {
            std::fs::write(p, s)?;
            eprintln!("-- {}: {}", tag, p.display());
        }
        None => println!("{}", s),
    }
    Ok(())
}

/// Распознать страницы PDF. На 2 ядрах и меньше — последовательно одним
/// движком (ORR и так занимает оба ядра), на 4+ — по движку на долю ядер,
/// страницы идут параллельно. Явный --jobs перекрывает автоматике.
fn run_pages(
    engine: Arc<OcrEngine>,
    ort_env: &ort::environment::Environment,
    images: std::sync::Arc<Vec<image::RgbImage>>,
    jobs_opt: Option<usize>,
    base_opts: &EngineOptions,
) -> Result<Vec<OcrResult>> {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let n = images.len().max(1);
    let jobs = match jobs_opt {
        Some(j) => j.min(8).max(1).min(n),
        None => {
            if cores <= 2 {
                1
            } else {
                cores.min(8).min(n)
            }
        }
    };

    if jobs <= 1 {
        let engine2 = engine.clone();
        let imgs = images.clone();
        return (0..images.len())
            .map(|i| engine2.recognize(&imgs[i]))
            .collect();
    }

    // несколько движков, каждому — своя доля ядер
    let intra = (cores / jobs).max(1);
    let mut opts = EngineOptions {
        models_dir: base_opts.models_dir.clone(),
        lang: base_opts.lang,
        det_limit: base_opts.det_limit,
        intra_threads: Some(intra),
    };
    if let Some(n) = base_opts.intra_threads {
        // явный лимит сверху задан — не превышаем его
        opts.intra_threads = Some(n.min(intra).max(1));
    }
    let mut engines: Vec<Arc<OcrEngine>> = Vec::with_capacity(jobs);
    engines.push(engine.clone()); // первый уже загружен — экономим время
    for _ in 1..jobs {
        engines.push(Arc::new(OcrEngine::load(&opts, ort_env)?));
    }
    eprintln!(
        "== параллельно: {} поток(ов) × {} intra-op поток(а) на движок",
        jobs, intra
    );

    let n_pages = images.len();
    let queue = std::sync::Arc::new(std::sync::Mutex::new(
        (0..n_pages).map(std::option::Option::Some).collect::<Vec<_>>(),
    ));
    let done: std::sync::Arc<std::sync::Mutex<Vec<std::option::Option<OcrResult>>>> =
        std::sync::Arc::new(std::sync::Mutex::new((0..n_pages).map(|_| None).collect()));
    let mut handles = Vec::with_capacity(engines.len());
    for eng in engines {
        let queue = queue.clone();
        let done = done.clone();
        let imgs = images.clone();
        let h = std::thread::spawn(move || -> Result<()> {
            loop {
                let next = {
                    let mut q = queue.lock().unwrap();
                    q.iter_mut().find(|s| s.is_some()).and_then(|s| s.take())
                };
                let Some(idx) = next else { break };
                let r = eng.recognize(&imgs[idx])?;
                done.lock().unwrap()[idx] = Some(r);
            }
            Ok(())
        });
        handles.push(h);
    }
    for h in handles {
        h.join().map_err(|_| anyhow::anyhow!("падение потока OCR"))??;
    }
    let results = done
        .lock()
        .unwrap()
        .drain(..)
        .map(|r| r.expect("страница без результата"))
        .collect();
    Ok(results)
}
