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
mod pdf_text;
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
    let mut xlsx_pages: Vec<(
        std::sync::Arc<Vec<image::RgbImage>>,
        Vec<OcrResult>,
        Option<std::sync::Arc<Vec<pdf_text::PageText>>>,
    )> = Vec::new();

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

            // гибрид: у «цифровых» PDF берём идеальный текстовый слой,
            // страницы-сканы читаем OCR (RUSTOCR_TEXT_LAYER=0 — выключить)
            let layer = text_layer_for(path, images.len() as u32);

            let t1 = std::time::Instant::now();
            let results = run_pages(
                engine.clone(),
                ort_env,
                images.clone(),
                cfg.jobs,
                base_opts,
                layer.clone(),
            )?;
            let ocr_ms = t1.elapsed().as_millis() as u64;
            let tl_pages = layer
                .as_deref()
                .map(|l| l.iter().filter(|p| p.usable()).count())
                .unwrap_or(0);
            if tl_pages > 0 {
                eprintln!(
                    "-- текстовый слой: {} из {} стр. без OCR",
                    tl_pages,
                    images.len()
                );
            }

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
            xlsx_pages.push((images, results, layer));
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
            xlsx_pages.push((std::sync::Arc::new(vec![img]), vec![res], None));
        }
    }

    if let Some(xlsx_path) = &cfg.xlsx {
        let mut imgs: Vec<image::RgbImage> = Vec::new();
        let mut ress: Vec<OcrResult> = Vec::new();
        for (i, r, _) in xlsx_pages.iter() {
            imgs.extend(i.iter().cloned());
            ress.extend(r.iter().cloned());
        }
        let n = table::write_xlsx(&engine, &imgs, &ress, xlsx_path)?;
        eprintln!("-- Excel: {} лист(ов) → {}", n, xlsx_path.display());
    }

    // сметный конвейер: иерархия + валидация (+ DTO ПАЗЛ)
    if cfg.estimate.is_some() || cfg.pazl.is_some() {
        for (path, (imgs, results, layer)) in files.iter().zip(xlsx_pages.iter()) {
            let t0 = std::time::Instant::now();
            let pages: Vec<estimate::PageInput> = results
                .iter()
                .enumerate()
                .map(|(n, r)| estimate::PageInput {
                    page: n + 1,
                    img: imgs.get(n),
                    result: r,
                })
                .collect();
            let mut doc = estimate::parse_estimate(Some(&engine), &pages);
            // П.2/П.3: арифметико-управляемый повтор — если арифметика не
            // сошлась, перечитываем PDF в большем DPI, «подозрительные»
            // страницы (бледные итоги, битые ячейки) — с выравниванием фона;
            // из двух вариантов остаётся лучший по валидации
            if let Some(doc2) = estimate_retry(
                &engine,
                path,
                cfg,
                imgs,
                results,
                layer.as_deref().map(|v| v.as_slice()),
                &doc,
            ) {
                doc = doc2;
            }
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

/// DPI повторного прохода сметы (env RUSTOCR_EST_RETRY_DPI, по умолчанию
/// 300 при исходном <300; 0 — выключить). Перечитывание имеет смысл только
/// для PDF (картинку перерендерить не из чего).
fn est_retry_dpi(cfg: &CliCfg) -> u32 {
    match std::env::var("RUSTOCR_EST_RETRY_DPI").ok().and_then(|v| v.parse::<u32>().ok()) {
        Some(0) => 0,
        Some(v) => v.clamp(72, 400),
        None => {
            if cfg.dpi < 300 {
                300
            } else {
                0
            }
        }
    }
}

/// Текстовый слой PDF (если есть) для всех страниц документа.
/// None — слой недоступен или выключен: все страницы идут в OCR.
fn text_layer_for(
    path: &Path,
    n_pages: u32,
) -> Option<std::sync::Arc<Vec<pdf_text::PageText>>> {
    if std::env::var("RUSTOCR_TEXT_LAYER").as_deref() == Ok("0") {
        return None;
    }
    match pdf_text::extract(path, 1, n_pages) {
        Ok(v) => Some(std::sync::Arc::new(v)),
        Err(e) => {
            eprintln!("-- текстовый слой недоступен ({}), читаю OCR", e);
            None
        }
    }
}

/// Повторный проход сметного конвейера (арифметика не сошлась):
/// перерендер в большем DPI, но перечитываются OCR-ом только ПОДОЗРИТЕЛЬНЫЕ
/// страницы (бледные итоги, битые ячейки — им ещё и выравнивается фон);
/// страницы из текстового слоя пересобираются точно (слой DPI-независим),
/// чистые OCR-страницы лишь масштабируются. Возвращает `Some(doc)`, только
/// если вариант стал лучше по валидации (score ниже) — перечитывание не
/// может ухудшить результат, только потратить время.
fn estimate_retry(
    engine: &Arc<OcrEngine>,
    path: &std::path::Path,
    cfg: &CliCfg,
    imgs: &std::sync::Arc<Vec<image::RgbImage>>,
    results: &[OcrResult],
    layer: Option<&[pdf_text::PageText]>,
    first: &estimate::EstimateDocument,
) -> Option<estimate::EstimateDocument> {
    let dpi2 = est_retry_dpi(cfg);
    if dpi2 == 0 || dpi2 <= cfg.dpi {
        return None;
    }
    if !pdf::is_pdf_path(path) {
        return None; // картинку перерендерить не из чего
    }
    let score1 = estimate::validation_score(first);
    if score1 == 0 {
        return None; // арифметика сошлась — перечитывать незачем
    }
    let suspects = estimate::suspect_pages(first);
    eprintln!(
        "-- Смета: арифметика не сошлась (оценка {}), перечитываю подозрительные страницы {:?} в {} DPI{}",
        score1,
        suspects,
        dpi2,
        if suspects.is_empty() {
            String::new()
        } else {
            " с выравниванием фона".to_string()
        }
    );
    let t0 = std::time::Instant::now();
    let mut images2 = pdf::render_pdf(path, dpi2).ok()?;
    if images2.len() != imgs.len() {
        return None; // рендер дал другое число страниц — не рискуем
    }
    for &p in &suspects {
        if p >= 1 && p <= images2.len() {
            crate::det::flatten_background(&mut images2[p - 1]);
        }
    }
    let factor = dpi2 as f32 / cfg.dpi.max(1) as f32;
    let mut results2: Vec<OcrResult> = Vec::with_capacity(images2.len());
    for (i, img2) in images2.iter().enumerate() {
        let tl = layer.and_then(|l| l.get(i)).filter(|p| p.usable());
        let r = match tl {
            Some(p) => p.to_ocr_result(img2.width(), img2.height()),
            None => {
                if suspects.contains(&(i + 1)) {
                    engine.recognize(img2).ok()?
                } else {
                    results.get(i)?.scaled(factor)
                }
            }
        };
        results2.push(r);
    }
    let images2 = std::sync::Arc::new(images2);
    let pages2: Vec<estimate::PageInput> = results2
        .iter()
        .enumerate()
        .map(|(n, r)| estimate::PageInput {
            page: n + 1,
            img: images2.get(n),
            result: r,
        })
        .collect();
    let second = estimate::parse_estimate(Some(engine), &pages2);
    let score2 = estimate::validation_score(&second);
    eprintln!(
        "-- Смета (повтор, {} DPI): {:.1} с, оценка арифметики {} → {} — {}",
        dpi2,
        t0.elapsed().as_secs_f64(),
        score1,
        score2,
        if score2 < score1 { "принят повторный вариант" } else { "оставлен первый вариант" }
    );
    (score2 < score1).then_some(second)
}

/// Результат страницы из текстового слоя (если он достаточно плотный).
/// None — страница идёт в OCR.
fn text_layer_result(
    layer: Option<&[pdf_text::PageText]>,
    idx: usize,
    img: &image::RgbImage,
) -> Option<OcrResult> {
    layer
        .and_then(|l| l.get(idx))
        .filter(|p| p.usable())
        .map(|p| p.to_ocr_result(img.width(), img.height()))
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

/// Распознать страницы PDF. Гибрид: страница с плотным текстовым слоем
/// собирается из слоя (без детекции и распознавания — быстро и без
/// ошибок OCR), остальных — обычный OCR. На 2 ядрах и меньше —
/// последовательно одним движком (ORR и так занимает оба ядра), на 4+ —
/// по движку на долю ядер, страницы идут параллельно. Явный --jobs
/// перекрывает автоматике.
fn run_pages(
    engine: Arc<OcrEngine>,
    ort_env: &ort::environment::Environment,
    images: std::sync::Arc<Vec<image::RgbImage>>,
    jobs_opt: Option<usize>,
    base_opts: &EngineOptions,
    layer: Option<std::sync::Arc<Vec<pdf_text::PageText>>>,
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
        let tl = layer.as_deref().map(|v| v.as_slice());
        return (0..images.len())
            .map(|i| match text_layer_result(tl, i, &imgs[i]) {
                Some(r) => Ok(r),
                None => engine2.recognize(&imgs[i]),
            })
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
        let layer = layer.clone();
        let h = std::thread::spawn(move || -> Result<()> {
            loop {
                let next = {
                    let mut q = queue.lock().unwrap();
                    q.iter_mut().find(|s| s.is_some()).and_then(|s| s.take())
                };
                let Some(idx) = next else { break };
                let r = match text_layer_result(layer.as_deref().map(|v| v.as_slice()), idx, &imgs[idx]) {
                    Some(r) => r,
                    None => eng.recognize(&imgs[idx])?,
                };
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
