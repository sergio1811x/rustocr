//! Конвейер OCR: детекция → вырезание строк → распознавание (RU+EN).

use crate::det::{self, DetBox, ProbMap};
use crate::geom::{order_quad, warp_quad, Pt};
use crate::rec::{self, RecBatch};
use anyhow::{Context, Result};
use image::RgbImage;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use ort::inputs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

const LATIN_FALLBACK_CONF: f32 = 0.90; // ниже — пробуем вторую модель
const LATIN_MARGIN: f32 = 0.05; // латиница должна быть заметно увереннее
const ROT_RETRY_CONF: f32 = 0.45; // средняя уверенность ниже — подозреваем поворот
const ROT_RETRY_MIN_BLOCKS: usize = 3; // минимум блоков, чтобы вообще пробовать
const ROT_MARGIN: f32 = 0.10; // повёрнутый вариант должен быть заметно лучше

/// Режим языка: авто (обе модели), только русский, только английский.
#[derive(Clone, Copy, PartialEq)]
enum LangMode {
    Auto,
    Ru,
    En,
}

impl LangMode {
    fn from_env() -> Self {
        match std::env::var("RUSTOCR_LANG").unwrap_or_default().to_lowercase().as_str() {
            "ru" | "cyr" | "cyrillic" => LangMode::Ru,
            "en" | "lat" | "latin" => LangMode::En,
            _ => LangMode::Auto,
        }
    }
}

#[derive(Clone)]
pub struct Block {
    pub text: String,
    pub confidence: f32,
    pub lang: &'static str, // "ru" | "en" | "mix"
    pub quad: [Pt; 4],
}

pub struct OcrResult {
    pub width: u32,
    pub height: u32,
    pub text: String,
    pub blocks: Vec<Block>,
    pub det_ms: u64,
    pub rec_ms: u64,
    pub total_ms: u64,
}

pub struct OcrEngine {
    _env: Environment, // держим ONNX Runtime живым, пока живут сессии
    det: Mutex<Session>,
    rec_cyr: Mutex<Session>,
    rec_lat: Mutex<Session>,
    lang_mode: LangMode,
    table_cyr: Vec<String>,
    table_lat: Vec<String>,
    det_in: String,
    det_out: String,
    rec_cyr_in: String,
    rec_cyr_out: String,
    rec_lat_in: String,
    rec_lat_out: String,
    det_limit: u32,
}

impl OcrEngine {
    /// Загрузить движок из каталога models/ (или RUSTOCR_MODELS).
    pub fn load(models_dir: &Path) -> Result<Self> {
        let det_path = models_dir.join("det.onnx");
        let rec_cyr_path = models_dir.join("rec_cyrillic.onnx");
        let rec_lat_path = models_dir.join("rec_latin.onnx");
        for p in [&det_path, &rec_cyr_path, &rec_lat_path] {
            if !p.exists() {
                anyhow::bail!("модель не найдена: {}", p.display());
            }
        }
        let env = ort::init()
            .build()
            .context("инициализация ONNX Runtime (проверьте ORT_DYLIB_PATH)")?;
        let det = Session::builder(&env)
            .context("builder детектора")?
            .commit_from_file(&det_path)
            .with_context(|| format!("загрузка {}", det_path.display()))?;
        let rec_cyr = Session::builder(&env)
            .context("builder распознавателя (кириллица)")?
            .commit_from_file(&rec_cyr_path)
            .with_context(|| format!("загрузка {}", rec_cyr_path.display()))?;
        let rec_lat = Session::builder(&env)
            .context("builder распознавателя (латиница)")?
            .commit_from_file(&rec_lat_path)
            .with_context(|| format!("загрузка {}", rec_lat_path.display()))?;
        let table_cyr = rec::load_table(&models_dir.join("cyrillic_dict.txt"))?;
        let table_lat = rec::load_table(&models_dir.join("latin_dict.txt"))?;
        let det_limit: u32 = std::env::var("RUSTOCR_DET_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(960);
        let lang_mode = LangMode::from_env();
        let io = |s: &Session| -> (String, String) {
            (
                s.inputs()[0].name().to_string(),
                s.outputs()[0].name().to_string(),
            )
        };
        let (det_in, det_out) = io(&det);
        let (rec_cyr_in, rec_cyr_out) = io(&rec_cyr);
        let (rec_lat_in, rec_lat_out) = io(&rec_lat);
        Ok(Self {
            _env: env,
            det: Mutex::new(det),
            rec_cyr: Mutex::new(rec_cyr),
            rec_lat: Mutex::new(rec_lat),
            lang_mode,
            table_cyr,
            table_lat,
            det_in,
            det_out,
            rec_cyr_in,
            rec_cyr_out,
            rec_lat_in,
            rec_lat_out,
            det_limit,
        })
    }

    /// Размер словаря кириллицы (для диагностики загрузки).
    pub fn table_size(&self) -> usize {
        self.table_cyr.len()
    }

    /// Распознать всю картинку (с авто-поворотом при неуверенном чтении).
    pub fn recognize(&self, img: &RgbImage) -> Result<OcrResult> {
        let base = self.recognize_inner(img)?;
        let avg = avg_confidence(&base);
        if avg >= ROT_RETRY_CONF || base.blocks.len() < ROT_RETRY_MIN_BLOCKS {
            return Ok(base);
        }
        // картинка читается плохо — возможно, она повёрнута: пробуем 90/180/270°
        let mut best = base;
        for angle in [90u16, 180, 270] {
            let rotated = match angle {
                90 => image::imageops::rotate90(img),
                180 => image::imageops::rotate180(img),
                _ => image::imageops::rotate270(img),
            };
            let candidate = self.recognize_inner(&rotated)?;
            if avg_confidence(&candidate) > avg_confidence(&best) + ROT_MARGIN {
                best = candidate;
            }
        }
        Ok(best)
    }

    /// Основной проход без повторов.
    fn recognize_inner(&self, img: &RgbImage) -> Result<OcrResult> {
        let t0 = Instant::now();

        // 1) Детекция текстовых блоков
        let inp = det::preprocess(img, self.det_limit);
        let t_det = Instant::now();
        let prob = self.run_det(&inp.data, inp.shape)?;
        let det_ms = t_det.elapsed().as_millis() as u64; // чисто инференс
        let boxes = det::postprocess(&prob, inp.ratio, img.width(), img.height());

        // 2) Вырезание строк
        let crops: Vec<RgbImage> = boxes
            .iter()
            .filter_map(|b| crop_box(img, b))
            .collect();

        // 3) Распознавание — модель по режиму языка
        let t_rec = Instant::now();
        let mut results: Vec<(String, f32, &'static str)> = match self.lang_mode {
            LangMode::En => self.run_rec_batch(
                &self.rec_lat,
                &self.table_lat,
                &self.rec_lat_in,
                &self.rec_lat_out,
                &crops,
            ),
            _ => self.run_rec_batch(
                &self.rec_cyr,
                &self.table_cyr,
                &self.rec_cyr_in,
                &self.rec_cyr_out,
                &crops,
            ),
        };

        // 4) Запасная модель для неуверенных строк (только в авто-режиме)
        if self.lang_mode == LangMode::Auto {
            let weak: Vec<usize> = results
                .iter()
                .enumerate()
                .filter(|(_, (t, c, _))| *c < LATIN_FALLBACK_CONF || t.trim().is_empty())
                .map(|(i, _)| i)
                .collect();
            if !weak.is_empty() {
                let weak_crops: Vec<RgbImage> =
                    weak.iter().map(|&i| crops[i].clone()).collect();
                let lat = self.run_rec_batch(
                    &self.rec_lat,
                    &self.table_lat,
                    &self.rec_lat_in,
                    &self.rec_lat_out,
                    &weak_crops,
                );
                for (k, &i) in weak.iter().enumerate() {
                    if lat[k].1 > results[i].1 + LATIN_MARGIN {
                        results[i] = lat[k].clone();
                    }
                }
            }
        }
        let rec_ms = t_rec.elapsed().as_millis() as u64;

        // 5) Сборка результата: текст + язык по составу символов
        let mut blocks: Vec<Block> = Vec::with_capacity(boxes.len());
        let mut lines: Vec<String> = Vec::new();
        for (i, b) in boxes.iter().enumerate() {
            if i >= results.len() {
                break;
            }
            let (text, conf, _) = &results[i];
            let lang = detect_lang(text);
            blocks.push(Block {
                text: text.clone(),
                confidence: *conf,
                lang,
                quad: b.quad,
            });
            if !text.trim().is_empty() {
                lines.push(text.trim().to_string());
            }
        }
        let text = lines.join("\n");
        let total_ms = t0.elapsed().as_millis() as u64;

        Ok(OcrResult {
            width: img.width(),
            height: img.height(),
            text,
            blocks,
            det_ms,
            rec_ms,
            total_ms,
        })
    }

    fn run_det(&self, data: &[f32], shape: [usize; 4]) -> Result<ProbMap> {
        let input = Tensor::from_array((shape, data.to_vec())).context("тензор детектора")?;
        let mut session = lock(&self.det);
        let outputs = session
            .run(inputs![self.det_in.as_str() => input])
            .context("запуск детектора")?;
        let out = &outputs[self.det_out.as_str()];
        let (oshape, raw) = out
            .try_extract_tensor::<f32>()
            .context("выход детектора")?;
        // [1, 1, H, W]
        let (h, w) = (oshape[2] as usize, oshape[3] as usize);
        if raw.len() != h * w {
            anyhow::bail!(
                "неожиданный размер выхода детектора: {} при {}×{}",
                raw.len(),
                h,
                w
            );
        }
        Ok(ProbMap {
            w,
            h,
            data: raw.to_vec(),
        })
    }

    /// Пачка строк для одного прогона сети: на плотных таблицах блоков
    /// сотни, и один общий тензор съедал гигабайты (убийство по OOM).
    /// Поэтому режем на порции по 16 строк и сортируем их по ширине
    /// (как batch_size + сортировка по max_wh_ratio в PaddleOCR):
    /// узкие строки паддятся до узкой пачки, а не до самой широкой строки.
    fn run_rec_batch(
        &self,
        rec: &Mutex<Session>,
        table: &[String],
        in_name: &str,
        out_name: &str,
        crops: &[RgbImage],
    ) -> Vec<(String, f32, &'static str)> {
        let n_total = crops.len();
        let mut results: Vec<(String, f32, &'static str)> = Vec::with_capacity(n_total);
        if n_total == 0 {
            return results;
        }
        // ширина/высота каждого кропа — по ней сортируем
        let ratios: Vec<f32> = crops
            .iter()
            .map(|c| c.width() as f32 / c.height().max(1) as f32)
            .collect();
        let mut order: Vec<usize> = (0..n_total).collect();
        order.sort_by(|&a, &b| ratios[a].partial_cmp(&ratios[b]).unwrap_or(std::cmp::Ordering::Equal));

        const CHUNK: usize = 16;
        let mut decoded: Vec<Option<(String, f32)>> = vec![None; n_total];
        for chunk in order.chunks(CHUNK) {
            let batch_crops: Vec<&RgbImage> = chunk.iter().map(|&i| &crops[i]).collect();
            let batch: RecBatch = rec::preprocess_batch(&batch_crops);
            let run = (|| -> Result<Vec<(String, f32)>> {
                let input = Tensor::from_array((batch.shape, batch.data)).context("тензор строк")?;
                let mut session = lock(rec);
                let outputs = session
                    .run(inputs![in_name => input])
                    .context("запуск распознавателя")?;
                let out = &outputs[out_name];
                let (oshape, raw) = out
                    .try_extract_tensor::<f32>()
                    .context("выход распознавателя")?;
                // [N, T, C]
                let (n, t, c) = (oshape[0] as usize, oshape[1] as usize, oshape[2] as usize);
                let per = t * c;
                let mut part = Vec::with_capacity(n);
                for i in 0..n.min(chunk.len()) {
                    let (text, conf) = rec::ctc_decode(&raw[i * per..(i + 1) * per], t, c, table);
                    part.push((text, conf));
                }
                Ok(part)
            })();
            match run {
                Ok(part) => {
                    for (k, &i) in chunk.iter().enumerate() {
                        if let Some(v) = part.get(k) {
                            decoded[i] = Some(v.clone());
                        }
                    }
                }
                Err(_) => {
                    eprintln!("[rustocr] пачка строк не распознана, пропускаю её");
                }
            }
        }
        for d in decoded {
            results.push(match d {
                Some((text, conf)) => (text, conf, "model"),
                None => (String::new(), 0.0, "model"),
            });
        }
        results
    }
}

/// Вырезать блок из картинки (перспектива → прямой прямоугольник).
fn crop_box(img: &RgbImage, b: &DetBox) -> Option<RgbImage> {
    let q = order_quad(&b.quad);
    let w_top = crate::geom::dist(q[0], q[1]);
    let w_bot = crate::geom::dist(q[2], q[3]);
    let h_left = crate::geom::dist(q[0], q[3]);
    let h_right = crate::geom::dist(q[1], q[2]);
    let w = w_top.max(w_bot).floor() as u32;
    let h = h_left.max(h_right).floor() as u32;
    if w < 3 || h < 3 {
        return None;
    }
    Some(warp_quad(img, &q, w.min(4096), h.min(1024)))
}

/// Средняя уверенность по непустым блокам.
fn avg_confidence(r: &OcrResult) -> f32 {
    let vals: Vec<f32> = r
        .blocks
        .iter()
        .filter(|b| !b.text.trim().is_empty())
        .map(|b| b.confidence)
        .collect();
    if vals.is_empty() {
        0.0
    } else {
        vals.iter().sum::<f32>() / vals.len() as f32
    }
}

/// Язык строки по составу символов.
fn detect_lang(text: &str) -> &'static str {
    let mut cyr = 0usize;
    let mut lat = 0usize;
    for ch in text.chars() {
        if ch.is_alphabetic() {
            if ('\u{0400}'..='\u{04FF}').contains(&ch) {
                cyr += 1;
            } else if ch.is_ascii_alphabetic() {
                lat += 1;
            }
        }
    }
    match (cyr, lat) {
        (0, 0) => "ru",
        (_, 0) => "ru",
        (0, _) => "en",
        _ => "mix",
    }
}

/// Найти каталог моделей с учётом переменной окружения.
pub fn models_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("RUSTOCR_MODELS").unwrap_or_else(|_| "models".to_string()),
    )
}

/// Захватить мьютекс, переживая отравление.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
