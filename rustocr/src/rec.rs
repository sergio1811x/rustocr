//! Распознаватель строк (CRNN из PP-OCRv5) и CTC-декодирование.

use crate::geom::resize_tri;
use image::RgbImage;
use std::path::Path;

pub const REC_H: u32 = 48;
pub const REC_W_DEFAULT: u32 = 320;
pub const REC_W_MAX: u32 = 1280;

/// Таблица классов: индекс 0 — CTC-blank, 1..=N — символы словаря, N+1 — пробел.
pub fn load_table(dict_path: &Path) -> anyhow::Result<Vec<String>> {
    let text = std::fs::read_to_string(dict_path)
        .map_err(|e| anyhow::anyhow!("не читается словарь {}: {}", dict_path.display(), e))?;
    let mut table: Vec<String> = vec![String::new()]; // blank
    for line in text.lines() {
        table.push(line.to_string());
    }
    table.push(" ".to_string());
    Ok(table)
}

pub struct RecBatch {
    pub data: Vec<f32>,
    pub shape: [usize; 4],
}

/// Подготовить пачку кропов: высота 48, ширина пропорционально,
/// паддинг нулями до общей ширины пачки (как max_wh_ratio в PaddleOCR).
/// Принимает ссылки: под-пачки нарезаются без клонов картинок.
pub fn preprocess_batch(crops: &[&RgbImage]) -> RecBatch {
    let rws: Vec<u32> = crops
        .iter()
        .map(|c| {
            let ratio = c.width() as f32 / c.height().max(1) as f32;
            ((REC_H as f32) * ratio).ceil() as u32
        })
        .map(|w| w.clamp(1, REC_W_MAX))
        .collect();
    let w_batch = rws.iter().copied().max().unwrap_or(REC_W_DEFAULT).max(REC_W_DEFAULT);
    let w_batch = w_batch.min(REC_W_MAX);
    let n = crops.len();
    let plane = (REC_H as usize) * (w_batch as usize);
    let mut data = vec![0.0f32; n * 3 * plane];
    for (i, crop) in crops.iter().enumerate() {
        let w_i = rws[i].min(w_batch);
        let resized = resize_tri(crop, w_i, REC_H);
        let base = i * 3 * plane;
        for y in 0..REC_H as usize {
            for x in 0..w_i as usize {
                let p = resized.get_pixel(x as u32, y as u32);
                let idx = base + y * (w_batch as usize) + x;
                // (v/255 - 0.5) / 0.5 — нормализация симметрична, порядок каналов неважен
                data[idx] = (p[0] as f32 / 255.0 - 0.5) / 0.5;
                data[base + plane + y * (w_batch as usize) + x] =
                    (p[1] as f32 / 255.0 - 0.5) / 0.5;
                data[base + 2 * plane + y * (w_batch as usize) + x] =
                    (p[2] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
    }
    RecBatch {
        data,
        shape: [n, 3, REC_H as usize, w_batch as usize],
    }
}

/// CTC-декодирование: argmax по моментам времени, схлопывание повторов,
/// пропуск blank; уверенность — средняя вероятность выбранных символов.
pub fn ctc_decode(probs: &[f32], t: usize, c: usize, table: &[String]) -> (String, f32) {
    let mut text = String::new();
    let mut confs: Vec<f32> = Vec::new();
    let mut prev: usize = usize::MAX;
    for ti in 0..t {
        let row = &probs[ti * c..(ti + 1) * c];
        let (mut bi, mut bv) = (0usize, f32::MIN);
        for (ci, &v) in row.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = ci;
            }
        }
        if bi != prev && bi != 0 && bi < table.len() {
            text.push_str(&table[bi]);
            confs.push(bv);
        }
        prev = bi;
    }
    let conf = if confs.is_empty() {
        0.0
    } else {
        confs.iter().sum::<f32>() / confs.len() as f32
    };
    (text, conf)
}
