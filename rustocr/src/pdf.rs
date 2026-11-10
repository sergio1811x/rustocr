//! Растеризация PDF в картинки: внешний рендерер pdftoppm (poppler-utils).
//! Ставится везде одной командой и понимает любые PDF, включая сканы.
//!
//! Надёжность: уникальная временная папка на запрос (параллельные запросы
//! не сталкиваются), число страниц проверяется по pdfinfo ДО рендера,
//! растеризация идёт порциями и ограничена по времени, DPI зажат в
//! разумный диапазон, слишком большая страница отвергается по заголовку.

use anyhow::{bail, Context, Result};
use image::RgbImage;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const DEFAULT_DPI: u32 = 200; // A4 → 1654×2339: достаточно для мелкой таблицы
pub const MAX_PAGES: usize = 200; // защита от «случайно скормили 5000 страниц»
pub const MIN_DPI: u32 = 72;
pub const MAX_DPI: u32 = 400;
pub const CHUNK_PAGES: usize = 8; // страниц за вызов pdftoppm — память не растёт с PDF
const RENDER_TIMEOUT_S: u64 = 600; // лимит времени растеризации одной порции
const MAX_PAGE_PIXELS: u64 = 120_000_000; // 120 Мп на страницу — выше отвергаем

/// Похожи ли байты на PDF (магия "%PDF-")?
pub fn is_pdf_bytes(bytes: &[u8]) -> bool {
    bytes.len() >= 5 && &bytes[..5] == b"%PDF-"
}

/// Похоже ли имя файла на PDF?
pub fn is_pdf_path(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("pdf"))
        .unwrap_or(false)
}

/// Зажать DPI в разумный диапазон.
pub fn clamp_dpi(dpi: u32) -> u32 {
    dpi.clamp(MIN_DPI, MAX_DPI)
}

/// Число страниц по pdfinfo — до рендеринга, чтобы сразу отвергать
/// гигантские документы, не тратя минуты на растеризацию.
pub fn page_count(path: &Path) -> Result<usize> {
    let out = Command::new("pdfinfo")
        .arg(path)
        .output()
        .map_err(|e| anyhow::anyhow!("не запустить pdfinfo (нужен poppler-utils): {}", e))?;
    if !out.status.success() {
        bail!(
            "PDF повреждён или не читается: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Pages:") {
            if let Ok(n) = rest.trim().parse::<usize>() {
                return Ok(n);
            }
        }
    }
    bail!("pdfinfo не сообщил число страниц")
}

/// DPI рендера PDF из окружения (RUSTOCR_PDF_DPI, запасное имя RUSTOCR_DPI).
pub fn dpi_from_env() -> u32 {
    std::env::var("RUSTOCR_PDF_DPI")
        .or_else(|_| std::env::var("RUSTOCR_DPI"))
        .ok()
        .and_then(|v| v.parse().ok())
        .map(clamp_dpi)
        .unwrap_or(DEFAULT_DPI)
}

/// Уникальное имя временной папки: PID + наносекунды + счётчик запросов.
fn temp_name() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("rustocr-pdf-{}-{}-{}", std::process::id(), nanos, seq)
}

/// Временная папка растеризации: уникальное имя (параллельные запросы
/// не сталкиваются) и гарантированная уборка при любом исходе (RAII).
pub struct PdfRender {
    dir: PathBuf,
}

impl PdfRender {
    pub fn new() -> Result<Self> {
        let dir = std::env::temp_dir().join(temp_name());
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("не создать каталог {}", dir.display()))?;
        Ok(Self { dir })
    }

    /// Куда положить сам PDF внутри временной папки.
    pub fn pdf_path(&self) -> PathBuf {
        self.dir.join("in.pdf")
    }

    /// Растеризовать страницы [first, last] в PNG и прочитать их.
    pub fn render_range(
        &self,
        path: &Path,
        first: u32,
        last: u32,
        dpi: u32,
    ) -> Result<Vec<RgbImage>> {
        let dpi = clamp_dpi(dpi);
        let prefix = self.dir.join("pg");
        let out = Command::new("timeout")
            .arg(RENDER_TIMEOUT_S.to_string())
            .arg("pdftoppm")
            .arg("-r")
            .arg(dpi.to_string())
            .arg("-png")
            .arg("-f")
            .arg(first.to_string())
            .arg("-l")
            .arg(last.to_string())
            .arg(path)
            .arg(&prefix)
            .output()
            .map_err(|e| {
                anyhow::anyhow!(
                    "не запустить timeout/pdftoppm (нужен poppler-utils): {}",
                    e
                )
            })?;
        if !out.status.success() {
            let msg = if out.status.code() == Some(124) {
                format!(
                    "превышен лимит времени растеризации ({} с)",
                    RENDER_TIMEOUT_S
                )
            } else {
                String::from_utf8_lossy(&out.stderr).trim().to_string()
            };
            bail!(
                "pdftoppm не смог растеризовать страницы {}–{}: {}",
                first,
                last,
                msg
            );
        }
        // pdftoppm называет файлы pg-01.png, pg-02.png, …
        let mut numbered: Vec<(u32, PathBuf)> = Vec::new();
        for entry in std::fs::read_dir(&self.dir)
            .with_context(|| format!("не прочитать {}", self.dir.display()))?
            .flatten()
        {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(num) = name
                .strip_prefix("pg-")
                .and_then(|s| s.strip_suffix(".png"))
                .and_then(|s| s.parse::<u32>().ok())
            {
                if num >= first && num <= last {
                    numbered.push((num, entry.path()));
                }
            }
        }
        numbered.sort_by_key(|(n, _)| *n);
        let mut imgs = Vec::with_capacity(numbered.len());
        for (num, p) in numbered {
            // гигантскую страницу отвергаем до декодирования — по заголовку
            let (w, h) = image::image_dimensions(&p)
                .with_context(|| format!("страница {} не читается", num))?;
            if (w as u64) * (h as u64) > MAX_PAGE_PIXELS {
                bail!(
                    "страница {} слишком большая ({}×{}): уменьшите DPI (максимум {})",
                    num,
                    w,
                    h,
                    MAX_DPI
                );
            }
            let img = crate::imgio::open_file(&p)
                .with_context(|| format!("декодирование страницы {}", num))?;
            // выцветший скан: выравниваем фон до распознавания (и текст,
            // и таблицы в Excel получают одинаково подготовленную страницу)
            let mut rgb = img.to_rgb8();
            crate::imgio::flatten_if_faded(&mut rgb);
            imgs.push(rgb);
            let _ = std::fs::remove_file(&p);
        }
        if imgs.is_empty() {
            bail!(
                "страницы {}–{} не растеризовались — пустой или повреждённый PDF?",
                first,
                last
            );
        }
        Ok(imgs)
    }
}

impl Drop for PdfRender {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Превратить PDF в набор картинок (по одной на страницу).
/// Сначала pdfinfo (число страниц и лимит), потом рендер порциями:
/// память не растёт с толщиной PDF. Картинки копятся все — они нужны
/// CLI для восстановления таблиц (раскладка колонок по пикселям).
pub fn render_pdf(path: &Path, dpi: u32) -> Result<Vec<RgbImage>> {
    let n = page_count(path)?;
    if n > MAX_PAGES {
        bail!("в PDF {} страниц — максимум {}", n, MAX_PAGES);
    }
    let r = PdfRender::new()?;
    let mut out = Vec::with_capacity(n);
    for first in (1..=n as u32).step_by(CHUNK_PAGES) {
        let last = (first + CHUNK_PAGES as u32 - 1).min(n as u32);
        out.extend(r.render_range(path, first, last, dpi)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_names_unique() {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..500 {
            assert!(seen.insert(temp_name()), "имя временной папки повторилось");
        }
    }

    #[test]
    fn dpi_clamped() {
        assert_eq!(clamp_dpi(10), MIN_DPI);
        assert_eq!(clamp_dpi(10000), MAX_DPI);
        assert_eq!(clamp_dpi(200), 200);
    }
}
