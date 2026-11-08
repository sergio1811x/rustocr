//! Растеризация PDF в картинки: внешний рендерер pdftoppm (poppler-utils).
//! Ставится везде одной командой и понимает любые PDF, включая сканы.

use anyhow::{bail, Context, Result};
use image::RgbImage;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const DEFAULT_DPI: u32 = 150; // A4 → 1240×1754: достаточно для мелкой таблицы
pub const MAX_PAGES: usize = 200; // защита от «случайно скормили 5000 страниц»

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

/// Превратить PDF в набор картинок (по одной на страницу).
pub fn render_pdf(path: &Path, dpi: u32) -> Result<Vec<RgbImage>> {
    let dir = std::env::temp_dir().join(format!(
        "rustocr-pdf-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("не создать каталог {}", dir.display()))?;
    let prefix = dir.join("pg");
    let out = Command::new("pdftoppm")
        .arg("-r")
        .arg(dpi.to_string())
        .arg("-png")
        .arg("-l")
        .arg(MAX_PAGES.to_string()) // дальше лимита и не рендерим
        .arg(path)
        .arg(&prefix)
        .output();

    let res = (|| -> Result<Vec<RgbImage>> {
        match out {
            Ok(o) if o.status.success() => {}
            Ok(o) => bail!(
                "pdftoppm завершился с ошибкой: {}\n{}",
                o.status,
                String::from_utf8_lossy(&o.stderr)
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => bail!(
                "не найдена программа pdftoppm (пакет poppler-utils).\n\
                 Установите её, например: sudo apt install poppler-utils"
            ),
            Err(e) => bail!("не удалось запустить pdftoppm: {}", e),
        }
        // pdftoppm называет файлы prefix-01.png, prefix-02.png, …
        let mut files: Vec<(u64, PathBuf)> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
            .filter_map(|p| {
                let stem = p.file_stem()?.to_str()?; // "pg-12"
                let num: u64 = stem.rsplit('-').next()?.parse().ok()?;
                Some((num, p))
            })
            .collect();
        files.sort_by_key(|(n, _)| *n);
        let mut imgs = Vec::with_capacity(files.len());
        for (_, p) in files {
            imgs.push(
                image::open(&p)
                    .with_context(|| format!("чтение {}", p.display()))?
                    .to_rgb8(),
            );
        }
        if imgs.is_empty() {
            bail!("pdftoppm не создал ни одной страницы — пустой или повреждённый PDF?");
        }
        Ok(imgs)
    })();
    let _ = std::fs::remove_dir_all(&dir); // прибираем за собой
    res
}
