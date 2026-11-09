//! Сборка XLSX из распознанных страниц. Таблицы восстанавливает модуль
//! estimate::structure (полная сетка строк/столбцов, склейка кусков ячеек,
//! перечтение разрезанных блоков) — здесь остаётся только раскладка
//! «строка × столбец → ячейка Excel». Страница без явной таблицы
//! выгружается одной колонкой текста.

use crate::estimate::structure;
use crate::pipeline::{OcrEngine, OcrResult};
use anyhow::{Context, Result};
use rust_xlsxwriter::Workbook;
use std::path::Path;

/// Записать все страницы в один XLSX (по листу на страницу).
/// `images[i]` — исходная картинка страницы `results[i]`.
pub fn write_xlsx(
    engine: &OcrEngine,
    images: &[image::RgbImage],
    results: &[OcrResult],
    path: &Path,
) -> Result<usize> {
    let mut wb = Workbook::new();
    for (i, r) in results.iter().enumerate() {
        let name = format!("Стр. {}", i + 1);
        let ws = wb
            .add_worksheet()
            .set_name(name.as_str())
            .with_context(|| format!("имя листа {}", name))?;
        let img = images.get(i);
        write_page(engine, img, ws, r)
            .with_context(|| format!("страница {}", i + 1))?;
    }
    wb.save(path)
        .with_context(|| format!("запись {}", path.display()))?;
    Ok(results.len())
}

/// Ошибки записи ячеек не игнорируем: неполный файл без сообщения —
/// хуже явной ошибки.
fn write_page(
    engine: &OcrEngine,
    img: Option<&image::RgbImage>,
    ws: &mut rust_xlsxwriter::Worksheet,
    r: &OcrResult,
) -> Result<()> {
    if let Some(grid) = page_grid(engine, img, r) {
        for (row, col, text) in grid {
            ws.write(row as u32, col as u16, text.as_str())
                .with_context(|| format!("ячейка R{}C{}", row, col))?;
        }
    } else {
        // не таблица: по строке на лист-строку
        for (i, line) in r.text.lines().enumerate() {
            ws.write(i as u32, 0u16, line)
                .with_context(|| format!("строка текста {}", i))?;
        }
    }
    Ok(())
}

/// Геометрия страницы → ячейки (строка, столбец, текст), если похоже на таблицу.
fn page_grid(
    engine: &OcrEngine,
    img: Option<&image::RgbImage>,
    r: &OcrResult,
) -> Option<Vec<(usize, usize, String)>> {
    let t = structure::analyze_page(Some(engine), img, r)?;
    let mut cells: Vec<(usize, usize, String)> = Vec::new();
    for (ri, row) in t.rows.iter().enumerate() {
        for c in &row.cells {
            // два куска в одной ячейке — склейка пробелом (на случай,
            // когда structure оставил их раздельными)
            if let Some(last) = cells.last_mut() {
                if last.0 == ri && last.1 == c.col {
                    last.2.push(' ');
                    last.2.push_str(&c.text);
                    continue;
                }
            }
            cells.push((ri, c.col, c.text.clone()));
        }
    }
    Some(cells)
}
