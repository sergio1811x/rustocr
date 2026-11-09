//! Сметный конвейер: OCR → структура таблиц → семантика (иерархия
//! раздел → подраздел → позиция → ресурсы) → валидация → структурный JSON
//! и DTO ПАЗЛ. Точка входа — `parse_estimate` (CLI/сервер отдают результат
//! в JSON).

pub mod model;
pub mod pazl;
pub mod semantics;
pub mod structure;
pub mod validate;

use crate::pipeline::{OcrEngine, OcrResult};
use image::RgbImage;

pub use model::EstimateDocument;

/// Страница на вход конвейера.
pub struct PageInput<'a> {
    /// номер страницы (1-based)
    pub page: usize,
    /// картинка нужна для поиска колонок по просветам и перечтения
    /// разрезанных ячеек; без неё — запасной чисто геометрический путь
    pub img: Option<&'a RgbImage>,
    pub result: &'a OcrResult,
}

/// Строки одной страницы в терминах семантического парсера.
pub fn page_rows(
    engine: Option<&OcrEngine>,
    img: Option<&RgbImage>,
    result: &OcrResult,
    page: usize,
) -> Vec<semantics::DocRow> {
    match structure::analyze_page(engine, img, result) {
        Some(t) => t
            .rows
            .iter()
            .filter(|r| !r.cells.is_empty())
            .map(|tr| semantics::DocRow {
                page,
                columnar: tr.columnar,
                cells: tr
                    .cells
                    .iter()
                    .map(|c| (c.col, c.colspan, c.text.clone()))
                    .collect(),
            })
            .collect(),
        None => result
            .text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| semantics::DocRow {
                page,
                columnar: false,
                cells: vec![(0, 1, l.trim().to_string())],
            })
            .collect(),
    }
}

/// Собрать документ из уже готовых строк (сервер собирает их по мере
/// рендера страниц — картинки не держатся в памяти).
pub fn parse_rows(rows: Vec<semantics::DocRow>, pages: usize) -> EstimateDocument {
    let mut doc = semantics::parse_document(rows);
    doc.pages = pages;
    validate::validate(&mut doc);
    doc
}

/// Полный конвейер по страницам (CLI: картинки уже в памяти).
pub fn parse_estimate(engine: Option<&OcrEngine>, pages: &[PageInput]) -> EstimateDocument {
    let mut rows: Vec<semantics::DocRow> = Vec::new();
    for p in pages {
        rows.extend(page_rows(engine, p.img, p.result, p.page));
    }
    parse_rows(rows, pages.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Полный конвейер на синтетической странице (без картинок и моделей):
    /// блоки → структура таблицы → иерархия → валидация.
    #[test]
    fn pipeline_from_blocks_to_document() {
        use crate::geom::Pt;
        use crate::pipeline::Block;

        let mk = |text: &str, x0: f32, y0: f32, x1: f32, y1: f32| Block {
            text: text.to_string(),
            confidence: 0.95,
            lang: "ru",
            quad: [
                Pt { x: x0, y: y0 },
                Pt { x: x1, y: y0 },
                Pt { x: x1, y: y1 },
                Pt { x: x0, y: y1 },
            ],
        };
        let cols: [(f32, f32); 4] = [
            (20.0, 60.0),
            (120.0, 400.0),
            (440.0, 500.0),
            (560.0, 640.0),
        ];
        let mut blocks = vec![
            mk("СМЕТА № 1 на работы", 20.0, 20.0, 640.0, 48.0),
            mk("Раздел 1. Общестроительные работы", 20.0, 80.0, 640.0, 108.0),
        ];
        // шапка
        let headers = ["№", "Наименование", "Ед.", "Сумма, руб"];
        for (ci, &(a, b)) in cols.iter().enumerate() {
            blocks.push(mk(headers[ci], a, 140.0, b, 168.0));
        }
        // позиция 1
        for (ci, &(a, b)) in cols.iter().enumerate() {
            let t = ["1", "Кладка стен", "м3", "200 000"][ci];
            blocks.push(mk(t, a, 200.0, b, 228.0));
        }
        // позиция 2
        for (ci, &(a, b)) in cols.iter().enumerate() {
            let t = ["2", "Устройство кровли", "м2", "300 000"][ci];
            blocks.push(mk(t, a, 260.0, b, 288.0));
        }
        // итог (объединённая ячейка «Итого по смете» + число)
        blocks.push(mk("Итого по смете", 120.0, 320.0, 400.0, 348.0));
        blocks.push(mk("500 000", 560.0, 320.0, 640.0, 348.0));

        let r = OcrResult {
            width: 700,
            height: 400,
            text: String::new(),
            blocks,
            skipped: 0,
            det_ms: 0,
            rec_ms: 0,
            total_ms: 0,
        };
        let doc = parse_estimate(None, &[PageInput { page: 1, img: None, result: &r }]);
        assert_eq!(doc.meta.number.as_deref(), Some("1"));
        assert_eq!(doc.sections.len(), 1);
        assert_eq!(doc.sections[0].title, "Общестроительные работы");
        assert_eq!(doc.sections[0].positions.len(), 2);
        assert_eq!(doc.sections[0].positions[0].name, "Кладка стен");
        assert_eq!(doc.sections[0].positions[0].total.as_ref().map(|n| n.value), Some(200000.0));
        assert_eq!(doc.totals.total.as_ref().map(|n| n.value), Some(500000.0));
        // валидация: итог сметы сходится с Σ позиций
        assert!(
            !doc.validation.errors.iter().any(|e| e.kind == "doc_total_mismatch"),
            "ошибки: {:?}",
            doc.validation.errors
        );
    }
}
