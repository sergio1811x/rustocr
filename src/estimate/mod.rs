//! Сметный конвейер: OCR → структура таблиц → семантика (иерархия
//! раздел → подраздел → позиция → ресурсы) → валидация → структурный JSON
//! и DTO ПАЗЛ. Точка входа — `parse_estimate` (CLI/сервер отдают результат
//! в JSON).

pub mod model;
pub mod pazl;
pub mod repair;
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

/// «Плохость» разобранной сметы по валидации: ошибки арифметики дороже
/// предупреждений. Используется retry-проходом (перечтение в большем DPI):
/// перечитываем документ только когда арифметика не сходится, и оставляем
/// лучший из двух вариантов — перечитывание никогда не ухудшает результат.
pub fn validation_score(doc: &EstimateDocument) -> u64 {
    let mut s = 0u64;
    for e in &doc.validation.errors {
        s += match e.kind.as_str() {
            "doc_total_mismatch" | "section_total_mismatch" | "total_with_vat_mismatch" => 10,
            _ => 10,
        };
    }
    for w in &doc.validation.warnings {
        s += match w.kind.as_str() {
            "qty_price_mismatch" => 2,
            "section_total_missing" | "doc_total_missing" => 3,
            "vat_mismatch" => 2,
            "auto_repair" => 1,
            _ => 0,
        };
    }
    s
}

/// Страницы с «подозрительными» итогами: разделы без итога (бледная печать
/// итогов «серым по серому» — детектор их не видит) и позиции, где
/// кол-во × цена ≠ сумма. На retry-проходе этим страницам принудительно
/// выравнивается фон (обычный текст уже распознался, а вот итоги «в тумане»).
pub fn suspect_pages(doc: &EstimateDocument) -> Vec<usize> {
    let mut v: Vec<usize> = Vec::new();
    let mut push_pages = |pages: &Vec<usize>| {
        for &p in pages {
            if p >= 1 && !v.contains(&p) {
                v.push(p);
            }
        }
    };
    for sec in &doc.sections {
        if sec.total.is_none() && !(sec.positions.is_empty() && sec.subsections.is_empty()) {
            push_pages(&sec.pages);
        }
        // итог есть, но арифметика с Σ позиций не сходится — ячейка итога
        // или одной из позиций прочитана с заменой цифр
        if let Some(t) = &sec.total {
            let mut computed = 0.0f64;
            for p in sec.all_positions() {
                if let Some(pt) = &p.total {
                    computed += pt.value;
                } else if let (Some(q), Some(pr)) = (&p.quantity, &p.price) {
                    computed += q.value * pr.value;
                }
            }
            for r in &sec.resources {
                if let Some(rt) = &r.total {
                    computed += rt.value;
                }
            }
            if computed > 0.0 && (t.value - computed).abs() > (2.0_f64).max(t.value * 0.005) {
                push_pages(&sec.pages);
            }
        }
    }
    for pos in doc.all_positions() {
        if let (Some(q), Some(pr), Some(t)) = (&pos.quantity, &pos.price, &pos.total) {
            let expect = q.value * pr.value;
            if (t.value - expect).abs() > (1.0_f64).max(expect * 0.02) {
                push_pages(&pos.pages);
            }
        }
    }
    v.sort_unstable();
    v
}

/// Запасной путь: страница без таблицы — простые текстовые строки.
fn text_rows(page: usize, result: &OcrResult) -> Vec<semantics::DocRow> {
    result
        .text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| semantics::DocRow {
            page,
            columnar: false,
            cells: vec![(0, 1, l.trim().to_string())],
        })
        .collect()
}

/// Собственные границы колонок каждой страницы (первый проход скелета).
fn own_edges(pages: &[PageInput]) -> Vec<Vec<f32>> {
    pages
        .iter()
        .map(|p| {
            structure::page_skeleton(p.img, p.result)
                .map(|s| s.boundaries)
                .unwrap_or_default()
        })
        .collect()
}

/// Кластеризация отсортированных границ с допуском 15 px → средние.
fn cluster_edges(mut edges: Vec<f32>) -> Vec<f32> {
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut out: Vec<f32> = Vec::new();
    let mut cluster: Vec<f32> = Vec::new();
    for e in edges {
        match cluster.last() {
            Some(&last) if e - last <= 15.0 => cluster.push(e),
            _ => {
                if !cluster.is_empty() {
                    out.push(cluster.iter().sum::<f32>() / cluster.len() as f32);
                }
                cluster.clear();
                cluster.push(e);
            }
        }
    }
    if !cluster.is_empty() {
        out.push(cluster.iter().sum::<f32>() / cluster.len() as f32);
    }
    out
}

/// Похожесть двух наборов границ: доля границ МЕНЬШЕГО набора, нашедших
/// пару в большом с допуском 15 px (0..1).
fn edge_affinity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let (small, big) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let matched = small
        .iter()
        .filter(|&&x| big.iter().any(|&y| (x - y).abs() <= 15.0))
        .count();
    matched as f32 / small.len() as f32
}

/// Границы колонок ПО СЕМЕЙСТВАМ страниц.
///
/// Склейка границ ВСЕГО документа («средняя температура») ломается на
/// Гранд-Сметах: после страниц позиций идут итоговые страницы со СВОИМИ
/// колонками (НР/СП/НДС, выкладки ресурсов) — их границы попадают в общий
/// набор и РЕЖУТ колонку наименований страниц позиций (перечитанные
/// куски дают «Сметная сто | имос ть 2 010,425»). Поэтому страницы
/// кластеризуются в семейства похожих наборов границ (совпадение ≥60%
/// меньшего набора): роли шапки, выученные на первой странице, действуют
/// на страницах её семейства, а страницы с другой таблицей раскладываются
/// по собственным границам. «Слепые» страницы (собственных границ <2 —
/// шапка затянула профиль просветов) примыкают к крупнейшему семейству.
fn family_boundaries(pages: &[PageInput]) -> Vec<Vec<f32>> {
    let own = own_edges(pages);
    let n = pages.len();

    // 1) жадная кластеризация: страница примыкает к первому семейству,
    //    с которым совпадает её собственный набор (или набор члена)
    let mut families: Vec<Vec<usize>> = Vec::new(); // индексы страниц
    for i in 0..n {
        let mut target: Option<usize> = None;
        for (fi, members) in families.iter().enumerate() {
            let ok = members
                .iter()
                .any(|&m| edge_affinity(&own[i], &own[m]) >= 0.6);
            if ok {
                target = Some(fi);
                break;
            }
        }
        match target {
            Some(fi) => families[fi].push(i),
            None => families.push(vec![i]),
        }
    }

    // 2) «слепые» страницы (<2 границ) примыкают к крупнейшему семейству
    if !families.is_empty() {
        let largest = (0..families.len())
            .max_by_key(|&fi| families[fi].len())
            .unwrap();
        for i in 0..n {
            if own[i].len() < 2 {
                // не в одиночном семействе — переносим
                let fi = families.iter().position(|m| m.contains(&i)).unwrap();
                if fi != largest && families[fi].len() == 1 {
                    families[fi].retain(|&m| m != i);
                    families[largest].push(i);
                }
            }
        }
        families.retain(|m| !m.is_empty());
    }

    // 3) границы каждого семейства — кластер объединения границ членов
    let mut result: Vec<Vec<f32>> = vec![Vec::new(); n];
    for members in &families {
        let edges: Vec<f32> = members
            .iter()
            .flat_map(|&m| own[m].iter().copied())
            .collect();
        let fam = cluster_edges(edges);
        for &m in members {
            result[m] = fam.clone();
        }
    }
    result
}

/// Полный конвейер по страницам (CLI: картинки уже в памяти).
pub fn parse_estimate(engine: Option<&OcrEngine>, pages: &[PageInput]) -> EstimateDocument {
    // границы колонок по семействам страниц: скелеты проходят дважды —
    // дёшево (геометрия и профиль просветов, без раскладки и перечтения)
    let fam = family_boundaries(pages);
    let mut rows: Vec<semantics::DocRow> = Vec::new();
    for (i, p) in pages.iter().enumerate() {
        let edges: &[f32] = fam
            .get(i)
            .map(|v| v.as_slice())
            .unwrap_or_default();
        let table = structure::page_skeleton(p.img, p.result)
            .and_then(|skel| structure::layout_page(skel, engine, p.img, edges));
        match table {
            Some(t) => rows.extend(
                t.rows
                    .iter()
                    .filter(|r| !r.cells.is_empty())
                    .map(|tr| semantics::DocRow {
                        page: p.page,
                        columnar: tr.columnar,
                        cells: tr
                            .cells
                            .iter()
                            .map(|c| (c.col, c.colspan, c.text.clone()))
                            .collect(),
                    }),
            ),
            None => rows.extend(text_rows(p.page, p.result)),
        }
    }
    parse_rows(rows, pages.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Скоринг валидации: ошибки дороже предупреждений; чистая смета = 0
    /// (retry-проход не запускается).
    #[test]
    fn validation_score_clean_is_zero() {
        use crate::estimate::model::{Num, Position, Section, Totals};
        let mk = |sec_total: Option<f64>| {
            let p = Position {
                line: Some(1),
                cipher: None,
                name: "Работа".to_string(),
                unit: None,
                quantity: Some(Num::new(2.0, "2")),
                price: Some(Num::new(100.0, "100")),
                total: Some(Num::new(200.0, "200")),
                resources: Vec::new(),
                pages: vec![1],
                raw: Vec::new(),
            };
            EstimateDocument {
                meta: Default::default(),
                sections: vec![Section {
                    number: None,
                    title: "Раздел".to_string(),
                    subsections: Vec::new(),
                    positions: vec![p],
                    resources: Vec::new(),
                    total: sec_total.map(|v| Num::new(v, v.to_string())),
                    pages: vec![1],
                }],
                totals: Totals {
                    total: sec_total.map(|v| Num::new(v, v.to_string())),
                    ..Default::default()
                },
                validation: Default::default(),
                unparsed_lines: Vec::new(),
                auto_repairs: 0,
                pages: 1,
            }
        };
        // сходится: score == 0, подозрительных страниц нет
        let mut doc = mk(Some(200.0));
        validate::validate(&mut doc);
        assert_eq!(
            validation_score(&doc),
            0,
            "warnings: {:?}",
            doc.validation.warnings
        );
        assert!(suspect_pages(&doc).is_empty());
        // расходится: score > 0, страница подозрительна
        let mut doc = mk(Some(999.0));
        validate::validate(&mut doc);
        assert!(validation_score(&doc) > 0);
        assert_eq!(suspect_pages(&doc), vec![1]);
    }

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
