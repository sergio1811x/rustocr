//! Восстановление структуры таблицы страницы: полная сетка «строка × столбец»
//! с объединёнными ячейками (colspan) и пометкой «настоящих» табличных строк.
//!
//! Это детерминированный «аналог SLANet»: нейросети PP-Structure решают ту же
//! задачу (сетка + спаны) для сложных таблиц без просветов; в сметах же
//! колонки разнесены устойчивыми просветами, и геометрия даёт результат
//! не хуже — при нулевой стоимости инференса. Интерфейс `PageTable` совместим
//! с выходом SLANet-класса моделей: при появлении ONNX-модели её вывод
//! мапится в те же структуры (см. README, «Сметный конвейер»).
//!
//! Алгоритм (перенесён из table.rs без изменения сути, проверен на 17 119
//! числах реальных смет):
//!   1) строки — группировка блоков по Y-центрам (допуск = половина медианной
//!      высоты блока);
//!   2) столбцы — вертикальные «просветы» в самой картинке по полосам Y,
//!      занятым табличными строками (линии сетки выкидываются); запасной путь
//!      без картинки — объединение интервалов блоков;
//!   3) склеенные ячейки (блок, накрывший границу колонок) режутся по границам
//!      и перечитываются распознавателем по кускам;
//!   4) colspan — сколько колонок накрывает непорезанный блок (объединённая
//!      ячейка «Итого по разделу…»).

use crate::geom::Pt;
use crate::pipeline::{Block, OcrEngine, OcrResult};
use image::RgbImage;

/// Геометрия блока: (x-центр, y-центр, ширина, высота).
pub fn block_geom(b: &Block) -> (f32, f32, f32, f32) {
    let xs = [b.quad[0].x, b.quad[1].x, b.quad[2].x, b.quad[3].x];
    let ys = [b.quad[0].y, b.quad[1].y, b.quad[2].y, b.quad[3].y];
    let x0 = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let x1 = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let y0 = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let y1 = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (x0 + (x1 - x0) / 2.0, y0 + (y1 - y0) / 2.0, x1 - x0, y1 - y0)
}

/// Ячейка восстановленной таблицы.
#[derive(Clone, Debug)]
pub struct TableCell {
    /// индекс колонки (0-based, слева направо)
    pub col: usize,
    /// сколько колонок накрывает исходный блок (объединённая ячейка)
    pub colspan: usize,
    pub text: String,
    /// уверенность OCR блока/куска
    pub conf: f32,
}

/// Строка страницы: табличная (несколько ячеек) или текстовая (одна «ячейка»).
#[derive(Clone, Debug)]
pub struct TableRow {
    /// средний Y-центр строки (для порядка чтения)
    pub y: f32,
    /// настоящая табличная строка: ≥2 блока с просветом между колонками
    pub columnar: bool,
    pub cells: Vec<TableCell>,
}

/// Восстановленная таблица страницы.
#[derive(Clone, Debug)]
pub struct PageTable {
    /// полосы колонок [x0, x1) слева направо
    pub bands: Vec<(f32, f32)>,
    /// строки сверху вниз
    pub rows: Vec<TableRow>,
    /// средняя высота блока на странице (для эвристик)
    pub med_h: f32,
}

impl PageTable {
    /// Число колонок.
    pub fn n_cols(&self) -> usize {
        self.bands.len()
    }
}

/// Анализ страницы: таблица, если похоже на неё (как в table.rs — иначе None,
/// и вызывающий работает с простыми текстовыми строками).
/// `engine` нужен только при наличии `img` (перечтение разрезанных ячеек).
pub fn analyze_page(
    engine: Option<&OcrEngine>,
    img: Option<&RgbImage>,
    r: &OcrResult,
) -> Option<PageTable> {
    let blocks: Vec<&Block> = r
        .blocks
        .iter()
        .filter(|b| !b.text.trim().is_empty())
        .collect();
    if blocks.len() < 9 {
        return None;
    }

    let mut heights: Vec<f32> = blocks.iter().map(|b| block_geom(b).3).collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_h = heights[heights.len() / 2].max(1.0);

    // 1) строки: группировка по Y-центрам (как в table.rs)
    let tol = 0.5 * med_h;
    let mut idx: Vec<usize> = (0..blocks.len()).collect();
    idx.sort_by(|&a, &b| {
        block_geom(blocks[a])
            .1
            .partial_cmp(&block_geom(blocks[b]).1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut row_cy: Vec<f32> = Vec::new();
    for &i in idx.iter() {
        let cy = block_geom(blocks[i]).1;
        let joined = rows
            .last_mut()
            .zip(row_cy.last_mut())
            .map(|(members, rcy)| {
                if (cy - *rcy).abs() <= tol {
                    members.push(i);
                    *rcy = (*rcy * (members.len() - 1) as f32 + cy) / members.len() as f32;
                    true
                } else {
                    false
                }
            })
            .unwrap_or(false);
        if !joined {
            rows.push(vec![i]);
            row_cy.push(cy);
        }
    }

    // 2) столбцы: просветы в картинке по полосам табличных строк
    let is_columnar = |m: &Vec<usize>| -> bool {
        if m.len() < 2 {
            return false;
        }
        let mut xs: Vec<(f32, f32)> = m
            .iter()
            .map(|&i| {
                let (cx, _, w, _) = block_geom(blocks[i]);
                (cx - w / 2.0, cx + w / 2.0)
            })
            .collect();
        xs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut max_gap = 0.0f32;
        for wnd in xs.windows(2) {
            let g = wnd[1].0 - wnd[0].1;
            if g > max_gap {
                max_gap = g;
            }
        }
        max_gap >= (med_h).max(25.0)
    };
    let table_rows: Vec<&Vec<usize>> = rows
        .iter()
        .filter(|m| m.len() >= 2 && is_columnar(m))
        .collect();
    let table_blocks: Vec<usize> = table_rows.iter().flat_map(|m| m.iter().copied()).collect();
    if table_blocks.is_empty() {
        return None;
    }

    let debug = std::env::var("RUSTOCR_TABLE_DEBUG").is_ok();

    let mut boundaries: Vec<f32> = Vec::new();
    if let Some(img) = img {
        let x_of = |i: usize| block_geom(blocks[i]).0 - block_geom(blocks[i]).2 / 2.0;
        let x_end = |i: usize| block_geom(blocks[i]).0 + block_geom(blocks[i]).2 / 2.0;
        let rx0 = table_blocks
            .iter()
            .map(|&i| x_of(i))
            .fold(f32::INFINITY, f32::min)
            .max(0.0) as u32;
        let rx1 = table_blocks
            .iter()
            .map(|&i| x_end(i))
            .fold(f32::NEG_INFINITY, f32::max)
            .min(img.width() as f32 - 1.0) as u32;
        if rx1 > rx0 {
            let w = (rx1 - rx0) as usize;
            let mut prof = vec![0u32; w];
            let mut span_px = 0u32;
            let is_dark = |x: u32, y: u32| -> bool {
                let p = img.get_pixel(x, y);
                (p[0] as u32 + p[1] as u32 + p[2] as u32) / 3 < 140
            };
            for row in &table_rows {
                let y0 = row
                    .iter()
                    .map(|&i| block_geom(blocks[i]).1 - block_geom(blocks[i]).3 / 2.0)
                    .fold(f32::NEG_INFINITY, f32::max)
                    .max(0.0) as u32;
                let y1 = row
                    .iter()
                    .map(|&i| block_geom(blocks[i]).1 + block_geom(blocks[i]).3 / 2.0)
                    .fold(f32::INFINITY, f32::min)
                    .min(img.height() as f32 - 1.0) as u32;
                if y1 <= y0 + 1 {
                    continue;
                }
                span_px += y1 - y0;
                for y in y0..y1 {
                    let mut darks = vec![false; w];
                    let mut dark_cnt = 0u32;
                    for xi in 0..w {
                        if is_dark(rx0 + xi as u32, y) {
                            darks[xi] = true;
                            dark_cnt += 1;
                        }
                    }
                    if dark_cnt as f32 > 0.6 * w as f32 {
                        continue; // сплошная линия сетки
                    }
                    let (mut longest, mut run, mut gap) = (0usize, 0usize, 0usize);
                    for &f in darks.iter() {
                        if f {
                            run += 1;
                            gap = 0;
                            longest = longest.max(run);
                        } else if run > 0 {
                            gap += 1;
                            if gap > 3 {
                                run = 0;
                            }
                        }
                    }
                    if longest as f32 > 0.5 * w as f32 {
                        continue; // сглаженная линия сетки
                    }
                    for xi in 0..w {
                        if darks[xi] {
                            prof[xi] += 1;
                        }
                    }
                }
            }
            if span_px > 0 {
                let thresh = ((span_px as f32) * 0.004).max(2.0);
                let mut run_start: Option<usize> = None;
                for xi in 0..=w {
                    let dark = xi < w && prof[xi] as f32 >= thresh;
                    match (run_start, dark) {
                        (Some(s), true) => {
                            if xi >= s + 18 {
                                boundaries.push((s + xi) as f32 / 2.0 + rx0 as f32);
                            }
                            run_start = None;
                        }
                        (None, false) => run_start = Some(xi),
                        _ => {}
                    }
                }
                let mut merged: Vec<f32> = Vec::new();
                for b in boundaries.drain(..) {
                    if let Some(last) = merged.last_mut() {
                        if b - *last < 20.0 {
                            *last = (*last + b) / 2.0;
                            continue;
                        }
                    }
                    merged.push(b);
                }
                boundaries = merged;
                if debug {
                    eprintln!(
                        "[table] просветы: порог {:.1}, границ {} → {:?}",
                        thresh,
                        boundaries.len(),
                        boundaries.iter().map(|b| b.round()).collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    let bands: Vec<(f32, f32)> = if boundaries.is_empty() {
        let mut ivs: Vec<(f32, f32)> = table_blocks
            .iter()
            .map(|&i| {
                let (cx, _, w, _) = block_geom(blocks[i]);
                (cx - w / 2.0, cx + w / 2.0)
            })
            .collect();
        ivs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut bs: Vec<(f32, f32)> = Vec::new();
        for (a, b) in ivs {
            if let Some(last) = bs.last_mut() {
                if a <= last.1 + 15.0 {
                    last.1 = last.1.max(b);
                    continue;
                }
            }
            bs.push((a, b));
        }
        bs
    } else {
        let left = table_blocks
            .iter()
            .map(|&i| block_geom(blocks[i]).0 - block_geom(blocks[i]).2 / 2.0)
            .fold(f32::INFINITY, f32::min);
        let right = table_blocks
            .iter()
            .map(|&i| block_geom(blocks[i]).0 + block_geom(blocks[i]).2 / 2.0)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut bs: Vec<(f32, f32)> = Vec::with_capacity(boundaries.len() + 1);
        let mut prev = left;
        for &b in &boundaries {
            bs.push((prev, b));
            prev = b;
        }
        bs.push((prev, right));
        bs
    };
    if bands.len() < 2 {
        return None; // один столбец — не таблица
    }

    // Похоже ли на таблицу: ≥2,0 блока на табличную строку. Двухколоночные
    // списочные сметы «наименование | цена» — тоже таблица.
    let avg_per_row = table_blocks.len() as f32 / table_rows.len() as f32;
    if avg_per_row < 2.0 || rows.len() < 3 {
        return None;
    }

    // 3) раскладка текста: куски блока → колонки; склеенные блоки режем
    //    по границам и перечитываем.
    let band_of = |x: f32| -> usize {
        let mut best = 0usize;
        let mut bd = f32::INFINITY;
        for (i, &(a, b)) in bands.iter().enumerate() {
            let c = (a + b) / 2.0;
            let d = (c - x).abs();
            if d < bd {
                bd = d;
                best = i;
            }
        }
        best
    };
    // сколько колонок накрывает диапазон x (объединённая ячейка)
    let span_of = |x0: f32, x1: f32| -> usize {
        let n = bands
            .iter()
            .filter(|&&(a, b)| {
                let c = (a + b) / 2.0;
                c >= x0 - 2.0 && c <= x1 + 2.0
            })
            .count();
        n.max(1)
    };

    // размещённые куски блока: (col, colspan, text, conf, x0)
    let mut placed: Vec<Vec<(usize, usize, String, f32, f32)>> = blocks
        .iter()
        .map(|b| {
            let (cx, _, w, _) = block_geom(b);
            vec![(
                band_of(cx),
                span_of(cx - w / 2.0, cx + w / 2.0),
                b.text.trim().to_string(),
                b.confidence,
                cx - w / 2.0,
            )]
        })
        .collect();

    if let (Some(engine), Some(img)) = (engine, img) {
        for (bi, blk) in blocks.iter().enumerate() {
            let (cx, _, w, _) = block_geom(blk);
            let x0 = cx - w / 2.0;
            let x1 = cx + w / 2.0;
            let mut cuts: Vec<f32> = Vec::new();
            for &bnd in &boundaries {
                if x0 + 6.0 < bnd && bnd < x1 - 6.0 {
                    cuts.push(bnd);
                }
            }
            if cuts.is_empty() {
                continue;
            }
            let mut edges = vec![x0];
            edges.extend(cuts.iter().copied());
            edges.push(x1);
            let mut sub_quads: Vec<[Pt; 4]> = Vec::new();
            let mut strip_range: Vec<(f32, f32)> = Vec::new();
            for pair in edges.windows(2) {
                let (xa, xb) = (pair[0], pair[1]);
                if xb - xa < 4.0 {
                    continue;
                }
                sub_quads.push(clamp_quad_x(&blk.quad, xa, xb));
                strip_range.push((xa, xb));
            }
            if sub_quads.is_empty() {
                continue;
            }
            let rec = engine.recognize_quads(img, &sub_quads);
            let mut parts: Vec<(usize, usize, String, f32, f32)> = Vec::new();
            for (k, slot) in rec.iter().enumerate() {
                if let Some((t, c)) = slot {
                    let t = t.trim();
                    if t.is_empty() || *c < 0.5 {
                        continue;
                    }
                    let (xa, xb) = strip_range[k];
                    parts.push((
                        band_of((xa + xb) / 2.0),
                        span_of(xa, xb),
                        t.to_string(),
                        *c,
                        xa,
                    ));
                }
            }
            if parts.len() >= 2 {
                placed[bi] = parts;
            }
        }
    }

    // 4) строки: слева направо; два куска в одной ячейке — склейка пробелом
    let mut out_rows: Vec<TableRow> = Vec::with_capacity(rows.len());
    for (ri, row) in rows.iter().enumerate() {
        let columnar = row.len() >= 2 && is_columnar(row);
        let mut pieces: Vec<(usize, usize, String, f32, f32)> = Vec::new();
        for &block_i in row.iter() {
            pieces.extend(placed[block_i].iter().cloned());
        }
        pieces.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));
        let mut cells: Vec<TableCell> = Vec::new();
        for (col, colspan, text, conf, _) in pieces {
            if let Some(last) = cells.last_mut() {
                if last.col == col {
                    last.text.push(' ');
                    last.text.push_str(&text);
                    continue;
                }
            }
            cells.push(TableCell {
                col,
                colspan,
                text,
                conf,
            });
        }
        let y = row_cy.get(ri).copied().unwrap_or_default();
        out_rows.push(TableRow { y, columnar, cells });
    }

    if debug {
        eprintln!(
            "[table] блоков: {}, строк: {}, полос: {}",
            blocks.len(),
            out_rows.len(),
            bands.len()
        );
    }

    Some(PageTable {
        bands,
        rows: out_rows,
        med_h,
    })
}

/// Подрезать X-координаты углов четырёхугольника в полосу [a, b].
fn clamp_quad_x(q: &[Pt; 4], a: f32, b: f32) -> [Pt; 4] {
    let clamp = |mut p: Pt| {
        p.x = p.x.clamp(a, b);
        p
    };
    [clamp(q[0]), clamp(q[1]), clamp(q[2]), clamp(q[3])]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Pt;

    fn mk_block(text: &str, x0: f32, y0: f32, x1: f32, y1: f32) -> Block {
        Block {
            text: text.to_string(),
            confidence: 0.95,
            lang: "ru",
            quad: [
                Pt { x: x0, y: y0 },
                Pt { x: x1, y: y0 },
                Pt { x: x1, y: y1 },
                Pt { x: x0, y: y1 },
            ],
        }
    }

    fn mk_result(blocks: Vec<Block>) -> OcrResult {
        OcrResult {
            width: 500,
            height: 400,
            text: String::new(),
            blocks,
            skipped: 0,
            det_ms: 0,
            rec_ms: 0,
            total_ms: 0,
        }
    }

    /// Сетка из синтетических блоков: 3 колонки × 4 строки (без картинки —
    /// запасной путь поиска полос по интервалам блоков).
    #[test]
    fn grid_from_blocks() {
        let cols: [(f32, f32); 3] = [(20.0, 120.0), (180.0, 260.0), (330.0, 460.0)];
        let mut blocks = Vec::new();
        for (ri, y) in [40.0, 110.0, 180.0, 250.0].iter().enumerate() {
            for (ci, &(a, b)) in cols.iter().enumerate() {
                blocks.push(mk_block(&format!("r{}c{}", ri, ci), a, *y, b, y + 30.0));
            }
        }
        let r = mk_result(blocks);
        let t = analyze_page(None, None, &r).expect("таблица не распознана");
        assert_eq!(t.n_cols(), 3, "полос: {:?}", t.bands);
        assert_eq!(t.rows.len(), 4);
        assert!(t.rows.iter().all(|r| r.columnar), "все строки табличные");
        assert_eq!(t.rows[1].cells[2].text, "r1c2");
        assert!(t.rows[1].cells.iter().all(|c| c.colspan == 1));
    }

    /// Полноширинная строка (заголовок раздела) — не табличная, colspan = 3.
    #[test]
    fn full_width_row_is_not_columnar() {
        let cols: [(f32, f32); 3] = [(20.0, 120.0), (180.0, 260.0), (330.0, 460.0)];
        let mut blocks = Vec::new();
        blocks.push(mk_block("Раздел 1. Фундамент", 20.0, 20.0, 460.0, 50.0));
        for (ri, y) in [90.0, 160.0, 230.0].iter().enumerate() {
            for (ci, &(a, b)) in cols.iter().enumerate() {
                blocks.push(mk_block(&format!("r{}c{}", ri, ci), a, *y, b, y + 30.0));
            }
        }
        let r = mk_result(blocks);
        let t = analyze_page(None, None, &r).expect("таблица");
        assert_eq!(t.rows.len(), 4);
        assert!(!t.rows[0].columnar, "полноширинная строка не табличная");
        assert_eq!(t.rows[0].cells.len(), 1);
        assert_eq!(t.rows[0].cells[0].colspan, 3, "накрывает все колонки");
        assert_eq!(t.rows[0].cells[0].text, "Раздел 1. Фундамент");
        assert!(t.rows[1].columnar);
    }

    /// Слишком мало блоков — не таблица.
    #[test]
    fn few_blocks_not_a_table() {
        let blocks = vec![mk_block("один", 0.0, 0.0, 10.0, 10.0)];
        assert!(analyze_page(None, None, &mk_result(blocks)).is_none());
    }
}
