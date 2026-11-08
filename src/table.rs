//! Сборка XLSX из распознанных страниц. Таблицы восстанавливаются геометрически:
//! блоки группируются в строки по вертикали, столбцы — по кластерам X-центров.
//! Блок, растянувшийся на несколько колонок (детектор иногда склеивает соседние
//! ячейки), режется по границам колонок и перечитывается по кускам.
//! Страница без явной таблицы выгружается одной колонкой текста.

use crate::geom::Pt;
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
        write_page(engine, img, ws, r);
    }
    wb.save(path)
        .with_context(|| format!("запись {}", path.display()))?;
    Ok(results.len())
}

fn write_page(
    engine: &OcrEngine,
    img: Option<&image::RgbImage>,
    ws: &mut rust_xlsxwriter::Worksheet,
    r: &OcrResult,
) {
    if let Some(grid) = page_grid(engine, img, r) {
        for (row, col, text) in grid {
            let _ = ws.write(row as u32, col as u16, text.as_str());
        }
    } else {
        // не таблица: по строке на лист-строку
        for (i, line) in r.text.lines().enumerate() {
            let _ = ws.write(i as u32, 0u16, line);
        }
    }
}

/// Геометрия блока: (x-центр, y-центр, ширина, высота).
fn block_geom(b: &crate::pipeline::Block) -> (f32, f32, f32, f32) {
    let xs = [b.quad[0].x, b.quad[1].x, b.quad[2].x, b.quad[3].x];
    let ys = [b.quad[0].y, b.quad[1].y, b.quad[2].y, b.quad[3].y];
    let x0 = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let x1 = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let y0 = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let y1 = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (x0 + (x1 - x0) / 2.0, y0 + (y1 - y0) / 2.0, x1 - x0, y1 - y0)
}

/// Геометрия страницы → ячейки (строка, столбец, текст), если похоже на таблицу.
fn page_grid(
    engine: &OcrEngine,
    img: Option<&image::RgbImage>,
    r: &OcrResult,
) -> Option<Vec<(usize, usize, String)>> {
    let blocks: Vec<&crate::pipeline::Block> =
        r.blocks.iter().filter(|b| !b.text.trim().is_empty()).collect();
    if blocks.len() < 9 {
        return None;
    }

    let mut heights: Vec<f32> = blocks.iter().map(|b| block_geom(b).3).collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_h = heights[heights.len() / 2].max(1.0); // медианная высота блока

    // 1) строки: группировка по Y-центрам с допуском в половину медианной
    //    высоты блока. Внутри строки центры выровнены (±3px), между строками
    //    — целый шаг таблицы; простое правило здесь надёжнее хитрых
    //    перекрытий, которые цепочкой склеивали соседние строки.
    let tol = 0.5 * med_h;
    let mut idx: Vec<usize> = (0..blocks.len()).collect();
    idx.sort_by(|&a, &b| {
        block_geom(blocks[a])
            .1
            .partial_cmp(&block_geom(blocks[b]).1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut row_cy: Vec<f32> = Vec::new(); // средний Y-центр строки
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

    // 2) столбцы: границы ищем по вертикальным «просветам» в самой картинке,
    //    причём только по полосам Y, занятым строками таблицы — так не мешают
    //    горизонтальные линии сетки, лежащие между строками. Запасной путь,
    //    если картинки нет — объединение интервалов блоков.
    // «Табличная» строка — с настоящим зазором между колонками: длинные
    // строки шапки детектор тоже режет на куски, но пробелы между кусками
    // маленькие, а между колонками — в рост высоты блока и больше.
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

    let mut boundaries: Vec<f32> = Vec::new(); // границы колонок (x)
    if let Some(img) = img {
        // область таблицы
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
            // полосы Y берём по строкам таблицы (объединение рамок блоков),
            // а горизонтальные линии сетки выкидываем: строка пикселей, в
            // которой тёмного больше 60% ширины — это линия, не текст
            let is_dark = |x: u32, y: u32| -> bool {
                let p = img.get_pixel(x, y);
                (p[0] as u32 + p[1] as u32 + p[2] as u32) / 3 < 140
            };
            for row in &table_rows {
                // ПЕРЕСЕЧЕНИЕ вертикальных рамок блоков строки: общее ядро
                // текста. Завышенная рамка («прилипла» линия сетки) не может
                // растянуть полосу — пересечение режет её до честной.
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
                    // флаги тёмных пикселей строки
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
                    // квазинепрерывный пробег (дырки ≤3 px от сглаживания):
                    // длинная почти сплошная тёмная полоса = линия сетки
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
                // долины: достаточно широкие пустые вертикальные полосы
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
                // близкие границы схлопываем (остатки линий сетки)
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
                    // мини-профиль: max по окнам в 20 px
                    let mut line = String::new();
                    for (k, chunk) in prof.chunks(20).enumerate() {
                        let m = chunk.iter().max().copied().unwrap_or(0);
                        let c = if m == 0 {
                            '.'
                        } else if (m as f32) < thresh {
                            '_'
                        } else if m < 30 {
                            '1'
                        } else if m < 100 {
                            '2'
                        } else {
                            '3'
                        };
                        line.push(c);
                        if k % 60 == 59 {
                            line.push('|');
                        }
                    }
                    eprintln!("[table] профиль (окна 20px): {}", line);
                }
            }
        }
    }

    let bands: Vec<(f32, f32)> = if boundaries.is_empty() {
        // запасной путь без картинки: объединение интервалов блоков
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
        // полосы между найденными границами
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
    if debug {
        for (ri, row) in rows.iter().enumerate() {
            let y0 = row.iter().map(|&i| block_geom(blocks[i]).1 - block_geom(blocks[i]).3 / 2.0).fold(f32::INFINITY, f32::min);
            let y1 = row.iter().map(|&i| block_geom(blocks[i]).1 + block_geom(blocks[i]).3 / 2.0).fold(f32::NEG_INFINITY, f32::max);
            let texts: Vec<&str> = row.iter().take(3).map(|&i| blocks[i].text.trim()).collect();
            eprintln!("[table] строка {}: y=[{:.0}..{:.0}] блоков={} {:?}", ri, y0, y1, row.len(), texts);
        }
        eprintln!(
            "[table] блоков: {}, строк: {}, полос: {}",
            blocks.len(),
            rows.len(),
            bands.len()
        );
        for (a, b) in &bands {
            eprintln!("[table]   полоса [{:.0}..{:.0}]", a, b);
        }
    }

    // похоже ли на таблицу
    let avg_per_row =
        table_blocks.len() as f32 / table_rows.len() as f32;
    if avg_per_row < 2.2 || rows.len() < 3 {
        return None;
    }

    // 3) собираем текст ячеек. Блок, накрывший границу между полосами
    //    (детектор склеил соседние ячейки), режем в середине зазора между
    //    полосами и перечитываем куски отдельно.
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
    let mut texts: Vec<Vec<(usize, String)>> = blocks
        .iter()
        .map(|b| {
            let (cx, _, _, _) = block_geom(b);
            vec![(band_of(cx), b.text.trim().to_string())]
        })
        .collect();

    if let Some(img) = img {
        for (bi, blk) in blocks.iter().enumerate() {
            let (cx, _, w, _) = block_geom(blk);
            let x0 = cx - w / 2.0;
            let x1 = cx + w / 2.0;
            // точки реза: границы колонок, лежащие внутри блока
            let mut cuts: Vec<f32> = Vec::new();
            for &bnd in &boundaries {
                if x0 + 6.0 < bnd && bnd < x1 - 6.0 {
                    cuts.push(bnd);
                }
            }
            if cuts.is_empty() {
                continue; // блок в своей полосе — резать нечего
            }
            let mut edges = vec![x0];
            edges.extend(cuts.iter().copied());
            edges.push(x1);
            let mut sub_quads: Vec<[Pt; 4]> = Vec::new();
            let mut strip_cols: Vec<usize> = Vec::new();
            for pair in edges.windows(2) {
                let (xa, xb) = (pair[0], pair[1]);
                if xb - xa < 4.0 {
                    continue; // щепка — нечего читать
                }
                sub_quads.push(clamp_quad_x(&blk.quad, xa, xb));
                strip_cols.push(band_of((xa + xb) / 2.0));
            }
            if sub_quads.is_empty() {
                continue;
            }
            let rec = engine.recognize_quads(img, &sub_quads);
            let mut parts: Vec<(usize, String)> = Vec::new();
            for (k, slot) in rec.iter().enumerate() {
                if let Some((t, c)) = slot {
                    let t = t.trim();
                    if t.is_empty() || *c < 0.5 {
                        if debug {
                            eprintln!("[table] кусок {} отброшен (conf {:.2}, текст {:?})", k, c, t);
                        }
                        continue;
                    }
                    parts.push((strip_cols[k], t.to_string()));
                }
            }
            if debug {
                eprintln!(
                    "[table] блок [{:.0}..{:.0}] «{}» режется в {:?} → {} частей",
                    x0, x1, blk.text.trim(), cuts, parts.len()
                );
            }
            if parts.len() >= 2 {
                texts[bi] = parts; // перечитанные куски вместо склейки
            }
        }
    }

    // 4) раскладка: строка × столбец (внутри строки — слева направо).
    //    Два блока в одной ячейке — склеиваем через пробел, а не затираем.
    let mut cells: Vec<(usize, usize, String)> = Vec::with_capacity(blocks.len());
    for (ri, row) in rows.iter().enumerate() {
        let mut placed: Vec<(usize, f32, String)> = Vec::new();
        for &block_i in row.iter() {
            let (cx, _, _, _) = block_geom(blocks[block_i]);
            for (col, text) in &texts[block_i] {
                placed.push((*col, cx, text.clone()));
            }
        }
        placed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (col, _, text) in placed {
            if let Some(last) = cells.last_mut() {
                if last.0 == ri && last.1 == col {
                    last.2.push(' ');
                    last.2.push_str(&text);
                    continue;
                }
            }
            cells.push((ri, col, text));
        }
    }
    Some(cells)
}

/// Подрезать X-координаты углов четырёхугольника в полосу [a, b]
/// (для разрезания склеенных ячеек по вертикальной границе).
fn clamp_quad_x(q: &[Pt; 4], a: f32, b: f32) -> [Pt; 4] {
    let clamp = |mut p: Pt| {
        p.x = p.x.clamp(a, b);
        p
    };
    [clamp(q[0]), clamp(q[1]), clamp(q[2]), clamp(q[3])]
}
