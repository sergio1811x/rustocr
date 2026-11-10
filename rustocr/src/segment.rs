//! Сегментация строки на глифы по вертикальному проекционному профилю,
//! разрезание слипшихся символов, вставка пробелов по межсимвольным зазорам.

use crate::layout::{median, Line};

#[derive(Clone, Copy, Debug)]
pub struct Seg {
    /// x1/y1 включительно (координаты страницы)
    pub x0: i32,
    pub x1: i32,
    pub y0: i32,
    pub y1: i32,
}

impl Seg {
    pub fn w(&self) -> i32 {
        self.x1 - self.x0 + 1
    }
    pub fn h(&self) -> i32 {
        self.y1 - self.y0 + 1
    }
    pub fn gap_to(&self, next: &Seg) -> i32 {
        next.x0 - self.x1 - 1
    }
}

/// Занятость колонок чернилами в пределах bbox строки.
fn column_occupancy(bin: &[bool], w: usize, h: usize, line: &Line) -> (usize, Vec<u32>) {
    let y0 = (line.y0.max(0) as usize).min(h);
    let y1 = (((line.y1 + 1) as usize).min(h)).max(y0);
    let x0 = (line.x0.max(0) as usize).min(w);
    let x1 = (((line.x1 + 1) as usize).min(w)).max(x0);
    let mut occ = vec![0u32; x1 - x0];
    for y in y0..y1 {
        let row = y * w;
        for x in x0..x1 {
            if bin[row + x] {
                occ[x - x0] += 1;
            }
        }
    }
    (x0, occ)
}

/// Проходы занятых колонок -> сегменты с tight-bbox по вертикали.
fn runs_to_segs(bin: &[bool], w: usize, occ: &[u32], x0: usize, y_lo: usize, y_hi: usize) -> Vec<Seg> {
    let mut segs = Vec::new();
    let mut x = 0usize;
    while x < occ.len() {
        if occ[x] == 0 {
            x += 1;
            continue;
        }
        let start = x;
        while x < occ.len() && occ[x] > 0 {
            x += 1;
        }
        let end = x; // не включая
        let mut ty0 = i32::MAX;
        let mut ty1 = i32::MIN;
        for yy in y_lo..y_hi {
            let row = yy * w;
            for xx in (x0 + start)..(x0 + end) {
                if bin[row + xx] {
                    if (yy as i32) < ty0 {
                        ty0 = yy as i32;
                    }
                    if (yy as i32) > ty1 {
                        ty1 = yy as i32;
                    }
                    break;
                }
            }
        }
        if ty1 < ty0 {
            continue;
        }
        segs.push(Seg {
            x0: (x0 + start) as i32,
            x1: (x0 + end - 1) as i32,
            y0: ty0,
            y1: ty1,
        });
    }
    segs
}

/// Первый проход: сегменты без разрезания (для оценки медианной ширины глифа).
pub fn segment_raw(bin: &[bool], w: usize, h: usize, line: &Line) -> Vec<Seg> {
    let (x0, occ) = column_occupancy(bin, w, h, line);
    let y_lo = (line.y0.max(0) as usize).min(h);
    let y_hi = (((line.y1 + 1) as usize).min(h)).max(y_lo);
    runs_to_segs(bin, w, &occ, x0, y_lo, y_hi)
}

/// Полная сегментация: группы сегментов. Группа длиной > 1 — результат
/// разрезания слипшихся символов (классифицируется и порознь, и слитно).
pub fn segment_groups(bin: &[bool], w: usize, h: usize, line: &Line, med_adv: i32) -> Vec<Vec<Seg>> {
    let (x0, occ) = column_occupancy(bin, w, h, line);
    let y_lo = (line.y0.max(0) as usize).min(h);
    let y_hi = (((line.y1 + 1) as usize).min(h)).max(y_lo);
    let base = runs_to_segs(bin, w, &occ, x0, y_lo, y_hi);
    if med_adv <= 0 {
        return base.into_iter().map(|s| vec![s]).collect();
    }
    base.into_iter()
        .map(|seg| {
            let a0 = (seg.x0 as usize) - x0;
            let len = seg.w() as usize;
            let len = len.min(occ.len() - a0.min(occ.len()));
            let a0 = a0.min(occ.len());
            split_recursive(bin, w, &occ[a0..a0 + len], x0 + a0, y_lo, y_hi, med_adv, 0)
        })
        .collect()
}

/// Рекурсивное разрезание широкого сегмента по «долине» профиля.
fn split_recursive(
    bin: &[bool],
    w: usize,
    occ: &[u32],
    off: usize,
    y_lo: usize,
    y_hi: usize,
    med_adv: i32,
    depth: u32,
) -> Vec<Seg> {
    let width = occ.len();
    let max_occ = occ.iter().copied().max().unwrap_or(0);
    if (width as f64) < med_adv as f64 * 1.45 || depth >= 3 || max_occ == 0 {
        return vec![single_seg(bin, w, occ, off, y_lo, y_hi)];
    }
    // долина в центральных 76% сегмента
    let lo = ((width as f64) * 0.12).ceil() as usize;
    let hi = (((width as f64) * 0.88).floor() as usize).min(width - 1) + 1;
    let thr = (max_occ as f64 * 0.14) as u32;
    let mut best_x = usize::MAX;
    let mut best_v = u32::MAX;
    for x in lo..hi {
        let v = occ[x];
        if v < best_v && v <= thr {
            best_v = v;
            best_x = x;
        }
    }
    let min_half = ((med_adv as f64) * 0.5).max(3.0) as usize;
    if best_x == usize::MAX || best_x < min_half || width - best_x < min_half {
        return vec![single_seg(bin, w, occ, off, y_lo, y_hi)];
    }
    let (l, r) = occ.split_at(best_x);
    let mut out = Vec::new();
    out.extend(split_recursive(bin, w, l, off, y_lo, y_hi, med_adv, depth + 1));
    out.extend(split_recursive(bin, w, r, off + best_x, y_lo, y_hi, med_adv, depth + 1));
    out
}

/// Один сегмент из куска профиля (без разрезания).
fn single_seg(bin: &[bool], w: usize, occ: &[u32], off: usize, y_lo: usize, y_hi: usize) -> Seg {
    let mut ty0 = i32::MAX;
    let mut ty1 = i32::MIN;
    let xs = off;
    let xe = off + occ.len();
    for yy in y_lo..y_hi {
        let row = yy * w;
        for xx in xs..xe {
            if bin[row + xx] {
                if (yy as i32) < ty0 {
                    ty0 = yy as i32;
                }
                if (yy as i32) > ty1 {
                    ty1 = yy as i32;
                }
                break;
            }
        }
    }
    if ty1 < ty0 {
        ty0 = y_lo as i32;
        ty1 = y_lo as i32;
    }
    Seg {
        x0: xs as i32,
        x1: (xe - 1) as i32,
        y0: ty0,
        y1: ty1,
    }
}

/// Медианная ширина сегментов по всем строкам (оценка «шага» символа).
pub fn median_advance(all_widths: &mut Vec<i32>) -> i32 {
    if all_widths.is_empty() {
        return 0;
    }
    let m = median(all_widths.clone());
    all_widths.retain(|w| (*w as f64) >= m as f64 * 0.4 && (*w as f64) <= m as f64 * 2.5);
    if all_widths.is_empty() {
        m
    } else {
        median(all_widths.clone())
    }
}

/// Число пробелов между единицами распознавания по массиву зазоров.
/// Двухэтапно: порог «межсловного» зазора, затем квантование по медиане
/// реальных межсловных зазоров.
pub fn space_counts_from_gaps(gaps: &[i32], med_adv: i32) -> Vec<usize> {
    if gaps.is_empty() {
        return Vec::new();
    }
    let intra: Vec<i32> = gaps
        .iter()
        .copied()
        .filter(|g| *g > 0 && (*g as f64) < med_adv as f64 * 0.5)
        .collect();
    let med_gap = if intra.is_empty() {
        (med_adv as f64 * 0.18) as i32
    } else {
        median(intra)
    };
    let thr = ((med_gap as f64) * 2.4).max(4.0) as i32;
    let wordgaps: Vec<i32> = gaps.iter().copied().filter(|g| *g > thr).collect();
    let unit = if wordgaps.is_empty() {
        (thr as f64 * 1.2).max(1.0)
    } else {
        median(wordgaps) as f64
    };
    if std::env::var("RUSTOCR_DEBUG").is_ok() {
        eprintln!(
            "[space] med_adv={} med_gap={} thr={} unit={} gaps={:?}",
            med_adv, med_gap, thr, unit, gaps
        );
    }
    gaps.iter()
        .map(|g| {
            if *g > thr {
                let ns = ((*g as f64) / unit).round() as i32;
                ns.clamp(1, 3) as usize
            } else {
                0
            }
        })
        .collect()
}
