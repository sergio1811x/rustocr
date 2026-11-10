//! Разбиение блобов на текстовые строки, слияние диакритики (точки «ё»,
//! бревис «й»), фильтрация мусора и крупных графических объектов.

use crate::cc::Blob;

pub fn median(mut v: Vec<i32>) -> i32 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

#[derive(Clone, Debug)]
pub struct Line {
    pub blobs: Vec<Blob>,
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Line {
    fn from_blob(b: &Blob) -> Line {
        Line {
            blobs: vec![*b],
            x0: b.x0,
            y0: b.y0,
            x1: b.x1,
            y1: b.y1,
        }
    }
    fn add(&mut self, b: &Blob) {
        self.x0 = self.x0.min(b.x0);
        self.y0 = self.y0.min(b.y0);
        self.x1 = self.x1.max(b.x1);
        self.y1 = self.y1.max(b.y1);
        self.blobs.push(*b);
    }
    pub fn h(&self) -> i32 {
        self.y1 - self.y0 + 1
    }
    pub fn w(&self) -> i32 {
        self.x1 - self.x0 + 1
    }
}

/// Робастная оценка медианной высоты символов по блобам.
pub fn median_char_height(blobs: &[Blob]) -> i32 {
    let mut hs: Vec<i32> = blobs
        .iter()
        .filter(|b| b.h() >= 4 && b.w() >= 2 && b.h() <= 300 && b.w() <= 300)
        .map(|b| b.h())
        .collect();
    if hs.is_empty() {
        return 0;
    }
    let m = median(hs.clone());
    // второй проход: только «ядро» распределения
    hs.retain(|h| (*h as f64) > m as f64 * 0.5 && (*h as f64) < m as f64 * 1.8);
    if hs.is_empty() {
        m
    } else {
        median(hs)
    }
}

/// Фильтрация блобов: шум, линейки, крупные объекты.
pub fn filter_blobs(blobs: Vec<Blob>, med_h: i32) -> Vec<Blob> {
    if med_h <= 0 {
        return blobs;
    }
    blobs
        .into_iter()
        .filter(|b| {
            let h = b.h();
            let w = b.w();
            if b.area < 4 || h < 4 || w < 2 {
                return false;
            }
            // крупные графические объекты (картинки, рамки)
            if h > 6 * med_h && w > 6 * med_h {
                return false;
            }
            // горизонтальные линейки таблиц
            if h <= 4 && w > 6 * med_h {
                return false;
            }
            // вертикальные линейки
            if w <= 3 && h > 4 * med_h {
                return false;
            }
            // сверхширокие «линии»
            if w > 25 * med_h {
                return false;
            }
            true
        })
        .collect()
}

/// Группировка блобов в строки: блоб присоединяется к текущей строке,
/// если его вертикальный перекрытие с ней ≥ 35% высоты блоба.
/// Затем мелкие строки-диакритики вливаются в соседнюю основную строку.
pub fn build_lines(mut blobs: Vec<Blob>, med_h: i32) -> Vec<Line> {
    blobs.sort_by_key(|b| (b.y0, b.x0));
    let mut lines: Vec<Line> = Vec::new();
    for b in &blobs {
        let mut assigned = false;
        if let Some(cur) = lines.last_mut() {
            let ov_top = b.y0.max(cur.y0);
            let ov_bot = b.y1.min(cur.y1);
            let overlap = ov_bot - ov_top;
            let need = ((b.h() as f64) * 0.35) as i32;
            if overlap >= need {
                cur.add(b);
                assigned = true;
            }
        }
        if !assigned {
            lines.push(Line::from_blob(b));
        }
    }

    // Слияние диакритики: маленькие строки рядом с большими
    if med_h > 0 {
        let mut merged: Vec<Line> = Vec::new();
        let mut i = 0usize;
        while i < lines.len() {
            let cur = lines[i].clone();
            let is_tiny = cur.h() < (med_h as f64 * 0.55) as i32 && cur.blobs.len() <= 4;
            if is_tiny {
                // попробуем влить в следующую строку (акценты СВЕРХУ)
                let mut done = false;
                if i + 1 < lines.len() {
                    let next = &mut lines[i + 1];
                    let gap = next.y0 - cur.y1;
                    let x_ov = cur.x1.max(next.x0) - cur.x0.min(next.x1); // >=0 при перекрытии
                    let overlap_x = (cur.x1.min(next.x1)) - (cur.x0.max(next.x0));
                    let _ = x_ov;
                    if gap > -(med_h / 2)
                        && gap < med_h + med_h / 2
                        && overlap_x > -(cur.w() / 2)
                    {
                        for b in &cur.blobs {
                            next.add(b);
                        }
                        done = true;
                    }
                }
                // или в предыдущую (акценты/запятые снизу уже в строке — редко)
                if !done && !merged.is_empty() {
                    let prev = merged.last_mut().unwrap();
                    let gap = cur.y0 - prev.y1;
                    let overlap_x = (cur.x1.min(prev.x1)) - (cur.x0.max(prev.x0));
                    if gap > -(med_h / 2)
                        && gap < med_h / 2
                        && overlap_x > -(cur.w() / 2)
                    {
                        for b in &cur.blobs {
                            prev.add(b);
                        }
                        done = true;
                    }
                }
                if done {
                    i += 1;
                    continue;
                }
            }
            merged.push(cur);
            i += 1;
        }
        lines = merged;
    }

    lines.sort_by_key(|l| (l.y0, l.x0));
    lines
}

/// Медиана нижних границ «ядровых» блобов строки — оценка базовой линии.
pub fn line_baseline(line: &Line, med_h: i32) -> i32 {
    let mut bottoms: Vec<i32> = line
        .blobs
        .iter()
        .filter(|b| b.h() >= (med_h as f64 * 0.45) as i32)
        .map(|b| b.y1)
        .collect();
    if bottoms.is_empty() {
        bottoms = line.blobs.iter().map(|b| b.y1).collect();
    }
    median(bottoms)
}
