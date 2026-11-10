//! Классификация глифа по шаблонам: Chamfer по направлениям контуров +
//! зонные плотности + аспект + битовая XOR-маска + положение базовой линии.

use crate::features::{bmp_xor, GlyphFeatures, GRID};
use crate::templates::{TemplateEntry, Templates};

#[derive(Clone, Copy, Debug)]
pub struct Weights {
    pub chamfer: f32,
    pub zone: f32,
    pub aspect: f32,
    pub xor: f32,
    pub bl: f32,
    pub relh: f32,
}

impl Default for Weights {
    fn default() -> Self {
        // подобрано координатным спуском (examples/tune.rs)
        Weights {
            chamfer: 0.7,
            zone: 0.4,
            aspect: 0.4,
            xor: 0.9,
            bl: 0.8,
            relh: 2.3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClassResult {
    pub ch: char,
    pub score: f32,
    pub conf: f32,
    /// вторая (конкурирующая) гипотеза
    pub alt: Option<(char, f32)>,
}

/// Префильтр кандидатов по аспекту/плотности/базовой линии/высоте.
#[inline]
fn passes_prefilter(g: &GlyphFeatures, t: &TemplateEntry) -> bool {
    let ar = (g.aspect / t.aspect).ln().abs();
    if ar > 0.42 {
        return false;
    }
    if (g.ink - t.ink).abs() > 0.30 {
        return false;
    }
    if (g.bl_off - t.bl_off).abs() > 0.36 {
        return false;
    }
    let rh = (g.rel_h / t.rel_h).ln().abs();
    if rh > 0.55 {
        return false;
    }
    true
}

/// Полный скор одного шаблона. `limit` — при превышении этого счёта
/// вычисление прерывается (все прочие слагаемые неотрицательны).
#[inline]
fn score_template(g: &GlyphFeatures, t: &TemplateEntry, w: &Weights, limit: f32) -> f32 {
    // Chamfer: признаки шаблона ищутся в DT-картах глифа (того же направления)
    let wc = w.chamfer;
    let inv_n = 1.0 / (t.n_feat as f32).max(1.0);
    let mut cost = 0f32;
    let feats = &t.feats;
    for (i, &(x, y, d)) in feats.iter().enumerate() {
        let dt = &g.dts[d as usize];
        cost += dt[y as usize * GRID + x as usize];
        // ранний выход: даже без остальных слагаемых счёт уже хуже предела
        if (i & 31) == 31 && wc * cost * inv_n > limit {
            return f32::MAX;
        }
    }
    let chamfer = cost * inv_n;

    // зоны: L1
    let mut zone = 0f32;
    for i in 0..16 {
        zone += (g.zone[i] - t.zone[i]).abs();
    }

    let ar = (g.aspect / t.aspect).ln().abs();
    let xor = bmp_xor(&g.bmp, &t.bmp);
    let bl = (g.bl_off - t.bl_off).abs();
    let rh = (g.rel_h / t.rel_h).ln().abs();

    let s = w.chamfer * chamfer + w.zone * zone + w.aspect * ar + w.xor * xor + w.bl * bl + w.relh * rh;
    if s > limit {
        return f32::MAX;
    }
    s
}

/// Классификация глифа: лучший символ, скор, уверенность (отрыв от конкурента).
pub fn classify(g: &GlyphFeatures, tpl: &Templates, w: &Weights) -> Option<ClassResult> {
    let mut best_char = '\0';
    let mut best_score = f32::MAX;
    let mut second_score = f32::MAX; // лучший скор среди ДРУГИХ символов
    let mut second_char: Option<char> = None;

    for t in &tpl.entries {
        if !passes_prefilter(g, t) {
            continue;
        }
        // лимит = текущий второй счёт: всё, что хуже него, не влияет ни на
        // лучший, ни на второй результат — можно прерывать без потерь
        let s = score_template(g, t, w, second_score);
        if s == f32::MAX {
            continue;
        }
        if s < best_score {
            if t.ch != best_char {
                second_score = best_score;
                second_char = if best_char == '\0' { None } else { Some(best_char) };
            }
            best_score = s;
            best_char = t.ch;
        } else if s < second_score && t.ch != best_char {
            second_score = s;
            second_char = Some(t.ch);
        }
    }

    if best_char == '\0' {
        return None;
    }
    let conf = match second_char {
        Some(_) => ((second_score - best_score) / second_score.max(1e-6)).clamp(0.0, 1.0),
        // единственный кандидат: уверенность неполная — конкурентов просто
        // не было в префильтре
        None => 0.75,
    };
    Some(ClassResult {
        ch: best_char,
        score: best_score,
        conf,
        alt: second_char.map(|c| (c, second_score)),
    })
}
