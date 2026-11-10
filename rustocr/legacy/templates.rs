//! Шаблоны глифов: рендер символов из встроенных TTF-шрифтов и извлечение
//! тех же признаков, что используются для распознаваемых глифов.

use ab_glyph::{Font, FontRef, PxScale, point};
use rayon::prelude::*;

use crate::features::{self, GlyphFeatures};

pub const FONT_FILES: [&[u8]; 8] = [
    include_bytes!("../fonts/DejaVuSans.ttf"),
    include_bytes!("../fonts/DejaVuSans-Bold.ttf"),
    include_bytes!("../fonts/DejaVuSerif.ttf"),
    include_bytes!("../fonts/DejaVuSerif-Bold.ttf"),
    include_bytes!("../fonts/LiberationSans-Regular.ttf"),
    include_bytes!("../fonts/LiberationSerif-Regular.ttf"),
    include_bytes!("../fonts/FreeSans.ttf"),
    include_bytes!("../fonts/FreeSerif.ttf"),
];

const RENDER_PX: f32 = 64.0;

/// x-высота шрифта в пикселях при рендере шаблонов (высота глифа «x»).
fn font_x_height(font: &FontRef) -> f64 {
    let gid = font.glyph_id('x');
    let glyph = gid.with_scale_and_position(PxScale::from(RENDER_PX), point(0.0, 0.0));
    if let Some(og) = font.outline_glyph(glyph) {
        let pb = og.px_bounds();
        if pb.height() > 1.0 {
            return pb.height() as f64;
        }
    }
    RENDER_PX as f64 * 0.5
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Lang {
    En,
    Ru,
    Both,
}

impl Lang {
    pub fn parse(s: &str) -> Lang {
        match s.to_ascii_lowercase().as_str() {
            "en" | "eng" | "english" => Lang::En,
            "ru" | "rus" | "russian" | "рус" => Lang::Ru,
            _ => Lang::Both,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Ru => "ru",
            Lang::Both => "both",
        }
    }
}

/// Алфавит для языка: печатаемый ASCII без пробела + кириллица + типографика.
pub fn charset(lang: Lang) -> Vec<char> {
    let mut out: Vec<char> = Vec::with_capacity(200);
    for c in 33u32..127 {
        out.push(char::from_u32(c).unwrap());
    }
    if lang != Lang::En {
        for c in 0x410u32..0x450 {
            out.push(char::from_u32(c).unwrap());
        }
        out.push('Ё');
        out.push('ё');
        out.extend(['«', '»', '—', '–', '…', '№', '“', '”', '‘', '’']);
    }
    if lang == Lang::Ru {
        // только кириллица: убираем латинские буквы, оставляем цифры/пунктуацию
        out.retain(|c| !c.is_ascii_alphabetic());
    }
    out.sort_unstable();
    out.dedup();
    out
}

pub struct TemplateEntry {
    pub ch: char,
    pub feats: Vec<(u8, u8, u8)>,
    pub zone: [f32; 16],
    pub aspect: f32,
    pub ink: f32,
    pub bmp: [u64; 32],
    pub bl_off: f32,
    pub rel_h: f32,
    pub n_feat: usize,
}

pub struct Templates {
    pub entries: Vec<TemplateEntry>,
}

/// Рендер одного символа: полутоновая битмапа + позиция базовой линии.
/// Начало координат — базовая линия глифа. Канва в точности накрывает
/// пиксельные границы контура (px_bounds).
fn render_char(font: &FontRef, ch: char) -> Option<(Vec<u8>, usize, usize, f64)> {
    let gid = font.glyph_id(ch);
    let glyph = gid.with_scale_and_position(PxScale::from(RENDER_PX), point(0.0, 0.0));
    let og = font.outline_glyph(glyph)?;
    let pb = og.px_bounds();
    if pb.width() < 0.5 || pb.height() < 0.5 {
        return None;
    }
    let w = ((pb.width().ceil() as usize)).max(1);
    let h = ((pb.height().ceil() as usize)).max(1);
    if w > 300 || h > 300 {
        return None;
    }
    // draw() выдаёт координаты 0..w x 0..h относительно px_bounds — канва совпадает с ними
    let mut buf = vec![255u8; w * h];
    og.draw(|x, y, cov| {
        if (x as usize) < w && (y as usize) < h {
            let ink = (cov * 255.0).round() as u32;
            let idx = y as usize * w + x as usize;
            let cur = 255 - buf[idx] as u32;
            buf[idx] = (255 - cur.max(ink).min(255)) as u8;
        }
    });
    // базовая линия (y=0 в системе глифа) в координатах канвы
    let baseline_y = -pb.min.y as f64;
    Some((buf, w, h, baseline_y))
}

impl Templates {
    /// Построение шаблонов (параллельно по символам каждого шрифта).
    pub fn build(lang: Lang) -> Templates {
        let chars = charset(lang);
        let mut entries: Vec<TemplateEntry> = Vec::with_capacity(chars.len() * FONT_FILES.len());
        for data in FONT_FILES.iter() {
            let font = match FontRef::try_from_slice(data) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let mut part: Vec<TemplateEntry> = chars
                .par_iter()
                .filter_map(|&ch| {
                    let (buf, w, h, baseline) = render_char(&font, ch)?;
                    let g: GlyphFeatures =
                        features::extract(&buf, w, h, None, baseline, font_x_height(&font))?;
                    let n_feat = g.feats.len();
                    Some(TemplateEntry {
                        ch,
                        feats: g.feats,
                        zone: g.zone,
                        aspect: g.aspect,
                        ink: g.ink,
                        bmp: g.bmp,
                        bl_off: g.bl_off,
                        rel_h: g.rel_h,
                        n_feat,
                    })
                })
                .collect();
            entries.append(&mut part);
        }
        Templates { entries }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_build_both() {
        let t = Templates::build(Lang::Both);
        assert!(t.entries.len() > 500, "шаблонов: {}", t.entries.len());
        let has_ru = t.entries.iter().any(|e| e.ch == 'Я');
        let has_en = t.entries.iter().any(|e| e.ch == 'Z');
        let has_digit = t.entries.iter().any(|e| e.ch == '7');
        assert!(has_ru && has_en && has_digit);
        // все шаблоны должны иметь признаки
        let bad: Vec<(char, usize, usize)> = t
            .entries
            .iter()
            .filter(|e| e.n_feat <= 10)
            .map(|e| (e.ch, e.n_feat, e.feats.len()))
            .collect();
        assert!(bad.is_empty(), "мало фич: {bad:?}");
        assert!(t.entries.iter().all(|e| e.n_feat > 10));
    }

    #[test]
    fn baseline_offset_descender() {
        // у «р» базовая линия внутри bbox (bl_off < 1), у «Р» — на нижней границе
        let font = FontRef::try_from_slice(FONT_FILES[0]).unwrap();
        let xh = font_x_height(&font);
        let (b, w, h, bl) = render_char(&font, 'p').unwrap();
        let g = features::extract(&b, w, h, None, bl, xh).unwrap();
        assert!(g.bl_off < 0.9, "p bl_off={}", g.bl_off);
        let (b, w, h, bl) = render_char(&font, 'P').unwrap();
        let g = features::extract(&b, w, h, None, bl, xh).unwrap();
        assert!(g.bl_off > 0.9, "P bl_off={}", g.bl_off);
        // высота строчной «x» ≈ 1.0 x-высоты, заглавной «X» — заметно больше
        let (b, w, h, bl) = render_char(&font, 'x').unwrap();
        let g = features::extract(&b, w, h, None, bl, xh).unwrap();
        assert!(g.rel_h > 0.85 && g.rel_h < 1.15, "x rel_h={}", g.rel_h);
        let (b, w, h, bl) = render_char(&font, 'X').unwrap();
        let g = features::extract(&b, w, h, None, bl, xh).unwrap();
        assert!(g.rel_h > 1.2, "X rel_h={}", g.rel_h);
    }
}
