//! Рендер текста в изображение — для бенчмарков, тестов и демо.

use ab_glyph::{Font, FontRef, PxScale, ScaleFont, point};
use image::{GrayImage, Luma};

/// Рендер многострочного текста заданным шрифтом.
/// `px` — размер шрифта, `line_gap` — дополнительный межстрочный интервал в px.
pub fn render_text(text: &str, font_bytes: &[u8], px: f32, line_gap: f32) -> GrayImage {
    let font = FontRef::try_from_slice(font_bytes).expect("шрифт не читается");
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return GrayImage::from_pixel(10, 10, Luma([255]));
    }
    let lh = (px * 1.42 + line_gap).max(px + 3.0);

    // ширина канвы по максимальной строке
    let scaled = font.as_scaled(px);
    let mut max_w = 0f32;
    for line in &lines {
        let mut x = 0f32;
        for ch in line.chars() {
            x += scaled.h_advance(font.glyph_id(ch));
        }
        max_w = max_w.max(x);
    }
    let w = (max_w + 44.0).ceil() as u32;
    let h = ((lines.len() as f32) * lh + px * 2.0).ceil() as u32 + 8;
    let mut img = GrayImage::from_pixel(w.max(16), h.max(16), Luma([255u8]));

    let mut baseline = px * 1.55;
    for line in &lines {
        let mut x = 22.0f32;
        for ch in line.chars() {
            let gid = font.glyph_id(ch);
            let adv = scaled.h_advance(gid);
            if ch != ' ' {
                let glyph = gid.with_scale_and_position(PxScale::from(px), point(x, baseline));
                if let Some(og) = font.outline_glyph(glyph) {
                    // draw() выдаёт координаты относительно px_bounds глифа —
                    // сдвигаем на целочисленный офсет его левого верхнего угла
                    let pb = og.px_bounds();
                    let ox = pb.min.x.floor() as i32;
                    let oy = pb.min.y.floor() as i32;
                    og.draw(|dx, dy, cov| {
                        let gx = ox + dx as i32;
                        let gy = oy + dy as i32;
                        if gx >= 0 && gy >= 0 && (gx as u32) < w && (gy as u32) < h {
                            let ink = (cov * 255.0).round() as u32;
                            let cur = 255 - img.get_pixel(gx as u32, gy as u32).0[0] as u32;
                            let _ = img.put_pixel(
                                gx as u32,
                                gy as u32,
                                Luma([(255 - cur.max(ink).min(255)) as u8]),
                            );
                        }
                    });
                }
            }
            x += adv;
        }
        baseline += lh;
    }
    img
}

/// Добавление равномерного шума (xorshift-генератор, детерминированный).
pub fn add_noise(img: &GrayImage, sigma: f32, seed: u64) -> GrayImage {
    let mut s = seed | 1u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let u = (next() % 4096) as f32 / 4096.0 - 0.5; // ~U(-0.5, 0.5)
        let v = p.0[0] as f32 + u * sigma * 3.46; // ~N(0, sigma) приближённо
        p.0[0] = v.clamp(0.0, 255.0) as u8;
    }
    out
}
