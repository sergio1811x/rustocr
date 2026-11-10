//! Декодирование картинок с лимитами: маленький PNG может «распаковаться»
//! в гигапиксельное изображение и съесть всю память (декомпрессионная
//! бомба). Лимиты отсекают такие файлы до выделения буфера.

use image::ImageReader;
use std::io::{BufReader, Cursor};
use std::path::Path;

const MAX_W: u32 = 20_000; // пикселей по стороне
const MAX_H: u32 = 20_000;
const MAX_ALLOC: u64 = 512 * 1024 * 1024; // 512 МБ на декод

fn limits() -> image::Limits {
    let mut l = image::Limits::default();
    l.max_image_width = Some(MAX_W);
    l.max_image_height = Some(MAX_H);
    l.max_alloc = Some(MAX_ALLOC);
    l
}

/// Декодировать картинку из байтов запроса (HTTP API).
pub fn decode_bytes(bytes: &[u8]) -> anyhow::Result<image::DynamicImage> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| anyhow::anyhow!("не определить формат картинки: {}", e))?;
    reader.limits(limits());
    reader
        .decode()
        .map_err(|e| anyhow::anyhow!("не удалось декодировать картинку: {}", e))
}

/// Открыть картинку с диска (CLI).
pub fn open_file(path: &Path) -> anyhow::Result<image::DynamicImage> {
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("не открыть {}: {}", path.display(), e))?;
    let mut reader = ImageReader::new(BufReader::new(file))
        .with_guessed_format()
        .map_err(|e| anyhow::anyhow!("не определить формат {}: {}", path.display(), e))?;
    reader.limits(limits());
    reader
        .decode()
        .map_err(|e| anyhow::anyhow!("не удалось декодировать {}: {}", path.display(), e))
}

/// Выцветшая страница: чернил темнее 120 нет вообще, но есть «светлые
/// чернила» (пиксели < 220) — печать выцвела до серого. У нормальной
/// страницы темнее 120 — 4–8% пикселей, у разреженной (подпись/штамп на
/// пустом листе) светлее 220 почти нет. Выравнивание фона превращает
/// бледный текст в обычный: на выцветшей копии сметы 254 числа → 143
/// без выравнивания и 252 с ним.
/// env RUSTOCR_FADE_FLATTEN: 0 — выкл, 1 — всегда, иначе авто по порогам.
pub fn flatten_if_faded(img: &mut image::RgbImage) {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let mode = *MODE.get_or_init(|| match std::env::var("RUSTOCR_FADE_FLATTEN").as_deref() {
        Ok("0") => 0,
        Ok("1") => 1,
        _ => 2, // авто
    });
    if mode == 0 {
        return;
    }
    if mode == 2 {
        // доли пикселей темнее 120 и темнее 220 по гистограмме яркости
        let raw = img.as_raw();
        let stride = (raw.len() / 400_000 / 3).max(1) * 3; // ≤ ~400 тыс. сэмплов
        let mut cum120 = 0u32;
        let mut cum220 = 0u32;
        let mut n = 0u32;
        let mut i = 0usize;
        while i + 2 < raw.len() {
            let g = (raw[i] as u32 * 30 + raw[i + 1] as u32 * 59 + raw[i + 2] as u32 * 11) / 100;
            if g < 120 {
                cum120 += 1;
            }
            if g < 220 {
                cum220 += 1;
            }
            n += 1;
            i += stride;
        }
        if n == 0 {
            return;
        }
        let f120 = cum120 as f32 / n as f32;
        let f220 = cum220 as f32 / n as f32;
        // нет тёмных чернил, но светлые есть — выцветший скан
        if !(f120 < 0.002 && f220 > 0.01) {
            return;
        }
    }
    crate::det::flatten_background(img);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_png() {
        let img = image::DynamicImage::new_rgb8(3, 2);
        let mut buf = Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let decoded = decode_bytes(buf.get_ref()).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (3, 2));
    }
}
