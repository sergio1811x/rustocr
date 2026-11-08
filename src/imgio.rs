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
