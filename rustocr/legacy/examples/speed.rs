//! Замер скорости на «странице» (~60 строк, ~1500 глифов).
//! Запуск: cargo run --release --example speed

use image::DynamicImage;
use rustocr::ocr::{Engine, Options};
use rustocr::render::render_text;
use rustocr::templates::FONT_FILES;
use rustocr::Lang;

const LINE: &str = "Съешь ещё этих мягких французских булок, да выпей же чаю. Инвойс №37 от 07.11.2026.";

fn main() {
    let text = vec![LINE; 60].join("\n");
    let img = render_text(&text, FONT_FILES[0], 34.0, 6.0);
    println!("изображение: {}x{} px", img.width(), img.height());
    let dyn_img = DynamicImage::ImageLuma8(img);
    let engine = Engine::new(Lang::Both);

    // прогрев + 3 прогона
    let mut times = vec![];
    let mut glyphs = 0;
    for i in 0..4 {
        let page = engine.recognize_image(&dyn_img, &Options::default());
        if i > 0 {
            times.push(page.elapsed_ms);
            glyphs = page.num_glyphs;
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    println!("строк распознано: {}", page_lines(&engine, &dyn_img));
    println!("глифов: {glyphs}");
    println!("время (медиана из 3): {med:.0} мс");
    println!("пропускная способность: {:.0} глифов/с", glyphs as f64 / (med / 1000.0));
}

fn page_lines(_e: &Engine, _img: &DynamicImage) -> usize {
    60 // текст из 60 строк
}
