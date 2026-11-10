//! Координатный спуск по весам классификатора на представительном наборе.
//! Запуск: cargo run --release --example tune

use image::DynamicImage;
use rustocr::classify::Weights;
use rustocr::ocr::{Engine, Options};
use rustocr::render::{add_noise, render_text};
use rustocr::templates::FONT_FILES;
use rustocr::Lang;

const EN_TEXT: &str = "The quick brown fox jumps over the lazy dog.
Pack my box with five dozen liquor jugs.
Invoice No. 4821 dated 07.11.2026, total: $1,249.50 (VAT 20%).";

const RU_TEXT: &str = "Съешь ещё этих мягких французских булок, да выпей же чаю.
Широкая электрификация южных губерний даст мощный толчок подъёму.
Счёт №37 от 07.11.2026 на сумму 12 450,50 руб. (НДС 20%).";

const CARLITO: &[u8] = include_bytes!("../examples_data/Carlito-Regular.ttf");

fn normalize_ws(s: &str) -> String {
    s.lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

fn accuracy(truth: &str, out: &str) -> f64 {
    let t = normalize_ws(truth);
    let o = normalize_ws(out);
    if t.is_empty() {
        return 1.0;
    }
    1.0 - (levenshtein(&t, &o) as f64 / t.chars().count() as f64)
}

fn main() {
    // представительнный набор: en + ru (dejavu), ru serif, шумный, unseen carlito
    let imgs: Vec<(&str, &str, DynamicImage)> = vec![
        (
            "en",
            EN_TEXT,
            DynamicImage::ImageLuma8(render_text(EN_TEXT, FONT_FILES[0], 34.0, 6.0)),
        ),
        (
            "ru",
            RU_TEXT,
            DynamicImage::ImageLuma8(render_text(RU_TEXT, FONT_FILES[0], 34.0, 6.0)),
        ),
        (
            "serif_ru",
            RU_TEXT,
            DynamicImage::ImageLuma8(render_text(RU_TEXT, FONT_FILES[2], 34.0, 6.0)),
        ),
        (
            "noisy_ru",
            RU_TEXT,
            DynamicImage::ImageLuma8(add_noise(
                &render_text(RU_TEXT, FONT_FILES[0], 34.0, 6.0),
                9.0,
                777,
            )),
        ),
        (
            "freeserif_en",
            EN_TEXT,
            DynamicImage::ImageLuma8(render_text(EN_TEXT, FONT_FILES[7], 34.0, 6.0)),
        ),
        (
            "carlito_ru",
            RU_TEXT,
            DynamicImage::ImageLuma8(render_text(RU_TEXT, CARLITO, 34.0, 6.0)),
        ),
    ];

    let mut engine = Engine::new(Lang::Both);
    let opts = Options::default();

    let mut eval = |w: &Weights| -> f64 {
        engine.weights = *w;
        let mut total = 0.0;
        for (_name, truth, img) in &imgs {
            let page = engine.recognize_image(img, &opts);
            let acc = accuracy(truth, &page.text());
            total += acc;
        }
        total / imgs.len() as f64
    };

    let mut w = Weights::default();
    let mut best = eval(&w);
    println!("старт: avg={:.4} при {:?}", best, w);

    // координатный спуск, 3 раунда
    let fields: [(&str, fn(&mut Weights, f32), Vec<f32>); 6] = [
        ("chamfer", |w, v| w.chamfer = v, vec![0.7, 1.0, 1.4, 1.8]),
        ("zone", |w, v| w.zone = v, vec![0.4, 0.8, 1.2]),
        ("aspect", |w, v| w.aspect = v, vec![0.4, 0.7, 1.0]),
        ("xor", |w, v| w.xor = v, vec![0.3, 0.55, 0.9]),
        ("bl", |w, v| w.bl = v, vec![0.5, 0.8, 1.1]),
        ("relh", |w, v| w.relh = v, vec![0.9, 1.3, 1.8, 2.3]),
    ];

    for round in 0..3 {
        let mut improved = false;
        for (name, setter, values) in &fields {
            for &v in values {
                let mut cand = w;
                setter(&mut cand, v);
                let score = eval(&cand);
                if score > best + 1e-6 {
                    best = score;
                    w = cand;
                    improved = true;
                    println!("раунд {}: {} = {:.2} -> avg={:.4}", round + 1, name, v, best);
                }
            }
        }
        if !improved {
            println!("раунд {}: улучшений нет, стоп", round + 1);
            break;
        }
    }
    println!("\nитоговые веса: {:?}", w);
    println!("avg точность на тюнинг-наборе: {:.4}", best);
    // финальная детализация
    engine.weights = w;
    for (name, truth, img) in &imgs {
        let page = engine.recognize_image(img, &opts);
        println!("  {:<10} {:.2}%", name, accuracy(truth, &page.text()) * 100.0);
    }
}
