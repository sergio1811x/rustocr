//! Бенчмарк: генерация тестовых изображений (разные шрифты/шум/перекос/размер)
//! и замер посимвольной точности + скорости.
//!
//! Запуск: cargo run --release --example bench

use std::time::Instant;

use image::{DynamicImage, GrayImage};
use rustocr::ocr::{Engine, Options};
use rustocr::render::{add_noise, render_text};
use rustocr::templates::FONT_FILES;
use rustocr::{gray, Lang};

const EN_TEXT: &str = "The quick brown fox jumps over the lazy dog.
Pack my box with five dozen liquor jugs.
Sphinx of black quartz, judge my vow.
Invoice No. 4821 dated 07.11.2026, total: $1,249.50 (VAT 20%).
Heavy boxes block the machines output vent today.";

const RU_TEXT: &str = "Съешь ещё этих мягких французских булок, да выпей же чаю.
Широкая электрификация южных губерний даст мощный толчок подъёму сельского хозяйства.
Ёлка, йогурт, подъезд, объявление и прочие трудные буквы.
Счёт №37 от 07.11.2026 на сумму 12 450,50 руб. (НДС 20%).
Флаги, книжки, чашки — всё в шкафу у Маши.";

const CARLITO: &[u8] = include_bytes!("../examples_data/Carlito-Regular.ttf");
const DEJAVU_MONO: &[u8] = include_bytes!("../examples_data/DejaVuSansMono.ttf");

struct Case {
    name: &'static str,
    truth: &'static str,
    img: GrayImage,
    lang: Lang,
    deskew: bool,
}

fn to_dyn(img: GrayImage) -> DynamicImage {
    DynamicImage::ImageLuma8(img)
}

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
    let d = levenshtein(&t, &o);
    1.0 - (d as f64 / t.chars().count() as f64)
}

fn rotate_img(img: &GrayImage, deg: f64) -> GrayImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let (r, nw, nh) = gray::rotate_bilinear(&img.clone().into_raw(), w, h, deg.to_radians());
    GrayImage::from_raw(nw as u32, nh as u32, r).expect("rotate")
}

fn main() {
    let out_dir = std::path::Path::new("examples/out");
    let _ = std::fs::create_dir_all(out_dir);

    // паддинги-строки для честного сравнения с normalize_ws
    let mut cases: Vec<Case> = vec![
        Case {
            name: "dejavu_sans_en_clean",
            truth: EN_TEXT,
            img: render_text(EN_TEXT, FONT_FILES[0], 34.0, 6.0),
            lang: Lang::Both,
            deskew: true,
        },
        Case {
            name: "dejavu_sans_ru_clean",
            truth: RU_TEXT,
            img: render_text(RU_TEXT, FONT_FILES[0], 34.0, 6.0),
            lang: Lang::Both,
            deskew: true,
        },
        Case {
            name: "dejavu_serif_ru_clean",
            truth: RU_TEXT,
            img: render_text(RU_TEXT, FONT_FILES[2], 34.0, 6.0),
            lang: Lang::Both,
            deskew: true,
        },
        Case {
            name: "dejavu_sansbold_en",
            truth: EN_TEXT,
            img: render_text(EN_TEXT, FONT_FILES[1], 34.0, 6.0),
            lang: Lang::Both,
            deskew: true,
        },
        Case {
            name: "liberation_sans_en",
            truth: EN_TEXT,
            img: render_text(EN_TEXT, FONT_FILES[4], 34.0, 6.0),
            lang: Lang::Both,
            deskew: true,
        },
        Case {
            name: "freesans_ru_SEEN",
            truth: RU_TEXT,
            img: render_text(RU_TEXT, FONT_FILES[6], 34.0, 6.0),
            lang: Lang::Both,
            deskew: true,
        },
        Case {
            name: "freeserif_en_SEEN",
            truth: EN_TEXT,
            img: render_text(EN_TEXT, FONT_FILES[7], 34.0, 6.0),
            lang: Lang::Both,
            deskew: true,
        },
        Case {
            name: "carlito_ru_UNSEEN",
            truth: RU_TEXT,
            img: render_text(RU_TEXT, CARLITO, 34.0, 6.0),
            lang: Lang::Both,
            deskew: true,
        },
        Case {
            name: "carlito_en_UNSEEN",
            truth: EN_TEXT,
            img: render_text(EN_TEXT, CARLITO, 34.0, 6.0),
            lang: Lang::Both,
            deskew: true,
        },
        Case {
            name: "dejavu_mono_ru_UNSEEN",
            truth: RU_TEXT,
            img: render_text(RU_TEXT, DEJAVU_MONO, 32.0, 6.0),
            lang: Lang::Both,
            deskew: true,
        },
        Case {
            name: "dejavu_ru_noisy_jpeg",
            truth: RU_TEXT,
            img: {
                let base = render_text(RU_TEXT, FONT_FILES[0], 34.0, 6.0);
                add_noise(&base, 9.0, 0xDEADBEE)
            },
            lang: Lang::Both,
            deskew: true,
        },
        Case {
            name: "dejavu_en_skew1.5",
            truth: EN_TEXT,
            img: rotate_img(&render_text(EN_TEXT, FONT_FILES[0], 34.0, 6.0), 1.5),
            lang: Lang::Both,
            deskew: true,
        },
        Case {
            name: "dejavu_ru_small13px",
            truth: RU_TEXT,
            img: render_text(RU_TEXT, FONT_FILES[0], 13.0, 3.0),
            lang: Lang::Both,
            deskew: true,
        },
        Case {
            name: "dejavu_en_big56px",
            truth: EN_TEXT,
            img: render_text(EN_TEXT, FONT_FILES[0], 56.0, 10.0),
            lang: Lang::Both,
            deskew: true,
        },
    ];

    // для jpeg-кейса: сохранить и перечитать (артефакты сжатия)
    for case in cases.iter_mut() {
        if case.name.contains("jpeg") {
            let p = out_dir.join(format!("{}.jpg", case.name));
            to_dyn(case.img.clone()).save(&p).expect("save jpg");
            case.img = image::open(&p).expect("reload").to_luma8();
        }
    }

    let t_build = Instant::now();
    let engine = Engine::new(Lang::Both);
    eprintln!("построение шаблонов: {:.0} мс", t_build.elapsed().as_secs_f64() * 1000.0);

    println!(
        "{:<28} {:>7} {:>9} {:>7} {:>9} {:>8}",
        "кейс", "мс", "точность", "глифов", "ср.conf", "перекос"
    );
    println!("{}", "-".repeat(74));
    let mut worst: Vec<(String, String)> = vec![];
    for case in &cases {
        let img = to_dyn(case.img.clone());
        let page = engine.recognize_image(&img, &Options { deskew: case.deskew });
        let acc = accuracy(case.truth, &page.text());
        println!(
            "{:<28} {:>7.0} {:>8.2}% {:>7} {:>9.3} {:>7.2}",
            case.name,
            page.elapsed_ms,
            acc * 100.0,
            page.num_glyphs,
            page.mean_conf,
            page.skew_deg
        );
        // сохранить вход и результат для ручной проверки
        let _ = to_dyn(case.img.clone()).save(out_dir.join(format!("{}.png", case.name)));
        let _ = std::fs::write(
            out_dir.join(format!("{}.txt", case.name)),
            page.text(),
        );
        // диагностика: худшие кейсы — первые расхождения
        if acc < 0.97 {
            let t = normalize_ws(case.truth);
            let o = normalize_ws(&page.text());
            let (ta, ob) = (t.chars().collect::<Vec<_>>(), o.chars().collect::<Vec<_>>());
            let mut msg = String::new();
            let mut diffs = 0;
            let n = ta.len().min(ob.len());
            for i in 0..n {
                if ta[i] != ob[i] && diffs < 12 {
                    let ctx: String = ta[i.saturating_sub(8)..(i + 8).min(n)].iter().collect();
                    msg.push_str(&format!(
                        "  {}->{} в «…{}…»\n",
                        ta[i], ob[i], ctx
                    ));
                    diffs += 1;
                }
            }
            if ob.len() != ta.len() {
                msg.push_str(&format!("  длины: {} vs {}\n", ta.len(), ob.len()));
            }
            worst.push((case.name.to_string(), msg));
        }
    }
    if !worst.is_empty() {
        println!("\nРасхождения:");
        for (name, msg) in worst {
            println!("[{name}]\n{msg}");
        }
    }
}
