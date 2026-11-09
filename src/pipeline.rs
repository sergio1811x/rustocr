//! Конвейер OCR: детекция → вырезание строк → распознавание (RU+EN).

use crate::det::{self, DetBox, ProbMap};
use crate::geom::{order_quad, warp_quad, Pt};
use crate::rec::{self, RecBatch};
use anyhow::{Context, Result};
use image::RgbImage;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use ort::inputs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

const LATIN_FALLBACK_CONF: f32 = 0.90; // ниже — пробуем вторую модель
const LATIN_MARGIN: f32 = 0.05; // латиница должна быть заметно увереннее
const ROT_RETRY_CONF: f32 = 0.45; // средняя уверенность ниже — подозреваем поворот
const ROT_RETRY_MIN_BLOCKS: usize = 3; // минимум блоков, чтобы вообще пробовать
const ROT_MARGIN: f32 = 0.10; // повёрнутый вариант должен быть заметно лучше

const DESKEW_MIN_DEG: f32 = 0.3; // меньше — джиттер прямых сканов, не трогаем
const DESKEW_MAX_DEG: f32 = 5.0; // больше — это не крен (фото/перспектива)

/// Выравнивание крена страницы (env RUSTOCR_DESKEW, по умолчанию 1).
/// Крен 1–2° (лист криво лёг в сканер) глушит детекцию тонких строк:
/// на реальной смете 254 числа → 175 (−31%). Угол берём как взвешенную
/// медиану углов «линейных» квадов детекции (ширина > 3·высоты),
/// выравниваем страницу и распознаём заново; квады возвращаем
/// в координаты исходной картинки. Прямые страницы — угол ~0,
/// повторный проход не запускается.
fn deskew_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("RUSTOCR_DESKEW")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Медианный угол «линейных» боксов (рад), вес — квадрат ширины: у коротких
/// боксов дрейф от крена субпиксельный на карте детекции, их угол квантуется
/// в 0° и топит оценку (newizba: 13 «нулей» против 9 наклонённых).
/// None — мало улик.
fn skew_angle(quads: &[[Pt; 4]]) -> Option<f32> {
    let mut acc: Vec<(f32, f32)> = Vec::new(); // (угол, вес)
    for q in quads {
        let o = order_quad(q);
        let w = crate::geom::dist(o[0], o[1]);
        let h = crate::geom::dist(o[0], o[3]).max(crate::geom::dist(o[1], o[2]));
        if w > 2.5 * h && h > 4.0 && w > 80.0 {
            acc.push(((o[1].y - o[0].y).atan2(o[1].x - o[0].x), w * w));
        }
    }
    if acc.len() < 5 {
        return None;
    }
    let total: f32 = acc.iter().map(|(_, w)| *w).sum();
    acc.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    if std::env::var("RUSTOCR_DESKEW_DEBUG").is_ok() {
        let meds = acc
            .iter()
            .map(|(a, _)| a.to_degrees())
            .collect::<Vec<_>>();
        eprintln!(
            "[skew] линейных: {}, углы: min {:.2}° / медиана-элем {:.2}° / max {:.2}°",
            acc.len(),
            meds.first().copied().unwrap_or(0.0),
            meds[meds.len() / 2],
            meds.last().copied().unwrap_or(0.0)
        );
    }
    let mut cum = 0.0;
    for (ang, w) in &acc {
        cum += *w;
        if cum >= total * 0.5 {
            return Some(*ang);
        }
    }
    None
}

/// Поворорот картинки на θ (рад) вокруг центра: out(x,y) = src(R_θ(x,y)).
/// Текст, наклонённый на +θ, становится горизонтальным.
fn rotate_page(img: &RgbImage, theta: f32) -> RgbImage {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let (cx, cy) = (w / 2.0, h / 2.0);
    let (s, c) = (theta.sin(), theta.cos());
    let rot = |x: f32, y: f32| -> Pt {
        let (dx, dy) = (x - cx, y - cy);
        Pt {
            x: cx + dx * c - dy * s,
            y: cy + dx * s + dy * c,
        }
    };
    let quad = [
        rot(0.0, 0.0),
        rot(w - 1.0, 0.0),
        rot(w - 1.0, h - 1.0),
        rot(0.0, h - 1.0),
    ];
    crate::geom::warp_quad(img, &quad, img.width(), img.height())
}

/// Квады результата из координат выровненной картинки — в исходные.
fn unrotate_quads(res: &mut OcrResult, theta: f32, w: u32, h: u32) {
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let (s, c) = (theta.sin(), theta.cos());
    for b in res.blocks.iter_mut() {
        for p in b.quad.iter_mut() {
            let (dx, dy) = (p.x - cx, p.y - cy);
            p.x = cx + dx * c - dy * s;
            p.y = cy + dx * s + dy * c;
        }
    }
}

const REC_LONG_RATIO: f32 = 26.0; // шире 26:1 (≈1280px при h=48) — строка сжимается

/// Запас вокруг детекторного квада при вырезании строки (env RUSTOCR_CROP_PAD,
/// 0..=16, по умолчанию 0 = выкл). Проверено на 14 сметах (17 119 чисел):
/// pad 2 → 98.1%, pad 4 → 94.2% против 99.2% без него. В тесных таблицах
/// запас втягивает цифры соседних ячеек в кроп (потеря хвостовых «,00»);
/// точечный выигрыш на бледных ячейках не окупает массовых потерь.
fn crop_pad() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("RUSTOCR_CROP_PAD")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.0)
            .clamp(0.0, 16.0)
    })
}

/// Максимальный размер порции строк в под-пачке (env RUSTOCR_REC_CHUNK, 1..=64).
fn rec_chunk() -> usize {
    std::env::var("RUSTOCR_REC_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16)
        .clamp(1, 64)
}

/// Лёгкая пост-обработка: типовые единицы измерения в сметах.
/// «м3»/«м2» — это м³/м² (строчная «м», за ней цифра, перед ней не буква:
/// «М300» — марка бетона, её не трогаем).
fn normalize_units(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < n {
        let ch = chars[i];
        let next_is_23 = i + 1 < n && (chars[i + 1] == '2' || chars[i + 1] == '3');
        // перед «м» не буква (цифра — можно: «75м2» = 75 м²)
        let prev_ok = i == 0 || !chars[i - 1].is_alphabetic();
        if ch == 'м' && next_is_23 && prev_ok {
            out.push('м');
            out.push(if chars[i + 1] == '2' { '²' } else { '³' });
            i += 2;
            continue;
        }
        out.push(ch);
        i += 1;
    }
    out
}

/// Замена «букв-двойников цифр» рядом с цифрами: б→6, З→3, О/о→0.
/// OCR на мелком шрифте путает их («б000» вместо «6000»). Заменяем букву
/// только если сосед — цифра: слова «под», «Зоя» не страдают.
/// Сосед считается и через один разделитель: «1-Зсорт» → «1-3сорт»
/// (дефис режет токен, но по факту это цифра диапазона; на 14 реальных
/// сметах такая правка вернула 100+ «потерянных» чисел в КД-документах).
fn normalize_digit_lookalikes(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let is_d = |c: char| c.is_ascii_digit();
    let sep = |c: char| matches!(c, '-' | '–' | '—' | '/');
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    // токен = буквы+цифры подряд
    while i < n {
        if chars[i].is_alphanumeric() {
            let start = i;
            while i < n && chars[i].is_alphanumeric() {
                i += 1;
            }
            let tok = &chars[start..i];
            let has_digit = tok.iter().any(|&c| is_d(c));
            // «цифра-разделитель-токен…»: 1-Зсорт, 2/Зм, 100-Зшт
            let digit_before_sep =
                start >= 2 && sep(chars[start - 1]) && is_d(chars[start - 2]);
            if has_digit {
                for (k, &c) in tok.iter().enumerate() {
                    let prev_d = k > 0 && is_d(tok[k - 1]);
                    let next_d = k + 1 < tok.len() && is_d(tok[k + 1]);
                    let cross = k == 0 && digit_before_sep;
                    let rep = match c {
                        'б' if next_d => Some('6'),
                        'З' if prev_d || next_d || cross => Some('3'),
                        'О' | 'о' if prev_d || next_d => Some('0'),
                        _ => None,
                    };
                    out.push(rep.unwrap_or(c));
                }
            } else if digit_before_sep && tok[0] == 'З' {
                // токен без своих цифр, но открывается З после «цифра-дефис»
                out.push('3');
                out.extend(tok[1..].iter());
            } else {
                out.extend(tok.iter());
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Одиночный символ О/о/O/o — это цифра 0 в изолированной ячейке таблицы:
/// без соседей модель не отличает букву от цифры. В реальных сметах
/// (РувитСтрой) целый столбец нулей читался буквой «о» — 16 «потерянных»
/// чисел. Осмысленного блока из одной буквы «о» в документах не бывает.
fn fix_lone_letter_o(text: &str) -> String {
    match text.trim() {
        "О" | "о" | "O" | "o" => "0".to_string(),
        _ => text.to_string(),
    }
}

/// Перегруппировка пробелов в больших числах. OCR часто смещает
/// разделитель тысяч: «2103 955,86» или «1 372277,35». Правило:
/// несколько числовых кусков через пробел, дробь — только в последнем —
/// соединяем и ставим пробелы по 3 разряда с конца.
/// Склейки соседних колонок вида «230518,62 0009» (последний кусок — шифр
/// без дроби, 4+ цифры) не трогаем.
fn normalize_numbers(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < n {
        if chars[i].is_ascii_digit() {
            // максимум из цифр/пробелов/запятых/точек, заканчивается цифрой
            let start = i;
            while i < n
                && (chars[i].is_ascii_digit()
                    || chars[i] == ' '
                    || chars[i] == '\u{a0}'
                    || chars[i] == ','
                    || chars[i] == '.')
            {
                i += 1;
            }
            let mut e = i;
            while e > start && !chars[e - 1].is_ascii_digit() {
                e -= 1;
            }
            i = e;
            let token: String = chars[start..e].iter().collect();
            out.push_str(&regroup_number(&token));
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Перегруппировка одного числового токена (куски через пробел).
/// Склеиваем только «разорванное» число, но не соседние ячейки таблицы:
/// • с дробью («2103 955,86», «1 372277,35») — каждый кусок до последнего
///   либо ровно 3 цифры, либо длина ≡ 1 (mod 3) — ведущая группа разрядов;
/// • без дроби — только классическая группировка «1 250 000»:
///   ≥ 3 куска, первый ≤ 3 цифр, остальные ровно по 3.
/// «150 0», «100 50», «4500 120» (склеенные ячейки) остаются как есть.
fn regroup_number(token: &str) -> String {
    let pieces: Vec<&str> = token.split([' ', '\u{a0}']).filter(|p| !p.is_empty()).collect();
    if pieces.len() < 2 {
        return token.to_string();
    }
    // дробь допустима только в последнем куске
    for p in &pieces[..pieces.len() - 1] {
        if p.contains(',') || p.contains('.') {
            return token.to_string();
        }
    }
    let last = pieces[pieces.len() - 1];
    let (int_last, frac) = match last.find([',', '.']) {
        Some(pos) => (&last[..pos], &last[pos..]),
        None => (last, ""),
    };
    // последний кусок без дроби и 4+ цифр — похоже на шифр расценки рядом
    if frac.is_empty() && int_last.len() >= 4 {
        return token.to_string();
    }
    if !frac.is_empty() {
        // разорванное число с копейками: куски должны выглядеть группами разрядов
        let ok = pieces[..pieces.len() - 1]
            .iter()
            .all(|p| p.len() == 3 || p.len() % 3 == 1);
        if !ok {
            return token.to_string();
        }
    } else {
        // без дроби: только «1 250 000» — классические тройки разрядов
        if pieces.len() < 3 || pieces[0].len() > 3 || pieces[1..].iter().any(|p| p.len() != 3) {
            return token.to_string();
        }
    }
    // нет ни одного крупного куска — мало данных, не трогаем
    let has3 = pieces.iter().any(|p| p.len() >= 3) || !frac.is_empty();
    if !has3 {
        return token.to_string();
    }
    let mut digits = String::new();
    for p in &pieces[..pieces.len() - 1] {
        digits.push_str(p);
    }
    digits.push_str(int_last);
    if digits.len() < 4 {
        return token.to_string();
    }
    // пробелы по 3 разряда с конца
    let d: Vec<char> = digits.chars().collect();
    let dl = d.len();
    let mut grouped = String::with_capacity(dl + dl / 3);
    for (k, ch) in d.iter().enumerate() {
        grouped.push(*ch);
        let rem = dl - k - 1;
        if rem > 0 && rem % 3 == 0 {
            grouped.push(' ');
        }
    }
    let mut res = grouped;
    res.push_str(frac);
    res
}

/// Режим языка: авто (обе модели), только русский, только английский.
#[derive(Clone, Copy, PartialEq)]
pub enum LangMode {
    Auto,
    Ru,
    En,
}

impl LangMode {
    fn from_env() -> Self {
        match std::env::var("RUSTOCR_LANG").unwrap_or_default().to_lowercase().as_str() {
            "ru" | "cyr" | "cyrillic" => LangMode::Ru,
            "en" | "lat" | "latin" => LangMode::En,
            _ => LangMode::Auto,
        }
    }
}

#[derive(Clone)]
pub struct Block {
    pub text: String,
    pub confidence: f32,
    pub lang: &'static str, // "ru" | "en" | "mix"
    pub quad: [Pt; 4],
}

#[derive(Clone)]
pub struct OcrResult {
    pub width: u32,
    pub height: u32,
    pub text: String,
    pub blocks: Vec<Block>,
    /// Фрагменты, не прочитанные ни одной моделью (сбой инференса).
    /// Виден в JSON как skipped_fragments — потеря текста не остаётся тайной.
    pub skipped: usize,
    pub det_ms: u64,
    pub rec_ms: u64,
    pub total_ms: u64,
}

/// Настройки движка. Заполняются из env-переменных, но при обработке
/// PDF можно создать несколько движков с ограниченными потоками.
pub struct EngineOptions {
    pub models_dir: PathBuf,
    pub lang: LangMode,
    pub det_limit: u32,
    /// Лимит intra-op потоков ONNX Runtime (None = все ядра).
    pub intra_threads: Option<usize>,
}

impl EngineOptions {
    pub fn from_env() -> Self {
        Self {
            models_dir: models_dir(),
            lang: LangMode::from_env(),
            det_limit: std::env::var("RUSTOCR_DET_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1280),
            intra_threads: None,
        }
    }
}

pub struct OcrEngine {
    _env: Environment, // держим ONNX Runtime живым, пока живут сессии
    det: Mutex<Session>,
    rec_cyr: Mutex<Session>,
    rec_lat: Mutex<Session>,
    lang_mode: LangMode,
    table_cyr: Vec<String>,
    table_lat: Vec<String>,
    det_in: String,
    det_out: String,
    rec_cyr_in: String,
    rec_cyr_out: String,
    rec_lat_in: String,
    rec_lat_out: String,
    det_limit: u32,
}

impl OcrEngine {
    /// Загрузить движок с заданными настройками. Окружение ONNX Runtime
    /// создаётся один раз на процесс и передаётся сюда (engine хранит клон).
    pub fn load(opts: &EngineOptions, env: &Environment) -> Result<Self> {
        let models_dir = &opts.models_dir;
        let det_path = models_dir.join("det.onnx");
        let rec_cyr_path = models_dir.join("rec_cyrillic.onnx");
        let rec_lat_path = models_dir.join("rec_latin.onnx");
        for p in [&det_path, &rec_cyr_path, &rec_lat_path] {
            if !p.exists() {
                anyhow::bail!("модель не найдена: {}", p.display());
            }
        }
        let build_session = |path: &Path| -> Result<Session> {
            let mut b = Session::builder(env).context("builder сессии")?;
            if let Some(n) = opts.intra_threads {
                b = b.with_intra_threads(n).map_err(|e| {
                    anyhow::anyhow!("настройка intra-op потоков: {}", e)
                })?;
            }
            b.commit_from_file(path)
                .with_context(|| format!("загрузка {}", path.display()))
        };
        let det = build_session(&det_path)?;
        let rec_cyr = build_session(&rec_cyr_path)?;
        let rec_lat = build_session(&rec_lat_path)?;
        let table_cyr = rec::load_table(&models_dir.join("cyrillic_dict.txt"))?;
        let table_lat = rec::load_table(&models_dir.join("latin_dict.txt"))?;
        let lang_mode = opts.lang;
        let det_limit = opts.det_limit;
        let io = |s: &Session| -> (String, String) {
            (
                s.inputs()[0].name().to_string(),
                s.outputs()[0].name().to_string(),
            )
        };
        let (det_in, det_out) = io(&det);
        let (rec_cyr_in, rec_cyr_out) = io(&rec_cyr);
        let (rec_lat_in, rec_lat_out) = io(&rec_lat);
        Ok(Self {
            _env: env.clone(),
            det: Mutex::new(det),
            rec_cyr: Mutex::new(rec_cyr),
            rec_lat: Mutex::new(rec_lat),
            lang_mode,
            table_cyr,
            table_lat,
            det_in,
            det_out,
            rec_cyr_in,
            rec_cyr_out,
            rec_lat_in,
            rec_lat_out,
            det_limit,
        })
    }

    /// Размер словаря кириллицы (для диагностики загрузки).
    pub fn table_size(&self) -> usize {
        self.table_cyr.len()
    }

    /// Распознать всю картинку (выравнивание крена + авто-поворот 90° при
    /// неуверенном чтении).
    pub fn recognize(&self, img: &RgbImage) -> Result<OcrResult> {
        let base = self.recognize_inner(img)?;
        let mut best = base;
        // 0) крен: тонкие строки пропадают из детекции — выравниваем и
        //    перечитываем; вариант с бóльшим числом блоков выигрывает.
        //    До двух проходов: на бледных страницах крен глушит и длинные
        //    боксы — первый угол недооценён, дожимаем остаток вторым.
        if deskew_enabled() {
            let mut straight_img: Option<RgbImage> = None;
            let mut cum_ang = 0.0f32;
            for _ in 0..2 {
                let quads: Vec<[Pt; 4]> = best.blocks.iter().map(|b| b.quad).collect();
                let Some(ang) = skew_angle(&quads) else { break };
                if std::env::var("RUSTOCR_DESKEW_DEBUG").is_ok() {
                    eprintln!(
                        "[deskew] угол {:.3}° (линейных квадов: {}), блоков сейчас {}",
                        ang.to_degrees(),
                        quads.len(),
                        best.blocks.len()
                    );
                }
                let deg = ang.to_degrees().abs();
                if !(DESKEW_MIN_DEG..=DESKEW_MAX_DEG).contains(&deg) {
                    break;
                }
                let base_img: &RgbImage = straight_img.as_ref().unwrap_or(img);
                let rotated = rotate_page(base_img, ang);
                let Ok(c) = self.recognize_inner(&rotated) else { break };
                // «не хуже» по блокам: частичная коррекция может не дать
                // прироста, но приближает страницу к прямой — остаток
                // дожмётся вторым проходом
                if c.blocks.len() >= best.blocks.len()
                    && avg_confidence(&c) >= avg_confidence(&best) - 0.05
                {
                    best = c; // квады — в координатах выровненной картинки
                    cum_ang += ang;
                    straight_img = Some(rotated);
                } else {
                    break;
                }
            }
            if cum_ang != 0.0 {
                unrotate_quads(&mut best, cum_ang, img.width(), img.height());
            }
        }
        let avg = avg_confidence(&best);
        if avg >= ROT_RETRY_CONF || best.blocks.len() < ROT_RETRY_MIN_BLOCKS {
            return Ok(best);
        }
        // картинка читается плохо — возможно, она повёрнута: пробуем 90/180/270°
        for angle in [90u16, 180, 270] {
            let rotated = match angle {
                90 => image::imageops::rotate90(img),
                180 => image::imageops::rotate180(img),
                _ => image::imageops::rotate270(img),
            };
            // неудача повёрнутого варианта не должна валить весь запрос
            let candidate = match self.recognize_inner(&rotated) {
                Ok(c) => c,
                Err(_) => continue,
            };
            if avg_confidence(&candidate) > avg_confidence(&best) + ROT_MARGIN {
                best = candidate;
            }
        }
        Ok(best)
    }

    /// Основной проход без повторов.
    fn recognize_inner(&self, img: &RgbImage) -> Result<OcrResult> {
        let t0 = Instant::now();

        // 1) Детекция текстовых блоков
        let inp = det::preprocess(img, self.det_limit);
        let t_det = Instant::now();
        let prob = self.run_det(&inp.data, inp.shape)?;
        let det_ms = t_det.elapsed().as_millis() as u64; // чисто инференс
        let boxes = det::postprocess(&prob, inp.ratio, img.width(), img.height());

        // 2) Вырезание строк
        let crops: Vec<RgbImage> = boxes
            .iter()
            .filter_map(|b| crop_box(img, b))
            .collect();

        // 3) Распознавание — модель по режиму языка
        let t_rec = Instant::now();
        let (mut results, mut lost) = match self.lang_mode {
            LangMode::En => self.run_rec_batch(
                &self.rec_lat,
                &self.table_lat,
                &self.rec_lat_in,
                &self.rec_lat_out,
                &crops,
            ),
            _ => self.run_rec_batch(
                &self.rec_cyr,
                &self.table_cyr,
                &self.rec_cyr_in,
                &self.rec_cyr_out,
                &crops,
            ),
        };

        // 3а) Длинные строки: ширина обрезана в 1280px и текст сжимается —
        // перечитываем по частям в полном разрешении и склеиваем
        for i in 0..results.len() {
            if i >= crops.len() {
                break;
            }
            let c = &crops[i];
            let ratio = c.width() as f32 / c.height().max(1) as f32;
            if ratio > REC_LONG_RATIO {
                if let Some((t2, c2)) = self.retry_long_crop(c) {
                    if !t2.trim().is_empty() && c2 > 0.0 {
                        results[i] = (t2, c2, "model");
                        lost[i] = false;
                    }
                }
            }
        }

        // 4) Запасная модель для неуверенных строк (только в авто-режиме).
        //    Строки из цифр и знаков пропускаем: обе модели читают их
        //    одинаково — перечитывание только тратит время.
        if self.lang_mode == LangMode::Auto {
            let weak: Vec<usize> = results
                .iter()
                .enumerate()
                .filter(|(_, (t, c, _))| {
                    (t.trim().is_empty() || *c < LATIN_FALLBACK_CONF) && !is_digitish(t)
                })
                .map(|(i, _)| i)
                .collect();
            if !weak.is_empty() {
                let weak_crops: Vec<RgbImage> =
                    weak.iter().map(|&i| crops[i].clone()).collect();
                let (lat, lat_lost) = self.run_rec_batch(
                    &self.rec_lat,
                    &self.table_lat,
                    &self.rec_lat_in,
                    &self.rec_lat_out,
                    &weak_crops,
                );
                for (k, &i) in weak.iter().enumerate() {
                    if lat[k].1 > results[i].1 + LATIN_MARGIN {
                        results[i] = lat[k].clone();
                    }
                    // вторая модель прочитала фрагмент — он не потерян
                    if !lat_lost[k] {
                        lost[i] = false;
                    }
                }
            }
        }
        let rec_ms = t_rec.elapsed().as_millis() as u64;
        let skipped = lost.iter().filter(|&&f| f).count();
        // все фрагменты сбойнули — это ошибка запроса, а не «пустая страница»
        if !crops.is_empty() && skipped == crops.len() {
            anyhow::bail!(
                "распознавание не удалось: не прочитан ни один из {} фрагментов",
                crops.len()
            );
        }

        // 5) Сборка результата: текст + язык по составу символов
        //    (пост-обработка под сметы: м³/м², буквы-двойники, разряды)
        let mut blocks: Vec<Block> = Vec::with_capacity(boxes.len());
        let mut lines: Vec<String> = Vec::new();
        for (i, b) in boxes.iter().enumerate() {
            if i >= results.len() {
                break;
            }
            let (raw, conf, _) = &results[i];
            let text = fix_lone_letter_o(&normalize_numbers(&normalize_digit_lookalikes(
                &normalize_units(raw),
            )));
            let lang = detect_lang(&text);
            blocks.push(Block {
                text: text.clone(),
                confidence: *conf,
                lang,
                quad: b.quad,
            });
            if !text.trim().is_empty() {
                lines.push(text.trim().to_string());
            }
        }
        let text = lines.join("\n");
        let total_ms = t0.elapsed().as_millis() as u64;

        Ok(OcrResult {
            width: img.width(),
            height: img.height(),
            text,
            blocks,
            skipped,
            det_ms,
            rec_ms,
            total_ms,
        })
    }

    /// Распознать текст в заданных областях картинки. Нужно для разбора
    /// таблиц: склеенный блок режется на части по границам колонок и
    /// перечитывается. Выравнивание результата — по индексам областей.
    pub fn recognize_quads(&self, img: &RgbImage, quads: &[[Pt; 4]]) -> Vec<Option<(String, f32)>> {
        let mut idxs: Vec<usize> = Vec::with_capacity(quads.len());
        let mut crops: Vec<RgbImage> = Vec::with_capacity(quads.len());
        for (i, q) in quads.iter().enumerate() {
            if let Some(c) = crop_quad(img, q) {
                idxs.push(i);
                crops.push(c);
            }
        }
        let (run, _) = match self.lang_mode {
            LangMode::En => self.run_rec_batch(
                &self.rec_lat,
                &self.table_lat,
                &self.rec_lat_in,
                &self.rec_lat_out,
                &crops,
            ),
            _ => self.run_rec_batch(
                &self.rec_cyr,
                &self.table_cyr,
                &self.rec_cyr_in,
                &self.rec_cyr_out,
                &crops,
            ),
        };
        let mut out: Vec<Option<(String, f32)>> = vec![None; quads.len()];
        for (k, &i) in idxs.iter().enumerate() {
            if let Some((text, conf, _)) = run.get(k) {
                // та же цепочка пост-обработки, что и для обычных блоков:
                // в ячейках таблиц числа важнее всего, «б000»/«1-Зсорт»
                // не должны утекать в XLSX сырыми
                let text = fix_lone_letter_o(&normalize_numbers(&normalize_digit_lookalikes(
                    &normalize_units(text),
                )));
                out[i] = Some((text, *conf));
            }
        }
        out
    }

    fn run_det(&self, data: &[f32], shape: [usize; 4]) -> Result<ProbMap> {
        let input = Tensor::from_array((shape, data.to_vec())).context("тензор детектора")?;
        let mut session = lock(&self.det);
        let outputs = session
            .run(inputs![self.det_in.as_str() => input])
            .context("запуск детектора")?;
        let out = &outputs[self.det_out.as_str()];
        let (oshape, raw) = out
            .try_extract_tensor::<f32>()
            .context("выход детектора")?;
        // [1, 1, H, W]
        let (h, w) = (oshape[2] as usize, oshape[3] as usize);
        if raw.len() != h * w {
            anyhow::bail!(
                "неожиданный размер выхода детектора: {} при {}×{}",
                raw.len(),
                h,
                w
            );
        }
        Ok(ProbMap {
            w,
            h,
            data: raw.to_vec(),
        })
    }

    /// Пачка строк для одного прогона сети: на плотных таблицах блоков
    /// сотни, и один общий тензор съедал гигабайты (убийство по OOM).
    /// Поэтому режем на порции и сортируем их по ширине
    /// (как batch_size + сортировка по max_wh_ratio в PaddleOCR):
    /// узкие строки паддятся до узкой пачки, а не до самой широкой строки.
    /// Возвращает (результаты по кропам, признак «фрагмент не прочитан»):
    /// потеря текста из-за сбоя видна вызывающей стороне.
    fn run_rec_batch(
        &self,
        rec: &Mutex<Session>,
        table: &[String],
        in_name: &str,
        out_name: &str,
        crops: &[RgbImage],
    ) -> (Vec<(String, f32, &'static str)>, Vec<bool>) {
        let n_total = crops.len();
        let mut results: Vec<(String, f32, &'static str)> = Vec::with_capacity(n_total);
        if n_total == 0 {
            return (results, Vec::new());
        }
        // ширина/высота каждого кропа — по ней сортируем
        let ratios: Vec<f32> = crops
            .iter()
            .map(|c| c.width() as f32 / c.height().max(1) as f32)
            .collect();
        let mut order: Vec<usize> = (0..n_total).collect();
        order.sort_by(|&a, &b| ratios[a].partial_cmp(&ratios[b]).unwrap_or(std::cmp::Ordering::Equal));

        // порция строк: env RUSTOCR_REC_CHUNK (по умолчанию 16), минимум 16 —
        // с сортировкой по ширине паддинг в пачке и так минимален
        let chunk_max = rec_chunk().max(16);
        // Порции собираем не просто по счёту, а по похожей ширине: строки
        // внутри порции паддятся до самой широкой, и «мешанина» из узкой
        // и широкой строки жгла процессор впустую.
        let mut chunks: Vec<Vec<usize>> = Vec::new();
        let mut cur: Vec<usize> = Vec::with_capacity(chunk_max);
        let (mut rmin, mut rmax) = (f32::INFINITY, 0.0f32);
        for &idx in order.iter() {
            let r = ratios[idx];
            let too_wide = !cur.is_empty()
                && (cur.len() >= chunk_max || r > rmin * 1.7 || r * 1.7 < rmax);
            if too_wide {
                chunks.push(std::mem::take(&mut cur));
                rmin = f32::INFINITY;
                rmax = 0.0;
            }
            cur.push(idx);
            rmin = rmin.min(r);
            rmax = rmax.max(r);
        }
        if !cur.is_empty() {
            chunks.push(cur);
        }
        let mut decoded: Vec<Option<(String, f32)>> = vec![None; n_total];
        for chunk in &chunks {
            let batch_crops: Vec<&RgbImage> = chunk.iter().map(|&i| &crops[i]).collect();
            let batch: RecBatch = rec::preprocess_batch(&batch_crops);
            let run = (|| -> Result<Vec<(String, f32)>> {
                let input = Tensor::from_array((batch.shape, batch.data)).context("тензор строк")?;
                let mut session = lock(rec);
                let outputs = session
                    .run(inputs![in_name => input])
                    .context("запуск распознавателя")?;
                let out = &outputs[out_name];
                let (oshape, raw) = out
                    .try_extract_tensor::<f32>()
                    .context("выход распознавателя")?;
                // [N, T, C]
                let (n, t, c) = (oshape[0] as usize, oshape[1] as usize, oshape[2] as usize);
                let per = t * c;
                let mut part = Vec::with_capacity(n);
                for i in 0..n.min(chunk.len()) {
                    let (text, conf) = rec::ctc_decode(&raw[i * per..(i + 1) * per], t, c, table);
                    part.push((text, conf));
                }
                Ok(part)
            })();
            match run {
                Ok(part) => {
                    for (k, &i) in chunk.iter().enumerate() {
                        if let Some(v) = part.get(k) {
                            decoded[i] = Some(v.clone());
                        }
                    }
                }
                Err(e) => {
                    // под-пачка не прошла — её строки останутся пустыми,
                    // но теперь это видно по skipped в ответе
                    eprintln!(
                        "[rustocr] под-пачка распознавателя не удалась ({} строк): {}",
                        chunk.len(),
                        e
                    );
                }
            }
        }
        let mut failed: Vec<bool> = Vec::with_capacity(n_total);
        for d in decoded {
            failed.push(d.is_none());
            results.push(match d {
                Some((text, conf)) => (text, conf, "model"),
                None => (String::new(), 0.0, "model"),
            });
        }
        (results, failed)
    }

    /// Перечитать очень длинную строку по частям в полном разрешении.
    /// Число частей: каждая — не шире 24:1 (меньше лимита 1280/48).
    /// Границы частей сдвигаются к ближайшему «просвету» — вертикальной
    /// полосе без текста (пробел между колонками сметы): фиксированная
    /// нарезка режет посреди чисел и теряет цифры («33 245,54» → «3»).
    /// Перекрытие между соседями сохраняется (в просвете оно пустое —
    /// склейка вставит пробел), последовательное слияние с вычитанием
    /// дублирующейся зоны.
    fn retry_long_crop(&self, crop: &RgbImage) -> Option<(String, f32)> {
        let (w, h) = crop.dimensions();
        let ratio = w as f32 / h.max(1) as f32;
        if h < 8 {
            return None;
        }
        let n = ((ratio / 24.0).ceil() as usize).max(2);
        let overlap = ((w as f32 * 0.10).round() as u32).max(4);
        // ширина части: (w + (n-1)*overlap) / n
        let pw = ((w as f32 + (n as f32 - 1.0) * overlap as f32) / n as f32).ceil() as u32;
        if pw < 8 {
            return None;
        }
        let gaps = gap_centers(crop);
        let max_shift = (pw as f32 * 0.35).round() as u32; // куда можно сдвинуть границу
        let mut crops: Vec<RgbImage> = Vec::with_capacity(n);
        let mut x = 0u32;
        while x < w {
            let ideal = x + pw;
            let end = if ideal < w {
                // граница — в просвет, если он в досягаемости; иначе как было
                let snapped = snap_to_gap(ideal, &gaps, max_shift);
                snapped.clamp(x + 8, w - 8)
            } else {
                w
            };
            crops.push(image::imageops::crop_imm(crop, x, 0, end - x, h).to_image());
            if end >= w {
                break;
            }
            x = end.saturating_sub(overlap);
        }
        if crops.len() < 2 {
            return None;
        }
        let mode_lat = self.lang_mode == LangMode::En;
        let (parts, _) = if mode_lat {
            self.run_rec_batch(
                &self.rec_lat,
                &self.table_lat,
                &self.rec_lat_in,
                &self.rec_lat_out,
                &crops,
            )
        } else {
            self.run_rec_batch(
                &self.rec_cyr,
                &self.table_cyr,
                &self.rec_cyr_in,
                &self.rec_cyr_out,
                &crops,
            )
        };
        if parts.len() != crops.len() {
            return None;
        }
        // последовательное слияние с дедупликацией перекрытий
        let mut acc = parts[0].0.trim().to_string();
        let mut wsum = acc.chars().count().max(1) as f32;
        let mut csum = parts[0].1 * wsum;
        for (t, c, _) in parts.iter().skip(1) {
            let t = t.trim();
            let tl = t.chars().count().max(1) as f32;
            wsum += tl;
            csum += c * tl;
            acc = merge_overlapping(&acc, t);
        }
        let conf = csum / wsum;
        Some((acc, conf))
    }
}

/// Склеить два текста, вычитая дублирующуюся зону перекрытия.
fn merge_overlapping(a: &str, b: &str) -> String {
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    let max_k = ac.len().min(bc.len()).min(20);
    let mut best = 0;
    for k in 1..=max_k {
        if ac[ac.len() - k..] == bc[..k] {
            best = k;
        }
    }
    let mut out = String::with_capacity(a.len() + b.len());
    out.push_str(a);
    if best == 0 {
        let last_a = ac.last().copied().unwrap_or(' ');
        let first_b = bc.first().copied().unwrap_or(' ');
        if last_a.is_alphanumeric() && first_b.is_alphanumeric() {
            out.push(' ');
        }
    }
    out.extend(bc[best..].iter());
    out
}

/// Профиль «чернил» по колонкам кропа: сколько тёмных пикселей в колонке x.
fn ink_profile(crop: &RgbImage) -> Vec<u32> {
    let (w, h) = crop.dimensions();
    let mut prof = vec![0u32; w as usize];
    for y in 0..h {
        for x in 0..w {
            let p = crop.get_pixel(x, y);
            // быстрая люминанция; фон кропа после warp — белый
            let lum = (p[0] as u32 + p[1] as u32 + p[2] as u32) / 3;
            if lum < 160 {
                prof[x as usize] += 1;
            }
        }
    }
    prof
}

/// Центры «просветов» — вертикальных полос практически без текста
/// (пробелы между колонками сметы). Ширина полосы ≥ 3 px.
fn gap_centers(crop: &RgbImage) -> Vec<u32> {
    let (w, h) = crop.dimensions();
    let prof = ink_profile(crop);
    // колонка «пустая», если тёмных пикселей ≤ 6% высоты
    let limit = ((h as f32) * 0.06).ceil() as u32;
    let mut centers = Vec::new();
    let mut run_start: Option<usize> = None;
    for (x, &v) in prof.iter().enumerate() {
        if v <= limit {
            if run_start.is_none() {
                run_start = Some(x);
            }
        } else if let Some(s) = run_start.take() {
            if x - s >= 3 {
                centers.push(((s + x) / 2) as u32);
            }
        }
    }
    if let Some(s) = run_start {
        if w as usize - s >= 3 {
            centers.push(((s + w as usize) / 2) as u32);
        }
    }
    centers
}

/// Сдвинуть границу разреза к ближайшему просвету (не дальше max_shift).
fn snap_to_gap(x: u32, centers: &[u32], max_shift: u32) -> u32 {
    let mut best = x;
    let mut best_d = u32::MAX;
    for &c in centers {
        let d = if c > x { c - x } else { x - c };
        if d < best_d && d <= max_shift {
            best_d = d;
            best = c;
        }
    }
    best
}

/// Вырезать блок из картинки (перспектива → прямой прямоугольник).
fn crop_box(img: &RgbImage, b: &DetBox) -> Option<RgbImage> {
    let mut q = b.quad;
    let pad = crop_pad();
    if pad > 0.0 {
        // расширяем квад от центра: каждый угол уходит наружу на pad
        // (для прямоугольного квада — ровно ±pad по обеим осям)
        let cx = (q[0].x + q[1].x + q[2].x + q[3].x) * 0.25;
        let cy = (q[0].y + q[1].y + q[2].y + q[3].y) * 0.25;
        for p in q.iter_mut() {
            p.x += (p.x - cx).signum() * pad;
            p.y += (p.y - cy).signum() * pad;
            p.x = p.x.clamp(0.0, (img.width() - 1) as f32);
            p.y = p.y.clamp(0.0, (img.height() - 1) as f32);
        }
    }
    crop_quad(img, &q)
}

fn crop_quad(img: &RgbImage, q: &[Pt; 4]) -> Option<RgbImage> {
    let q = order_quad(q);
    let w_top = crate::geom::dist(q[0], q[1]);
    let w_bot = crate::geom::dist(q[2], q[3]);
    let h_left = crate::geom::dist(q[0], q[3]);
    let h_right = crate::geom::dist(q[1], q[2]);
    let w = w_top.max(w_bot).floor() as u32;
    let h = h_left.max(h_right).floor() as u32;
    if w < 3 || h < 3 {
        return None;
    }
    Some(warp_quad(img, &q, w.min(4096), h.min(1024)))
}

impl OcrResult {
    /// JSON-представление результата (общее для CLI и HTTP-сервера).
    pub fn to_json(&self) -> serde_json::Value {
        let blocks: Vec<serde_json::Value> = self
            .blocks
            .iter()
            .map(|b| {
                let xs = [b.quad[0].x, b.quad[1].x, b.quad[2].x, b.quad[3].x];
                let ys = [b.quad[0].y, b.quad[1].y, b.quad[2].y, b.quad[3].y];
                let x0 = xs.iter().cloned().fold(f32::INFINITY, f32::min);
                let y0 = ys.iter().cloned().fold(f32::INFINITY, f32::min);
                let x1 = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let y1 = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                serde_json::json!({
                    "text": b.text,
                    "confidence": (b.confidence * 1000.0).round() / 1000.0,
                    "lang": b.lang,
                    "box": [x0.round() as i32, y0.round() as i32,
                            (x1 - x0).round() as i32, (y1 - y0).round() as i32],
                    "quad": [
                        b.quad[0].x, b.quad[0].y, b.quad[1].x, b.quad[1].y,
                        b.quad[2].x, b.quad[2].y, b.quad[3].x, b.quad[3].y,
                    ],
                })
            })
            .collect();
        serde_json::json!({
            "text": self.text,
            "width": self.width,
            "height": self.height,
            "blocks": blocks,
            "skipped_fragments": self.skipped,
            "timing_ms": {"det": self.det_ms, "rec": self.rec_ms, "total": self.total_ms},
        })
    }
}

/// Средняя уверенность по непустым блокам.
fn avg_confidence(r: &OcrResult) -> f32 {
    let vals: Vec<f32> = r
        .blocks
        .iter()
        .filter(|b| !b.text.trim().is_empty())
        .map(|b| b.confidence)
        .collect();
    if vals.is_empty() {
        0.0
    } else {
        vals.iter().sum::<f32>() / vals.len() as f32
    }
}

/// Похоже ли строка на «чисто цифровую» (цифры/знаки/пробелы):
/// таким строкам вторая модель ничего не добавит.
fn is_digitish(text: &str) -> bool {
    let mut alnum = 0usize;
    let mut digits = 0usize;
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            alnum += 1;
            if ch.is_numeric() {
                digits += 1;
            }
        }
    }
    alnum > 0 && (digits as f32 / alnum as f32) >= 0.8
}

/// Язык строки по составу символов.
fn detect_lang(text: &str) -> &'static str {
    let mut cyr = 0usize;
    let mut lat = 0usize;
    for ch in text.chars() {
        if ch.is_alphabetic() {
            if ('\u{0400}'..='\u{04FF}').contains(&ch) {
                cyr += 1;
            } else if ch.is_ascii_alphabetic() {
                lat += 1;
            }
        }
    }
    match (cyr, lat) {
        (0, 0) => "ru",
        (_, 0) => "ru",
        (0, _) => "en",
        _ => "mix",
    }
}

/// Найти каталог моделей с учётом переменной окружения.
pub fn models_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("RUSTOCR_MODELS").unwrap_or_else(|_| "models".to_string()),
    )
}

/// Захватить мьютекс, переживая отравление.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units() {
        assert_eq!(normalize_units("м3"), "м³");
        assert_eq!(normalize_units("м2"), "м²");
        assert_eq!(normalize_units("75м2"), "75м²");
        assert_eq!(normalize_units("М300"), "М300"); // марка бетона
        assert_eq!(normalize_units("объем 3"), "объем 3");
        assert_eq!(normalize_units("м3 м3"), "м³ м³");
    }

    #[test]
    fn gap_snapping() {
        use image::Rgb;
        // белая полоса 400×20, два «слова» тёмными пикселями: [10..80) и [300..380)
        let mut img = image::RgbImage::from_pixel(400, 20, Rgb([255, 255, 255]));
        for x in 10..80 {
            for y in 4..16 {
                img.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        for x in 300..380 {
            for y in 4..16 {
                img.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        let gaps = gap_centers(&img);
        // один просвет между словами (краевые полосы тоже пустые,
        // но центрируются на краях — для снапа это тоже безопасно)
        let mid = gaps
            .iter()
            .copied()
            .find(|&c| c > 100 && c < 290)
            .expect("просвет между словами не найден");
        // граница рядом с серединой сдвигается в просвет
        assert_eq!(snap_to_gap(190, &gaps, 200), mid);
        // далеко от просвета — остаётся на месте
        assert_eq!(snap_to_gap(190, &[50], 10), 190);
        // просвет за пределом сдвига — не используется
        assert_eq!(snap_to_gap(190, &[60], 100), 190);
    }

    #[test]
    fn lone_letter_o_is_zero() {
        assert_eq!(fix_lone_letter_o("о"), "0");
        assert_eq!(fix_lone_letter_o("О "), "0");
        assert_eq!(fix_lone_letter_o("O"), "0");
        assert_eq!(fix_lone_letter_o("o"), "0");
        // не одиночные — не трогаем
        assert_eq!(fix_lone_letter_o("под"), "под");
        assert_eq!(fix_lone_letter_o("0"), "0");
        assert_eq!(fix_lone_letter_o("окно"), "окно");
        assert_eq!(fix_lone_letter_o("1 о"), "1 о");
    }

    #[test]
    fn lookalikes() {
        assert_eq!(normalize_digit_lookalikes("25х100хб000мм"), "25х100х6000мм");
        assert_eq!(normalize_digit_lookalikes("б000"), "6000");
        assert_eq!(normalize_digit_lookalikes("2О25 г"), "2025 г");
        // обычные слова не трогаем
        assert_eq!(normalize_digit_lookalikes("под ocean Зоя"), "под ocean Зоя");
        assert_eq!(normalize_digit_lookalikes("Вес 1 доски"), "Вес 1 доски");
        // З в начале слова без цифр рядом — буква
        assert_eq!(normalize_digit_lookalikes("Зсорт"), "Зсорт");
        // 1Зсорт: З после цифры → 3
        assert_eq!(normalize_digit_lookalikes("1Зсорт"), "13сорт");
        // через один разделитель: дефис режет токен, но это цифра диапазона
        assert_eq!(normalize_digit_lookalikes("1-Зсорт"), "1-3сорт");
        assert_eq!(normalize_digit_lookalikes("100-Зшт"), "100-3шт");
        assert_eq!(normalize_digit_lookalikes("2/Зм"), "2/3м");
        assert_eq!(normalize_digit_lookalikes("6000-Зсорт2"), "6000-3сорт2");
        // два разделителя или буква перед дефисом — обычный текст, не трогаем
        assert_eq!(normalize_digit_lookalikes("1 - Зсорт"), "1 - Зсорт");
        assert_eq!(normalize_digit_lookalikes("п.-Зсорт"), "п.-Зсорт");
    }

    #[test]
    fn numbers_regroup() {
        assert_eq!(regroup_number("2103 955,86"), "2 103 955,86");
        assert_eq!(regroup_number("1 372277,35"), "1 372 277,35");
        assert_eq!(regroup_number("268 777,90"), "268 777,90");
        assert_eq!(regroup_number("3 000,00"), "3 000,00");
        assert_eq!(regroup_number("1 250 000"), "1 250 000");
        // шифр рядом с числом — не трогаем
        assert_eq!(regroup_number("230518,62 0009"), "230518,62 0009");
        // мелкие куски — не трогаем
        assert_eq!(regroup_number("10 20"), "10 20");
        assert_eq!(regroup_number("630,06"), "630,06");
        assert_eq!(regroup_number("5,953"), "5,953");
        // соседние ячейки таблицы — не склеиваем (было: «1 500», «10 050»)
        assert_eq!(regroup_number("150 0"), "150 0");
        assert_eq!(regroup_number("100 50"), "100 50");
        assert_eq!(regroup_number("4500 120"), "4500 120");
        assert_eq!(regroup_number("6000 25"), "6000 25");
        // кол-во + цена рядом — не склеиваем
        assert_eq!(regroup_number("12 0,5"), "12 0,5");
        assert_eq!(regroup_number("3 17,65"), "3 17,65");
    }

    #[test]
    fn numbers_in_text() {
        assert_eq!(
            normalize_numbers("Итого 2103 955,86 руб."),
            "Итого 2 103 955,86 руб."
        );
        assert_eq!(normalize_numbers("5,953"), "5,953");
        assert_eq!(normalize_numbers("д 12 мм"), "д 12 мм");
    }
}
