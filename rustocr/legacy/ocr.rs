//! Оркестрация пайплайна:
//! бинаризация -> макет страницы -> сегментация -> классификация -> текст.

use std::time::Instant;

use image::{imageops, DynamicImage, GrayImage};
use rayon::prelude::*;

use crate::cc;
use crate::classify;
use crate::features;
use crate::gray;
use crate::layout::{self, Line};
use crate::segment::{self, Seg};
use crate::skew;
use crate::templates::{Lang, Templates};

const SAUVOLA_K: f64 = 0.34;

/// Нерезкая маска 3x3: g' = g + amount * (g - box3(g)).
fn unsharp3(gray: &mut [u8], w: usize, h: usize, amount: f32) {
    if w < 3 || h < 3 {
        return;
    }
    let mut blur = vec![0u32; gray.len()];
    for y in 0..h {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(h - 1);
        for x in 0..w {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(w - 1);
            let s = gray[ym * w + xm] as u32 + gray[ym * w + x] as u32 + gray[ym * w + xp] as u32
                + gray[y * w + xm] as u32 + gray[y * w + x] as u32 + gray[y * w + xp] as u32
                + gray[yp * w + xm] as u32 + gray[yp * w + x] as u32 + gray[yp * w + xp] as u32;
            blur[y * w + x] = s / 9;
        }
    }
    for i in 0..gray.len() {
        let v = gray[i] as f32 + amount * (gray[i] as f32 - blur[i] as f32);
        gray[i] = v.round().clamp(0.0, 255.0) as u8;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// выправлять перекоск страницы
    pub deskew: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { deskew: true }
    }
}

#[derive(Clone, Debug)]
pub struct CharOut {
    pub ch: char,
    pub conf: f32,
    pub bbox: [i32; 4],
}

#[derive(Clone, Debug)]
pub struct WordOut {
    pub text: String,
    pub conf: f32,
    pub bbox: [i32; 4],
}

#[derive(Clone, Debug)]
pub struct LineOut {
    pub text: String,
    pub conf: f32,
    pub bbox: [i32; 4],
    pub chars: Vec<CharOut>,
    pub words: Vec<WordOut>,
    /// разрыв абзаца после этой строки
    pub para_break: bool,
}

#[derive(Clone, Debug)]
pub struct Page {
    pub width: usize,
    pub height: usize,
    pub skew_deg: f64,
    pub elapsed_ms: f64,
    pub num_glyphs: usize,
    pub mean_conf: f32,
    pub lines: Vec<LineOut>,
}

impl Page {
    /// Полный текст страницы (с абзацными разрывами).
    pub fn text(&self) -> String {
        let mut s = String::new();
        for (i, l) in self.lines.iter().enumerate() {
            if i > 0 {
                s.push('\n');
            }
            s.push_str(&l.text);
            if l.para_break {
                s.push('\n');
            }
        }
        s
    }
}

pub struct Engine {
    pub lang: Lang,
    templates: Templates,
    pub weights: classify::Weights,
}

impl Engine {
    /// Построение движка (рендер шаблонов ~0.5–1.5 с, делается один раз).
    pub fn new(lang: Lang) -> Engine {
        Engine {
            lang,
            templates: Templates::build(lang),
            weights: classify::Weights::default(),
        }
    }

    pub fn recognize_image(&self, img: &DynamicImage, opts: &Options) -> Page {
        let (g, w, h) = gray::to_luma(img);
        self.recognize(g, w, h, opts)
    }

    pub fn recognize_gray(&self, img: &GrayImage, opts: &Options) -> Page {
        let (w, h) = (img.width() as usize, img.height() as usize);
        self.recognize(img.clone().into_raw(), w, h, opts)
    }

    /// Основной вход: полутоновое изображение (RowMajor, w x h).
    pub fn recognize(&self, gray_in: Vec<u8>, w_in: usize, h_in: usize, opts: &Options) -> Page {
        let t0 = Instant::now();
        let mut gray = gray_in;
        let mut w = w_in;
        let mut h = h_in;
        if w == 0 || h == 0 {
            return Page {
                width: 0,
                height: 0,
                skew_deg: 0.0,
                elapsed_ms: 0.0,
                num_glyphs: 0,
                mean_conf: 0.0,
                lines: vec![],
            };
        }

        // --- 1. первичная бинаризация и оценка размеров
        let win0 = (31usize.min(w.min(h)).max(3)) | 1;
        let mut bin = gray::sauvola(&gray, w, h, win0, SAUVOLA_K);
        if gray::ink_ratio(&bin) > 0.5 {
            gray::invert(&mut bin);
        }
        let mut blobs = cc::connected_components(&bin, w, h, 4);
        let mut med_h = layout::median_char_height(&blobs);
        let dbg = std::env::var("RUSTOCR_DEBUG").is_ok();
        let mut tmark = t0;
        let mark = |dbg: bool, tmark: &mut std::time::Instant, name: &str| {
            if dbg {
                let ms = tmark.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[time] {name}: {ms:.0} мс");
                *tmark = std::time::Instant::now();
            }
        };

        // --- 2. апскейл мелкого текста / подстройка окна бинаризации
        if med_h > 0 && med_h < 15 && w * h < 30_000_000 {
            let f = (26.0 / med_h as f64).clamp(1.6, 4.0);
            let nw = ((w as f64) * f).round() as u32;
            let nh = ((h as f64) * f).round() as u32;
            if nw > 1 && nh > 1 && (nw as usize) * (nh as usize) < 80_000_000 {
                let buf = GrayImage::from_raw(w as u32, h as u32, gray).expect("raw gray");
                let resized = imageops::resize(&buf, nw, nh, imageops::FilterType::Lanczos3);
                let mut gray2 = resized.into_raw();
                // восстановление резкости после апскейла (края слишком мягкие
                // для сопоставления с чёткими шаблонами)
                unsharp3(&mut gray2, nw as usize, nh as usize, 1.35);
                gray = gray2;
                w = nw as usize;
                h = nh as usize;
                let win = (((med_h as f64) * f * 2.0).clamp(21.0, 81.0).round() as usize) | 1;
                bin = gray::sauvola(&gray, w, h, win, SAUVOLA_K);
                if gray::ink_ratio(&bin) > 0.5 {
                    gray::invert(&mut bin);
                }
                blobs = cc::connected_components(&bin, w, h, 4);
                med_h = layout::median_char_height(&blobs);
            }
        } else if med_h > 0 {
            let win2 = (((med_h as f64) * 2.2).clamp(21.0, 101.0).round() as usize) | 1;
            if (win2 as i32 - win0 as i32).abs() > 10 {
                bin = gray::sauvola(&gray, w, h, win2, SAUVOLA_K);
                if gray::ink_ratio(&bin) > 0.5 {
                    gray::invert(&mut bin);
                }
                blobs = cc::connected_components(&bin, w, h, 4);
                med_h = layout::median_char_height(&blobs);
            }
        }

        // --- 3. определение и выправление перекоса
        let mut skew = 0.0f64;
        if opts.deskew {
            skew = skew::estimate_skew(&bin, w, h);
            if skew.abs() > 0.12 {
                let (g2, w2, h2) = gray::rotate_bilinear(&gray, w, h, -skew.to_radians());
                let win = (31usize.min(w2.min(h2)).max(3)) | 1;
                let mut bin2 = gray::sauvola(&g2, w2, h2, win, SAUVOLA_K);
                if gray::ink_ratio(&bin2) > 0.5 {
                    gray::invert(&mut bin2);
                }
                let skew2 = skew::estimate_skew(&bin2, w2, h2);
                if skew2.abs() < skew.abs() {
                    // поворот помог — принимаем
                    gray = g2;
                    w = w2;
                    h = h2;
                    bin = bin2;
                    blobs = cc::connected_components(&bin, w, h, 4);
                    med_h = layout::median_char_height(&blobs);
                }
            }
        }

        // --- 4. фильтрация блобов и построение строк
        mark(dbg, &mut tmark, "бинаризация/блобы/перекос");
        let blobs = layout::filter_blobs(blobs, med_h);
        let lines = layout::build_lines(blobs, med_h);

        // --- 5. оценка медианного шага символа
        let mut all_w: Vec<i32> = Vec::new();
        for l in &lines {
            for s in segment::segment_raw(&bin, w, h, l) {
                all_w.push(s.w());
            }
        }
        let med_adv = segment::median_advance(&mut all_w);
        mark(dbg, &mut tmark, "макет/сегментация");

        // --- 6. распознавание строк (параллельно)
        let mut line_outs: Vec<LineOut> = lines
            .par_iter()
            .map(|l| self.recognize_line(&bin, &gray, w, h, l, med_h, med_adv))
            .collect();
        line_outs.retain(|l| !l.chars.is_empty());
        mark(dbg, &mut tmark, "распознавание");

        // --- 7. абзацные разрывы
        if line_outs.len() > 2 {
            let lhs: Vec<i32> = line_outs.iter().map(|l| l.bbox[3] - l.bbox[1] + 1).collect();
            let med_lh = layout::median(lhs) as f64;
            if med_lh > 0.0 {
                for i in 1..line_outs.len() {
                    let gap = line_outs[i].bbox[1] - line_outs[i - 1].bbox[3];
                    if gap as f64 > med_lh * 1.7 {
                        line_outs[i - 1].para_break = true;
                    }
                }
            }
        }

        let num_glyphs: usize = line_outs.iter().map(|l| l.chars.len()).sum();
        let mean_conf = if num_glyphs > 0 {
            line_outs
                .iter()
                .flat_map(|l| l.chars.iter())
                .map(|c| c.conf)
                .sum::<f32>()
                / num_glyphs as f32
        } else {
            0.0
        };

        Page {
            width: w,
            height: h,
            skew_deg: skew,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
            num_glyphs,
            mean_conf,
            lines: line_outs,
        }
    }

    /// Распознавание одной строки.
    fn recognize_line(
        &self,
        bin: &[bool],
        gray: &[u8],
        w: usize,
        h: usize,
        line: &Line,
        med_h: i32,
        med_adv: i32,
    ) -> LineOut {
        let baseline = layout::line_baseline(line, med_h);
        // построчная оценка x-высоты: медиана высот «ядровых» блобов строки
        // (надёжнее глобальной, особенно на мелком/смешанном тексте)
        let x_height = {
            let hs: Vec<i32> = line
                .blobs
                .iter()
                .filter(|b| {
                    let h = b.h();
                    h >= (med_h as f32 * 0.55) as i32 && h <= (med_h as f32 * 1.15) as i32
                })
                .map(|b| b.h())
                .collect();
            if hs.is_empty() {
                med_h
            } else {
                layout::median(hs)
            }
        };
        let groups = segment::segment_groups(bin, w, h, line, med_adv);

        // единица распознавания: 1 сегмент, разрезанная группа (split) или
        // слитая пара (association — «ы» = «ь» + «I» и т.п.)
        struct Unit {
            segs: Vec<Seg>,
            /// результат слитного чтения (если ассоциация решила сливать)
            merged_result: Option<CharOut>,
        }
        let flat: Vec<Seg> = groups.iter().flat_map(|g| g.iter().copied()).collect();
        // кэш классификаций одиночных сегментов (заполняется при попытках ассоциации)
        let mut pre_cache: Vec<Option<CharOut>> = vec![None; flat.len()];

        // --- ассоциация: малый зазор + узкий сегмент → пробуем слитное чтение
        let assoc_thr = ((med_adv as f32) * 0.32) as i32 + 1;
        let narrow = (med_adv as f32 * 0.5) as i32;
        let mut units: Vec<Unit> = Vec::with_capacity(flat.len());
        let mut i = 0usize;
        while i < flat.len() {
            let mut merged = false;
            if i + 1 < flat.len() {
                let gap = flat[i].gap_to(&flat[i + 1]);
                let suspicious = gap <= 2
                    || flat[i].w() <= narrow
                    || flat[i + 1].w() <= narrow
                    || (flat[i].h() - flat[i + 1].h()).abs() > (med_h / 2);
                if gap <= assoc_thr && gap >= -1 && suspicious {
                    let mseg = Seg {
                        x0: flat[i].x0,
                        x1: flat[i + 1].x1,
                        y0: flat[i].y0.min(flat[i + 1].y0),
                        y1: flat[i].y1.max(flat[i + 1].y1),
                    };
                    let ca = match &pre_cache[i] {
                        Some(c) => Some(c.clone()),
                        None => self.recognize_seg(bin, gray, w, h, &flat[i], baseline, x_height),
                    };
                    let cb = match &pre_cache[i + 1] {
                        Some(c) => Some(c.clone()),
                        None => self.recognize_seg(bin, gray, w, h, &flat[i + 1], baseline, x_height),
                    };
                    if ca.is_some() {
                        pre_cache[i] = ca.clone();
                    }
                    if cb.is_some() {
                        pre_cache[i + 1] = cb.clone();
                    }
                    // дорогую слитную гипотезу считаем только для «сомнительных» пар:
                    // обе части неуверенны либо одна из них — узкий штрих
                    let need_merged = match (&ca, &cb) {
                        (Some(a), Some(b)) => {
                            (a.conf < 0.82 || b.conf < 0.82)
                                || is_stem(a.ch)
                                || is_stem(b.ch)
                                || gap <= 2
                        }
                        _ => true,
                    };
                    let cm = if need_merged {
                        self.recognize_seg(bin, gray, w, h, &mseg, baseline, x_height)
                    } else {
                        None
                    };
                    if let (Some(a), Some(b), Some(m)) = (ca, cb, cm) {
                        let avg = (a.conf + b.conf) * 0.5;
                        // сливаем только при уверенной слитной гипотезе:
                        // мусорные сегменты (conf≈0 у частей) не склеиваем
                        let do_merge = (m.conf > 0.40
                            && m.conf >= avg + 0.15
                            && m.conf > a.conf
                            && m.conf > b.conf
                            && m.ch != a.ch
                            && m.ch != b.ch)
                            // спец-случай «ы» = «ь» + узкий штрих: штрих часто
                            // читается как «I»/«l»/«!»/«1» с завышенным conf,
                            // поэтому не требуем превосходства над частями
                            || (is_stem(a.ch) || is_stem(b.ch))
                                && (m.ch == 'ы' || m.ch == 'Ы')
                                && m.conf >= 0.30
                                && m.ch != a.ch
                                && m.ch != b.ch;
                        if std::env::var("RUSTOCR_DEBUG").is_ok() {
                            eprintln!(
                                "[assoc] «{}»+«{}»->«{}»: conf {:.2} vs {:.2}/{:.2} ({}), gap={}, w={}",
                                a.ch, b.ch, m.ch, m.conf, a.conf, b.conf,
                                if do_merge { "сливаем" } else { "нет" }, gap, mseg.w()
                            );
                        }
                        if do_merge {
                            units.push(Unit {
                                segs: vec![flat[i], flat[i + 1]],
                                merged_result: Some(m),
                            });
                            merged = true;
                        }
                    }
                }
            }
            if !merged {
                units.push(Unit {
                    segs: vec![flat[i]],
                    merged_result: None,
                });
                i += 1;
            } else {
                i += 2;
            }
        }

        // --- классификация единиц
        let mut chars: Vec<CharOut> = Vec::new();
        for (ui, unit) in units.iter().enumerate() {
            if let Some(m) = &unit.merged_result {
                chars.push(m.clone());
                continue;
            }
            if unit.segs.len() == 1 {
                // используем кэш от ассоциации, если есть
                let idx = flat.iter().position(|s| s.x0 == unit.segs[0].x0 && s.x1 == unit.segs[0].x1);
                let cached = idx.and_then(|k| pre_cache[k].clone());
                if let Some(c) = cached
                    .or_else(|| self.recognize_seg(bin, gray, w, h, &unit.segs[0], baseline, x_height))
                {
                    chars.push(c);
                }
                let _ = ui;
            } else {
                // разрезанная группа (split): сравниваем «разрезанное» и «слитное»
                let mut parts: Vec<CharOut> = Vec::new();
                for s in &unit.segs {
                    if let Some(c) = self.recognize_seg(bin, gray, w, h, s, baseline, x_height) {
                        parts.push(c);
                    }
                }
                let merged_seg = Seg {
                    x0: unit.segs.first().map(|s| s.x0).unwrap_or(0),
                    x1: unit.segs.last().map(|s| s.x1).unwrap_or(0),
                    y0: unit.segs.iter().map(|s| s.y0).min().unwrap_or(0),
                    y1: unit.segs.iter().map(|s| s.y1).max().unwrap_or(0),
                };
                let merged = self.recognize_seg(bin, gray, w, h, &merged_seg, baseline, x_height);
                let avg_conf = if parts.is_empty() {
                    0.0
                } else {
                    parts.iter().map(|c| c.conf).sum::<f32>() / parts.len() as f32
                };
                match &merged {
                    Some(m) if !parts.is_empty() && m.conf >= avg_conf - 0.12 => {
                        chars.push(m.clone())
                    }
                    Some(m) if parts.is_empty() => chars.push(m.clone()),
                    _ => chars.extend(parts),
                }
            }
        }

        // --- пробелы: зазоры меряем по чернилам ВЫШЕ базовой линии —
        // хвост «j»/запятые не съедают пробел, а перекладина «T» учитывается
        let band_bot = ((baseline + 1).max(0) as usize).min(h);
        let unit_extent = |unit: &Unit| -> (i32, i32) {
            let mut left: Option<i32> = None;
            let mut right: Option<i32> = None;
            for s in &unit.segs {
                let x0 = (s.x0.max(0) as usize).min(w.saturating_sub(1));
                let x1 = ((s.x1 as usize).min(w - 1)).max(x0);
                for x in x0..=x1 {
                    let mut has_ink = false;
                    for y in 0..band_bot {
                        if bin[y * w + x] {
                            has_ink = true;
                            break;
                        }
                    }
                    if has_ink {
                        let xi = x as i32;
                        left = Some(left.map_or(xi, |l| l.min(xi)));
                        right = Some(right.map_or(xi, |r| r.max(xi)));
                    }
                }
            }
            match (left, right) {
                (Some(l), Some(r)) => (l, r),
                _ => (
                    unit.segs.first().map(|s| s.x0).unwrap_or(0),
                    unit.segs.last().map(|s| s.x1).unwrap_or(0),
                ),
            }
        };
        let gaps: Vec<i32> = (0..units.len().saturating_sub(1))
            .map(|i| {
                let (_, cur_right) = unit_extent(&units[i]);
                let (next_left, _) = unit_extent(&units[i + 1]);
                next_left - cur_right - 1
            })
            .collect();
        let counts = segment::space_counts_from_gaps(&gaps, med_adv);

        if std::env::var("RUSTOCR_DEBUG").is_ok() {
            let txt: String = chars.iter().map(|c| c.ch).collect();
            eprintln!("[line] '{} ' gaps={:?} counts={:?}", txt, gaps, counts);
        }

        // постобработка: скрипты и цифробуквенные неоднозначности
        harmonize_scripts(&mut chars, self.lang);
        digit_context(&mut chars);

        // текст и слова
        let mut text = String::new();
        let mut words: Vec<WordOut> = Vec::new();
        let mut cur: Vec<CharOut> = Vec::new();
        fn flush(cur: &mut Vec<CharOut>, words: &mut Vec<WordOut>) {
            if cur.is_empty() {
                return;
            }
            let bbox = [
                cur.iter().map(|c| c.bbox[0]).min().unwrap(),
                cur.iter().map(|c| c.bbox[1]).min().unwrap(),
                cur.iter().map(|c| c.bbox[2]).max().unwrap(),
                cur.iter().map(|c| c.bbox[3]).max().unwrap(),
            ];
            let text: String = cur.iter().map(|c| c.ch).collect();
            let conf = cur.iter().map(|c| c.conf).sum::<f32>() / cur.len() as f32;
            words.push(WordOut { text, conf, bbox });
            cur.clear();
        }
        for (i, c) in chars.iter().enumerate() {
            text.push(c.ch);
            cur.push(c.clone());
            if i < counts.len() {
                for _ in 0..counts[i] {
                    text.push(' ');
                    flush(&mut cur, &mut words);
                }
            }
        }
        flush(&mut cur, &mut words);

        let conf = if chars.is_empty() {
            0.0
        } else {
            chars.iter().map(|c| c.conf).sum::<f32>() / chars.len() as f32
        };
        LineOut {
            text,
            conf,
            bbox: [line.x0, line.y0, line.x1, line.y1],
            chars,
            words,
            para_break: false,
        }
    }

    /// Распознавание одного сегмента (глифа).
    fn recognize_seg(
        &self,
        bin: &[bool],
        gray: &[u8],
        w: usize,
        h: usize,
        seg: &Seg,
        baseline: i32,
        x_height: i32,
    ) -> Option<CharOut> {
        if w == 0 || h == 0 {
            return None;
        }
        let x0 = (seg.x0.max(0) as usize).min(w - 1);
        let x1 = ((seg.x1 as usize).min(w - 1)).max(x0);
        let y0 = (seg.y0.max(0) as usize).min(h - 1);
        let y1 = ((seg.y1 as usize).min(h - 1)).max(y0);
        let pw = x1 - x0 + 1;
        let ph = y1 - y0 + 1;
        if pw == 0 || ph == 0 || pw > 400 || ph > 400 {
            return None;
        }
        // площадь чернил в сегменте
        let mut area = 0usize;
        for y in y0..=y1 {
            let row = y * w;
            for x in x0..=x1 {
                if bin[row + x] {
                    area += 1;
                }
            }
        }
        if area < 4 {
            return None;
        }
        // полутоновый патч + бинарная маска (стабильный tight-bbox)
        let mut patch = vec![255u8; pw * ph];
        let mut mask = vec![false; pw * ph];
        for y in 0..ph {
            let srow = (y0 + y) * w;
            let drow = y * pw;
            for x in 0..pw {
                patch[drow + x] = gray[srow + x0 + x];
                mask[drow + x] = bin[srow + x0 + x];
            }
        }
        let g = features::extract(
            &patch,
            pw,
            ph,
            Some(&mask),
            baseline as f64 - y0 as f64,
            x_height as f64,
        )?;
        let r = classify::classify(&g, &self.templates, &self.weights)?;
        Some(CharOut {
            ch: r.ch,
            conf: r.conf,
            bbox: [seg.x0, seg.y0, seg.x1, seg.y1],
        })
    }
}

// ---------------------------------------------------------------------------
// Постобработка: скрипты и цифро-буквенные неоднозначности
// ---------------------------------------------------------------------------

fn is_cyrillic(c: char) -> bool {
    ('А'..='я').contains(&c) || c == 'Ё' || c == 'ё'
}

fn is_latin(c: char) -> bool {
    c.is_ascii_alphabetic()
}

/// Узкий «штриховой» символ — типичная правая половина «ы» при разбиении.
fn is_stem(c: char) -> bool {
    matches!(c, 'I' | 'l' | '1' | '!' | '|')
}

/// Символ-«двойник» с совпадающей формой в другом алфавите.
fn shape_twin(c: char) -> Option<char> {
    Some(match c {
        'A' => 'А',
        'А' => 'A',
        'B' => 'В',
        'В' => 'B',
        'C' => 'С',
        'С' => 'C',
        'E' => 'Е',
        'Е' => 'E',
        'H' => 'Н',
        'Н' => 'H',
        'K' => 'К',
        'К' => 'K',
        'M' => 'М',
        'М' => 'M',
        'O' => 'О',
        'О' => 'O',
        'P' => 'Р',
        'Р' => 'P',
        'T' => 'Т',
        'Т' => 'T',
        'X' => 'Х',
        'Х' => 'X',
        'a' => 'а',
        'а' => 'a',
        'c' => 'с',
        'с' => 'c',
        'e' => 'е',
        'е' => 'e',
        'o' => 'о',
        'о' => 'o',
        'p' => 'р',
        'р' => 'p',
        'x' => 'х',
        'х' => 'x',
        'y' => 'у',
        'у' => 'y',
        _ => return None,
    })
}

/// «Мягкие» пары: похожие, но не идентичные формы — применяются только при
/// низкой уверенности и сильном большинстве алфавита в строке.
fn soft_twin(c: char) -> Option<char> {
    Some(match c {
        'т' => 'T',
        'T' => 'т',
        'г' => 'r',
        'r' => 'г',
        'и' => 'u',
        'u' => 'и',
        _ => return None,
    })
}

/// Приведение «двойников» к доминирующему алфавиту строки.
fn harmonize_scripts(chars: &mut [CharOut], lang: Lang) {
    if lang != Lang::Both || chars.is_empty() {
        return;
    }
    let mut lat = 0f32;
    let mut cyr = 0f32;
    for c in chars.iter() {
        if shape_twin(c.ch).is_some() {
            continue; // неоднозначные не голосуют
        }
        if is_latin(c.ch) {
            lat += c.conf.max(0.05);
        } else if is_cyrillic(c.ch) {
            cyr += c.conf.max(0.05);
        }
    }
    if lat == 0.0 && cyr == 0.0 {
        return;
    }
    let want_latin = lat >= cyr;
    let strong = (lat.max(cyr)) > 2.0 * lat.min(cyr).max(1e-6);
    for c in chars.iter_mut() {
        if let Some(t) = shape_twin(c.ch) {
            if is_latin(t) == want_latin {
                c.ch = t;
            }
        } else if strong && c.conf < 0.55 {
            // мягкие пары: только при сильном большинстве и низкой уверенности
            if let Some(t) = soft_twin(c.ch) {
                if is_latin(t) == want_latin {
                    c.ch = t;
                }
            }
        }
    }
}

/// Разрешение цифро-буквенных неоднозначностей по соседям
/// (0/О, 1/l, 2/З, 5/S, 8/В).
fn digit_context(chars: &mut [CharOut]) {
    let n = chars.len();
    for i in 0..n {
        let c = chars[i].ch;
        let pair: Option<(char, char)> = match c {
            '0' | 'O' | 'О' => Some(('0', 'О')),
            'o' | 'о' => Some(('0', 'о')),
            '1' | 'l' | 'I' => Some(('1', 'l')),
            '2' | 'Z' | 'З' => Some(('2', 'З')),
            '5' | 'S' => Some(('5', 'S')),
            '8' | 'B' | 'В' => Some(('8', 'В')),
            _ => None,
        };
        let Some((d, l)) = pair else { continue };
        if c != d && c != l {
            continue;
        }
        let prev = if i > 0 { Some(chars[i - 1].ch) } else { None };
        let next = if i + 1 < n { Some(chars[i + 1].ch) } else { None };
        let mut dvotes = 0i32;
        let mut lvotes = 0i32;
        for nb in [prev, next].into_iter().flatten() {
            if nb.is_ascii_digit() || matches!(nb, '.' | ',') {
                dvotes += 1;
            } else if nb.is_alphabetic() {
                lvotes += 1;
            }
        }
        if dvotes >= 2 {
            chars[i].ch = d;
        } else if lvotes >= 2 {
            chars[i].ch = l;
        }
    }
}
