//! Текстовый слой PDF: «цифровые» сметы (Гранд-Смета, 1С и т.п.) уже
//! содержат идеальный текст с координатами — читать их OCR-ом значит
//! вносить ошибки в то, что и так напечатано. Гибрид: страница с плотным
//! текстовым слоем берётся из `pdftotext -bbox-layout` (poppler, локально,
//! без сети), страница-скан — обычным OCR. Результат собирается в те же
//! `Block`/`OcrResult`, что выходят из детектора: таблицы, семантика и
//! валидация работают без изменений.
//!
//! Тонкости:
//!   • координаты pdftotext — в пунктах PDF (72/дюйм), блоки OCR — в
//!     пикселях растеризованной страницы; масштаб считается по размерам
//!     картинки и точен при любом DPI;
//!   • слова группируются в визуальные строки по Y-центрам, внутри строки
//!     режутся на «чанки» по X-просветам — это имитирует боксы детектора
//!     (широкие пробелы выключенного текста детектор тоже не склеивает);
//!   • текст слоя помечается `lang = "txt"` и уверенностью 1.0: перечтение
//!     разрезанных ячеек его пропускает — идеальный текст не перечитывают
//!     заново OCR-ом;
//!   • NBSP (неразрывный пробел в числах «1 545,38») нормализуется в
//!     обычный пробел, дальше — та же цепочка пост-обработки, что у OCR.

use crate::geom::Pt;
use crate::pipeline::{Block, OcrResult};
use anyhow::{bail, Context, Result};
use std::path::Path;
use std::process::Command;

const EXTRACT_TIMEOUT_S: u64 = 120;

/// Слово текстового слоя в координатах пунктов PDF.
#[derive(Clone, Debug)]
pub struct WordPt {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub text: String,
}

/// Текстовый слой одной страницы.
#[derive(Clone, Debug)]
pub struct PageText {
    /// размер страницы в пунктах (0 — не удалось прочитать из вывода)
    pub w_pt: f32,
    pub h_pt: f32,
    pub words: Vec<WordPt>,
}

/// Достать текстовый слой диапазона страниц [first, last].
/// Ошибка или пустой вывод — не фатально: вызывающий просто уходит в OCR.
pub fn extract(path: &Path, first: u32, last: u32) -> Result<Vec<PageText>> {
    let out = Command::new("timeout")
        .arg(EXTRACT_TIMEOUT_S.to_string())
        .arg("pdftotext")
        .arg("-f")
        .arg(first.to_string())
        .arg("-l")
        .arg(last.to_string())
        .arg("-bbox-layout")
        .arg(path)
        .arg("-")
        .output()
        .context("не запустить pdftotext (нужен poppler-utils)")?;
    if !out.status.success() {
        bail!(
            "pdftotext вернул код {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let html = String::from_utf8_lossy(&out.stdout).into_owned();
    Ok(parse_bbox_html(&html, first, last))
}

impl PageText {
    /// Достаточно ли текста, чтобы доверять слою (иначе — OCR страницы).
    /// Пороги подобраны по корпусу: титульные листы и строки шапки
    /// проходят, «пустые» страницы и огрызки — нет.
    pub fn usable(&self) -> bool {
        if self.w_pt <= 0.0 || self.h_pt <= 0.0 || self.words.is_empty() {
            return false;
        }
        let chars: usize = self
            .words
            .iter()
            .map(|w| w.text.chars().filter(|c| !c.is_whitespace()).count())
            .sum();
        self.words.len() >= 25 && chars >= 150
    }

    /// Собрать OcrResult в пиксельных координатах растеризованной страницы
    /// `img_w × img_h`. Слово → визуальная строка (по Y) → чанк (по
    /// X-просвету) → `Block` с `lang = "txt"`.
    pub fn to_ocr_result(&self, img_w: u32, img_h: u32) -> OcrResult {
        let t0 = std::time::Instant::now();
        // `/Rotate 90|270`: pdftotext отдаёт координаты слов в системе
        // ПОВЁРНУТОЙ (видимой) страницы, но width/height тега <page> —
        // размеры неповёрнутой. Рендер pdftoppm тоже в видимой системе,
        // поэтому при несовпадении пропорций размеры меняем местами.
        let (ew, eh) = if (img_w.max(1) as f32 / img_h.max(1) as f32
            - self.w_pt.max(1.0) / self.h_pt.max(1.0))
        .abs()
            < 0.05
        {
            (self.w_pt, self.h_pt)
        } else {
            (self.h_pt, self.w_pt)
        };
        let sx = img_w as f32 / ew.max(1.0);
        let sy = img_h as f32 / eh.max(1.0);

        // слова в пикселях, без пустых
        let mut ws: Vec<(f32, f32, f32, f32, String)> = self
            .words
            .iter()
            .filter_map(|w| {
                let t = w.text.replace('\u{a0}', " ");
                if t.trim().is_empty() {
                    return None;
                }
                let x0 = (w.x0 * sx).max(0.0);
                let y0 = (w.y0 * sy).max(0.0);
                Some((x0, y0, w.x1 * sx, w.y1 * sy, t))
            })
            .collect();

        let med_h = {
            let mut hs: Vec<f32> = ws.iter().map(|w| w.3 - w.1).collect();
            hs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            hs.get(hs.len() / 2).copied().unwrap_or(1.0).max(1.0)
        };

        // визуальные строки по Y-центрам (допуск как у скелета таблиц)
        ws.sort_by(|a, b| {
            let ya = (a.1 + a.3) / 2.0;
            let yb = (b.1 + b.3) / 2.0;
            ya.partial_cmp(&yb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let tol = 0.6 * med_h;
        let mut rows: Vec<Vec<usize>> = Vec::new();
        let mut rows_cy: Vec<f32> = Vec::new();
        for i in 0..ws.len() {
            let cy = (ws[i].1 + ws[i].3) / 2.0;
            let joined = rows
                .last_mut()
                .zip(rows_cy.last_mut())
                .map(|(m, rcy)| {
                    if (cy - *rcy).abs() <= tol {
                        m.push(i);
                        *rcy = (*rcy * (m.len() - 1) as f32 + cy) / m.len() as f32;
                        true
                    } else {
                        false
                    }
                })
                .unwrap_or(false);
            if !joined {
                rows.push(vec![i]);
                rows_cy.push(cy);
            }
        }

        // строки → чанки по X-просветам → блоки
        let gap_thr = (0.9 * med_h).max(8.0);
        let mut blocks: Vec<Block> = Vec::new();
        let mut lines: Vec<String> = Vec::new();
        for row in &rows {
            let mut idx: Vec<usize> = row.clone();
            idx.sort_by(|&a, &b| ws[a].0.partial_cmp(&ws[b].0).unwrap_or(std::cmp::Ordering::Equal));
            let mut chunks: Vec<Vec<usize>> = Vec::new();
            for &i in &idx {
                let new_chunk = match chunks.last() {
                    Some(prev) => ws[i].0 - ws[*prev.last().unwrap()].2 > gap_thr,
                    None => true,
                };
                if new_chunk {
                    chunks.push(vec![i]);
                } else {
                    chunks.last_mut().unwrap().push(i);
                }
            }
            let mut row_text: Vec<String> = Vec::new();
            for ch in &chunks {
                let x0 = ws[*ch.first().unwrap()].0;
                let x1 = ws[*ch.last().unwrap()].2;
                let y0 = ch.iter().map(|&i| ws[i].1).fold(f32::INFINITY, f32::min);
                let y1 = ch.iter().map(|&i| ws[i].3).fold(f32::NEG_INFINITY, f32::max);
                let raw: Vec<&str> = ch.iter().map(|&i| ws[i].4.trim()).collect();
                let text = crate::pipeline::normalize_block_text(&raw.join(" "));
                if text.trim().is_empty() {
                    continue;
                }
                row_text.push(text.trim().to_string());
                blocks.push(Block {
                    text: text.trim().to_string(),
                    confidence: 1.0,
                    lang: "txt",
                    quad: [
                        Pt { x: x0, y: y0 },
                        Pt { x: x1, y: y0 },
                        Pt { x: x1, y: y1 },
                        Pt { x: x0, y: y1 },
                    ],
                });
            }
            if !row_text.is_empty() {
                lines.push(row_text.join(" "));
            }
        }

        OcrResult {
            width: img_w,
            height: img_h,
            text: lines.join("\n"),
            blocks,
            skipped: 0,
            det_ms: 0,
            rec_ms: t0.elapsed().as_millis() as u64,
            total_ms: t0.elapsed().as_millis() as u64,
        }
    }
}

/// Разбор вывода `pdftotext -bbox-layout` (XHTML: <page><flow><line><word>).
/// Формат машинный и стабильный — достаточно линейного сканера тегов.
fn parse_bbox_html(html: &str, first: u32, last: u32) -> Vec<PageText> {
    let b = html.as_bytes();
    let want = (last.saturating_sub(first) + 1) as usize;
    let mut pages: Vec<PageText> = Vec::with_capacity(want);
    let mut i = 0usize;
    while i < b.len() {
        if b[i] != b'<' {
            i += 1;
            continue;
        }
        if let Some(end) = find_byte(b, i, b'>') {
            if starts_with(b, i, b"<page") {
                let tag = &b[i..end];
                pages.push(PageText {
                    w_pt: attr_f32(tag, b"width").unwrap_or(0.0),
                    h_pt: attr_f32(tag, b"height").unwrap_or(0.0),
                    words: Vec::new(),
                });
            } else if starts_with(b, i, b"<word") {
                let tag = &b[i..end];
                let content_start = end + 1;
                let content_end = find_byte(b, content_start, b'<').unwrap_or(b.len());
                let raw = String::from_utf8_lossy(&b[content_start..content_end]).into_owned();
                if let Some(page) = pages.last_mut() {
                    page.words.push(WordPt {
                        x0: attr_f32(tag, b"xMin").unwrap_or(0.0),
                        y0: attr_f32(tag, b"yMin").unwrap_or(0.0),
                        x1: attr_f32(tag, b"xMax").unwrap_or(0.0),
                        y1: attr_f32(tag, b"yMax").unwrap_or(0.0),
                        text: decode_entities(&raw),
                    });
                }
                i = content_end;
                continue;
            }
            i = end + 1;
        } else {
            break;
        }
    }
    // страниц меньше, чем просили (крайний диапазон) — дополняем пустыми:
    // пустой слой => unusable => страница уходит в OCR
    while pages.len() < want {
        pages.push(PageText {
            w_pt: 0.0,
            h_pt: 0.0,
            words: Vec::new(),
        });
    }
    pages
}

fn find_byte(b: &[u8], from: usize, needle: u8) -> Option<usize> {
    b.iter().skip(from).position(|&c| c == needle).map(|p| p + from)
}

fn starts_with(b: &[u8], at: usize, prefix: &[u8]) -> bool {
    b.len() >= at + prefix.len() && &b[at..at + prefix.len()] == prefix
}

/// Значение числового атрибута `name="..."` внутри тега.
fn attr_f32(tag: &[u8], name: &[u8]) -> Option<f32> {
    let mut pat = Vec::with_capacity(name.len() + 2);
    pat.extend_from_slice(name);
    pat.extend_from_slice(b"=\"");
    let start = tag
        .windows(pat.len())
        .position(|w| w == pat.as_slice())?
        + pat.len();
    let end = find_byte(tag, start, b'"')?;
    std::str::from_utf8(&tag[start..end])
        .ok()?
        .parse()
        .ok()
}

/// XML-entities: &amp; &lt; &gt; &quot; &apos; &#NN; &#xHH;
fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(p) = rest.find('&') {
        out.push_str(&rest[..p]);
        rest = &rest[p..];
        let end = rest.find(';');
        let (ent, tail) = match end {
            Some(e) if e <= 12 => (&rest[1..e], &rest[e + 1..]),
            _ => {
                // не entity — просто амперсанд
                out.push('&');
                rest = &rest[1..];
                continue;
            }
        };
        let rep = match ent {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => {
                if let Some(hex) = ent.strip_prefix("#x").or_else(|| ent.strip_prefix("#X")) {
                    u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
                } else if let Some(dec) = ent.strip_prefix('#') {
                    dec.parse::<u32>().ok().and_then(char::from_u32)
                } else {
                    None
                }
            }
        };
        match rep {
            Some(c) => out.push(c),
            None => {
                out.push_str(&rest[..end.unwrap() + 1]);
            }
        }
        rest = tail;
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page_text_from(words: Vec<WordPt>) -> PageText {
        PageText {
            w_pt: 595.0,
            h_pt: 842.0,
            words,
        }
    }

    fn word(text: &str, x0: f32, y0: f32, x1: f32, y1: f32) -> WordPt {
        WordPt {
            x0,
            y0,
            x1,
            y1,
            text: text.to_string(),
        }
    }

    /// Разбор машинного XHTML pdftotext: страницы, слова, entities.
    #[test]
    fn parse_bbox_output() {
        let html = concat!(
            "<html><body><doc>",
            "<page width=\"595.30\" height=\"841.90\">",
            "<flow><line xMin=\"40\" yMin=\"20\" xMax=\"200\" yMax=\"32\">",
            "<word xMin=\"40\" yMin=\"20\" xMax=\"90\" yMax=\"32\">Локальный</word>",
            "<word xMin=\"95\" yMin=\"20\" xMax=\"200\" yMax=\"32\">сметный &amp; расчёт</word>",
            "</line></flow>",
            "</page>",
            "<page width=\"595.30\" height=\"841.90\">",
            "<flow><line>",
            "<word xMin=\"10\" yMin=\"5\" xMax=\"20\" yMax=\"15\">&#8470;</word>",
            "</line></flow>",
            "</page>",
            "</doc></body></html>",
        );
        let pages = parse_bbox_html(html, 1, 2);
        assert_eq!(pages.len(), 2);
        assert!((pages[0].w_pt - 595.3).abs() < 0.01);
        assert_eq!(pages[0].words.len(), 2);
        assert_eq!(pages[0].words[1].text, "сметный & расчёт");
        assert_eq!(pages[1].words[0].text, "\u{2116}");
        // диапазон длиннее вывода — дополнен пустыми страницами
        let pages = parse_bbox_html(html, 1, 4);
        assert_eq!(pages.len(), 4);
        assert!(!pages[3].usable());
    }

    /// Числовые атрибуты ищутся только внутри своего тега.
    #[test]
    fn attr_parsing() {
        let tag = br#"<word xMin="38.3" yMin="49.0" xMax="45.1" yMax="57.2">"#;
        assert_eq!(attr_f32(tag, b"xMin"), Some(38.3));
        assert_eq!(attr_f32(tag, b"yMax"), Some(57.2));
        assert_eq!(attr_f32(tag, b"nope"), None);
    }

    /// Entities, включая числовые и «не-entity» амперсанды.
    #[test]
    fn entity_decoding() {
        assert_eq!(decode_entities("A &amp; B"), "A & B");
        assert_eq!(decode_entities("&lt;смета&gt;"), "<смета>");
        assert_eq!(decode_entities("10&#160;500"), "10\u{a0}500");
        assert_eq!(decode_entities("R&D; нет"), "R&D; нет");
        assert_eq!(decode_entities("&#x414;"), "\u{414}");
    }

    /// Страница с достаточным слоем — usable; огрызок — нет.
    #[test]
    fn usability_thresholds() {
        let words: Vec<WordPt> = (0..30)
            .map(|i| word(&format!("слово{}", i), 10.0, 10.0 + i as f32, 40.0, 20.0 + i as f32))
            .collect();
        let pt = page_text_from(words);
        assert!(pt.usable());
        let thin = page_text_from(vec![word("мало", 0.0, 0.0, 10.0, 10.0)]);
        assert!(!thin.usable());
    }

    /// Слой → OcrResult: строки по Y, чанки по X-просветам, NBSP,
    /// нормализация чисел и пост-обработка как у OCR.
    #[test]
    fn to_ocr_result_groups_and_chunks() {
        // три «строки» по 2 чанка: колонка шифра и колонка суммы
        let mut words = Vec::new();
        for (ri, y) in [20.0, 40.0, 60.0].iter().enumerate() {
            let y1 = y + 10.0;
            words.push(word(&format!("ТЕР{:02}-1", ri + 1), 40.0, *y, 90.0, y1));
            // просвет 90 → 200: новый чанк; внутри — NBSP в числе
            words.push(word(&format!("1\u{a0}0{:02},50", ri + 1), 200.0, *y, 260.0, y1));
        }
        let pt = page_text_from(words);
        let r = pt.to_ocr_result(1190, 1684); // 595×842 → ×2
        assert_eq!(r.blocks.len(), 6, "по 2 чанка на строку");
        assert_eq!(r.width, 1190);
        // координаты масштабированы ×2
        let (x0, x1) = {
            let q = &r.blocks[0].quad;
            let xs = [q[0].x, q[1].x, q[2].x, q[3].x];
            (
                xs.iter().copied().fold(f32::INFINITY, f32::min),
                xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            )
        };
        assert!((x0 - 80.0).abs() < 0.01, "x0={}", x0);
        assert!((x1 - 180.0).abs() < 0.01, "x1={}", x1);
        // NBSP → обычный пробел, число перегруппировано в «1 001,50»
        assert_eq!(r.blocks[1].text, "1 001,50");
        // блоки слоя помечены и не перечитываются
        assert_eq!(r.blocks[0].lang, "txt");
        assert_eq!(r.blocks[0].confidence, 1.0);
        // текст по визуальным строкам
        assert_eq!(r.text.lines().count(), 3);
        assert!(r.text.contains("ТЕР01-1 1 001,50"));
    }

    /// Слой с выровненным по ширине именем: широкий пробел режет чанк,
    /// но в одной полосе куски склеиваются пробелом (как у детектора).
    #[test]
    fn justified_name_splits_into_chunks() {
        let mut words = Vec::new();
        // колонка №
        words.push(word("1", 40.0, 100.0, 48.0, 110.0));
        // имя: слова с большими межсловными пробелами (выключка)
        words.push(word("Устройство", 120.0, 100.0, 190.0, 110.0));
        words.push(word("железобетонных", 340.0, 100.0, 430.0, 110.0));
        // сумма
        words.push(word("2\u{a0}500,00", 500.0, 100.0, 560.0, 110.0));
        let pt = page_text_from(words);
        let r = pt.to_ocr_result(595, 842);
        assert_eq!(r.blocks.len(), 4, "каждое слово имени — свой чанк");
    }

    /// Полный конвейер по слою: как тест pipeline_from_blocks_to_document,
    /// но вход — текстовый слой. Перечтение разрезанных ячеек не трогает
    /// блоки «txt» — colspan объединённой ячейки выживает.
    #[test]
    fn estimate_pipeline_from_text_layer() {
        let mut words = Vec::new();
        // титульная строка (7 слов одним чанком — не табличная строка)
        for (i, w) in [
            "Локальный",
            "сметный",
            "расчёт",
            "№1",
            "на",
            "общестроительные",
            "работы",
        ]
        .iter()
        .enumerate()
        {
            let x = 20.0 + i as f32 * 45.0;
            words.push(word(w, x, 20.0, x + 40.0, 32.0));
        }
        // строка заголовков таблицы: № | наименование | сумма
        words.push(word("№", 20.0, 80.0, 60.0, 92.0));
        words.push(word("наименование", 120.0, 80.0, 240.0, 92.0));
        words.push(word("сумма", 500.0, 80.0, 580.0, 92.0));
        // раздел: слова вплотную — один чанк, не таблица
        words.push(word("Раздел", 20.0, 130.0, 80.0, 142.0));
        words.push(word("1.", 85.0, 130.0, 100.0, 142.0));
        words.push(word("Стены", 105.0, 130.0, 150.0, 142.0));
        // позиции (y-строки): № | имя | ед. | сумма
        let rows = [
            ("Кладка", "200\u{a0}000"),
            ("Кровля", "300\u{a0}000"),
            ("Отделка", "150\u{a0}000"),
            ("Утепление", "50\u{a0}000"),
        ];
        for (ri, (name, sum)) in rows.iter().enumerate() {
            let y = 170.0 + ri as f32 * 40.0;
            let y1 = y + 12.0;
            words.push(word(&(ri + 1).to_string(), 20.0, y, 32.0, y1));
            words.push(word(name, 120.0, y, 180.0, y1));
            words.push(word("м3", 430.0, y, 450.0, y1));
            words.push(word(sum, 500.0, y, 580.0, y1));
        }
        // итог по смете: подпись одним чанком в колонке наименования,
        // число — в колонке суммы (объединённая ячейка не перечитывается)
        words.push(word("Итого", 120.0, 350.0, 160.0, 362.0));
        words.push(word("по", 165.0, 350.0, 180.0, 362.0));
        words.push(word("смете", 185.0, 350.0, 230.0, 362.0));
        words.push(word("700\u{a0}000", 500.0, 350.0, 580.0, 362.0));
        let pt = page_text_from(words);
        assert!(pt.usable(), "слой достаточно плотный");
        let r = pt.to_ocr_result(595, 842);
        let doc = crate::estimate::parse_estimate(None, &[crate::estimate::PageInput {
            page: 1,
            img: None,
            result: &r,
        }]);
        assert_eq!(doc.sections.len(), 1, "раздел распознан");
        assert_eq!(doc.sections[0].positions.len(), 4, "все позиции на месте");
        assert_eq!(doc.sections[0].positions[0].name, "Кладка");
        assert_eq!(
            doc.totals.total.as_ref().map(|n| n.value),
            Some(700000.0),
            "итог сметы = 700 000"
        );
        assert!(
            !doc.validation.errors.iter().any(|e| e.kind == "doc_total_mismatch"),
            "арифметика сходится: {:?}",
            doc.validation.errors
        );
    }
}
