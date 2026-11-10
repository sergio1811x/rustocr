//! Простой JSON-вывод без внешних зависимостей.

use crate::ocr::Page;

/// Экранирование строки для JSON (UTF-8 содержимое пишется как есть).
pub fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

fn f2(v: f32) -> String {
    format!("{:.3}", v)
}

/// Полная выдача страницы: текст + строки + слова + символы с координатами.
pub fn page_to_json(page: &Page) -> String {
    let mut s = String::with_capacity(4096);
    s.push_str("{\"text\":\"");
    s.push_str(&esc(&page.text()));
    s.push_str("\",\"width\":");
    s.push_str(&page.width.to_string());
    s.push_str(",\"height\":");
    s.push_str(&page.height.to_string());
    s.push_str(",\"skew_deg\":");
    s.push_str(&format!("{:.2}", page.skew_deg));
    s.push_str(",\"num_glyphs\":");
    s.push_str(&page.num_glyphs.to_string());
    s.push_str(",\"mean_conf\":");
    s.push_str(&f2(page.mean_conf));
    s.push_str(",\"elapsed_ms\":");
    s.push_str(&format!("{:.1}", page.elapsed_ms));
    s.push_str(",\"lines\":[");
    for (li, line) in page.lines.iter().enumerate() {
        if li > 0 {
            s.push(',');
        }
        s.push_str("{\"text\":\"");
        s.push_str(&esc(&line.text));
        s.push_str("\",\"conf\":");
        s.push_str(&f2(line.conf));
        s.push_str(",\"bbox\":[");
        s.push_str(&line.bbox.map(|v| v.to_string()).join(","));
        s.push_str("],\"words\":[");
        for (wi, wd) in line.words.iter().enumerate() {
            if wi > 0 {
                s.push(',');
            }
            s.push_str("{\"text\":\"");
            s.push_str(&esc(&wd.text));
            s.push_str("\",\"conf\":");
            s.push_str(&f2(wd.conf));
            s.push_str(",\"bbox\":[");
            s.push_str(&wd.bbox.map(|v| v.to_string()).join(","));
            s.push_str("]}");
        }
        s.push_str("],\"chars\":[");
        for (ci, ch) in line.chars.iter().enumerate() {
            if ci > 0 {
                s.push(',');
            }
            s.push_str("{\"ch\":\"");
            s.push_str(&esc(&ch.ch.to_string()));
            s.push_str("\",\"conf\":");
            s.push_str(&f2(ch.conf));
            s.push_str(",\"bbox\":[");
            s.push_str(&ch.bbox.map(|v| v.to_string()).join(","));
            s.push_str("]}");
        }
        s.push_str("]}");
    }
    s.push_str("]}");
    s
}
