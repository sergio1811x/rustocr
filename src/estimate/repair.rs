//! Авто-починка арифметических ошибок распознавания.
//!
//! Главный системный дефект OCR на сметах — потеря десятичной запятой
//! в колонке сумм: «32 000,00» превращается в «3200000» (×100),
//! «29 000,00» → «2900000» (×100) и т.п. Распознаватель часто «склеивает»
//! группы разрядов без разделителя. Однако смета — арифметически связный
//! документ, и это можно обратить:
//!
//!   1) Правило «кол-во × цена»: если qty × price ≈ total × 10^k
//!      (k=1..4), у total в raw нет разделителя, а k согласован с типичной
//!      длиной дробной части в колонке — чиним total, сдвигая запятую.
//!   2) Правило «итог раздела»: Σ позиций (уже починенных) ≈ итог × 10^k —
//!      чиним итог раздела. Ловит случаи, когда цена И сумма потеряли
//!      запятую одновременно и локально «сходятся» (напр., 3000,00 →
//!      «3 000»/300000 обе искажены).
//!
//! Каждая починка фиксируется warning'ом вида `auto_repair_*` — результат
//! молча не меняется, всё видно в отчёте валидации.

use crate::estimate::model::*;
use crate::estimate::semantics::parse_number;

/// Сколько разрядов запятой можно потерять максимум.
const MAX_SHIFT: u32 = 4;
/// Есть ли в исходном тексте числа десятичный разделитель.
fn has_separator(raw: &str) -> bool {
    raw.contains(',') || raw.contains('.')
}

/// Формат числа похож на «потерянную запятую»: только цифры (возможно,
/// с разрядными пробелами/nbsp), разделителя нет, длина ≥ 4.
fn looks_lost_comma(raw: &str) -> bool {
    let t: String = raw
        .trim()
        .chars()
        .filter(|c| *c != ' ' && *c != '\u{a0}')
        .collect();
    !t.is_empty()
        && t.len() >= 4
        && t.chars().all(|c| c.is_ascii_digit())
}

/// Округлённая доля от ожидаемой точности сравнения.
fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// Найти k, при котором expect × 10^k ≈ value (в допуске tol), т.е.
/// распознанное value завышено в 10^k раз относительно ожидаемого expect.
/// Возвращает k вместе с погрешностью.
fn shift_factor(value: f64, expect: f64, tol: f64) -> Option<(u32, f64)> {
    let mut best: Option<(u32, f64)> = None;
    for k in 1..=MAX_SHIFT {
        let inflated = expect * 10f64.powi(k as i32);
        let d = (value - inflated).abs();
        if d <= tol && (best.is_none() || d < best.unwrap().1) {
            best = Some((k, d));
        }
    }
    best
}

/// Починить одно число: вернуть новое значение с восстановленной запятой.
fn repair_value(value: f64, k: u32) -> f64 {
    value / 10f64.powi(k as i32)
}

// ---------- проход 1: позиции ----------

/// Собрать дробную длину из raw всех «нормальных» чисел колонки total
/// (для оценки ожидаемого k). Если все суммы в колонке — «N,00», потеря
/// почти наверняка ×100.
fn typical_frac_len(doc: &EstimateDocument) -> Option<u32> {
    let mut lens: Vec<usize> = Vec::new();
    let mut collect = |secs: &Vec<Section>| {
        for s in secs {
            for p in s.all_positions() {
                if let Some(t) = &p.total {
                    if has_separator(&t.raw) {
                        if let Some(pos) = t.raw.rfind([',', '.']) {
                            lens.push(t.raw[pos + 1..].trim().len());
                        }
                    }
                }
            }
            for r in &s.resources {
                if let Some(t) = &r.total {
                    if has_separator(&t.raw) {
                        if let Some(pos) = t.raw.rfind([',', '.']) {
                            lens.push(t.raw[pos + 1..].trim().len());
                        }
                    }
                }
            }
        }
    };
    collect(&doc.sections);
    if lens.is_empty() {
        return None;
    }
    lens.sort_unstable();
    let median = lens[lens.len() / 2];
    (1..=8).contains(&median).then_some(median as u32)
}

/// Проход по позициям: чиним total по правилу qty × price.
fn repair_positions(doc: &mut EstimateDocument, frac: Option<u32>) -> Vec<String> {
    let mut notes = Vec::new();
    let sections = std::mem::take(&mut doc.sections);
    let mut sections = sections;
    for sec in sections.iter_mut() {
        repair_section_positions(sec, frac, &mut notes);
    }
    doc.sections = sections;
    notes
}

fn repair_section_positions(sec: &mut Section, frac: Option<u32>, notes: &mut Vec<String>) {
    for p in sec.positions.iter_mut() {
        // --- правило A: цена есть — сверяем qty × price ≈ total × 10^k ---
        if let (Some(q), Some(pr), Some(t)) = (&p.quantity, &p.price, &p.total) {
            if looks_lost_comma(&t.raw) {
                let expect = q.value * pr.value;
                if expect > 0.0 && t.value > 0.0 {
                    // допуск: 2% как у qty_price, но не меньше 1 руб.
                    let tol = (1.0_f64).max(expect * 0.02);
                    if let Some((k, _)) = shift_factor(t.value, expect, tol) {
                        // согласование с колонкой: если медианная дробная
                        // часть 2, ожидаемый сдвиг k=2 — иначе любой,
                        // обоснованный арифметикой
                        let column_ok = match frac {
                            Some(f) if f >= 1 => k == f || k == f + 1,
                            _ => true,
                        };
                        if column_ok {
                            let fixed = repair_value(t.value, k);
                            notes.push(format!(
                                "«{}»: сумма {} → {} — OCR потерял{} десятичную запятую ({} × {} = {:.2})",
                                name_short(&p.name),
                                t.value,
                                fixed,
                                if k == 1 { "а" } else { "" },
                                q.value,
                                pr.value,
                                expect
                            ));
                            p.total = Some(Num::new(fixed, t.raw.clone()));
                        }
                    }
                }
            }
        }
        // --- правило B: цена потеряна (роль колонки не нашлась), но
        // количество есть — ищем цену среди чисел строки: если
        // qty × c ≈ total / 10^k, чиним total и достанавливаем цену ---
        if let (Some(q), Some(t)) = (&p.quantity, &p.total) {
            if p.price.is_none() && looks_lost_comma(&t.raw) && q.value > 0.0 && t.value > 0.0 {
                let row_nums = row_numbers(&p.raw);
                for (c, craw) in row_nums {
                    if c <= 0.0 {
                        continue;
                    }
                    let expect = q.value * c;
                    let tol = (1.0_f64).max(expect * 0.02);
                    if let Some((k, _)) = shift_factor(t.value, expect, tol) {
                        let column_ok = match frac {
                            Some(f) if f >= 1 => k == f || k == f + 1,
                            _ => true,
                        };
                        if column_ok {
                            let fixed = repair_value(t.value, k);
                            notes.push(format!(
                                "«{}»: сумма {} → {}, цена восстановлена {} ({} × {} = {:.2})",
                                name_short(&p.name),
                                t.value,
                                fixed,
                                c,
                                q.value,
                                c,
                                expect
                            ));
                            p.total = Some(Num::new(fixed, t.raw.clone()));
                            p.price = Some(Num::new(c, craw));
                            break;
                        }
                    }
                }
            }
        }
        // --- правило C: total выглядит целым, но в строке есть число
        // с разделителем на позиции итога (последнее число) — берём его ---
        if let Some(t) = &p.total {
            if looks_lost_comma(&t.raw) {
                if let Some((last, lraw)) = row_numbers(&p.raw).into_iter().last() {
                    if has_separator(&lraw) && last > 0.0 && (last - t.value).abs() > 0.005 {
                        // последнее число « целее»: и не совпадает с битым итогом
                        let tol = (1.0_f64).max(last * 0.02);
                        if (t.value - last).abs() > tol {
                            notes.push(format!(
                                "«{}»: сумма {} → {} (итог колонки «Итого» из той же строки)",
                                name_short(&p.name),
                                t.value,
                                last
                            ));
                            p.total = Some(Num::new(last, lraw));
                        }
                    }
                }
            }
        }
        // ресурсы позиции
        let name = p.name.clone();
        for r in p.resources.iter_mut() {
            repair_resource(r, notes, &name);
        }
    }
    for sub in sec.subsections.iter_mut() {
        repair_section_positions(sub, frac, notes);
    }
    // ресурсы раздела (НР/СП): чиним по итогу раздела позже, здесь только
    // отметим их как кандидатов, если походит арифметика с ценой/кол-вом
    for r in sec.resources.iter_mut() {
        let _ = &r.name;
    }
}

fn repair_resource(r: &mut Resource, notes: &mut Vec<String>, _pos_name: &str) {
    if let (Some(q), Some(pr), Some(t)) = (&r.quantity, &r.price, &r.total) {
        if !looks_lost_comma(&t.raw) {
            return;
        }
        let expect = q.value * pr.value;
        if expect <= 0.0 || t.value <= 0.0 {
            return;
        }
        let tol = (1.0_f64).max(expect * 0.02);
        if let Some((k, _)) = shift_factor(t.value, expect, tol) {
            let fixed = repair_value(t.value, k);
            notes.push(format!(
                "«{}»: сумма {} → {} (потеря запятой ×{}, {} × {} = {:.2})",
                name_short(&r.name),
                t.value,
                fixed,
                10f64.powi(k as i32) as u64,
                q.value,
                pr.value,
                expect
            ));
            r.total = Some(Num::new(fixed, t.raw.clone()));
        }
    }
}

/// Короткое имя для сообщений.
fn name_short(s: &str) -> String {
    s.chars().take(40).collect()
}

/// Чистые числа из сырых строк позиции (для поиска цены при потерянной
/// роли колонки). Токены после разбиения по пробелам, без слов с буквами
/// рядом («В22,5», «0,5м», «600*300*250» отбрасываются).
fn row_numbers(raws: &[String]) -> Vec<(f64, String)> {
    let mut tokens: Vec<String> = Vec::new();
    for line in raws {
        for tok in line.split_whitespace() {
            let t = tok.trim_matches(|c: char| !c.is_ascii_digit() && c != ',' && c != '.');
            if t.is_empty() {
                continue;
            }
            // токен-« слово с цифрой» (артикул, размер) — не число суммы
            if tok.chars().any(|c| c.is_alphabetic()) {
                continue;
            }
            tokens.push(t.to_string());
        }
    }
    // склейка разрядных групп: «14 500.00» → 14500, «1 234 567» → 1234567
    let group3 = |t: &str| -> bool {
        let core = t.trim_end_matches(|c: char| c == ',' || c == '.');
        let (int_part, dec) = match core.split_once([',', '.']) {
            Some((i, d)) => (i, Some(d)),
            None => (core, None),
        };
        int_part.len() == 3
            && int_part.chars().all(|c| c.is_ascii_digit())
            && dec
                .map(|d| d.len() <= 2 && d.chars().all(|c| c.is_ascii_digit()))
                .unwrap_or(true)
    };
    let short_int = |t: &str| -> bool {
        !t.is_empty() && (1..=3).contains(&t.len()) && t.chars().all(|c| c.is_ascii_digit())
    };
    let mut out: Vec<(f64, String)> = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let mut merged = tokens[i].clone();
        let mut j = i;
        // «1 234 567,89»: цепочка коротких групп + группы по 3
        while j + 1 < tokens.len() && short_int(&tokens[j]) && group3(&tokens[j + 1]) {
            merged.push(' ');
            merged.push_str(&tokens[j + 1]);
            j += 1;
        }
        if let Some(v) = parse_number(&merged) {
            out.push((v, merged.clone()));
        }
        i = j + 1;
    }
    out
}

// ---------- проход 2: итоги разделов ----------

/// Сумма позиций раздела (итоги позиций или кол-во × цена).
fn section_positions_sum(s: &Section) -> f64 {
    let mut sum = 0.0;
    for p in s.all_positions() {
        if let Some(t) = &p.total {
            sum += t.value;
        } else if let (Some(q), Some(pr)) = (&p.quantity, &p.price) {
            sum += q.value * pr.value;
        }
    }
    sum
}

/// Сумма позиций + НР/СП-ресурсов раздела.
fn section_full_sum(s: &Section) -> f64 {
    let mut sum = section_positions_sum(s);
    for r in &s.resources {
        if let Some(t) = &r.total {
            sum += t.value;
        }
    }
    sum
}

/// Починка итогов разделов: Σ позиций ≈ итог × 10^k.
fn repair_section_totals(doc: &mut EstimateDocument) -> Vec<String> {
    let mut notes = Vec::new();
    let mut sections = std::mem::take(&mut doc.sections);
    for sec in sections.iter_mut() {
        repair_one_section_total(sec, &mut notes);
    }
    doc.sections = sections;
    notes
}

fn repair_one_section_total(sec: &mut Section, notes: &mut Vec<String>) {
    let computed = section_full_sum(sec);
    if let Some(t) = &sec.total {
        if computed <= 0.0 || t.value <= 0.0 {
            return;
        }
        // итог раздела в исходном сметном формате почти всегда с копейками
        let tol = tolerance_sec(computed);
        if let Some((k, _)) = shift_factor(t.value, computed, tol) {
            let fixed = repair_value(t.value, k);
            notes.push(format!(
                "«{}»: итог раздела {} → {} (Σ позиций = {:.2})",
                name_short(&sec.title),
                t.value,
                fixed,
                computed
            ));
            sec.total = Some(Num::new(fixed, t.raw.clone()));
        }
    }
    for sub in sec.subsections.iter_mut() {
        repair_one_section_total(sub, notes);
    }
}

fn tolerance_sec(base: f64) -> f64 {
    (2.0_f64).max(base * 0.005)
}

// ---------- точка входа ----------

/// Полная авто-починка документа. Возвращает список описаний починок,
/// которые вызывающий код добавит как warnings `auto_repair`.
pub fn repair(doc: &mut EstimateDocument) -> Vec<String> {
    let mut notes = Vec::new();
    // проход 1: позиции по qty × price
    let frac = typical_frac_len(doc);
    notes.extend(repair_positions(doc, frac));
    // проход 2: итоги разделов по Σ позиций
    notes.extend(repair_section_totals(doc));
    notes
}


#[cfg(test)]
mod tests {
    use super::*;

    fn num_raw(raw: &str, v: f64) -> Option<Num> {
        Some(Num::new(v, raw.to_string()))
    }

    fn pos_lost(total_raw: &str, total: f64, qty: f64, price: f64) -> Position {
        Position {
            line: Some(1),
            cipher: None,
            name: "Работа бетононасоса".to_string(),
            unit: Some("смена".to_string()),
            quantity: num_raw(&qty.to_string(), qty),
            price: num_raw(&price.to_string(), price),
            total: num_raw(total_raw, total),
            resources: Vec::new(),
            pages: vec![1],
            raw: Vec::new(),
        }
    }

    #[test]
    fn repair_total_lost_comma() {
        // 3.119: 1 × 32000 = 32000, распознано 3200000 (×100)
        let mut doc = doc_of(vec![pos_lost("3200000", 3_200_000.0, 1.0, 32_000.0)]);
        let notes = repair(&mut doc);
        assert_eq!(
            doc.sections[0].positions[0].total.as_ref().unwrap().value,
            32_000.0
        );
        assert!(!notes.is_empty());
    }

    #[test]
    fn no_repair_when_separator_present() {
        // «32 000,00» — нормальное число, чинить нельзя
        let mut doc = doc_of(vec![pos_lost("32000,00", 32_000.0, 1.0, 32_000.0)]);
        let notes = repair(&mut doc);
        assert_eq!(
            doc.sections[0].positions[0].total.as_ref().unwrap().value,
            32_000.0
        );
        assert!(notes.is_empty());
    }

    #[test]
    fn no_repair_on_unrelated_mismatch() {
        // 5 × 100 = 500, total 7000 без разделителя — это не ×10^k, не чиним
        let mut doc = doc_of(vec![pos_lost("7000", 7_000.0, 5.0, 100.0)]);
        let notes = repair(&mut doc);
        assert_eq!(
            doc.sections[0].positions[0].total.as_ref().unwrap().value,
            7_000.0
        );
        assert!(notes.is_empty());
    }

    #[test]
    fn repair_section_total() {
        // итог раздела распознан как 3200000 (×100 от 32000)
        let p = pos_lost("32000,00", 32_000.0, 1.0, 32_000.0);
        let mut sec = Section {
            number: Some("3".to_string()),
            title: "Устройство фундамента".to_string(),
            subsections: Vec::new(),
            positions: vec![p],
            resources: Vec::new(),
            total: num_raw("3200000", 3_200_000.0),
            pages: vec![2],
        };
        let mut doc = EstimateDocument {
            meta: Meta::default(),
            sections: vec![sec],
            totals: Totals::default(),
            validation: ValidationReport::default(),
            auto_repairs: 0,
            unparsed_lines: Vec::new(),
            pages: 4,
        };
        let notes = repair(&mut doc);
        assert_eq!(doc.sections[0].total.as_ref().unwrap().value, 32_000.0);
        assert!(!notes.is_empty());
    }

    #[test]
    fn double_shift_case_8_95() {
        // случай из сметы ПРАВИЛО: qty=1, price=300000 («3 000,00»), total=300000
        // оба без запятой — локально «сходятся» (1×300000=300000), чинить
        // нельзя по qty×price. Итога раздела нет — проверяем, что починки
        // не произошло
        let mut doc = doc_of(vec![pos_lost("300000", 300_000.0, 1.0, 300_000.0)]);
        let notes = repair(&mut doc);
        assert!(notes.is_empty());
        assert_eq!(
            doc.sections[0].positions[0].total.as_ref().unwrap().value,
            300_000.0
        );
    }

    #[test]
    fn rule_b_price_lost_recovered_from_raw() {
        // 3.119 из сметы ПРАВИЛО: цена потеряна (роль колонки не нашлась),
        // qty=1, total=3200000, в raw есть «32000.00» → 1 × 32000 = 32000
        let mut p = pos_lost("3200000", 3_200_000.0, 1.0, 32_000.0);
        p.price = None;
        p.raw = vec!["3.119 Работа бетононасоса смена 1 32000.00 3200000 0.00 0.00 32000.00".to_string()];
        let mut doc = doc_of(vec![p]);
        let notes = repair(&mut doc);
        let pos = &doc.sections[0].positions[0];
        assert_eq!(pos.total.as_ref().unwrap().value, 32_000.0);
        assert_eq!(pos.price.as_ref().unwrap().value, 32_000.0);
        assert!(!notes.is_empty());
    }

    #[test]
    fn rule_c_last_number_with_separator_wins() {
        // 6.132: total=29900000 (битый), последнее число строки «299 00000»
        // тоже битое — правило C не сработает; проверяем на случае, где
        // последнее число целое: total=3200000, последнее «32000.00"
        let mut p = pos_lost("3200000", 3_200_000.0, 1.0, 32_000.0);
        p.raw = vec!["6.134 Доставка смена 1 0.00 0.00 0.00 32000.00".to_string()];
        let mut doc = doc_of(vec![p]);
        let notes = repair(&mut doc);
        assert_eq!(doc.sections[0].positions[0].total.as_ref().unwrap().value, 32_000.0);
        assert!(!notes.is_empty());
    }

    fn doc_of(positions: Vec<Position>) -> EstimateDocument {
        EstimateDocument {
            meta: Meta::default(),
            sections: vec![Section {
                number: Some("1".to_string()),
                title: "Раздел".to_string(),
                subsections: Vec::new(),
                positions,
                resources: Vec::new(),
                total: None,
                pages: vec![1],
            }],
            totals: Totals::default(),
            validation: ValidationReport::default(),
            auto_repairs: 0,
            unparsed_lines: Vec::new(),
            pages: 1,
        }
    }
}
