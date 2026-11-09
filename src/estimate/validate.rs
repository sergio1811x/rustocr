//! Арифметическая и логическая валидация разобранной сметы:
//! количество × цена = сумма; итоги разделов = Σ позиций; итог сметы =
//! Σ разделов; НДС и «Всего с НДС»; пропущенные номера позиций; строки,
//! которые парсер не понял.

use crate::estimate::model::*;
use crate::estimate::repair;

/// Допуск для «количество × цена = сумма» (округления в сметах обычны).
const QTY_PRICE_TOL: f64 = 0.02; // 2%
/// Допуск для итогов (руб., абсолют).
const TOTAL_TOL_RUB: f64 = 2.0;
/// Допуск для итогов (доля).
const TOTAL_TOL_SHARE: f64 = 0.005; // 0.5%

pub fn validate(doc: &mut EstimateDocument) {
    // СНАЧАЛА авто-починка: она меняет значения, а валидация ниже уже
    // проверяет починенный документ и не генерит ложных ошибок про
    // расхождения, которые мы сами устранили.
    let notes = repair::repair(doc);

    let mut rep = std::mem::take(&mut doc.validation);
    rep.errors.clear();
    rep.warnings.clear();
    // пересчитываем счётчики заново; из парсера берём только табличную статистику
    let table_rows = rep.stats.table_rows;
    let unclassified_rows = rep.stats.unclassified_rows;
    rep.stats = ValidationStats {
        table_rows,
        unclassified_rows,
        ..Default::default()
    };

    // отчёт о починках — в warnings, одной строкой на починку
    for n in &notes {
        rep.warn("auto_repair", n.clone(), None);
    }

    // --- позиции: количество × цена = сумма ---
    let sections = doc.sections.clone();
    for sec in &sections {
        validate_section(sec, &mut rep);
    }

    // --- итог документа ---
    let sum_sections: f64 = sections
        .iter()
        .map(|s| section_sum(s))
        .filter(|v| *v > 0.0)
        .sum();
    let n_with_total = sections.iter().filter(|s| section_sum(s) > 0.0).count();
    if let Some(t) = &doc.totals.total {
        if n_with_total > 0 {
            let diff = (sum_sections - t.value).abs();
            if diff > tolerance(t.value) {
                rep.error(
                    "doc_total_mismatch",
                    format!(
                        "Итог сметы {} ≠ Σ итогов разделов {} (разница {:.2} руб.)",
                        t.value, sum_sections, diff
                    ),
                    Some("документ".to_string()),
                );
            }
        }
    } else if n_with_total > 1 && sum_sections > 0.0 {
        rep.warn(
            "doc_total_missing",
            format!("Итог сметы не распознан (Σ разделов = {:.2} руб.)", sum_sections),
            Some("документ".to_string()),
        );
    }

    // --- НДС ---
    if let (Some(total), Some(vat)) = (&doc.totals.total, &doc.totals.vat) {
        if let Some(rate) = doc.totals.vat_rate {
            let expect = total.value * rate;
            let diff = (vat.value - expect).abs();
            if diff > tolerance(expect) {
                rep.warn(
                    "vat_mismatch",
                    format!(
                        "НДС {} не соответствует {}% от {} (ожидалось {:.2})",
                        vat.value,
                        (rate * 100.0).round(),
                        total.value,
                        expect
                    ),
                    Some("документ".to_string()),
                );
            }
        }
    }
    if let (Some(total), Some(vat), Some(twv)) = (
        &doc.totals.total,
        &doc.totals.vat,
        &doc.totals.total_with_vat,
    ) {
        let expect = total.value + vat.value;
        if (twv.value - expect).abs() > tolerance(expect) {
            rep.error(
                "total_with_vat_mismatch",
                format!(
                    "«Всего с НДС» {} ≠ Итого {} + НДС {} = {:.2}",
                    twv.value, total.value, vat.value, expect
                ),
                Some("документ".to_string()),
            );
        }
    }

    // --- пропущенные позиции ---
    check_numbering(&sections, &mut rep);

    // --- неклассифицированные строки ---
    if rep.stats.unclassified_rows > 0 {
        let examples: Vec<String> = doc
            .unparsed_lines
            .iter()
            .take(3)
            .map(|s| {
                let cut: String = s.chars().take(60).collect();
                format!("«{}»", cut)
            })
            .collect();
        rep.warn(
            "unclassified_rows",
            format!(
                "Строк таблицы не разобрано: {} из {}{}",
                rep.stats.unclassified_rows,
                rep.stats.table_rows,
                if examples.is_empty() {
                    String::new()
                } else {
                    format!(" — например: {}", examples.join("; "))
                }
            ),
            None,
        );
    }
    if rep.stats.positions == 0 {
        rep.warn(
            "no_positions",
            "Не распознано ни одной позиции сметы — проверьте формат документа".to_string(),
            None,
        );
    }

    doc.validation = rep;
}

fn tolerance(base: f64) -> f64 {
    (TOTAL_TOL_RUB).max(base.abs() * TOTAL_TOL_SHARE)
}

/// Сумма раздела: явный итог, иначе Σ позиций (итоги позиций или кол-во × цена).
fn section_sum(s: &Section) -> f64 {
    if let Some(t) = &s.total {
        return t.value;
    }
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

/// Сумма позиции для сверки итога раздела: итог позиции + накладные
/// расходы и сметная прибыль её ресурсных строк (в выкладках ГС/РИК итог
/// раздела = Σ позиций + Σ НР/СП, привязанных к позициям).
fn position_sum(p: &Position) -> Option<f64> {
    let mut sum = p.total.as_ref().map(|t| t.value).unwrap_or(0.0);
    let mut has = p.total.is_some();
    for r in &p.resources {
        if matches!(r.kind, ResourceKind::Overhead | ResourceKind::Profit) {
            if let Some(t) = &r.total {
                sum += t.value;
                has = true;
            }
        }
    }
    if has {
        Some(sum)
    } else {
        None
    }
}

fn validate_section(sec: &Section, rep: &mut ValidationReport) {
    rep.stats.sections += 1;
    for p in sec.positions.iter() {
        rep.stats.positions += 1;
        rep.stats.resources += p.resources.len();
        let loc = format!(
            "Раздел {}{}, позиция {}",
            sec.number.as_deref().unwrap_or(""),
            if sec.number.is_some() { ", " } else { "" },
            p.line
                .map(|l| l.to_string())
                .unwrap_or_else(|| p.name.chars().take(30).collect())
        );
        // количество × цена = сумма
        if let (Some(q), Some(pr), Some(t)) = (&p.quantity, &p.price, &p.total) {
            let expect = q.value * pr.value;
            let diff = (t.value - expect).abs();
            if diff > (1.0_f64).max(expect * QTY_PRICE_TOL) {
                rep.warn(
                    "qty_price_mismatch",
                    format!(
                        "{} × {} = {:.2} ≠ сумма {} (разница {:.2} руб.)",
                        q.value, pr.value, expect, t.value, diff
                    ),
                    Some(loc.clone()),
                );
            }
            rep.stats.qty_price_checked += 1;
        }
        // пустые поля
        if p.name.is_empty() || p.name == "(без названия)" {
            rep.warn("position_no_name", "У позиции нет наименования".to_string(), Some(loc.clone()));
        }
        if p.total.is_none() && p.quantity.is_none() {
            rep.warn(
                "position_no_numbers",
                "У позиции нет ни количества, ни суммы".to_string(),
                Some(loc),
            );
        }
    }
    // итог раздела = Σ позиций
    let sum_positions: f64 = sec
        .positions
        .iter()
        .chain(sec.subsections.iter().flat_map(|s| s.positions.iter()))
        .filter_map(position_sum)
        .sum();
    let sub_sums: f64 = sec.subsections.iter().map(section_sum).sum();
    let computed = if sum_positions > 0.0 { sum_positions } else { sub_sums };
    if let Some(t) = &sec.total {
        if computed > 0.0 {
            let diff = (t.value - computed).abs();
            if diff > tolerance(t.value) {
                rep.error(
                    "section_total_mismatch",
                    format!(
                        "Итог раздела «{}» {} ≠ Σ позиций {:.2} (разница {:.2} руб.)",
                        sec.title, t.value, computed, diff
                    ),
                    Some(format!(
                        "Раздел {}",
                        sec.number.as_deref().unwrap_or(&sec.title)
                    )),
                );
            }
            rep.stats.section_totals_checked += 1;
        }
    } else if computed > 0.0 {
        rep.warn(
            "section_total_missing",
            format!(
                "Итог раздела «{}» не распознан (Σ позиций = {:.2} руб.)",
                sec.title, computed
            ),
            Some(format!("Раздел {}", sec.number.as_deref().unwrap_or(&sec.title))),
        );
    }
    for sub in &sec.subsections {
        validate_section(sub, rep);
    }
}

/// Дыры и повторы в нумерации позиций.
fn check_numbering(sections: &[Section], rep: &mut ValidationReport) {
    let mut lines: Vec<u32> = Vec::new();
    for sec in sections {
        collect_lines(sec, &mut lines);
    }
    if lines.len() < 3 {
        return;
    }
    let mut sorted = lines.clone();
    sorted.sort_unstable();
    sorted.dedup();
    // повторы номеров
    let mut counts: std::collections::BTreeMap<u32, usize> = Default::default();
    for l in &lines {
        *counts.entry(*l).or_default() += 1;
    }
    let dupes: Vec<u32> = counts
        .into_iter()
        .filter(|(_, c)| *c > 1)
        .map(|(v, _)| v)
        .collect();
    if !dupes.is_empty() {
        rep.warn(
            "position_duplicate",
            format!("Повторяющиеся номера позиций: {:?}", dupes),
            None,
        );
    }
    let missing: Vec<u32> = (sorted.first().copied().unwrap_or(1)..=sorted.last().copied().unwrap_or(1))
        .filter(|n| !sorted.contains(n))
        .collect();
    if !missing.is_empty() && missing.len() < 20 {
        rep.warn(
            "position_missing",
            format!("Пропущенные номера позиций: {:?}", missing),
            None,
        );
    }
}

fn collect_lines(sec: &Section, out: &mut Vec<u32>) {
    for p in &sec.positions {
        if let Some(l) = p.line {
            out.push(l);
        }
    }
    for sub in &sec.subsections {
        collect_lines(sub, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num(v: f64) -> Option<Num> {
        Some(Num::new(v, v.to_string()))
    }

    fn pos(line: u32, qty: f64, price: f64, total: f64) -> Position {
        Position {
            line: Some(line),
            cipher: None,
            name: format!("Работа {}", line),
            unit: Some("шт".to_string()),
            quantity: num(qty),
            price: num(price),
            total: num(total),
            resources: Vec::new(),
            pages: vec![1],
            raw: Vec::new(),
        }
    }

    #[test]
    fn qty_price_and_section_total_ok() {
        let mut doc = EstimateDocument {
            meta: Meta::default(),
            sections: vec![Section {
                number: Some("1".to_string()),
                title: "Земляные работы".to_string(),
                subsections: Vec::new(),
                positions: vec![pos(1, 10.0, 100.0, 1000.0), pos(2, 5.0, 200.0, 1000.0)],
                resources: Vec::new(),
                total: num(2000.0),
                pages: vec![1],
            }],
            totals: Totals {
                total: num(2000.0),
                vat_rate: Some(0.2),
                vat: num(400.0),
                total_with_vat: num(2400.0),
            },
            validation: ValidationReport::default(),
            unparsed_lines: Vec::new(),
            auto_repairs: 0,
            pages: 1,
        };
        validate(&mut doc);
        assert!(doc.validation.errors.is_empty(), "ошибки: {:?}", doc.validation.errors);
        assert!(doc.validation.warnings.is_empty(), "предупреждения: {:?}", doc.validation.warnings);
        assert_eq!(doc.validation.stats.qty_price_checked, 2);
        assert_eq!(doc.validation.stats.section_totals_checked, 1);
    }

    #[test]
    fn mismatch_is_caught() {
        let mut doc = EstimateDocument {
            meta: Meta::default(),
            sections: vec![Section {
                number: Some("1".to_string()),
                title: "Раздел".to_string(),
                subsections: Vec::new(),
                positions: vec![pos(1, 10.0, 100.0, 900.0)], // 10×100=1000 ≠ 900
                resources: Vec::new(),
                total: num(1500.0), // Σ = 999 ≠ 1500
                pages: vec![1],
            }],
            totals: Totals {
                total: num(1600.0), // Σ разделов = 1500 ≠ 1600
                vat_rate: Some(0.2),
                vat: num(400.0), // 20% от 1600 = 320 ≠ 400
                total_with_vat: num(1900.0), // 1600+400 = 2000 ≠ 1900
            },
            validation: ValidationReport::default(),
            unparsed_lines: Vec::new(),
            auto_repairs: 0,
            pages: 1,
        };
        validate(&mut doc);
        let kinds: Vec<&str> = doc.validation.warnings.iter().map(|i| i.kind.as_str()).collect();
        assert!(kinds.contains(&"qty_price_mismatch"));
        let ekind: Vec<&str> = doc.validation.errors.iter().map(|i| i.kind.as_str()).collect();
        assert!(ekind.contains(&"section_total_mismatch"));
        assert!(ekind.contains(&"doc_total_mismatch"));
        assert!(ekind.contains(&"total_with_vat_mismatch"));
        assert!(kinds.contains(&"vat_mismatch"));
    }

    #[test]
    fn missing_numbering() {
        let mut doc = EstimateDocument {
            meta: Meta::default(),
            sections: vec![Section {
                number: None,
                title: "(без разделов)".to_string(),
                subsections: Vec::new(),
                positions: vec![pos(1, 1.0, 1.0, 1.0), pos(2, 1.0, 1.0, 1.0), pos(5, 1.0, 1.0, 1.0)],
                resources: Vec::new(),
                total: num(3.0),
                pages: vec![1],
            }],
            totals: Totals::default(),
            validation: ValidationReport::default(),
            unparsed_lines: Vec::new(),
            auto_repairs: 0,
            pages: 1,
        };
        validate(&mut doc);
        let kinds: Vec<&str> = doc.validation.warnings.iter().map(|i| i.kind.as_str()).collect();
        assert!(kinds.contains(&"position_missing"), "warnings: {:?}", kinds);
    }
}
