//! Локальный семантический парсер сметы: классифицирует строки
//! восстановленной таблицы (раздел / позиция / ресурс / итог / шапка),
//! вытаскивает числа, шифры расценок, единицы измерения. Работает офлайн,
//! без нейросетей — и служит базой, которую LLM-парсер (pazl.rs) может
//! уточнять, но не обязан.

use crate::estimate::model::*;

/// Строка документа в порядке чтения (со всех страниц).
pub struct DocRow {
    pub page: usize,
    pub columnar: bool,
    /// (колонка, colspan, текст) слева направо
    pub cells: Vec<(usize, usize, String)>,
}

impl DocRow {
    fn joined(&self) -> String {
        self.cells
            .iter()
            .map(|(_, _, t)| t.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }
    fn lower(&self) -> String {
        self.joined().to_lowercase()
    }
    /// числовые ячейки (колонка, значение, текст), слева направо
    fn numbers(&self) -> Vec<(usize, f64, String)> {
        self.cells
            .iter()
            .filter_map(|(col, _, t)| parse_number(t).map(|v| (*col, v, t.trim().to_string())))
            .collect()
    }
}

// ---------- примитивы: числа, единицы, шифры ----------

/// Разбор числа из сметы: «1 234,56» → 1234.56. Строго: только цифры,
/// пробелы/nbsp (группы разрядов) и один-два разделителя «,»/«.».
pub fn parse_number(s: &str) -> Option<f64> {
    let raw = s.trim();
    if raw.is_empty() {
        return None;
    }
    let t: String = raw.chars().filter(|c| *c != ' ' && *c != '\u{a0}').collect();
    if t.is_empty() {
        return None;
    }
    let mut seps: Vec<usize> = Vec::new();
    for (i, c) in t.char_indices() {
        if c.is_ascii_digit() {
            continue;
        }
        if c == ',' || c == '.' {
            seps.push(i);
        } else {
            return None; // посторонний символ — не число
        }
    }
    if seps.is_empty() {
        return t.parse::<f64>().ok().filter(|v| v.is_finite());
    }
    let last = *seps.last().unwrap();
    if last == t.len() - 1 {
        return None; // «1,» — обрыв
    }
    if seps.len() > 2 {
        return None;
    }
    if seps.len() == 2 {
        let first = seps[0];
        let after_last = &t[last + 1..];
        let after_first = &t[first + 1..last];
        if after_last.len() <= 2 {
            // «1.234,56»: последний разделитель — десятичный
            let int_part: String = t[..first].chars().chain(t[first + 1..last].chars()).collect();
            let mut norm = int_part;
            norm.push('.');
            norm.push_str(after_last);
            return norm.parse::<f64>().ok();
        }
        // «1.234.567» — оба разрядные
        if after_first.len() == 3 && after_last.len() == 3 {
            let int_part: String = t.chars().filter(|c| c.is_ascii_digit()).collect();
            return int_part.parse::<f64>().ok();
        }
        return None;
    }
    t.replace(',', ".").parse::<f64>().ok()
}

/// Канонизация единицы измерения: «м2»/«М²» → «м²», «шт.» → «шт»,
/// «маш.-ч» → «маш-ч». Донормализация раскладки: латинские буквы-омофоны
/// (M2, M3, Mn, Maw, WT, ET, Me, K-T, MП) заменяются на русские единицы —
/// «М» и «П» в русской раскладке падают на те же клавиши, что латинские
/// V/T, а распознаватель возвращает их латиницей. «мл» как «м²» не
/// трогаем: в сметах «мл» сам по себе не встречается, а вот подмена
/// «м²» → «мл» встречается (см. корпус), поэтому «мл» → «м²» при
/// сопутствующих признаках делает вызывающий код, не канонизация.
pub fn normalize_unit(s: &str) -> String {
    let t = s.trim().to_lowercase().replace('\u{a0}', " ");
    // латинские омофоны, характерные для единиц измерения:
    // «м2» в латинице — «m2», «мп» — «mn»/«mп», «шт» — «et»/«шт»,
    // «маш» — «maw»/«мaw», «к-т» — «k-t», «мест» — «me»/«мecт»
    let t = t
        .replace("м2", "м²")
        .replace("м3", "м³")
        .replace("м^2", "м²")
        .replace("м^3", "м³")
        .replace("m2", "м²")
        .replace("m3", "м³")
        .replace("m^2", "м²")
        .replace("m^3", "м³");
    let t: String = t.chars().filter(|c| *c != '.').collect();
    let t = t.trim().to_string();
    // однозначные латинские последовательности, которых в русских сметах
    // не бывает, зато бывают при перепутанной раскладке
    let t = match t.as_str() {
        "mn" | "mп" | "мп" => "мп".to_string(),
        "maw" | "мaw" => "маш".to_string(),
        "wt" | "et" | "me" => "шт".to_string(),
        "k-t" => "к-т".to_string(),
        "mп" => "мп".to_string(),
        "мл" => "мл".to_string(), // честное «мл» — редкая единица, оставляем
        _ => t,
    };
    t.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Словарь единиц измерения смет (после канонизации, без точек).
const UNITS: &[&str] = &[
    "м", "м²", "м³", "км", "дм", "см", "мм", "т", "кг", "г", "ц", "шт", "компл", "л", "мл",
    "упак", "мес", "смена", "маш-ч", "чел-ч", "чел-мин", "т-км", "квт-ч", "гкал", "мб", "руб",
    "тыс руб", "млн руб", "%", "тыс шт", "пог м", "мп",
];

/// Похоже ли значение ячейки на единицу измерения.
pub fn is_unit(s: &str) -> bool {
    let u = normalize_unit(s);
    if u.is_empty() {
        return false;
    }
    if UNITS.contains(&u.as_str()) {
        return true;
    }
    // составные: «м³/т», «шт/м²», «кг/м» …
    if u.matches('/').count() == 1 {
        if let Some((a, b)) = u.split_once('/') {
            return UNITS.contains(&a.trim()) && UNITS.contains(&b.trim());
        }
    }
    // числовой префикс: «100 м²», «1 000 шт»
    if let Some((num, rest)) = u.split_once(' ') {
        if !num.is_empty() && num.chars().all(|c| c.is_ascii_digit()) {
            return UNITS.contains(&rest);
        }
    }
    false
}

/// Шифр расценки позиции: «ГЭСН 01-01-013-01», «ФЕР06-01-001-01»,
/// «01-01-013-01», «ТЕР 12-01-001-1», «ПР-123», «12.01.001.01».
pub fn position_cipher(s: &str) -> Option<String> {
    let t = s.trim().trim_start_matches("№").trim();
    if t.is_empty() {
        return None;
    }
    let lower = t.to_lowercase();
    let labels = [
        "гэсн", "фер", "тер", "мэр", "оэр", "фсрр", "фссц", "тссц", "пгэ", "снир",
    ];
    let strip_label = |x: &str| -> String {
        for l in labels {
            if let Some(rest) = x.strip_prefix(l) {
                return rest.trim_start_matches(|c: char| c == ' ' || c == '№' || c == '-').to_string();
            }
        }
        x.to_string()
    };
    let code = strip_label(&lower);
    // кодовая часть: XXX-XX-XXX(-XX) — ГЭСН-формат, или XX.XX.XXX.XX — точечный
    let digits_dashes: String = code
        .trim_start_matches(|c: char| c.is_ascii_alphabetic())
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '-' || *c == '.')
        .collect();
    let parts: Vec<&str> = digits_dashes.split('-').collect();
    if parts.len() >= 3
        && parts[..3].iter().all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
        && parts[0].len() == 2
        && parts[1].len() == 2
        && parts[2].len() == 3
    {
        return Some(t.to_string());
    }
    let dparts: Vec<&str> = digits_dashes.split('.').collect();
    if dparts.len() >= 4
        && dparts[..4].iter().all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
    {
        return Some(t.to_string());
    }
    // «ПР-123», «ЭМ-123», «ЗП-123»
    if let Some(tail) = lower.strip_prefix("пр-").or_else(|| lower.strip_prefix("эм-")).or_else(|| lower.strip_prefix("зп-")) {
        if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) {
            return Some(t.to_string());
        }
    }
    None
}

/// Код ресурса (не расценка): «1.1-1-2345», «101-0797», «060148».
pub fn resource_code(s: &str) -> Option<String> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    // 1.1-1-2345
    if let Some((head, rest)) = t.split_once('.') {
        if !head.is_empty() && head.len() <= 2 && head.chars().all(|c| c.is_ascii_digit()) {
            let rparts: Vec<&str> = rest.split('-').collect();
            if rparts.len() >= 3
                && rparts[..2].iter().all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
                && rparts[2].len() >= 3
                && rparts[2].chars().all(|c| c.is_ascii_digit())
            {
                return Some(t.to_string());
            }
        }
    }
    // 101-0797
    let parts: Vec<&str> = t.split('-').collect();
    if parts.len() == 2
        && parts[0].len() == 3
        && parts[1].len() == 4
        && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit()))
    {
        return Some(t.to_string());
    }
    // 060148 — код машины (6 цифр)
    if t.len() == 6 && t.chars().all(|c| c.is_ascii_digit()) {
        return Some(t.to_string());
    }
    None
}

/// Похоже ли на дату: «07.12.2021», «7.12.21».
fn is_date(s: &str) -> bool {
    let t = s.trim();
    let parts: Vec<&str> = t.split('.').collect();
    if parts.len() != 3 {
        return false;
    }
    parts[0].len() >= 1
        && parts[0].len() <= 2
        && parts[1].len() <= 2
        && parts.iter().all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
        && parts[2].len() >= 2
}

// ---------- роли колонок из шапки ----------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Role {
    Line,
    Cipher,
    Name,
    Unit,
    Qty,
    Price,
    Total,
}

fn role_of_header(text: &str) -> Option<Role> {
    let t = text.trim().to_lowercase();
    if t.is_empty() {
        return None;
    }
    if t.contains("п/п") || t == "№" || t.starts_with("№") || t.contains("номер") {
        return Some(Role::Line);
    }
    if t.contains("изм") || t == "ед" || t.starts_with("ед ") {
        return Some(Role::Unit);
    }
    if t.contains("кол") || t.contains("объем") || t.contains("объём") {
        return Some(Role::Qty);
    }
    if t.contains("обоснов") || t.contains("шифр") || t.contains("артикул") {
        return Some(Role::Cipher);
    }
    if t.contains("наимен") || t.contains("описан") {
        return Some(Role::Name);
    }
    if t.contains("цена") || t.contains("за ед") || t.contains("стоимость единицы") {
        return Some(Role::Price);
    }
    if t == "итого" || t == "итого," {
        // голая колонка «Итого» (коммерческие сметы с двухуровневой шапкой)
        return Some(Role::Total);
    }
    if t.contains("всего") || t.contains("сумма") || t.contains("стоимост") || t.contains("затрат") {
        return Some(Role::Total);
    }
    None
}

/// Похоже ли на строку шапки таблицы: только подписи колонок, без чисел.
fn is_header_row(row: &DocRow) -> bool {
    if !row.numbers().is_empty() {
        return false;
    }
    let lower = row.lower();
    if lower.contains("наименован") {
        return true;
    }
    let marks = [
        "№", "ед", "кол", "цена", "сумма", "всего", "обоснов", "стоимост", "затрат", "труд",
        "машин", "материал",
    ];
    let hits = marks.iter().filter(|m| lower.contains(*m)).count();
    hits >= 3
}

// ---------- итоги ----------

#[derive(Clone, Copy, PartialEq, Debug)]
enum TotalKind {
    Section,
    Sub,
    Doc,
    Vat,
    DocWithVat,
}

fn totals_kind(row: &DocRow) -> Option<(TotalKind, Option<f64>)> {
    totals_kind_text(&row.lower())
}

fn totals_kind_text(lower: &str) -> Option<(TotalKind, Option<f64>)> {
    if lower.starts_with("ндс") || lower.starts_with("сумма ндс") {
        return Some((TotalKind::Vat, parse_vat_rate(lower)));
    }
    if lower.starts_with("всего с ндс") || lower.starts_with("итого с ндс") {
        return Some((TotalKind::DocWithVat, None));
    }
    // разбивка по видам работ/затрат в конце Гранд-Сметы и «прямые затраты» —
    // промежуточные суммы, не итог раздела и не итог сметы
    if lower.contains("прямые затраты")
        || (lower.starts_with("итого")
            && (lower.contains("строительные работы")
                || lower.contains("монтажные работы")
                || lower.contains("прочие работы")
                || lower.contains("прочие затраты")))
    {
        return None;
    }
    if lower.contains("по разделу") {
        return Some((TotalKind::Section, None));
    }
    if lower.contains("по подразделу") {
        return Some((TotalKind::Sub, None));
    }
    if lower.contains("по главе") {
        return Some((TotalKind::Section, None));
    }
    if lower.contains("по главам") {
        return Some((TotalKind::Doc, None)); // итог сводного сметного расчёта
    }
    if lower.contains("по смете") || lower.contains("по объекту") || lower.contains("по расч") {
        return Some((TotalKind::Doc, None));
    }
    if lower.contains("по позиц") || lower.starts_with("справочно") {
        return None;
    }
    if lower.starts_with("итого") || lower.starts_with("всего") {
        return Some((TotalKind::Section, None)); // раздел или документ — решает контекст
    }
    None
}

/// Ставка НДС из «(20%)» / «(22 %)».
fn parse_vat_rate(lower: &str) -> Option<f64> {
    let mut from = 0usize;
    while let Some(i) = lower[from..].find('(') {
        let start = from + i + 1;
        if let Some(close) = lower[start..].find(')') {
            let inner = lower[start..start + close].trim().trim_end_matches('%').trim();
            if let Some(v) = parse_number(inner) {
                return Some(v / 100.0);
            }
        }
        from = start;
    }
    None
}

/// Последнее число в тексте со «мусором»: «Итого по смете 1 234 567,89 руб.»
fn parse_number_tail(joined: &str) -> Option<f64> {
    let toks: Vec<&str> = joined.split([' ', '\u{a0}']).filter(|t| !t.is_empty()).collect();
    let mut best: Option<f64> = None;
    let mut i = 0usize;
    while i < toks.len() {
        if !toks[i].starts_with(|c: char| c.is_ascii_digit()) {
            i += 1;
            continue;
        }
        let start = i;
        let mut acc = String::new();
        let mut last_value: Option<f64> = None;
        while i < toks.len() {
            let t = toks[i].trim_end_matches(|c: char| c == '.' || c == ',');
            if parse_number(t).is_some() {
                acc.push_str(t);
                if let Some(v) = parse_number(&acc) {
                    last_value = Some(v);
                }
                i += 1;
            } else {
                break;
            }
        }
        if let Some(v) = last_value {
            best = Some(v);
        }
        if i == start {
            i += 1;
        }
    }
    best
}

/// Итоговая сумма из строки: число с конца с валидацией групп разрядов.
/// Понимает порезанные ячейки и буквенные хвосты («1 341 653P» → 1341653),
/// не склеивает номер раздела с суммой («по разделу 1 6 172,50» → 6172.5).
fn parse_total_number(joined: &str) -> Option<f64> {
    let toks: Vec<&str> = joined.split([' ', '\u{a0}']).filter(|t| !t.trim().is_empty()).collect();
    // числовой токен: цифры/разделители + допустимый буквенный хвост («руб», «P»)
    let clean = |t: &str| -> Option<String> {
        let t = t.trim();
        let core: String = t
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == ',' || *c == '.')
            .collect();
        let rest: String = t.chars().skip(core.len()).collect();
        let rest_ok = rest.chars().all(|c| c.is_alphabetic() || c == '.');
        if core.is_empty() || !rest_ok {
            None
        } else {
            Some(core)
        }
    };
    // правый числовой токен
    let mut i = toks.len();
    while i > 0 && clean(toks[i - 1]).is_none() {
        i -= 1;
    }
    if i == 0 {
        return None;
    }
    let right = clean(toks[i - 1])?;
    let (int_part, dec) = match right.find([',', '.']) {
        Some(p) => (right[..p].to_string(), Some(right[p..].to_string())),
        None => (right.clone(), None),
    };
    if int_part.is_empty() {
        return None;
    }
    if int_part.len() > 3 {
        return parse_number(&right);
    }
    let mut groups: Vec<String> = vec![int_part];
    i -= 1;
    // средние группы — строго по 3 цифры
    while i > 0 {
        if let Some(c) = clean(toks[i - 1]) {
            if c.len() == 3 && !c.contains(',') && !c.contains('.') {
                groups.push(c);
                i -= 1;
                continue;
            }
        }
        break;
    }
    // ведущая группа 1–3 цифры: с дробью — соседнюю слева; без дроби —
    // только при наличии средней группы (иначе это № раздела/позиции)
    let has_middle = groups.len() > 1;
    if i > 0 && (dec.is_some() || has_middle) {
        if let Some(c) = clean(toks[i - 1]) {
            if (1..=3).contains(&c.len()) && !c.contains(',') && !c.contains('.') {
                groups.push(c);
            }
        }
    }
    let mut digits = String::new();
    for g in groups.iter().rev() {
        digits.push_str(g);
    }
    let mut s = digits;
    if let Some(d) = dec {
        s.push_str(&d);
    }
    parse_number(&s)
}

/// Итоговая сумма из табличной строки: берём крайнюю правую числовую
/// ЯЧЕЙКУ (граница ячейки отделяет подпись от числа — в отличие от
/// склейки всей строки). Число, разрезанное границей колонок на куски
/// («1» | «341 653P»), собирается из соседних числовых ячеек.
fn total_from_row(row: &DocRow) -> Option<f64> {
    // терпимый разбор ячейки: буквенный мусор срезаем, пробелы внутри — ок
    let lenient = |t: &str| -> Option<f64> {
        let t = t.trim();
        if t.contains('(') || t.contains('%') || t.contains('=') {
            return None; // «(20%)» — ставка, не сумма
        }
        let cleaned: String = t
            .chars()
            .filter(|c| c.is_ascii_digit() || *c == ',' || *c == '.' || *c == ' ' || *c == '\u{a0}')
            .collect();
        parse_number(cleaned.trim())
    };
    let right_idx = (0..row.cells.len())
        .rev()
        .find(|&i| lenient(&row.cells[i].2).is_some())?;
    // фрагменты разрезанного числа — чистые короткие цифровые ячейки слева
    let mut parts: Vec<String> = vec![row.cells[right_idx].2.trim().to_string()];
    let mut j = right_idx;
    while j > 0 {
        j -= 1;
        let prev = row.cells[j].2.trim();
        if !prev.is_empty()
            && prev.chars().all(|c| c.is_ascii_digit())
            && (1..=3).contains(&prev.chars().count())
        {
            parts.insert(0, prev.to_string());
        } else {
            break;
        }
    }
    lenient(&parts.join(" "))
}

// ---------- ресурсы ----------

fn resource_kind(lower: &str) -> Option<ResourceKind> {
    // сокращения выкладок Гранд-Сметы: «НР (4596,89 руб.): 80%*0,94 от ФОТ»,
    // «СП (2750,8 руб.): 45% от ФОТ»
    if lower.starts_with("нр (")
        || lower.starts_with("нр(")
        || lower.starts_with("нр ")
        || lower.contains("накладные расходы")
    {
        return Some(ResourceKind::Overhead);
    }
    if lower.starts_with("сп (")
        || lower.starts_with("сп(")
        || lower.starts_with("сп ")
        || lower.contains("сметная прибыль")
    {
        return Some(ResourceKind::Profit);
    }
    if lower.contains("затраты труда рабочих") || lower.contains("затраты труда (строителей)") {
        return Some(ResourceKind::Labor);
    }
    if lower.contains("труда машинистов") {
        return Some(ResourceKind::MachineLabor);
    }
    if lower.contains("эксплуатация машин") || lower.contains("эксплуатация строительных") {
        return Some(ResourceKind::Machine);
    }
    if lower.contains("транспортные расходы") || lower.contains("перевозк") {
        return Some(ResourceKind::Transport);
    }
    if lower.contains("материалы") {
        return Some(ResourceKind::Material);
    }
    None
}

/// Число в скобках: «НР (4596,89 руб.): 80%*0,94 от ФОТ» → 4596.89.
fn paren_number(s: &str) -> Option<f64> {
    let open = s.find('(')?;
    let close = open + s[open..].find(')')?;
    let inner: String = s[open + 1..close]
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == ',' || *c == '.')
        .collect();
    if inner.is_empty() {
        return None;
    }
    parse_number(&inner)
}

/// Строка-шкала номеров колонок выкладок ГС/Смета.RU: «1 2 3 … 11».
/// Допускаем до двух аномалий (склейка соседних номеров «9 10»→«910» при
/// порезке колонок, непрочитанный номер) — главное: числа начинаются с 1,
/// идут подряд и их не меньше пяти.
fn is_column_scale_row(row: &DocRow) -> bool {
    let mut smalls: Vec<i64> = Vec::new();
    let mut outliers = 0usize;
    for (_, _, t) in &row.cells {
        let t = t.trim();
        if t.is_empty() {
            continue;
        }
        match parse_number(t) {
            Some(v) if v.fract() == 0.0 && (0.0..=40.0).contains(&v) => smalls.push(v as i64),
            Some(_) => outliers += 1,
            None => return false,
        }
    }
    if smalls.len() < 5 || outliers > 1 {
        return false;
    }
    smalls.sort_unstable();
    smalls.dedup();
    if smalls.first() != Some(&1) {
        return false;
    }
    let span = *smalls.last().unwrap() as usize;
    smalls.len() + 2 >= span.max(5)
}

/// Название раздела из подписи итога: «ИТОГО ФУНДАМЕНТ расчетное $ 9847»
/// → «ФУНДАМЕНТ» (слова до числа/заполнителя).
fn totals_name_hint(joined: &str) -> Option<String> {
    let words: Vec<&str> = joined.split_whitespace().collect();
    let first = words.first()?.trim_matches(|c: char| !c.is_alphanumeric());
    let fl = first.to_lowercase();
    let rest_words: Vec<&str> = if fl == "итого" || fl == "всего" {
        words[1..].to_vec()
    } else if fl.starts_with("итого") {
        // «ИТОГО:» / склейка — редкость, считаем первым словом названия
        vec![words[0]]
    } else {
        return None;
    };
    let mut parts: Vec<&str> = Vec::new();
    for w in rest_words {
        let wl = w
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if w.contains(|c: char| c.is_ascii_digit())
            || w.contains('$')
            || w.contains('=')
            || w.contains(':')
        {
            break;
        }
        if matches!(wl.as_str(), "расчетное" | "расчётное" | "по" | "факту" | "с" | "ндс") {
            break;
        }
        parts.push(w);
    }
    let hint: String = parts.join(" ");
    if hint.chars().filter(|c| c.is_alphabetic()).count() < 3 {
        return None;
    }
    Some(hint)
}

/// Голый заголовок раздела: короткая строка без цифр и двоеточий («Кладка»).
fn bare_title(joined: &str) -> bool {
    let t = joined.trim();
    let n = t.chars().count();
    if !(3..=40).contains(&n) {
        return false;
    }
    if t.contains(':') || t.contains('(') || t.contains('№') || t.contains('%') {
        return false;
    }
    if t.chars().any(|c| c.is_ascii_digit()) {
        return false;
    }
    if t.chars().filter(|c| c.is_alphabetic()).count() < 3 {
        return false;
    }
    if is_unit(t) {
        return false;
    }
    let lower = t.to_lowercase();
    if totals_kind_text(&lower).is_some() || meta_field(&lower).is_some() {
        return false;
    }
    true
}

/// Строка целиком из чисел (допустимы буквенные хвосты «руб», «P»).
fn pure_numeric_row(row: &DocRow) -> bool {
    let mut any = false;
    for (_, _, t) in &row.cells {
        let t = t.trim();
        if t.is_empty() {
            continue;
        }
        let cleaned: String = t
            .chars()
            .filter(|c| c.is_ascii_digit() || *c == ',' || *c == '.' || *c == ' ' || *c == '\u{a0}')
            .collect();
        if parse_number(cleaned.trim()).is_none() {
            return false;
        }
        any = true;
    }
    any
}

fn lower_of(s: &str) -> String {
    s.to_lowercase()
}

/// Отладочная печать классификации строк: RUSTOCR_ESTIMATE_DEBUG=1.
fn dtag(tag: &str, joined: &str) {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *V.get_or_init(|| std::env::var("RUSTOCR_ESTIMATE_DEBUG").is_ok()) {
        let cut: String = joined.chars().take(90).collect();
        eprintln!("[est:{:>9}] {}", tag, cut);
    }
}

fn section_header(row: &DocRow) -> Option<(Option<String>, String, bool)> {
    // (номер, название, это подраздел)
    let joined = row.joined();
    let trimmed = joined.trim_start();
    let lower = trimmed.to_lowercase();
    let is_sub = lower.starts_with("подраздел");
    let is_sec = (lower.starts_with("раздел") || lower.starts_with("глава")) && !is_sub;
    if !is_sec && !is_sub {
        return None;
    }
    // срезаем слово «Раздел»/«РАЗДЕЛ»/«Подраздел» (любой регистр)
    let rest: String = trimmed
        .chars()
        .skip_while(|c| c.is_alphabetic())
        .collect::<String>()
        .trim_start()
        .to_string();
    let rest = rest.trim_start_matches("№").trim_start().to_string();
    let mut number = None;
    let mut title_rest = rest.clone();
    let num_end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(rest.len());
    if num_end > 0 {
        let maybe = rest[..num_end].trim_end_matches('.');
        if !maybe.is_empty() && maybe.chars().all(|c| c.is_ascii_digit() || c == '.') {
            number = Some(maybe.to_string());
            title_rest = rest[num_end..]
                .trim_start_matches(|c: char| matches!(c, '.' | ')' | ':' | ' ' | '-'))
                .to_string();
        }
    }
    let title = if title_rest.trim().is_empty() {
        "(без названия)".to_string()
    } else {
        title_rest.trim().to_string()
    };
    Some((number, title, is_sub))
}

// ---------- шапка документа ----------

fn meta_field(lower: &str) -> Option<&'static str> {
    if lower.contains("заказчик") {
        return Some("customer");
    }
    if lower.contains("подрядчик") || lower.contains("исполнител") {
        return Some("contractor");
    }
    if lower.contains("объект") {
        return Some("object");
    }
    if lower.contains("дата") || lower.contains("составлен") {
        return Some("date");
    }
    if lower.contains("смета") || lower.contains("сметный расчёт") || lower.contains("сметный расчет") {
        return Some("title");
    }
    None
}

fn split_label_value(joined: &str) -> (String, String) {
    if let Some(pos) = joined.find(':') {
        let label = joined[..pos].trim().to_string();
        let value = joined[pos + 1..].trim().to_string();
        if !value.is_empty() {
            return (label, value);
        }
    }
    (joined.trim().to_string(), String::new())
}

/// Арифметически согласованная тройка (кол-во, цена, сумма) из чисел
/// строки: a×b ≈ c (допуск 2% + 0,5 руб. на округления). Из всех вариантов
/// берём с наибольшей суммой — в выкладках ГС/РИК «всего по позиции» больше
/// служебных «в т.ч. оплаты труда» и трудоёмкости.
fn arithmetic_triple(numbers: &[(usize, f64, String)]) -> Option<(Num, Num, Num)> {
    let mut best: Option<(f64, usize, usize, usize)> = None; // (|сумма|, i, j, k)
    for i in 0..numbers.len() {
        for j in (i + 1)..numbers.len() {
            for k in (j + 1)..numbers.len() {
                let (a, b, c) = (numbers[i].1, numbers[j].1, numbers[k].1);
                if a == 0.0 || b == 0.0 || c == 0.0 {
                    continue;
                }
                let ok = (a * b - c).abs() <= 0.02 * c.abs() + 0.5;
                if !ok {
                    continue;
                }
                let key = c.abs();
                if best.map(|(bk, ..)| key > bk).unwrap_or(true) {
                    best = Some((key, i, j, k));
                }
            }
        }
    }
    best.map(|(_, i, j, k)| {
        (
            Num::new(numbers[i].1, numbers[i].2.clone()),
            Num::new(numbers[j].1, numbers[j].2.clone()),
            Num::new(numbers[k].1, numbers[k].2.clone()),
        )
    })
}

/// Оценка суммы позиции для сверки итогов (в parser'e, до валидатора).
fn est_position_total(p: &Position) -> f64 {
    if let Some(t) = &p.total {
        t.value
    } else if let (Some(q), Some(pr)) = (&p.quantity, &p.price) {
        q.value * pr.value
    } else {
        0.0
    }
}

fn est_section_sum(s: &Section) -> f64 {
    if let Some(t) = &s.total {
        return t.value;
    }
    s.positions.iter().map(est_position_total).sum::<f64>()
        + s
            .subsections
            .iter()
            .map(est_section_sum)
            .sum::<f64>()
        + s
            .resources
            .iter()
            .filter_map(|r| r.total.as_ref().map(|t| t.value))
            .sum::<f64>()
}

// ---------- парсер документа ----------

pub fn parse_document(rows: Vec<DocRow>) -> EstimateDocument {
    let mut p = Parser::default();
    // предпросмотр: голое «Итого»/«Всего», за которым ≤3 строк идут «НДС» или
    // «ВСЕГО по смете», — итог документа (финальная сводка Гранд-Сметы), а не
    // итог текущего раздела
    let doc_total = mark_doc_totals(&rows);
    for (i, row) in rows.into_iter().enumerate() {
        p.next_is_doc_total = doc_total[i];
        p.feed(row);
    }
    p.finish()
}

fn mark_doc_totals(rows: &[DocRow]) -> Vec<bool> {
    let mut out = vec![false; rows.len()];
    for i in 0..rows.len() {
        // «Итого 2031228,96 2389,4»: первое слово — итоговое, дальше только числа
        let lower = rows[i].lower();
        let toks: Vec<&str> = lower.split_whitespace().collect();
        let head = toks.first().copied().unwrap_or_default();
        if head != "итого" && head != "всего" {
            continue;
        }
        let rest_numeric = toks[1..]
            .iter()
            .all(|t| parse_number(t.trim_end_matches('.')).is_some());
        if !rest_numeric {
            continue;
        }
        for w in rows.iter().skip(i + 1).take(3) {
            let wl = w.lower();
            if wl.starts_with("ндс") || wl.starts_with("всего с ндс") || wl.contains("по смете") {
                out[i] = true;
                break;
            }
            if totals_kind_text(&wl).is_some() {
                break; // другой итог ближе — это не финальная сводка
            }
        }
    }
    out
}

#[derive(Default)]
struct Parser {
    meta: Meta,
    sections: Vec<Section>,
    cur_section: Option<Section>,
    cur_sub: Option<Section>,
    cur_position: Option<Position>,
    name_col: Option<usize>,
    roles: Vec<Option<Role>>,
    /// колонки чисел последней строки-позиции (для различения
    /// списочной строки от продолжения: у списочной единственное число
    /// в той же колонке, что у предыдущей позиции)
    prev_number_cols: Vec<usize>,
    saw_table_row: bool,
    saw_section: bool,
    totals: Totals,
    stats: ValidationStats,
    unparsed: Vec<String>,
    pages: usize,
    /// итог, у которого сумма не в этой строке — ждём число ниже
    pending_total: Option<(TotalKind, Option<f64>, Option<String>)>,
    /// сколько строк можно ждать сумму отложенного итога
    /// только что был итог/его заполнители — голая строка после = новый раздел
    after_total: bool,
    /// метка из предпросмотра: текущая строка — финальное «Итого» документа
    next_is_doc_total: bool,
}

impl Parser {
    fn feed(&mut self, row: DocRow) {
        if row.columnar {
            self.stats.table_rows += 1;
        }
        if row.page > self.pages {
            self.pages = row.page;
        }
        // заполнители итогов («расчетное», «по факту», «$») — пропускаем
        let joined = row.joined();
        let lower = joined.to_lowercase();
        if matches!(
            lower.trim(),
            "расчетное" | "расчётное" | "по факту" | "$" | "-" | "—" | "=" | ""
        ) {
            return; // после итога — продолжаем ждать сумму
        }
        // отложенный итог: строка целиком из чисел — это его сумма
        if let Some((kind, rate, hint)) = self.pending_total.take() {
            if pure_numeric_row(&row) {
                let num = total_from_row(&row)
                    .or_else(|| parse_total_number(&joined));
                if let Some(v) = num {
                    if let Some(h) = hint {
                        self.rename_implicit_section(&h);
                    }
                    dtag("итог+сумма", &joined);
                    self.apply_total(kind, rate, Some(v), &joined);
                    self.after_total = true;
                    return;
                }
            }
            // не дождались — итог отменяется, строка обработается ниже
        }
        if row.columnar {
            self.feed_columnar(row);
        } else {
            self.feed_line(row);
        }
    }

    fn feed_line(&mut self, row: DocRow) {
        let joined = row.joined();
        if joined.trim().is_empty() {
            return;
        }
        // раздел может быть и полноширинной строкой
        if let Some((number, title, is_sub)) = section_header(&row) {
            self.start_section(number, title, is_sub);
            return;
        }
        // итоги вне таблицы: «Итого по смете 123 руб.»
        if self.saw_table_row {
            let doc_mark = std::mem::take(&mut self.next_is_doc_total);
            if let Some((kind0, rate)) = totals_kind(&row) {
                let kind = if doc_mark && kind0 == TotalKind::Section {
                    TotalKind::Doc
                } else {
                    kind0
                };
                let hint = totals_name_hint(&lower_of(&joined));
                self.flush_position();
                let num = self.resolve_total_num(kind, &row, &joined);
                if num.is_none() {
                    // сумма ниже по документу — ждём
                    if let Some(h) = &hint {
                        self.rename_implicit_section(h);
                    }
                    self.pending_total = Some((kind, rate, hint));
                    self.after_total = true;
                    dtag("итог-ждёт", &joined);
                    return;
                }
                self.apply_total(kind, rate, num, &joined);
                dtag(&format!("итог={:?}", num), &joined);
                self.after_total = true;
                return;
            }
            // голый заголовок после итога — новый раздел («Кладка», «Монолит»)
            if self.after_total && bare_title(&joined) {
                dtag("раздел-голый", &joined);
                self.start_section(None, joined.trim().to_string(), false);
                self.after_total = false;
                return;
            }
        }
        // голый заголовок до первой табличной строки — раздел («Материалы»)
        if !self.saw_table_row && !self.saw_section && bare_title(&joined) {
            dtag("раздел-ранний", &joined);
            self.start_section(None, joined.trim().to_string(), false);
            return;
        }
        // шапка / заметки
        if !self.saw_table_row && !self.saw_section {
            self.feed_meta_line(&joined);
        } else {
            self.push_unparsed(&joined);
        }
    }

    fn feed_meta_line(&mut self, joined: &str) {
        let lower = joined.to_lowercase();
        let (_, value) = split_label_value(joined);
        match meta_field(&lower) {
            Some("customer") => self.meta.customer = Some(non_empty(value, joined)),
            Some("contractor") => self.meta.contractor = Some(non_empty(value, joined)),
            Some("object") => self.meta.object = Some(non_empty(value, joined)),
            Some("date") => self.meta.date = Some(non_empty(value, joined)),
            Some("title") => {
                if self.meta.title.is_none() {
                    // «Локальная смета № 02-01-01 на …»
                    if let Some(npos) = joined.find('№') {
                        let tail = joined[npos + '№'.len_utf8()..].trim();
                        let n: String =
                            tail.chars().take_while(|c| !c.is_whitespace()).collect();
                        if !n.is_empty() {
                            self.meta.number = Some(n);
                        }
                    }
                    self.meta.title = Some(non_empty(value, joined));
                }
            }
            _ => {
                if self.meta.notes.len() < 12 && joined.trim().len() > 3 {
                    self.meta.notes.push(joined.trim().to_string());
                }
            }
        }
    }

    fn feed_columnar(&mut self, row: DocRow) {
        let joined = row.joined();
        if joined.trim().is_empty() {
            return;
        }
        self.saw_table_row = true;

        // 1) заголовок раздела внутри сетки
        if let Some((number, title, is_sub)) = section_header(&row) {
            self.start_section(number, title, is_sub);
            return;
        }

        // 2) шапка таблицы → роли колонок
        if is_header_row(&row) {
            self.learn_roles(&row);
            return;
        }

        // 3) итоги
        let doc_mark = std::mem::take(&mut self.next_is_doc_total);
        if let Some((kind0, rate)) = totals_kind(&row) {
            // метка предпросмотра: финальное «Итого …» перед НДС/«ВСЕГО по
            // смете» — итог документа, а не текущего раздела
            let kind = if doc_mark && kind0 == TotalKind::Section {
                TotalKind::Doc
            } else {
                kind0
            };
            let hint = totals_name_hint(&lower_of(&joined));
            self.flush_position();
            let num = self.resolve_total_num(kind, &row, &joined);
            if let Some(h) = &hint {
                self.rename_implicit_section(h);
            }
            if num.is_none() {
                self.pending_total = Some((kind, rate, hint));
                self.after_total = true;
                dtag("итог-ждёт", &joined);
                return;
            }
            self.apply_total(kind, rate, num, &joined);
            dtag(&format!("итог={:?}", num), &joined);
            self.after_total = true;
            return;
        }

        // 3а) строка-шкала номеров колонок («1 2 3 … 11» выкладок ГС) — не
        // позиция и не итог
        if is_column_scale_row(&row) {
            dtag("шкала кол", &joined);
            return;
        }

        // 3б) продолжение позиции: без № и шифра, текст в колонке
        // наименования начинается со строчной буквы или скобки —
        // «(10 м³ основания) 82,01 3,65 25,55», «траншеях шириной до 2 м…»
        if self.cur_position.is_some() && self.is_position_continuation(&row) {
            if let Some(pos) = self.cur_position.as_mut() {
                // чисто текстовое продолжение (без чисел и единиц) —
                // склеиваем в название: «Устройство монолитного» +
                // «фундамента из бетона В22,5»
                let has_numbers = row
                    .cells
                    .iter()
                    .any(|(_, _, t)| parse_number(t.trim()).is_some());
                if !has_numbers {
                    let text = row
                        .cells
                        .iter()
                        .filter_map(|(_, _, t)| {
                            let t = t.trim();
                            (!t.is_empty() && parse_number(t).is_none() && !is_unit(t))
                                .then_some(t)
                        })
                        .next()
                        .map(|s| s.to_string());
                    if let Some(text) = text {
                        if !pos.name.starts_with('(') {
                            pos.name.push(' ');
                            pos.name.push_str(&text);
                        }
                    }
                }
                pos.raw.push(joined.clone());
                if pos.pages.last() != Some(&row.page) {
                    pos.pages.push(row.page);
                }
            }
            dtag("продолж-е", &joined);
            return;
        }

        // 4) позиция
        if let Some(pos) = self.parse_position(&row) {
            dtag("позиция", &joined);
            self.after_total = false;
            self.flush_position();
            self.cur_position = Some(pos);
            return;
        }

        // 5) ресурс
        if let Some(res) = self.parse_resource(&row) {
            dtag("ресурс", &joined);
            self.after_total = false;
            self.attach_resource(res, &joined);
            return;
        }

        // 6) маркер блока / продолжение названия
        if self.try_continuation_or_marker(&row, &joined) {
            return;
        }

        self.push_unparsed(&joined);
        dtag("не разобрано", &joined);
    }

    // --- роли колонок ---

    fn learn_roles(&mut self, row: &DocRow) {
        for (col, span, text) in &row.cells {
            if let Some(role) = role_of_header(text) {
                // объединённая ячейка-группа («Стоимость материалов, машин и
                // механизмов» на 2 колонки) описывает блок, а не отдельную
                // колонку: роль вешаем на ПОСЛЕДНЮЮ колонку спана — там
                // под-заголовок «Итого»/сумма, а «За ед» стартовой колонки
                // не затирается (двухуровневые шапки коммерческих смет)
                let target = if *span > 1 { col + span - 1 } else { *col };
                if self.roles.len() <= target {
                    self.roles.resize_with(target + 1, || None);
                }
                if self.roles[target].is_none() {
                    self.roles[target] = Some(role);
                }
            }
        }
    }

    fn role_at(&self, col: usize) -> Option<Role> {
        self.roles.get(col).copied().flatten()
    }

    /// Продолжение позиции на следующей строке: нет ни №, ни шифра расценки,
    /// это не ресурс (НР/СП/труд), а текст в колонке наименования начинается
    /// со строчной буквы или скобки — перенос названия или вторые строки
    /// данных многострочных позиций ГС/РИК/Смета.RU.
    fn is_position_continuation(&self, row: &DocRow) -> bool {
        if self.cell_int_line(row).is_some() {
            return false;
        }
        if row
            .cells
            .iter()
            .any(|(_, _, t)| position_cipher(t).is_some())
        {
            return false;
        }
        if resource_kind(&row.lower()).is_some() {
            return false;
        }
        // списочная строка «труба для заезда | 54000»: одно число,
        // и оно в той же колонке, что числа предыдущей позиции — это
        // НОВАЯ позиция, а не продолжение (ГС-продолжения несут числа
        // в других колонках — трудоёмкость, маш.-ч)
        let numbers = row.numbers();
        if numbers.len() == 1 {
            let (ncol, _, _) = numbers[0];
            if self.prev_number_cols.contains(&ncol) {
                return false;
            }
        }
        let first_text = row.cells.iter().find(|(_, _, t)| {
            let t = t.trim();
            !t.is_empty() && parse_number(t).is_none() && !is_unit(t)
        });
        let Some((col, _, text)) = first_text else {
            return false;
        };
        let t = text.trim();
        let starts_lower = t.chars().next().map(|c| c.is_lowercase()).unwrap_or(false);
        if !starts_lower && !t.starts_with('(') {
            return false;
        }
        match self.name_col {
            Some(nc) => *col == nc,
            None => true,
        }
    }

    /// Выбор числа-итога из строки. В выкладках ГС рядом с суммой идут
    /// трудоёмкость (чел.-ч) и «занятые обслуживанием машин» — правое число
    /// оказывается неверным. Если чисел несколько, берём ближайшее к
    /// Σ позиций (для документа — к Σ разделов и Σ+НДС).
    fn resolve_total_num(&self, kind: TotalKind, row: &DocRow, joined: &str) -> Option<f64> {
        let fallback = total_from_row(row).or_else(|| parse_total_number(joined));
        let mut candidates: Vec<f64> = row
            .cells
            .iter()
            .filter_map(|(_, _, t)| parse_number(t.trim()))
            .collect();
        candidates.retain(|v| *v != 0.0);
        candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        if candidates.len() <= 1 {
            return fallback;
        }
        let mut anchors: Vec<f64> = Vec::new();
        match kind {
            TotalKind::Doc | TotalKind::DocWithVat => {
                let mut s: f64 = self.sections.iter().map(est_section_sum).sum();
                if let Some(sec) = &self.cur_section {
                    s += est_section_sum(sec);
                }
                if let Some(sub) = &self.cur_sub {
                    s += est_section_sum(sub);
                }
                if s > 0.0 {
                    anchors.push(s);
                    if let Some(vat) = &self.totals.vat {
                        anchors.push(s + vat.value);
                    }
                }
            }
            TotalKind::Section | TotalKind::Sub => {
                if let Some(t) = self.cur_sub.as_ref().or(self.cur_section.as_ref()) {
                    let s = est_section_sum(t);
                    if s > 0.0 {
                        anchors.push(s);
                    }
                }
            }
            TotalKind::Vat => {}
        }
        if anchors.is_empty() {
            return fallback;
        }
        let mut best: Option<(f64, f64)> = None; // (расстояние, значение)
        for &c in &candidates {
            for &a in &anchors {
                let d = (c - a).abs();
                if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                    best = Some((d, c));
                }
            }
        }
        // ни один кандидат не похож на Σ (мусор от скана) — прежний путь
        let fits = best
            .map(|(d, _)| anchors.iter().any(|a| d <= 0.35 * a.max(1.0)))
            .unwrap_or(false);
        if fits {
            best.map(|(_, v)| v)
        } else {
            fallback
        }
    }

    // --- позиции ---

    fn cell_int_line(&self, row: &DocRow) -> Option<(usize, u32)> {
        let line_col = (0..self.roles.len()).find(|c| self.roles[*c] == Some(Role::Line));
        let mut first_nonempty: Option<usize> = None;
        for (col, _, text) in &row.cells {
            if text.trim().is_empty() {
                continue;
            }
            if first_nonempty.is_none() {
                first_nonempty = Some(*col);
            }
            // без роли смотрим только первую непустую ячейку
            let consider = match line_col {
                Some(lc) => *col == lc,
                None => Some(*col) == first_nonempty,
            };
            if !consider {
                if line_col.is_none() {
                    return None; // первая непустая ячейка — не №
                }
                continue;
            }
            let t = text.trim().trim_end_matches('.');
            if !t.is_empty() && t.len() <= 4 && t.chars().all(|c| c.is_ascii_digit()) {
                if let Ok(v) = t.parse::<u32>() {
                    if (1..=9999).contains(&v) {
                        return Some((*col, v));
                    }
                }
            }
            if line_col.is_none() {
                return None;
            }
        }
        None
    }

    fn parse_position(&mut self, row: &DocRow) -> Option<Position> {
        let mut numbers = row.numbers();
        // хвостовые нули — служебные колонки коммерческих смет («в т.ч. …» = 0):
        // «Разметка | м² | 1 | 150 | 150 | 0» → кол-во 1, цена 150, сумма 150;
        // «… | 14 | 156,8 | 0» (сумма не прочиталась) → кол-во 14, цена 156,8
        let mut dropped_zero = false;
        while numbers.len() >= 3 {
            let (_, v, _) = numbers.last().unwrap();
            if *v == 0.0 {
                numbers.pop();
                dropped_zero = true;
            } else {
                break;
            }
        }
        let line = self.cell_int_line(row);
        let line_col = line.map(|(c, _)| c);

        // шифр расценки в любой ячейке
        let mut cipher: Option<String> = None;
        let mut cipher_col: Option<usize> = None;
        for (i, (col, _, text)) in row.cells.iter().enumerate() {
            if Some(*col) == line_col {
                continue;
            }
            if let Some(c) = position_cipher(text) {
                cipher_col = Some(*col);
                // метка «ГЭСН» отдельной ячейкой перед кодом
                if i > 0 {
                    let prev = row.cells[i - 1].2.trim().to_lowercase();
                    if matches!(
                        prev.as_str(),
                        "гэсн" | "фер" | "тер" | "мэр" | "оэр" | "фсрр" | "фссц" | "тссц"
                    ) {
                        cipher = Some(format!("{} {}", row.cells[i - 1].2.trim(), c));
                    }
                }
                cipher = cipher.or(Some(c));
                break;
            }
        }

        let is_position = match (&line, &cipher) {
            (Some(_), _) if !numbers.is_empty() || cipher.is_some() || row.cells.len() >= 3 => true,
            (None, Some(_)) => true,
            // «Арматура | 195000» — списочная смета в 2 колонки: текст + число
            (None, None)
                if row.cells.iter().filter(|(_, _, t)| !t.trim().is_empty()).count() == 2
                    && resource_kind(&row.lower()).is_none() =>
            {
                let non_empty: Vec<&(usize, usize, String)> = row
                    .cells
                    .iter()
                    .filter(|(_, _, t)| !t.trim().is_empty())
                    .collect();
                let (a, b) = (non_empty[0].2.trim(), non_empty[1].2.trim());
                let a_text = a.chars().filter(|c| c.is_alphabetic()).count() >= 3
                    && parse_number(a).is_none()
                    && !is_unit(a);
                a_text && parse_number(b).is_some()
            }
            // коммерческая таблица без № и шифров: [наименование | ед | числа…]
            // (и журнал расходов: дата | поставщик | сумма | комментарий)
            (None, None) if resource_kind(&row.lower()).is_none() => {
                let last_num_col = row.numbers().last().map(|(c, _, _)| *c);
                let has_code = row.cells.iter().any(|(col, _, t)| {
                    resource_code(t).is_some()
                        && !(t.trim().len() == 6 && Some(*col) == last_num_col)
                });
                if has_code {
                    false
                } else {
                    let first = row
                        .cells
                        .iter()
                        .find(|(_, _, t)| !t.trim().is_empty())
                        .map(|(_, _, t)| t.trim())
                        .unwrap_or("");
                    // дата в первой ячейке — журнал расходов: строка всё равно
                    // позиция (дата | поставщик | сумма | комментарий)
                    let texty = (first.chars().filter(|c| c.is_alphabetic()).count() >= 3
                        || is_date(first))
                        && parse_number(first).is_none()
                        && !is_unit(first);
                    texty
                        && numbers.len() >= 1
                        && row.cells.iter().filter(|(_, _, t)| !t.trim().is_empty()).count() >= 3
                }
            }
            _ => false,
        };
        if !is_position {
            return None;
        }

        // название: колонка роли Name, иначе самая длинная текстовая ячейка
        let name_col_hint = (0..self.roles.len()).find(|c| self.roles[*c] == Some(Role::Name));
        let mut name: Option<(usize, String)> = None;
        if let Some(nc) = name_col_hint {
            if let Some((col, _, text)) = row.cells.iter().find(|(c, _, _)| *c == nc) {
                if !text.trim().is_empty() {
                    name = Some((*col, text.trim().to_string()));
                }
            }
        }
        if name.is_none() {
            let mut best: Option<(usize, String)> = None;
            for (col, _, text) in &row.cells {
                if Some(*col) == line_col || Some(*col) == cipher_col {
                    continue;
                }
                if position_cipher(text).is_some() || is_unit(text) || parse_number(text).is_some() {
                    continue;
                }
                let letters = text.chars().filter(|c| c.is_alphabetic()).count();
                if letters >= 3
                    && best
                        .as_ref()
                        .map(|(_, t)| t.chars().count())
                        .unwrap_or(0)
                        < text.trim().chars().count()
                {
                    best = Some((*col, text.trim().to_string()));
                }
            }
            name = best;
        }
        let (name_col, name_text) = name.unwrap_or((usize::MAX, "(без названия)".to_string()));
        if name_col != usize::MAX {
            self.name_col = Some(name_col);
        }

        // единица измерения
        let unit = if let Some(uc) = (0..self.roles.len()).find(|c| self.roles[*c] == Some(Role::Unit))
        {
            row.cells
                .iter()
                .find(|(c, _, _)| *c == uc)
                .and_then(|(_, _, t)| if is_unit(t) { Some(normalize_unit(t)) } else { None })
        } else {
            row.cells
                .iter()
                .filter(|(c, _, _)| Some(*c) != line_col && Some(*c) != cipher_col)
                .find(|(_, _, t)| is_unit(t))
                .map(|(_, _, t)| normalize_unit(t))
        };

        let (qty, price, total) = self.pick_position_numbers(&numbers, dropped_zero);
        self.prev_number_cols = numbers.iter().map(|(c, _, _)| *c).collect();

        Some(Position {
            line: line.map(|(_, v)| v),
            cipher,
            name: name_text,
            unit,
            quantity: qty,
            price,
            total,
            resources: Vec::new(),
            pages: vec![row.page],
            raw: vec![row.joined()],
        })
    }

    fn pick_position_numbers(
        &self,
        numbers: &[(usize, f64, String)],
        dropped_zero: bool,
    ) -> (Option<Num>, Option<Num>, Option<Num>) {
        let by_role = |role: Role| -> Option<Num> {
            for (col, v, raw) in numbers {
                if self.role_at(*col) == Some(role) {
                    return Some(Num::new(*v, raw.clone()));
                }
            }
            None
        };
        // итог позиции — в ПРАВОЙ колонке с ролью Total: при
        // двухуровневой шапке «За ед | Итого» ×2 + «Итого» промежуточные
        // суммы материалов/работ тоже помечены Total, позиционный итог —
        // крайний правый
        let total_by_role = |numbers: &[(usize, f64, String)]| -> Option<Num> {
            let mut last = None;
            for (col, v, raw) in numbers {
                if self.role_at(*col) == Some(Role::Total) && *v != 0.0 {
                    last = Some(Num::new(*v, raw.clone()));
                }
            }
            last
        };
        // цена — первый ненулевой Price (при «За ед»×2 нулевая цена
        // материалов означает строку работ)
        let price_by_role = |numbers: &[(usize, f64, String)]| -> Option<Num> {
            for (col, v, raw) in numbers {
                if self.role_at(*col) == Some(Role::Price) && *v != 0.0 {
                    return Some(Num::new(*v, raw.clone()));
                }
            }
            None
        };
        let qty = by_role(Role::Qty);
        let price = price_by_role(numbers);
        let total = total_by_role(numbers);
        if qty.is_some() || price.is_some() || total.is_some() {
            return (qty, price, total);
        }
        // ролей нет: позиционно. ≥3 числа: (кол-во, цена, сумма); 2: (кол-во,
        // цена) если хвостовой ноль сброшен — суммы не было, иначе (кол-во,
        // сумма); 1: сумма
        match numbers.len() {
            0 => (None, None, None),
            1 => {
                let (_, v, r) = &numbers[0];
                (None, None, Some(Num::new(*v, r.clone())))
            }
            2 => {
                let (_, v1, r1) = &numbers[0];
                let (_, v2, r2) = &numbers[1];
                if dropped_zero {
                    // «… | 14 | 156,8 | 0» — сумма не прочиталась
                    (
                        Some(Num::new(*v1, r1.clone())),
                        Some(Num::new(*v2, r2.clone())),
                        None,
                    )
                } else {
                    (Some(Num::new(*v1, r1.clone())), None, Some(Num::new(*v2, r2.clone())))
                }
            }
            _ => {
                // сначала арифметика: (кол-во) × (цена) ≈ (сумма) — устойчиво к
                // лишним служебным колонкам (№ пп, «в т.ч. оплата труда»,
                // трудоёмкость) выкладок ГС/РИК/Смета.RU: в «7 | 1240,49 | … |
                // 8683,43 | …» правильная тройка (7, 1240,49, 8683,43), а не
                // «первое и два последних» с № пп в роли количества
                if let Some((q, p, t)) = arithmetic_triple(numbers) {
                    return (Some(q), Some(p), Some(t));
                }
                let n = numbers.len();
                let (_, vq, rq) = &numbers[0];
                let (_, vp, rp) = &numbers[n - 2];
                let (_, vt, rt) = &numbers[n - 1];
                (
                    Some(Num::new(*vq, rq.clone())),
                    Some(Num::new(*vp, rp.clone())),
                    Some(Num::new(*vt, rt.clone())),
                )
            }
        }
    }

    // --- ресурсы ---

    fn parse_resource(&mut self, row: &DocRow) -> Option<Resource> {
        let joined = row.joined();
        let lower = joined.to_lowercase();
        let numbers = row.numbers();

        // ключевой ресурс («Затраты труда рабочих | чел.-ч | 123,45»)
        if let Some(kind) = resource_kind(&lower) {
            if numbers.is_empty() {
                return None; // заголовок блока («Материалы», «Накладные расходы»)
            }
            let unit = row
                .cells
                .iter()
                .find(|(_, _, t)| is_unit(t))
                .map(|(_, _, t)| normalize_unit(t));
            let name = row
                .cells
                .iter()
                .filter(|(_, _, t)| !is_unit(t) && parse_number(t).is_none() && t.trim().len() > 2)
                .max_by_key(|(_, _, t)| t.trim().chars().count())
                .map(|(_, _, t)| t.trim().to_string())
                .unwrap_or_else(|| joined.trim().to_string());
            // НР/СП/транспорт — одна цифра это итог; труд/машины — количество
            let cost_like = matches!(
                kind,
                ResourceKind::Overhead | ResourceKind::Profit | ResourceKind::Transport
            );
            // «НР (4596,89 руб.): 80%*0,94 от ФОТ» — сумма в скобках, а не в
            // процентах/коэффициентах; иначе — наибольшее число строки
            let (qty, price, total) = if cost_like {
                let v = paren_number(&joined).or_else(|| {
                    numbers
                        .iter()
                        .map(|(_, v, _)| *v)
                        .fold(f64::NEG_INFINITY, f64::max)
                        .max(-1.0)
                        .into()
                });
                match v {
                    Some(v) => (None, None, Some(Num::new(v, extract_number_raw(&joined).unwrap_or_default()))),
                    None => (None, None, None),
                }
            } else {
                match numbers.len() {
                    0 => (None, None, None),
                    1 => {
                        let (_, v, r) = &numbers[0];
                        (Some(Num::new(*v, r.clone())), None, None)
                    }
                    2 => {
                        let (_, v1, r1) = &numbers[0];
                        let (_, v2, r2) = &numbers[1];
                        (
                            Some(Num::new(*v1, r1.clone())),
                            None,
                            Some(Num::new(*v2, r2.clone())),
                        )
                    }
                    _ => {
                        let n = numbers.len();
                        let (_, vq, rq) = &numbers[0];
                        let (_, vp, rp) = &numbers[n - 2];
                        let (_, vt, rt) = &numbers[n - 1];
                        (
                            Some(Num::new(*vq, rq.clone())),
                            Some(Num::new(*vp, rp.clone())),
                            Some(Num::new(*vt, rt.clone())),
                        )
                    }
                }
            };
            return Some(Resource {
                kind,
                code: None,
                name,
                unit,
                quantity: qty,
                price,
                total,
            });
        }

        // код ресурса в ячейке: 6-значный «код машины» не должен быть
        // последним числом строки — иначе это сумма («… 110000 арматура»)
        let last_num_col = row.numbers().last().map(|(c, _, _)| *c);
        let mut code_col: Option<usize> = None;
        let mut code: Option<String> = None;
        for (col, _, text) in &row.cells {
            if resource_code(text).is_some() {
                if text.trim().len() == 6 && Some(*col) == last_num_col {
                    continue; // чистые 6 цифр в последней числовой ячейке — сумма
                }
                code = resource_code(text);
                code_col = Some(*col);
                break;
            }
        }
        let code = code?;
        let kind = if code.len() == 6 {
            ResourceKind::Machine
        } else {
            ResourceKind::Material
        };
        let name = row
            .cells
            .iter()
            .filter(|(c, _, t)| Some(*c) != code_col && !is_unit(t) && parse_number(t).is_none())
            .max_by_key(|(_, _, t)| t.trim().chars().count())
            .map(|(_, _, t)| t.trim().to_string())
            .filter(|t| t.chars().filter(|c| c.is_alphabetic()).count() >= 3);
        let unit = row
            .cells
            .iter()
            .find(|(_, _, t)| is_unit(t))
            .map(|(_, _, t)| normalize_unit(t));
        let (qty, price, total) = match numbers.len() {
            0 => (None, None, None),
            1 => {
                let (_, v, r) = &numbers[0];
                (Some(Num::new(*v, r.clone())), None, None)
            }
            2 => {
                let (_, v1, r1) = &numbers[0];
                let (_, v2, r2) = &numbers[1];
                (Some(Num::new(*v1, r1.clone())), None, Some(Num::new(*v2, r2.clone())))
            }
            _ => {
                let n = numbers.len();
                let (_, vq, rq) = &numbers[0];
                let (_, vp, rp) = &numbers[n - 2];
                let (_, vt, rt) = &numbers[n - 1];
                (
                    Some(Num::new(*vq, rq.clone())),
                    Some(Num::new(*vp, rp.clone())),
                    Some(Num::new(*vt, rt.clone())),
                )
            }
        };
        Some(Resource {
            kind,
            code: Some(code),
            name: name.unwrap_or_else(|| "(ресурс)".to_string()),
            unit,
            quantity: qty,
            price,
            total,
        })
    }

    fn attach_resource(&mut self, res: Resource, joined: &str) {
        if let Some(pos) = self.cur_position.as_mut() {
            pos.resources.push(res);
            pos.raw.push(joined.to_string());
            self.stats.resources += 1;
        } else {
            // ресурс вне позиции — на уровень раздела (напр., НР по разделу)
            self.ensure_section();
            let target = self.cur_sub.as_mut().or(self.cur_section.as_mut());
            if let Some(sec) = target {
                sec.resources.push(res);
                self.stats.resources += 1;
            }
        }
    }

    // --- маркеры и продолжения ---

    fn try_continuation_or_marker(&mut self, row: &DocRow, joined: &str) -> bool {
        let non_empty: Vec<&(usize, usize, String)> = row
            .cells
            .iter()
            .filter(|(_, _, t)| !t.trim().is_empty())
            .collect();
        if non_empty.len() != 1 {
            return false;
        }
        let (col, _, text) = non_empty[0];
        let lower = text.trim().to_lowercase();
        // маркеры блоков ресурсов и служебные строки
        if matches!(
            lower.as_str(),
            "материалы"
                | "машины и механизмы"
                | "трудовые ресурсы"
                | "средний разряд работы"
                | "средний разряд"
                | "машины"
                | "механизмы"
                | "итого по позиции"
                | "всего по позиции"
        ) {
            return true;
        }
        // продолжение названия позиции/ресурса (в колонке наименования)
        let name_col_ok = self.name_col.map(|nc| nc == *col).unwrap_or(true);
        let letters = text.trim().chars().filter(|c| c.is_alphabetic()).count();
        let is_pure_number = parse_number(text.trim()).is_some();
        if name_col_ok && !is_pure_number && letters >= 2 {
            if let Some(pos) = self.cur_position.as_mut() {
                if !pos.resources.is_empty() {
                    let r = pos.resources.last_mut().unwrap();
                    r.name.push(' ');
                    r.name.push_str(text.trim());
                } else if !pos.name.starts_with('(') {
                    pos.name.push(' ');
                    pos.name.push_str(text.trim());
                }
                pos.raw.push(joined.to_string());
                return true;
            }
        }
        false
    }

    // --- структура ---

    fn blank_section() -> Section {
        Section {
            number: None,
            title: "(без разделов)".to_string(),
            subsections: Vec::new(),
            positions: Vec::new(),
            resources: Vec::new(),
            total: None,
            pages: Vec::new(),
        }
    }

    /// Назвать безымянный раздел по подписи итога: «ИТОГО ФУНДАМЕНТ» →
    /// раздел получает название «ФУНДАМЕНТ».
    fn rename_implicit_section(&mut self, hint: &str) {
        let hint = hint.trim().trim_matches(|c: char| !c.is_alphanumeric());
        if hint.chars().count() < 3 {
            return;
        }
        if let Some(sec) = self.cur_section.as_mut() {
            if sec.number.is_none() && sec.title == "(без разделов)" {
                sec.title = hint.to_string();
            }
        }
    }

    fn ensure_section(&mut self) {
        if self.cur_section.is_none() {
            self.cur_section = Some(Self::blank_section());
        }
    }

    fn start_section(&mut self, number: Option<String>, title: String, is_sub: bool) {
        self.saw_section = true;
        self.name_col = None;
        self.flush_position();
        if is_sub {
            self.flush_sub();
            self.ensure_section();
            self.cur_sub = Some(Section {
                number,
                title,
                subsections: Vec::new(),
                positions: Vec::new(),
                resources: Vec::new(),
                total: None,
                pages: Vec::new(),
            });
        } else {
            self.flush_sub();
            self.flush_section();
            self.cur_section = Some(Section {
                number,
                title,
                subsections: Vec::new(),
                positions: Vec::new(),
                resources: Vec::new(),
                total: None,
                pages: Vec::new(),
            });
        }
    }

    fn flush_position(&mut self) {
        if let Some(pos) = self.cur_position.take() {
            self.stats.positions += 1;
            if self.cur_sub.is_some() {
                self.cur_sub.as_mut().unwrap().positions.push(pos);
            } else {
                self.ensure_section();
                self.cur_section
                    .get_or_insert_with(Self::blank_section)
                    .positions
                    .push(pos);
            }
        }
    }

    fn flush_sub(&mut self) {
        if let Some(sub) = self.cur_sub.take() {
            self.stats.sections += 1;
            self.ensure_section();
            self.cur_section
                .get_or_insert_with(Self::blank_section)
                .subsections
                .push(sub);
        }
    }

    fn flush_section(&mut self) {
        self.flush_position();
        self.flush_sub();
        if let Some(sec) = self.cur_section.take() {
            self.stats.sections += 1;
            self.sections.push(sec);
        }
    }

    fn flush_all_open(&mut self) {
        self.flush_position();
        self.flush_sub();
        self.flush_section();
    }

    // --- итоги ---

    fn apply_total(&mut self, kind: TotalKind, rate: Option<f64>, num: Option<f64>, joined: &str) {
        let mk = |v: f64| Num::new(v, extract_number_raw(joined).unwrap_or_default());
        match kind {
            TotalKind::Vat => {
                self.totals.vat_rate = rate.or(self.totals.vat_rate);
                if let Some(v) = num {
                    self.totals.vat = Some(mk(v));
                }
            }
            TotalKind::DocWithVat => {
                if rate.is_some() {
                    self.totals.vat_rate = rate;
                }
                if let Some(v) = num {
                    self.totals.total_with_vat = Some(mk(v));
                }
            }
            TotalKind::Doc => {
                if rate.is_some() {
                    self.totals.vat_rate = rate;
                }
                if let Some(v) = num {
                    self.totals.total = Some(mk(v));
                }
            }
            TotalKind::Sub => {
                if let Some(v) = num {
                    let target = self.cur_sub.as_mut().or(self.cur_section.as_mut());
                    match target {
                        Some(t) => t.total = Some(mk(v)),
                        None => self.totals.total = Some(mk(v)),
                    }
                }
            }
            TotalKind::Section => {
                if let Some(v) = num {
                    if let Some(sec) = self.cur_section.as_mut() {
                        sec.total = Some(mk(v));
                    } else if let Some(last) = self.sections.last_mut() {
                        last.total = Some(mk(v));
                    } else if self.cur_sub.is_some() {
                        self.cur_sub.as_mut().unwrap().total = Some(mk(v));
                    } else {
                        self.totals.total = Some(mk(v));
                    }
                }
            }
        }
    }

    fn push_unparsed(&mut self, joined: &str) {
        self.stats.unclassified_rows += 1;
        if self.unparsed.len() < 200 {
            self.unparsed.push(joined.trim().to_string());
        }
    }

    fn finish(mut self) -> EstimateDocument {
        self.flush_all_open();
        // единственный безымянный раздел — вся смета: его итог становится
        // итогом документа (коммерческие сметы без «Разделов»)
        if self.sections.len() == 1 {
            let sec = &mut self.sections[0];
            if sec.number.is_none() && sec.title == "(без разделов)" {
                if let Some(t) = &sec.total {
                    let v = t.clone();
                    self.totals.total = Some(v);
                }
            }
        }
        // итог документа по сумме разделов, если не распознан явно
        if self.totals.total.is_none() {
            let named: Vec<&Section> = self.sections.iter().filter(|s| s.total.is_some()).collect();
            if named.len() == 1 {
                if let Some(t) = &named[0].total {
                    let v = t.clone();
                    self.totals.total = Some(v);
                }
            }
        }
        EstimateDocument {
            meta: self.meta,
            sections: self.sections,
            totals: self.totals,
            validation: ValidationReport {
                errors: Vec::new(),
                warnings: Vec::new(),
                stats: self.stats,
            },
            unparsed_lines: self.unparsed,
            auto_repairs: 0,
            pages: self.pages,
        }
    }
}

fn non_empty(value: String, full: &str) -> String {
    if value.is_empty() {
        full.trim().to_string()
    } else {
        value
    }
}

/// Исходный текст числа из строки (для Num.raw).
fn extract_number_raw(joined: &str) -> Option<String> {
    let mut best: Option<String> = None;
    for tok in joined.split(|c: char| c == ' ' || c == '\u{a0}') {
        if parse_number(tok).is_some() {
            best = Some(tok.to_string());
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(cells: &[(&str, usize)], columnar: bool) -> DocRow {
        DocRow {
            page: 1,
            columnar,
            cells: cells.iter().map(|(t, c)| (*c, 1, t.to_string())).collect(),
        }
    }

    #[test]
    fn numbers_ru() {
        assert_eq!(parse_number("1 234,56"), Some(1234.56));
        assert_eq!(parse_number("123456"), Some(123456.0));
        assert_eq!(parse_number("0,5"), Some(0.5));
        assert_eq!(parse_number("12.5"), Some(12.5));
        assert_eq!(parse_number("1.234,56"), Some(1234.56));
        assert_eq!(parse_number("1.234.567"), Some(1234567.0));
        assert_eq!(parse_number("1 234 567"), Some(1234567.0));
        assert_eq!(parse_number("2 103 955,86"), Some(2103955.86));
        assert_eq!(parse_number("м³"), None);
        assert_eq!(parse_number("12-3"), None);
        assert_eq!(parse_number(""), None);
        assert_eq!(parse_number("20%"), None);
        assert_eq!(parse_number("В22,5"), None);
    }

    #[test]
    fn number_tail() {
        assert_eq!(parse_number_tail("Итого по смете 1 234 567,89 руб."), Some(1234567.89));
        assert_eq!(parse_number_tail("НДС (20%) 1 234,50"), Some(1234.5));
        assert_eq!(parse_number_tail("Всего с НДС 7 407,00"), Some(7407.0));
        assert_eq!(parse_number_tail("без чисел"), None);
    }

    #[test]
    fn total_number_groups() {
        // разрезанное на ячейки число с «рублём»-хвостом
        assert_eq!(parse_total_number("ИТОГО 1 341 653P"), Some(1341653.0));
        // номер раздела не склеивается с суммой (без дроби и средней группы)
        assert_eq!(parse_total_number("Итого по разделу 1 500"), Some(500.0));
        assert_eq!(parse_total_number("Итого по разделу 1 6 172,50"), Some(6172.5));
        assert_eq!(parse_total_number("Итого по смете 1 234 567,89"), Some(1234567.89));
        assert_eq!(parse_total_number("Всего 500 000"), Some(500000.0));
        assert_eq!(parse_total_number("НДС (20%)"), None);
        assert_eq!(parse_total_number("НДС 1234,50 руб."), Some(1234.5));
        assert_eq!(parse_total_number("Итого 1234567,89"), Some(1234567.89));
    }

    #[test]
    fn total_from_cells() {
        // подпись и число в разных ячейках — граница ячейки решает
        let r = row(&[("Итого по разделу 1", 0), ("500", 4)], true);
        assert_eq!(total_from_row(&r), Some(500.0));
        // число, разрезанное границей колонок
        let r = row(&[("ИТОГО", 0), ("1", 2), ("341 653P", 3)], true);
        assert_eq!(total_from_row(&r), Some(1341653.0));
        // дробь в одной ячейке
        let r = row(&[("Итого по смете", 0), ("1 234 567,89", 4)], true);
        assert_eq!(total_from_row(&r), Some(1234567.89));
        // ставка НДС — не сумма
        let r = row(&[("НДС (20%)", 0), ("1 234,50", 3)], true);
        assert_eq!(total_from_row(&r), Some(1234.5));
    }

    #[test]
    fn units() {
        assert!(is_unit("м³"));
        assert!(is_unit("м3"));
        assert!(is_unit("м2"));
        assert!(is_unit("шт."));
        assert!(is_unit("маш.-ч"));
        assert!(is_unit("чел.-ч"));
        assert!(is_unit("100 м²"));
        assert!(is_unit("т"));
        assert!(is_unit("компл"));
        assert!(is_unit("м³/т"));
        assert!(!is_unit("бетон"));
        assert!(!is_unit("1 234,56"));
        assert_eq!(normalize_unit("М2"), "м²");
        assert_eq!(normalize_unit("Маш.-ч"), "маш-ч");
    }

    #[test]
    fn ciphers() {
        assert!(position_cipher("ГЭСН 01-01-013-01").is_some());
        assert!(position_cipher("01-01-013-01").is_some());
        assert!(position_cipher("ФЕР06-01-001-01").is_some());
        assert!(position_cipher("ТЕР 12-01-001-1").is_some());
        assert!(position_cipher("ПР-123").is_some());
        assert!(position_cipher("12.01.001.01").is_some());
        assert!(position_cipher("Разработка грунта").is_none());
        assert!(position_cipher("1.1-1-2345").is_none(), "код ресурса — не расценка");
        assert!(resource_code("1.1-1-2345").is_some());
        assert!(resource_code("101-0797").is_some());
        assert!(resource_code("060148").is_some());
        assert!(resource_code("ГЭСН 01-01-013-01").is_none());
    }

    #[test]
    fn sections_and_positions() {
        let rows = vec![
            row(&[("ЛОКАЛЬНАЯ СМЕТА № 02-01-01 на фундаменты", 0)], false),
            row(&[("Объект: Жилой дом", 0)], false),
            row(&[("Заказчик: ООО Ромашка", 0)], false),
            row(&[("РАЗДЕЛ 1. Земляные работы", 0)], false),
            row(
                &[
                    ("№", 0),
                    ("Обоснование", 1),
                    ("Наименование", 2),
                    ("Ед. изм.", 3),
                    ("Кол-во", 4),
                    ("Цена, руб", 5),
                    ("Всего, руб", 6),
                ],
                true,
            ),
            row(
                &[
                    ("1", 0),
                    ("ГЭСН 01-01-013-01", 1),
                    ("Разработка грунта экскаватором", 2),
                    ("м3", 3),
                    ("123,45", 4),
                    ("50,00", 5),
                    ("6 172,50", 6),
                ],
                true,
            ),
            row(
                &[("", 0), ("", 1), ("Затраты труда рабочих", 2), ("чел.-ч", 3), ("12,5", 4)],
                true,
            ),
            row(&[("", 0), ("", 1), ("Итого по разделу 1", 2), ("6 172,50", 6)], true),
            row(&[("Итого по смете", 0), ("6 172,50", 3)], true),
            row(&[("НДС (20%)", 0), ("1 234,50", 3)], true),
            row(&[("Всего с НДС", 0), ("7 407,00", 3)], true),
        ];
        let doc = parse_document(rows);
        assert_eq!(doc.meta.number.as_deref(), Some("02-01-01"));
        assert_eq!(doc.meta.object.as_deref(), Some("Жилой дом"));
        assert_eq!(doc.meta.customer.as_deref(), Some("ООО Ромашка"));
        assert_eq!(doc.sections.len(), 1);
        let sec = &doc.sections[0];
        assert_eq!(sec.number.as_deref(), Some("1"));
        assert_eq!(sec.title, "Земляные работы");
        assert_eq!(sec.positions.len(), 1);
        let pos = &sec.positions[0];
        assert_eq!(pos.line, Some(1));
        assert_eq!(pos.cipher.as_deref(), Some("ГЭСН 01-01-013-01"));
        assert_eq!(pos.name, "Разработка грунта экскаватором");
        assert_eq!(pos.unit.as_deref(), Some("м³"));
        assert_eq!(pos.quantity.as_ref().map(|n| n.value), Some(123.45));
        assert_eq!(pos.price.as_ref().map(|n| n.value), Some(50.0));
        assert_eq!(pos.total.as_ref().map(|n| n.value), Some(6172.5));
        assert_eq!(pos.resources.len(), 1);
        assert_eq!(pos.resources[0].kind, ResourceKind::Labor);
        assert_eq!(pos.resources[0].quantity.as_ref().map(|n| n.value), Some(12.5));
        assert_eq!(sec.total.as_ref().map(|n| n.value), Some(6172.5));
        assert_eq!(doc.totals.total.as_ref().map(|n| n.value), Some(6172.5));
        assert_eq!(doc.totals.vat_rate, Some(0.2));
        assert_eq!(doc.totals.vat.as_ref().map(|n| n.value), Some(1234.5));
        assert_eq!(doc.totals.total_with_vat.as_ref().map(|n| n.value), Some(7407.0));
    }

    #[test]
    fn commercial_estimate_without_sections() {
        let rows = vec![
            row(&[("СМЕТА на ремонт помещения", 0)], false),
            row(
                &[
                    ("№", 0),
                    ("Наименование работ", 1),
                    ("Ед. изм.", 2),
                    ("Кол-во", 3),
                    ("Цена", 4),
                    ("Сумма", 5),
                ],
                true,
            ),
            row(
                &[("1", 0), ("Демонтаж пола", 1), ("м2", 2), ("50", 3), ("100", 4), ("5 000", 5)],
                true,
            ),
            row(
                &[("2", 0), ("Укладка ламината", 1), ("м2", 2), ("50", 3), ("800", 4), ("40 000", 5)],
                true,
            ),
            row(&[("Итого:", 0), ("45 000", 5)], true),
        ];
        let doc = parse_document(rows);
        assert_eq!(doc.sections.len(), 1);
        assert_eq!(doc.sections[0].title, "(без разделов)");
        assert_eq!(doc.sections[0].positions.len(), 2);
        assert_eq!(doc.totals.total.as_ref().map(|n| n.value), Some(45000.0));
        assert_eq!(doc.sections[0].total.as_ref().map(|n| n.value), Some(45000.0));
    }

    #[test]
    fn continuation_of_long_name() {
        let rows = vec![
            row(
                &[("1", 0), ("Устройство монолитного", 1), ("м3", 2), ("10", 3), ("5 000", 4)],
                true,
            ),
            row(&[("", 0), ("фундамента из бетона В22,5", 1)], true),
        ];
        let doc = parse_document(rows);
        assert_eq!(
            doc.sections[0].positions[0].name,
            "Устройство монолитного фундамента из бетона В22,5"
        );
    }

    #[test]
    fn material_resource_row() {
        let rows = vec![
            row(
                &[("1", 0), ("Кладка стен", 1), ("м3", 2), ("100", 3), ("2 000", 4), ("200 000", 5)],
                true,
            ),
            row(
                &[
                    ("", 0),
                    ("КИРПИЧ КЕРАМИЧЕСКИЙ", 1),
                    ("1.1-1-1234", 2),
                    ("шт", 3),
                    ("40 000", 4),
                    ("15", 5),
                    ("600 000", 6),
                ],
                true,
            ),
        ];
        let doc = parse_document(rows);
        let pos = &doc.sections[0].positions[0];
        assert_eq!(pos.resources.len(), 1);
        let r = &pos.resources[0];
        assert_eq!(r.kind, ResourceKind::Material);
        assert_eq!(r.code.as_deref(), Some("1.1-1-1234"));
        assert_eq!(r.name, "КИРПИЧ КЕРАМИЧЕСКИЙ");
        assert_eq!(r.unit.as_deref(), Some("шт"));
        assert_eq!(r.quantity.as_ref().map(|n| n.value), Some(40000.0));
        assert_eq!(r.price.as_ref().map(|n| n.value), Some(15.0));
        assert_eq!(r.total.as_ref().map(|n| n.value), Some(600000.0));
    }

    #[test]
    fn overhead_profit_rows() {
        let rows = vec![
            row(
                &[("1", 0), ("Кладка стен", 1), ("м3", 2), ("100", 3), ("2 000", 4), ("200 000", 5)],
                true,
            ),
            row(&[("Накладные расходы", 0), ("30 000", 4)], true),
            row(&[("Сметная прибыль", 0), ("20 000", 4)], true),
            row(&[("Итого по смете", 0), ("250 000", 4)], true),
        ];
        let doc = parse_document(rows);
        let pos = &doc.sections[0].positions[0];
        assert_eq!(pos.resources.len(), 2);
        assert_eq!(pos.resources[0].kind, ResourceKind::Overhead);
        assert_eq!(pos.resources[0].total.as_ref().map(|n| n.value), Some(30000.0));
        assert_eq!(pos.resources[1].kind, ResourceKind::Profit);
        assert_eq!(doc.totals.total.as_ref().map(|n| n.value), Some(250000.0));
    }

    #[test]
    fn subsections() {
        let rows = vec![
            row(&[("Раздел 1. Монтаж", 0)], false),
            row(&[("1", 0), ("Работа А", 1), ("шт", 2), ("1", 3), ("100", 4)], true),
            row(&[("Подраздел 1.1. Электрика", 0)], false),
            row(&[("2", 0), ("Работа Б", 1), ("шт", 2), ("2", 3), ("200", 4)], true),
            row(&[("Итого по разделу 1", 0), ("500", 4)], true),
        ];
        let doc = parse_document(rows);
        assert_eq!(doc.sections.len(), 1);
        let sec = &doc.sections[0];
        assert_eq!(sec.positions.len(), 1);
        assert_eq!(sec.subsections.len(), 1);
        assert_eq!(sec.subsections[0].number.as_deref(), Some("1.1"));
        assert_eq!(sec.subsections[0].title, "Электрика");
        assert_eq!(sec.subsections[0].positions.len(), 1);
        assert_eq!(sec.total.as_ref().map(|n| n.value), Some(500.0));
    }
}
