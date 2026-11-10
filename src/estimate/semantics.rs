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
/// Несколько слипшихся чисел ячейки («179000 4,2 75 180» после слияния
/// колонок) — НЕ число: дробный разделитель допустим только в последнем
/// пробельном токене, иначе это разные значения, склеенные OCR.
pub fn parse_number(s: &str) -> Option<f64> {
    let raw = s.trim();
    if raw.is_empty() {
        return None;
    }
    {
        let toks: Vec<&str> = raw.split([' ', '\u{a0}']).filter(|t| !t.is_empty()).collect();
        for (i, tok) in toks.iter().enumerate() {
            if (tok.contains(',') || tok.contains('.')) && i + 1 != toks.len() {
                return None;
            }
        }
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

/// Код ресурса (не расценка): «1.1-1-2345», «1.102-0008», «101-0797»,
/// «060148». Код может нести OCR-мусор в той же ячейке («Уд 1.102-0008»)
/// или разорванную точкой пару токенов («3 102-0061»).
pub fn resource_code(s: &str) -> Option<String> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    if resource_code_exact(t) {
        return Some(t.to_string());
    }
    // «3 102-0061» — точка прочиталась пробелом (пробуем ДО токен-скана:
    // иначе голый «102-0061» перехватит проверку раньше составного кода)
    let toks: Vec<&str> = t.split_whitespace().collect();
    for w in toks.windows(2) {
        let joined = format!("{}.{}", w[0], w[1]);
        if resource_code_exact(&joined) {
            return Some(joined);
        }
    }
    // «Уд 1.102-0008», «ya 2.102-0049» — код внутри ячейки с мусором.
    // Шестизначные числа внутри текста НЕ считаем кодами: это цены/суммы
    // («Работа автокрана смена 320000 1,5 48 000») — код машины занимает
    // ячейку целиком и стоит в начале строки
    for tok in t.split_whitespace() {
        if is_six_digit(tok) {
            continue;
        }
        if resource_code_exact(tok) {
            return Some(tok.to_string());
        }
    }
    None
}

fn resource_code_exact(t: &str) -> bool {
    // 1.1-1-2345 и 1.102-0008
    if let Some((head, rest)) = t.split_once('.') {
        if !head.is_empty() && head.len() <= 2 && head.chars().all(|c| c.is_ascii_digit()) {
            let rparts: Vec<&str> = rest.split('-').collect();
            if (rparts.len() >= 3
                && rparts[..2]
                    .iter()
                    .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
                && rparts[2].len() >= 3
                && rparts[2].chars().all(|c| c.is_ascii_digit()))
                || (rparts.len() == 2
                    && rparts
                        .iter()
                        .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
                    && rparts[0].len() == 3
                    && rparts[1].len() == 4)
            {
                return true;
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
        return true;
    }
    // 060148 — код машины (6 цифр)
    is_six_digit(t)
}

/// Ячейка из ровно шести цифр («060148», «320000»): может быть кодом
/// машины — а может быть ценой/суммой. Различает положение в строке
/// (см. `is_first_nonempty`).
fn is_six_digit(t: &str) -> bool {
    let t = t.trim();
    t.len() == 6 && t.chars().all(|c| c.is_ascii_digit())
}

/// Ячейка с индексом i — первая непустая в строке. Код машины всегда
/// стоит первой непустой ячейкой («060148 Кран…»); шестизначное число
/// правее наименования («Работа автокрана смена 320000 1,5 48 000») —
/// это цена, а не код.
fn is_first_nonempty(row: &DocRow, i: usize) -> bool {
    row.cells[..i]
        .iter()
        .all(|(_, _, t)| t.trim().is_empty())
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

/// Свернуть латинские буквы-двойники в кириллицу — ТОЛЬКО для матчинга
/// ключевых слов (шапки/итоги): OCR регулярно печатает «Bcero» вместо
/// «Всего», «cметe» вместо «смете», и роли колонок/итоги теряются.
/// Выходной текст документов не меняется — только распознавание маркеров.
fn fold_latin_lookalikes(lower: &str) -> String {
    lower
        .chars()
        .map(|c| match c {
            'a' => 'а',
            'b' => 'в',
            'c' => 'с',
            'e' => 'е',
            'h' => 'н',
            'k' => 'к',
            'm' => 'м',
            'o' => 'о',
            'p' => 'р',
            'r' => 'г', // «Bcero» → «Всего»: Г читается латинской r
            't' => 'т',
            'x' => 'х',
            'y' => 'у',
            'n' => 'и', // «no cмeтe» → «по смете»
            'і' => 'и', // украинская І — частая подмена кириллической И
            _ => c,
        })
        .collect()
}

/// Строка без пробелов с согнутыми латинскими двойниками: каноническая
/// форма для матчинга маркеров, разорванных OCR-пробелами («htt p://»,
/// «по с смете», «Ра аздел»). Пробелы OCR вставляет ВНУТРИ слов.
fn despace_fold(lower: &str) -> String {
    fold_latin_lookalikes(&lower.chars().filter(|c| !c.is_whitespace()).collect::<String>())
}

/// Только буквы, слитно, в нижнем регистре с двойниками: форма для
/// нечёткого сравнения ключевых слов («затра ты» → «затраты»,
/// «прибыл 1ь» → «прибыль»).
fn letters_squeezed(lower: &str) -> String {
    fold_latin_lookalikes(&lower.chars().filter(|c| c.is_alphabetic()).collect::<String>())
}

/// Расстояние Левенштейна по символам (строки-ключевые слова короткие).
fn edit_distance_chars(a: &[char], b: &[char]) -> usize {
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
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

fn edit_distance(a: &str, b: &str) -> usize {
    edit_distance_chars(&a.chars().collect::<Vec<_>>(), &b.chars().collect::<Vec<_>>())
}

/// Нечёткое вхождение ключевого слова (в форме `letters_squeezed`) в
/// строку: OCR рвёт и искажает ключевые слова («Сметна ая прибыл 1ь» =
/// «Сметная прибыль»), а терять маркеры ресурсов/итогов нельзя. Сначала
/// дешёвый предфильтр по общим символам, затем скользящие окна с
/// расстоянием ≤ 1–2 (длинное слово — больше допуск).
fn fuzzy_contains(hay: &str, needle: &str) -> bool {
    let hay: Vec<char> = hay.chars().collect();
    let needle: Vec<char> = needle.chars().collect();
    let (n, m) = (hay.len(), needle.len());
    if m == 0 || n + 2 < m {
        return false;
    }
    let max_d = if m >= 10 { 2 } else { 1 };
    // предфильтр: общих символов (multiset) не меньше m - max_d
    let mut need: [usize; 64] = [0; 64];
    let idx = |c: char| -> Option<usize> {
        // только кириллица и латиница нижнего регистра — всё, что даёт fold
        let u = c as u32;
        if ('а' as u32..='я' as u32).contains(&u) {
            Some((u - 'а' as u32) as usize)
        } else if c.is_ascii_lowercase() {
            Some(32 + (u - 'a' as u32) as usize)
        } else {
            None
        }
    };
    for &c in &needle {
        if let Some(i) = idx(c) {
            need[i] += 1;
        }
    }
    let required = m.saturating_sub(max_d);
    let mut common = 0usize;
    'outer: for &c in &hay {
        if let Some(i) = idx(c) {
            if need[i] > 0 {
                need[i] -= 1;
                common += 1;
                if common >= required {
                    break 'outer;
                }
            }
        }
    }
    if common < required {
        return false;
    }
    for start in 0..n {
        for len in m..=(m + max_d).min(n - start) {
            if edit_distance_chars(&hay[start..start + len], &needle) <= max_d {
                return true;
            }
        }
    }
    false
}

/// Колонтитулы, URL, контакты, логотипы — «мебель» страницы, не
/// содержимое сметы: не позиции, не итоги и не «неразобранные» строки.
/// Вход — строка в нижнем регистре (свернём на всякий случай).
fn is_junk_line(lower: &str) -> bool {
    let owned;
    let lower = if lower.chars().any(|c| c.is_uppercase()) {
        owned = lower.to_lowercase();
        owned.as_str()
    } else {
        lower
    };
    let sq = despace_fold(lower);
    if sq.contains("http")
        || sq.contains("www.")
        || sq.contains(".ru")
        || sq.contains(".com")
        || sq.contains("@")
        || sq.contains("e-mail")
        || sq.contains("email")
        || sq.contains("mail.ru")
        || sq.contains("тел.")
        || sq.contains("+7(")
        || sq.contains("факс")
        || sq.contains("страниц")
        || sq.contains("верси")
        || sq.contains("гранд-смет")
        || sq.contains("грандсмет")
        || sq.contains("ул.")
    {
        return true;
    }
    // «Сметный Калькулятор» — колонтитул Прайм-калькуляторов (шапка каждой
    // страницы), не содержимое
    if sq.contains("сметныйкалькулятор") {
        return true;
    }
    let lz = letters_squeezed(lower);
    if lz.contains("грандсмет") {
        return true;
    }
    // «Ver 6.2.7» / «версия 6.2.7 20221031» — латинский след программы
    lz.starts_with("ver") && lower.chars().any(|c| c.is_ascii_digit())
}

fn role_of_header(text: &str) -> Option<Role> {
    let t = fold_latin_lookalikes(&text.trim().to_lowercase());
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
    if t == "итого" || t == "итого," || t.starts_with("итогов") {
        // голая колонка «Итого»/«Итоговая стоимость» (коммерческие сметы
        // с двухуровневой шапкой)
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
    let lower = fold_latin_lookalikes(&row.lower());
    if lower.contains("наименован") {
        return true;
    }
    let marks = [
        "№", "ед", "кол", "цена", "сумма", "всего", "обоснов", "стоимост", "затрат", "труд",
        "машин", "материал",
        // трёхуровневые шапки Гранд-Сметы: строка листьев «Всего | Осн.З/п |
        // Эк.Маш | 3/пМех» — крупных марок всего две, узнаём по мелким
        "маш", "мех", "осн", "з/п", "3/п",
    ];
    let hits = marks.iter().filter(|m| lower.contains(*m)).count();
    hits >= 3
}

/// Промежуточные сводки Гранд-Сметы в конце документа: «Итого прямые
/// затраты…», «Итого строительные/монтажные работы», «Непредвиденные
/// затраты 2%». Это НЕ итоги (не раздел/документ) и НЕ позиции — но без
/// фильтра становятся «позициями» и ломают Σ разделов. Нечёткое
/// сравнение — OCR рвёт слова пробелами и глотает буквы.
fn is_intermediate_summary(lower: &str) -> bool {
    let lower = &fold_latin_lookalikes(lower);
    let lz = letters_squeezed(lower);
    is_intermediate_summary_text(lower, &lz)
}

fn is_intermediate_summary_text(lower: &str, lz: &str) -> bool {
    lower.contains("прямые затраты")
        || lower.contains("непредвиден")
        || lower.starts_with("индекс")
        || lower.contains("в том числе зарплата")
        || lower.starts_with("материалы в текущем")
        || (lz.starts_with("итого")
            && (lower.contains("с накладными")
                || lower.contains("с индексацией")
                || lower.contains("строительные работы")
                || lower.contains("монтажные работы")
                || lower.contains("прочие работы")
                || lower.contains("прочие затраты")
                || fuzzy_contains(lz, "прямыезатраты")
                || fuzzy_contains(lz, "непредвиденныезатраты")
                || fuzzy_contains(lz, "строительныеработы")
                || fuzzy_contains(lz, "монтажныеработы")
                || fuzzy_contains(lz, "прочиеработы")
                || fuzzy_contains(lz, "прочиезатраты")))
}

/// Строка шапки ДОКУМЕНТА до табличного заголовка: «СОГЛАСОВАНО…»,
/// «Сметная стоимость 2010,425 тыс. руб.», «Составлен(а)…». Такие
/// строки в Гранд-Сметах лежат в сетке (метка | значение) и без
/// фильтра становятся «позициями» сметы.
fn meta_keyword_row(lower: &str) -> bool {
    [
        "согласован",
        "утвержда",
        "сметная стоимость",
        "сметная трудоемк",
        "средства на оплату",
        "составлен",
        "основание:",
        "наименование стройки",
        "строительных работ",
        "монтажных работ",
        "тыс. руб",
        "тыс.руб",
    ]
    .iter()
    .any(|k| lower.contains(k))
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
    // «НДС» целиком и осколки OCR: «H 18%», «нд C 18%», «ндс 18%» —
    // буквы до цифр образуют обрывок «ндс» (латинские двойники c/h),
    // дальше обязательна ставка с процентом
    let lower = &fold_latin_lookalikes(lower);
    let squeezed: String = lower.chars().filter(|c| !c.is_whitespace()).collect();
    // каноническая форма для нечётких маркеров: «Итог о прям мые затра ты»
    // → «итогопряммыезатраты» (пробелы внутри слов схлопываются)
    let lz = letters_squeezed(lower);
    // голова строки — итоговое слово, даже разорванное OCR и с мусорной
    // буквой впереди («и Ітог и по смете:», «BC ЕГО п о смете»): копим
    // буквы первых токенов, пока не опознаем слово (расстояние ≤ 1)
    let head = {
        let mut acc = String::new();
        let mut hit = false;
        for t in lower.split_whitespace().take(4) {
            let w: String = t.chars().filter(|c| c.is_alphabetic()).collect();
            if w.is_empty() {
                continue;
            }
            acc.push_str(&w);
            let a = letters_squeezed(&acc);
            if a.chars().count() >= 5
                && ["итого", "итоги", "всего", "подытог"]
                    .iter()
                    .any(|m| a == *m || edit_distance(&a, m) <= 1)
            {
                hit = true;
                break;
            }
            if a.chars().count() >= 8 {
                break; // слово длиннее итогового — не оно
            }
        }
        hit
    };
    if squeezed.starts_with("ндс")
        || squeezed.starts_with("суммандс")
        || vat_label_mangle(&squeezed)
    {
        return Some((TotalKind::Vat, parse_vat_rate(lower)));
    }
    if lower.starts_with("всего с ндс") || lower.starts_with("итого с ндс") {
        return Some((TotalKind::DocWithVat, None));
    }
    // «Итого с непредвиденными» — база документа (перед НДС), а НЕ итог
    // текущего раздела: иначе затирает итог последнего раздела (нечётко:
    // «Итог о с непредвиден…» — OCR рвёт слова)
    if head
        && (lower.contains("непредвиден") || fuzzy_contains(&lz, "непредвиденными"))
    {
        return Some((TotalKind::Doc, None));
    }
    // разбивка по видам работ/затрат в конце Гранд-Сметы и «прямые затраты» —
    // промежуточные суммы, не итог раздела и не итог сметы (нечёткое
    // сравнение: OCR рвёт слова пробелами — «прям мые затра ты»)
    if is_intermediate_summary_text(lower, &lz) {
        return None;
    }
    if lower.contains("по разделу") || squeezed.contains("поразделу") {
        return Some((TotalKind::Section, None));
    }
    if lower.contains("по подразделу") || squeezed.contains("поподразделу") {
        return Some((TotalKind::Sub, None));
    }
    // «Итого по всем разделам» — итог ДОКУМЕНТА (хвост Прайм-калькуляторов:
    // …Всего по разделу: 564863,12 / Итого по всем разделам 8 109 921,92 /
    // НДС / Всего по смете). Не узнанный — затирает итог последнего раздела
    if lower.contains("по всем разделам")
        || lower.contains("по всем подразделам")
        || squeezed.contains("повсемразделам")
    {
        return Some((TotalKind::Doc, None));
    }
    if lower.contains("по главе") {
        return Some((TotalKind::Section, None));
    }
    if lower.contains("по главам") {
        return Some((TotalKind::Doc, None)); // итог сводного сметного расчёта
    }
    // «по смете»/«по объекту»/«по расч» — итог ДОКУМЕНТА, но только с
    // итоговой головой строки: «Итого прямые затраты ПО СМЕТЕ в ценах
    // 2001г.» — промежуточная сводка (мусорный итог 3,3 млрд в корпусе
    // был именно отсюда). Нечётко: OCR вставляет лишние буквы («по с смете»)
    // и рвёт слова пробелами
    if head
        && (lower.contains("по смете")
            || squeezed.contains("посмете")
            || fuzzy_contains(&lz, "посмете")
            || lower.contains("по объекту")
            || squeezed.contains("пообъекту")
            || lower.contains("по расч")
            || squeezed.contains("порасч"))
    {
        return Some((TotalKind::Doc, None));
    }
    if lower.contains("по позиц") || lower.starts_with("справочно") {
        return None;
    }
    if head {
        // итог дома/комплектации — итог документа, а не раздела:
        // «Итого дом в комплектации "КОРОБКА"…» (опциональные разделы
        // идут после него и в него не входят)
        if lz.starts_with("итогодом")
            || lz.starts_with("всегодом")
            || lz.contains("домвкомплектации")
            || lz.starts_with("итоговсего")
            || lz.contains("итогоподому")
            || lz.contains("всегоподому")
        {
            return Some((TotalKind::Doc, None));
        }
        return Some((TotalKind::Section, None)); // раздел или документ — решает контекст
    }
    None
}

/// Ставка НДС из «(20%)» / «(22 %)» либо «НДС 18%» без скобок.
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
    // без скобок: токен вида «18%»/«18 %»
    let mut rest = lower;
    while let Some(i) = rest.find('%') {
        let head = rest[..i].trim_end();
        let digits: String = head
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit() || *c == ',' || *c == '.')
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if let Some(v) = parse_number(&digits) {
            return Some(v / 100.0);
        }
        rest = &rest[i + 1..];
    }
    None
}

/// Осколок «НДС»: буквенный префикс — обрывок «ндс» из визуальных
/// двойников (h→н, латинская c→с), а остаток обязан начинаться с цифры
/// и содержать процент: «h18%», «ндc18%», «н18%» — это «НДС 18%».
fn vat_label_mangle(squeezed: &str) -> bool {
    let letters: String = squeezed.chars().take_while(|c| c.is_alphabetic()).collect();
    if letters.is_empty() {
        return false;
    }
    let rest = &squeezed[letters.len()..];
    if !rest.starts_with(|c: char| c.is_ascii_digit()) || !rest.contains('%') {
        return false;
    }
    let mapped: Option<String> = letters
        .chars()
        // «о» — просечка скана внутри обрывка («ндо C 18%» = «НДС 18%»)
        .filter(|c| *c != 'о' && *c != 'o')
        .map(|c| match c {
            'h' | 'н' => Some('н'),
            'д' => Some('д'),
            'с' => Some('с'),
            'c' => Some('с'), // латинская C — двойник С
            _ => None,
        })
        .collect();
    matches!(mapped.as_deref(), Some("н") | Some("нд") | Some("ндс") | Some("дс") | Some("нс"))
}

/// Голая итоговая строка: итоговое слово (даже разорванное OCR) + только
/// числа/короткие обрывки. В хвосте «Итоги по смете:» такие строки —
/// бегущие итоги документа («Итого 200149,96», «Итого 208644,33»),
/// а НЕ итоги текущего раздела.
fn bare_total_line(lower: &str) -> bool {
    let toks: Vec<&str> = lower.split_whitespace().collect();
    if toks.is_empty() {
        return false;
    }
    // голова: кусочки итогового слова, склеенные подряд; запоминаем,
    // сколько токенов съедено — их в проверке хвоста не учитываем
    let mut acc = String::new();
    let mut consumed = 0usize;
    let mut head_ok = false;
    for (i, t) in toks.iter().enumerate().take(3) {
        let w: String = t.chars().filter(|c| c.is_alphabetic()).collect();
        if w.is_empty() {
            continue;
        }
        acc.push_str(&w.to_lowercase());
        let lz = letters_squeezed(&acc);
        if ["итого", "итоги", "всего", "подытог", "подытога"]
            .iter()
            .any(|m| lz == *m || edit_distance(&lz, m) <= 1)
        {
            head_ok = true;
            consumed = i + 1;
            break;
        }
        if lz.chars().count() > 7 {
            break;
        }
    }
    if !head_ok {
        return false;
    }
    // прочие токены — только числа и обрывки ≤3 букв (куски итогового
    // слова): «ито го 2 00149. 96» — да; «итого по разделу 1 6 172,50» — нет
    toks[consumed..].iter().all(|t| {
        let w: String = t.chars().filter(|c| c.is_alphabetic()).collect();
        w.is_empty() || w.chars().count() <= 3
    })
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
    if let Some(v) = cut_total_value(row) {
        return Some(v);
    }
    // фрагменты разрезанного числа — чистые короткие цифровые ячейки слева
    // (группы разрядов «1» | «341 653P»). Склейку делаем только когда
    // правый осколок БЕЗ десятичного разделителя: «57» | «391.62» — это
    // РАЗНЫЕ метрики строки (трудоёмкость и сумма), а не куски одного числа
    let right_has_sep = row.cells[right_idx]
        .2
        .trim()
        .contains(',')
        || row.cells[right_idx].2.trim().contains('.');
    let mut parts: Vec<String> = vec![row.cells[right_idx].2.trim().to_string()];
    let mut j = right_idx;
    while j > 0 && !right_has_sep {
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

/// Итог, разрезанный границей колонок: десятичный разделитель ушёл на
/// стык соседних ячеек — «1 544414» | «.6», «2 00149.» | «96»,
/// «1» | «544414» | «.6». Разделитель на границе ячеек не бывает у
/// РАЗНЫХ метрик («57» | «391.62» — трудоёмкость и сумма: разделителя
/// на стыке нет), поэтому такая склейка — сильное доказательство и
/// она приоритетнее якорного выбора кандидатов в `resolve_total_num`.
fn cut_total_value(row: &DocRow) -> Option<f64> {
    let lenient = |t: &str| -> Option<f64> {
        let t = t.trim();
        if t.contains('(') || t.contains('%') || t.contains('=') {
            return None;
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
    if right_idx == 0 {
        return None;
    }
    let left = row.cells[right_idx - 1].2.trim().to_string();
    let right = row.cells[right_idx].2.trim().to_string();
    let numish = |t: &str| -> bool {
        !t.is_empty()
            && t.chars().count() <= 10
            && t.chars().all(|c| c.is_ascii_digit() || c == ',' || c == '.' || c == ' ' || c == '\u{a0}')
    };
    let sep_boundary = right.starts_with(',') || right.starts_with('.')
        || left.ends_with(',') || left.ends_with('.');
    if !sep_boundary || !numish(&left) {
        return None;
    }
    let joined = format!("{}{}", left, right);
    let mut v = lenient(&joined)?;
    // ведущие группы разрядов того же числа: «1» | «544414» | «.6»
    // → «1 544414.6»; «2» | «35588.» | «66» → «2 35588.66»
    let mut joined = joined;
    let mut k = right_idx - 1;
    while k > 0 {
        k -= 1;
        let head = row.cells[k].2.trim();
        if !head.is_empty()
            && head.chars().all(|c| c.is_ascii_digit())
            && (1..=3).contains(&head.chars().count())
        {
            let cand = format!("{} {}", head, joined);
            if let Some(v2) = lenient(&cand) {
                joined = cand;
                v = v2;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    Some(v)
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
    // нечёткий слой: OCR рвёт ключевые слова пробелами и глотает буквы
    // («с Сметна ая прибыл 1ь» = «Сметная прибыль» — иначе строка итога
    // становится фантомной позицией). Материалы нечётко НЕ ищем: слово
    // частое, ложные срабатывания дороже находок
    let lz = letters_squeezed(lower);
    if fuzzy_contains(&lz, "накладныерасходы") {
        return Some(ResourceKind::Overhead);
    }
    if fuzzy_contains(&lz, "сметнаяприбыль") {
        return Some(ResourceKind::Profit);
    }
    if fuzzy_contains(&lz, "затратытрударабочих") || fuzzy_contains(&lz, "затратытрудастроителей") {
        return Some(ResourceKind::Labor);
    }
    if fuzzy_contains(&lz, "трудамашинистов") {
        return Some(ResourceKind::MachineLabor);
    }
    if fuzzy_contains(&lz, "эксплуатациямашин") {
        return Some(ResourceKind::Machine);
    }
    if fuzzy_contains(&lz, "транспортныерасходы") {
        return Some(ResourceKind::Transport);
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
    title_like(joined) && !joined.trim().chars().any(|c| c.is_ascii_digit())
}

/// Заголовок НОВОГО документа в составном файле (ЛСР + ОСР + приложение):
/// «Объектный сметный расчёт…», «Локальная смета…» — хвост «Итоги по
/// смете» предыдущего документа на этом кончается.
fn new_doc_header(lower: &str) -> bool {
    let f = despace_fold(lower);
    f.contains("сметныйрасчет")
        || f.contains("сметныйрасчёт")
        || f.contains("локальнаясмета")
        || f.contains("объектныйсметный")
        || f.contains("сводныйсметный")
}

/// Похоже ли на название раздела: без служебных знаков, контактов и
/// чисел; начинается с ЗАГЛАВНОЙ буквы. Фрагменты примечаний, адресов и
/// логотипов («г. Москва», «с ТРОИМ», «www.vfstroy.ru», «FULLHOUSE»)
/// отсекаются — раньше они становились мусорными разделами.
fn title_like(t: &str) -> bool {
    let t = t.trim();
    let n = t.chars().count();
    if !(3..=60).contains(&n) {
        return false;
    }
    if t.contains(':') || t.contains('(') || t.contains('№') || t.contains('%') {
        return false;
    }
    if t.contains('@') || t.contains("www") || t.contains(".ru") || t.contains(".com") {
        return false;
    }
    // тире/дефис в начале — пункт списка или приписка, не название раздела
    if t.starts_with('-') || t.starts_with('–') || t.starts_with('—') {
        return false;
    }
    if t.chars().filter(|c| c.is_alphabetic()).count() < 3 {
        return false;
    }
    if is_unit(t) {
        return false;
    }
    // примечания/адреса начинаются со строчной буквы; логотипы — целиком
    // латинские капсом
    if t.chars().next().map(|c| c.is_lowercase()).unwrap_or(false) {
        return false;
    }
    if t
        .chars()
        .filter(|c| c.is_alphabetic())
        .all(|c| c.is_ascii_alphabetic() && c.is_uppercase())
    {
        return false;
    }
    let lower = t.to_lowercase();
    if totals_kind_text(&lower).is_some() || meta_field(&lower).is_some() {
        return false;
    }
    true
}

/// Название раздела в строке с номером: «Строительство 1го этажа» —
/// заглавная буква, без служебных знаков; цифры внутри слов («1го»)
/// допустимы, отдельные числа — нет.
fn section_title_ok(title: &str) -> bool {
    let t = title.trim();
    if !(4..=80).contains(&t.chars().count()) {
        return false;
    }
    if t.contains(':') || t.contains('@') || t.contains("www") {
        return false;
    }
    // название раздела начинается с БУКВЫ: «3 110*0,94% = 133.28…» —
    // разорванный хвост позиции (НР/СП-арифметика), а не раздел №3
    if !t.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) {
        return false;
    }
    // рекламные блоки коммерческих смет («ПОЧЕМУ«ПРАВИЛО»?»)
    if t.contains('?') || t.contains('!') {
        return false;
    }
    if t.chars().filter(|c| c.is_alphabetic()).count() < 4 {
        return false;
    }
    if is_unit(t) {
        return false;
    }
    if t.chars().next().map(|c| c.is_lowercase()).unwrap_or(false) {
        return false;
    }
    let lower = t.to_lowercase();
    if totals_kind_text(&lower).is_some() {
        return false;
    }
    true
}

/// Заголовок раздела коммерческой сметы БЕЗ слова «Раздел»:
/// ведущее целое 1..=99 и текстовое название — либо в двух ячейках
/// («1» | «Строительство фундамента»), либо в одной («2 Строительство
/// 1го этажа»). Чисел, кроме ведущего номера, в строке нет — у позиций
/// всегда есть кол-во/цена/сумма, и это отличает заголовок от данных.
fn numbered_section_header(row: &DocRow) -> Option<(u32, String)> {
    let non_empty: Vec<&(usize, usize, String)> = row
        .cells
        .iter()
        .filter(|(_, _, t)| !t.trim().is_empty())
        .collect();
    if non_empty.is_empty() {
        return None;
    }
    // форма 1: первая ячейка — целое, остальные — название
    let head = non_empty[0].2.trim().trim_end_matches('.');
    if let Some(n) = head.parse::<u32>().ok().filter(|v| (1..=99).contains(v)) {
        if non_empty.len() >= 2 {
            let rest: Vec<&str> = non_empty[1..].iter().map(|(_, _, t)| t.trim()).collect();
            if rest.iter().any(|t| parse_number(t).is_some()) {
                return None;
            }
            let title = rest.join(" ");
            if section_title_ok(&title) {
                return Some((n, title));
            }
        }
    }
    // форма 2: всё в одной ячейке «2 Строительство 1го этажа»
    if non_empty.len() == 1 {
        let t = non_empty[0].2.trim();
        let mut toks = t.split_whitespace();
        let head = toks.next()?.trim_end_matches('.');
        if let Some(n) = head
            .parse::<u32>()
            .ok()
            .filter(|v| (1..=99).contains(v))
        {
            let title: Vec<&str> = toks.collect();
            if title.is_empty() || title.iter().any(|w| parse_number(w).is_some()) {
                return None;
            }
            let title = title.join(" ");
            if section_title_ok(&title) {
                return Some((n, title));
            }
        }
    }
    None
}

/// Строка — целиком целое 1..=99: номер раздела, вынесенный отдельной
/// строкой («2» над «Строительство 1го этажа»). Трёхзначные — служебные
/// обрезки чисел («652» из итоговой строки sm_id), не номера разделов.
fn lone_int(row: &DocRow) -> Option<u32> {
    let joined = row.joined();
    let t = joined.trim().trim_end_matches('.');
    if !(1..=2).contains(&t.chars().count()) {
        return None;
    }
    if !t.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    t.parse::<u32>().ok().filter(|v| (1..=99).contains(v))
}

/// Маркер блока примечаний: «Примечания:» / «Примечание:».
fn notes_marker(lower: &str) -> bool {
    let t = lower
        .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')' || c == ' ')
        .trim();
    t.starts_with("примечани")
}

/// Raw — только цифры и разрядные пробелы: похоже на потерянную запятую.
fn raw_digits_only(raw: &str) -> bool {
    let t: String = raw.chars().filter(|c| *c != ' ' && *c != '\u{a0}').collect();
    t.len() >= 4 && t.chars().all(|c| c.is_ascii_digit())
}

/// Двухценовая раскладка «материалы + работы»: кол-во встречается в строке
/// дважды, цены p1, p2 и стоимости c1 = q×p1, c2 = q×p2 тоже, а
/// c1 + c2 = итог. Возвращает (кол-во, суммарная цена за единицу).
fn dual_price_pair(numbers: &[(usize, f64, String)], total: f64) -> Option<(Num, Num)> {
    if total <= 0.0 {
        return None;
    }
    let close = |a: f64, b: f64| (a - b).abs() <= b.abs().max(1.0) * 0.005 + 0.01;
    for (i, nq) in numbers.iter().enumerate() {
        let q = nq.1;
        if q <= 0.0 {
            continue;
        }
        // кол-во должно встретиться минимум дважды
        if !numbers
            .iter()
            .enumerate()
            .any(|(j, nj)| j != i && (nj.1 - q).abs() < 1e-9)
        {
            continue;
        }
        for c1n in numbers {
            let c1 = c1n.1;
            if c1 <= 0.0 || c1 >= total {
                continue;
            }
            let p1 = c1 / q;
            let Some(p1n) = numbers.iter().find(|n| close(n.1, p1)) else {
                continue;
            };
            let c2 = total - c1;
            let p2 = c2 / q;
            let Some(p2n) = numbers.iter().find(|n| close(n.1, p2)) else {
                continue;
            };
            // вторая стоимость тоже должна быть в строке
            if !numbers.iter().any(|n| close(n.1, c2)) {
                continue;
            }
            let price = p1n.1 + p2n.1;
            return Some((
                Num::new(q, nq.2.clone()),
                Num::new(price, format!("{} + {}", p1n.2, p2n.2)),
            ));
        }
    }
    None
}

/// Кол-во и цена по арифметике строки при известном итоге: q × c = total
/// (кол-во ищем среди чисел ≤ 1000), либо q × c = total × 10^k при цене,
/// склеенной OCR без запятой (raw из одних цифр) — цена восстанавливается
/// сдвигом запятой на k разрядов. «10 0000» × «3,0» = «30 000» × 10 →
/// кол-во 3, цена 10 000.
fn qty_price_by_arith(numbers: &[(usize, f64, String)], total: f64) -> Option<(Num, Num)> {
    if total <= 0.0 {
        return None;
    }
    let tol = 0.02 * total + 0.5;
    // чистая пара
    for (qi, nq) in numbers.iter().enumerate() {
        let q = nq.1;
        if q <= 0.0 || q > 1000.0 {
            continue;
        }
        for (ci, nc) in numbers.iter().enumerate() {
            let c = nc.1;
            if ci == qi || c <= 0.0 {
                continue;
            }
            if (q * c - total).abs() <= tol {
                return Some((Num::new(q, nq.2.clone()), Num::new(c, nc.2.clone())));
            }
        }
    }
    // цена склеена без запятой
    for (_qi, nq) in numbers.iter().enumerate() {
        let q = nq.1;
        if q <= 0.0 || q > 1000.0 {
            continue;
        }
        for nc in numbers {
            let c = nc.1;
            if c <= 0.0 {
                continue;
            }
            let prod = q * c;
            if prod <= total {
                continue;
            }
            for k in 1..=3u32 {
                let f = 10f64.powi(k as i32);
                if (prod - total * f).abs() <= tol && raw_digits_only(&nc.2) {
                    let fixed = c / f;
                    return Some((Num::new(q, nq.2.clone()), Num::new(fixed, nc.2.clone())));
                }
            }
        }
    }
    None
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
    let toks: Vec<&str> = trimmed.split_whitespace().collect();
    if toks.is_empty() {
        return None;
    }
    // 1) маркер начала: строгий («Раздел 1. …») или нечёткий — OCR рвёт
    //    слово пробелами и глотает буквы: «аздел 1. Земляные…»,
    //    «Ра аздел 2. Фундамент» (в корпусе из-за этого пропадали ВСЕ
    //    разделы документа)
    let lower0 = toks[0].to_lowercase();
    let strict_sub = lower0.starts_with("подраздел");
    let strict_sec = (lower0.starts_with("раздел") || lower0.starts_with("глава")) && !strict_sub;
    let (fuzzy_sub, fuzzy_sec, consumed) = if strict_sub || strict_sec {
        (false, false, 1usize)
    } else {
        fuzzy_section_prefix(&toks)
    };
    let is_sub = strict_sub || fuzzy_sub;
    let is_sec = strict_sec || fuzzy_sec;
    if !is_sec && !is_sub {
        return None;
    }
    // 2) токены после маркера: пропускаем «№», точки, дефисы
    let rest: Vec<&str> = toks[consumed..].to_vec();
    let mut ri = 0usize;
    while ri < rest.len() {
        let t = rest[ri].trim_matches(|c: char| matches!(c, '.' | '-' | '–' | ')'));
        if t == "№" || t.is_empty() {
            ri += 1;
        } else {
            break;
        }
    }
    // 3) номер: ведущие цифры с точками («1», «2.1»)
    let mut number = None;
    let mut ti = ri;
    if ti < rest.len() {
        let t = rest[ti];
        let digits: String = t
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        let digits = digits.trim_end_matches('.');
        if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit() || c == '.') {
            number = Some(digits.to_string());
            ti += 1;
        }
    }
    // 4) название — остальное, без служебных хвостов
    let title: Vec<&str> = rest[ti..]
        .iter()
        .map(|t| t.trim_matches(|c: char| matches!(c, '.' | ':' | '—')))
        .filter(|t| !t.is_empty())
        .collect();
    let title = if title.is_empty() {
        "(без названия)".to_string()
    } else {
        title.join(" ")
    };
    Some((number, title, is_sub))
}

/// Нечёткий маркер «Раздел»/«Подраздел»/«Глава» в начале строки: копим
/// слова, пока форма `letters_squeezed` не станет похожа на маркер
/// (расстояние ≤ 1). Возвращает (подраздел?, раздел?, съедено токенов).
/// Копим до ЛУЧШЕГО совпадения: «Разде л 1 …» — по первому токену
/// расстояние 1 («разде»), по второму — 0 («раздел»): съесть оба,
/// иначе оторванный хвост марки попадает в название раздела.
fn fuzzy_section_prefix(toks: &[&str]) -> (bool, bool, usize) {
    let mut acc = String::new();
    // (расстояние, это подраздел?, съедено токенов)
    let mut best: Option<(usize, bool, usize)> = None;
    for (i, t) in toks.iter().take(3).enumerate() {
        let word: String = t.chars().filter(|c| c.is_alphabetic()).collect();
        if word.is_empty() {
            continue;
        }
        acc.push_str(&word.to_lowercase());
        let acc_lz = letters_squeezed(&acc);
        if acc_lz.len() >= 5 {
            let d_sub = edit_distance(&acc_lz, "подраздел");
            let d_sec = edit_distance(&acc_lz, "раздел");
            let d_ch = edit_distance(&acc_lz, "глава");
            let (d, is_sub) = if d_sub <= d_sec && d_sub <= d_ch {
                (d_sub, true)
            } else if d_sec <= d_ch {
                (d_sec, false)
            } else {
                (d_ch, false)
            };
            if d <= 1 && best.map(|(bd, _, _)| d < bd).unwrap_or(true) {
                best = Some((d, is_sub, i + 1));
            }
        }
        if acc_lz.len() > 12 {
            break; // маркер длиннее не бывает — дальше не копим
        }
    }
    match best {
        Some((_, is_sub, consumed)) => (is_sub, !is_sub, consumed),
        None => (false, false, 0),
    }
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
        if head != "итого" && head != "всего" && head != "итоги" {
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
            if let Some((k, _)) = totals_kind_text(&wl) {
                if matches!(k, TotalKind::Vat | TotalKind::DocWithVat) {
                    out[i] = true;
                    break;
                }
                if matches!(k, TotalKind::Doc) {
                    continue; // «Итого с непредвиденными» — не прерывает цепочку
                }
                break; // Section/Sub ближе — это не финальная сводка
            }
            if wl.contains("по смете") {
                out[i] = true;
                break;
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
    /// шапка таблицы уже виделась — до неё строки-«метка | значение»
    /// (СОГЛАСОВАНО, Сметная стоимость…) — шапка документа, не позиции
    saw_header_row: bool,
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
    /// хвост «Итоги по смете:» пошёл: голые «Итого N» — бегущие итоги
    /// ДОКУМЕНТА, а строки разбивки по видам работ — не позиции
    in_summary_tail: bool,
    /// «Примечания:» прошли — дальше только текст, не смета
    in_notes: bool,
    /// голый номер раздела («2» отдельной строкой) — название ждём ниже
    pending_section_no: Option<u32>,
}

impl Parser {
    fn feed(&mut self, row: DocRow) {
        dtag(&format!("строка col={}", row.columnar), &row.joined());
        if std::env::var("RUSTOCR_ESTIMATE_DEBUG").as_deref() == Ok("2") {
            let cells: Vec<String> = row
                .cells
                .iter()
                .map(|(c, s, t)| format!("[{}+{}]«{}»", c, s, t))
                .collect();
            eprintln!("[est:cells] {}", cells.join(" "));
        }
        if row.columnar {
            self.stats.table_rows += 1;
        }
        if row.page > self.pages {
            self.pages = row.page;
        }
        // заполнители итогов («расчетное», «по факту», «$») — пропускаем
        let joined = row.joined();
        let lower = joined.to_lowercase();
        // «Примечания:» — конец сметной части: контакты и пояснения после
        // него не становятся разделами и позициями
        if !self.in_notes && !row.columnar && self.saw_table_row && notes_marker(&lower) {
            self.in_notes = true;
            dtag("примечания", &joined);
            return;
        }
        if self.in_notes {
            self.push_unparsed(&joined, row.columnar);
            return;
        }
        // составной файл (ЛСР + ОСР + приложение): заголовок следующего
        // документа кончает хвост «Итоги по смете» предыдущего
        if self.in_summary_tail && new_doc_header(&lower) {
            self.in_summary_tail = false;
            dtag("новый-документ", &joined);
        }
        // колонтитулы, URL, контакты, логотипы — не содержимое сметы:
        // не позиции, не итоги и не «неразобранные» строки статистики
        if is_junk_line(&lower) {
            dtag("мусор", &joined);
            return;
        }
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
        let lower = joined.to_lowercase();
        // раздел может быть и полноширинной строкой
        if let Some((number, title, is_sub)) = section_header(&row) {
            self.start_section(number, title, is_sub);
            return;
        }
        // отложенный номер раздела («2» отдельной строкой): эта строка —
        // название («Строительство 1го этажа»)
        if let Some(n) = self.pending_section_no.take() {
            if section_title_ok(&joined) {
                dtag("раздел-номер", &joined);
                self.start_section(Some(n.to_string()), joined.trim().to_string(), false);
                self.after_total = false;
                return;
            }
        }
        // «N Название» одной строкой или голый «N» (номер раздела
        // коммерческой сметы без слова «Раздел») — сразу после итога
        // или до первого раздела
        if self.after_total || (self.saw_table_row && !self.saw_section) {
            if let Some((n, title)) = numbered_section_header(&row) {
                dtag("раздел-номер", &joined);
                self.start_section(Some(n.to_string()), title, false);
                self.after_total = false;
                return;
            }
            if self.after_total {
                if let Some(n) = lone_int(&row) {
                    self.pending_section_no = Some(n);
                    self.after_total = false;
                    dtag("раздел-номер", &joined);
                    return;
                }
            }
        }
        // итоги вне таблицы: «Итого по смете 123 руб.»
        if self.saw_table_row {
            let doc_mark = std::mem::take(&mut self.next_is_doc_total);
            if let Some((kind0, rate)) = totals_kind(&row) {
                let mut kind = if doc_mark && kind0 == TotalKind::Section {
                    TotalKind::Doc
                } else {
                    kind0
                };
                if kind == TotalKind::Section
                    && self.in_summary_tail
                    && bare_total_line(&lower)
                {
                    kind = TotalKind::Doc; // бегущий итог хвоста «Итоги по смете:»
                }
                if matches!(kind, TotalKind::Doc | TotalKind::DocWithVat) {
                    self.in_summary_tail = true;
                }
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
            // промежуточные сводки финальных страниц ГС вне сетки — тоже
            // закрывают текущую позицию, но не открывают «после итога»-пути
            if is_intermediate_summary(&lower) {
                dtag("сводка-промеж", &joined);
                self.flush_position();
                self.push_unparsed(&joined, false);
                return;
            }
            // НР/СП/«В том числе» — конец текущей позиции: ресурс уровня
            // сметы («Накладные расходы 13055,93»), а не позиция и не
            // продолжение. after_total НЕ ставим: пер-позиционные НР/СП
            // («НР (3067 руб.)…») идут среди позиций, и обрезки чисел
            // дальше («13») не должны становиться номерами разделов (sm_id)
            if self.is_summary_resource_line(&lower) {
                self.flush_position();
                if let Some(res) = self.parse_resource(&row) {
                    dtag("ресурс-сводка", &joined);
                    self.attach_resource(res, &joined);
                    return;
                }
                dtag("сводка-конец", &joined);
                self.push_unparsed(&joined, false);
                return;
            }
            // перенос названия позиции: полноширинная строка вне сетки со
            // строчной буквы/скобки — «естественного залегания
            // корчевателями-…» (в galaktion_lsr таких строк десятки)
            if !self.after_total
                && self.cur_position.is_some()
                && self.is_text_continuation(&joined, &lower)
            {
                let page = row.page;
                if let Some(pos) = self.cur_position.as_mut() {
                    let has_numbers = joined.chars().any(|c| c.is_ascii_digit());
                    if !has_numbers {
                        if !pos.name.starts_with('(') {
                            pos.name.push(' ');
                            pos.name.push_str(joined.trim());
                        }
                    }
                    pos.raw.push(joined.clone());
                    if pos.pages.last() != Some(&page) {
                        pos.pages.push(page);
                    }
                }
                dtag("продолж-е", &joined);
                return;
            }
            // мета-поля в хвосте документа («Подрядчик:…», «Заказчики:…»)
            if meta_field(&lower).is_some() {
                self.feed_meta_line(&joined);
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
            self.push_unparsed(&joined, false);
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

        // 1а) отложенный номер раздела: эта строка — название
        if let Some(n) = self.pending_section_no.take() {
            if section_title_ok(&joined) {
                dtag("раздел-номер", &joined);
                self.start_section(Some(n.to_string()), joined.trim().to_string(), false);
                self.after_total = false;
                return;
            }
        }

        // 1б) заголовок раздела «N Название» — коммерческие сметы без
        // слова «Раздел»: сразу после итога или до первого раздела;
        // у позиций всегда есть числа, у такого заголовка — нет
        if (self.after_total || !self.saw_section) && self.saw_table_row {
            if let Some((n, title)) = numbered_section_header(&row) {
                dtag("раздел-номер", &joined);
                self.start_section(Some(n.to_string()), title, false);
                self.after_total = false;
                return;
            }
        }

        // 2) шапка таблицы → роли колонок
        if is_header_row(&row) {
            self.saw_header_row = true;
            self.learn_roles(&row);
            return;
        }

        // 2а) до шапки таблицы: строки «метка | значение» — шапка ДОКУМЕНТА
        // (Гранд-Смета: «Сметная стоимость 2010,425 тыс. руб.» лежит в
        // сетке), а не позиции сметы — иначе вырастает фантомный «раздел»
        // из шапки и валидация считается от мусорных сумм
        if !self.saw_header_row
            && !self.saw_section
            && meta_keyword_row(&row.lower())
        {
            dtag("шапка-док", &joined);
            self.feed_meta_line(&joined);
            return;
        }

        // 3) итоги
        let doc_mark = std::mem::take(&mut self.next_is_doc_total);
        if let Some((kind0, rate)) = totals_kind(&row) {
            // метка предпросмотра: финальное «Итого …» перед НДС/«ВСЕГО по
            // смете» — итог документа, а не текущего раздела
            let mut kind = if doc_mark && kind0 == TotalKind::Section {
                TotalKind::Doc
            } else {
                kind0
            };
            if kind == TotalKind::Section
                && self.in_summary_tail
                && bare_total_line(&row.lower())
            {
                kind = TotalKind::Doc; // бегущий итог хвоста «Итоги по смете:»
            }
            if matches!(kind, TotalKind::Doc | TotalKind::DocWithVat) {
                self.in_summary_tail = true;
            }
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

        // 3б) промежуточные сводки финальных страниц Гранд-Сметы
        // («прямые затраты…», «итого строительные работы», «непредвиденные
        // затраты 2%») — не позиции: иначе ломают Σ позиций раздела;
        // ЗАКРЫВАЮТ текущую позицию (НР/СП дальше — на уровень раздела),
        // но НЕ открывают «после итога»-пути: обрезки чисел сводки («652»)
        // не должны становиться номерами новых разделов (sm_id)
        if is_intermediate_summary(&row.lower()) {
            dtag("сводка-промеж", &joined);
            self.flush_position();
            // сводка РАСПОЗНАНА — не «неразобранная строка таблицы»:
            // в unclassified_rows не считаем
            self.push_unparsed(&joined, false);
            return;
        }

        // 3в) продолжение позиции: без № и шифра, текст в колонке
        // наименования начинается со строчной буквы или скобки —
        // «(10 м³ основания) 82,01 3,65 25,55», «траншеях шириной до 2 м…»
        // (в хвосте сводки позиций уже нет — продолжений тоже)
        if !self.in_summary_tail && self.cur_position.is_some() && self.is_position_continuation(&row) {
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

        // 4) позиция (в хвосте «Итоги по смете:» позиций уже нет —
        // только ресурсы и неразобранные строки разбивки по видам работ)
        if !self.in_summary_tail {
            if let Some(pos) = self.parse_position(&row) {
                dtag("позиция", &joined);
                self.after_total = false;
                self.flush_position();
                self.cur_position = Some(pos);
                return;
            }
        }

        // 5а) хвост «Итоги по смете:»: разбивка по видам работ и блок
        // «В том числе» — не позиции, не ресурсы и не продолжения (позиции
        // уже закрыты), просто сырьё для LLM. НР/СП раздела стоят ВЫШЕ
        // маркера «Итоги по смете:» — тайл ещё не включён — и ресурсами
        // стать успеют; дубль из «В том числе» ниже сюда отфильтруется
        if self.in_summary_tail {
            self.flush_position();
            self.push_unparsed(&joined, true);
            dtag("хвост-сводки", &joined);
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

        self.push_unparsed(&joined, true);
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
            // продолжение живёт в КОЛОНКЕ наименования — не правее: ярлык
            // («водоснабжение», «для ускосин») в колонке МАТЕРИАЛОВ слева
            // от наименования — это метка новой позиции, не продолжение
            // (karkas_kd1: 61 позиция терялась при релаксации влево/вправо)
            Some(nc) => *col == nc,
            None => true,
        }
    }

    /// Строка-закрыватель позиций финальной сводки: НР/СП/«В том числе».
    /// С таких строк начинается хвост Гранд-Сметы — позиции после них
    /// закрыты, а сами они не позиции и не продолжения названий.
    fn is_summary_resource_line(&self, lower: &str) -> bool {
        // сворачиваем латинских двойников и внутрисловные пробелы:
        // «B том числе:» (латинская B) — обычный шум OCR
        let f = despace_fold(lower);
        f.starts_with("втомчисле")
            || f.starts_with("накладные")
            || f.starts_with("сметнаяприбыль")
            || f.starts_with("нр(")
            || f.starts_with("сп(")
            // «НР 1234…» / «СП 2750,8…» — сокращение с суммой, не слово
            || (f.starts_with("нр") || f.starts_with("сп"))
                && f.as_bytes().get(2).is_some_and(|&c| c.is_ascii_digit())
    }

    /// Полноширинная строка-перенос названия позиции: со строчной буквы
    /// или скобки, без двоеточий и точки на конце (у приксельных примечаний
    /// «оплачивается по факту его использования.» точка есть — у переносов
    /// названий почти никогда), не итог, не ресурс, не мета и не мусор.
    fn is_text_continuation(&self, joined: &str, lower: &str) -> bool {
        let t = joined.trim();
        if t.is_empty() {
            return false;
        }
        let first = t.chars().next().unwrap();
        if !(first.is_lowercase() || first == '(' || first == '«' || first == '"') {
            return false;
        }
        if t.contains(':') {
            return false; // «в том числе:», «нормативная трудоемкость: …»
        }
        if t.ends_with('.') {
            return false; // приписки-примечания, а не кусок названия
        }
        if t.chars().filter(|c| c.is_alphabetic()).count() < 3 {
            return false;
        }
        if t.chars().count() > 160 {
            return false;
        }
        let lz = letters_squeezed(lower);
        // итоговые слова — даже разорванные («и тог го вес»)
        if lz.starts_with("итог") || lz.starts_with("всег") || lz.starts_with("подытог") {
            return false;
        }
        if totals_kind_text(lower).is_some() || totals_kind_text(&lz).is_some() {
            return false;
        }
        if let Some(k) = resource_kind(lower) {
            if matches!(
                k,
                ResourceKind::Overhead | ResourceKind::Profit | ResourceKind::Transport
            ) {
                return false;
            }
        }
        if meta_field(lower).is_some() || meta_keyword_row(lower) {
            return false;
        }
        if is_intermediate_summary(lower) {
            return false;
        }
        if is_junk_line(lower) {
            return false;
        }
        true
    }

    /// Выбор числа-итога из строки. В выкладках ГС рядом с суммой идут
    /// трудоёмкость (чел.-ч) и «занятые обслуживанием машин» — правое число
    /// оказывается неверным. Если чисел несколько, берём ближайшее к
    /// Σ позиций (для документа — к Σ разделов и Σ+НДС).
    fn resolve_total_num(&self, kind: TotalKind, row: &DocRow, joined: &str) -> Option<f64> {
        // разрезанный границей колонок итог — сильное доказательство:
        // десятичный разделитель на стыке ячеек не бывает у разных метрик
        if let Some(v) = cut_total_value(row) {
            return Some(v);
        }
        let fallback = total_from_row(row).or_else(|| parse_total_number(joined));
        let mut candidates: Vec<f64> = row
            .cells
            .iter()
            .filter_map(|(_, _, t)| parse_number(t.trim()))
            .collect();
        // дополнительные кандидаты: ведущая группа разрядов, приклеенная
        // OCR к подписи итога («…е 39» | «126» → 39126; «…е 33» | «158» →
        // 33158). Хвост-цифры (1–3) смешанной ячейки + соседняя ЧИСТАЯ
        // трёхзначная ячейка. Якоря отсеют ложные сборки
        for w in row.cells.windows(2) {
            let t1 = w[0].2.trim();
            let t2 = w[1].2.trim();
            let tail: String = t1
                .chars()
                .rev()
                .take_while(|c| c.is_ascii_digit())
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            let t2_pure3 = t2.len() == 3 && t2.chars().all(|c| c.is_ascii_digit());
            if (1..=3).contains(&tail.len()) && !tail.starts_with('0') && t2_pure3 {
                if let Ok(v) = format!("{}{}", tail, t2).parse::<f64>() {
                    candidates.push(v);
                }
            }
        }
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
        if std::env::var("RUSTOCR_ESTIMATE_DEBUG").is_ok() {
            eprintln!(
                "[est:total-resolve] kind={:?} якоря={:?} кандидаты={:?}",
                kind, anchors, candidates
            );
        }
        let mut best: Option<(f64, f64, f64)> = None; // (расстояние, значение, якорь)
        for &c in &candidates {
            for &a in &anchors {
                let d = (c - a).abs();
                if best.map(|(bd, _, _)| d < bd).unwrap_or(true) {
                    best = Some((d, c, a));
                }
            }
        }
        // ни один кандидат не похож на СВОЙ якорь (мусор от скана) — прежний
        // путь. Порог меряем от якоря пары: иначе крупный якорь Σ+НДС
        // «легализует» мусорный кандидат, ближайший к малому якорю Σ
        // (сантехприборы: кандидат 126 при Σ=1920,74 и Σ+НДС=7888,74)
        let fits = best
            .map(|(d, _, a)| d <= 0.5 * a.max(1.0))
            .unwrap_or(false);
        if fits {
            best.map(|(_, v, _)| v)
        } else if candidates.len() >= 2 {
            // якоря не сошлись, кандидатов несколько. Два частых случая:
            // 1) деньги — САМОЕ КРУПНОЕ число строки, правый край —
            //    трудоёмкость («Итого 2031228,96 2389,4» → 2031228,96;
            //    «ВСЕГО по смете 39 126 | 115,59» → 39126);
            // 2) итог разрезан границей ячеек на осколки «126» | «115,59»
            //    → 126115,59 (ячеечная склейка: правое число + чистые
            //    1–3-значные ячейки слева).
            // Склейку делаем ТОЛЬКО по границам ячеек — в слитом тексте
            // «39 126 115,59» это одно число и трудоёмкость, не осколки.
            // Максимум считаем по РЕАЛЬНЫМ числам ячеек: склейные
            // кандидаты windows (хвост подписи + чистая тройка:
            // «Итого по разделу 1» | «500» → фантомные 1500) здесь
            // права голоса не имеют — без якоря они не отличимы от мусора.
            let mut real: Vec<f64> = row
                .cells
                .iter()
                .filter_map(|(_, _, t)| parse_number(t.trim()))
                .filter(|v| *v != 0.0)
                .collect();
            real.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            real.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
            let maxc = real.last().copied();
            let cellglue = total_from_row(row);
            match (maxc, cellglue) {
                (Some(m), Some(c)) => Some(if c > m { c } else { m }),
                (Some(m), None) => Some(m),
                (None, Some(c)) => Some(c),
                (None, None) => fallback,
            }
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

        // строка-ОБРЫВОК чисел (хвост разрезанного числа, обрывок колонки
        // «5 | 70636.95 | 442.78»): цифра-«номер» + числа, но ни одной
        // текстовой ячейки и ни шифра — фантомная позиция «(без названия)»,
        // к которой ещё и цепляются ресурсы. Настоящая позиция всегда
        // имеет наименование (или шифр расценки).
        let has_text_cell = row
            .cells
            .iter()
            .any(|(_, _, t)| t.trim().chars().filter(|c| c.is_alphabetic()).count() >= 2);
        if !has_text_cell && cipher.is_none() {
            return None;
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
                let has_code = row.cells.iter().enumerate().any(|(i, (col, _, t))| {
                    resource_code(t).is_some()
                        && !(is_six_digit(t)
                            && (Some(*col) == last_num_col || !is_first_nonempty(row, i)))
                });
                if has_code {
                    false
                } else {
                    // первая ТЕКСТОВАЯ ячейка строки — не обязательно первая
                    // непустая: в объектных сметных расчётах шифр («101-01-01»)
                    // стоит отдельной ячейкой перед наименованием
                    let first = row
                        .cells
                        .iter()
                        .find(|(_, _, t)| {
                            let t = t.trim();
                            t.chars().filter(|c| c.is_alphabetic()).count() >= 3 || is_date(t)
                        })
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
            // кол-во/цена без ролей (шапка искажена OCR) — восстановим из
            // арифметики строки при известном итоге: двухценовая раскладка
            // «материалы + работы», чистая пара q × c = total или цена,
            // склеенная без запятой (q × c = total × 10^k)
            if total.is_some() && price.is_none() {
                let tval = total.as_ref().unwrap().value;
                if let Some((dq, dp)) = dual_price_pair(numbers, tval) {
                    if qty
                        .as_ref()
                        .map(|q| (q.value - dq.value).abs() < 1e-9)
                        .unwrap_or(true)
                    {
                        let q = qty.clone().unwrap_or(dq);
                        return (Some(q), Some(dp), total);
                    }
                }
                if let Some((aq, ap)) = qty_price_by_arith(numbers, tval) {
                    if qty
                        .as_ref()
                        .map(|q| (q.value - aq.value).abs() < 1e-9)
                        .unwrap_or(true)
                    {
                        let q = qty.clone().unwrap_or(aq);
                        return (Some(q), Some(ap), total);
                    }
                }
                if qty.is_none() {
                    if let Some((q, p, t)) = arithmetic_triple(numbers) {
                        if (t.value - tval).abs() <= 0.02 * tval + 0.5 {
                            // «6 780,0 × 1,0»: если «кол-во» оказалось больше
                            // «цены» при малой цене — порядок спутан
                            let (q, p) = if q.value > p.value && p.value <= 1000.0 {
                                (p, q)
                            } else {
                                (q, p)
                            };
                            return (Some(q), Some(p), total);
                        }
                    }
                }
            }
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
                // двухценовая раскладка «материалы + работы»: кол-во ×2,
                // стоимости ×2, их сумма — правое число строки
                if let Some(last) = numbers.last() {
                    let tv = last.1;
                    if let Some((q, p)) = dual_price_pair(numbers, tv) {
                        return (Some(q), Some(p), Some(Num::new(tv, last.2.clone())));
                    }
                }
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
        // последним числом строки — иначе это сумма («… 110000 арматура»);
        // и вообще код машины стоит ПЕРВОЙ непустой ячейкой строки
        // («060148 Кран…»), а цена — правее наименования («Работа
        // автокрана смена 320000 1,5 48 000»: 320000 — цена, не код)
        let last_num_col = row.numbers().last().map(|(c, _, _)| *c);
        let mut code_col: Option<usize> = None;
        let mut code: Option<String> = None;
        for (i, (col, _, text)) in row.cells.iter().enumerate() {
            if resource_code(text).is_some() {
                if is_six_digit(text)
                    && (Some(*col) == last_num_col || !is_first_nonempty(row, i))
                {
                    continue; // чистые 6 цифр не в начале строки — цена/сумма
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
        // продолжение названия позиции/ресурса (в колонке наименования
        // или правее — длинные блоки режутся границами)
        let name_col_ok = self.name_col.map(|nc| nc <= *col + 1).unwrap_or(true);
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
        // заголовок раздела — новое начало: «голый» заголовок сразу после
        // него (подраздел Гранд-Сметы «ремонт малой сцены») не должен
        // приниматься за следующий раздел по правилу «после итога»
        self.after_total = false;
        // раздел после хвоста сводки — например, приложение; хвост кончился
        self.in_summary_tail = false;
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

    fn push_unparsed(&mut self, joined: &str, columnar: bool) {
        if columnar {
            // неклассифицированные СТРОКИ ТАБЛИЦЫ: текстовые строки вне
            // таблицы (шапка документа, примечания) в статистику ошибок
            // распознавания таблиц не смешиваем
            self.stats.unclassified_rows += 1;
        }
        if self.unparsed.len() < 200 {
            self.unparsed.push(joined.trim().to_string());
        }
    }

    fn finish(mut self) -> EstimateDocument {
        self.flush_all_open();
        // полностью ПУСТЫЕ разделы (ни позиций, ни подразделов, ни ресурсов,
        // ни итога, ни номера) — мусор: строки шапки «Земляные работы»
        // (вид работ в шапке сметы), обрывки. Первый раздел коммерческих
        // смет «(без разделов)» никогда не пуст — позиции уже внутри
        self.sections.retain(|s| {
            !(s.positions.is_empty()
                && s.subsections.is_empty()
                && s.resources.is_empty()
                && s.total.is_none()
                && s.number.is_none()
                && s.title != "(без разделов)")
        });
        // единственный безымянный раздел — вся смета: его итог становится
        // итогом документа (коммерческие сметы без «Разделов»), но НЕ
        // затирает явно распознанный «Всего по смете» (1721671 ≠ Σ
        // раздела с командировочными — primer_smet_3-1)
        if self.sections.len() == 1 && self.totals.total.is_none() {
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
    fn fuzzy_matching_of_torn_keywords() {
        // OCR рвёт ключевые слова пробелами и глотает буквы
        let lz = |s: &str| letters_squeezed(&s.to_lowercase());
        assert!(fuzzy_contains(&lz("с Сметна ая прибыл 1ь"), "сметнаяприбыль"));
        assert!(fuzzy_contains(&lz("прям мые затра ты"), "прямыезатраты"));
        assert!(fuzzy_contains(&lz("наклад ные расходы"), "накладныерасходы"));
        // обычные слова не должны случайно совпадать
        assert!(!fuzzy_contains(&lz("гидроизоляция фундамента боковая"), "сметнаяприбыль"));
        assert!(!fuzzy_contains(&lz("кладка стен из кирпича"), "накладныерасходы"));
        assert!(!fuzzy_contains(&lz("устройство кровли из металлочерепицы"), "прямыезатраты"));
        // разорванное «по смете» схлопывается
        assert!(despace_fold("и по с смете").contains("поссмете"));
    }

    #[test]
    fn junk_lines_filtered() {
        assert!(is_junk_line("htt p://www.smet ta-mosc ow.ru/ c траница 1"));
        assert!(is_junk_line("Гра нд-CMET A"));
        assert!(is_junk_line("г.Москва а, Электролитный проезд 3 стр.28. Тел.: +7 (495) 64-614-64"));
        assert!(is_junk_line("версия 6.2.7 20221031"));
        assert!(is_junk_line("e-mail: zhitnik@vfstroy.ru"));
        // содержимое сметы мусором НЕ считается
        assert!(!is_junk_line("кладка стен из кирпича м2 1200"));
        assert!(!is_junk_line("итого по разделу 1 6 172,50"));
        assert!(!is_junk_line("устройство кровли из профилированной металлочерепицы"));
    }

    #[test]
    fn mangled_section_headers_recognized() {
        // «аздел 1. Землянные работы» — OCR глотнул «Р»
        let r = row(&[("аздел 1. Землян ные работы", 0)], false);
        assert_eq!(
            section_header(&r),
            Some((Some("1".to_string()), "Землян ные работы".to_string(), false))
        );
        // «Ра аздел 2. Фундамент» — лишний пробел внутри слова
        let r = row(&[("Ра", 1), ("аздел 2. Фундам мент", 2), ("Перекр ытие", 3)], true);
        assert_eq!(
            section_header(&r),
            Some((Some("2".to_string()), "Фундам мент Перекр ытие".to_string(), false))
        );
        // строгий путь не сломался
        let r = row(&[("Раздел 1. Земляные работы", 0)], false);
        assert_eq!(
            section_header(&r),
            Some((Some("1".to_string()), "Земляные работы".to_string(), false))
        );
        // подраздел
        let r = row(&[("Подраздел 2.1 Кладка", 0)], false);
        let (n, _, is_sub) = section_header(&r).unwrap();
        assert_eq!((n.as_deref(), is_sub), (Some("2.1"), true));
        // позиция с «раздел...»-подобным словом — НЕ раздел
        let r = row(&[("Кладка перегородок из пазогребневых плит", 0)], false);
        assert_eq!(section_header(&r), None);
    }

    #[test]
    fn summary_tails_not_totals_or_positions() {
        // «Итого прямые затраты ПО СМЕТЕ в ценах 2001г.» — промежуточная
        // сводка, а не итог документа (был мусорный итог 3,3 млрд)
        assert!(is_intermediate_summary(
            "ітог о прям мые затра ты по смете в цен ах 2001г. 180347.4"
        ));
        assert_eq!(totals_kind_text("ітог о прям мые затра ты по смете в цен ах 2001г."), None);
        // «Итоги» (мн.ч.) — итог документа
        assert!(matches!(totals_kind_text("итоги по смете:"), Some((TotalKind::Doc, _))));
        // замусоленная «Сметная прибыль» — ресурс, а не фантомная позиция
        assert_eq!(resource_kind("с сметна ая прибыл 1ь"), Some(ResourceKind::Profit));
        assert_eq!(
            resource_kind("наклад ные расходы"),
            Some(ResourceKind::Overhead)
        );
        // обычные «по смете» с итоговой головой — по-прежнему итог документа
        assert!(matches!(totals_kind_text("всего по смете 1544414.6"), Some((TotalKind::Doc, _))));
    }

    #[test]
    fn cut_total_fragments_joined() {
        // колоночная граница разрубила итог: «1 544414» | «.6» → 1544414.6
        let r = row(&[("BC ЕГО п о смете", 2), ("1 544414", 22), (".6", 23)], true);
        assert_eq!(total_from_row(&r), Some(1544414.6));
        // «2 00149.» | «96» → 200149.96
        let r = row(&[("Ито го", 2), ("2 00149.", 22), ("96", 23)], true);
        assert_eq!(total_from_row(&r), Some(200149.96));
        // «2 08644.» | «33» → 208644.33
        let r = row(&[("Ито го", 2), ("2 08644.", 22), ("33", 23)], true);
        assert_eq!(total_from_row(&r), Some(208644.33));
        // РАЗНЫЕ метрики не склеиваются: «57» | «391.62» → 391.62 (как было)
        let r = row(&[("…", 2), ("57", 26), ("391.62", 28)], true);
        assert_eq!(total_from_row(&r), Some(391.62));
        // прежний путь не сломался: «1» | «341 653P» → 1341653
        let r = row(&[("ИТОГО", 2), ("1", 22), ("341 653P", 23)], true);
        assert_eq!(total_from_row(&r), Some(1341653.0));
    }

    #[test]
    fn bare_total_lines() {
        // голые бегущие итоги хвоста «Итоги по смете:»
        assert!(bare_total_line("ито го 2 00149. 96"));
        assert!(bare_total_line("итого 204152.96"));
        assert!(bare_total_line("всего по смете 1544414.6") == false); // «по смете» — отдельное правило
        // итог раздела с подписью — НЕ голый
        assert!(!bare_total_line("итого по разделу 1 6 172,50"));
        assert!(!bare_total_line("итого прямые затраты по смете 180347.4"));
        assert!(!bare_total_line("итого с непредвиденными 212817.22"));
        // обычная позиция — не итог
        assert!(!bare_total_line("кладка стен м2 1200 300 000"));
    }

    #[test]
    fn fullwidth_continuation_line() {
        // перенос названия позиции: строчная буква, без двоеточий/точки
        let p = Parser::default();
        assert!(p.is_text_continuation("естественног о залегания корчевателями л-", "естественног о залегания корчевателями л-"));
        assert!(p.is_text_continuation("кустарник и м иелколесье средние", "кустарник и м иелколесье средние"));
        // приписки и служебные строки — НЕ продолжения
        assert!(!p.is_text_continuation("в том числе:", "в том числе:"));
        assert!(!p.is_text_continuation("оплачивается по факту его использования.", "оплачивается по факту его использования."));
        assert!(!p.is_text_continuation("н орм мативная т руд оемкость: 238 35.00 ч.-час.", "н орм мативная т руд оемкость: 238 35.00 ч.-час."));
        assert!(!p.is_text_continuation("накладные расходы 13055.93", "накладные расходы 13055.93"));
        assert!(!p.is_text_continuation("Земляные работы, выполняемые вручную", "земляные работы, выполняемые вручную"));
    }

    #[test]
    fn latin_lookalikes_folded_for_matching() {
        // «Bcero» (латинские B и r) читается как «Всего» — только для
        // матчинга ключевых слов, исходный текст не меняется
        assert_eq!(fold_latin_lookalikes("bcero"), "всего");
        assert_eq!(fold_latin_lookalikes("cмeтe"), "смете");
        assert_eq!(role_of_header("Bcero"), Some(Role::Total));
        assert_eq!(role_of_header("Bcero ("), Some(Role::Total));
        assert_eq!(role_of_header("Koл."), Some(Role::Qty));
    }

    #[test]
    fn vat_mangle_fragments() {
        // осколки OCR от «НДС 18%»
        assert!(vat_label_mangle("h18%"));
        assert!(vat_label_mangle("ндc18%"));
        assert!(vat_label_mangle("ндоc18%")); // «ндо C 18%» с просечкой
        assert!(vat_label_mangle("н18%"));
        assert!(!vat_label_mangle("ндс"));
        assert!(!vat_label_mangle("итог18"));
        // в контексте итогов (текст приходит нижним регистром, как из row.lower())
        assert!(matches!(
            totals_kind_text("нд c 18% 306674,94"),
            Some((TotalKind::Vat, _))
        ));
        assert!(matches!(
            totals_kind_text("итого с непредвиденными 1703749,65"),
            Some((TotalKind::Doc, _))
        ));
        assert_eq!(parse_vat_rate("ндс 18%"), Some(0.18));
        assert_eq!(parse_vat_rate("ндс (20%)"), Some(0.2));
    }

    #[test]
    fn resource_codes_with_ocr_junk() {
        // код с маркером-мусором и разорванной точкой
        assert_eq!(resource_code("Уд 1.102-0008"), Some("1.102-0008".to_string()));
        assert_eq!(resource_code("ya 2.102-0049"), Some("2.102-0049".to_string()));
        assert_eq!(resource_code("3 102-0061"), Some("3.102-0061".to_string()));
        // шестизначные числа ВНУТРИ текста — цены, не коды машин
        assert_eq!(resource_code("Работа автокрана смена 320000 1,5 48 000"), None);
        // код машины — чистая ячейка из 6 цифр
        assert_eq!(resource_code("060148"), Some("060148".to_string()));
        assert_eq!(resource_code("101-0797"), Some("101-0797".to_string()));
    }

    #[test]
    fn intermediate_summaries_not_positions() {
        assert!(is_intermediate_summary(
            "итого прямые затраты по смете в ценах 2001г. 196082,29"
        ));
        assert!(is_intermediate_summary("непредвиденные затраты 2% 33406,86"));
        assert!(is_intermediate_summary("итого строительные работы 1645550,13"));
        assert!(!is_intermediate_summary("итого по разделу 1 500,00"));
        assert!(!is_intermediate_summary("кладка стен"));
    }

    #[test]
    fn numbers_glued_cells_rejected() {
        // несколько чисел, склеенных в одну ячейку — НЕ число
        assert_eq!(parse_number("179000 4,2 75 180"), None);
        assert_eq!(parse_number("6 208,0 1,0 8 820,0"), None);
        assert_eq!(parse_number("1,0 6 208"), None);
        // одно число — как было
        assert_eq!(parse_number("1 234,56"), Some(1234.56));
        assert_eq!(parse_number("10 0000"), Some(100000.0));
        assert_eq!(parse_number("2781 031"), Some(2781031.0));
        assert_eq!(parse_number("13 560,0"), Some(13560.0));
    }

    #[test]
    fn numbered_sections_recognized() {
        // «1 Строительство фундамента» — номер в отдельной ячейке
        let r = row(&[("1", 0), ("Строительство фундамента", 1)], true);
        assert_eq!(
            numbered_section_header(&r),
            Some((1, "Строительство фундамента".to_string()))
        );
        // «2 Строительство 1го этажа» — всё в одной ячейке, цифра в слове
        let r = row(&[("2 Строительство 1го этажа", 0)], false);
        assert_eq!(
            numbered_section_header(&r),
            Some((2, "Строительство 1го этажа".to_string()))
        );
        // позиция с числами — НЕ заголовок
        let r = row(
            &[("3", 0), ("Кладка стен", 1), ("м3", 2), ("80,0", 3), ("1 650,0", 4), ("132 000", 5)],
            true,
        );
        assert_eq!(numbered_section_header(&r), None);
        // голый номер и мусорные фрагменты примечаний
        assert!(lone_int(&row(&[("2", 0)], false)).is_some());
        // трёхзначные — обрезки чисел, не номера разделов
        assert_eq!(lone_int(&row(&[("123", 0)], false)), None);
        assert!(lone_int(&row(&[("2.", 0)], false)).is_some(), "точка после номера — ок");
        assert_eq!(lone_int(&row(&[("1234", 0)], false)), None);
    }

    #[test]
    fn notes_and_junk_titles_filtered() {
        assert!(notes_marker("примечания:"));
        assert!(notes_marker("1. примечания"));
        assert!(!notes_marker("1. земляные работы"));
        // фрагменты примечаний/адресов/логотипы — не названия разделов
        assert!(!title_like("определения отметки чистого пола"), "строчная буква");
        assert!(!title_like("строительного мусора"), "строчная буква");
        assert!(!title_like("г. Москва"), "строчная буква");
        assert!(!title_like("www.vfstroy.ru"));
        assert!(!title_like("FULLHOUSE"), "латинский капс-логотип");
        assert!(!bare_title("Кладка 2 этаж")); // «2» — отдельное число
        assert!(title_like("Монтаж кровли"));
        assert!(title_like("Строительные леса"));
    }

    #[test]
    fn doc_total_kind_for_dom() {
        // «Итого дом в комплектации "КОРОБКА"» — итог ДОКУМЕНТА
        assert_eq!(
            totals_kind_text("итого дом в комплектации \"коротбка\""),
            Some((TotalKind::Doc, None))
        );
        assert_eq!(totals_kind_text("итого строительство 1го этажа"), Some((TotalKind::Section, None)));
        assert_eq!(totals_kind_text("итого по смете"), Some((TotalKind::Doc, None)));
    }

    #[test]
    fn dual_price_and_arith_recovery() {
        // двухценовая строка: 1650×80 + 1494×80 = 251 520
        let nums: Vec<(usize, f64, String)> = [
            (2usize, 1650.0f64, "1 650,0"),
            (3, 80.0, "80,0"),
            (4, 1494.0, "1 494,0"),
            (5, 80.0, "80,0"),
            (6, 132000.0, "132 000"),
            (7, 119520.0, "119 520"),
            (8, 251520.0, "251 520"),
        ]
        .iter()
        .map(|(a, b, c)| (*a, *b, c.to_string()))
        .collect();
        let (q, p) = dual_price_pair(&nums, 251520.0).expect("двухценовая пара");
        assert_eq!(q.value, 80.0);
        assert!((p.value - 3144.0).abs() < 1e-6);
        // чистая пара: 1 × 6780 = 6780
        let nums2: Vec<(usize, f64, String)> = [
            (2, 6780.0, "6 780,0"),
            (3, 1.0, "1,0"),
            (6, 6780.0, "6 780"),
            (8, 6780.0, "6 780"),
        ]
        .iter()
        .map(|(a, b, c)| (*a, *b, c.to_string()))
        .collect();
        let (q, p) = qty_price_by_arith(&nums2, 6780.0).expect("чистая пара");
        assert_eq!((q.value, p.value), (1.0, 6780.0));
        // цена склеена: 3 × 100000 = 30000 × 10 → цена 10 000
        let nums3: Vec<(usize, f64, String)> = [
            (2, 100000.0, "10 0000"),
            (3, 3.0, "3,0"),
            (6, 30000.0, "30 000"),
            (8, 30000.0, "30 000"),
        ]
        .iter()
        .map(|(a, b, c)| (*a, *b, c.to_string()))
        .collect();
        let (q, p) = qty_price_by_arith(&nums3, 30000.0).expect("склеенная цена");
        assert_eq!((q.value, p.value), (3.0, 10000.0));
    }

    #[test]
    fn commercial_estimate_numbered_sections_flow() {
        // конвейер мини-сметы в стиле Full Houses: «1 Строительство
        // фундамента», позиции без №, «Итого …», «Итого дом…», «Примечания»
        let rows = vec![
            row(&[("СМЕТА на строительно-монтажные работы", 0)], false),
            row(&[("№", 0), ("Наименование", 1), ("Ед.", 2), ("Кол-во", 3), ("Цена", 4), ("Стоимость", 5)], true),
            row(&[("1", 0), ("Строительство фундамента", 1)], true),
            row(&[("Вынос проекта дома на участок", 1), ("компл", 2), ("1,0", 3), ("6 780,0", 4), ("6 780", 5)], true),
            row(&[("Итого строительство фундамента:", 1), ("6 780", 5)], true),
            row(&[("Итого дом в комплектации \"КОРОБКА\":", 1), ("6 780", 5)], true),
            row(&[("Примечания:", 0)], false),
            row(&[("1. Земляные работы рассчитаны для ровного участка", 0)], false),
        ];
        let doc = parse_document(rows);
        assert_eq!(doc.sections.len(), 1, "разделы: {:?}", doc.sections.iter().map(|s| &s.title).collect::<Vec<_>>());
        assert_eq!(doc.sections[0].number.as_deref(), Some("1"));
        assert_eq!(doc.sections[0].title, "Строительство фундамента");
        assert_eq!(doc.sections[0].positions.len(), 1);
        assert_eq!(doc.sections[0].total.as_ref().unwrap().value, 6780.0);
        assert_eq!(doc.totals.total.as_ref().unwrap().value, 6780.0);
        // примечания не стали разделами
        assert!(doc.sections.iter().all(|s| s.title != "1. Земляные работы рассчитаны для ровного участка"));
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
