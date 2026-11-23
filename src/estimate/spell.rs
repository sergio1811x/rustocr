//! Офлайн-словарь ru_RU (Hunspell dic из LibreOffice/dictionaries) для
//! проверки и починки ключевых слов сметного парсера: рваные OCR-слова
//! («Фундам мент», «Проемь ы») склеиваются, если склейка есть в словаре.
//! Словарь закэширован в models/dict/ru_RU.dic — интернет не нужен.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::OnceLock;

fn models_dir() -> PathBuf {
    std::env::var("RUSTOCR_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("models"))
}

fn load_dict() -> HashSet<String> {
    let mut set = HashSet::new();
    let p = models_dir().join("dict/ru_RU.dic");
    if let Ok(txt) = std::fs::read_to_string(&p) {
        // формат Hunspell: первая строка — число слов, далее «слово/флаги»
        for line in txt.lines().skip(1) {
            let w = line.split('/').next().unwrap_or("").trim().to_lowercase();
            let n = w.chars().count();
            if n >= 2 && n <= 30 && w.chars().all(|c| c.is_alphabetic() || c == '-') {
                set.insert(w);
            }
        }
    }
    set
}

fn dict() -> &'static HashSet<String> {
    static D: OnceLock<HashSet<String>> = OnceLock::new();
    D.get_or_init(load_dict)
}

/// Слово есть в словаре (ё нормализуется к «е», как в словаре).
/// Hunspell dic хранит ЛЕММЫ с флагами аффиксов («проем/K»), поэтому
/// частые словоформы («проемы») сводим к лемме отрезанием флексии.
pub fn is_word(w: &str) -> bool {
    let w = w.trim().to_lowercase();
    if w.is_empty() {
        return false;
    }
    let flat = w.replace('ё', "е");
    if dict().contains(&w) || dict().contains(&flat) {
        return true;
    }
    // словоформа → лемма: «проемы» → «проем», «магазина» → «магазин».
    // «а/я/у/ю/е» — частые падежные флексии; без них осмысленное слово
    // считалось рваным, и склейка портила заголовок («магазина и кафе»
    // → «магазини кафе», golden smeta_objektnaya_moyka)
    const ENDINGS: [&str; 19] = [
        "ами", "ями", "ов", "ев", "ам", "ям", "ах", "ях", "ой", "ых", "ий", "ый", "ы", "и", "а",
        "я", "у", "ю", "е",
    ];
    for e in ENDINGS {
        if let Some(stem) = flat.strip_suffix(e) {
            if stem.chars().count() >= 4 && dict().contains(stem) {
                return true;
            }
        }
    }
    false
}

/// Склейка рваной пары: «фундам» + «мент» → «фундамент». Убирает
/// удвоение буквы на стыке (фундам|мент → фундаммент), результат
/// обязан быть словарным словом.
pub fn fix_torn(a: &str, b: &str) -> Option<String> {
    let a = a.trim().trim_end_matches('-').to_lowercase();
    let b = b.trim().to_lowercase();
    if a.chars().count() < 3 || b.chars().count() < 1 {
        return None;
    }
    // второй фрагмент — самостоятельное слово («и», «в», «на»):
    // рваный хвост так не выглядит, склейка скорее порвёт пару слов
    if b.chars().count() <= 2 && is_word(&b) {
        return None;
    }
    let mut cands = Vec::with_capacity(3);
    let joined = format!("{a}{b}");
    cands.push(joined);
    // удвоенная буква на стыке склейки
    let a_cut: String = a.chars().take(a.chars().count() - 1).collect();
    cands.push(format!("{a_cut}{b}"));
    let b_cut: String = b.chars().skip(1).collect();
    cands.push(format!("{a}{b_cut}"));
    cands.into_iter().find(|c| is_word(c))
}

/// Короткие предлоги, которые могут «отклеиться» от слова при OCR
/// («про изводство», «пере крытие» — стыковое слово длинное)
fn is_glued_prefix(w: &str) -> bool {
    matches!(w, "в" | "с" | "к" | "о" | "у" | "на" | "по" | "за" | "от" | "до" | "из" | "про")
}

/// Починка рваных слов в заголовке/ключевой фразе: «Фундам мент» →
/// «Фундамент», «про изводство» → «производство». Правит ТОЛЬКО пары,
/// склейка которых — словарное слово; осмысленные пары («под ключ»,
/// «по смете») не трогает. Регистр первой буквы сохраняется.
pub fn fix_title(title: &str) -> String {
    if dict().is_empty() {
        // словаря нет — работает как раньше
        return title.to_string();
    }
    let toks: Vec<&str> = title.split_whitespace().collect();
    let mut out: Vec<String> = Vec::with_capacity(toks.len());
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        if i + 1 < toks.len() {
            let u = toks[i + 1];
            // рваное слово: первый фрагмент сам по себе не слово (OCR режет
            // середину слова: «фундам|мент»), либо короткий предлог отклеился
            // от словарного корня («про изводство»). Осмысленные пары из
            // двух слов («под ключ», «по смете») не трогаем.
            let glue_allowed = !is_word(t) || (is_glued_prefix(t) && !is_word(u));
            if glue_allowed {
                if let Some(f) = fix_torn(t, u) {
                    // регистр: «Фундам мент» → «Фундамент»
                    let up = t.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
                    let mut cs = f.chars();
                    let styled = match cs.next() {
                        Some(c) => {
                            let c2 = if up { c.to_uppercase().next().unwrap_or(c) } else { c };
                            once_char(c2).chain(cs).collect::<String>()
                        }
                        None => f,
                    };
                    out.push(styled);
                    i += 2;
                    continue;
                }
            }
        }
        out.push(t.to_string());
        i += 1;
    }
    out.join(" ")
}

fn once_char(c: char) -> std::iter::Once<char> {
    std::iter::once(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torn_words_joined() {
        // «Фундам мент» (удвоение стыка), «Проемь ы», «про изводство»
        assert_eq!(fix_title("Фундам мент"), "Фундамент");
        assert_eq!(fix_title("Проемь ы"), "Проемы");
        assert_eq!(fix_title("про изводство работ"), "производство работ");
        // осмысленные пары не трогаем
        assert_eq!(fix_title("под ключ"), "под ключ");
        assert_eq!(fix_title("по смете"), "по смете");
        assert_eq!(fix_title("Земляные работы"), "Земляные работы");
        // регрессия moyka: «магазина и кафе» ≠ «магазини кафе» — союз «и»
        // не хвост рваного слова, а «магазина» — словарная словоформа
        assert_eq!(
            fix_title("Строительство мойки легкового транспорта, магазина и кафе"),
            "Строительство мойки легкового транспорта, магазина и кафе"
        );
        assert_eq!(fix_title("магазин и кафе"), "магазин и кафе");
        // рваное слово остаётся чинимым
        assert_eq!(fix_title("Фундам мент"), "Фундамент");
    }
}
