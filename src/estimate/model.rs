//! Модель иерархии сметы: документ → разделы → позиции → ресурсы,
//! плюс итоги и отчёт валидации. Сериализуется в структурный JSON.

use serde::{Deserialize, Serialize};

/// Число из сметы: значение + исходный текст (для трассировки и LLM).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Num {
    pub value: f64,
    #[serde(rename = "raw")]
    pub raw: String,
}

impl Num {
    pub fn new(value: f64, raw: impl Into<String>) -> Self {
        Self {
            value,
            raw: raw.into(),
        }
    }
}

/// Тип ресурса позиции.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    /// затраты труда рабочих
    Labor,
    /// эксплуатация машин
    Machine,
    /// оплата труда машинистов
    MachineLabor,
    /// материалы
    Material,
    /// транспортные расходы
    Transport,
    /// накладные расходы
    Overhead,
    /// сметная прибыль
    Profit,
    Other,
}

impl ResourceKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceKind::Labor => "labor",
            ResourceKind::Machine => "machine",
            ResourceKind::MachineLabor => "machine_labor",
            ResourceKind::Material => "material",
            ResourceKind::Transport => "transport",
            ResourceKind::Overhead => "overhead",
            ResourceKind::Profit => "profit",
            ResourceKind::Other => "other",
        }
    }
}

/// Ресурс позиции (материал / механизм / трудозатраты).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Resource {
    pub kind: ResourceKind,
    /// код ресурса (101-0797, 1.1-1-234, 060148…)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantity: Option<Num>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// цена за единицу
    pub price: Option<Num>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// всего затрат по ресурсу
    pub total: Option<Num>,
}

/// Позиция сметы (работа по расценке или строка коммерческой сметы).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Position {
    /// № п/п (если распознан)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    /// шифр расценки (ГЭСН 01-01-013-01, ФЕР…, ТЕР…)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cipher: Option<String>,
    /// наименование
    pub name: String,
    /// единица измерения
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// количество
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantity: Option<Num>,
    /// цена за единицу
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price: Option<Num>,
    /// всего затрат по позиции
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<Num>,
    /// ресурсы позиции (материалы, механизмы, труд)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<Resource>,
    /// страницы, где встретилась позиция
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pages: Vec<usize>,
    /// исходные строки (ячейки таблицы) — сырьё для LLM-парсера и отладки
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub raw: Vec<String>,
}

/// Раздел (или подраздел — вложенность один уровень).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Section {
    /// номер раздела («1», «2.1»)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub number: Option<String>,
    /// название («Земляные работы»)
    pub title: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subsections: Vec<Section>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub positions: Vec<Position>,
    /// ресурсы вне позиций (напр., НР/СП по разделу)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<Resource>,
    /// «Итого по разделу»
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<Num>,
    /// страницы раздела
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pages: Vec<usize>,
}

impl Section {
    /// Все позиции раздела вместе с подразделами (в порядке документа).
    pub fn all_positions(&self) -> Vec<&Position> {
        let mut v: Vec<&Position> = self.positions.iter().collect();
        for s in &self.subsections {
            v.extend(s.all_positions());
        }
        v
    }
}

/// Итоги по документу.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Totals {
    /// «Итого по смете» (без НДС)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<Num>,
    /// ставка НДС (0.2 — «НДС (20%)»)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vat_rate: Option<f64>,
    /// сумма НДС
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vat: Option<Num>,
    /// «Всего с НДС»
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_with_vat: Option<Num>,
}

/// Шапка документа (что удалось вытащить из текста).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Meta {
    /// «Локальная смета № …» / название
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub number: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub customer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contractor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
    /// прочие строки шапки («Сметная стоимость…», «Норматив трудозатрат…»)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

/// Проблема, найденная валидатором.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Issue {
    /// short_kind: qty_price_mismatch / section_total_mismatch / …
    pub kind: String,
    pub message: String,
    /// где: «Раздел 2, позиция 5» / «документ»
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
}

/// Сводка, что проверено.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ValidationStats {
    pub positions: usize,
    pub sections: usize,
    pub resources: usize,
    /// позиций, где проверено «количество × цена = сумма»
    pub qty_price_checked: usize,
    /// разделов с проверенным итогом
    pub section_totals_checked: usize,
    /// строк таблицы, не классифицированных парсером
    pub unclassified_rows: usize,
    /// всего табличных строк
    pub table_rows: usize,
}

/// Отчёт валидации.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ValidationReport {
    pub errors: Vec<Issue>,
    pub warnings: Vec<Issue>,
    pub stats: ValidationStats,
}

impl ValidationReport {
    pub fn error(&mut self, kind: &str, message: String, location: Option<String>) {
        self.errors.push(Issue {
            kind: kind.to_string(),
            message,
            location,
        });
    }
    pub fn warn(&mut self, kind: &str, message: String, location: Option<String>) {
        self.warnings.push(Issue {
            kind: kind.to_string(),
            message,
            location,
        });
    }
}

/// Документ-смета целиком.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EstimateDocument {
    pub meta: Meta,
    pub sections: Vec<Section>,
    pub totals: Totals,
    pub validation: ValidationReport,
    /// строки документа, которые парсер не понял (сырьё для LLM)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unparsed_lines: Vec<String>,
    /// сколько страниц было на входе
    pub pages: usize,
}

impl EstimateDocument {
    /// Все позиции документа по порядку (разделы + подразделы).
    pub fn all_positions(&self) -> Vec<&Position> {
        let mut v = Vec::new();
        for s in &self.sections {
            v.extend(s.all_positions());
        }
        v
    }
}
