//! DTO ПАЗЛ — обменный формат сметы для загрузки в сметные системы —
//! и парсер, который его строит. Два пути:
//!   • правила: детерминированное отображение иерархии (semantics.rs) в DTO;
//!   • LLM: распознанная структура (включая сырые строки таблицы) уходит
//!     в OpenAI-совместимый чат-API с жёсткой схемой ответа; результат
//!     проверяется и при любой ошибке откатывается на правила.
//! Включается переменными окружения: RUSTOCR_LLM_URL (например
//! https://api.openai.com/v1 или http://localhost:11434/v1),
//! RUSTOCR_LLM_KEY, RUSTOCR_LLM_MODEL (по умолчанию gpt-4o-mini),
//! RUSTOCR_LLM_TIMEOUT_SECS (по умолчанию 120).

use crate::estimate::model::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ---------- DTO ----------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PazlEstimate {
    /// версия формата
    pub format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub number: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub customer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contractor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
    #[serde(default = "default_currency")]
    pub currency: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vat_rate: Option<f64>,
    pub sections: Vec<PazlSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vat: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_with_vat: Option<f64>,
    /// чем заполнено: «rules» или «llm»
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parser: Option<String>,
}

fn default_currency() -> String {
    "RUB".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PazlSection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub title: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub positions: Vec<PazlPosition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PazlPosition {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    /// шифр расценки (ГЭСН/ФЕР/ТЕР…)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cipher: Option<String>,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// количество
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantity: Option<f64>,
    /// цена за единицу
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit_price: Option<f64>,
    /// стоимость труда в составе позиции
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labor: Option<f64>,
    /// стоимость эксплуатации машин
    #[serde(skip_serializing_if = "Option::is_none")]
    pub machines: Option<f64>,
    /// стоимость материалов
    #[serde(skip_serializing_if = "Option::is_none")]
    pub materials: Option<f64>,
    /// всего по позиции
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<PazlResource>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PazlResource {
    /// labor | machine | machine_labor | material | transport | overhead | profit | other
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit_price: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
}

// ---------- правила: иерархия → DTO ----------

pub fn to_pazl(doc: &EstimateDocument) -> PazlEstimate {
    PazlEstimate {
        format: "pazl.estimate.v1".to_string(),
        number: doc.meta.number.clone(),
        title: doc.meta.title.clone(),
        object: doc.meta.object.clone(),
        customer: doc.meta.customer.clone(),
        contractor: doc.meta.contractor.clone(),
        date: doc.meta.date.clone(),
        currency: default_currency(),
        vat_rate: doc.totals.vat_rate,
        sections: doc
            .sections
            .iter()
            .map(|s| {
                let mut ps = to_pazl_section(s);
                let sub_positions: Vec<PazlPosition> =
                    s.subsections.iter().flat_map(to_pazl_section_positions).collect();
                ps.positions.extend(sub_positions);
                ps
            })
            .collect(),
        total: doc.totals.total.as_ref().map(|n| n.value),
        vat: doc.totals.vat.as_ref().map(|n| n.value),
        total_with_vat: doc.totals.total_with_vat.as_ref().map(|n| n.value),
        parser: Some("rules".to_string()),
    }
}

fn to_pazl_section(s: &Section) -> PazlSection {
    PazlSection {
        code: s.number.clone(),
        title: s.title.clone(),
        positions: to_pazl_section_positions(s),
        total: s.total.as_ref().map(|n| n.value),
    }
}

fn to_pazl_section_positions(s: &Section) -> Vec<PazlPosition> {
    s.positions
        .iter()
        .map(|p| {
            let mut labor = 0.0;
            let mut machines = 0.0;
            let mut materials = 0.0;
            for r in &p.resources {
                let v = r.total.as_ref().map(|n| n.value).unwrap_or(0.0);
                match r.kind {
                    ResourceKind::Labor => labor += v,
                    ResourceKind::Machine | ResourceKind::MachineLabor => machines += v,
                    ResourceKind::Material => materials += v,
                    _ => {}
                }
            }
            PazlPosition {
                line: p.line,
                cipher: p.cipher.clone(),
                name: p.name.clone(),
                unit: p.unit.clone(),
                quantity: p.quantity.as_ref().map(|n| n.value),
                unit_price: p.price.as_ref().map(|n| n.value),
                labor: if labor > 0.0 { Some(labor) } else { None },
                machines: if machines > 0.0 { Some(machines) } else { None },
                materials: if materials > 0.0 { Some(materials) } else { None },
                total: p.total.as_ref().map(|n| n.value),
                resources: p
                    .resources
                    .iter()
                    .map(|r| PazlResource {
                        kind: r.kind.as_str().to_string(),
                        code: r.code.clone(),
                        name: r.name.clone(),
                        unit: r.unit.clone(),
                        quantity: r.quantity.as_ref().map(|n| n.value),
                        unit_price: r.price.as_ref().map(|n| n.value),
                        total: r.total.as_ref().map(|n| n.value),
                    })
                    .collect(),
            }
        })
        .collect()
}

// ---------- LLM-парсер ----------

pub struct LlmConfig {
    pub url: String,
    pub key: Option<String>,
    pub model: String,
    pub timeout: Duration,
}

impl LlmConfig {
    /// None — LLM не настроен (RUSTOCR_LLM_URL пуст).
    pub fn from_env() -> Option<Self> {
        let url = std::env::var("RUSTOCR_LLM_URL").unwrap_or_default();
        if url.trim().is_empty() {
            return None;
        }
        Some(Self {
            url: url.trim().trim_end_matches('/').to_string(),
            key: std::env::var("RUSTOCR_LLM_KEY").ok().filter(|k| !k.is_empty()),
            model: std::env::var("RUSTOCR_LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string()),
            timeout: Duration::from_secs(
                std::env::var("RUSTOCR_LLM_TIMEOUT_SECS")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(120)
                    .clamp(5, 600),
            ),
        })
    }
}

/// Итоговый парсер: LLM если настроен и справился, иначе правила.
/// Возвращает (DTO, кем заполнено, ошибка LLM если была).
pub fn parse_pazl(doc: &EstimateDocument) -> (PazlEstimate, &'static str, Option<String>) {
    let rules = to_pazl(doc);
    let Some(cfg) = LlmConfig::from_env() else {
        return (rules, "rules", None);
    };
    match llm_parse(&cfg, doc) {
        Ok(mut p) => {
            p.parser = Some("llm".to_string());
            // поля, которые LLM не заполнил, добираем из правил
            if p.number.is_none() {
                p.number = rules.number.clone();
            }
            if p.title.is_none() {
                p.title = rules.title.clone();
            }
            if p.total.is_none() {
                p.total = rules.total;
            }
            if p.vat.is_none() {
                p.vat = rules.vat;
            }
            if p.total_with_vat.is_none() {
                p.total_with_vat = rules.total_with_vat;
            }
            (p, "llm", None)
        }
        Err(e) => (rules, "rules", Some(e.to_string())),
    }
}

fn llm_parse(cfg: &LlmConfig, doc: &EstimateDocument) -> anyhow::Result<PazlEstimate> {
    let user_payload = llm_user_payload(doc);
    let system = SYSTEM_PROMPT.to_string();
    let mut req = ureq::post(&format!("{}/chat/completions", cfg.url))
        .timeout(cfg.timeout)
        .set("Content-Type", "application/json");
    if let Some(key) = &cfg.key {
        req = req.set("Authorization", &format!("Bearer {}", key));
    }
    let body = serde_json::json!({
        "model": cfg.model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_payload}
        ]
    });
    let resp: serde_json::Value = req.send_json(body)?.into_json()?;
    let content = resp
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("в ответе LLM нет choices[0].message.content"))?;
    let json = strip_code_fence(content);
    let parsed: PazlEstimate = serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("ответ LLM не соответствует схеме DTO: {}", e))?;
    // sanity: что-то meaningful
    let n_pos: usize = parsed.sections.iter().map(|s| s.positions.len()).sum();
    if n_pos == 0 && parsed.total.is_none() {
        anyhow::bail!("LLM вернул пустую смету (0 позиций, без итога)");
    }
    Ok(parsed)
}

/// Компактная выгрузка распознанной структуры для LLM: иерархия + сырые
/// строки таблицы (именно по ним LLM уточняет неоднозначные поля).
fn llm_user_payload(doc: &EstimateDocument) -> String {
    let mut v = serde_json::to_value(doc).unwrap_or(serde_json::json!({}));
    // сырые строки уже внутри позиций (raw); unparsed_lines тоже полезны
    let _ = &mut v;
    let s = serde_json::to_string(&v).unwrap_or_default();
    // страховка от гигантских промптов
    const LIMIT: usize = 400_000;
    if s.len() > LIMIT {
        let cut: String = s.chars().take(LIMIT).collect();
        return format!("{} …(обрезано)", cut);
    }
    s
}

/// Убрать ```json … ``` вокруг ответа модели.
fn strip_code_fence(content: &str) -> String {
    let t = content.trim();
    let t = t.strip_prefix("```json").or_else(|| t.strip_prefix("```")).unwrap_or(t);
    let t = t.strip_suffix("```").unwrap_or(t);
    // модель может добавить пояснение до/после JSON — берём внешний {...}
    let start = t.find('{');
    let end = t.rfind('}');
    match (start, end) {
        (Some(a), Some(b)) if b > a => t[a..=b].to_string(),
        _ => t.to_string(),
    }
}

const SYSTEM_PROMPT: &str = r#"Ты — парсер строительных смет. Тебе дают JSON с результатом OCR-распознавания сметы: иерархию «раздел → позиция → ресурсы», итоги, неразобранные строки, а внутри позиций — исходные строки таблицы (поле raw).

Верни СТРОГО JSON (без пояснений, без markdown) — DTO сметы по схеме:
{
  "format": "pazl.estimate.v1",
  "number": string|null,        // номер сметы
  "title": string|null,
  "object": string|null,        // объект
  "customer": string|null,      // заказчик
  "contractor": string|null,    // подрядчик
  "date": string|null,
  "currency": "RUB",
  "vat_rate": number|null,      // 0.2 значит 20%
  "sections": [
    {
      "code": string|null,      // номер раздела
      "title": string,
      "positions": [
        {
          "line": integer|null, // № п/п
          "cipher": string|null,// шифр расценки ГЭСН/ФЕР/ТЕР
          "name": string,       // наименование работы/материала
          "unit": string|null,  // ед. изм.: м³, м², т, шт, чел.-ч…
          "quantity": number|null,
          "unit_price": number|null,
          "labor": number|null,       // стоимость труда в позиции
          "machines": number|null,    // эксплуатация машин
          "materials": number|null,   // материалы
          "total": number|null,       // всего по позиции
          "resources": [
            {"kind":"labor|machine|machine_labor|material|transport|overhead|profit|other",
             "code":string|null, "name":string, "unit":string|null,
             "quantity":number|null, "unit_price":number|null, "total":number|null}
          ]
        }
      ],
      "total": number|null
    }
  ],
  "total": number|null,         // итого по смете без НДС
  "vat": number|null,           // сумма НДС
  "total_with_vat": number|null
}

Правила:
- Используй ТОЛЬКО данные из входного JSON; ничего не выдумывай, числа не пересчитывай.
- Поле raw позиции — исходные строки таблицы: если поле распознано плохо (None или подозрительно), восстанови его из raw (например «м3» → «м³», перепутанные колонки).
- Неразобранные строки (unparsed_lines), похожие на позиции или ресурсы, добавь как позиции/ресурсы.
- Числа пиши как number: «1 234,56» → 1234.56.
- Разделы и позиции сохраняй в исходном порядке. Не добавляй пустые разделы."#;

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_doc() -> EstimateDocument {
        let pos = Position {
            line: Some(1),
            cipher: Some("ГЭСН 01-01-013-01".to_string()),
            name: "Разработка грунта".to_string(),
            unit: Some("м³".to_string()),
            quantity: Some(Num::new(123.45, "123,45".to_string())),
            price: Some(Num::new(50.0, "50,00".to_string())),
            total: Some(Num::new(6172.5, "6 172,50".to_string())),
            resources: vec![
                Resource {
                    kind: ResourceKind::Labor,
                    code: None,
                    name: "Затраты труда рабочих".to_string(),
                    unit: Some("чел.-ч".to_string()),
                    quantity: Some(Num::new(12.5, "12,5".to_string())),
                    price: None,
                    total: Some(Num::new(100.0, "100".to_string())),
                },
                Resource {
                    kind: ResourceKind::Material,
                    code: Some("1.1-1-2345".to_string()),
                    name: "Песок".to_string(),
                    unit: Some("м³".to_string()),
                    quantity: Some(Num::new(10.0, "10".to_string())),
                    price: Some(Num::new(30.0, "30".to_string())),
                    total: Some(Num::new(300.0, "300".to_string())),
                },
            ],
            pages: vec![1],
            raw: vec!["1 ГЭСН 01-01-013-01 Разработка грунта м3 123,45 50,00 6 172,50".to_string()],
        };
        EstimateDocument {
            meta: Meta {
                title: Some("Локальная смета".to_string()),
                number: Some("02-01-01".to_string()),
                object: Some("Жилой дом".to_string()),
                customer: Some("ООО Ромашка".to_string()),
                contractor: None,
                date: None,
                notes: Vec::new(),
            },
            sections: vec![Section {
                number: Some("1".to_string()),
                title: "Земляные работы".to_string(),
                subsections: Vec::new(),
                positions: vec![pos],
                resources: Vec::new(),
                total: Some(Num::new(6172.5, "6 172,50".to_string())),
                pages: vec![1],
            }],
            totals: Totals {
                total: Some(Num::new(6172.5, "6 172,50".to_string())),
                vat_rate: Some(0.2),
                vat: Some(Num::new(1234.5, "1 234,50".to_string())),
                total_with_vat: Some(Num::new(7407.0, "7 407,00".to_string())),
            },
            validation: ValidationReport::default(),
            unparsed_lines: Vec::new(),
            auto_repairs: 0,
            pages: 1,
        }
    }

    #[test]
    fn rules_mapping() {
        let p = to_pazl(&sample_doc());
        assert_eq!(p.format, "pazl.estimate.v1");
        assert_eq!(p.number.as_deref(), Some("02-01-01"));
        assert_eq!(p.sections.len(), 1);
        let s = &p.sections[0];
        assert_eq!(s.code.as_deref(), Some("1"));
        assert_eq!(s.positions.len(), 1);
        let pos = &s.positions[0];
        assert_eq!(pos.cipher.as_deref(), Some("ГЭСН 01-01-013-01"));
        assert_eq!(pos.quantity, Some(123.45));
        assert_eq!(pos.unit_price, Some(50.0));
        assert_eq!(pos.total, Some(6172.5));
        assert_eq!(pos.labor, Some(100.0));
        assert_eq!(pos.materials, Some(300.0));
        assert_eq!(pos.machines, None);
        assert_eq!(pos.resources.len(), 2);
        assert_eq!(pos.resources[0].kind, "labor");
        assert_eq!(pos.resources[1].code.as_deref(), Some("1.1-1-2345"));
        assert_eq!(p.total, Some(6172.5));
        assert_eq!(p.vat, Some(1234.5));
        assert_eq!(p.total_with_vat, Some(7407.0));
        assert_eq!(p.parser.as_deref(), Some("rules"));
        // сериализация и обратный разбор (схема стабильна)
        let j = serde_json::to_string(&p).unwrap();
        let back: PazlEstimate = serde_json::from_str(&j).unwrap();
        assert_eq!(back.sections[0].positions[0].name, "Разработка грунта");
    }

    #[test]
    fn fence_stripping() {
        assert_eq!(
            strip_code_fence("```json\n{\"a\":1}\n```"),
            "{\"a\":1}"
        );
        assert_eq!(strip_code_fence("Вот ответ:\n{\"a\":1}\nНадеюсь, верно"), "{\"a\":1}");
        assert_eq!(strip_code_fence("{\"a\":1}"), "{\"a\":1}");
    }

    #[test]
    fn fallback_without_env() {
        std::env::remove_var("RUSTOCR_LLM_URL");
        let (p, who, err) = parse_pazl(&sample_doc());
        assert_eq!(who, "rules");
        assert!(err.is_none());
        assert_eq!(p.parser.as_deref(), Some("rules"));
    }
}
