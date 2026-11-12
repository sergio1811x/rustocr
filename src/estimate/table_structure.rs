//! П.4: TableStructureRec wired — SLANeXt-wired ONNX (PP-Structure, Apache-2.0).
//!
//! «Второй голос» для геометрического восстановления таблиц: SLANeXt по
//! картинке страницы предсказывает последовательность HTML-токенов
//! (`<tbody><tr><td></td>…</tr>…`) с боксом на каждую ячейку (в т.ч.
//! объединённые — colspan/rowspan задаются токенами). Геометрия по белым
//! просветам остаётся основным путём (нулевая стоимость, проверена на
//! корпусе); нейросеть подключается как «спасение» для страниц, где
//! просветы заблокированы (двухуровневые шапки, плотные коммерческие
//! бланки) — и даёт границы колонок туда же, в `layout_page`.
//!
//! Декод сверен с эталоном PaddleOCR v2.7 (ppocr/postprocess/
//! table_postprocess.py + label_ops.py):
//!   * словарь 50 классов = [sos] + merge(48 токенов: убрать `<td>`,
//!     добавить `<td></td>`) + [eos]; id0=sos пропускается, eos обрезает;
//!   * бокс ячейки — сырые [0..1] (БЕЗ sigmoid) на ИНДЕКСЕ шага с td-токеном,
//!     умноженные на ширину/высоту ОРИГИНАЛЬНОЙ картинки;
//!   * препроцессинг: resize по длинной стороне ≤512 (bilinear/triangle),
//!     BGR, нормализация ImageNet (mean/std), паддинг нулями до 512×512,
//!     NCHW.
//!
//! Тяжёлый onnx (367 МБ) лежит в models/table/ вне git; при его отсутствии
//! модуль просто не активируется (None), конвейер работает по-старому.

use crate::geom::resize_tri;
use crate::pipeline::OcrEngine;
use image::RgbImage;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;

const IN_SIZE: u32 = 512;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406]; // RGB; подаём BGR — порядок ниже
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Ячейка нейросетевой таблицы (координаты — в пикселях страницы).
#[derive(Clone, Debug)]
pub struct NeuralCell {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub rowspan: usize,
    pub colspan: usize,
}

/// Таблица страницы по версии SLANeXt: строки ячеек сверху вниз.
#[derive(Clone, Debug)]
pub struct NeuralTable {
    pub rows: Vec<Vec<NeuralCell>>,
    /// средняя уверенность токенов структуры
    pub conf: f32,
}

impl NeuralTable {
    /// Число колонок сетки (по самой широкой строке).
    pub fn n_cols(&self) -> usize {
        self.rows
            .iter()
            .map(|r| r.iter().map(|c| c.colspan).sum::<usize>())
            .max()
            .unwrap_or(0)
    }

    pub fn n_cells(&self) -> usize {
        self.rows.iter().map(Vec::len).sum()
    }

    /// Вертикальные границы колонок: середины межклеточных зазоров по всем
    /// строкам, кластеризованные. Объединённая ячейка не даёт «своей»
    /// границы, но соседние строки с обычными ячейками — дают.
    pub fn col_boundaries(&self) -> Vec<f32> {
        let mut cand: Vec<f32> = Vec::new();
        for row in &self.rows {
            for w in row.windows(2) {
                cand.push((w[0].x1 + w[1].x0) / 2.0);
            }
        }
        cand.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let tol = 14.0f32;
        let mut out: Vec<f32> = Vec::new();
        let mut acc: Vec<f32> = Vec::new();
        for e in cand {
            match acc.last() {
                Some(&last) if e - last <= tol => acc.push(e),
                _ => {
                    if !acc.is_empty() {
                        out.push(acc.iter().sum::<f32>() / acc.len() as f32);
                    }
                    acc.clear();
                    acc.push(e);
                }
            }
        }
        if !acc.is_empty() {
            out.push(acc.iter().sum::<f32>() / acc.len() as f32);
        }
        // ограничитель здравого смысла: больше 18 границ в смете не бывает
        out.into_iter().take(18).collect()
    }
}

/// Режим нейро-структуры (env RUSTOCR_TABLE_STRUCT):
///   * по умолчанию (пусто) — ВЫКЛЮЧЕНО: на плотных коммерческих бланках
///     без сетки сеть пере-дробляет колонки (karkas_kd1: 266→238 поз,
///     dominanta: 148→138, furanfleks: итог документа рассыпался) —
///     геометрия по просветам на корпусе стабильнее;
///   * 1/always — границы нейросети предпочтительнее геометрии;
///   * smart — сеть только на «слепых» страницах (<2 границ) и при
///     строго большем числе колонок.
#[derive(PartialEq, Clone, Copy)]
enum Mode {
    Off,
    Smart,
    Always,
}

fn mode() -> Mode {
    match std::env::var("RUSTOCR_TABLE_STRUCT").as_deref() {
        Ok("0") | Ok("off") | Ok("false") => Mode::Off,
        Ok("1") | Ok("always") | Ok("true") => Mode::Always,
        Ok("smart") => Mode::Smart,
        _ => Mode::Off,
    }
}

fn debug_enabled() -> bool {
    std::env::var("RUSTOCR_TABLE_STRUCT_DEBUG").is_ok()
}

/// Движок структуры таблиц (одна ONNX-сессия на процесс).
pub struct TableStructEngine {
    session: Mutex<Session>,
    /// 48 токенов после merge: `<td>` убран, `<td></td>` добавлен в конец
    dict: Vec<String>,
    in_name: String,
    /// имя выхода боксов ([1, T, 8]) и структуры ([1, T, 50])
    out0: String,
    out1: String,
}

impl TableStructEngine {
    /// Загрузить модель. `None` — файла нет или выключено env-ом.
    pub fn load(env: &Environment, models_dir: &Path) -> Option<Self> {
        if mode() == Mode::Off {
            return None;
        }
        let onnx = models_dir.join("table/slanext_wired.onnx");
        let dict_path = models_dir.join("table/slanext_wired_struct_dict.txt");
        if !onnx.exists() || !dict_path.exists() {
            return None; // тяжёлая модель вне git — офлайн-конвейер работает без неё
        }
        let dict = load_dict(&dict_path)?;
        let session = Session::builder(env)
            .and_then(|mut b| b.commit_from_file(&onnx))
            .map_err(|e| {
                eprintln!("[table-struct] не загрузился {}: {}", onnx.display(), e);
                e
            })
            .ok()?;
        let in_name = session.inputs()[0].name().to_string();
        let out0 = session.outputs()[0].name().to_string();
        let out1 = session.outputs()[1].name().to_string();
        if debug_enabled() {
            eprintln!(
                "[table-struct] SLANeXt-wired загружен: {} токенов, вход «{}», выходы «{}»/«{}»",
                dict.len(),
                in_name,
                out0,
                out1
            );
        }
        Some(Self {
            session: Mutex::new(session),
            dict,
            in_name,
            out0,
            out1,
        })
    }

    /// Распознать структуру таблиц на странице. `None` — сеть не нашла
    /// таблицу или неуверенна.
    pub fn recognize(&self, img: &RgbImage) -> Option<NeuralTable> {
        let (data, shape) = preprocess(img);
        let input = Tensor::from_array((shape, data)).ok()?;
        let mut session = self.session.lock().ok()?;
        let outputs = session
            .run(ort::inputs![self.in_name.as_str() => input])
            .ok()?;
        let (oshape0, raw0) = outputs[self.out0.as_str()]
            .try_extract_tensor::<f32>()
            .ok()?;
        let (oshape1, raw1) = outputs[self.out1.as_str()]
            .try_extract_tensor::<f32>()
            .ok()?;
        // [1, T, K]: K=8 — боксы, K=len(dict)+2 — структура
        let (t0, k0) = (oshape0[1] as usize, oshape0[2] as usize);
        let (t1, k1) = (oshape1[1] as usize, oshape1[2] as usize);
        let (boxes, probs, t, kbox, c_struct) = if k0 == 8 && k1 != 8 {
            (raw0, raw1, t1, 8usize, k1)
        } else if k1 == 8 && k0 != 8 {
            (raw1, raw0, t0, 8usize, k0)
        } else {
            return None; // неожиданный формат выходов
        };
        if t == 0 || boxes.len() != t * 8 || probs.len() != t * c_struct {
            return None;
        }
        decode(
            &self.dict,
            boxes,
            probs,
            t,
            kbox,
            c_struct,
            img.width(),
            img.height(),
        )
    }
}

/// Словарь 48 токенов: merge по TableLabelDecode (убрать `<td>`,
/// добавить `<td></td>`), спец-токены sos/eos НЕ входят в файл.
fn load_dict(path: &Path) -> Option<Vec<String>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut toks: Vec<String> = text
        .lines()
        .map(|l| l.trim_end_matches(['\r', '\n']).to_string())
        .filter(|l| !l.trim().is_empty())
        .collect();
    toks.retain(|t| t != "<td>");
    if !toks.contains(&"<td></td>".to_string()) {
        toks.push("<td></td>".to_string());
    }
    if toks.len() + 2 != 50 {
        eprintln!(
            "[table-struct] словарь странного размера: {} (ожидалось 48)",
            toks.len()
        );
        return None;
    }
    Some(toks)
}

/// Препроцессинг: длинная сторона ≤512 (triangle ≈ cv2.INTER_LINEAR),
/// BGR, ImageNet-нормализация, паддинг нулями справа/снизу, NCHW.
fn preprocess(img: &RgbImage) -> (Vec<f32>, [usize; 4]) {
    let (w, h) = img.dimensions();
    let ratio = IN_SIZE as f32 / w.max(h).max(1) as f32;
    let nw = ((w as f32 * ratio) as u32).clamp(1, IN_SIZE);
    let nh = ((h as f32 * ratio) as u32).clamp(1, IN_SIZE);
    let small = resize_tri(img, nw, nh);
    let mut data = vec![0f32; (IN_SIZE * IN_SIZE * 3) as usize];
    let plane = (IN_SIZE * IN_SIZE) as usize;
    for y in 0..nh {
        for x in 0..nw {
            let p = small.get_pixel(x, y);
            // BGR: канал 0 — синий
            let (b, g, r) = (p[2] as f32, p[1] as f32, p[0] as f32);
            let i = (y as usize) * IN_SIZE as usize + x as usize;
            data[i] = (b / 255.0 - MEAN[2]) / STD[2];
            data[plane + i] = (g / 255.0 - MEAN[1]) / STD[1];
            data[2 * plane + i] = (r / 255.0 - MEAN[0]) / STD[0];
        }
    }
    (data, [1, 3, IN_SIZE as usize, IN_SIZE as usize])
}

/// Декод выхода: поток токенов (argmax по классам) + боксы на td-токенах.
/// Выделено отдельно, чтобы тестировать без ONNX-модели.
fn decode(
    dict: &[String],
    boxes: &[f32],
    probs: &[f32],
    t: usize,
    kbox: usize,
    c_struct: usize,
    img_w: u32,
    img_h: u32,
) -> Option<NeuralTable> {
    if dict.len() + 2 != c_struct || kbox != 8 {
        return None;
    }
    let eos = dict.len() as i64 + 1;
    let mut cells: Vec<Vec<NeuralCell>> = Vec::new();
    let mut row: Vec<NeuralCell> = Vec::new();
    let mut cur: Option<NeuralCell> = None;
    let mut confs: Vec<f32> = Vec::new();
    let mut in_cell_attr = false; // между `<td` и `>` идут только colspan/rowspan

    let close_cell = |row: &mut Vec<NeuralCell>, cur: &mut Option<NeuralCell>| {
        if let Some(c) = cur.take() {
            row.push(c);
        }
    };

    for step in 0..t {
        let base = step * c_struct;
        let mut id = 0i64;
        let mut best = f32::NEG_INFINITY;
        for (k, v) in probs[base..base + c_struct].iter().enumerate() {
            if *v > best {
                best = *v;
                id = k as i64;
            }
        }
        if step > 0 && id == eos {
            break;
        }
        if id <= 0 || id as usize > dict.len() {
            continue; // sos и мусорные id
        }
        let tok = &dict[id as usize - 1];
        let is_td = tok == "<td></td>" || tok == "<td" || tok == "<td>";
        if tok == "</tr>" {
            close_cell(&mut row, &mut cur);
            in_cell_attr = false;
            if !row.is_empty() {
                cells.push(std::mem::take(&mut row));
            }
        } else if tok == ">" {
            in_cell_attr = false;
        } else if is_td {
            close_cell(&mut row, &mut cur);
            in_cell_attr = tok == "<td";
            let b = &boxes[step * 8..step * 8 + 8];
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for c in 0..4 {
                let bx = b[c * 2] * img_w as f32;
                let by = b[c * 2 + 1] * img_h as f32;
                x0 = x0.min(bx);
                y0 = y0.min(by);
                x1 = x1.max(bx);
                y1 = y1.max(by);
            }
            cur = Some(NeuralCell {
                x0,
                y0,
                x1,
                y1,
                rowspan: 1,
                colspan: 1,
            });
            confs.push(best);
        } else if in_cell_attr {
            if let Some(v) = tok
                .strip_prefix(' ')
                .and_then(|s| s.strip_prefix("colspan=\""))
                .and_then(|s| s.strip_suffix('"'))
                .and_then(|s| s.parse::<usize>().ok())
            {
                if let Some(c) = cur.as_mut() {
                    c.colspan = c.colspan.max(v);
                }
            } else if let Some(v) = tok
                .strip_prefix(' ')
                .and_then(|s| s.strip_prefix("rowspan=\""))
                .and_then(|s| s.strip_suffix('"'))
                .and_then(|s| s.parse::<usize>().ok())
            {
                if let Some(c) = cur.as_mut() {
                    c.rowspan = c.rowspan.max(v);
                }
            }
        }
    }
    close_cell(&mut row, &mut cur);
    if !row.is_empty() {
        cells.push(row);
    }

    let table = NeuralTable {
        rows: cells,
        conf: if confs.is_empty() {
            0.0
        } else {
            confs.iter().sum::<f32>() / confs.len() as f32
        },
    };
    if table.n_cells() < 8 || table.n_cols() < 2 || table.conf < 0.9 {
        return None;
    }
    Some(table)
}

/// «Спасение» границ: геометрия по просветам — основной голос; SLANeXt
/// подключается, когда геометрия слабая, и принимается при строго большем
/// числе колонок. Используется и из `own_edges` (семейства страниц), и из
/// `analyze_page` (одиночная страница).
pub fn rescue_boundaries(
    engine: Option<&OcrEngine>,
    img: Option<&RgbImage>,
    geo: &[f32],
) -> Vec<f32> {
    let m = mode();
    if m == Mode::Off || img.is_none() {
        return geo.to_vec();
    }
    let Some(e) = engine.and_then(OcrEngine::table_struct) else {
        return geo.to_vec();
    };
    let geo_cols = geo.len() + 1;
    let run = match m {
        Mode::Always => true,
        // строгое спасение: страница СЛЕПАЯ — собственных границ нет
        // вовсе (<2); сейчас она заимствует границы семейства, нейросеть
        // даёт ей свои. Здоровая геометрия не перевбирается: на плотных
        // коммерческих бланках сеть склонна к пере-дроблению колонок
        Mode::Smart => geo.len() < 2,
        Mode::Off => false,
    };
    if !run {
        return geo.to_vec();
    }
    let Some(table) = e.recognize(img.unwrap()) else {
        if debug_enabled() {
            eprintln!(
                "[table-struct] сеть не дала таблицу; геометрия остаётся ({} границ)",
                geo.len()
            );
        }
        return geo.to_vec();
    };
    let neural = table.col_boundaries();
    let neural_cols = neural.len() + 1;
    let accept = match m {
        Mode::Always => neural_cols >= 2,
        _ => neural_cols > geo_cols && neural_cols >= 4,
    };
    if debug_enabled() {
        eprintln!(
            "[table-struct] геометрия: {} кол (границы {:?}); сеть: {} кол, conf {:.3} → {}",
            geo_cols,
            geo.iter().map(|b| b.round()).collect::<Vec<_>>(),
            neural_cols,
            table.conf,
            if accept { "ПРИНЯТО" } else { "отклонено" }
        );
    }
    if accept {
        neural
    } else {
        geo.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dict() -> Vec<String> {
        // порядок реального файла models/table/slanext_wired_struct_dict.txt
        let mut toks: Vec<String> = vec![
            "<thead>", "</thead>", "<tbody>", "</tbody>", "<tr>", "</tr>", "<td>",
            "<td", ">", "</td>",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        for n in 2..=20 {
            toks.push(format!(" colspan=\"{}\"", n));
        }
        for n in 2..=20 {
            toks.push(format!(" rowspan=\"{}\"", n));
        }
        // merge: убрать <td>, добавить <td></td> в конец — 48 токенов
        toks.retain(|t| t != "<td>");
        toks.push("<td></td>".to_string());
        toks
    }

    /// Собрать вход декодера: probs с argmax по нужным id; боксы кладутся
    /// НА ПОЗИЦИИ td-токенов (id7=`<td`, id48=`<td></td>`), как в реальном
    /// выходе сети, и нормализуются в [0..1].
    fn make_inputs(ids: &[i64], cells: &[[u8; 8]]) -> (Vec<f32>, Vec<f32>, usize, usize) {
        let c = 50usize;
        let t = ids.len();
        let mut probs = vec![0f32; t * c];
        for (i, &id) in ids.iter().enumerate() {
            probs[i * c + id as usize] = 1.0;
        }
        let mut boxes = vec![0f32; t * 8];
        let mut ci = 0;
        for (i, &id) in ids.iter().enumerate() {
            if id == 7 || id == 48 {
                if ci < cells.len() {
                    for (j, v) in cells[ci].iter().enumerate() {
                        boxes[i * 8 + j] = *v as f32 / 255.0;
                    }
                    ci += 1;
                }
            }
        }
        (boxes, probs, t, 8)
    }

    /// 3 строки × 3 колонки на странице 600×400: сетка и границы.
    #[test]
    fn decode_grid_and_boundaries() {
        // ids: <tbody>(3) <tr>(5) <td></td>×3 </tr>(6) ... </tbody>(4)
        let td = 48i64; // <td></td> в merge-словаре — последний, id 48
        let ids = vec![
            0i64, 3, 5, td, td, td, 6, 5, td, td, td, 6, 5, td, td, td, 6, 4, 49,
        ];
        // колонки x: 0..100, 100..200, 200..300; строки y: 10..50, 60..100, 110..150
        let mkcell = |x0: u8, y0: u8, x1: u8, y1: u8| [x0, y0, x1, y0, x1, y1, x0, y1];
        let mut cells: Vec<[u8; 8]> = Vec::new();
        for &y in [10u8, 60, 110].iter() {
            for &(x0, x1) in [(0u8, 100u8), (100, 200), (200, 255)].iter() {
                cells.push(mkcell(x0, y, x1, y + 40));
            }
        }
        let (boxes, probs, t, kbox) = make_inputs(&ids, &cells);
        let table = decode(&dict(), &boxes, &probs, t, kbox, 50, 600, 400)
            .expect("таблица не декодирована");
        assert_eq!(table.rows.len(), 3);
        assert_eq!(table.n_cols(), 3);
        assert_eq!(table.rows[0][2].x1, 600.0); // 255/255 * 600
        let b = table.col_boundaries();
        assert_eq!(b.len(), 2, "границы: {:?}", b);
        // 100/255*600 ≈ 235; 200/255*600 ≈ 470
        assert!((b[0] - 235.3).abs() < 6.0, "первая граница: {}", b[0]);
        assert!((b[1] - 470.5).abs() < 6.0, "вторая граница: {}", b[1]);
        assert!(table.conf > 0.99);
    }

    /// Ячейка с colspan=2 (`<td` + ` colspan="2"` + `>`), затем `</td>`;
    /// вторая-третья-четвёртая строки — обычные (10 ячеек ≥ порога 8).
    #[test]
    fn decode_colspan_cell() {
        // id: <tbody>=3 <tr>=5 <td=7 colspan"2"=10 >=8 </td>=9 </tr>=6 <td></td>=48
        let ids = vec![
            0i64, 3, 5, 7, 10, 8, 9, 6, 5, 48, 48, 48, 6, 5, 48, 48, 48, 6, 5, 48, 48,
            48, 6, 4, 49,
        ];
        let mkcell = |x0: u8, y0: u8, x1: u8, y1: u8| [x0, y0, x1, y0, x1, y1, x0, y1];
        let mut cells: Vec<[u8; 8]> = vec![
            mkcell(0, 10, 127, 50), // объединённая — накрывает две колонки
        ];
        for &y in [60u8, 110, 160].iter() {
            for &(x0, x1) in [(0u8, 127u8), (127, 200), (200, 255)].iter() {
                cells.push(mkcell(x0, y, x1, y + 40));
            }
        }
        let (boxes, probs, t, kbox) = make_inputs(&ids, &cells);
        let table = decode(&dict(), &boxes, &probs, t, kbox, 50, 400, 300)
            .expect("таблица не декодирована");
        assert_eq!(table.rows.len(), 4);
        assert_eq!(table.rows[0].len(), 1);
        assert_eq!(table.rows[0][0].colspan, 2, "colspan не прочитан");
        assert_eq!(table.rows[1].len(), 3);
        assert_eq!(table.rows[3].len(), 3);
        assert_eq!(table.n_cols(), 3, "сетка: 1×2 + 3×3");
        let b = table.col_boundaries();
        assert_eq!(b.len(), 2, "границы: {:?}", b);
        // 127/255*400 ≈ 199; 200/255*400 ≈ 314
        assert!((b[0] - 199.2).abs() < 6.0, "первая: {}", b[0]);
        assert!((b[1] - 313.7).abs() < 6.0, "вторая: {}", b[1]);
    }

    /// Мусорный поток (одинокие токены) — None.
    #[test]
    fn decode_garbage_is_none() {
        let ids = vec![0i64, 5, 6, 5, 6, 49];
        let (boxes, probs, t, kbox) = make_inputs(&ids, &[]);
        assert!(decode(&dict(), &boxes, &probs, t, kbox, 50, 600, 400).is_none());
    }

    /// Режимы читаются из env; по умолчанию (пусто) — выключено.
    #[test]
    fn mode_default_off() {
        std::env::remove_var("RUSTOCR_TABLE_STRUCT");
        assert!(matches!(mode(), Mode::Off));
    }
}
