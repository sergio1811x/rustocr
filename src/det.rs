//! Детектор текстовых блоков (DBNet из PP-OCRv5).

use crate::geom::{
    convex_hull, dist, expand_rect, min_area_rect, order_quad, rect_area, rect_peri, resize_tri,
    Pt,
};
use image::RgbImage;

pub const DET_THRESH: f32 = 0.3; // порог бинаризации карты вероятностей
pub const BOX_THRESH: f32 = 0.6; // минимальная средняя уверенность блока
pub const UNCLIP_RATIO: f32 = 1.5; // насколько расширить блок вокруг текста

pub struct DetInput {
    pub data: Vec<f32>,
    pub shape: [usize; 4],
    pub ratio: f32, // масштаб «оригинал → вход сети»
}

/// Подготовка картинки: длинная сторона ≤ limit, паддинг до кратности 32,
/// нормализация ImageNet в порядке BGR, раскладка NCHW.
pub fn preprocess(img: &RgbImage, limit: u32) -> DetInput {
    let (w, h) = img.dimensions();
    let long = w.max(h);
    let ratio = if long > limit {
        limit as f32 / long as f32
    } else {
        1.0
    };
    let rw = (((w as f32) * ratio).round() as u32).max(1);
    let rh = (((h as f32) * ratio).round() as u32).max(1);
    let resized = resize_tri(img, rw, rh);
    let pw = (rw + 31) / 32 * 32;
    let ph = (rh + 31) / 32 * 32;
    let c = (ph as usize) * (pw as usize);
    let mut data = vec![0.0f32; 3 * c];
    for y in 0..rh as usize {
        for x in 0..rw as usize {
            let p = resized.get_pixel(x as u32, y as u32);
            let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
            let i = y * (pw as usize) + x;
            // порядок BGR — так модель обучалась (см. конфиг det.yml)
            data[i] = (b / 255.0 - 0.485) / 0.229;
            data[c + i] = (g / 255.0 - 0.456) / 0.224;
            data[2 * c + i] = (r / 255.0 - 0.406) / 0.225;
        }
    }
    DetInput {
        data,
        shape: [1, 3, ph as usize, pw as usize],
        ratio,
    }
}

#[derive(Clone)]
pub struct DetBox {
    pub quad: [Pt; 4], // углы [tl, tr, br, bl] в координатах исходной картинки
    pub score: f32,
}

/// Карта вероятностей «здесь текст».
pub struct ProbMap {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

/// Постобработка: бинаризация → компоненты связности (8-соседей) →
/// минимальный прямоугольник → расширение (unclip) → координаты оригинала.
pub fn postprocess(prob: &ProbMap, ratio: f32, orig_w: u32, orig_h: u32) -> Vec<DetBox> {
    let (w, h) = (prob.w, prob.h);
    let n = w * h;
    let mut mask = vec![false; n];
    for i in 0..n {
        mask[i] = prob.data[i] > DET_THRESH;
    }
    let mut visited = vec![false; n];
    let mut boxes: Vec<DetBox> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let offs: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    for start in 0..n {
        if !mask[start] || visited[start] {
            continue;
        }
        visited[start] = true;
        stack.clear();
        stack.push(start);
        let mut pts: Vec<Pt> = Vec::new();
        let (mut x0, mut x1) = (usize::MAX, 0usize);
        let (mut y0, mut y1) = (usize::MAX, 0usize);
        while let Some(i) = stack.pop() {
            let x = i % w;
            let y = i / w;
            pts.push(Pt {
                x: x as f32,
                y: y as f32,
            });
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
            for (dx, dy) in offs {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if mask[j] && !visited[j] {
                    visited[j] = true;
                    stack.push(j);
                }
            }
        }
        if pts.len() < 3 {
            continue;
        }
        // средняя вероятность по bbox компонента (как box_score_fast в PaddleOCR)
        let mut sum = 0.0f32;
        for yy in y0..=y1 {
            let row = yy * w;
            for xx in x0..=x1 {
                sum += prob.data[row + xx];
            }
        }
        let score = sum / (((y1 - y0 + 1) * (x1 - x0 + 1)) as f32);
        if score < BOX_THRESH {
            continue;
        }
        let hull = convex_hull(&pts);
        let rect = min_area_rect(&hull);
        let area = rect_area(&rect);
        let peri = rect_peri(&rect);
        if area < 1.0 || peri < 1.0 {
            continue;
        }
        let d = area * UNCLIP_RATIO / peri;
        let mut quad = expand_rect(&rect, d);
        // обратно в координаты исходной картинки
        for p in quad.iter_mut() {
            p.x = (p.x / ratio).clamp(0.0, orig_w as f32 - 1.0);
            p.y = (p.y / ratio).clamp(0.0, orig_h as f32 - 1.0);
        }
        boxes.push(DetBox { quad, score });
    }
    sort_boxes(&mut boxes);
    boxes
}

/// Сортировка блоков «как читают»: сверху вниз, внутри строки — слева направо.
fn sort_boxes(boxes: &mut Vec<DetBox>) {
    if boxes.len() <= 1 {
        return;
    }
    let quad_h = |q: &[Pt; 4]| -> f32 {
        let o = order_quad(q);
        dist(o[0], o[3]).max(dist(o[1], o[2]))
    };
    let mut heights: Vec<f32> = boxes.iter().map(|b| quad_h(&b.quad)).collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_h = heights[heights.len() / 2].max(1.0);

    let yc = |b: &DetBox| -> f32 {
        b.quad.iter().map(|p| p.y).sum::<f32>() / 4.0
    };
    let xc = |b: &DetBox| -> f32 {
        b.quad.iter().map(|p| p.x).sum::<f32>() / 4.0
    };
    boxes.sort_by(|a, b| yc(a).partial_cmp(&yc(b)).unwrap());
    // группировка в строки по разрыву y-центра
    let mut groups: Vec<Vec<usize>> = vec![vec![0]];
    for i in 1..boxes.len() {
        let last = *groups.last().unwrap().last().unwrap();
        if yc(&boxes[i]) - yc(&boxes[last]) > 0.5 * med_h {
            groups.push(vec![i]);
        } else {
            groups.last_mut().unwrap().push(i);
        }
    }
    let mut sorted_idx: Vec<usize> = Vec::with_capacity(boxes.len());
    for mut g in groups {
        g.sort_by(|&a, &b| xc(&boxes[a]).partial_cmp(&xc(&boxes[b])).unwrap());
        sorted_idx.extend(g);
    }
    let mut result: Vec<DetBox> = Vec::with_capacity(boxes.len());
    for i in sorted_idx {
        result.push(boxes[i].clone());
    }
    *boxes = result;
}
