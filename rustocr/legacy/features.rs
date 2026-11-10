//! Признаки глифа: нормализация в квадрат с сохранением пропорций,
//! контуры Собеля с квантованием направления на 8 секторов,
//! Chamfer-преобразования по направлениям, зонные плотности, битовая маска.
//!
//! Одна и та же функция `extract` применяется и к распознаваемым глифам,
//! и к шаблонам — этим гарантируется сопоставимость признаков.

/// Размер сетки признаков.
pub const GRID: usize = 48;
/// Порог отсечки расстояния в Chamfer-метчинге (3-4 метрика: ~5 пикселей).
pub const CAP: f32 = 15.0;
/// Нижний порог модуля градиента (пол подавления шума).
const EDGE_FLOOR: f32 = 0.12;

#[derive(Clone, Debug)]
pub struct GlyphFeatures {
    /// Контурные пиксели: (x, y, направление градиента 0..8)
    pub feats: Vec<(u8, u8, u8)>,
    /// 8 Chamfer-карт (по одному на направление)
    pub dts: Vec<Vec<f32>>,
    /// Зонные плотности чернил 4x4 (сумма = 1)
    pub zone: [f32; 16],
    /// Аспект tight-bbox: w/h
    pub aspect: f32,
    /// Средняя плотность чернил в tight-bbox
    pub ink: f32,
    /// 32x32 битовая маска (строка = один u64)
    pub bmp: [u64; 32],
    /// Относительное положение базовой линии: (baseline - y0) / h
    pub bl_off: f32,
    /// Высота глифа относительно x-высоты строки/шрифта
    pub rel_h: f32,
}

/// Билинейная нормализация полутонового патча в квадрат size x size
/// с сохранением пропорций и центрированием. Значения — «чернила» в [0,1].
pub fn normalize_patch(src: &[u8], w: usize, h: usize, size: usize) -> Vec<f32> {
    let mut out = vec![0f32; size * size];
    if w == 0 || h == 0 || size == 0 {
        return out;
    }
    let scale = size as f64 / (w.max(h) as f64);
    let nw = (((w as f64) * scale).round() as usize).max(1);
    let nh = (((h as f64) * scale).round() as usize).max(1);
    let ox = (size.saturating_sub(nw)) / 2;
    let oy = (size.saturating_sub(nh)) / 2;
    for y in 0..nh.min(size) {
        let sy = (((y as f64) + 0.5) / scale - 0.5).clamp(0.0, (h - 1) as f64);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(h - 1);
        let fy = (sy - y0 as f64) as f32;
        for x in 0..nw.min(size) {
            let sx = (((x as f64) + 0.5) / scale - 0.5).clamp(0.0, (w - 1) as f64);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let fx = (sx - x0 as f64) as f32;
            let v00 = src[y0 * w + x0] as f32;
            let v01 = src[y0 * w + x1] as f32;
            let v10 = src[y1 * w + x0] as f32;
            let v11 = src[y1 * w + x1] as f32;
            let v = v00 * (1.0 - fx) * (1.0 - fy)
                + v01 * fx * (1.0 - fy)
                + v10 * (1.0 - fx) * fy
                + v11 * fx * fy;
            out[(oy + y) * size + (ox + x)] = (255.0 - v) / 255.0;
        }
    }
    out
}

/// Box-размытие 3x3.
fn blur3(p: &[f32]) -> Vec<f32> {
    let n = GRID;
    let mut out = vec![0f32; p.len()];
    for y in 0..n {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(n - 1);
        for x in 0..n {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(n - 1);
            let s = p[ym * n + xm] + p[ym * n + x] + p[ym * n + xp]
                + p[y * n + xm] + p[y * n + x] + p[y * n + xp]
                + p[yp * n + xm] + p[yp * n + x] + p[yp * n + xp];
            out[y * n + x] = s / 9.0;
        }
    }
    out
}

/// Контурные пиксели Собеля с 8 направлениями градиента.
/// Порог адаптивный: 35% от 87.5-го перцентиля магнитуд — устойчиво и к
/// мягким краям мелких глифов (после апскейла), и к контрасту сканов.
fn sobel_feats(p: &[f32]) -> Vec<(u8, u8, u8)> {
    let n = GRID;
    // (x, y, dir, mag)
    let mut cands: Vec<(u8, u8, u8, f32)> = Vec::with_capacity(1024);
    for y in 0..n {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(n - 1);
        for x in 0..n {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(n - 1);
            let gx = (p[ym * n + xp] + 2.0 * p[y * n + xp] + p[yp * n + xp])
                - (p[ym * n + xm] + 2.0 * p[y * n + xm] + p[yp * n + xm]);
            let gy = (p[yp * n + xm] + 2.0 * p[yp * n + x] + p[yp * n + xp])
                - (p[ym * n + xm] + 2.0 * p[ym * n + x] + p[ym * n + xp]);
            let mag = (gx * gx + gy * gy).sqrt();
            if mag < EDGE_FLOOR {
                continue;
            }
            let ang = gy.atan2(gx); // (-pi, pi]
            let sector = ((ang / std::f32::consts::FRAC_PI_4).round() as i32 + 8) % 8;
            cands.push((x as u8, y as u8, sector as u8, mag));
        }
    }
    if cands.is_empty() {
        return vec![];
    }
    // адаптивный порог: 35% от 87.5-го перцентиля магнитуд
    let mut mags: Vec<f32> = cands.iter().map(|c| c.3).collect();
    mags.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let idx = ((mags.len() as f64) * 0.125) as usize;
    let p875 = mags[idx.min(mags.len() - 1)];
    let thr = (p875 * 0.35).max(EDGE_FLOOR);
    cands
        .into_iter()
        .filter(|c| c.3 >= thr)
        .map(|(x, y, d, _)| (x, y, d))
        .collect()
}

/// true если координата найдена (не usize::MAX).
#[inline]
fn x1_ok(v: usize) -> bool {
    v != usize::MAX
}

/// Chamfer-преобразование (3-4) булевой маски, с потолком CAP.
fn chamfer_dt(mask: &[bool]) -> Vec<f32> {
    const INF: f32 = 1e9;
    let n = GRID;
    let mut d = Vec::with_capacity(mask.len());
    for &m in mask {
        d.push(if m { 0.0 } else { INF });
    }
    // прямой проход
    for y in 0..n {
        for x in 0..n {
            let i = y * n + x;
            if d[i] == 0.0 {
                continue;
            }
            let mut m = d[i];
            if x > 0 {
                m = m.min(d[i - 1] + 3.0);
            }
            if y > 0 {
                m = m.min(d[i - n] + 3.0);
            }
            if x > 0 && y > 0 {
                m = m.min(d[i - n - 1] + 4.0);
            }
            if x + 1 < n && y > 0 {
                m = m.min(d[i - n + 1] + 4.0);
            }
            d[i] = m;
        }
    }
    // обратный проход
    for y in (0..n).rev() {
        for x in (0..n).rev() {
            let i = y * n + x;
            if d[i] == 0.0 {
                continue;
            }
            let mut m = d[i];
            if x + 1 < n {
                m = m.min(d[i + 1] + 3.0);
            }
            if y + 1 < n {
                m = m.min(d[i + n] + 3.0);
            }
            if x + 1 < n && y + 1 < n {
                m = m.min(d[i + n + 1] + 4.0);
            }
            if x > 0 && y + 1 < n {
                m = m.min(d[i + n - 1] + 4.0);
            }
            d[i] = m;
        }
    }
    for v in d.iter_mut() {
        if *v > CAP {
            *v = CAP;
        }
    }
    d
}

/// Полное извлечение признаков из полутонового патча глифа.
/// `mask` — бинарная маска того же патча (если есть — tight-bbox по ней,
/// это стабильнее на мягких/мелких глифах, чем порог по полутону);
/// `baseline_y` — базовая линия в системе патча; `x_height` — для rel_h.
/// Возвращает None для пустого патча.
pub fn extract(
    gray: &[u8],
    w: usize,
    h: usize,
    mask: Option<&[bool]>,
    baseline_y: f64,
    x_height: f64,
) -> Option<GlyphFeatures> {
    if w == 0 || h == 0 {
        return None;
    }
    // два кандидата tight-bbox: по маске (стабильно на мягких глифах) и
    // по порогу полутона (устойчиво к шуму). Если маска раздута более чем
    // на 35% относительно полутонового (спекл шума) — берём полутоновый.
    let bbox_from = |pred: &dyn Fn(usize) -> bool| -> (usize, usize, usize, usize, usize) {
        let mut x0 = usize::MAX;
        let mut y0 = usize::MAX;
        let mut x1 = 0usize;
        let mut y1 = 0usize;
        let mut cnt = 0usize;
        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                if pred(row + x) {
                    cnt += 1;
                    if x < x0 {
                        x0 = x;
                    }
                    if x > x1 {
                        x1 = x;
                    }
                    if y < y0 {
                        y0 = y;
                    }
                    if y > y1 {
                        y1 = y;
                    }
                }
            }
        }
        (x0, y0, x1, y1, cnt)
    };
    let gb = bbox_from(&|i| gray[i] < 170);
    let mb = match mask {
        Some(m) => bbox_from(&|i| m[i]),
        None => gb,
    };
    let (x0, y0, x1, y1, ink_px) = if mb.4 >= 3 && x1_ok(mb.2) && x1_ok(gb.2) {
        let mw = (mb.2 as i64 - mb.0 as i64) + 1;
        let mh = (mb.3 as i64 - mb.1 as i64) + 1;
        let gw = (gb.2 as i64 - gb.0 as i64) + 1;
        let gh = (gb.3 as i64 - gb.1 as i64) + 1;
        if mw <= gw * 135 / 100 && mh <= gh * 135 / 100 {
            mb
        } else {
            gb
        }
    } else {
        gb
    };
    if ink_px < 3 || x1 < x0 {
        return None;
    }
    let tw = x1 - x0 + 1;
    let th = y1 - y0 + 1;
    // вырезаем tight-патч
    let mut crop = vec![255u8; tw * th];
    for y in 0..th {
        let src_row = (y0 + y) * w;
        let dst_row = y * tw;
        for x in 0..tw {
            crop[dst_row + x] = gray[src_row + x0 + x];
        }
    }
    let patch = normalize_patch(&crop, tw, th, GRID);
    let patch32 = normalize_patch(&crop, tw, th, 32);
    let blurred = blur3(&patch);

    // битовая маска 32x32
    let mut bmp = [0u64; 32];
    for y in 0..32 {
        for x in 0..32 {
            if patch32[y * 32 + x] >= 0.45 {
                bmp[y] |= 1u64 << x;
            }
        }
    }

    // контуры
    let feats = sobel_feats(&blurred);

    // 8 Chamfer-карт по направлениям
    let mut masks = vec![vec![false; GRID * GRID]; 8];
    for &(x, y, d) in &feats {
        masks[d as usize][y as usize * GRID + x as usize] = true;
    }
    let dts: Vec<Vec<f32>> = masks.iter().map(|m| chamfer_dt(m)).collect();

    // зонные плотности 4x4
    let cell = GRID / 4;
    let mut zone = [0f32; 16];
    let mut total = 0f32;
    for cy in 0..4 {
        for cx in 0..4 {
            let mut s = 0f32;
            for y in cy * cell..(cy + 1) * cell {
                for x in cx * cell..(cx + 1) * cell {
                    s += blurred[y * GRID + x];
                }
            }
            zone[cy * 4 + cx] = s;
            total += s;
        }
    }
    if total > 1e-6 {
        for z in zone.iter_mut() {
            *z /= total;
        }
    }

    let _ink = blurred.iter().sum::<f32>() / (GRID * GRID) as f32;
    // плотность по tight-bbox (не по квадрату)
    let ink_tight = ink_px as f32 / (tw * th) as f32;
    let bl_off = ((baseline_y - y0 as f64) / th as f64) as f32;
    let rel_h = if x_height > 1.0 {
        (th as f64 / x_height) as f32
    } else {
        1.0
    };

    Some(GlyphFeatures {
        feats,
        dts,
        zone,
        aspect: tw as f32 / th as f32,
        ink: ink_tight,
        bmp,
        bl_off: bl_off.clamp(-0.5, 1.5),
        rel_h: rel_h.clamp(0.2, 3.5),
    })
}

/// XOR-расстояние двух битовых масок (нормированное).
pub fn bmp_xor(a: &[u64; 32], b: &[u64; 32]) -> f32 {
    let mut diff = 0u32;
    let mut uni = 0u32;
    for i in 0..32 {
        diff += (a[i] ^ b[i]).count_ones();
        uni += (a[i] | b[i]).count_ones();
    }
    if uni == 0 {
        0.0
    } else {
        diff as f32 / uni as f32
    }
}
