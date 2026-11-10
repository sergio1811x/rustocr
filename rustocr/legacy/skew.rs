//! Оценка перекоса страницы: максимизация дисперсии проекционного профиля.

/// Возвращает угол перекоса в градусах: положительный — строки текста
/// «спускаются» вправо (в экранных координатах, y вниз).
/// Диапазон поиска: ±4° (грубый шаг 0.5°, точный 0.05°).
pub fn estimate_skew(bin: &[bool], w: usize, h: usize) -> f64 {
    if w * h == 0 {
        return 0.0;
    }
    // равномерная подвыборка точек чернил (не более ~300 тыс.)
    let stride = ((w * h) / 300_000).max(1);
    let mut pts: Vec<(i32, i32)> = Vec::new();
    let mut i = 0usize;
    while i < bin.len() {
        if bin[i] {
            pts.push(((i % w) as i32, (i / w) as i32));
        }
        i += stride;
    }
    if pts.len() < 30 {
        return 0.0;
    }

    let max_off = ((w as f64) * 4.0f64.to_radians().tan()).ceil() as i32 + 2;
    let nbins = (h as i32 + 2 * max_off + 2) as usize;

    let eval = |ang_deg: f64| -> f64 {
        let t = ang_deg.to_radians().tan();
        let mut prof = vec![0f64; nbins];
        for &(x, y) in &pts {
            let r = (y as f64 - (x as f64) * t).round() as i64 + max_off as i64;
            if r >= 0 && (r as usize) < nbins {
                prof[r as usize] += 1.0;
            }
        }
        prof.iter().map(|p| p * p).sum::<f64>()
    };

    // грубый поиск
    let mut best = 0.0f64;
    let mut best_score = eval(0.0);
    let mut a = -4.0f64;
    while a <= 4.0 {
        let s = eval(a);
        if s > best_score {
            best_score = s;
            best = a;
        }
        a += 0.5;
    }
    // уточнение
    let mut a = best - 0.45;
    while a <= best + 0.45 {
        let s = eval(a);
        if s > best_score {
            best_score = s;
            best = a;
        }
        a += 0.05;
    }
    // нормируем мелкий дрейф нуля
    if best.abs() < 0.06 {
        return 0.0;
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_horizontal_as_zero() {
        let w = 200usize;
        let h = 100usize;
        let mut bin = vec![false; w * h];
        for y in [10usize, 40, 70] {
            for x in 5..150 {
                bin[y * w + x] = true;
            }
        }
        assert_eq!(estimate_skew(&bin, w, h), 0.0);
    }

    #[test]
    fn detects_small_skew() {
        let w = 400usize;
        let h = 200usize;
        let mut bin = vec![false; w * h];
        let tan = 1.0f64.to_radians().tan();
        for y0 in [20.0f64, 90.0, 160.0] {
            for x in 10..380 {
                let y = (y0 + x as f64 * tan).round() as usize;
                if y < h {
                    bin[y * w + x] = true;
                }
            }
        }
        let ang = estimate_skew(&bin, w, h);
        assert!((ang - 1.0).abs() < 0.25, " ожидался ~1°, получили {ang}");
    }
}
