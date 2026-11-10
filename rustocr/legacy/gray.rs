//! Полутоновое изображение: интегральные суммы, адаптивная бинаризация
//! Сауволы, поворот для выправления перекоса.

/// Приведение к полутоновому (BT.709). Возвращает (пиксели, ширина, высота).
pub fn to_luma(img: &image::DynamicImage) -> (Vec<u8>, usize, usize) {
    let l = img.to_luma8();
    let (w, h) = (l.width() as usize, l.height() as usize);
    (l.into_raw(), w, h)
}

/// Интегральное изображение (суммы и суммы квадратов) для быстрых
/// локальных статистик при бинаризации.
pub struct Integral {
    w: usize,
    h: usize,
    stride: usize,
    sum: Vec<u64>,
    sq: Vec<u64>,
}

impl Integral {
    pub fn new(gray: &[u8], w: usize, h: usize) -> Self {
        let stride = w + 1;
        let mut sum = vec![0u64; stride * (h + 1)];
        let mut sq = vec![0u64; stride * (h + 1)];
        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                let v = gray[row + x] as u64;
                let i = (y + 1) * stride + (x + 1);
                let up = i - stride;
                sum[i] = v + sum[i - 1] + sum[up] - sum[up - 1];
                sq[i] = v * v + sq[i - 1] + sq[up] - sq[up - 1];
            }
        }
        Integral { w, h, stride, sum, sq }
    }

    fn raw(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> (u64, u64) {
        let s = self.stride;
        let a = y0 * s + x0;
        let b = y1 * s + x1;
        let c = y0 * s + x1;
        let d = y1 * s + x0;
        (
            self.sum[b] - self.sum[c] - self.sum[d] + self.sum[a],
            self.sq[b] - self.sq[c] - self.sq[d] + self.sq[a],
        )
    }

    /// Среднее и СКО в прямоугольнике [x0,x1) x [y0,y1), окно клампится к границам.
    pub fn mean_std(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> (f64, f64) {
        let x0 = x0.min(self.w);
        let y0 = y0.min(self.h);
        let x1 = x1.min(self.w).max(x0);
        let y1 = y1.min(self.h).max(y0);
        let n = (((x1 - x0) * (y1 - y0)).max(1)) as f64;
        let (s, sq) = self.raw(x0, y0, x1, y1);
        let m = s as f64 / n;
        let var = (sq as f64 / n - m * m).max(0.0);
        (m, var.sqrt())
    }
}

/// Адаптивная бинаризация Сауволы: t = m * (1 + k*(s/R - 1)).
/// Возвращает карту `true` = чернила.
pub fn sauvola(gray: &[u8], w: usize, h: usize, win: usize, k: f64) -> Vec<bool> {
    let int = Integral::new(gray, w, h);
    let half = win / 2;
    let mut out = vec![false; w * h];
    let r = 128.0;
    for y in 0..h {
        let y0 = y.saturating_sub(half);
        let y1 = (y + half + 1).min(h);
        for x in 0..w {
            let x0 = x.saturating_sub(half);
            let x1 = (x + half + 1).min(w);
            let (m, s) = int.mean_std(x0, y0, x1, y1);
            let t = m * (1.0 + k * (s / r - 1.0));
            out[y * w + x] = (gray[y * w + x] as f64) < t;
        }
    }
    out
}

/// Доля чернил в бинарной карте.
pub fn ink_ratio(bin: &[bool]) -> f64 {
    if bin.is_empty() {
        return 0.0;
    }
    bin.iter().filter(|b| **b).count() as f64 / bin.len() as f64
}

/// Инверсия (светлый текст на тёмном фоне).
pub fn invert(bin: &mut [bool]) {
    for b in bin.iter_mut() {
        *b = !*b;
    }
}

/// Билинейный поворот с расширением холста (фон белый).
/// Знак: положительный угол — содержимое поворачивается против часовой
/// стрелки в экранных координатах (y вниз); выправление перекоса θ
/// выполняется вызовом с углом -θ.
pub fn rotate_bilinear(src: &[u8], w: usize, h: usize, angle: f64) -> (Vec<u8>, usize, usize) {
    if angle.abs() < 1e-5 || w < 2 || h < 2 {
        return (src.to_vec(), w, h);
    }
    let s = angle.sin();
    let c = angle.cos();
    let (as_, ac) = (s.abs(), c.abs());
    let nw = ((w as f64 * ac + h as f64 * as_).ceil() as usize).max(2);
    let nh = ((w as f64 * as_ + h as f64 * ac).ceil() as usize).max(2);
    let (cx, cy) = ((w as f64 - 1.0) / 2.0, (h as f64 - 1.0) / 2.0);
    let (ncx, ncy) = ((nw as f64 - 1.0) / 2.0, (nh as f64 - 1.0) / 2.0);
    let mut dst = vec![255u8; nw * nh];
    for y in 0..nh {
        let dy = y as f64 - ncy;
        for x in 0..nw {
            let dx = x as f64 - ncx;
            // обратное отображение
            let sx = cx + dx * c + dy * s;
            let sy = cy - dx * s + dy * c;
            if sx < 0.0 || sy < 0.0 || sx > (w - 1) as f64 || sy > (h - 1) as f64 {
                continue;
            }
            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let fx = sx - x0 as f64;
            let fy = sy - y0 as f64;
            let v00 = src[y0 * w + x0] as f64;
            let v01 = src[y0 * w + x1] as f64;
            let v10 = src[y1 * w + x0] as f64;
            let v11 = src[y1 * w + x1] as f64;
            let v = v00 * (1.0 - fx) * (1.0 - fy)
                + v01 * fx * (1.0 - fy)
                + v10 * (1.0 - fx) * fy
                + v11 * fx * fy;
            dst[y * nw + x] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    (dst, nw, nh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sauvola_finds_dark_text() {
        // 50x10, белый фон, чёрная полоса по центру
        let w = 50usize;
        let h = 10usize;
        let mut gray = vec![255u8; w * h];
        for y in 3..7 {
            for x in 10..40 {
                gray[y * w + x] = 30;
            }
        }
        let bin = sauvola(&gray, w, h, 15, 0.34);
        let mut fg = 0usize;
        for y in 3..7 {
            for x in 10..40 {
                if bin[y * w + x] {
                    fg += 1;
                }
            }
        }
        assert!(fg > 100, "полоса должна распознаваться как чернила, fg={fg}");
        // полоса 120 пикселей из 500 — это 24%
        assert!(ink_ratio(&bin) < 0.35);
    }
}
