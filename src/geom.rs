//! Геометрия: выпуклая оболочка, минимальный прямоугольник, перспективное вырезание.

use image::RgbImage;

#[derive(Clone, Copy, Debug)]
pub struct Pt {
    pub x: f32,
    pub y: f32,
}

pub fn dist(a: Pt, b: Pt) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

fn cross(o: Pt, a: Pt, b: Pt) -> f32 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

/// Выпуклая оболочка (цепочка Эндрю), результат — против часовой стрелки.
pub fn convex_hull(points: &[Pt]) -> Vec<Pt> {
    let mut pts = points.to_vec();
    if pts.len() < 3 {
        pts.dedup_by(|a, b| a.x == b.x && a.y == b.y);
        return pts;
    }
    pts.sort_by(|a, b| {
            a.x.partial_cmp(&b.x)
                .unwrap()
                .then(a.y.partial_cmp(&b.y).unwrap())
    });
    let mut lower: Vec<Pt> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Pt> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Минимальный по площади прямоугольник вокруг оболочки («вращающиеся линейки»).
/// Возвращает 4 угла против часовой стрелки.
pub fn min_area_rect(hull: &[Pt]) -> [Pt; 4] {
    let n = hull.len();
    if n == 0 {
        return [Pt { x: 0.0, y: 0.0 }; 4];
    }
    if n < 3 {
        let a = hull[0];
        let b = hull[n - 1];
        return [a, b, b, a];
    }
    let mut best_area = f32::MAX;
    let mut best = [Pt { x: 0.0, y: 0.0 }; 4];
    for i in 0..n {
        let a = hull[i];
        let b = hull[(i + 1) % n];
        let ex = b.x - a.x;
        let ey = b.y - a.y;
        let l = (ex * ex + ey * ey).sqrt();
        if l < 1e-6 {
            continue;
        }
        let (ux, uy) = (ex / l, ey / l); // направление ребра
        let (vx, vy) = (-uy, ux); // перпендикуляр
        let (mut minu, mut maxu, mut minv, mut maxv) =
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for p in hull {
            let u = p.x * ux + p.y * uy;
            let v = p.x * vx + p.y * vy;
            minu = minu.min(u);
            maxu = maxu.max(u);
            minv = minv.min(v);
            maxv = maxv.max(v);
        }
        let area = (maxu - minu) * (maxv - minv);
        if area < best_area {
            best_area = area;
            // обратное преобразование для ортонормированных осей — транспонирование
            let corner = |u: f32, v: f32| Pt {
                x: u * ux + v * vx,
                y: u * uy + v * vy,
            };
            best = [
                corner(minu, minv),
                corner(maxu, minv),
                corner(maxu, maxv),
                corner(minu, maxv),
            ];
        }
    }
    best
}

/// Площадь прямоугольника из 4 углов (CCW).
pub fn rect_area(c: &[Pt; 4]) -> f32 {
    dist(c[0], c[1]) * dist(c[0], c[3])
}

/// Периметр прямоугольника из 4 углов (CCW).
pub fn rect_peri(c: &[Pt; 4]) -> f32 {
    2.0 * (dist(c[0], c[1]) + dist(c[0], c[3]))
}

/// Расширить прямоугольник (углы CCW) на d по всем сторонам.
pub fn expand_rect(c: &[Pt; 4], d: f32) -> [Pt; 4] {
    let w = dist(c[0], c[1]);
    let h = dist(c[0], c[3]);
    if w < 1e-3 || h < 1e-3 {
        return *c;
    }
    let ux = ((c[1].x - c[0].x) / w, (c[1].y - c[0].y) / w);
    let vy = ((c[3].x - c[0].x) / h, (c[3].y - c[0].y) / h);
    let o = Pt {
        x: c[0].x - d * ux.0 - d * vy.0,
        y: c[0].y - d * ux.1 - d * vy.1,
    };
    let w2 = w + 2.0 * d;
    let h2 = h + 2.0 * d;
    [
        o,
        Pt {
            x: o.x + w2 * ux.0,
            y: o.y + w2 * ux.1,
        },
        Pt {
            x: o.x + w2 * ux.0 + h2 * vy.0,
            y: o.y + w2 * ux.1 + h2 * vy.1,
        },
        Pt {
            x: o.x + h2 * vy.0,
            y: o.y + h2 * vy.1,
        },
    ]
}

/// Упорядочить 4 точки: [верх-лево, верх-право, низ-право, низ-лево].
pub fn order_quad(q: &[Pt; 4]) -> [Pt; 4] {
    let mut s = [0.0f32; 4];
    let mut d = [0.0f32; 4];
    for i in 0..4 {
        s[i] = q[i].x + q[i].y;
        d[i] = q[i].y - q[i].x;
    }
    let tl = q[argmin(&s)];
    let br = q[argmax(&s)];
    let tr = q[argmin(&d)];
    let bl = q[argmax(&d)];
    [tl, tr, br, bl]
}

fn argmin(v: &[f32; 4]) -> usize {
    let mut m = 0;
    for i in 1..4 {
        if v[i] < v[m] {
            m = i;
        }
    }
    m
}

fn argmax(v: &[f32; 4]) -> usize {
    let mut m = 0;
    for i in 1..4 {
        if v[i] > v[m] {
            m = i;
        }
    }
    m
}

/// Билинейная выборка пикселя (координаты могут быть дробными, клэмп по краям).
fn bilinear(src: &RgbImage, x: f32, y: f32) -> image::Rgb<u8> {
    let (sw, sh) = (src.width() as i32, src.height() as i32);
    let x = x.clamp(0.0, (sw - 1) as f32);
    let y = y.clamp(0.0, (sh - 1) as f32);
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = (x0 + 1).min(sw - 1);
    let y1 = (y0 + 1).min(sh - 1);
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let p00 = src.get_pixel(x0 as u32, y0 as u32);
    let p10 = src.get_pixel(x1 as u32, y0 as u32);
    let p01 = src.get_pixel(x0 as u32, y1 as u32);
    let p11 = src.get_pixel(x1 as u32, y1 as u32);
    let mut out = [0u8; 3];
    for ch in 0..3 {
        let a = p00[ch] as f32 * (1.0 - dx) + p10[ch] as f32 * dx;
        let b = p01[ch] as f32 * (1.0 - dx) + p11[ch] as f32 * dx;
        out[ch] = (a * (1.0 - dy) + b * dy).round().clamp(0.0, 255.0) as u8;
    }
    image::Rgb::<u8>(out)
}

/// Решить систему 8×8 методом Гаусса с выбором ведущего элемента.
fn gauss_solve(a: &mut [[f64; 8]; 8], b: &mut [f64; 8]) -> [f64; 8] {
    for col in 0..8 {
        let mut piv = col;
        for r in (col + 1)..8 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        if d.abs() < 1e-12 {
            a[col][col] += 1e-12;
            continue;
        }
        for r in (col + 1)..8 {
            let f = a[r][col] / a[col][col];
            if f == 0.0 {
                continue;
            }
            for c in col..8 {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0f64; 8];
    for r in (0..8).rev() {
        let mut s = b[r];
        for c in (r + 1)..8 {
            s -= a[r][c] * x[c];
        }
        x[r] = s / a[r][r];
    }
    x
}

/// Гомография из 4 соответствий src→dst (h8 = 1), 9 коэффициентов.
fn solve_homography(src: &[Pt; 4], dst: &[Pt; 4]) -> [f64; 9] {
    let mut a = [[0.0f64; 8]; 8];
    let mut b = [0.0f64; 8];
    for i in 0..4 {
        let (x, y) = (src[i].x as f64, src[i].y as f64);
        let (u, v) = (dst[i].x as f64, dst[i].y as f64);
        let r = 2 * i;
        a[r][0] = x;
        a[r][1] = y;
        a[r][2] = 1.0;
        a[r][6] = -u * x;
        a[r][7] = -u * y;
        b[r] = u;
        let r2 = 2 * i + 1;
        a[r2][3] = x;
        a[r2][4] = y;
        a[r2][5] = 1.0;
        a[r2][6] = -v * x;
        a[r2][7] = -v * y;
        b[r2] = v;
    }
    let h = gauss_solve(&mut a, &mut b);
    [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]
}

/// Вырезать четырёхугольник quad ([tl,tr,br,bl]) в прямоугольник w×h
/// с выравниванием перспективы и билинейной выборкой.
pub fn warp_quad(src: &RgbImage, quad: &[Pt; 4], w: u32, h: u32) -> RgbImage {
    let dst = [
        Pt { x: 0.0, y: 0.0 },
        Pt {
            x: w as f32 - 1.0,
            y: 0.0,
        },
        Pt {
            x: w as f32 - 1.0,
            y: h as f32 - 1.0,
        },
        Pt {
            x: 0.0,
            y: h as f32 - 1.0,
        },
    ];
    // гомография dst→src, чтобы для каждого пикселя результата сразу знать источник
    let m = solve_homography(&dst, quad);
    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f64, y as f64);
            let z = m[6] * fx + m[7] * fy + 1.0;
            let sx = ((m[0] * fx + m[1] * fy + m[2]) / z) as f32;
            let sy = ((m[3] * fx + m[4] * fy + m[5]) / z) as f32;
            out.put_pixel(x, y, bilinear(src, sx, sy));
        }
    }
    out
}

/// Ресайз с антиалиасингом (треугольный фильтр ≈ cv2.INTER_LINEAR).
pub fn resize_tri(img: &RgbImage, w: u32, h: u32) -> RgbImage {
    image::imageops::resize(img, w, h, image::imageops::FilterType::Triangle)
}
