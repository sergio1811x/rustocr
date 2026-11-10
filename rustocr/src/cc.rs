//! Связные компоненты (8-связность) на бинарной маске — «блобы».

#[derive(Clone, Copy, Debug)]
pub struct Blob {
    /// x1/y1 включительно
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
    pub area: u32,
}

impl Blob {
    pub fn w(&self) -> i32 {
        self.x1 - self.x0 + 1
    }
    pub fn h(&self) -> i32 {
        self.y1 - self.y0 + 1
    }
}

fn find(parent: &mut Vec<u32>, x: u32) -> u32 {
    let mut root = x;
    while parent[root as usize] != root {
        root = parent[root as usize];
    }
    // сжатие пути
    let mut cur = x;
    while parent[cur as usize] != cur {
        let next = parent[cur as usize];
        parent[cur as usize] = root;
        cur = next;
    }
    root
}

fn union(parent: &mut Vec<u32>, a: u32, b: u32) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[rb as usize] = ra;
    }
}

/// Разметка связных компонент; отбрасывает блобы площадью меньше `min_area`.
pub fn connected_components(bin: &[bool], w: usize, h: usize, min_area: u32) -> Vec<Blob> {
    let mut parent: Vec<u32> = Vec::new();
    let mut label = vec![0u32; w * h];

    for y in 0..h {
        let row = y * w;
        let up_row = row.wrapping_sub(w); // строка y-1 (валидна при y>0)
        for x in 0..w {
            if !bin[row + x] {
                continue;
            }
            let mut nb = [0u32; 4];
            let mut n = 0;
            if y > 0 {
                if x > 0 && bin[up_row + x - 1] {
                    nb[n] = label[up_row + x - 1];
                    n += 1;
                }
                if bin[up_row + x] {
                    nb[n] = label[up_row + x];
                    n += 1;
                }
                if x + 1 < w && bin[up_row + x + 1] {
                    nb[n] = label[up_row + x + 1];
                    n += 1;
                }
            }
            if x > 0 && bin[row + x - 1] {
                nb[n] = label[row + x - 1];
                n += 1;
            }

            let lab = if n == 0 {
                // новая метка: parent[0] — неиспользуемый слот, метки 1..=len-1
                if parent.is_empty() {
                    parent.push(0);
                }
                let new_lab = parent.len() as u32;
                parent.push(new_lab);
                new_lab
            } else {
                let m = nb[..n].iter().copied().min().unwrap();
                for i in 1..n {
                    if nb[i] != m {
                        union(&mut parent, m, nb[i]);
                    }
                }
                m
            };
            label[row + x] = lab;
        }
    }

    // Второй проход: компактные корни + накопление bbox/площади
    let mut blobs: Vec<Blob> = Vec::new();
    let mut root_to_idx: Vec<Option<usize>> = vec![None; parent.len() + 1];
    for y in 0..h {
        for x in 0..w {
            let l = label[y * w + x];
            if l == 0 {
                continue;
            }
            let r = find(&mut parent, l) as usize;
            let idx = match root_to_idx[r] {
                Some(i) => i,
                None => {
                    blobs.push(Blob {
                        x0: x as i32,
                        y0: y as i32,
                        x1: x as i32,
                        y1: y as i32,
                        area: 0,
                    });
                    root_to_idx[r] = Some(blobs.len() - 1);
                    blobs.len() - 1
                }
            };
            let b = &mut blobs[idx];
            if (x as i32) < b.x0 {
                b.x0 = x as i32;
            }
            if (x as i32) > b.x1 {
                b.x1 = x as i32;
            }
            if (y as i32) < b.y0 {
                b.y0 = y as i32;
            }
            if (y as i32) > b.y1 {
                b.y1 = y as i32;
            }
            b.area += 1;
        }
    }
    blobs.retain(|b| b.area >= min_area);
    blobs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_letters_two_blobs() {
        // 'I' и 'l' — два вертикальных штриха
        let w = 9usize;
        let h = 7usize;
        let mut bin = vec![false; w * h];
        for y in 1..6 {
            bin[y * w + 1] = true; // штрих 1
            bin[y * w + 7] = true; // штрих 2 (по диагонали не соединяются)
        }
        let blobs = connected_components(&bin, w, h, 1);
        assert_eq!(blobs.len(), 2);
        assert_eq!(blobs[0].w(), 1);
        assert_eq!(blobs[0].h(), 5);
    }

    #[test]
    fn diagonal_is_one_blob() {
        let w = 5usize;
        let h = 5usize;
        let mut bin = vec![false; w * h];
        for i in 0..5 {
            bin[i * w + i] = true;
        }
        let blobs = connected_components(&bin, w, h, 1);
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].area, 5);
    }
}
