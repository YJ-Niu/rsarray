// SPDX-License-Identifier: MIT
use crate::*;

#[pyfunction]
fn invert(x: &NdArray) -> NdArray {
    NdArray {
        imag: None,
        data: x.data.mapv(|v| if v == 0.0 { 1.0 } else { 0.0 }),
    }
}

#[pyfunction]
fn bitwise_and(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) & (b as i64);
        bits as f64
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn bitwise_or(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) | (b as i64);
        bits as f64
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn bitwise_xor(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) ^ (b as i64);
        bits as f64
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn left_shift(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) << (b as i64);
        bits as f64
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn right_shift(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) >> (b as i64);
        bits as f64
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn bitwise_not(x: &NdArray) -> NdArray {
    NdArray {
        imag: None,
        data: x.data.mapv(|v| {
            let bits = !(v as i64);
            bits as f64
        }),
    }
}

#[pyfunction]
fn binary_repr(num: i64, width: Option<usize>) -> String {
    let w = if let Some(w) = width {
        w
    } else {
        let bits = 64 - num.abs().leading_zeros();
        if bits == 0 { 1 } else { bits as usize }
    };

    if num >= 0 {
        format!("{num:0w$b}")
    } else {
        let val = (1 << w) + num;
        format!("{val:0w$b}")
    }
}

/// 逐元素统计绝对值二进制表示中置位比特数（与 bitwise_count 一致，输出 uint8）。
#[pyfunction]
fn bitwise_count(x: &NdArray) -> NdArray {
    NdArray {
        imag: None,
        data: x.data.mapv(|v| (v.abs() as i64).count_ones() as f64),
    }
}

// ========== 比特打包 ==========
/// 将布尔/整数数组按比特打包为 uint8
#[pyfunction]
#[pyo3(signature = (a, bitorder="big"))]
fn packbits(a: &NdArray, bitorder: &str) -> NdArray {
    let bits: Vec<u8> = a
        .data
        .iter()
        .map(|&v| if v != 0.0 { 1u8 } else { 0u8 })
        .collect();
    let n = bits.len().div_ceil(8);
    let mut out = vec![0u8; n];
    for (i, &b) in bits.iter().enumerate() {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        if bitorder == "big" {
            out[byte_idx] |= b << (7 - bit_idx);
        } else {
            out[byte_idx] |= b << bit_idx;
        }
    }
    let result: Vec<f64> = out.iter().map(|&v| v as f64).collect();
    let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
        .expect("打包比特后构造数组形状必然合法");
    NdArray {
        imag: None,
        data: arr,
    }
}

// ========== 比特展开 ==========
/// 将 uint8 数组展开为比特
#[pyfunction]
#[pyo3(signature = (a, count=None, bitorder="big"))]
fn unpackbits(a: &NdArray, count: Option<usize>, bitorder: &str) -> NdArray {
    let vals: Vec<u8> = a.data.iter().map(|&v| v as u8).collect();
    let mut out: Vec<u8> = Vec::with_capacity(vals.len() * 8);
    for &iv in &vals {
        let mut bits = [0u8; 8];
        for (j, bit) in bits.iter_mut().enumerate() {
            *bit = if bitorder == "big" {
                (iv >> (7 - j)) & 1
            } else {
                (iv >> j) & 1
            };
        }
        out.extend(bits.iter());
    }
    if let Some(c) = count {
        out.truncate(c);
    }
    let result: Vec<f64> = out.iter().map(|&v: &u8| v as f64).collect();
    let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
        .expect("展开比特后构造数组形状必然合法");
    NdArray {
        imag: None,
        data: arr,
    }
}

/// 比特展开（core 直连）：a 为数值数组（任意布局，按逻辑 C 序读取），
/// bitorder 为 big/little。
/// axis=None → 展平后 1-D 展开（长度 8*size）；
/// axis=Some(ax) → 位轴（长度 n）替换原 ax 轴，其他维保持原始顺序不变。
/// count：正数 → 截断/零填充到该长度；负数 → total + count（< 0 抛 ValueError）；
/// None → 全展开长度。
#[pyfunction]
#[pyo3(signature = (a, axis=None, count=None, bitorder="big"))]
fn unpackbits_core(
    a: &NdArray,
    axis: Option<usize>,
    count: Option<isize>,
    bitorder: &str,
) -> PyResult<NdArray> {
    if a.imag.is_some() {
        return Err(PyTypeError::new_err(
            "unpackbits complex not supported in core path",
        ));
    }
    let big = match bitorder {
        "big" => true,
        "little" => false,
        other => {
            return Err(PyValueError::new_err(format!(
                "bitorder must be 'big' or 'little', got {other:?}"
            )));
        }
    };
    let nd = a.data.ndim();
    let expand_byte = |v: u8| -> [u8; 8] {
        let mut b = [0u8; 8];
        for (j, bit) in b.iter_mut().enumerate() {
            *bit = if big {
                (v >> (7 - j)) & 1
            } else {
                (v >> j) & 1
            };
        }
        b
    };
    let make_out = |shape: Vec<usize>, flat: Vec<u8>| {
        let result: Vec<f64> = flat.iter().map(|&v| v as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    };
    match axis {
        None => {
            let vals: Vec<u8> = a.data.iter().map(|&v| v as u8).collect();
            let total = 8 * vals.len();
            let n_i = match count {
                Some(c) if c >= 0 => c,
                Some(c) => total as isize + c,
                None => total as isize,
            };
            if n_i < 0 {
                return Err(PyValueError::new_err(format!(
                    "Invalid count {} for unpacking {} bits",
                    count.unwrap_or(0),
                    total
                )));
            }
            let n = n_i as usize;
            let mut out = vec![0u8; n];
            for (i, &v) in vals.iter().enumerate() {
                let b = expand_byte(v);
                for (j, &bv) in b.iter().enumerate() {
                    let pos = i * 8 + j;
                    if pos < n {
                        out[pos] = bv;
                    }
                }
            }
            make_out(vec![n], out)
        }
        Some(ax) => {
            if ax >= nd {
                return Err(PyValueError::new_err("axis out of bounds for unpackbits"));
            }
            let ashp: Vec<usize> = a.data.shape().to_vec();
            let other: Vec<usize> = (0..nd).filter(|&i| i != ax).collect();
            let other_shape: Vec<usize> = other.iter().map(|&i| ashp[i]).collect();
            let n_other: usize = other_shape.iter().product();
            let k = ashp[ax];
            let total = 8 * k;
            let n_i = match count {
                Some(c) if c >= 0 => c,
                Some(c) => total as isize + c,
                None => total as isize,
            };
            if n_i < 0 {
                return Err(PyValueError::new_err(format!(
                    "Invalid count {} for unpacking {} bits",
                    count.unwrap_or(0),
                    total
                )));
            }
            let n = n_i as usize;
            // 源按逻辑 C 序展平（任意布局安全）
            let sv: Vec<f64> = a.data.iter().copied().collect();
            let mut strides = vec![1usize; nd];
            for i in (1..nd).rev() {
                strides[i - 1] = strides[i] * ashp[i];
            }
            let other_strides: Vec<usize> = other.iter().map(|&i| strides[i]).collect();
            let outsize = n * n_other;
            let mut out = vec![0u8; outsize];
            for r in 0..n_other {
                let mut rem = r;
                let mut base = 0usize;
                for j in (0..other.len()).rev() {
                    let d = rem % other_shape[j];
                    rem /= other_shape[j];
                    base += d * other_strides[j];
                }
                for av in 0..k {
                    let v = sv[base + av * strides[ax]] as u8;
                    let b = expand_byte(v);
                    for (bit, &bv) in b.iter().enumerate() {
                        // 全位索引 = av*8 + bit（漏 av*8 会让同 r 各字节互相覆盖）
                        let pos = (av * 8 + bit) * n_other + r;
                        if pos < outsize {
                            out[pos] = bv;
                        }
                    }
                }
            }
            // (n, *other) 布局 → 最终形状（位轴放回 ax 位置，其他轴原始顺序）：
            // r = pre_c * post_prod + post_c；flat_f = (pre_c * n + bit) * post_prod + post_c
            let post_prod: usize = (ax + 1..nd).map(|i| ashp[i]).product();
            let mut final_out = vec![0u8; outsize];
            for r in 0..n_other {
                let pre = r / post_prod;
                let post = r % post_prod;
                for bit in 0..n {
                    final_out[(pre * n + bit) * post_prod + post] = out[bit * n_other + r];
                }
            }
            let mut final_shape = ashp;
            final_shape[ax] = n;
            make_out(final_shape, final_out)
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(invert, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_and, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_or, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_xor, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_not, m)?)?;
    m.add_function(wrap_pyfunction!(left_shift, m)?)?;
    m.add_function(wrap_pyfunction!(right_shift, m)?)?;
    m.add_function(wrap_pyfunction!(binary_repr, m)?)?;
    m.add_function(wrap_pyfunction!(packbits, m)?)?;
    m.add_function(wrap_pyfunction!(unpackbits, m)?)?;
    m.add_function(wrap_pyfunction!(unpackbits_core, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_count, m)?)?;
    Ok(())
}
