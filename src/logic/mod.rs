// SPDX-License-Identifier: MIT
use crate::*;

#[pyfunction]
fn all(a: &NdArray) -> bool {
    a.all()
}

#[pyfunction]
fn any(a: &NdArray) -> bool {
    a.any()
}

#[pyfunction]
fn isnan(x: &NdArray) -> NdArray {
    NdArray {
        imag: None,
        data: x.data.mapv(|v| if v.is_nan() { 1.0 } else { 0.0 }),
    }
}

#[pyfunction]
fn isinf(x: &NdArray) -> NdArray {
    NdArray {
        imag: None,
        data: x.data.mapv(|v| if v.is_infinite() { 1.0 } else { 0.0 }),
    }
}

#[pyfunction]
fn isfinite(x: &NdArray) -> NdArray {
    NdArray {
        imag: None,
        data: x.data.mapv(|v| if v.is_finite() { 1.0 } else { 0.0 }),
    }
}

/// 逐元素判断虚部是否为 0（isreal）。非 complex 全 True；
/// complex 用 abs(imag) < 1e-15（对齐原 Python 语义，避免 tolist + 递归）。
#[pyfunction]
fn isreal(x: &NdArray) -> NdArray {
    let data = match x.imag.as_ref() {
        None => x.data.mapv(|_| 1.0),
        Some(im) => im
            .clone()
            .mapv(|iv| if iv.abs() < 1e-15 { 1.0 } else { 0.0 }),
    };
    NdArray { imag: None, data }
}

/// 逐元素判断虚部是否非 0（iscomplex）。非 complex 全 False；
/// complex 用 imag != 0（对齐原 __init__ 语义，避免 tolist + 递归）。
#[pyfunction]
fn iscomplex(x: &NdArray) -> NdArray {
    let data = match x.imag.as_ref() {
        None => x.data.mapv(|_| 0.0),
        Some(im) => im.clone().mapv(|iv| if iv != 0.0 { 1.0 } else { 0.0 }),
    };
    NdArray { imag: None, data }
}

#[pyfunction]
fn maximum(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a.max(b))?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn minimum(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a.min(b))?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn allclose(a: &NdArray, b: &NdArray, rtol: f64, atol: f64) -> bool {
    if a.data.len() != b.data.len() {
        return false;
    }
    for (va, vb) in a.data.iter().zip(b.data.iter()) {
        if (va - vb).abs() > atol + rtol * vb.abs() {
            return false;
        }
    }
    true
}

#[pyfunction]
fn array_equal(a: &NdArray, b: &NdArray) -> bool {
    if a.data.shape() != b.data.shape() {
        return false;
    }
    a.data.iter().zip(b.data.iter()).all(|(x, y)| x == y)
}

#[pyfunction]
fn greater(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a > b { 1.0 } else { 0.0 })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn less(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a < b { 1.0 } else { 0.0 })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn equal(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    // 数组语义：== 为精确 IEEE 比较——inf==inf 为 True、nan==nan 为 False。
    // 不得用 |a-b|<eps 容差：inf-inf=nan 会使 inf==inf 误判为 False。
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a == b { 1.0 } else { 0.0 })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn not_equal(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    // 与 equal 对偶：nan!=nan 为 True（IEEE 不等），inf!=inf 为 False。
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a != b { 1.0 } else { 0.0 })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn greater_equal(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a >= b { 1.0 } else { 0.0 })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn less_equal(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a <= b { 1.0 } else { 0.0 })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn logical_and(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 }
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn logical_or(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 }
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn logical_xor(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        if (a != 0.0) != (b != 0.0) { 1.0 } else { 0.0 }
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

/// 逐元素 isclose（数组语义）：|a-b| <= atol + rtol*|b|；相等值（含 inf==inf /
/// -inf==-inf）恒 close；NaN 对仅 equal_nan 时 True；非有限非相等对恒 False。
/// 复数取模，实/虚部分别判有限。
/// 支持通用 N 维广播（ndarray 零拷贝广播视图）。
/// 注：pyo3 0.30 的 signature 宏对 `=false` 布尔默认值会静默丢弃参数，
/// 统一用 Option<bool> + None 默认值
#[pyfunction]
#[pyo3(signature = (a, b, rtol=1e-05, atol=1e-08, equal_nan=None))]
fn isclose(
    a: &NdArray,
    b: &NdArray,
    rtol: f64,
    atol: f64,
    equal_nan: Option<bool>,
) -> PyResult<NdArray> {
    let equal_nan = equal_nan.unwrap_or(false);
    let out_shape =
        broadcast_shape(a.data.shape(), b.data.shape()).map_err(PyValueError::new_err)?;
    // broadcast 返回零拷贝视图；物化为 owned 以复用 `&owned` Zip part 惯例
    // （ndarray 0.17 中 `&视图` 不是合法的 Zip part）
    let ar = a
        .data
        .broadcast(IxDyn(&out_shape))
        .ok_or_else(|| PyValueError::new_err("broadcast shape mismatch"))?
        .to_owned();
    let br = b
        .data
        .broadcast(IxDyn(&out_shape))
        .ok_or_else(|| PyValueError::new_err("broadcast shape mismatch"))?
        .to_owned();
    let is_complex = a.imag.is_some() || b.imag.is_some();
    let mut out = Array::zeros(IxDyn(&out_shape));
    if is_complex {
        let ai = match &a.imag {
            Some(im) => im
                .broadcast(IxDyn(&out_shape))
                .ok_or_else(|| PyValueError::new_err("broadcast shape mismatch"))?
                .to_owned(),
            None => Array::zeros(ar.dim().clone()),
        };
        let bi = match &b.imag {
            Some(im) => im
                .broadcast(IxDyn(&out_shape))
                .ok_or_else(|| PyValueError::new_err("broadcast shape mismatch"))?
                .to_owned(),
            None => Array::zeros(br.dim().clone()),
        };
        Zip::from(&mut out)
            .and(&ar)
            .and(&br)
            .and(&ai)
            .and(&bi)
            .for_each(|o: &mut f64, x, y, xi, yi| {
                let a_nan = x.is_nan() || xi.is_nan();
                let b_nan = y.is_nan() || yi.is_nan();
                *o = if a_nan && b_nan {
                    f64::from(equal_nan)
                } else if x == y && xi == yi {
                    1.0
                } else if x.is_finite() && xi.is_finite() && y.is_finite() && yi.is_finite() {
                    let d_re = x - y;
                    let d_im = xi - yi;
                    let dist = (d_re * d_re + d_im * d_im).sqrt();
                    let mag_b = (y * y + yi * yi).sqrt();
                    f64::from(dist <= atol + rtol * mag_b)
                } else {
                    0.0
                };
            });
    } else {
        Zip::from(&mut out).and(&ar).and(&br).for_each(|o, x, y| {
            *o = if x.is_nan() && y.is_nan() {
                f64::from(equal_nan)
            } else if x == y {
                1.0
            } else if x.is_finite() && y.is_finite() {
                f64::from((x - y).abs() <= atol + rtol * y.abs())
            } else {
                0.0
            };
        });
    }
    Ok(NdArray {
        imag: None,
        data: out,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(greater, m)?)?;
    m.add_function(wrap_pyfunction!(less, m)?)?;
    m.add_function(wrap_pyfunction!(equal, m)?)?;
    m.add_function(wrap_pyfunction!(not_equal, m)?)?;
    m.add_function(wrap_pyfunction!(greater_equal, m)?)?;
    m.add_function(wrap_pyfunction!(less_equal, m)?)?;
    m.add_function(wrap_pyfunction!(logical_and, m)?)?;
    m.add_function(wrap_pyfunction!(logical_or, m)?)?;
    m.add_function(wrap_pyfunction!(logical_xor, m)?)?;
    m.add_function(wrap_pyfunction!(isclose, m)?)?;
    m.add_function(wrap_pyfunction!(allclose, m)?)?;
    m.add_function(wrap_pyfunction!(array_equal, m)?)?;
    m.add_function(wrap_pyfunction!(all, m)?)?;
    m.add_function(wrap_pyfunction!(any, m)?)?;
    m.add_function(wrap_pyfunction!(isnan, m)?)?;
    m.add_function(wrap_pyfunction!(isinf, m)?)?;
    m.add_function(wrap_pyfunction!(isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(isreal, m)?)?;
    m.add_function(wrap_pyfunction!(iscomplex, m)?)?;
    m.add_function(wrap_pyfunction!(maximum, m)?)?;
    m.add_function(wrap_pyfunction!(minimum, m)?)?;
    Ok(())
}
