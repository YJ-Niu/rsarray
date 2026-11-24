// SPDX-License-Identifier: MIT
use ndarray::linalg::Dot;
use ndarray::{Array, Ix2, Ix3, IxDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::NdArray;

/// 点积
#[pyfunction]
fn dot(_py: Python<'_>, a: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    _py.detach(move || {
        let a_data = &a.data;
        let b_data = &b.data;
        let a_shape = a_data.shape();
        let b_shape = b_data.shape();

        // 形状校验（避免底层内核 panic），随后交给 ndarray 的 Dot：
        // 其 IxDyn 实现会把 1D/2D 组合分派到优化的 gemv/gemm 内核。
        let compatible = match (a_shape.len(), b_shape.len()) {
            (1, 1) => a_shape[0] == b_shape[0],
            (2, 2) => a_shape[1] == b_shape[0],
            (1, 2) => a_shape[0] == b_shape[0],
            (2, 1) => a_shape[1] == b_shape[0],
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported shapes for dot product: {:?} and {:?}",
                    a_shape, b_shape
                )));
            }
        };
        if !compatible {
            return Err(PyValueError::new_err(format!(
                "Incompatible shapes for dot product: {:?} and {:?}",
                a_shape, b_shape
            )));
        }
        Ok(NdArray {
            imag: None,
            data: a_data.dot(b_data),
        })
    })
}

/// 向量点积（展开为一维）
#[pyfunction]
fn vdot(_py: Python<'_>, a: &NdArray, b: &NdArray) -> PyResult<f64> {
    _py.detach(move || {
        let a_flat: Vec<f64> = a.data.iter().copied().collect();
        let b_flat: Vec<f64> = b.data.iter().copied().collect();
        if a_flat.len() != b_flat.len() {
            return Err(PyValueError::new_err(format!(
                "vdot requires same number of elements: {} vs {}",
                a_flat.len(),
                b_flat.len()
            )));
        }
        let result: f64 = a_flat.iter().zip(b_flat.iter()).map(|(x, y)| x * y).sum();
        Ok(result)
    })
}

/// 内积
#[pyfunction]
fn inner(_py: Python<'_>, a: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    _py.detach(move || {
        let a_data = &a.data;
        let b_data = &b.data;
        let a_shape = a_data.shape().to_vec();
        let b_shape = b_data.shape().to_vec();

        if a_shape.len() == 1 && b_shape.len() == 1 {
            if a_shape[0] != b_shape[0] {
                return Err(PyValueError::new_err(format!(
                    "Incompatible shapes for inner product: {:?} and {:?}",
                    a_shape, b_shape
                )));
            }
            let result: f64 = a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).sum();
            return Ok(NdArray {
                imag: None,
                data: Array::from_elem(IxDyn(&[]), result),
            });
        }

        if a_shape.len() == 2 && b_shape.len() == 2 {
            if a_shape[1] != b_shape[1] {
                return Err(PyValueError::new_err(format!(
                    "Incompatible shapes for inner product: {:?} and {:?}",
                    a_shape, b_shape
                )));
            }
            // inner(a, b)[i, j] = Σ_k a[i,k]·b[j,k] = a · bᵀ。
            // 交给 ndarray 的 Dot（f64 时走 BLAS gemm），而非朴素三重循环。
            let a2 = a_data
                .view()
                .into_dimensionality::<Ix2>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let b2 = b_data
                .view()
                .into_dimensionality::<Ix2>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let c = a2.dot(&b2.t());
            return Ok(NdArray {
                imag: None,
                data: c.into_dyn(),
            });
        }

        Err(PyValueError::new_err(format!(
            "Unsupported shapes for inner product: {:?} and {:?}",
            a_shape, b_shape
        )))
    })
}

/// 矩阵乘法
#[pyfunction]
fn matmul(_py: Python<'_>, a: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    _py.detach(move || {
        let a_data = &a.data;
        let b_data = &b.data;
        let a_shape = a_data.shape().to_vec();
        let b_shape = b_data.shape().to_vec();

        let a_imag = &a.imag;
        let b_imag = &b.imag;
        let has_imag = a_imag.is_some() || b_imag.is_some();

        match (a_shape.len(), b_shape.len()) {
            (1, 1) | (2, 2) | (2, 1) | (1, 2) => {
                if *a_shape.last().unwrap() != b_shape[0] {
                    return Err(PyValueError::new_err(format!(
                        "Incompatible shapes for matmul: {:?} and {:?}",
                        a_shape, b_shape
                    )));
                }

                let real_part = a_data.dot(b_data);

                if !has_imag {
                    return Ok(NdArray {
                        imag: None,
                        data: real_part,
                    });
                }

                let zero_a = a_data.clone() * 0.0;
                let zero_b = b_data.clone() * 0.0;
                let a_i = a_imag.as_ref().unwrap_or(&zero_a);
                let b_i = b_imag.as_ref().unwrap_or(&zero_b);
                let real_part = real_part - a_i.dot(b_i);
                let imag_part = a_data.dot(b_i) + a_i.dot(b_data);

                return Ok(NdArray {
                    imag: Some(imag_part),
                    data: real_part,
                });
            }
            _ => {}
        }

        if a_shape.len() >= 3 && b_shape.len() == 2 {
            let batch_dims: Vec<usize> = a_shape[..a_shape.len() - 2].to_vec();
            let m = a_shape[a_shape.len() - 2];
            let n = a_shape[a_shape.len() - 1];
            let p = b_shape[1];
            if n != b_shape[0] {
                return Err(PyValueError::new_err(format!(
                    "Incompatible shapes for matmul: {:?} and {:?}",
                    a_shape, b_shape
                )));
            }
            let batch_size: usize = batch_dims.iter().product();

            let a2 = a_data
                .to_shape((batch_size * m, n))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let b2 = b_data
                .to_shape((n, p))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            if !has_imag {
                let real_part = a2.dot(&b2);
                let mut result_shape = batch_dims.clone();
                result_shape.extend_from_slice(&[m, p]);
                let real_arr = real_part
                    .into_shape_with_order(result_shape)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                return Ok(NdArray {
                    imag: None,
                    data: real_arr,
                });
            }

            let zero_a = a_data.clone() * 0.0;
            let zero_b = b_data.clone() * 0.0;
            let a_i = a_imag.as_ref().unwrap_or(&zero_a);
            let b_i = b_imag.as_ref().unwrap_or(&zero_b);

            let a_i_2 = a_i
                .to_shape((batch_size * m, n))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let b_i_2 = b_i
                .to_shape((n, p))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            let real_part = a2.dot(&b2) - a_i_2.dot(&b_i_2);
            let imag_part = a2.dot(&b_i_2) + a_i_2.dot(&b2);

            let mut result_shape = batch_dims.clone();
            result_shape.extend_from_slice(&[m, p]);
            let real_arr = real_part
                .into_shape_with_order(result_shape.clone())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let imag_arr = imag_part
                .into_shape_with_order(result_shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            return Ok(NdArray {
                imag: Some(imag_arr),
                data: real_arr,
            });
        }

        // stacked（批处理）matmul：批维广播 (B...,m,k) @ (C...,k,n) → (B',...,m,n)，
        // 1-D 边归一成 (1,k)/(k,1) 参与计算再 reshape 回 (B',...,m,)/(B',...,n,)。
        // Rust rayon 并行内核，替代 _matmul_fallback 的逐批 Python 循环。
        if let Some(res) = batched_matmul(a, b) {
            return res;
        }

        Err(PyValueError::new_err(format!(
            "Unsupported shapes for matmul: {:?} and {:?}",
            a_shape, b_shape
        )))
    })
}

/// 批维广播：右对齐逐维，须相等或有一侧为 1。
/// 返回 (out_batch_shape, a_广播(取 batch 0), b_广播)；不匹配返回 None。
fn broadcast_batches(a_batch: &[usize], b_batch: &[usize]) -> Option<(Vec<usize>, bool, bool)> {
    let nd = a_batch.len().max(b_batch.len());
    let mut out = vec![1usize; nd];
    let mut a_bc = false;
    let mut b_bc = false;
    for p in 0..nd {
        let da = if p < a_batch.len() {
            a_batch[a_batch.len() - 1 - p]
        } else {
            1
        };
        let db = if p < b_batch.len() {
            b_batch[b_batch.len() - 1 - p]
        } else {
            1
        };
        match (da, db) {
            (1, 1) => {}
            (1, _) => {
                out[nd - 1 - p] = db;
                a_bc = true;
            }
            (_, 1) => {
                out[nd - 1 - p] = da;
                b_bc = true;
            }
            (x, y) if x == y => out[nd - 1 - p] = x,
            _ => return None,
        }
    }
    Some((out, a_bc, b_bc))
}

/// 批量 matmul 实数内核：a (B,m,k) @ b (B,k,n) → (B,m,n) flat（C 序），rayon 按批并行。
/// a_bc/b_bc 为 true 时该侧恒取 batch 0（其批维是被广播的 1）。
#[allow(clippy::too_many_arguments)]
fn matmul_batched_real(
    a: &Array<f64, Ix3>,
    b: &Array<f64, Ix3>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    a_bc: bool,
    b_bc: bool,
) -> Vec<f64> {
    let results: Vec<Vec<f64>> = crate::threadpool::with_pool(|| {
        (0..batch)
            .into_par_iter()
            .map(|bi| {
                let ai = if a_bc { 0 } else { bi };
                let bj = if b_bc { 0 } else { bi };
                let mut out = vec![0.0f64; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f64;
                        for kk in 0..k {
                            sum += a[[ai, i, kk]] * b[[bj, kk, j]];
                        }
                        out[i * n + j] = sum;
                    }
                }
                out
            })
            .collect()
    });
    let mut flat = Vec::with_capacity(batch * m * n);
    for r in results {
        flat.extend(r);
    }
    flat
}

/// 批量 matmul 复数内核：(re,im) 逐元素乘加融合，输出 (B,m,n) flat 的 re/im。
#[allow(clippy::too_many_arguments)]
fn matmul_batched_complex(
    ar: &Array<f64, Ix3>,
    ai: &Array<f64, Ix3>,
    br: &Array<f64, Ix3>,
    bi: &Array<f64, Ix3>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    a_bc: bool,
    b_bc: bool,
) -> (Vec<f64>, Vec<f64>) {
    let results: Vec<(Vec<f64>, Vec<f64>)> = crate::threadpool::with_pool(|| {
        (0..batch)
            .into_par_iter()
            .map(|b| {
                let ab = if a_bc { 0 } else { b };
                let bb = if b_bc { 0 } else { b };
                let mut re = vec![0.0f64; m * n];
                let mut im = vec![0.0f64; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut srr = 0.0f64;
                        let mut sri = 0.0f64;
                        let mut sir = 0.0f64;
                        let mut sii = 0.0f64;
                        for kk in 0..k {
                            let ar_ = ar[[ab, i, kk]];
                            let ai_ = ai[[ab, i, kk]];
                            let br_ = br[[bb, kk, j]];
                            let bi_ = bi[[bb, kk, j]];
                            srr += ar_ * br_;
                            sri += ar_ * bi_;
                            sir += ai_ * br_;
                            sii += ai_ * bi_;
                        }
                        re[i * n + j] = srr - sii;
                        im[i * n + j] = sri + sir;
                    }
                }
                (re, im)
            })
            .collect()
    });
    let cap = batch * m * n;
    let mut re = Vec::with_capacity(cap);
    let mut im = Vec::with_capacity(cap);
    for (rr, ii) in results {
        re.extend(rr);
        im.extend(ii);
    }
    (re, im)
}

/// stacked（批处理）matmul 入口：(B...,m,k) @ (C...,k,n) → (B',...,m,n)（批维广播），
/// 1-D 边归一成 (1,k)/(k,1)（结果去掉归一维）。
/// 形状无法按批处理 matmul 解释时返回 None，由调用方报 Unsupported。
fn batched_matmul(a: &NdArray, b: &NdArray) -> Option<PyResult<NdArray>> {
    let a_shape = a.data.shape().to_vec();
    let b_shape = b.data.shape().to_vec();
    let a_len = a_shape.len();
    let b_len = b_shape.len();
    if a_len == 0 || b_len == 0 {
        return None;
    }
    let (m, k_a, a_batch) = if a_len == 1 {
        (1, a_shape[0], Vec::new())
    } else {
        (
            a_shape[a_len - 2],
            a_shape[a_len - 1],
            a_shape[..a_len - 2].to_vec(),
        )
    };
    // b 是 (..., k, n)：倒数第二维是内维 k_b，最后一维是输出维 n
    let (n, k_b, b_batch) = if b_len == 1 {
        (1, b_shape[0], Vec::new())
    } else {
        (
            b_shape[b_len - 1],
            b_shape[b_len - 2],
            b_shape[..b_len - 2].to_vec(),
        )
    };
    if k_a != k_b {
        return None;
    }
    let k = k_a;
    let (batch_shape, a_bc, b_bc) = broadcast_batches(&a_batch, &b_batch)?;
    let a_tot: usize = a_batch.iter().product();
    let b_tot: usize = b_batch.iter().product();
    // to_shape 返回 CowArray（可能是视图），to_owned 落为拥有的 Ix3 数组
    let a3 = a.data.to_shape((a_tot, m, k)).ok()?.to_owned();
    let b3 = b.data.to_shape((b_tot, k, n)).ok()?.to_owned();
    let batch: usize = batch_shape.iter().product();

    let mut out_shape = batch_shape;
    if a_len > 1 {
        out_shape.push(m);
    }
    if b_len > 1 {
        out_shape.push(n);
    }

    if !a.has_imag() && !b.has_imag() {
        let flat = matmul_batched_real(&a3, &b3, batch, m, k, n, a_bc, b_bc);
        let data = Array::from_shape_vec(IxDyn(&out_shape), flat)
            .ok()?
            .into_dyn();
        return Some(Ok(NdArray { imag: None, data }));
    }

    let a_im = match a.imag.as_ref() {
        Some(im) => im.to_shape((a_tot, m, k)).ok()?.to_owned(),
        None => Array::zeros((a_tot, m, k)),
    };
    let b_im = match b.imag.as_ref() {
        Some(im) => im.to_shape((b_tot, k, n)).ok()?.to_owned(),
        None => Array::zeros((b_tot, k, n)),
    };
    let (re, im) = matmul_batched_complex(&a3, &a_im, &b3, &b_im, batch, m, k, n, a_bc, b_bc);
    let data = Array::from_shape_vec(IxDyn(&out_shape), re)
        .ok()?
        .into_dyn();
    let imag = Array::from_shape_vec(IxDyn(&out_shape), im)
        .ok()?
        .into_dyn();
    Some(Ok(NdArray {
        imag: Some(imag),
        data,
    }))
}

/// 矩阵求逆 (2x2, 3x3)
#[pyfunction]
fn inv(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();

    // 3D 批量 (B, 2, 2) 快速路径：rayon 并行 + 复数支持
    if shape.len() == 3 && shape[1] == 2 && shape[2] == 2 {
        return inv_batch_2x2(a);
    }

    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("inv requires a square matrix"));
    }
    let n = shape[0];
    match n {
        2 => {
            let det = a.data[[0, 0]] * a.data[[1, 1]] - a.data[[0, 1]] * a.data[[1, 0]];
            if det == 0.0 {
                return Err(PyValueError::new_err("Singular matrix"));
            }
            let inv_det = 1.0 / det;
            let result = vec![
                a.data[[1, 1]] * inv_det,
                -a.data[[0, 1]] * inv_det,
                -a.data[[1, 0]] * inv_det,
                a.data[[0, 0]] * inv_det,
            ];
            let arr = Array::from_shape_vec((2, 2), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray {
                imag: None,
                data: arr.into_dyn(),
            })
        }
        3 => {
            let a11 = a.data[[0, 0]];
            let a12 = a.data[[0, 1]];
            let a13 = a.data[[0, 2]];
            let a21 = a.data[[1, 0]];
            let a22 = a.data[[1, 1]];
            let a23 = a.data[[1, 2]];
            let a31 = a.data[[2, 0]];
            let a32 = a.data[[2, 1]];
            let a33 = a.data[[2, 2]];

            let det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
                + a13 * (a21 * a32 - a22 * a31);

            if det == 0.0 {
                return Err(PyValueError::new_err("Singular matrix"));
            }
            let inv_det = 1.0 / det;

            let result = vec![
                (a22 * a33 - a23 * a32) * inv_det,
                (a13 * a32 - a12 * a33) * inv_det,
                (a12 * a23 - a13 * a22) * inv_det,
                (a23 * a31 - a21 * a33) * inv_det,
                (a11 * a33 - a13 * a31) * inv_det,
                (a13 * a21 - a11 * a23) * inv_det,
                (a21 * a32 - a22 * a31) * inv_det,
                (a12 * a31 - a11 * a32) * inv_det,
                (a11 * a22 - a12 * a21) * inv_det,
            ];
            let arr = Array::from_shape_vec((3, 3), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray {
                imag: None,
                data: arr.into_dyn(),
            })
        }
        _ => Err(PyValueError::new_err(
            "inv only supports 2x2 and 3x3 matrices",
        )),
    }
}

/// 批量 2x2 矩阵求逆，支持实数和复数，rayon 并行
fn inv_batch_2x2(a: &NdArray) -> PyResult<NdArray> {
    let batch = a.data.shape()[0];
    let a_re = &a.data;

    if a.has_imag() {
        let a_im = a.imag.as_ref().unwrap();

        // 每个矩阵返回 [r00_re, r00_im, r01_re, r01_im, r10_re, r10_im, r11_re, r11_im]
        let results: Vec<[f64; 8]> = crate::threadpool::with_pool(|| {
            (0..batch)
                .into_par_iter()
                .map(|b| {
                    let a00r = a_re[[b, 0, 0]];
                    let a00i = a_im[[b, 0, 0]];
                    let a01r = a_re[[b, 0, 1]];
                    let a01i = a_im[[b, 0, 1]];
                    let a10r = a_re[[b, 1, 0]];
                    let a10i = a_im[[b, 1, 0]];
                    let a11r = a_re[[b, 1, 1]];
                    let a11i = a_im[[b, 1, 1]];

                    // det = a00*a11 - a01*a10 (复数乘法)
                    let det_r = a00r * a11r - a00i * a11i - a01r * a10r + a01i * a10i;
                    let det_i = a00r * a11i + a00i * a11r - a01r * a10i - a01i * a10r;

                    // inv_det = conj(det) / |det|^2
                    let det_sq = det_r * det_r + det_i * det_i;
                    let inv_det_r = det_r / det_sq;
                    let inv_det_i = -det_i / det_sq;

                    // inv = inv_det * [[a11, -a01], [-a10, a00]]
                    let r00r = a11r * inv_det_r - a11i * inv_det_i;
                    let r00i = a11r * inv_det_i + a11i * inv_det_r;
                    let r01r = -(a01r * inv_det_r - a01i * inv_det_i);
                    let r01i = -(a01r * inv_det_i + a01i * inv_det_r);
                    let r10r = -(a10r * inv_det_r - a10i * inv_det_i);
                    let r10i = -(a10r * inv_det_i + a10i * inv_det_r);
                    let r11r = a00r * inv_det_r - a00i * inv_det_i;
                    let r11i = a00r * inv_det_i + a00i * inv_det_r;

                    [r00r, r00i, r01r, r01i, r10r, r10i, r11r, r11i]
                })
                .collect()
        });

        let mut re_vec = Vec::with_capacity(batch * 4);
        let mut im_vec = Vec::with_capacity(batch * 4);
        for r in &results {
            re_vec.extend_from_slice(&[r[0], r[2], r[4], r[6]]);
            im_vec.extend_from_slice(&[r[1], r[3], r[5], r[7]]);
        }

        let re_arr = Array::from_shape_vec((batch, 2, 2), re_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let im_arr = Array::from_shape_vec((batch, 2, 2), im_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(NdArray {
            imag: Some(im_arr.into_dyn()),
            data: re_arr.into_dyn(),
        })
    } else {
        // 实数批量 2x2 求逆
        let results: Vec<[f64; 4]> = crate::threadpool::with_pool(|| {
            (0..batch)
                .into_par_iter()
                .map(|b| {
                    let a00 = a_re[[b, 0, 0]];
                    let a01 = a_re[[b, 0, 1]];
                    let a10 = a_re[[b, 1, 0]];
                    let a11 = a_re[[b, 1, 1]];
                    let det = a00 * a11 - a01 * a10;
                    let inv_det = 1.0 / det;
                    [a11 * inv_det, -a01 * inv_det, -a10 * inv_det, a00 * inv_det]
                })
                .collect()
        });

        let re_vec: Vec<f64> = results.iter().flat_map(|r| r.iter().copied()).collect();
        let re_arr = Array::from_shape_vec((batch, 2, 2), re_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(NdArray {
            imag: None,
            data: re_arr.into_dyn(),
        })
    }
}

/// 行列式 (2x2, 3x3)
#[pyfunction]
fn det(a: &NdArray) -> PyResult<f64> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("det requires a square matrix"));
    }
    let n = shape[0];
    match n {
        2 => Ok(a.data[[0, 0]] * a.data[[1, 1]] - a.data[[0, 1]] * a.data[[1, 0]]),
        3 => {
            let a11 = a.data[[0, 0]];
            let a12 = a.data[[0, 1]];
            let a13 = a.data[[0, 2]];
            let a21 = a.data[[1, 0]];
            let a22 = a.data[[1, 1]];
            let a23 = a.data[[1, 2]];
            let a31 = a.data[[2, 0]];
            let a32 = a.data[[2, 1]];
            let a33 = a.data[[2, 2]];
            Ok(
                a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
                    + a13 * (a21 * a32 - a22 * a31),
            )
        }
        _ => Err(PyValueError::new_err(
            "det only supports 2x2 and 3x3 matrices",
        )),
    }
}

fn matmul_impl(a: &Array<f64, IxDyn>, b: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
    let m = a.shape()[0];
    let n = a.shape()[1];
    let p = b.shape()[1];
    let mut result = vec![0.0; m * p];
    for i in 0..m {
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[[i, k]] * b[[k, j]];
            }
            result[i * p + j] = sum;
        }
    }
    Array::from_shape_vec((m, p), result).unwrap().into_dyn()
}

fn solve2x2(a: &Array<f64, IxDyn>, b: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
    let inv_det = 1.0 / (a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]]);
    let a11 = a[[1, 1]] * inv_det;
    let a12 = -a[[0, 1]] * inv_det;
    let a21 = -a[[1, 0]] * inv_det;
    let a22 = a[[0, 0]] * inv_det;
    let b_shape = b.shape().to_vec();
    if b_shape.len() == 1 {
        let x0 = a11 * b[0] + a12 * b[1];
        let x1 = a21 * b[0] + a22 * b[1];
        Array::from_shape_vec(IxDyn(&[2]), vec![x0, x1])
            .unwrap()
            .into_dyn()
    } else {
        let n = b_shape[1];
        let mut result = vec![0.0; 2 * n];
        for j in 0..n {
            result[j] = a11 * b[[0, j]] + a12 * b[[1, j]];
            result[n + j] = a21 * b[[0, j]] + a22 * b[[1, j]];
        }
        Array::from_shape_vec((2, n), result).unwrap().into_dyn()
    }
}

fn solve3x3(a: &Array<f64, IxDyn>, b: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
    let a11 = a[[0, 0]];
    let a12 = a[[0, 1]];
    let a13 = a[[0, 2]];
    let a21 = a[[1, 0]];
    let a22 = a[[1, 1]];
    let a23 = a[[1, 2]];
    let a31 = a[[2, 0]];
    let a32 = a[[2, 1]];
    let a33 = a[[2, 2]];
    let det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31);
    let inv_det = 1.0 / det;
    let inv = [
        (a22 * a33 - a23 * a32) * inv_det,
        (a13 * a32 - a12 * a33) * inv_det,
        (a12 * a23 - a13 * a22) * inv_det,
        (a23 * a31 - a21 * a33) * inv_det,
        (a11 * a33 - a13 * a31) * inv_det,
        (a13 * a21 - a11 * a23) * inv_det,
        (a21 * a32 - a22 * a31) * inv_det,
        (a12 * a31 - a11 * a32) * inv_det,
        (a11 * a22 - a12 * a21) * inv_det,
    ];
    let b_shape = b.shape().to_vec();
    if b_shape.len() == 1 {
        let x0 = inv[0] * b[0] + inv[1] * b[1] + inv[2] * b[2];
        let x1 = inv[3] * b[0] + inv[4] * b[1] + inv[5] * b[2];
        let x2 = inv[6] * b[0] + inv[7] * b[1] + inv[8] * b[2];
        Array::from_shape_vec(IxDyn(&[3]), vec![x0, x1, x2])
            .unwrap()
            .into_dyn()
    } else {
        let n = b_shape[1];
        let mut result = vec![0.0; 3 * n];
        for j in 0..n {
            result[j] = inv[0] * b[[0, j]] + inv[1] * b[[1, j]] + inv[2] * b[[2, j]];
            result[n + j] = inv[3] * b[[0, j]] + inv[4] * b[[1, j]] + inv[5] * b[[2, j]];
            result[2 * n + j] = inv[6] * b[[0, j]] + inv[7] * b[[1, j]] + inv[8] * b[[2, j]];
        }
        Array::from_shape_vec((3, n), result).unwrap().into_dyn()
    }
}

fn eig2x2(a: &Array<f64, IxDyn>) -> (Array<f64, IxDyn>, Array<f64, IxDyn>) {
    let a11 = a[[0, 0]];
    let a12 = a[[0, 1]];
    let a21 = a[[1, 0]];
    let a22 = a[[1, 1]];
    let tr = a11 + a22;
    let det = a11 * a22 - a12 * a21;
    let disc = tr * tr - 4.0 * det;
    let sqrt_disc = disc.sqrt();
    let lambda1 = (tr + sqrt_disc) / 2.0;
    let lambda2 = (tr - sqrt_disc) / 2.0;
    let (v1_0, v1_1) = if a12.abs() > 1e-12 || (lambda1 - a11).abs() > 1e-12 {
        (a12, lambda1 - a11)
    } else {
        (lambda1 - a22, a21)
    };
    let norm1 = (v1_0 * v1_0 + v1_1 * v1_1).sqrt();
    let (v1_0, v1_1) = (v1_0 / norm1, v1_1 / norm1);
    let (v2_0, v2_1) = if a12.abs() > 1e-12 || (lambda2 - a11).abs() > 1e-12 {
        (a12, lambda2 - a11)
    } else {
        (lambda2 - a22, a21)
    };
    let norm2 = (v2_0 * v2_0 + v2_1 * v2_1).sqrt();
    let (v2_0, v2_1) = (v2_0 / norm2, v2_1 / norm2);
    let eigenvalues = Array::from_shape_vec(IxDyn(&[2]), vec![lambda1, lambda2])
        .unwrap()
        .into_dyn();
    let eigenvectors = Array::from_shape_vec((2, 2), vec![v1_0, v2_0, v1_1, v2_1])
        .unwrap()
        .into_dyn();
    (eigenvalues, eigenvectors)
}

fn eig3x3(a: &Array<f64, IxDyn>) -> (Array<f64, IxDyn>, Array<f64, IxDyn>) {
    let a11 = a[[0, 0]];
    let a12 = a[[0, 1]];
    let a13 = a[[0, 2]];
    let a21 = a[[1, 0]];
    let a22 = a[[1, 1]];
    let a23 = a[[1, 2]];
    let a31 = a[[2, 0]];
    let a32 = a[[2, 1]];
    let a33 = a[[2, 2]];
    let tr = a11 + a22 + a33;
    let m = (a11 * a22 - a12 * a21) + (a11 * a33 - a13 * a31) + (a22 * a33 - a23 * a32);
    let d = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31);
    let p = -tr;
    let q = m;
    let r = -d;
    let p2 = p * p;
    let alpha = q - p2 / 3.0;
    let beta = 2.0 * p2 * p / 27.0 - p * q / 3.0 + r;
    let disc = beta * beta / 4.0 + alpha * alpha * alpha / 27.0;
    if disc > 1e-10 {
        panic!("Complex eigenvalues not supported in eig3x3");
    }
    let sqrt_neg_alpha_3 = (-alpha / 3.0).sqrt();
    let cos_arg = (-beta / (2.0 * sqrt_neg_alpha_3.powi(3))).clamp(-1.0, 1.0);
    let theta = cos_arg.acos() / 3.0;
    let two_sqrt = 2.0 * sqrt_neg_alpha_3;
    let pi = std::f64::consts::PI;
    let t0 = two_sqrt * theta.cos();
    let t1 = two_sqrt * (theta - 2.0 * pi / 3.0).cos();
    let t2 = two_sqrt * (theta - 4.0 * pi / 3.0).cos();
    let shift = p / 3.0;
    let eigenvalues = vec![t0 - shift, t1 - shift, t2 - shift];
    let eig_arr = Array::from_shape_vec(IxDyn(&[3]), eigenvalues.clone())
        .unwrap()
        .into_dyn();
    let mut eigvecs = vec![0.0; 9];
    for (idx, &lam) in eigenvalues.iter().enumerate() {
        let b00 = a11 - lam;
        let b01 = a12;
        let b02 = a13;
        let b10 = a21;
        let b11 = a22 - lam;
        let b12 = a23;
        let b20 = a31;
        let b21 = a32;
        let b22 = a33 - lam;
        let (vx, vy, vz) = {
            let vx = b01 * b12 - b02 * b11;
            let vy = b02 * b10 - b00 * b12;
            let vz = b00 * b11 - b01 * b10;
            let n = (vx * vx + vy * vy + vz * vz).sqrt();
            if n > 1e-12 {
                (vx / n, vy / n, vz / n)
            } else {
                let vx = b01 * b22 - b02 * b21;
                let vy = b02 * b20 - b00 * b22;
                let vz = b00 * b21 - b01 * b20;
                let n = (vx * vx + vy * vy + vz * vz).sqrt();
                if n > 1e-12 {
                    (vx / n, vy / n, vz / n)
                } else {
                    let vx = b11 * b22 - b12 * b21;
                    let vy = b12 * b20 - b10 * b22;
                    let vz = b10 * b21 - b11 * b20;
                    let n = (vx * vx + vy * vy + vz * vz).sqrt();
                    (vx / n, vy / n, vz / n)
                }
            }
        };
        eigvecs[idx] = vx;
        eigvecs[3 + idx] = vy;
        eigvecs[6 + idx] = vz;
    }
    let evec_arr = Array::from_shape_vec((3, 3), eigvecs).unwrap().into_dyn();
    (eig_arr, evec_arr)
}

#[pyfunction]
#[pyo3(signature = (x, ord=None, axis=None))]
fn norm(_py: Python<'_>, x: &NdArray, ord: Option<f64>, axis: Option<isize>) -> PyResult<NdArray> {
    _py.detach(move || {
        let shape = x.data.shape().to_vec();
        let data = &x.data;
        let ord = ord.unwrap_or(2.0);
        if let Some(ax) = axis {
            let ax = if ax < 0 {
                (shape.len() as isize + ax) as usize
            } else {
                ax as usize
            };
            if ax >= shape.len() {
                return Err(PyValueError::new_err("axis out of bounds".to_string()));
            }
            let mut out_shape = shape.clone();
            out_shape.remove(ax);
            let out_size: usize = out_shape.iter().product();
            let axis_size = shape[ax];
            let mut result = vec![0.0; out_size];

            #[allow(clippy::needless_range_loop)]
            for i in 0..out_size {
                let mut non_ax_indices: Vec<usize> = Vec::with_capacity(shape.len() - 1);
                let mut temp = i;
                let other_dims: Vec<usize> = shape
                    .iter()
                    .enumerate()
                    .filter(|(d, _)| *d != ax)
                    .map(|(_, &s)| s)
                    .collect();
                for &dim in other_dims.iter().rev() {
                    non_ax_indices.push(temp % dim);
                    temp /= dim;
                }
                non_ax_indices.reverse();

                let mut s = 0.0;
                for k in 0..axis_size {
                    let mut full_indices = non_ax_indices.clone();
                    full_indices.insert(ax, k);
                    let v = data[full_indices.as_slice()];
                    if ord == 1.0 {
                        s += v.abs();
                    } else if ord == 2.0 {
                        s += v * v;
                    } else if ord == f64::INFINITY {
                        s = s.max(v.abs());
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "Unsupported norm order: {}",
                            ord
                        )));
                    }
                }
                if ord == 2.0 {
                    s = s.sqrt();
                }
                result[i] = s;
            }
            let arr = Array::from_shape_vec(IxDyn(out_shape.as_slice()), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray {
                imag: None,
                data: arr,
            })
        } else if shape.len() == 1 {
            if ord == 1.0 {
                let val: f64 = data.iter().map(|v| v.abs()).sum();
                Ok(NdArray {
                    imag: None,
                    data: Array::from_elem(IxDyn(&[]), val),
                })
            } else if ord == 2.0 {
                let val: f64 = data.iter().map(|v| v * v).sum::<f64>().sqrt();
                Ok(NdArray {
                    imag: None,
                    data: Array::from_elem(IxDyn(&[]), val),
                })
            } else if ord == f64::INFINITY {
                let val = data.iter().fold(0.0f64, |a, b| a.max(b.abs()));
                Ok(NdArray {
                    imag: None,
                    data: Array::from_elem(IxDyn(&[]), val),
                })
            } else {
                Err(PyValueError::new_err(format!(
                    "Unsupported norm order for 1D array: {}",
                    ord
                )))
            }
        } else if shape.len() == 2 {
            if ord == 1.0 {
                let ncols = shape[1];
                let mut max_sum = 0.0;
                for j in 0..ncols {
                    let s: f64 = (0..shape[0]).map(|i| data[[i, j]].abs()).sum();
                    if s > max_sum {
                        max_sum = s;
                    }
                }
                Ok(NdArray {
                    imag: None,
                    data: Array::from_elem(IxDyn(&[]), max_sum),
                })
            } else if ord == 2.0 {
                let val: f64 = data.iter().map(|v| v * v).sum::<f64>().sqrt();
                Ok(NdArray {
                    imag: None,
                    data: Array::from_elem(IxDyn(&[]), val),
                })
            } else if ord == f64::INFINITY {
                let ncols = shape[1];
                let mut max_sum = 0.0;
                for i in 0..shape[0] {
                    let s: f64 = (0..ncols).map(|j| data[[i, j]].abs()).sum();
                    if s > max_sum {
                        max_sum = s;
                    }
                }
                Ok(NdArray {
                    imag: None,
                    data: Array::from_elem(IxDyn(&[]), max_sum),
                })
            } else {
                Err(PyValueError::new_err(format!(
                    "Unsupported norm order for matrix: {}",
                    ord
                )))
            }
        } else {
            let val: f64 = if ord == 2.0 {
                data.iter().map(|v| v * v).sum::<f64>().sqrt()
            } else {
                data.iter().map(|v| v.abs()).sum()
            };
            Ok(NdArray {
                imag: None,
                data: Array::from_elem(IxDyn(&[]), val),
            })
        }
    })
}

#[pyfunction]
fn solve(a: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    let a_shape = a.data.shape().to_vec();
    let b_shape = b.data.shape().to_vec();
    if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
        return Err(PyValueError::new_err("solve requires a square matrix A"));
    }
    let n = a_shape[0];
    if !(b_shape.len() == 1 || b_shape.len() == 2) {
        return Err(PyValueError::new_err("b must be 1D or 2D"));
    }
    let b_dim0 = b_shape[0];
    if b_dim0 != n {
        return Err(PyValueError::new_err(format!(
            "Incompatible shapes: A {:?} and b {:?}",
            a_shape, b_shape
        )));
    }

    let a_has_imag = a.imag.is_some();
    let b_has_imag = b.imag.is_some();
    let has_imag = a_has_imag || b_has_imag;

    match n {
        2 => {
            let det = a.data[[0, 0]] * a.data[[1, 1]] - a.data[[0, 1]] * a.data[[1, 0]];
            if det == 0.0 {
                return Err(PyValueError::new_err("Singular matrix"));
            }

            if !has_imag {
                return Ok(NdArray {
                    imag: None,
                    data: solve2x2(&a.data, &b.data),
                });
            }

            let zero_a = a.data.clone() * 0.0;
            let zero_b = b.data.clone() * 0.0;
            let a_i = a.imag.as_ref().unwrap_or(&zero_a);
            let b_i = b.imag.as_ref().unwrap_or(&zero_b);

            let det_real = det;
            let det_imag = a.data[[0, 0]] * a_i[[1, 1]] + a_i[[0, 0]] * a.data[[1, 1]]
                - a.data[[0, 1]] * a_i[[1, 0]]
                - a_i[[0, 1]] * a.data[[1, 0]];

            let det_sq = det_real * det_real + det_imag * det_imag;
            if det_sq < 1e-24 {
                return Err(PyValueError::new_err("Singular matrix"));
            }

            let inv_det_real = det_real / det_sq;
            let inv_det_imag = -det_imag / det_sq;

            let a11 = a.data[[0, 0]];
            let a12 = a.data[[0, 1]];
            let a21 = a.data[[1, 0]];
            let a22 = a.data[[1, 1]];
            let ai11 = a_i[[0, 0]];
            let ai12 = a_i[[0, 1]];
            let ai21 = a_i[[1, 0]];
            let ai22 = a_i[[1, 1]];

            let inv_a11_r = a22 * inv_det_real + ai22 * inv_det_imag;
            let inv_a11_i = -a22 * inv_det_imag + ai22 * inv_det_real;
            let inv_a12_r = -a12 * inv_det_real - ai12 * inv_det_imag;
            let inv_a12_i = a12 * inv_det_imag - ai12 * inv_det_real;
            let inv_a21_r = -a21 * inv_det_real - ai21 * inv_det_imag;
            let inv_a21_i = a21 * inv_det_imag - ai21 * inv_det_real;
            let inv_a22_r = a11 * inv_det_real + ai11 * inv_det_imag;
            let inv_a22_i = -a11 * inv_det_imag + ai11 * inv_det_real;

            let b_shape = b.data.shape().to_vec();
            let real_result: Array<f64, IxDyn>;
            let imag_result: Array<f64, IxDyn>;

            if b_shape.len() == 1 {
                let b0 = b.data[0];
                let b1 = b.data[1];
                let bi0 = b_i[0];
                let bi1 = b_i[1];

                let x0_r = inv_a11_r * b0 + inv_a12_r * b1 - inv_a11_i * bi0 - inv_a12_i * bi1;
                let x0_i = inv_a11_i * b0 + inv_a12_i * b1 + inv_a11_r * bi0 + inv_a12_r * bi1;
                let x1_r = inv_a21_r * b0 + inv_a22_r * b1 - inv_a21_i * bi0 - inv_a22_i * bi1;
                let x1_i = inv_a21_i * b0 + inv_a22_i * b1 + inv_a21_r * bi0 + inv_a22_r * bi1;

                real_result = Array::from_shape_vec(IxDyn(&[2]), vec![x0_r, x1_r])
                    .unwrap()
                    .into_dyn();
                imag_result = Array::from_shape_vec(IxDyn(&[2]), vec![x0_i, x1_i])
                    .unwrap()
                    .into_dyn();
            } else {
                let b_n = b_shape[1];
                let mut r_result = vec![0.0; 2 * b_n];
                let mut i_result = vec![0.0; 2 * b_n];

                for j in 0..b_n {
                    let bj0 = b.data[[0, j]];
                    let bj1 = b.data[[1, j]];
                    let bij0 = b_i[[0, j]];
                    let bij1 = b_i[[1, j]];

                    r_result[j] =
                        inv_a11_r * bj0 + inv_a12_r * bj1 - inv_a11_i * bij0 - inv_a12_i * bij1;
                    i_result[j] =
                        inv_a11_i * bj0 + inv_a12_i * bij0 + inv_a11_r * bj1 + inv_a12_r * bij1;
                    r_result[b_n + j] =
                        inv_a21_r * bj0 + inv_a22_r * bj1 - inv_a21_i * bij0 - inv_a22_i * bij1;
                    i_result[b_n + j] =
                        inv_a21_i * bj0 + inv_a22_i * bij0 + inv_a21_r * bj1 + inv_a22_r * bij1;
                }

                real_result = Array::from_shape_vec((2, b_n), r_result)
                    .unwrap()
                    .into_dyn();
                imag_result = Array::from_shape_vec((2, b_n), i_result)
                    .unwrap()
                    .into_dyn();
            }

            Ok(NdArray {
                imag: Some(imag_result),
                data: real_result,
            })
        }
        3 => {
            let a11 = a.data[[0, 0]];
            let a12 = a.data[[0, 1]];
            let a13 = a.data[[0, 2]];
            let a21 = a.data[[1, 0]];
            let a22 = a.data[[1, 1]];
            let a23 = a.data[[1, 2]];
            let a31 = a.data[[2, 0]];
            let a32 = a.data[[2, 1]];
            let a33 = a.data[[2, 2]];
            let det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
                + a13 * (a21 * a32 - a22 * a31);
            if det == 0.0 {
                return Err(PyValueError::new_err("Singular matrix"));
            }

            if !has_imag {
                return Ok(NdArray {
                    imag: None,
                    data: solve3x3(&a.data, &b.data),
                });
            }

            let zero_a = a.data.clone() * 0.0;
            let zero_b = b.data.clone() * 0.0;
            let a_i = a.imag.as_ref().unwrap_or(&zero_a);
            let b_i = b.imag.as_ref().unwrap_or(&zero_b);

            let ai11 = a_i[[0, 0]];
            let ai12 = a_i[[0, 1]];
            let ai13 = a_i[[0, 2]];
            let ai21 = a_i[[1, 0]];
            let ai22 = a_i[[1, 1]];
            let ai23 = a_i[[1, 2]];
            let ai31 = a_i[[2, 0]];
            let ai32 = a_i[[2, 1]];
            let ai33 = a_i[[2, 2]];

            let det_i = ai11 * (a22 * a33 - a23 * a32)
                + a11 * (ai22 * a33 + a22 * ai33 - ai23 * a32 - a23 * ai32)
                - ai12 * (a21 * a33 - a23 * a31)
                - a12 * (ai21 * a33 + a21 * ai33 - ai23 * a31 - a23 * ai31)
                + ai13 * (a21 * a32 - a22 * a31)
                + a13 * (ai21 * a32 + a21 * ai32 - ai22 * a31 - a22 * ai31);

            let det_sq = det * det + det_i * det_i;
            if det_sq < 1e-24 {
                return Err(PyValueError::new_err("Singular matrix"));
            }

            let inv_det_r = det / det_sq;
            let inv_det_i = -det_i / det_sq;

            let cof11_r = (a22 * a33 - a23 * a32) * inv_det_r
                - (ai22 * a33 + a22 * ai33 - ai23 * a32 - a23 * ai32) * inv_det_i;
            let cof11_i = (a22 * a33 - a23 * a32) * inv_det_i
                + (ai22 * a33 + a22 * ai33 - ai23 * a32 - a23 * ai32) * inv_det_r;
            let cof12_r = -(a21 * a33 - a23 * a31) * inv_det_r
                + (ai21 * a33 + a21 * ai33 - ai23 * a31 - a23 * ai31) * inv_det_i;
            let cof12_i = -(a21 * a33 - a23 * a31) * inv_det_i
                - (ai21 * a33 + a21 * ai33 - ai23 * a31 - a23 * ai31) * inv_det_r;
            let cof13_r = (a21 * a32 - a22 * a31) * inv_det_r
                - (ai21 * a32 + a21 * ai32 - ai22 * a31 - a22 * ai31) * inv_det_i;
            let cof13_i = (a21 * a32 - a22 * a31) * inv_det_i
                + (ai21 * a32 + a21 * ai32 - ai22 * a31 - a22 * ai31) * inv_det_r;
            let cof21_r = -(a12 * a33 - a13 * a32) * inv_det_r
                + (ai12 * a33 + a12 * ai33 - ai13 * a32 - a13 * ai32) * inv_det_i;
            let cof21_i = -(a12 * a33 - a13 * a32) * inv_det_i
                - (ai12 * a33 + a12 * ai33 - ai13 * a32 - a13 * ai32) * inv_det_r;
            let cof22_r = (a11 * a33 - a13 * a31) * inv_det_r
                - (ai11 * a33 + a11 * ai33 - ai13 * a31 - a13 * ai31) * inv_det_i;
            let cof22_i = (a11 * a33 - a13 * a31) * inv_det_i
                + (ai11 * a33 + a11 * ai33 - ai13 * a31 - a13 * ai31) * inv_det_r;
            let cof23_r = -(a11 * a32 - a12 * a31) * inv_det_r
                + (ai11 * a32 + a11 * ai32 - ai12 * a31 - a12 * ai31) * inv_det_i;
            let cof23_i = -(a11 * a32 - a12 * a31) * inv_det_i
                - (ai11 * a32 + a11 * ai32 - ai12 * a31 - a12 * ai31) * inv_det_r;
            let cof31_r = (a12 * a23 - a13 * a22) * inv_det_r
                - (ai12 * a23 + a12 * ai23 - ai13 * a22 - a13 * ai22) * inv_det_i;
            let cof31_i = (a12 * a23 - a13 * a22) * inv_det_i
                + (ai12 * a23 + a12 * ai23 - ai13 * a22 - a13 * ai22) * inv_det_r;
            let cof32_r = -(a11 * a23 - a13 * a21) * inv_det_r
                + (ai11 * a23 + a11 * ai23 - ai13 * a21 - a13 * ai21) * inv_det_i;
            let cof32_i = -(a11 * a23 - a13 * a21) * inv_det_i
                - (ai11 * a23 + a11 * ai23 - ai13 * a21 - a13 * ai21) * inv_det_r;
            let cof33_r = (a11 * a22 - a12 * a21) * inv_det_r
                - (ai11 * a22 + a11 * ai22 - ai12 * a21 - a12 * ai21) * inv_det_i;
            let cof33_i = (a11 * a22 - a12 * a21) * inv_det_i
                + (ai11 * a22 + a11 * ai22 - ai12 * a21 - a12 * ai21) * inv_det_r;

            let b_shape = b.data.shape().to_vec();
            let real_result: Array<f64, IxDyn>;
            let imag_result: Array<f64, IxDyn>;

            if b_shape.len() == 1 {
                let b0 = b.data[0];
                let b1 = b.data[1];
                let b2 = b.data[2];
                let bi0 = b_i[0];
                let bi1 = b_i[1];
                let bi2 = b_i[2];

                let x0_r = cof11_r * b0 + cof12_r * b1 + cof13_r * b2
                    - cof11_i * bi0
                    - cof12_i * bi1
                    - cof13_i * bi2;
                let x0_i = cof11_i * b0
                    + cof12_i * b1
                    + cof13_i * b2
                    + cof11_r * bi0
                    + cof12_r * bi1
                    + cof13_r * bi2;
                let x1_r = cof21_r * b0 + cof22_r * b1 + cof23_r * b2
                    - cof21_i * bi0
                    - cof22_i * bi1
                    - cof23_i * bi2;
                let x1_i = cof21_i * b0
                    + cof22_i * b1
                    + cof23_i * b2
                    + cof21_r * bi0
                    + cof22_r * bi1
                    + cof23_r * bi2;
                let x2_r = cof31_r * b0 + cof32_r * b1 + cof33_r * b2
                    - cof31_i * bi0
                    - cof32_i * bi1
                    - cof33_i * bi2;
                let x2_i = cof31_i * b0
                    + cof32_i * b1
                    + cof33_i * b2
                    + cof31_r * bi0
                    + cof32_r * bi1
                    + cof33_r * bi2;

                real_result = Array::from_shape_vec(IxDyn(&[3]), vec![x0_r, x1_r, x2_r])
                    .unwrap()
                    .into_dyn();
                imag_result = Array::from_shape_vec(IxDyn(&[3]), vec![x0_i, x1_i, x2_i])
                    .unwrap()
                    .into_dyn();
            } else {
                let b_n = b_shape[1];
                let mut r_result = vec![0.0; 3 * b_n];
                let mut i_result = vec![0.0; 3 * b_n];

                for j in 0..b_n {
                    let bj0 = b.data[[0, j]];
                    let bj1 = b.data[[1, j]];
                    let bj2 = b.data[[2, j]];
                    let bij0 = b_i[[0, j]];
                    let bij1 = b_i[[1, j]];
                    let bij2 = b_i[[2, j]];

                    r_result[j] = cof11_r * bj0 + cof12_r * bj1 + cof13_r * bj2
                        - cof11_i * bij0
                        - cof12_i * bij1
                        - cof13_i * bij2;
                    i_result[j] = cof11_i * bj0
                        + cof12_i * bij0
                        + cof13_i * bij2
                        + cof11_r * bij1
                        + cof12_r * bj2
                        + cof13_r * bj1;
                    r_result[b_n + j] = cof21_r * bj0 + cof22_r * bj1 + cof23_r * bj2
                        - cof21_i * bij0
                        - cof22_i * bij1
                        - cof23_i * bij2;
                    i_result[b_n + j] = cof21_i * bj0
                        + cof22_i * bij0
                        + cof23_i * bij2
                        + cof21_r * bij1
                        + cof22_r * bj2
                        + cof23_r * bj1;
                    r_result[2 * b_n + j] = cof31_r * bj0 + cof32_r * bj1 + cof33_r * bj2
                        - cof31_i * bij0
                        - cof32_i * bij1
                        - cof33_i * bij2;
                    i_result[2 * b_n + j] = cof31_i * bj0
                        + cof32_i * bij0
                        + cof33_i * bij2
                        + cof31_r * bij1
                        + cof32_r * bj2
                        + cof33_r * bj1;
                }

                real_result = Array::from_shape_vec((3, b_n), r_result)
                    .unwrap()
                    .into_dyn();
                imag_result = Array::from_shape_vec((3, b_n), i_result)
                    .unwrap()
                    .into_dyn();
            }

            Ok(NdArray {
                imag: Some(imag_result),
                data: real_result,
            })
        }
        _ => Err(PyValueError::new_err(
            "solve only supports 2x2 and 3x3 matrices",
        )),
    }
}

#[pyfunction]
fn eig(a: &NdArray) -> PyResult<(NdArray, NdArray)> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("eig requires a square matrix"));
    }
    let n = shape[0];
    match n {
        2 => {
            let a11 = a.data[[0, 0]];
            let a12 = a.data[[0, 1]];
            let a21 = a.data[[1, 0]];
            let a22 = a.data[[1, 1]];
            let tr = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = tr * tr - 4.0 * det;
            if disc < 0.0 {
                return Err(PyValueError::new_err(
                    "eig does not support complex eigenvalues",
                ));
            }
            let (evals, evecs) = eig2x2(&a.data);
            Ok((
                NdArray {
                    imag: None,
                    data: evals,
                },
                NdArray {
                    imag: None,
                    data: evecs,
                },
            ))
        }
        3 => {
            let a11 = a.data[[0, 0]];
            let a12 = a.data[[0, 1]];
            let a13 = a.data[[0, 2]];
            let a21 = a.data[[1, 0]];
            let a22 = a.data[[1, 1]];
            let a23 = a.data[[1, 2]];
            let a31 = a.data[[2, 0]];
            let a32 = a.data[[2, 1]];
            let a33 = a.data[[2, 2]];
            let tr = a11 + a22 + a33;
            let m_val = (a11 * a22 - a12 * a21) + (a11 * a33 - a13 * a31) + (a22 * a33 - a23 * a32);
            let d = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
                + a13 * (a21 * a32 - a22 * a31);
            let p = -tr;
            let q = m_val;
            let r = -d;
            let p2 = p * p;
            let alpha = q - p2 / 3.0;
            let beta = 2.0 * p2 * p / 27.0 - p * q / 3.0 + r;
            let disc = beta * beta / 4.0 + alpha * alpha * alpha / 27.0;
            if disc > 1e-10 {
                return Err(PyValueError::new_err(
                    "eig does not support complex eigenvalues",
                ));
            }
            let (evals, evecs) = eig3x3(&a.data);
            Ok((
                NdArray {
                    imag: None,
                    data: evals,
                },
                NdArray {
                    imag: None,
                    data: evecs,
                },
            ))
        }
        _ => Err(PyValueError::new_err(
            "eig only supports 2x2 and 3x3 matrices",
        )),
    }
}

#[pyfunction]
fn eigvals(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("eigvals requires a square matrix"));
    }
    let n = shape[0];
    match n {
        2 => {
            let a11 = a.data[[0, 0]];
            let a12 = a.data[[0, 1]];
            let a21 = a.data[[1, 0]];
            let a22 = a.data[[1, 1]];
            let tr = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = tr * tr - 4.0 * det;
            if disc < 0.0 {
                return Err(PyValueError::new_err(
                    "eigvals does not support complex eigenvalues",
                ));
            }
            let sqrt_disc = disc.sqrt();
            let lambda1 = (tr + sqrt_disc) / 2.0;
            let lambda2 = (tr - sqrt_disc) / 2.0;
            let arr = Array::from_shape_vec(IxDyn(&[2]), vec![lambda1, lambda2])
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray {
                imag: None,
                data: arr,
            })
        }
        3 => {
            let a11 = a.data[[0, 0]];
            let a12 = a.data[[0, 1]];
            let a13 = a.data[[0, 2]];
            let a21 = a.data[[1, 0]];
            let a22 = a.data[[1, 1]];
            let a23 = a.data[[1, 2]];
            let a31 = a.data[[2, 0]];
            let a32 = a.data[[2, 1]];
            let a33 = a.data[[2, 2]];
            let tr = a11 + a22 + a33;
            let m_val = (a11 * a22 - a12 * a21) + (a11 * a33 - a13 * a31) + (a22 * a33 - a23 * a32);
            let d = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
                + a13 * (a21 * a32 - a22 * a31);
            let p = -tr;
            let q = m_val;
            let r = -d;
            let p2 = p * p;
            let alpha = q - p2 / 3.0;
            let beta = 2.0 * p2 * p / 27.0 - p * q / 3.0 + r;
            let disc = beta * beta / 4.0 + alpha * alpha * alpha / 27.0;
            if disc > 1e-10 {
                return Err(PyValueError::new_err(
                    "eigvals does not support complex eigenvalues",
                ));
            }
            let sqrt_neg_alpha_3 = (-alpha / 3.0).sqrt();
            let cos_arg = (-beta / (2.0 * sqrt_neg_alpha_3.powi(3))).clamp(-1.0, 1.0);
            let theta = cos_arg.acos() / 3.0;
            let two_sqrt = 2.0 * sqrt_neg_alpha_3;
            let pi = std::f64::consts::PI;
            let t0 = two_sqrt * theta.cos();
            let t1 = two_sqrt * (theta - 2.0 * pi / 3.0).cos();
            let t2 = two_sqrt * (theta - 4.0 * pi / 3.0).cos();
            let shift = p / 3.0;
            let eigenvalues = vec![t0 - shift, t1 - shift, t2 - shift];
            let arr = Array::from_shape_vec(IxDyn(&[3]), eigenvalues)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray {
                imag: None,
                data: arr,
            })
        }
        _ => Err(PyValueError::new_err(
            "eigvals only supports 2x2 and 3x3 matrices",
        )),
    }
}

#[pyfunction]
fn svd(a: &NdArray) -> PyResult<(NdArray, NdArray, NdArray)> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != 2 || shape[1] != 2 {
        return Err(PyValueError::new_err("svd only supports 2x2 matrices"));
    }
    let a11 = a.data[[0, 0]];
    let a12 = a.data[[0, 1]];
    let a21 = a.data[[1, 0]];
    let a22 = a.data[[1, 1]];
    let b11 = a11 * a11 + a21 * a21;
    let b12 = a11 * a12 + a21 * a22;
    let b22 = a12 * a12 + a22 * a22;
    let tr = b11 + b22;
    let det = b11 * b22 - b12 * b12;
    let disc = tr * tr - 4.0 * det;
    let sqrt_disc = disc.sqrt();
    let lambda1 = (tr + sqrt_disc) / 2.0;
    let lambda2 = (tr - sqrt_disc) / 2.0;
    let sigma1 = lambda1.sqrt();
    let sigma2 = lambda2.sqrt();
    let (v1x, v1y) = if b12.abs() > 1e-12 {
        (b12, lambda1 - b11)
    } else {
        (1.0, 0.0)
    };
    let v1_norm = (v1x * v1x + v1y * v1y).sqrt();
    let (v1x, v1y) = (v1x / v1_norm, v1y / v1_norm);
    let (v2x, v2y) = if b12.abs() > 1e-12 {
        (b12, lambda2 - b11)
    } else {
        (0.0, 1.0)
    };
    let v2_norm = (v2x * v2x + v2y * v2y).sqrt();
    let (v2x, v2y) = (v2x / v2_norm, v2y / v2_norm);
    let u1x = (a11 * v1x + a12 * v1y) / sigma1;
    let u1y = (a21 * v1x + a22 * v1y) / sigma1;
    let u1_norm = (u1x * u1x + u1y * u1y).sqrt();
    let (u1x, u1y) = (u1x / u1_norm, u1y / u1_norm);
    let u2x = (a11 * v2x + a12 * v2y) / sigma2;
    let u2y = (a21 * v2x + a22 * v2y) / sigma2;
    let u2_norm = (u2x * u2x + u2y * u2y).sqrt();
    let (u2x, u2y) = (u2x / u2_norm, u2y / u2_norm);
    let u_data = vec![u1x, u2x, u1y, u2y];
    let s_data = vec![sigma1, 0.0, 0.0, sigma2];
    let vt_data = vec![v1x, v1y, v2x, v2y];
    let u_arr =
        Array::from_shape_vec((2, 2), u_data).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let s_arr =
        Array::from_shape_vec((2, 2), s_data).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let vt_arr =
        Array::from_shape_vec((2, 2), vt_data).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        NdArray {
            imag: None,
            data: u_arr.into_dyn(),
        },
        NdArray {
            imag: None,
            data: s_arr.into_dyn(),
        },
        NdArray {
            imag: None,
            data: vt_arr.into_dyn(),
        },
    ))
}

#[pyfunction]
fn qr(a: &NdArray) -> PyResult<(NdArray, NdArray)> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("qr requires a 2D array"));
    }
    let m = shape[0];
    let n = shape[1];
    if m != n || (m != 2 && m != 3) {
        return Err(PyValueError::new_err(
            "qr only supports 2x2 and 3x3 matrices",
        ));
    }
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(n);
    for j in 0..n {
        let mut col = Vec::with_capacity(m);
        for i in 0..m {
            col.push(a.data[[i, j]]);
        }
        cols.push(col);
    }
    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut r_vals = vec![0.0; n * n];
    for j in 0..n {
        let mut v = cols[j].clone();
        for i in 0..j {
            let dot: f64 = (0..m).map(|k| cols[j][k] * q_cols[i][k]).sum();
            r_vals[i * n + j] = dot;
            for k in 0..m {
                v[k] -= dot * q_cols[i][k];
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        r_vals[j * n + j] = norm;
        let q_col: Vec<f64> = v.into_iter().map(|x| x / norm).collect();
        q_cols.push(q_col);
    }
    let mut q_data = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..m {
            q_data[i * n + j] = q_cols[j][i];
        }
    }
    let q_arr =
        Array::from_shape_vec((m, n), q_data).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let r_arr =
        Array::from_shape_vec((n, n), r_vals).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        NdArray {
            imag: None,
            data: q_arr.into_dyn(),
        },
        NdArray {
            imag: None,
            data: r_arr.into_dyn(),
        },
    ))
}

#[pyfunction]
fn cholesky(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("cholesky requires a square matrix"));
    }
    let n = shape[0];
    match n {
        2 => {
            let l11 = a.data[[0, 0]].sqrt();
            let l21 = a.data[[1, 0]] / l11;
            let l22 = (a.data[[1, 1]] - l21 * l21).sqrt();
            let result = vec![l11, 0.0, l21, l22];
            let arr = Array::from_shape_vec((2, 2), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray {
                imag: None,
                data: arr.into_dyn(),
            })
        }
        3 => {
            let l11 = a.data[[0, 0]].sqrt();
            let l21 = a.data[[1, 0]] / l11;
            let l31 = a.data[[2, 0]] / l11;
            let l22 = (a.data[[1, 1]] - l21 * l21).sqrt();
            let l32 = (a.data[[2, 1]] - l31 * l21) / l22;
            let l33 = (a.data[[2, 2]] - l31 * l31 - l32 * l32).sqrt();
            let result = vec![l11, 0.0, 0.0, l21, l22, 0.0, l31, l32, l33];
            let arr = Array::from_shape_vec((3, 3), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray {
                imag: None,
                data: arr.into_dyn(),
            })
        }
        _ => Err(PyValueError::new_err(
            "cholesky only supports 2x2 and 3x3 matrices",
        )),
    }
}

#[pyfunction]
fn matrix_power(a: &NdArray, n: i32) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err(
            "matrix_power requires a square matrix",
        ));
    }
    let n_dim = shape[0];
    if n_dim != 2 && n_dim != 3 {
        return Err(PyValueError::new_err(
            "matrix_power only supports 2x2 and 3x3 matrices",
        ));
    }
    if n == 0 {
        let mut result = vec![0.0; n_dim * n_dim];
        for i in 0..n_dim {
            result[i * n_dim + i] = 1.0;
        }
        let arr = Array::from_shape_vec((n_dim, n_dim), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray {
            imag: None,
            data: arr.into_dyn(),
        });
    }
    if n > 0 {
        let mut result = a.data.clone();
        for _ in 1..n {
            result = matmul_impl(&result, &a.data);
        }
        Ok(NdArray {
            imag: None,
            data: result,
        })
    } else {
        let inv_arr = inv(&NdArray {
            imag: None,
            data: a.data.clone(),
        })?;
        let abs_n = (-n) as u32;
        let mut result = inv_arr.data.clone();
        for _ in 1..abs_n {
            result = matmul_impl(&result, &inv_arr.data);
        }
        Ok(NdArray {
            imag: None,
            data: result,
        })
    }
}

#[pyfunction]
fn pinv(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("pinv requires a 2D array"));
    }
    let m = shape[0];
    let n = shape[1];
    if m == n && m <= 3 {
        return inv(a);
    }
    if m == 2 && n == 2 {
        return inv(a);
    }
    let mut ata = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += a.data[[k, i]] * a.data[[k, j]];
            }
            ata[i * n + j] = sum;
        }
    }
    let ata_arr =
        Array::from_shape_vec((n, n), ata).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ata_nd = NdArray {
        imag: None,
        data: ata_arr.into_dyn(),
    };
    let ata_inv = inv(&ata_nd)?;
    let mut result = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += ata_inv.data[[i, k]] * a.data[[j, k]];
            }
            result[i * m + j] = sum;
        }
    }
    let arr =
        Array::from_shape_vec((n, m), result).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr.into_dyn(),
    })
}

#[pyfunction]
fn trace(a: &NdArray) -> PyResult<f64> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("trace requires a square matrix"));
    }
    let n = shape[0];
    let mut result = 0.0;
    for i in 0..n {
        result += a.data[[i, i]];
    }
    Ok(result)
}

#[pyfunction]
fn diagonal(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("diagonal requires a 2D array"));
    }
    let n = shape[0].min(shape[1]);
    let mut result = vec![0.0; n];
    for (i, r) in result.iter_mut().enumerate() {
        *r = a.data[[i, i]];
    }
    let arr = Array::from_shape_vec(IxDyn(&[n]), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

#[pyfunction]
fn svdvals(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("svdvals requires a 2D array"));
    }
    let m = shape[0];
    let n = shape[1];
    if m != n || (m != 2 && m != 3) {
        return Err(PyValueError::new_err(
            "svdvals only supports 2x2 and 3x3 matrices",
        ));
    }
    match m {
        2 => {
            let a11 = a.data[[0, 0]];
            let a12 = a.data[[0, 1]];
            let a21 = a.data[[1, 0]];
            let a22 = a.data[[1, 1]];
            let b11 = a11 * a11 + a21 * a21;
            let b12 = a11 * a12 + a21 * a22;
            let b22 = a12 * a12 + a22 * a22;
            let tr = b11 + b22;
            let det = b11 * b22 - b12 * b12;
            let disc = tr * tr - 4.0 * det;
            let sqrt_disc = disc.sqrt();
            let lambda1 = (tr + sqrt_disc) / 2.0;
            let lambda2 = (tr - sqrt_disc) / 2.0;
            let sigma1 = lambda1.sqrt();
            let sigma2 = lambda2.sqrt();
            let arr = Array::from_shape_vec(IxDyn(&[2]), vec![sigma1, sigma2])
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray {
                imag: None,
                data: arr,
            })
        }
        3 => {
            let a11 = a.data[[0, 0]];
            let a12 = a.data[[0, 1]];
            let a13 = a.data[[0, 2]];
            let a21 = a.data[[1, 0]];
            let a22 = a.data[[1, 1]];
            let a23 = a.data[[1, 2]];
            let a31 = a.data[[2, 0]];
            let a32 = a.data[[2, 1]];
            let a33 = a.data[[2, 2]];
            let b11 = a11 * a11 + a21 * a21 + a31 * a31;
            let b12 = a11 * a12 + a21 * a22 + a31 * a32;
            let b13 = a11 * a13 + a21 * a23 + a31 * a33;
            let b22 = a12 * a12 + a22 * a22 + a32 * a32;
            let b23 = a12 * a13 + a22 * a23 + a32 * a33;
            let b33 = a13 * a13 + a23 * a23 + a33 * a33;
            let tr = b11 + b22 + b33;
            let m_val = b11 * b22 - b12 * b12 + b11 * b33 - b13 * b13 + b22 * b33 - b23 * b23;
            let d = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b13 * b23)
                + b13 * (b12 * b23 - b22 * b13);
            let p = -tr;
            let q = m_val;
            let r = -d;
            let p2 = p * p;
            let alpha = q - p2 / 3.0;
            let beta = 2.0 * p2 * p / 27.0 - p * q / 3.0 + r;
            let disc = beta * beta / 4.0 + alpha * alpha * alpha / 27.0;
            if disc > 1e-10 {
                return Err(PyValueError::new_err(
                    "svdvals does not support complex singular values",
                ));
            }
            let sqrt_neg_alpha_3 = (-alpha / 3.0).sqrt();
            let cos_arg = (-beta / (2.0 * sqrt_neg_alpha_3.powi(3))).clamp(-1.0, 1.0);
            let theta = cos_arg.acos() / 3.0;
            let two_sqrt = 2.0 * sqrt_neg_alpha_3;
            let pi = std::f64::consts::PI;
            let t0 = two_sqrt * theta.cos();
            let t1 = two_sqrt * (theta - 2.0 * pi / 3.0).cos();
            let t2 = two_sqrt * (theta - 4.0 * pi / 3.0).cos();
            let shift = p / 3.0;
            let sigma1 = (t0 - shift).sqrt();
            let sigma2 = (t1 - shift).sqrt();
            let sigma3 = (t2 - shift).sqrt();
            let arr = Array::from_shape_vec(IxDyn(&[3]), vec![sigma1, sigma2, sigma3])
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray {
                imag: None,
                data: arr,
            })
        }
        _ => Err(PyValueError::new_err(
            "svdvals only supports 2x2 and 3x3 matrices",
        )),
    }
}

#[pyfunction]
fn eigh(a: &NdArray) -> PyResult<(NdArray, NdArray)> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("eigh requires a square matrix"));
    }
    let n = shape[0];
    if n != 2 && n != 3 {
        return Err(PyValueError::new_err(
            "eigh only supports 2x2 and 3x3 matrices",
        ));
    }
    eig(a)
}

#[pyfunction]
fn solve_banded(lower: usize, upper: usize, ab: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    let ab_shape = ab.data.shape().to_vec();
    let b_shape = b.data.shape().to_vec();

    if ab_shape.len() != 2 {
        return Err(PyValueError::new_err("ab must be a 2D array"));
    }
    if b_shape.len() != 1 && b_shape.len() != 2 {
        return Err(PyValueError::new_err("b must be 1D or 2D"));
    }

    let n = ab_shape[1];
    let m = ab_shape[0];

    if m != lower + upper + 1 {
        return Err(PyValueError::new_err(
            "ab rows must equal lower + upper + 1",
        ));
    }
    if b_shape[0] != n {
        return Err(PyValueError::new_err("b dimensions must match matrix size"));
    }

    if n != 2 && n != 3 {
        return Err(PyValueError::new_err(
            "solve_banded only supports 2x2 and 3x3 systems",
        ));
    }

    let mut full = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..m {
            let row_in_full = (j + i) as isize - upper as isize;
            if row_in_full >= 0 && row_in_full < n as isize {
                full[row_in_full as usize * n + j] = ab.data[[i, j]];
            }
        }
    }

    let full_arr =
        Array::from_shape_vec((n, n), full).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let full_nd = NdArray {
        imag: None,
        data: full_arr.into_dyn(),
    };

    solve(&full_nd, b)
}

/// 复数幅值（hypot，与 Python abs(complex) 一致）
#[inline]
fn gj_mag(z: (f64, f64)) -> f64 {
    z.0.hypot(z.1)
}

/// 复数除法（朴素公式 (ac+bd)/m + (bc-ad)/m，m = c²+d²）
#[inline]
fn gj_cdiv(x: (f64, f64), y: (f64, f64)) -> (f64, f64) {
    let m = y.0 * y.0 + y.1 * y.1;
    ((x.0 * y.0 + x.1 * y.1) / m, (x.1 * y.0 - x.0 * y.1) / m)
}

/// a - f*w（f*w 先乘后减，运算次序与 Python complex 一致）
#[inline]
fn gj_csub_mul(a: (f64, f64), f: (f64, f64), w: (f64, f64)) -> (f64, f64) {
    (a.0 - (f.0 * w.0 - f.1 * w.1), a.1 - (f.0 * w.1 + f.1 * w.0))
}

/// 高斯-约当消元核心：解 Ax=b（列主元，任意 n、k 个右端）。
/// b=None → 右端为单位阵（即矩阵求逆）。语义与 Python
/// _gauss_jordan_solve/_gauss_jordan_inv 对齐：eps = n * max|A| * 2.22e-16
/// （max|A|==0 → 1e-10，且 max|A| 只统计 A）；主元为严格大于扫描
/// （并列取最先，NaN 不会胜过既有最优）；奇异 → Err("Singular matrix")；
/// 消元跳过 factor==0 的行。元素统一 (re, im) 对，实数输入 im=0。
#[pyfunction]
#[pyo3(signature = (a, b=None))]
fn gj_solve_core(a: &NdArray, b: Option<&NdArray>) -> PyResult<NdArray> {
    let a_shape: Vec<usize> = a.data.shape().to_vec();
    if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
        return Err(PyValueError::new_err(
            "gj_solve_core: a must be square (n, n)",
        ));
    }
    let n = a_shape[0];
    let k = match &b {
        None => n,
        Some(b) => match b.data.shape().to_vec().as_slice() {
            [m] if *m == n => 1,
            [m, kk] if *m == n => *kk,
            _ => {
                return Err(PyValueError::new_err(
                    "gj_solve_core: b must be (n,) or (n, k)",
                ));
            }
        },
    };
    let complex = a.imag.is_some() || b.is_some_and(|x| x.imag.is_some());
    // 读入（逻辑 C 序展平，任意布局安全）
    let mut work: Vec<(f64, f64)> = a.data.iter().map(|&r| (r, 0.0)).collect();
    if let Some(im) = &a.imag {
        for (w, &i) in work.iter_mut().zip(im.iter()) {
            w.1 = i;
        }
    }
    let mut rhs: Vec<(f64, f64)>;
    if let Some(b) = b {
        rhs = b.data.iter().map(|&r| (r, 0.0)).collect();
        if let Some(im) = &b.imag {
            for (w, &i) in rhs.iter_mut().zip(im.iter()) {
                w.1 = i;
            }
        }
    } else {
        rhs = (0..n * n)
            .map(|p| {
                let (i, j) = (p / n, p % n);
                (if i == j { 1.0 } else { 0.0 }, 0.0)
            })
            .collect();
    }
    let max_abs = work.iter().map(|z| gj_mag(*z)).fold(0.0f64, f64::max);
    let eps = if max_abs != 0.0 {
        n as f64 * max_abs * 2.220446049250313e-16
    } else {
        1e-10
    };
    for col in 0..n {
        let mut pivot = col;
        let mut best = gj_mag(work[col * n + col]);
        for r in (col + 1)..n {
            let v = gj_mag(work[r * n + col]);
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < eps {
            return Err(PyValueError::new_err("Singular matrix"));
        }
        if pivot != col {
            for j in 0..n {
                work.swap(col * n + j, pivot * n + j);
            }
            for j in 0..k {
                rhs.swap(col * k + j, pivot * k + j);
            }
        }
        let pv = work[col * n + col];
        for j in 0..n {
            work[col * n + j] = gj_cdiv(work[col * n + j], pv);
        }
        for j in 0..k {
            rhs[col * k + j] = gj_cdiv(rhs[col * k + j], pv);
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = work[r * n + col];
            if factor.0 == 0.0 && factor.1 == 0.0 {
                continue;
            }
            for j in 0..n {
                let w = work[col * n + j];
                work[r * n + j] = gj_csub_mul(work[r * n + j], factor, w);
            }
            for j in 0..k {
                let w = rhs[col * k + j];
                rhs[r * k + j] = gj_csub_mul(rhs[r * k + j], factor, w);
            }
        }
    }
    let out_real: Vec<f64> = rhs.iter().map(|z| z.0).collect();
    let data = Array::from_shape_vec(IxDyn(&[n, k]), out_real)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let imag = if complex {
        let v: Vec<f64> = rhs.iter().map(|z| z.1).collect();
        Some(
            Array::from_shape_vec(IxDyn(&[n, k]), v)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        )
    } else {
        None
    };
    Ok(NdArray { data, imag })
}

/// 列内积（实数，C 序 flat 访问）
fn jcol_dot(work: &[f64], mm: usize, nn: usize, p: usize, q: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..mm {
        s += work[i * nn + p] * work[i * nn + q];
    }
    s
}

/// 单侧 Jacobi SVD 共享实现（仅实数），语义与 Python _svd_jacobi
/// （实数分支）逐条对齐：m<n 转置为 A^T 分解、末端回映；100 sweeps，
/// 收敛 off <= 1e-28*sum(d²)+1e-300；ag <= 1e-15*sqrt(alpha*beta) 跳过
/// 旋转；半角 t/z 公式（[cp,cq] -> [c*cp+s*cq, -s*cp+c*cq]）；sig 降序
/// 稳定排序（并列/NaN 保持原序）；U 对 sp <= 1e-15*smax 的列留零；
/// full 且 mm>nn 时 Gram-Schmidt 补基（ne < 1e-8 跳过）。
/// 返回 (u flat, u shape, sig, vh flat, vh shape)：full → u (m,m)、
/// vh (n,n)；reduced → u (m,k)、vh (k,n)。
fn svd_jacobi_impl(
    data: &[f64],
    m: usize,
    n: usize,
    full_matrices: bool,
) -> (Vec<f64>, (usize, usize), Vec<f64>, Vec<f64>, (usize, usize)) {
    let k = m.min(n);
    let transposed = m < n;
    let (mm, nn) = if transposed { (n, m) } else { (m, n) };
    let mut work: Vec<f64> = if transposed {
        let mut w = vec![0.0; mm * nn];
        for i in 0..m {
            for j in 0..n {
                w[j * nn + i] = data[i * n + j];
            }
        }
        w
    } else {
        data.to_vec()
    };
    let mut v_acc: Vec<f64> = vec![0.0; nn * nn];
    for i in 0..nn {
        v_acc[i * nn + i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        let diag: Vec<f64> = (0..nn)
            .map(|p| jcol_dot(&work, mm, nn, p, p).abs())
            .collect();
        for p in 0..nn {
            for q in (p + 1)..nn {
                let g = jcol_dot(&work, mm, nn, p, q);
                off += g * g;
            }
        }
        if off <= 1e-28 * diag.iter().map(|d| d * d).sum::<f64>() + 1e-300 {
            break;
        }
        for p in 0..nn {
            for q in (p + 1)..nn {
                let alpha = jcol_dot(&work, mm, nn, p, p).abs();
                let beta = jcol_dot(&work, mm, nn, q, q).abs();
                let gamma = jcol_dot(&work, mm, nn, p, q);
                let ag = gamma.abs();
                if ag <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                let z = (beta - alpha) / (2.0 * gamma);
                let t = if z >= 0.0 { -1.0 } else { 1.0 } / (z.abs() + (1.0 + z * z).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for i in 0..mm {
                    let wp = work[i * nn + p];
                    let wq = work[i * nn + q];
                    work[i * nn + p] = c * wp + s * wq;
                    work[i * nn + q] = -s * wp + c * wq;
                }
                for i in 0..nn {
                    let vp = v_acc[i * nn + p];
                    let vq = v_acc[i * nn + q];
                    v_acc[i * nn + p] = c * vp + s * vq;
                    v_acc[i * nn + q] = -s * vp + c * vq;
                }
            }
        }
    }
    let sig_raw: Vec<f64> = (0..nn)
        .map(|p| jcol_dot(&work, mm, nn, p, p).abs().sqrt())
        .collect();
    let mut order: Vec<usize> = (0..nn).collect();
    order.sort_by(|&x, &y| {
        sig_raw[y]
            .partial_cmp(&sig_raw[x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sig: Vec<f64> = order.iter().map(|&p| sig_raw[p]).collect();
    let smax = sig.first().copied().unwrap_or(0.0);
    let mut u_part: Vec<f64> = vec![0.0; mm * nn];
    for (new_p, &old_p) in order.iter().enumerate() {
        let sp = sig[new_p];
        if sp > 1e-15 * smax {
            for i in 0..mm {
                u_part[i * nn + new_p] = work[i * nn + old_p] / sp;
            }
        }
    }
    let mut vh_mat: Vec<f64> = vec![0.0; nn * nn];
    for i in 0..nn {
        for j in 0..nn {
            vh_mat[i * nn + j] = v_acc[j * nn + order[i]];
        }
    }
    // u_part 补零至 (mm, mm)，供 full 补基与转置分支索引
    let mut u_pad: Vec<f64> = vec![0.0; mm * mm];
    for i in 0..mm {
        for j in 0..nn {
            u_pad[i * mm + j] = u_part[i * nn + j];
        }
    }
    if full_matrices && mm > nn {
        let mut basis: Vec<Vec<f64>> = (0..nn)
            .map(|p| (0..mm).map(|i| u_pad[i * mm + p]).collect())
            .collect();
        for i in 0..mm {
            let mut e: Vec<f64> = (0..mm).map(|r| if r == i { 1.0 } else { 0.0 }).collect();
            for b in basis.iter() {
                let proj: f64 = (0..mm).map(|r| b[r] * e[r]).sum();
                for r in 0..mm {
                    e[r] -= proj * b[r];
                }
            }
            let ne = (0..mm).map(|r| e[r] * e[r]).sum::<f64>().sqrt();
            if ne < 1e-8 {
                continue;
            }
            basis.push(e.iter().map(|&x| x / ne).collect());
            if basis.len() == mm {
                break;
            }
        }
        for (col_idx, col) in basis.iter().enumerate().skip(nn) {
            for i in 0..mm {
                u_pad[i * mm + col_idx] = col[i];
            }
        }
    }
    let (u_full, vh_full): (Vec<f64>, Vec<f64>) = if transposed {
        let mut u_out = vec![0.0; nn * nn];
        for i in 0..nn {
            for j in 0..nn {
                u_out[j * nn + i] = vh_mat[i * nn + j];
            }
        }
        let mut vh_out = vec![0.0; mm * mm];
        for i in 0..mm {
            for j in 0..mm {
                vh_out[i * mm + j] = u_pad[j * mm + i];
            }
        }
        (u_out, vh_out)
    } else {
        (u_pad, vh_mat)
    };
    let mut u_out = u_full;
    let mut vh_out = vh_full;
    let mut u_shape = (m, m);
    let mut v_shape = (n, n);
    if !full_matrices {
        let nu: Vec<f64> = (0..m)
            .flat_map(|i| u_out[i * m..i * m + k].iter().copied())
            .collect();
        let nv = vh_out[..k * n].to_vec();
        u_shape = (m, k);
        v_shape = (k, n);
        u_out = nu;
        vh_out = nv;
    }
    (u_out, u_shape, sig, vh_out, v_shape)
}

/// SVD 输入校验：2-d 非空实数矩阵；复数/空 → Err（Python 侧回退
/// 纯 Python Jacobi）。返回 (m, n, flat data)。
fn svd_jacobi_check(a: &NdArray) -> PyResult<(usize, usize, Vec<f64>)> {
    let shape: Vec<usize> = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(PyValueError::new_err(
            "svd_jacobi: a must be non-empty (m, n)",
        ));
    }
    if a.imag.is_some() {
        return Err(PyValueError::new_err("svd_jacobi: complex unsupported"));
    }
    let data: Vec<f64> = a.data.iter().copied().collect();
    Ok((shape[0], shape[1], data))
}

/// SVD 奇异值（仅实数，任意 m×n）：1-d 降序 sig（len = min(m,n)）。
/// 复数/空矩阵返回 Err，由 Python 侧回退纯 Python Jacobi。
#[pyfunction]
fn svd_jacobi_sig(a: &NdArray) -> PyResult<NdArray> {
    let (m, n, data) = svd_jacobi_check(a)?;
    let (_, _, sig, _, _) = svd_jacobi_impl(&data, m, n, false);
    let arr = Array::from_shape_vec(IxDyn(&[sig.len()]), sig)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        data: arr,
        imag: None,
    })
}

/// 完整 SVD（仅实数）：(u, s, vh)——full：u (m,m)、vh (n,n)；
/// reduced：u (m,k)、vh (k,n)，s 降序 (k,)。语义与 Python
/// _svd_jacobi（实数分支）一致；复数/空矩阵返回 Err。
#[pyfunction]
fn svd_jacobi_uv(a: &NdArray, full_matrices: bool) -> PyResult<(NdArray, NdArray, NdArray)> {
    let (m, n, data) = svd_jacobi_check(a)?;
    let (u, us, sig, vh, vs) = svd_jacobi_impl(&data, m, n, full_matrices);
    let u_arr = Array::from_shape_vec(IxDyn(&[us.0, us.1]), u)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let s_arr = Array::from_shape_vec(IxDyn(&[sig.len()]), sig)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let vh_arr = Array::from_shape_vec(IxDyn(&[vs.0, vs.1]), vh)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        NdArray {
            data: u_arr,
            imag: None,
        },
        NdArray {
            data: s_arr,
            imag: None,
        },
        NdArray {
            data: vh_arr,
            imag: None,
        },
    ))
}

pub fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(vdot, m)?)?;
    m.add_function(wrap_pyfunction!(inner, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(inv, m)?)?;
    m.add_function(wrap_pyfunction!(det, m)?)?;
    m.add_function(wrap_pyfunction!(norm, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(eig, m)?)?;
    m.add_function(wrap_pyfunction!(eigvals, m)?)?;
    m.add_function(wrap_pyfunction!(svd, m)?)?;
    m.add_function(wrap_pyfunction!(qr, m)?)?;
    m.add_function(wrap_pyfunction!(cholesky, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_power, m)?)?;
    m.add_function(wrap_pyfunction!(pinv, m)?)?;
    m.add_function(wrap_pyfunction!(trace, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(svdvals, m)?)?;
    m.add_function(wrap_pyfunction!(eigh, m)?)?;
    m.add_function(wrap_pyfunction!(solve_banded, m)?)?;
    m.add_function(wrap_pyfunction!(gj_solve_core, m)?)?;
    m.add_function(wrap_pyfunction!(svd_jacobi_sig, m)?)?;
    m.add_function(wrap_pyfunction!(svd_jacobi_uv, m)?)?;
    Ok(())
}
