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
    // b12 == 0（A^T A 已对角）时特征向量是 e1/e2：λ1 对应 max(b11, b22)
    // 的方向——固定取 e1 会使 U/Vh 与降序 s 不配套（如 diag(2,4)）
    let (v1x, v1y) = if b12.abs() > 1e-12 {
        (b12, lambda1 - b11)
    } else if b11 >= b22 {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let v1_norm = (v1x * v1x + v1y * v1y).sqrt();
    let (v1x, v1y) = (v1x / v1_norm, v1y / v1_norm);
    let (v2x, v2y) = if b12.abs() > 1e-12 {
        (b12, lambda2 - b11)
    } else if b11 >= b22 {
        (0.0, 1.0)
    } else {
        (1.0, 0.0)
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

/// 复共轭
#[inline]
fn cj(z: (f64, f64)) -> (f64, f64) {
    (z.0, -z.1)
}

/// 复 × 实标量
#[inline]
fn sc_mul(z: (f64, f64), r: f64) -> (f64, f64) {
    (z.0 * r, z.1 * r)
}

/// conj(a) * b
#[inline]
fn cj_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 + a.1 * b.1, a.0 * b.1 - a.1 * b.0)
}

/// 复 ÷ 实标量
#[inline]
fn cdiv_r(z: (f64, f64), r: f64) -> (f64, f64) {
    (z.0 / r, z.1 / r)
}

/// 复除复（CPython c_quot Smith 算法语义，与 Python complex/complex 位级一致）
fn cdiv_c(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    if b.1.abs() <= b.0.abs() {
        let ratio = b.1 / b.0;
        let den = b.0 + ratio * b.1;
        ((a.0 + a.1 * ratio) / den, (a.1 - a.0 * ratio) / den)
    } else {
        let ratio = b.0 / b.1;
        let den = b.1 + ratio * b.0;
        ((a.0 * ratio + a.1) / den, (a.1 * ratio - a.0) / den)
    }
}

/// 复乘法（运算次序与 Python complex 一致）
#[inline]
fn gj_cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// 复列内积 Σ conj(w[i,p]) * w[i,q]
fn jcol_dot_c(work: &[(f64, f64)], mm: usize, nn: usize, p: usize, q: usize) -> (f64, f64) {
    let mut s = (0.0, 0.0);
    for i in 0..mm {
        let t = cj_mul(work[i * nn + p], work[i * nn + q]);
        s = (s.0 + t.0, s.1 + t.1);
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

/// 单侧 Jacobi SVD 复数共享实现。语义与 Python _svd_jacobi 复数分支
/// 逐条对齐：m<n 转置为 A^T（纯转置）分解、末端回映；100 sweeps，收敛
/// off <= 1e-28*sum(d²)+1e-300（off = Σ|交叉|²，d = |列自内积|）；
/// ag <= 1e-15*sqrt(alpha*beta) 跳过旋转；先将交叉项相位归一到 |gamma|
/// （phase = conj(gamma)/|gamma|，work 列 q 与 v_acc 列 q 乘 phase），
/// 其后半角 t/z 与旋转同实数版（c, s 实数）；sig 降序稳定排序
/// （并列/NaN 保持原序）；U 对 sp <= 1e-15*smax 的列留零；Vh = V^H
/// （共轭转置 + 重排）；full 且 mm>nn 时复 Gram-Schmidt 补基
/// （ne < 1e-8 跳过）；转置分支 u_out = Vh^T、vh_out = U^T（纯转置）；
/// 非有限复数边缘 |phase|<=0（Python 会在 `z >= 0` 抛 TypeError）→
/// 返回 Err，由 Python 兜底复现同一异常。
/// 返回 (u re, u im, u shape, sig, vh re, vh im, vh shape)。
fn svd_jacobi_impl_c(
    re: &[f64],
    im: &[f64],
    m: usize,
    n: usize,
    full_matrices: bool,
) -> PyResult<(
    Vec<f64>,
    Vec<f64>,
    (usize, usize),
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    (usize, usize),
)> {
    let k = m.min(n);
    let transposed = m < n;
    let (mm, nn) = if transposed { (n, m) } else { (m, n) };
    let mut work: Vec<(f64, f64)> = if transposed {
        let mut w = vec![(0.0, 0.0); mm * nn];
        for i in 0..m {
            for j in 0..n {
                w[j * nn + i] = (re[i * n + j], im[i * n + j]);
            }
        }
        w
    } else {
        re.iter().zip(im.iter()).map(|(&r, &i)| (r, i)).collect()
    };
    let mut v_acc: Vec<(f64, f64)> = vec![(0.0, 0.0); nn * nn];
    for i in 0..nn {
        v_acc[i * nn + i] = (1.0, 0.0);
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        let diag: Vec<f64> = (0..nn)
            .map(|p| gj_mag(jcol_dot_c(&work, mm, nn, p, p)))
            .collect();
        for p in 0..nn {
            for q in (p + 1)..nn {
                let g = gj_mag(jcol_dot_c(&work, mm, nn, p, q));
                off += g * g;
            }
        }
        if off <= 1e-28 * diag.iter().map(|d| d * d).sum::<f64>() + 1e-300 {
            break;
        }
        for p in 0..nn {
            for q in (p + 1)..nn {
                let alpha = gj_mag(jcol_dot_c(&work, mm, nn, p, p));
                let beta = gj_mag(jcol_dot_c(&work, mm, nn, q, q));
                let gamma = jcol_dot_c(&work, mm, nn, p, q);
                let ag = gj_mag(gamma);
                if ag <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                let phase = cdiv_r(cj(gamma), ag);
                if ag <= 0.0 || gj_mag(phase) <= 0.0 {
                    // 非有限边缘：Python 对复数 `z >= 0` 抛 TypeError，
                    // 这里统一 Err，Python 兜底复现同一异常
                    return Err(PyValueError::new_err(
                        "svd_jacobi: non-finite complex entry",
                    ));
                }
                for i in 0..mm {
                    work[i * nn + q] = gj_cmul(work[i * nn + q], phase);
                }
                for i in 0..nn {
                    v_acc[i * nn + q] = gj_cmul(v_acc[i * nn + q], phase);
                }
                let z = (beta - alpha) / (2.0 * ag);
                let t = if z >= 0.0 { -1.0 } else { 1.0 } / (z.abs() + (1.0 + z * z).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for i in 0..mm {
                    let wp = work[i * nn + p];
                    let wq = work[i * nn + q];
                    let cwp = sc_mul(wp, c);
                    let cwq = sc_mul(wq, c);
                    let swp = sc_mul(wp, s);
                    let swq = sc_mul(wq, s);
                    work[i * nn + p] = (cwp.0 + swq.0, cwp.1 + swq.1);
                    work[i * nn + q] = (cwq.0 - swp.0, cwq.1 - swp.1);
                }
                for i in 0..nn {
                    let vp = v_acc[i * nn + p];
                    let vq = v_acc[i * nn + q];
                    let cvp = sc_mul(vp, c);
                    let cvq = sc_mul(vq, c);
                    let svp = sc_mul(vp, s);
                    let svq = sc_mul(vq, s);
                    v_acc[i * nn + p] = (cvp.0 + svq.0, cvp.1 + svq.1);
                    v_acc[i * nn + q] = (cvq.0 - svp.0, cvq.1 - svp.1);
                }
            }
        }
    }
    let sig_raw: Vec<f64> = (0..nn)
        .map(|p| gj_mag(jcol_dot_c(&work, mm, nn, p, p)).sqrt())
        .collect();
    let mut order: Vec<usize> = (0..nn).collect();
    order.sort_by(|&x, &y| {
        sig_raw[y]
            .partial_cmp(&sig_raw[x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sig: Vec<f64> = order.iter().map(|&p| sig_raw[p]).collect();
    let smax = sig.first().copied().unwrap_or(0.0);
    let mut u_part: Vec<(f64, f64)> = vec![(0.0, 0.0); mm * nn];
    for (new_p, &old_p) in order.iter().enumerate() {
        let sp = sig[new_p];
        if sp > 1e-15 * smax {
            for i in 0..mm {
                u_part[i * nn + new_p] = cdiv_r(work[i * nn + old_p], sp);
            }
        }
    }
    // Vh = V^H：Vh[i][j] = conj(v_acc[j][order[i]])
    let mut vh_mat: Vec<(f64, f64)> = vec![(0.0, 0.0); nn * nn];
    for i in 0..nn {
        for j in 0..nn {
            vh_mat[i * nn + j] = cj(v_acc[j * nn + order[i]]);
        }
    }
    // u_part 补零至 (mm, mm)
    let mut u_pad: Vec<(f64, f64)> = vec![(0.0, 0.0); mm * mm];
    for i in 0..mm {
        for j in 0..nn {
            u_pad[i * mm + j] = u_part[i * nn + j];
        }
    }
    if full_matrices && mm > nn {
        // 复 Gram-Schmidt：proj = Σ conj(b[r]) e[r]，e -= proj * b
        let mut basis: Vec<Vec<(f64, f64)>> = (0..nn)
            .map(|p| (0..mm).map(|i| u_pad[i * mm + p]).collect())
            .collect();
        for i in 0..mm {
            let mut e: Vec<(f64, f64)> = (0..mm)
                .map(|r| if r == i { (1.0, 0.0) } else { (0.0, 0.0) })
                .collect();
            for b in basis.iter() {
                let mut proj = (0.0, 0.0);
                for r in 0..mm {
                    let t = cj_mul(b[r], e[r]);
                    proj = (proj.0 + t.0, proj.1 + t.1);
                }
                for r in 0..mm {
                    let t = gj_cmul(proj, b[r]);
                    e[r] = (e[r].0 - t.0, e[r].1 - t.1);
                }
            }
            let ne = (0..mm).map(|r| gj_mag(e[r]).powi(2)).sum::<f64>().sqrt();
            if ne < 1e-8 {
                continue;
            }
            basis.push(e.iter().map(|&z| cdiv_r(z, ne)).collect());
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
    let (u_full, vh_full): (Vec<(f64, f64)>, Vec<(f64, f64)>) = if transposed {
        let mut u_out = vec![(0.0, 0.0); nn * nn];
        for i in 0..nn {
            for j in 0..nn {
                u_out[j * nn + i] = vh_mat[i * nn + j];
            }
        }
        let mut vh_out = vec![(0.0, 0.0); mm * mm];
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
        let nu: Vec<(f64, f64)> = (0..m)
            .flat_map(|i| u_out[i * m..i * m + k].iter().copied())
            .collect();
        let nv = vh_out[..k * n].to_vec();
        u_shape = (m, k);
        v_shape = (k, n);
        u_out = nu;
        vh_out = nv;
    }
    Ok((
        u_out.iter().map(|z| z.0).collect(),
        u_out.iter().map(|z| z.1).collect(),
        u_shape,
        sig,
        vh_out.iter().map(|z| z.0).collect(),
        vh_out.iter().map(|z| z.1).collect(),
        v_shape,
    ))
}

/// SVD 输入校验：2-d 非空矩阵（实数/复数）；空 → Err（Python 侧回退
/// 纯 Python Jacobi）。返回 (m, n, flat real, flat imag — 实数为 None)。
fn svd_jacobi_check(a: &NdArray) -> PyResult<(usize, usize, Vec<f64>, Option<Vec<f64>>)> {
    let shape: Vec<usize> = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(PyValueError::new_err(
            "svd_jacobi: a must be non-empty (m, n)",
        ));
    }
    let data: Vec<f64> = a.data.iter().copied().collect();
    let imag = a.imag.as_ref().map(|arr| arr.iter().copied().collect());
    Ok((shape[0], shape[1], data, imag))
}

/// SVD 奇异值（任意 m×n，实数/复数）：1-d 降序 sig（len = min(m,n)）。
/// 空矩阵返回 Err，由 Python 侧回退纯 Python Jacobi；非有限复数边缘
/// 亦 Err（Python 同样情形抛 TypeError）。
#[pyfunction]
fn svd_jacobi_sig(a: &NdArray) -> PyResult<NdArray> {
    let (m, n, data, imag) = svd_jacobi_check(a)?;
    let sig = match imag {
        None => svd_jacobi_impl(&data, m, n, false).2,
        Some(im) => svd_jacobi_impl_c(&data, &im, m, n, false)?.3,
    };
    let arr = Array::from_shape_vec(IxDyn(&[sig.len()]), sig)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        data: arr,
        imag: None,
    })
}

/// 完整 SVD（任意 m×n，实数/复数）：(u, s, vh)——full：u (m,m)、vh
/// (n,n)；reduced：u (m,k)、vh (k,n)，s 降序 (k,)。语义与 Python
/// _svd_jacobi 实数/复数分支一致；空矩阵返回 Err。
#[pyfunction]
fn svd_jacobi_uv(a: &NdArray, full_matrices: bool) -> PyResult<(NdArray, NdArray, NdArray)> {
    let (m, n, data, imag) = svd_jacobi_check(a)?;
    match imag {
        None => {
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
        Some(im) => {
            let (u_re, u_im, us, sig, vh_re, vh_im, vs) =
                svd_jacobi_impl_c(&data, &im, m, n, full_matrices)?;
            let u_arr = Array::from_shape_vec(IxDyn(&[us.0, us.1]), u_re)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let u_imag = Array::from_shape_vec(IxDyn(&[us.0, us.1]), u_im)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let s_arr = Array::from_shape_vec(IxDyn(&[sig.len()]), sig)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let vh_arr = Array::from_shape_vec(IxDyn(&[vs.0, vs.1]), vh_re)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let vh_imag = Array::from_shape_vec(IxDyn(&[vs.0, vs.1]), vh_im)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok((
                NdArray {
                    data: u_arr,
                    imag: Some(u_imag),
                },
                NdArray {
                    data: s_arr,
                    imag: None,
                },
                NdArray {
                    data: vh_arr,
                    imag: Some(vh_imag),
                },
            ))
        }
    }
}

/// SVD 后组装核心（pinv / lstsq），内部复用已验证的 Jacobi SVD
/// （实/复）；语义与 pinv._one / lstsq 的纯 Python 组装逐条对齐：
/// cutoff = rcond * s_max（s_max = max(s)），rank = count(s > cutoff)；
/// pinv 模式（b=None）：P[i][r] = Σ_j conj(vt[j][i])·s_inv[j]·
/// conj(u[r][j])，s_inv==0 时跳过；
/// lstsq 模式（b: (m,) / (m,r)）：uhb[j] = Σ_i conj(u[i][j])·b[i][c]，
/// x[i][c] = Σ_j conj(vt[j][i])·s_inv[j]·uhb[j]；residuals =
/// 逐列 Σ_i |b - A x|²，仅 m>n 且 rank==n 且 r>0 时非空。
/// 返回 (P 或 x, residuals（pinv 模式为空）, rank, s)。
#[pyfunction]
#[pyo3(signature = (a, b=None, rcond=1e-15, mode=0))]
fn svd_assemble_core(
    a: &NdArray,
    b: Option<&NdArray>,
    rcond: f64,
    mode: u8,
) -> PyResult<(NdArray, NdArray, i64, NdArray)> {
    fn build_2d(re: Vec<f64>, im: Option<Vec<f64>>, rows: usize, cols: usize) -> PyResult<NdArray> {
        let data = Array::from_shape_vec(IxDyn(&[rows, cols]), re)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let imag = match im {
            Some(v) => Some(
                Array::from_shape_vec(IxDyn(&[rows, cols]), v)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
            None => None,
        };
        Ok(NdArray { data, imag })
    }
    let (m, n, data, imag) = svd_jacobi_check(a)?;
    let (u, s, vh) = match &imag {
        None => {
            let (u, _us, s, vh, _vs) = svd_jacobi_impl(&data, m, n, false);
            (
                u.iter().map(|&r| (r, 0.0)).collect::<Vec<_>>(),
                s,
                vh.iter().map(|&r| (r, 0.0)).collect::<Vec<_>>(),
            )
        }
        Some(im) => {
            let (u_re, u_im, _us, s, vh_re, vh_im, _vs) =
                svd_jacobi_impl_c(&data, im, m, n, false)?;
            (
                u_re.iter()
                    .zip(u_im.iter())
                    .map(|(&r, &i)| (r, i))
                    .collect(),
                s,
                vh_re
                    .iter()
                    .zip(vh_im.iter())
                    .map(|(&r, &i)| (r, i))
                    .collect(),
            )
        }
    };
    let is_c = a.imag.is_some();
    let k = m.min(n);
    let s_max = s.iter().cloned().fold(0.0f64, f64::max);
    let cutoff = rcond * s_max;
    let rank = s.iter().filter(|v| **v > cutoff).count() as i64;
    let s_inv: Vec<f64> = s
        .iter()
        .map(|v| if *v > cutoff { 1.0 / v } else { 0.0 })
        .collect();
    let s_arr = Array::from_shape_vec(IxDyn(&[s.len()]), s.clone())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let s_out = NdArray {
        data: s_arr,
        imag: None,
    };
    if mode == 0 {
        // pinv：P = V·D⁺·U^H（s_inv 为 0 的三元组跳过，语义同 pinv._one）
        if b.is_some() {
            return Err(PyValueError::new_err(
                "svd_assemble_core: pinv mode takes no b",
            ));
        }
        let mut p_re = vec![0.0; n * m];
        let mut p_im = vec![0.0; n * m];
        for j in 0..k {
            let si = s_inv[j];
            if si == 0.0 {
                continue;
            }
            for i in 0..n {
                let a1 = sc_mul(cj(vh[j * n + i]), si);
                if a1 == (0.0, 0.0) {
                    continue;
                }
                for r in 0..m {
                    let t = gj_cmul(a1, cj(u[r * k + j]));
                    p_re[i * m + r] += t.0;
                    p_im[i * m + r] += t.1;
                }
            }
        }
        let p = build_2d(p_re, if is_c { Some(p_im) } else { None }, n, m)?;
        let res_arr = Array::from_shape_vec(IxDyn(&[0]), vec![])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok((
            p,
            NdArray {
                data: res_arr,
                imag: None,
            },
            rank,
            s_out,
        ));
    }
    // lstsq：b 校验 (m,) 或 (m, r)
    let b_arr = match b {
        Some(x) => x,
        None => {
            return Err(PyValueError::new_err(
                "svd_assemble_core: lstsq mode requires b",
            ));
        }
    };
    let bs: Vec<usize> = b_arr.data.shape().to_vec();
    let n_rhs = match bs.as_slice() {
        [mm] if *mm == m => 1,
        [mm, rr] if *mm == m => *rr,
        _ => {
            return Err(PyValueError::new_err(
                "svd_assemble_core: b must be (m,) or (m, r)",
            ));
        }
    };
    let is_bc = b_arr.imag.is_some();
    let bre: Vec<f64> = b_arr.data.iter().copied().collect();
    let bim: Vec<f64> = b_arr
        .imag
        .as_ref()
        .map(|x| x.iter().copied().collect())
        .unwrap_or_default();
    let bvec: Vec<(f64, f64)> = if is_bc {
        bre.iter().zip(bim.iter()).map(|(&r, &i)| (r, i)).collect()
    } else {
        bre.iter().map(|&r| (r, 0.0)).collect()
    };
    let aim: Vec<f64> = match &imag {
        Some(v) => v.clone(),
        None => vec![0.0; m * n],
    };
    // x = Vh^H · diag(1/s) · U^H · b（运算次序与 Python 逐条对齐）
    let mut x_re = vec![0.0; n * n_rhs];
    let mut x_im = vec![0.0; n * n_rhs];
    for c in 0..n_rhs {
        let mut uhb: Vec<(f64, f64)> = vec![(0.0, 0.0); k];
        for j in 0..k {
            for i in 0..m {
                let t = cj_mul(u[i * k + j], bvec[i * n_rhs + c]);
                uhb[j] = (uhb[j].0 + t.0, uhb[j].1 + t.1);
            }
        }
        for i in 0..n {
            let mut acc = (0.0, 0.0);
            for j in 0..k {
                let t = gj_cmul(sc_mul(cj(vh[j * n + i]), s_inv[j]), uhb[j]);
                acc = (acc.0 + t.0, acc.1 + t.1);
            }
            x_re[i * n_rhs + c] = acc.0;
            x_im[i * n_rhs + c] = acc.1;
        }
    }
    // 残差：m>n 且满列秩时逐列 Σ |b - A x|²（与 Python 同次序）
    let compute_res = m > n && rank == n as i64 && n_rhs > 0;
    let mut res_vals: Vec<f64> = vec![];
    if compute_res {
        for c in 0..n_rhs {
            let mut acc = 0.0f64;
            for i in 0..m {
                let mut r = bvec[i * n_rhs + c];
                for kk in 0..n {
                    let t = gj_cmul(
                        (data[i * n + kk], aim[i * n + kk]),
                        (x_re[kk * n_rhs + c], x_im[kk * n_rhs + c]),
                    );
                    r = (r.0 - t.0, r.1 - t.1);
                }
                acc += gj_mag(r).powi(2);
            }
            res_vals.push(acc);
        }
    }
    let x = build_2d(
        x_re,
        if is_c || is_bc { Some(x_im) } else { None },
        n,
        n_rhs,
    )?;
    let res_arr = Array::from_shape_vec(IxDyn(&[res_vals.len()]), res_vals)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        x,
        NdArray {
            data: res_arr,
            imag: None,
        },
        rank,
        s_out,
    ))
}

/// Householder QR 核心（任意 m×n，实/复），语义与 Python
/// _qr_householder 逐条对齐：alpha 相位约定（复数 alpha =
/// -x0/|x0|·||x||（|x0|=0 → -||x||），实数 -sign(x0)·||x||，均取
/// 首分量相加避相消）；v = x - alpha·e1，H = I - 2vv^H/v^Hv；
/// R 就地上三角，Q 同步累积（Q_new = Q_old @ H，复数时 factor
/// 取共轭后乘 conj(v)）；norm==0 或 vv==0 跳过该列；反射残留的
/// 下三角显式置零。want_complete：Q (m,m)、R (m,n)；否则
/// Q (m,k)、R (k,n)，k=min(m,n)。空矩阵 → Err（Python 兜底）。
#[pyfunction]
#[pyo3(signature = (a, want_complete))]
fn qr_householder_core(a: &NdArray, want_complete: bool) -> PyResult<(NdArray, NdArray)> {
    let shape: Vec<usize> = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(PyValueError::new_err(
            "qr_householder: cannot factor an empty matrix",
        ));
    }
    let m = shape[0];
    let n = shape[1];
    let k = m.min(n);
    let is_c = a.imag.is_some();
    let mut work: Vec<(f64, f64)> = match &a.imag {
        None => a.data.iter().map(|&r| (r, 0.0)).collect(),
        Some(im) => a
            .data
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| (r, i))
            .collect(),
    };
    let mut q: Vec<(f64, f64)> = vec![(0.0, 0.0); m * m];
    for i in 0..m {
        q[i * m + i] = (1.0, 0.0);
    }
    for j in 0..k {
        let width = m - j;
        let mut norm2 = 0.0f64;
        for i in 0..width {
            norm2 += gj_mag(work[(j + i) * n + j]).powi(2);
        }
        let norm = norm2.sqrt();
        if norm == 0.0 {
            continue;
        }
        let lead = work[j * n + j];
        let alpha: (f64, f64) = if is_c {
            let mag = gj_mag(lead);
            if mag > 0.0 {
                // 注意 -lead 是取负 (-lr, -li)，不是共轭
                sc_mul(cdiv_r((-lead.0, -lead.1), mag), norm)
            } else {
                (-norm, 0.0)
            }
        } else {
            ((if lead.0 < 0.0 { -norm } else { norm }), 0.0)
        };
        let mut v: Vec<(f64, f64)> = (0..width).map(|i| work[(j + i) * n + j]).collect();
        v[0] = (v[0].0 - alpha.0, v[0].1 - alpha.1);
        let mut vv = 0.0f64;
        for t in v.iter() {
            vv += gj_mag(*t).powi(2);
        }
        if vv == 0.0 {
            continue;
        }
        // R 更新（列 c = j..n）：rows -= (2·dot2(v, col)/vv)·v
        for c in j..n {
            let mut d = (0.0, 0.0);
            for i in 0..width {
                let t = cj_mul(v[i], work[(j + i) * n + c]);
                d = (d.0 + t.0, d.1 + t.1);
            }
            let f = (2.0 * d.0 / vv, 2.0 * d.1 / vv);
            for i in 0..width {
                let w = work[(j + i) * n + c];
                let t = gj_cmul(f, v[i]);
                work[(j + i) * n + c] = (w.0 - t.0, w.1 - t.1);
            }
        }
        // Q 更新（行 r）：Q_seg -= conj(2·dot2(Q_seg, conj(v))/vv)·conj(v)
        for r_row in 0..m {
            let mut d = (0.0, 0.0);
            for i in 0..width {
                let t = cj_mul(q[r_row * m + j + i], cj(v[i]));
                d = (d.0 + t.0, d.1 + t.1);
            }
            let mut f = (2.0 * d.0 / vv, 2.0 * d.1 / vv);
            if is_c {
                f = cj(f);
            }
            for i in 0..width {
                let w = q[r_row * m + j + i];
                let t = gj_cmul(f, cj(v[i]));
                q[r_row * m + j + i] = (w.0 - t.0, w.1 - t.1);
            }
        }
    }
    // 反射残留的下三角显式置零（j < min(i, n)）
    for i in 0..m {
        for j in 0..i.min(n) {
            work[i * n + j] = (0.0, 0.0);
        }
    }
    let q_re: Vec<f64> = q.iter().map(|z| z.0).collect();
    let q_im: Vec<f64> = q.iter().map(|z| z.1).collect();
    let r_re: Vec<f64> = work.iter().map(|z| z.0).collect();
    let r_im: Vec<f64> = work.iter().map(|z| z.1).collect();
    let (q_re, q_im, q_shape) = if want_complete {
        (q_re, q_im, (m, m))
    } else {
        let nu: Vec<f64> = (0..m)
            .flat_map(|i| q_re[i * m..i * m + k].iter().copied())
            .collect();
        let ni: Vec<f64> = (0..m)
            .flat_map(|i| q_im[i * m..i * m + k].iter().copied())
            .collect();
        (nu, ni, (m, k))
    };
    let (r_re, r_im, r_shape) = if want_complete {
        (r_re, r_im, (m, n))
    } else {
        (r_re[..k * n].to_vec(), r_im[..k * n].to_vec(), (k, n))
    };
    let q_arr = Array::from_shape_vec(IxDyn(&[q_shape.0, q_shape.1]), q_re)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let r_arr = Array::from_shape_vec(IxDyn(&[r_shape.0, r_shape.1]), r_re)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let q_imag = if is_c {
        Some(
            Array::from_shape_vec(IxDyn(&[q_shape.0, q_shape.1]), q_im)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        )
    } else {
        None
    };
    let r_imag = if is_c {
        Some(
            Array::from_shape_vec(IxDyn(&[r_shape.0, r_shape.1]), r_im)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        )
    } else {
        None
    };
    Ok((
        NdArray {
            data: q_arr,
            imag: q_imag,
        },
        NdArray {
            data: r_arr,
            imag: r_imag,
        },
    ))
}

/// 复平方根主支（cmath.sqrt 语义）
#[inline]
fn csqrt(z: (f64, f64)) -> (f64, f64) {
    let (r, i) = z;
    if r == 0.0 && i == 0.0 {
        return (0.0, 0.0);
    }
    let m = gj_mag(z);
    let re = ((m + r) / 2.0).sqrt();
    let im = if i >= 0.0 {
        ((m - r) / 2.0).sqrt()
    } else {
        -((m - r) / 2.0).sqrt()
    };
    (re, im)
}

/// 2x2 块特征值（降序），语义同 Python _2x2
fn eig2(work: &[(f64, f64)], n: usize, lo: usize) -> ((f64, f64), (f64, f64)) {
    let a = work[lo * n + lo];
    let b = work[lo * n + lo + 1];
    let c = work[(lo + 1) * n + lo];
    let d = work[(lo + 1) * n + lo + 1];
    let tr = (a.0 + d.0, a.1 + d.1);
    // det = a*d - b*c（复数）
    let det = (
        a.0 * d.0 - a.1 * d.1 - (b.0 * c.0 - b.1 * c.1),
        a.0 * d.1 + a.1 * d.0 - (b.0 * c.1 + b.1 * c.0),
    );
    // disc = sqrt(tr² - 4det)
    let disc = csqrt((
        tr.0 * tr.0 - tr.1 * tr.1 - 4.0 * det.0,
        2.0 * tr.0 * tr.1 - 4.0 * det.1,
    ));
    let e1 = ((tr.0 + disc.0) / 2.0, (tr.1 + disc.1) / 2.0);
    let e2 = ((tr.0 - disc.0) / 2.0, (tr.1 - disc.1) / 2.0);
    (e1, e2)
}

/// Hessenberg 相似约化（复 Householder），语义与 Python
/// _hessenberg_reduce 逐条对齐
fn hessenberg_reduce(work: &mut [(f64, f64)], n: usize) {
    if n <= 2 {
        return;
    }
    for j in 0..(n - 2) {
        let lo = j + 1;
        let mut norm2 = 0.0f64;
        for t in lo..n {
            norm2 += gj_mag(work[t * n + j]).powi(2);
        }
        let norm_x = norm2.sqrt();
        if norm_x == 0.0 {
            continue;
        }
        let x0 = work[lo * n + j];
        let sgn: (f64, f64) = if gj_mag(x0) > 0.0 {
            cdiv_r(x0, gj_mag(x0))
        } else {
            (1.0, 0.0)
        };
        let mut v: Vec<(f64, f64)> = (lo..n).map(|t| work[t * n + j]).collect();
        v[0] = (v[0].0 + sgn.0 * norm_x, v[0].1 + sgn.1 * norm_x);
        let mut nv2 = 0.0f64;
        for t in v.iter() {
            nv2 += gj_mag(*t).powi(2);
        }
        let nv = nv2.sqrt();
        if nv == 0.0 {
            continue;
        }
        for t in v.iter_mut() {
            *t = cdiv_r(*t, nv);
        }
        // w = v^H A[:, j..]（先从未修改的 A 完整算出）
        for c in j..n {
            let mut w = (0.0, 0.0);
            for t in lo..n {
                let p = cj_mul(v[t - lo], work[t * n + c]);
                w = (w.0 + p.0, w.1 + p.1);
            }
            for r in lo..n {
                let t = sc_mul(v[r - lo], 2.0);
                let wv = work[r * n + c];
                let t2 = gj_cmul(t, w);
                work[r * n + c] = (wv.0 - t2.0, wv.1 - t2.1);
            }
        }
        // 右乘 P：(AP)[r][c] = A[r][c] - 2(Av)[r]·conj(v[c])
        let mut avs = vec![(0.0, 0.0); n];
        for r in 0..n {
            let mut acc = (0.0, 0.0);
            for t in lo..n {
                let p = gj_cmul(work[r * n + t], v[t - lo]);
                acc = (acc.0 + p.0, acc.1 + p.1);
            }
            avs[r] = acc;
        }
        for c in lo..n {
            let vc = cj(v[c - lo]);
            for r in 0..n {
                let t = sc_mul(avs[r], 2.0);
                let t2 = gj_cmul(t, vc);
                let wv = work[r * n + c];
                work[r * n + c] = (wv.0 - t2.0, wv.1 - t2.1);
            }
        }
    }
}

/// H[i:m, i:m] - qI 的 Householder QR，返回 (Q, R)（语义同 _qr_factor）
fn qr_factor(
    work: &[(f64, f64)],
    n: usize,
    i: usize,
    m: usize,
    q_shift: (f64, f64),
) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let k = m - i;
    let mut r = vec![(0.0, 0.0); k * k];
    for r_ in 0..k {
        for c in 0..k {
            let mut v = work[(i + r_) * n + i + c];
            if r_ == c {
                v = (v.0 - q_shift.0, v.1 - q_shift.1);
            }
            r[r_ * k + c] = v;
        }
    }
    let mut qq: Vec<(f64, f64)> = vec![(0.0, 0.0); k * k];
    for c in 0..k {
        qq[c * k + c] = (1.0, 0.0);
    }
    if k == 0 {
        return (qq, r);
    }
    for j in 0..k.saturating_sub(1) {
        let mut norm2 = 0.0f64;
        for t in j..k {
            norm2 += gj_mag(r[t * k + j]).powi(2);
        }
        let norm_x = norm2.sqrt();
        if norm_x == 0.0 {
            continue;
        }
        let x0 = r[j * k + j];
        let sgn: (f64, f64) = if gj_mag(x0) > 0.0 {
            cdiv_r(x0, gj_mag(x0))
        } else {
            (1.0, 0.0)
        };
        let mut v: Vec<(f64, f64)> = (j..k).map(|t| r[t * k + j]).collect();
        v[0] = (v[0].0 + sgn.0 * norm_x, v[0].1 + sgn.1 * norm_x);
        let mut nv2 = 0.0f64;
        for t in v.iter() {
            nv2 += gj_mag(*t).powi(2);
        }
        let nv = nv2.sqrt();
        if nv == 0.0 {
            continue;
        }
        for t in v.iter_mut() {
            *t = cdiv_r(*t, nv);
        }
        // R <- P R
        for c in j..k {
            let mut w = (0.0, 0.0);
            for t in j..k {
                let p = cj_mul(v[t - j], r[t * k + c]);
                w = (w.0 + p.0, w.1 + p.1);
            }
            for r_ in j..k {
                let t = sc_mul(v[r_ - j], 2.0);
                let t2 = gj_cmul(t, w);
                let wv = r[r_ * k + c];
                r[r_ * k + c] = (wv.0 - t2.0, wv.1 - t2.1);
            }
        }
        // Q <- Q P
        for r_ in 0..k {
            let mut qv = (0.0, 0.0);
            for t in j..k {
                let p = gj_cmul(qq[r_ * k + t], v[t - j]);
                qv = (qv.0 + p.0, qv.1 + p.1);
            }
            if qv == (0.0, 0.0) {
                continue;
            }
            for c in j..k {
                let t = sc_mul(qv, 2.0);
                let t2 = gj_cmul(t, cj(v[c - j]));
                let wv = qq[r_ * k + c];
                qq[r_ * k + c] = (wv.0 - t2.0, wv.1 - t2.1);
            }
        }
    }
    (qq, r)
}

/// 单位移 QR 步：B = block - qI = QR → block = RQ + qI（原地回写）
fn qr_shift_step(work: &mut [(f64, f64)], n: usize, i: usize, m: usize, q_shift: (f64, f64)) {
    let k = m - i;
    let (qq, r) = qr_factor(work, n, i, m, q_shift);
    for r_ in 0..k {
        for c in 0..k {
            let mut acc = (0.0, 0.0);
            for t in 0..k {
                let p = gj_cmul(r[r_ * k + t], qq[t * k + c]);
                acc = (acc.0 + p.0, acc.1 + p.1);
            }
            let v = if r_ == c {
                (acc.0 + q_shift.0, acc.1 + q_shift.1)
            } else {
                acc
            };
            work[(i + r_) * n + i + c] = v;
        }
    }
}

/// Francis 显式双移位步：移位取尾部 2x2 块的两个特征值
fn qr_double_step(work: &mut [(f64, f64)], n: usize, i: usize, m: usize) {
    let (a, b, c, d) = (
        work[(m - 2) * n + m - 2],
        work[(m - 2) * n + m - 1],
        work[(m - 1) * n + m - 2],
        work[(m - 1) * n + m - 1],
    );
    let tr = (a.0 + d.0, a.1 + d.1);
    let det = (
        a.0 * d.0 - a.1 * d.1 - (b.0 * c.0 - b.1 * c.1),
        a.0 * d.1 + a.1 * d.0 - (b.0 * c.1 + b.1 * c.0),
    );
    let disc = csqrt((
        tr.0 * tr.0 - tr.1 * tr.1 - 4.0 * det.0,
        2.0 * tr.0 * tr.1 - 4.0 * det.1,
    ));
    qr_shift_step(
        work,
        n,
        i,
        m,
        ((tr.0 + disc.0) / 2.0, (tr.1 + disc.1) / 2.0),
    );
    qr_shift_step(
        work,
        n,
        i,
        m,
        ((tr.0 - disc.0) / 2.0, (tr.1 - disc.1) / 2.0),
    );
}

/// dlahqr 式窗口递归驱动（语义同 _qr_iter_eigvals 的 _window）
fn qr_window(
    work: &mut [(f64, f64)],
    n: usize,
    lo0: usize,
    mut hi: usize,
    evals: &mut Vec<(f64, f64)>,
    it: &mut usize,
) -> PyResult<()> {
    let eps = 2.220446049250313e-16;
    let lo = lo0;
    let mut stall = 0usize;
    while hi + 1 > lo {
        let size = hi - lo + 1;
        if size == 1 {
            evals.push(work[lo * n + lo]);
            return Ok(());
        }
        if size == 2 {
            let (e1, e2) = eig2(work, n, lo);
            evals.push(e1);
            evals.push(e2);
            return Ok(());
        }
        // 从底部向上找第一个可收缩次对角元
        let mut split: Option<usize> = None;
        for jj in (lo..hi).rev() {
            let sub = work[(jj + 1) * n + jj];
            if sub == (0.0, 0.0) {
                split = Some(jj);
                break;
            }
            if gj_mag(sub)
                <= eps * (gj_mag(work[jj * n + jj]) + gj_mag(work[(jj + 1) * n + jj + 1]))
            {
                work[(jj + 1) * n + jj] = (0.0, 0.0);
                split = Some(jj);
                break;
            }
        }
        if let Some(sp) = split {
            stall = 0;
            let tsize = hi - sp;
            if tsize == 1 {
                evals.push(work[hi * n + hi]);
                hi = sp;
                continue;
            }
            if tsize == 2 {
                let (e1, e2) = eig2(work, n, sp + 1);
                evals.push(e1);
                evals.push(e2);
                hi = sp;
                continue;
            }
            qr_window(work, n, sp + 1, hi, evals, it)?;
            hi = sp;
            continue;
        }
        // 双移位 QR 步；每 10 步无收缩改用异常移位
        if stall % 10 == 9 {
            let exc =
                0.75f64 * (gj_mag(work[hi * n + hi - 1]) + gj_mag(work[(hi - 1) * n + hi - 2]));
            let base = work[hi * n + hi];
            qr_shift_step(work, n, lo, hi + 1, (base.0 + 0.75 * exc, base.1));
        } else {
            qr_double_step(work, n, lo, hi + 1);
        }
        stall += 1;
        *it += 1;
        if *it > 60 * n {
            return Err(PyValueError::new_err(
                "QR eigenvalue iteration did not converge",
            ));
        }
    }
    Ok(())
}

/// n≥9 大矩阵特征值核心：Hessenberg 相似约化 + dlahqr 式双移位 QR
/// 迭代（语义与 Python _hessenberg_reduce/_qr_iter_eigvals 逐条对齐，
/// 含尾块 1x1/2x2 提取、每 10 步异常移位、60n 步不收敛报错）。
/// 实输入复算术残留的微小虚部归零（|imag| < 1e-11·max(1,|real|)，
/// 同 Python 版）；不收敛 → Err，由 Python 回退 FL+DK。
#[pyfunction]
fn eigvals_qr_core(a: &NdArray) -> PyResult<NdArray> {
    let shape: Vec<usize> = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err(
            "eigvals_qr_core: a must be square (n, n)",
        ));
    }
    let n = shape[0];
    let mut work: Vec<(f64, f64)> = match &a.imag {
        None => a.data.iter().map(|&r| (r, 0.0)).collect(),
        Some(im) => a
            .data
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| (r, i))
            .collect(),
    };
    hessenberg_reduce(&mut work, n);
    let mut evals: Vec<(f64, f64)> = vec![];
    let mut it = 0usize;
    if n > 0 {
        qr_window(&mut work, n, 0, n - 1, &mut evals, &mut it)?;
    }
    let out: Vec<(f64, f64)> = evals
        .iter()
        .map(|z| {
            if z.1.abs() < 1e-11 * f64::max(1.0, z.0.abs()) {
                (z.0, 0.0)
            } else {
                *z
            }
        })
        .collect();
    let out_re: Vec<f64> = out.iter().map(|z| z.0).collect();
    let out_im: Vec<f64> = out.iter().map(|z| z.1).collect();
    let data = Array::from_shape_vec(IxDyn(&[n]), out_re)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let imag = Array::from_shape_vec(IxDyn(&[n]), out_im)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        data,
        imag: Some(imag),
    })
}

/// 批量特征值：stacked (..., n, n) 一次 FFI 内逐批 QR 迭代（消除
/// Python 侧逐批内核调用 + wrap/tolist 往返）。要求 n ≥ 9（n<8 的
/// FL+DK 小矩阵语义保持 Python），n<9 或非方阵 → Err。
/// 返回 (evals (B, n) complex, ok (B,) f64)：单批不收敛 → 该行 NaN
/// 且 ok=0，由 Python 侧对该批回退纯 Python。
#[pyfunction]
fn eigvals_qr_batch_core(a: &NdArray) -> PyResult<(NdArray, NdArray)> {
    let shape: Vec<usize> = a.data.shape().to_vec();
    if shape.len() < 3 {
        return Err(PyValueError::new_err(
            "eigvals_qr_batch_core: a must be (..., n, n)",
        ));
    }
    let n = shape[shape.len() - 1];
    if shape[shape.len() - 2] != n {
        return Err(PyValueError::new_err(
            "eigvals_qr_batch_core: a must be square (..., n, n)",
        ));
    }
    if n < 9 {
        return Err(PyValueError::new_err(
            "eigvals_qr_batch_core: n < 9 stays on the Python FL+DK path",
        ));
    }
    let b: usize = shape[..shape.len() - 2].iter().product();
    let nn = n * n;
    let mut out_re = vec![f64::NAN; b * n];
    let mut out_im = vec![f64::NAN; b * n];
    let mut ok = vec![1.0f64; b];
    for bi in 0..b {
        let base = bi * nn;
        let mut work: Vec<(f64, f64)> = match &a.imag {
            None => a
                .data
                .iter()
                .skip(base)
                .take(nn)
                .map(|&r| (r, 0.0))
                .collect(),
            Some(im) => a
                .data
                .iter()
                .zip(im.iter())
                .skip(base)
                .take(nn)
                .map(|(&r, &i)| (r, i))
                .collect(),
        };
        hessenberg_reduce(&mut work, n);
        let mut evals: Vec<(f64, f64)> = vec![];
        let mut it = 0usize;
        match qr_window(&mut work, n, 0, n - 1, &mut evals, &mut it) {
            Ok(()) => {
                for (i, z) in evals.iter().enumerate() {
                    if z.1.abs() < 1e-11 * f64::max(1.0, z.0.abs()) {
                        out_re[bi * n + i] = z.0;
                        out_im[bi * n + i] = 0.0;
                    } else {
                        out_re[bi * n + i] = z.0;
                        out_im[bi * n + i] = z.1;
                    }
                }
            }
            Err(_) => {
                ok[bi] = 0.0;
            }
        }
    }
    let data = Array::from_shape_vec(IxDyn(&[b, n]), out_re)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let imag = Array::from_shape_vec(IxDyn(&[b, n]), out_im)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ok_arr =
        Array::from_shape_vec(IxDyn(&[b]), ok).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        NdArray {
            data,
            imag: Some(imag),
        },
        NdArray {
            data: ok_arr,
            imag: None,
        },
    ))
}

/// 对称/Hermitian 幂迭代特征分解（语义与 Python _eig_power_iteration
/// 逐条对齐）：对每个 k 从 e_k 起步、至多 100 次幂迭代（norm² < 1e-24
/// 提前终止）、Rayleigh 商 v^H·A·v 取特征值、特征向量按列存放、无
/// deflation。输出 evals (n,) 与 evecs (n,n)（complex 双通道）。
#[pyfunction]
fn eig_power_core(a: &NdArray) -> PyResult<(NdArray, NdArray)> {
    let shape: Vec<usize> = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] || shape[0] == 0 {
        return Err(PyValueError::new_err(
            "eig_power_core: a must be non-empty (n, n)",
        ));
    }
    let n = shape[0];
    let mat: Vec<(f64, f64)> = match &a.imag {
        None => a.data.iter().map(|&r| (r, 0.0)).collect(),
        Some(im) => a
            .data
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| (r, i))
            .collect(),
    };
    let matmul = |v: &[(f64, f64)]| -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| {
                let mut acc = (0.0, 0.0);
                for j in 0..n {
                    let p = gj_cmul(mat[i * n + j], v[j]);
                    acc = (acc.0 + p.0, acc.1 + p.1);
                }
                acc
            })
            .collect()
    };
    let mut evals = vec![(0.0, 0.0); n];
    let mut evecs = vec![(0.0, 0.0); n * n];
    for k in 0..n {
        let mut v: Vec<(f64, f64)> = vec![(0.0, 0.0); n];
        v[k] = (1.0, 0.0);
        for _ in 0..100 {
            let new_v = matmul(&v);
            let norm_sq: f64 = new_v.iter().map(|x| gj_mag(*x).powi(2)).sum();
            if norm_sq < 1e-24 {
                break;
            }
            let norm = norm_sq.sqrt();
            v = new_v.iter().map(|x| cdiv_r(*x, norm)).collect();
        }
        let mat_v = matmul(&v);
        let mut lam = (0.0, 0.0);
        for i in 0..n {
            let p = cj_mul(v[i], mat_v[i]);
            lam = (lam.0 + p.0, lam.1 + p.1);
        }
        evals[k] = lam;
        for i in 0..n {
            evecs[i * n + k] = v[i];
        }
    }
    let evals_re: Vec<f64> = evals.iter().map(|z| z.0).collect();
    let evals_im: Vec<f64> = evals.iter().map(|z| z.1).collect();
    let evecs_re: Vec<f64> = evecs.iter().map(|z| z.0).collect();
    let evecs_im: Vec<f64> = evecs.iter().map(|z| z.1).collect();
    let evals_arr = Array::from_shape_vec(IxDyn(&[n]), evals_re)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let evals_im_arr = Array::from_shape_vec(IxDyn(&[n]), evals_im)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let evecs_arr = Array::from_shape_vec(IxDyn(&[n, n]), evecs_re)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let evecs_im_arr = Array::from_shape_vec(IxDyn(&[n, n]), evecs_im)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        NdArray {
            data: evals_arr,
            imag: Some(evals_im_arr),
        },
        NdArray {
            data: evecs_arr,
            imag: Some(evecs_im_arr),
        },
    ))
}

/// 复数列主元高斯消元（语义与 Python _solve_pivot 逐条对齐）：
/// 奇异时用最大可用枢轴继续、零枢轴跳过、回代求 x。
fn solve_pivot_c(a: &[(f64, f64)], b: &[(f64, f64)], n: usize) -> Vec<(f64, f64)> {
    let mut work = a.to_vec();
    let mut rhs = b.to_vec();
    for col in 0..n {
        let mut p = col;
        let mut best = gj_mag(work[col * n + col]);
        for r in col + 1..n {
            let m = gj_mag(work[r * n + col]);
            if m > best {
                best = m;
                p = r;
            }
        }
        if p != col {
            // 整行交换（work 是扁平行主序 n×n；Python 交换的是行）
            for j in 0..n {
                work.swap(col * n + j, p * n + j);
            }
            rhs.swap(col, p);
        }
        let pv = work[col * n + col];
        if pv == (0.0, 0.0) {
            continue;
        }
        for r in col + 1..n {
            let f = cdiv_c(work[r * n + col], pv);
            if f != (0.0, 0.0) {
                for c2 in col..n {
                    let t = gj_cmul(f, work[col * n + c2]);
                    let w = work[r * n + c2];
                    work[r * n + c2] = (w.0 - t.0, w.1 - t.1);
                }
                let t = gj_cmul(f, rhs[col]);
                rhs[r] = (rhs[r].0 - t.0, rhs[r].1 - t.1);
            }
        }
    }
    let mut x = vec![(0.0, 0.0); n];
    for r in (0..n).rev() {
        let pv = work[r * n + r];
        if pv == (0.0, 0.0) {
            continue;
        }
        let mut s = rhs[r];
        for c2 in r + 1..n {
            let t = gj_cmul(work[r * n + c2], x[c2]);
            s = (s.0 - t.0, s.1 - t.1);
        }
        x[r] = cdiv_c(s, pv);
    }
    x
}

/// 通用 n×n 特征向量（语义与 Python _eigenvectors_general 逐条对齐）：
/// 对每个特征值 λ 构造正规方程 B = (A−λI)^H(A−λI)（B 近奇异，解的
/// 方向即零空间 = 特征向量），依次以坐标方向 e_t 为右端解 Bv = e_t，
/// 解范数 > 1e-8 时归一化采用；全失败用 e_0 兜底。evals 顺序与输出
/// 列一一对应。返回 (n,n) 行主序 evecs（out[i][k] = 第 k 向量第 i 分量）。
#[pyfunction]
fn eig_vectors_core(a: &NdArray, evals: &NdArray) -> PyResult<NdArray> {
    let shape: Vec<usize> = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] || shape[0] == 0 {
        return Err(PyValueError::new_err(
            "eig_vectors_core: a must be non-empty (n, n)",
        ));
    }
    let n = shape[0];
    let mat: Vec<(f64, f64)> = match &a.imag {
        None => a.data.iter().map(|&r| (r, 0.0)).collect(),
        Some(im) => a
            .data
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| (r, i))
            .collect(),
    };
    if evals.data.len() != n {
        return Err(PyValueError::new_err(
            "eig_vectors_core: evals length mismatch",
        ));
    }
    let evals_v: Vec<(f64, f64)> = match &evals.imag {
        None => evals.data.iter().map(|&r| (r, 0.0)).collect(),
        Some(im) => evals
            .data
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| (r, i))
            .collect(),
    };
    let mut out = vec![(0.0, 0.0); n * n];
    for (k, &lam) in evals_v.iter().enumerate() {
        // B = (A−λI)^H (A−λI)
        let mut b_mat = vec![(0.0, 0.0); n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = (0.0, 0.0);
                for kk in 0..n {
                    let mut aik = mat[kk * n + i];
                    if kk == i {
                        aik = (aik.0 - lam.0, aik.1 - lam.1);
                    }
                    let mut akj = mat[kk * n + j];
                    if kk == j {
                        akj = (akj.0 - lam.0, akj.1 - lam.1);
                    }
                    let p = cj_mul(aik, akj);
                    s = (s.0 + p.0, s.1 + p.1);
                }
                b_mat[i * n + j] = s;
            }
        }
        // 依次以 e_t 为右端解 Bv = e_t，解范数 > 1e-8 采用
        let mut v: Option<Vec<(f64, f64)>> = None;
        for t in 0..n {
            let rhs: Vec<(f64, f64)> = (0..n)
                .map(|i| if i == t { (1.0, 0.0) } else { (0.0, 0.0) })
                .collect();
            let cand = solve_pivot_c(&b_mat, &rhs, n);
            let nrm_sq: f64 = cand.iter().map(|z| gj_mag(*z).powi(2)).sum();
            let nrm = nrm_sq.sqrt();
            if nrm > 1e-8 {
                v = Some(cand.iter().map(|z| cdiv_r(*z, nrm)).collect());
                break;
            }
        }
        let v = v.unwrap_or_else(|| {
            let mut x = vec![(0.0, 0.0); n];
            x[0] = (1.0, 0.0);
            x
        });
        for i in 0..n {
            out[i * n + k] = v[i];
        }
    }
    let out_re: Vec<f64> = out.iter().map(|z| z.0).collect();
    let out_im: Vec<f64> = out.iter().map(|z| z.1).collect();
    let data = Array::from_shape_vec(IxDyn(&[n, n]), out_re)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let imag = Array::from_shape_vec(IxDyn(&[n, n]), out_im)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        data,
        imag: Some(imag),
    })
}

/// 1-D/全归约中位数（语义与 Python _median_py axis=None 对齐）：
/// nan_filter=False（默认，median）——NaN 传染（任一元素 NaN → 结果
/// NaN）；nan_filter=True（nanmedian）——过滤 NaN 后取中位，过滤后
/// 空则 NaN（全 NaN 输入，警告由 Python 侧处理）。排序 complex 按
/// (real, imag) 字典序（同 _ckey）；奇数取 vs[n/2]，偶数取
/// (vs[n/2-1]+vs[n/2])/2。返回 (1,) complex NdArray（实输入 imag=0）。
#[pyfunction]
#[pyo3(signature = (a, nan_filter))]
fn median_core(a: &NdArray, nan_filter: bool) -> PyResult<NdArray> {
    let n = a.data.len();
    if n == 0 {
        return Err(PyValueError::new_err("empty input"));
    }
    let has_imag = a.imag.is_some();
    let mut vals: Vec<(f64, f64)> = match &a.imag {
        None => a.data.iter().map(|&r| (r, 0.0)).collect(),
        Some(im) => a
            .data
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| (r, i))
            .collect(),
    };
    if nan_filter {
        vals.retain(|z| !(z.0.is_nan() || z.1.is_nan()));
    }
    // 中位索引基于过滤后长度（nanmedian 语义：NaN 不计入）
    let n = vals.len();
    // NaN 传染（median，未过滤时）：任一元素 NaN → 结果 NaN
    let nan = vals.iter().any(|z| z.0.is_nan() || z.1.is_nan());
    let r = if nan {
        (f64::NAN, f64::NAN)
    } else {
        if has_imag {
            // complex 按 (real, imag) 字典序（同 _ckey）
            vals.sort_by(|x, y| {
                x.0.partial_cmp(&y.0)
                    .unwrap()
                    .then(x.1.partial_cmp(&y.1).unwrap())
            });
        } else {
            // O(n) 选择替代全排序：中位只需要中位元素本身
            let half = n / 2;
            vals.select_nth_unstable_by(half, |x, y| x.0.partial_cmp(&y.0).unwrap());
        }
        if n % 2 == 1 {
            vals[n / 2]
        } else {
            // 偶数：选出较大中位元素后，在其前缀中选较小中位元素
            if has_imag {
                let x = vals[n / 2 - 1];
                let y = vals[n / 2];
                ((x.0 + y.0) / 2.0, (x.1 + y.1) / 2.0)
            } else {
                let y = vals[n / 2].0;
                let half = n / 2;
                vals[..=half]
                    .select_nth_unstable_by(half - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
                let lo = vals[half - 1].0;
                ((lo + y) / 2.0, 0.0)
            }
        }
    };
    let data = Array::from_shape_vec(IxDyn(&[1]), vec![r.0])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let imag = Array::from_shape_vec(IxDyn(&[1]), vec![r.1])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    // 实数输入返回实通道（Python 侧 tolist 得 float，不是 complex）
    Ok(NdArray {
        data,
        imag: if has_imag { Some(imag) } else { None },
    })
}

/// 1-D/全归约 nanargmax/nanargmin（语义与 Python nanargmax/nanargmin
/// 对齐）：忽略 NaN 后按 (real, imag) 字典序（同 _ckey）取最大/最小
/// 值的原索引（严格比较 → 首个命中）。全 NaN/空 → Err，由 Python
/// 侧 raise ValueError。
#[pyfunction]
#[pyo3(signature = (a, find_min))]
fn nanarg_core(a: &NdArray, find_min: bool) -> PyResult<usize> {
    let vals: Vec<(f64, f64)> = match &a.imag {
        None => a.data.iter().map(|&r| (r, 0.0)).collect(),
        Some(im) => a
            .data
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| (r, i))
            .collect(),
    };
    let mut best_i: Option<usize> = None;
    let mut best_v = (0.0, 0.0);
    for (i, &z) in vals.iter().enumerate() {
        if z.0.is_nan() || z.1.is_nan() {
            continue;
        }
        let take = match best_i {
            None => true,
            Some(_) => {
                if find_min {
                    z.0 < best_v.0 || (z.0 == best_v.0 && z.1 < best_v.1)
                } else {
                    z.0 > best_v.0 || (z.0 == best_v.0 && z.1 > best_v.1)
                }
            }
        };
        if take {
            best_i = Some(i);
            best_v = z;
        }
    }
    best_i.ok_or_else(|| PyValueError::new_err("All-NaN slice encountered"))
}

/// 实数数组是否含 NaN/inf（复数查双通道）——lstsq 等的非有限快速
/// 检查（替代 Python 全量 tolist+逐元素扫描）。
#[pyfunction]
fn has_nonfinite_core(a: &NdArray) -> PyResult<bool> {
    Ok(a.data.iter().any(|v| v.is_nan() || v.is_infinite())
        || a.imag
            .as_ref()
            .is_some_and(|im| im.iter().any(|v| v.is_nan() || v.is_infinite())))
}

/// 实数数组 NaN→val 替换（_replace_nan 的 O(n) 内核，替代布尔掩码
/// setitem 慢路径）。复数输入 → Err（Python 兜底）。
#[pyfunction]
#[pyo3(signature = (a, val))]
fn replace_nan_core(a: &NdArray, val: f64) -> PyResult<NdArray> {
    if a.imag.is_some() {
        return Err(PyValueError::new_err("complex input"));
    }
    let data: Vec<f64> = a
        .data
        .iter()
        .map(|&v| if v.is_nan() { val } else { v })
        .collect();
    let arr = Array::from_shape_vec(IxDyn(a.data.shape()), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        data: arr,
        imag: None,
    })
}

/// 1-D/全归约百分位数（语义与 Python _percentile_1d/'linear' 对齐）：
/// nan_filter=True 过滤 NaN（nanpercentile），False 时 NaN 传染（结果
/// NaN）；q01∈[0,1]，虚指数 vidx = q01·(n-1)，clamp 后线性插值
/// （hi 越界取 NaN，同 at()）。n 为原元素数（含被过滤的 NaN）。
#[pyfunction]
#[pyo3(signature = (a, q01, nan_filter))]
fn percentile_core(a: &NdArray, q01: f64, nan_filter: bool) -> PyResult<f64> {
    let n = a.data.len();
    if n == 0 {
        return Err(PyValueError::new_err("empty input"));
    }
    let mut vals: Vec<f64> = if nan_filter {
        a.data.iter().copied().filter(|v| !v.is_nan()).collect()
    } else {
        a.data.iter().copied().collect()
    };
    // 引擎语义：插值的 n = 过滤后长度（nanpercentile 忽略 NaN 计数）
    let n = vals.len();
    if n == 0 {
        return Err(PyValueError::new_err("empty input"));
    }
    // NaN 传染（median/percentile 未过滤时）：任一 NaN → 结果 NaN
    if !nan_filter && vals.iter().any(|v| v.is_nan()) {
        return Ok(f64::NAN);
    }
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n_f = n as f64;
    let vidx = (q01 * (n_f - 1.0)).clamp(0.0, n_f - 1.0);
    let lo = vidx.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = vidx - lo as f64;
    Ok(vals[lo] + (vals[hi] - vals[lo]) * frac)
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
    m.add_function(wrap_pyfunction!(svd_assemble_core, m)?)?;
    m.add_function(wrap_pyfunction!(qr_householder_core, m)?)?;
    m.add_function(wrap_pyfunction!(eigvals_qr_core, m)?)?;
    m.add_function(wrap_pyfunction!(eig_power_core, m)?)?;
    m.add_function(wrap_pyfunction!(eig_vectors_core, m)?)?;
    m.add_function(wrap_pyfunction!(median_core, m)?)?;
    m.add_function(wrap_pyfunction!(nanarg_core, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_core, m)?)?;
    m.add_function(wrap_pyfunction!(has_nonfinite_core, m)?)?;
    m.add_function(wrap_pyfunction!(replace_nan_core, m)?)?;
    m.add_function(wrap_pyfunction!(eigvals_qr_batch_core, m)?)?;
    Ok(())
}
