// SPDX-License-Identifier: MIT
use ndarray::{Array, IxDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rand::{Rng, SeedableRng};
use rand_distr::Distribution;
use rayon::prelude::*;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use crate::NdArray;

thread_local! {
    static THREAD_RNG: std::cell::RefCell<Option<::rand::rngs::StdRng>> = const { std::cell::RefCell::new(None) };
}

static GLOBAL_SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

fn get_global_seed() -> u64 {
    *GLOBAL_SEED.get_or_init(rand::random)
}

fn with_thread_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut ::rand::rngs::StdRng) -> R,
{
    THREAD_RNG.with(|cell| {
        let mut rng = cell.borrow_mut();
        if rng.is_none() {
            let seed = get_global_seed();
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::thread::current().id().hash(&mut hasher);
            let thread_seed = seed.wrapping_add(hasher.finish().wrapping_mul(0x9e3779b97f4a7c15));
            *rng = Some(::rand::rngs::StdRng::seed_from_u64(thread_seed));
        }
        f(rng.as_mut().unwrap())
    })
}

fn parse_shape_from_args(args: &Bound<'_, PyTuple>) -> Vec<usize> {
    let mut shape = Vec::with_capacity(args.len());
    for item in args.iter() {
        if let Ok(val) = item.extract::<usize>() {
            shape.push(val);
        }
    }
    shape
}

fn parse_size_arg(size: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<usize>> {
    match size {
        None => Ok(vec![]),
        Some(s) => {
            if let Ok(val) = s.extract::<usize>() {
                Ok(vec![val])
            } else if let Ok(tup) = s.cast::<PyTuple>() {
                let mut result = Vec::with_capacity(tup.len());
                for item in tup.iter() {
                    result.push(item.extract::<usize>()?);
                }
                Ok(result)
            } else if let Ok(list) = s.cast::<pyo3::types::PyList>() {
                let mut result = Vec::with_capacity(list.len());
                for item in list.iter() {
                    result.push(item.extract::<usize>()?);
                }
                Ok(result)
            } else {
                Err(PyValueError::new_err("size must be an integer or tuple"))
            }
        }
    }
}

/// splitmix64：把一个 64 位种子快速扩散成新的 64 位种子，
/// 用于从一个基种子派生出互不相关的分块种子。
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn make_ndarray_parallel(
    shape: &[usize],
    base_seed: u64,
    dist: impl rand_distr::Distribution<f64> + Sync + Send + Clone + 'static,
) -> NdArray {
    // 固定分块大小：让分块边界（以及每块派生的种子）与线程数无关，
    // 从而保证相同 base_seed 得到完全一致、可复现的结果。
    const CHUNK: usize = 1 << 16;
    let total: usize = shape.iter().product();

    let mut values = vec![0.0f64; total];

    values
        .par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(i, chunk)| {
            let seed = splitmix64(base_seed ^ (i as u64).wrapping_mul(0x9e3779b97f4a7c15));
            let mut local_rng = ::rand::rngs::StdRng::seed_from_u64(seed);
            for elem in chunk {
                *elem = dist.sample(&mut local_rng);
            }
        });

    let arr = Array::from_shape_vec(IxDyn(shape), values).unwrap();
    NdArray {
        imag: None,
        data: arr,
    }
}

fn make_ndarray_single(
    rng: &mut impl rand::Rng,
    shape: &[usize],
    dist: impl rand_distr::Distribution<f64>,
) -> NdArray {
    if shape.is_empty() {
        return NdArray {
            imag: None,
            data: Array::from_elem(IxDyn(&[]), dist.sample(rng)),
        };
    }
    let total: usize = shape.iter().product();

    let mut values = Vec::with_capacity(total);
    values.resize(total, 0.0);

    for elem in values.iter_mut() {
        *elem = dist.sample(rng);
    }

    let arr = Array::from_shape_vec(IxDyn(shape), values).unwrap();
    NdArray {
        imag: None,
        data: arr,
    }
}

#[pyfunction]
fn seed(val: u64) {
    let _ = GLOBAL_SEED.set(val);
    THREAD_RNG.with(|rng| {
        *rng.borrow_mut() = Some(::rand::rngs::StdRng::seed_from_u64(val));
    });
}

#[pyfunction]
fn get_state() -> u64 {
    get_global_seed()
}

#[pyfunction]
#[pyo3(signature = (*args), name = "rand")]
fn random_rand(_py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<NdArray> {
    let shape = parse_shape_from_args(args);
    let dist = ::rand_distr::Uniform::<f64>::new(0.0, 1.0)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    if shape.is_empty() {
        with_thread_rng(|rng| Ok(make_ndarray_single(rng, &shape, dist)))
    } else {
        let base = with_thread_rng(|rng| rng.next_u64());
        Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
    }
}

#[pyfunction]
#[pyo3(signature = (*args))]
fn randn(_py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<NdArray> {
    let shape = parse_shape_from_args(args);
    let normal =
        ::rand_distr::Normal::new(0.0, 1.0).map_err(|e| PyValueError::new_err(e.to_string()))?;

    if shape.is_empty() {
        with_thread_rng(|rng| Ok(make_ndarray_single(rng, &shape, normal)))
    } else {
        let base = with_thread_rng(|rng| rng.next_u64());
        Ok(_py.detach(move || make_ndarray_parallel(&shape, base, normal)))
    }
}

#[pyfunction]
#[pyo3(signature = (low, high, size=None), name = "randint")]
fn random_randint(low: i64, high: i64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist =
        ::rand_distr::Uniform::new(low, high).map_err(|e| PyValueError::new_err(e.to_string()))?;
    with_thread_rng(|rng| {
        if shape.is_empty() {
            return Ok(NdArray {
                imag: None,
                data: Array::from_elem(IxDyn(&[]), dist.sample(rng) as f64),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| dist.sample(rng) as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    })
}

#[pyclass(name = "RandomState")]
struct PyRandomState {
    rng: Mutex<::rand::rngs::StdRng>,
}

#[pymethods]
impl PyRandomState {
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> Self {
        PyRandomState {
            rng: Mutex::new(::rand::rngs::StdRng::seed_from_u64(
                seed.unwrap_or_else(rand::random),
            )),
        }
    }

    #[pyo3(signature = (*args))]
    fn rand(&self, _py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<NdArray> {
        let shape = parse_shape_from_args(args);
        let dist = ::rand_distr::Uniform::<f64>::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if shape.is_empty() {
            let mut rng = self.rng.lock().unwrap();
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            let base = self.rng.lock().unwrap().next_u64();
            Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
        }
    }

    #[pyo3(signature = (*args))]
    fn randn(&self, _py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<NdArray> {
        let shape = parse_shape_from_args(args);
        let normal = ::rand_distr::Normal::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if shape.is_empty() {
            let mut rng = self.rng.lock().unwrap();
            Ok(make_ndarray_single(&mut rng, &shape, normal))
        } else {
            let base = self.rng.lock().unwrap().next_u64();
            Ok(_py.detach(move || make_ndarray_parallel(&shape, base, normal)))
        }
    }

    #[pyo3(signature = (low, high=None, size=None))]
    fn randint(
        &self,
        low: i64,
        high: Option<i64>,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NdArray> {
        let hi = high.unwrap_or(low);
        let actual_low = if high.is_some() { low } else { 0 };
        let actual_high = hi;
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Uniform::new(actual_low, actual_high)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = self.rng.lock().unwrap();
        if shape.is_empty() {
            return Ok(NdArray {
                imag: None,
                data: Array::from_elem(IxDyn(&[]), dist.sample(&mut rng) as f64),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| dist.sample(&mut rng) as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    }

    #[pyo3(signature = (size=None))]
    fn random(&self, _py: Python<'_>, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Uniform::<f64>::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if shape.is_empty() {
            let mut rng = self.rng.lock().unwrap();
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            let base = self.rng.lock().unwrap().next_u64();
            Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
        }
    }

    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn normal(
        &self,
        _py: Python<'_>,
        loc: f64,
        scale: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Normal::new(loc, scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if shape.is_empty() {
            let mut rng = self.rng.lock().unwrap();
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            let base = self.rng.lock().unwrap().next_u64();
            Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
        }
    }

    #[pyo3(signature = (low=0.0, high=1.0, size=None))]
    fn uniform(
        &self,
        _py: Python<'_>,
        low: f64,
        high: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Uniform::new(low, high)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if shape.is_empty() {
            let mut rng = self.rng.lock().unwrap();
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            let base = self.rng.lock().unwrap().next_u64();
            Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
        }
    }

    fn seed(&self, val: u64) {
        *self.rng.lock().unwrap() = ::rand::rngs::StdRng::seed_from_u64(val);
    }
}

#[pyclass(name = "Generator")]
struct PyGenerator {
    rng: Mutex<::rand::rngs::StdRng>,
}

impl PyGenerator {
    fn spawn(&self) -> ::rand::rngs::StdRng {
        ::rand::rngs::StdRng::seed_from_u64(self.rng.lock().unwrap().next_u64())
    }
}

#[pymethods]
impl PyGenerator {
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> Self {
        PyGenerator {
            rng: Mutex::new(::rand::rngs::StdRng::seed_from_u64(
                seed.unwrap_or_else(rand::random),
            )),
        }
    }

    #[pyo3(signature = (size=None))]
    fn random(&self, _py: Python<'_>, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Uniform::<f64>::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if shape.is_empty() {
            let mut rng = self.spawn();
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            let base = self.rng.lock().unwrap().next_u64();
            Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
        }
    }

    #[pyo3(signature = (size=None))]
    fn standard_normal(
        &self,
        _py: Python<'_>,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let normal = ::rand_distr::Normal::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if shape.is_empty() {
            let mut rng = self.spawn();
            Ok(make_ndarray_single(&mut rng, &shape, normal))
        } else {
            let base = self.rng.lock().unwrap().next_u64();
            Ok(_py.detach(move || make_ndarray_parallel(&shape, base, normal)))
        }
    }

    #[pyo3(signature = (low, high=None, size=None, endpoint=false))]
    fn integers(
        &self,
        low: i64,
        high: Option<i64>,
        size: Option<&Bound<'_, PyAny>>,
        endpoint: bool,
    ) -> PyResult<NdArray> {
        let hi = high.unwrap_or(low);
        let actual_low = if high.is_some() { low } else { 0 };
        let actual_high = if endpoint { hi + 1 } else { hi };
        if actual_high <= actual_low {
            return Err(PyValueError::new_err("high must be > low"));
        }
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Uniform::new(actual_low, actual_high)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = self.spawn();
        if shape.is_empty() {
            return Ok(NdArray {
                imag: None,
                data: Array::from_elem(IxDyn(&[]), dist.sample(&mut rng) as f64),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| dist.sample(&mut rng) as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    }

    #[pyo3(signature = (a, size=None, replace=true))]
    fn choice<'py>(
        &self,
        py: Python<'py>,
        a: &Bound<'_, PyAny>,
        size: Option<usize>,
        replace: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut rng = self.spawn();
        let vals: Vec<f64> = if let Ok(arr) = a.extract::<NdArray>() {
            arr.data.iter().copied().collect()
        } else if let Ok(list) = a.cast::<pyo3::types::PyList>() {
            let mut v = Vec::with_capacity(list.len());
            for item in list.iter() {
                v.push(item.extract::<f64>()?);
            }
            v
        } else {
            return Err(PyTypeError::new_err("a must be array-like"));
        };

        if vals.is_empty() {
            return Err(PyValueError::new_err("a must be non-empty"));
        }

        let n = size.unwrap_or(1);
        if !replace && n > vals.len() {
            return Err(PyValueError::new_err(
                "Cannot take a larger sample than population when 'replace=false'",
            ));
        }

        let dist = ::rand_distr::Uniform::new(0usize, vals.len())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let result: Vec<f64> = if replace {
            (0..n).map(|_| vals[dist.sample(&mut rng)]).collect()
        } else {
            let mut indices: Vec<usize> = (0..vals.len()).collect();
            for i in (1..vals.len()).rev() {
                let j_dist = ::rand_distr::Uniform::new(0usize, i + 1)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let j = j_dist.sample(&mut rng);
                indices.swap(i, j);
            }
            indices.truncate(n);
            indices.into_iter().map(|i| vals[i]).collect()
        };

        if n == 1 && size.is_none() {
            return Ok(result[0].into_pyobject(py).unwrap().into_any());
        }

        let arr = Array::from_shape_vec(IxDyn(&[n]), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Bound::new(
            py,
            NdArray {
                imag: None,
                data: arr,
            },
        )?
        .into_any())
    }

    fn shuffle(&self, a: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut rng = self.spawn();
        if let Ok(mut arr) = a.extract::<NdArray>() {
            if arr.data.ndim() != 1 {
                return Err(PyValueError::new_err("shuffle only works for 1D arrays"));
            }
            let len = arr.data.len();
            let mut result: Vec<f64> = arr.data.iter().copied().collect();
            for i in (1..len).rev() {
                let j_dist = ::rand_distr::Uniform::new(0usize, i + 1)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let j = j_dist.sample(&mut rng);
                result.swap(i, j);
            }
            arr.data = Array::from_shape_vec(IxDyn(&[len]), result).unwrap();
            Ok(())
        } else {
            Err(PyTypeError::new_err("Expected ndarray"))
        }
    }

    #[pyo3(signature = (a))]
    fn permutation<'py>(
        &self,
        py: Python<'py>,
        a: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut rng = self.spawn();
        if let Ok(n) = a.extract::<usize>() {
            let mut indices: Vec<f64> = (0..n).map(|i| i as f64).collect();
            for i in (1..n).rev() {
                let j_dist = ::rand_distr::Uniform::new(0usize, i + 1)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let j = j_dist.sample(&mut rng);
                indices.swap(i, j);
            }
            let arr = Array::from_shape_vec(IxDyn(&[n]), indices)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Bound::new(
                py,
                NdArray {
                    imag: None,
                    data: arr,
                },
            )?
            .into_any())
        } else if let Ok(arr) = a.extract::<NdArray>() {
            let vals: Vec<f64> = arr.data.iter().copied().collect();
            let n = vals.len();
            let mut indices: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j_dist = ::rand_distr::Uniform::new(0usize, i + 1)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let j = j_dist.sample(&mut rng);
                indices.swap(i, j);
            }
            let result: Vec<f64> = indices.into_iter().map(|i| vals[i]).collect();
            let result_arr = Array::from_shape_vec(arr.data.dim(), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Bound::new(
                py,
                NdArray {
                    imag: None,
                    data: result_arr.into_dyn(),
                },
            )?
            .into_any())
        } else {
            Err(PyTypeError::new_err("Expected integer or ndarray"))
        }
    }

    #[pyo3(signature = (low=0.0, high=1.0, size=None))]
    fn uniform(
        &self,
        _py: Python<'_>,
        low: f64,
        high: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Uniform::new(low, high)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if shape.is_empty() {
            let mut rng = self.spawn();
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            let base = self.rng.lock().unwrap().next_u64();
            Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
        }
    }

    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn normal(
        &self,
        _py: Python<'_>,
        loc: f64,
        scale: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Normal::new(loc, scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if shape.is_empty() {
            let mut rng = self.spawn();
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            let base = self.rng.lock().unwrap().next_u64();
            Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
        }
    }

    #[pyo3(signature = (a, b, size=None))]
    fn beta(
        &self,
        _py: Python<'_>,
        a: f64,
        b: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist =
            ::rand_distr::Beta::new(a, b).map_err(|e| PyValueError::new_err(e.to_string()))?;

        if shape.is_empty() {
            let mut rng = self.spawn();
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            let base = self.rng.lock().unwrap().next_u64();
            Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
        }
    }

    #[pyo3(signature = (shape_param, scale=1.0, size=None))]
    fn gamma(
        &self,
        _py: Python<'_>,
        shape_param: f64,
        scale: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Gamma::new(shape_param, scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if shape.is_empty() {
            let mut rng = self.spawn();
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            let base = self.rng.lock().unwrap().next_u64();
            Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
        }
    }

    #[pyo3(signature = (scale=1.0, size=None))]
    fn exponential(
        &self,
        _py: Python<'_>,
        scale: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Exp::new(1.0 / scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if shape.is_empty() {
            let mut rng = self.spawn();
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            let base = self.rng.lock().unwrap().next_u64();
            Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
        }
    }

    #[pyo3(signature = (n, p, size=None))]
    fn binomial(&self, n: u64, p: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist =
            ::rand_distr::Binomial::new(n, p).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = self.spawn();
        if shape.is_empty() {
            return Ok(NdArray {
                imag: None,
                data: Array::from_elem(IxDyn(&[]), dist.sample(&mut rng) as f64),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| dist.sample(&mut rng) as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    }

    #[pyo3(signature = (lam, size=None))]
    fn poisson(&self, lam: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist =
            ::rand_distr::Poisson::new(lam).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = self.spawn();
        if shape.is_empty() {
            return Ok(NdArray {
                imag: None,
                data: Array::from_elem(IxDyn(&[]), dist.sample(&mut rng)),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| dist.sample(&mut rng)).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    }

    #[pyo3(signature = (a, scale=1.0, size=None))]
    fn weibull(
        &self,
        _py: Python<'_>,
        a: f64,
        scale: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Weibull::new(a, scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if shape.is_empty() {
            let mut rng = self.spawn();
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            let base = self.rng.lock().unwrap().next_u64();
            Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
        }
    }

    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn logistic(&self, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let uniform = ::rand_distr::Uniform::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = self.spawn();
        if shape.is_empty() {
            let u: f64 = uniform.sample(&mut rng);
            let val = loc + scale * (u / (1.0 - u)).ln();
            return Ok(NdArray {
                imag: None,
                data: Array::from_elem(IxDyn(&[]), val),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total)
            .map(|_| {
                let u: f64 = uniform.sample(&mut rng);
                loc + scale * (u / (1.0 - u)).ln()
            })
            .collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    }

    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn cauchy(
        &self,
        _py: Python<'_>,
        loc: f64,
        scale: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Cauchy::new(loc, scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if shape.is_empty() {
            let mut rng = self.spawn();
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            let base = self.rng.lock().unwrap().next_u64();
            Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
        }
    }

    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn laplace(&self, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let uniform = ::rand_distr::Uniform::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = self.spawn();
        if shape.is_empty() {
            let u: f64 = uniform.sample(&mut rng);
            let val = if u < 0.5 {
                loc + scale * (2.0 * u).ln()
            } else {
                loc - scale * (2.0 * (1.0 - u)).ln()
            };
            return Ok(NdArray {
                imag: None,
                data: Array::from_elem(IxDyn(&[]), val),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total)
            .map(|_| {
                let u: f64 = uniform.sample(&mut rng);
                if u < 0.5 {
                    loc + scale * (2.0 * u).ln()
                } else {
                    loc - scale * (2.0 * (1.0 - u)).ln()
                }
            })
            .collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    }
}

#[pyfunction]
#[pyo3(signature = (low=0.0, high=1.0, size=None))]
fn uniform(
    _py: Python<'_>,
    low: f64,
    high: f64,
    size: Option<&Bound<'_, PyAny>>,
) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist =
        ::rand_distr::Uniform::new(low, high).map_err(|e| PyValueError::new_err(e.to_string()))?;

    if shape.is_empty() {
        with_thread_rng(|rng| Ok(make_ndarray_single(rng, &shape, dist)))
    } else {
        let base = with_thread_rng(|rng| rng.next_u64());
        Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
    }
}

#[pyfunction]
#[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
fn normal(
    _py: Python<'_>,
    loc: f64,
    scale: f64,
    size: Option<&Bound<'_, PyAny>>,
) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist =
        ::rand_distr::Normal::new(loc, scale).map_err(|e| PyValueError::new_err(e.to_string()))?;

    if shape.is_empty() {
        with_thread_rng(|rng| Ok(make_ndarray_single(rng, &shape, dist)))
    } else {
        let base = with_thread_rng(|rng| rng.next_u64());
        Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, size=None))]
fn beta(_py: Python<'_>, a: f64, b: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist = ::rand_distr::Beta::new(a, b).map_err(|e| PyValueError::new_err(e.to_string()))?;

    if shape.is_empty() {
        with_thread_rng(|rng| Ok(make_ndarray_single(rng, &shape, dist)))
    } else {
        let base = with_thread_rng(|rng| rng.next_u64());
        Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
    }
}

#[pyfunction]
#[pyo3(signature = (shape_param, scale=1.0, size=None))]
fn gamma(
    _py: Python<'_>,
    shape_param: f64,
    scale: f64,
    size: Option<&Bound<'_, PyAny>>,
) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist = ::rand_distr::Gamma::new(shape_param, scale)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    if shape.is_empty() {
        with_thread_rng(|rng| Ok(make_ndarray_single(rng, &shape, dist)))
    } else {
        let base = with_thread_rng(|rng| rng.next_u64());
        Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
    }
}

#[pyfunction]
#[pyo3(signature = (scale=1.0, size=None))]
fn exponential(_py: Python<'_>, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist =
        ::rand_distr::Exp::new(1.0 / scale).map_err(|e| PyValueError::new_err(e.to_string()))?;

    if shape.is_empty() {
        with_thread_rng(|rng| Ok(make_ndarray_single(rng, &shape, dist)))
    } else {
        let base = with_thread_rng(|rng| rng.next_u64());
        Ok(_py.detach(move || make_ndarray_parallel(&shape, base, dist)))
    }
}

// ---------------------------------------------------------------------------
// MT19937 位级流（legacy RandomState 位兼容）：播种在 Python 侧完成（已与
// 标准逐位对照），此类只负责 twist/u32/res53/有界区间/极坐标正态的批量
// 流生成；状态（mt 缓冲 + 高斯缓存）是单一事实源，经 state/set_state 同步。
// ---------------------------------------------------------------------------
#[pyclass]
struct Mt19937Stream {
    mt: [u32; 624],
    mti: usize,
    gauss_cache: Option<f64>,
}

/// 最小位掩码 ≥ m（顶位以下全置 1）；等价 Python (1<<m.bit_length())-1，
/// 覆盖 u64 全域（含 2^64-1，无移位溢出）
#[inline]
fn gen_mask64(mut m: u64) -> u64 {
    m |= m >> 1;
    m |= m >> 2;
    m |= m >> 4;
    m |= m >> 8;
    m |= m >> 16;
    m |= m >> 32;
    m
}

impl Mt19937Stream {
    /// 最小位掩码 ≥ m（顶位以下全置 1）；u32 上 >>32 恒为 0，与 Python 版等价
    #[inline]
    fn gen_mask(mut m: u32) -> u32 {
        m |= m >> 1;
        m |= m >> 2;
        m |= m >> 4;
        m |= m >> 8;
        m |= m >> 16;
        m
    }
}

#[pymethods]
impl Mt19937Stream {
    #[new]
    #[pyo3(signature = (mt, mti=624))]
    fn new(mt: Vec<u32>, mti: usize) -> PyResult<Self> {
        if mt.len() != 624 {
            return Err(PyValueError::new_err("mt must have 624 entries"));
        }
        let mut arr = [0u32; 624];
        arr.copy_from_slice(&mt);
        Ok(Self {
            mt: arr,
            mti: mti.min(624),
            gauss_cache: None,
        })
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.mt[i] & 0x80000000) + (self.mt[(i + 1) % 624] & 0x7fffffff);
            let mut v = self.mt[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                v ^= 0x9908b0df;
            }
            self.mt[i] = v;
        }
        self.mti = 0;
    }

    fn u32(&mut self) -> u32 {
        if self.mti >= 624 {
            self.twist();
        }
        let mut x = self.mt[self.mti];
        self.mti += 1;
        x ^= x >> 11;
        x ^= (x << 7) & 0x9d2c5680;
        x ^= (x << 15) & 0xefc60000;
        x ^= x >> 18;
        x
    }

    /// res53 double：两个 32 位输出合成 53 位（(a*2^26+b)/2^53）
    fn res53(&mut self) -> f64 {
        let a = self.u32() >> 5;
        let b = self.u32() >> 6;
        (a as f64 * 67108864.0 + b as f64) * (1.0 / 9007199254740992.0)
    }

    #[pyo3(name = "random_sample")]
    fn random_sample1(&mut self) -> f64 {
        self.res53()
    }

    fn random_samples(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.res53()).collect()
    }

    /// rk_interval：掩码拒绝（与 Python 现行实现逐位一致；maxv 支持至
    /// 2^64-1，>2^32 恒接受——与 Python v=u32&mask 行为一致）
    fn interval(&mut self, maxv: u64) -> u64 {
        if maxv == 0 {
            return 0;
        }
        let mask = gen_mask64(maxv);
        loop {
            let v = self.u32() as u64 & mask;
            if v <= maxv {
                return v;
            }
        }
    }

    fn intervals(&mut self, rngs: Vec<u64>) -> Vec<u64> {
        rngs.into_iter().map(|r| self.interval(r)).collect()
    }

    /// legacy _rand_uint8：每 32-bit 字出 4 个 8-bit 值（LSB 起）、掩码拒绝，
    /// 单次调用内 buf 跨元素共享
    // 返回 Vec<u32> 而非 Vec<u8>：pyo3 会把 Vec<u8> 转成 bytes 而非 int 列表
    fn intervals_u8(&mut self, rngs: Vec<u32>) -> Vec<u32> {
        let mut out = Vec::with_capacity(rngs.len());
        let mut buf: u32 = 0;
        let mut bcnt: u32 = 0;
        for rng in rngs {
            if rng == 0 {
                out.push(0);
                continue;
            }
            let mask = Self::gen_mask(rng);
            let val = loop {
                if bcnt == 0 {
                    buf = self.u32();
                    bcnt = 3;
                } else {
                    buf >>= 8;
                    bcnt -= 1;
                }
                let v = (buf & 0xff) & mask;
                if v <= rng {
                    break v;
                }
            };
            out.push(val);
        }
        out
    }

    /// legacy _rand_uint16：每 32-bit 字出 2 个 16-bit 值（低半字先）
    fn intervals_u16(&mut self, rngs: Vec<u32>) -> Vec<u16> {
        let mut out = Vec::with_capacity(rngs.len());
        let mut buf: u32 = 0;
        let mut bcnt: u32 = 0;
        for rng in rngs {
            if rng == 0 {
                out.push(0);
                continue;
            }
            let mask = Self::gen_mask(rng);
            let val = loop {
                if bcnt == 0 {
                    buf = self.u32();
                    bcnt = 1;
                } else {
                    buf >>= 16;
                    bcnt -= 1;
                }
                let v = (buf & 0xffff) & mask;
                if v <= rng {
                    break v;
                }
            };
            out.push(val as u16);
        }
        out
    }

    /// legacy _rand_bool：每 32-bit 字出 32 个 bit（LSB 起）、无拒绝；
    /// rng==0 元素不消耗随机数
    fn intervals_bool(&mut self, rngs: Vec<u32>) -> Vec<u32> {
        let mut out = Vec::with_capacity(rngs.len());
        let mut buf: u32 = 0;
        let mut bcnt: u32 = 0;
        for rng in rngs {
            if rng == 0 {
                out.push(0);
                continue;
            }
            if bcnt == 0 {
                buf = self.u32();
                bcnt = 31;
            } else {
                buf >>= 1;
                bcnt -= 1;
            }
            out.push(u32::from(buf & 1 != 0));
        }
        out
    }

    /// legacy bounded-integer 标量 lo/hi 内核（与 Python intervals_u8/u16/
    /// intervals 标量路径逐位一致）：u8=4 值/字、u16=2 值/字（LSB 起、掩码
    /// 拒绝）、其他宽度=interval（1 字/值）。直接返回最终值（lo+v）的
    /// f64 列表，避免 2N 元素 int 经 Python 往返。rng = hi-1-lo（闭包上界）。
    fn bounded_integers(&mut self, lo: i64, rng: u64, n: usize, bits: u32) -> Vec<f64> {
        let lof = lo as f64;
        let mut out = Vec::with_capacity(n);
        if bits == 8 {
            let mask = gen_mask64(rng as u32 as u64);
            let mut buf: u32 = 0;
            let mut bcnt: u32 = 0;
            for _ in 0..n {
                if rng == 0 {
                    out.push(lof);
                    continue;
                }
                let v = loop {
                    if bcnt == 0 {
                        buf = self.u32();
                        bcnt = 3;
                    } else {
                        buf >>= 8;
                        bcnt -= 1;
                    }
                    let c = (buf & 0xff) as u64 & mask;
                    if c <= rng {
                        break c;
                    }
                };
                out.push(lof + v as f64);
            }
        } else if bits == 16 {
            let mask = gen_mask64(rng as u32 as u64);
            let mut buf: u32 = 0;
            let mut bcnt: u32 = 0;
            for _ in 0..n {
                if rng == 0 {
                    out.push(lof);
                    continue;
                }
                let v = loop {
                    if bcnt == 0 {
                        buf = self.u32();
                        bcnt = 1;
                    } else {
                        buf >>= 16;
                        bcnt -= 1;
                    }
                    let c = (buf & 0xffff) as u64 & mask;
                    if c <= rng {
                        break c;
                    }
                };
                out.push(lof + v as f64);
            }
        } else {
            for v in self.intervals(vec![rng; n]) {
                out.push(lof + v as f64);
            }
        }
        out
    }

    /// legacy rk_gauss（极坐标 Box-Muller，带缓存；缓存存于流状态，
    /// 跨调用保留）
    fn standard_normals(&mut self, n: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            if let Some(c) = self.gauss_cache.take() {
                out.push(c);
                continue;
            }
            let (v1, v2, s) = loop {
                let v1 = 2.0 * self.res53() - 1.0;
                let v2 = 2.0 * self.res53() - 1.0;
                let s = v1 * v1 + v2 * v2;
                if s > 0.0 && s < 1.0 {
                    break (v1, v2, s);
                }
            };
            let r = (-2.0 * s.ln() / s).sqrt();
            self.gauss_cache = Some(v1 * r);
            out.push(v2 * r);
        }
        out
    }

    fn gauss_cache(&self) -> Option<f64> {
        self.gauss_cache
    }

    fn state(&self) -> (Vec<u32>, usize) {
        (self.mt.to_vec(), self.mti)
    }

    fn set_state(&mut self, mt: Vec<u32>, mti: usize, gauss_cache: Option<f64>) -> PyResult<()> {
        if mt.len() != 624 {
            return Err(PyValueError::new_err("mt must have 624 entries"));
        }
        self.mt.copy_from_slice(&mt);
        self.mti = mti.min(624);
        self.gauss_cache = gauss_cache;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PCG64 位级流（XSL-RR-128-64）：播种（SeedSequence → initstate/inc 链）在
// Python 侧完成（已逐位对照），此类只负责 step/next64/next32/通用 res53/
// random_interval/极坐标正态的批量流生成。
// ---------------------------------------------------------------------------
const PCG64_MULT128: u128 = (2549297995355413924u128 << 64) | 4865540595714422341;

#[pyclass]
struct Pcg64Stream {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
    gauss_cache: Option<f64>,
}

#[pymethods]
impl Pcg64Stream {
    #[new]
    #[pyo3(signature = (state, inc, has_uint32=0, uinteger=0))]
    fn new(state: u128, inc: u128, has_uint32: u32, uinteger: u32) -> Self {
        Self {
            state,
            inc,
            has_uint32: has_uint32 != 0,
            uinteger,
            gauss_cache: None,
        }
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG64_MULT128)
            .wrapping_add(self.inc);
    }

    /// 一个 64 位输出（先步进后输出，XSL-RR：hi^lo 旋转）
    fn next64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        let x = hi ^ lo;
        let rot = (self.state >> 122) as u32;
        x.rotate_right(rot)
    }

    fn next32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            self.uinteger
        } else {
            let n = self.next64();
            self.has_uint32 = true;
            self.uinteger = (n >> 32) as u32;
            n as u32
        }
    }

    /// 通用位生成器 uniform：(next64 >> 11) / 2^53（取高 53 位）
    fn uniform_sample(&mut self) -> f64 {
        let n = self.next64();
        (n >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    #[pyo3(name = "random_sample")]
    fn random_sample1(&mut self) -> f64 {
        self.uniform_sample()
    }

    fn random_samples(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.uniform_sample()).collect()
    }

    fn next32_one(&mut self) -> u32 {
        self.next32()
    }

    fn next64_one(&mut self) -> u64 {
        self.next64()
    }

    /// random_interval（distributions.c）：掩码拒绝 + uint32 流
    /// （i ≤ 2^32-1 走 next32，否则走 next64）
    fn intervals(&mut self, rngs: Vec<u64>) -> Vec<u64> {
        let mut out = Vec::with_capacity(rngs.len());
        for i in rngs {
            if i == 0 {
                out.push(0);
                continue;
            }
            let mask = gen_mask64(i);
            let v = if i <= u64::from(u32::MAX) {
                loop {
                    let v = self.next32() as u64 & mask;
                    if v <= i {
                        break v;
                    }
                }
            } else {
                loop {
                    let v = self.next64() & mask;
                    if v <= i {
                        break v;
                    }
                }
            };
            out.push(v);
        }
        out
    }

    /// 缓冲式有界采样（next32 字为源）：与 MT19937 版同构，仅字源不同
    // 返回 Vec<u32> 而非 Vec<u8>：pyo3 会把 Vec<u8> 转成 bytes 而非 int 列表
    fn intervals_u8(&mut self, rngs: Vec<u32>) -> Vec<u32> {
        let mut out = Vec::with_capacity(rngs.len());
        let mut buf: u32 = 0;
        let mut bcnt: u32 = 0;
        for rng in rngs {
            if rng == 0 {
                out.push(0);
                continue;
            }
            let mask = Mt19937Stream::gen_mask(rng);
            let val = loop {
                if bcnt == 0 {
                    buf = self.next32();
                    bcnt = 3;
                } else {
                    buf >>= 8;
                    bcnt -= 1;
                }
                let v = (buf & 0xff) & mask;
                if v <= rng {
                    break v;
                }
            };
            out.push(val);
        }
        out
    }

    fn intervals_u16(&mut self, rngs: Vec<u32>) -> Vec<u16> {
        let mut out = Vec::with_capacity(rngs.len());
        let mut buf: u32 = 0;
        let mut bcnt: u32 = 0;
        for rng in rngs {
            if rng == 0 {
                out.push(0);
                continue;
            }
            let mask = Mt19937Stream::gen_mask(rng);
            let val = loop {
                if bcnt == 0 {
                    buf = self.next32();
                    bcnt = 1;
                } else {
                    buf >>= 16;
                    bcnt -= 1;
                }
                let v = (buf & 0xffff) & mask;
                if v <= rng {
                    break v;
                }
            };
            out.push(val as u16);
        }
        out
    }

    fn intervals_bool(&mut self, rngs: Vec<u32>) -> Vec<u32> {
        let mut out = Vec::with_capacity(rngs.len());
        let mut buf: u32 = 0;
        let mut bcnt: u32 = 0;
        for rng in rngs {
            if rng == 0 {
                out.push(0);
                continue;
            }
            if bcnt == 0 {
                buf = self.next32();
                bcnt = 31;
            } else {
                buf >>= 1;
                bcnt -= 1;
            }
            out.push(u32::from(buf & 1 != 0));
        }
        out
    }

    /// legacy bounded-integer 标量 lo/hi 内核（与 Python intervals_u8/u16/
    /// intervals 标量路径逐位一致；字源为 next32）：u8=4 值/字、u16=2 值/字
    /// （LSB 起、掩码拒绝）、其他宽度=interval（≤2^32 走 next32 流、否则
    /// next64 流）。直接返回最终值（lo+v）的 f64 列表。rng = hi-1-lo。
    fn bounded_integers(&mut self, lo: i64, rng: u64, n: usize, bits: u32) -> Vec<f64> {
        let lof = lo as f64;
        let mut out = Vec::with_capacity(n);
        if bits == 8 {
            let mask = gen_mask64(rng as u32 as u64);
            let mut buf: u32 = 0;
            let mut bcnt: u32 = 0;
            for _ in 0..n {
                if rng == 0 {
                    out.push(lof);
                    continue;
                }
                let v = loop {
                    if bcnt == 0 {
                        buf = self.next32();
                        bcnt = 3;
                    } else {
                        buf >>= 8;
                        bcnt -= 1;
                    }
                    let c = (buf & 0xff) as u64 & mask;
                    if c <= rng {
                        break c;
                    }
                };
                out.push(lof + v as f64);
            }
        } else if bits == 16 {
            let mask = gen_mask64(rng as u32 as u64);
            let mut buf: u32 = 0;
            let mut bcnt: u32 = 0;
            for _ in 0..n {
                if rng == 0 {
                    out.push(lof);
                    continue;
                }
                let v = loop {
                    if bcnt == 0 {
                        buf = self.next32();
                        bcnt = 1;
                    } else {
                        buf >>= 16;
                        bcnt -= 1;
                    }
                    let c = (buf & 0xffff) as u64 & mask;
                    if c <= rng {
                        break c;
                    }
                };
                out.push(lof + v as f64);
            }
        } else {
            for v in self.intervals(vec![rng; n]) {
                out.push(lof + v as f64);
            }
        }
        out
    }

    /// 极坐标 Box-Muller（legacy RandomState(PCG64()) 语义，带缓存）
    fn standard_normals(&mut self, n: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            if let Some(c) = self.gauss_cache.take() {
                out.push(c);
                continue;
            }
            let (v1, v2, s) = loop {
                let v1 = 2.0 * self.uniform_sample() - 1.0;
                let v2 = 2.0 * self.uniform_sample() - 1.0;
                let s = v1 * v1 + v2 * v2;
                if s > 0.0 && s < 1.0 {
                    break (v1, v2, s);
                }
            };
            let r = (-2.0 * s.ln() / s).sqrt();
            self.gauss_cache = Some(v1 * r);
            out.push(v2 * r);
        }
        out
    }

    fn gauss_cache(&self) -> Option<f64> {
        self.gauss_cache
    }

    fn state(&self) -> (u128, u128, u32, u32) {
        (
            self.state,
            self.inc,
            u32::from(self.has_uint32),
            self.uinteger,
        )
    }

    fn set_state(&mut self, state: u128, inc: u128, has_uint32: u32, uinteger: u32) {
        self.state = state;
        self.inc = inc;
        self.has_uint32 = has_uint32 != 0;
        self.uinteger = uinteger;
    }
}

pub fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(seed, m)?)?;
    m.add_function(wrap_pyfunction!(random_rand, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(random_randint, m)?)?;
    m.add_function(wrap_pyfunction!(uniform, m)?)?;
    m.add_function(wrap_pyfunction!(normal, m)?)?;
    m.add_function(wrap_pyfunction!(beta, m)?)?;
    m.add_function(wrap_pyfunction!(gamma, m)?)?;
    m.add_function(wrap_pyfunction!(exponential, m)?)?;
    m.add_function(wrap_pyfunction!(get_state, m)?)?;
    m.add_class::<PyRandomState>()?;
    m.add_class::<PyGenerator>()?;
    m.add_class::<Mt19937Stream>()?;
    m.add_class::<Pcg64Stream>()?;
    Ok(())
}
