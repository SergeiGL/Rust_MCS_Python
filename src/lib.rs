use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use nalgebra::{SVector, SMatrix};
use Rust_MCS::{mcs};

use std::cell::RefCell;


thread_local! {
    static PY_FUNC: RefCell<Option<PyObject>> = RefCell::new(None);
}

// Store `f` into TLS so that `rust_callback::<N>` can find it.
// We clone the `PyObject` under the GIL and stash it.
fn set_pyfunc(f: PyObject) {
    PY_FUNC.with(|cell| {
        *cell.borrow_mut() = Some(f);
    });
}

// Clear out the Python function from TLS. Call this immediately after `mcs(...)` returns.
fn clear_pyfunc() {
    PY_FUNC.with(|cell| {
        *cell.borrow_mut() = None;
    });
}


// A real function pointer (monomorphized per `N`) which calls back into Python.
// Signature matches what `mcs(...)` expects: `fn(&SVector<f64, N>) -> f64`.
// Inside, we do `Python::with_gil(...)`, reconstruct a `PyArray1<f64>` from `x`,
// then call the Python function that was stashed by `set_pyfunc`.
fn rust_callback<const N: usize>(x: &SVector<f64, N>) -> f64 {
    Python::with_gil(|py| {
        // Grab the Python function from TLS
        let maybe_func: Option<PyObject> = PY_FUNC.with(|cell| {
            cell
                .borrow()    
                .as_ref()                      // Option<&Py<PyAny>>
                .map(|f: &Py<PyAny>| f.clone_ref(py))
        });
        
        if let Some(py_func) = maybe_func {
            // Turn `x: &SVector<f64, N>` into a Vec<f64> and then into a PyArray1
            let x_vec: Vec<f64> = x.iter().cloned().collect();
            let x_array = PyArray1::from_vec(py, x_vec);

            // Call Python function
            match py_func.call1(py, (x_array,)) {
                Ok(py_val) => {
                    // Try to extract an f64; on failure, return +∞
                    py_val
                        .extract::<f64>(py)
                        .unwrap_or_else(|_| {
                            eprintln!("Warning: Failed to extract f64 from Python function result");
                            f64::INFINITY
                        })
                }
                Err(_) => {
                    eprintln!("Warning: Failed to call Python function");
                    f64::INFINITY
                }
            }
        } else {
            // If for some reason TLS is empty, return +∞
            f64::INFINITY
        }
    })
}

#[pyfunction]
fn mcs_py(
    py: Python,
    n: usize,
    func: PyObject,
    u: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    nsweeps: usize,
    nf: usize,
    local: usize,
    gamma: f64,
    smax: usize,
    hess: PyReadonlyArray2<f64>,
) -> PyResult<(PyObject, f64, usize, usize, String)> {
    match n{
        1 => mcs_wrapper::<1>(py, func, u, v, nsweeps, nf, local, gamma, smax, hess),
        2 => mcs_wrapper::<2>(py, func, u, v, nsweeps, nf, local, gamma, smax, hess),
        3 => mcs_wrapper::<3>(py, func, u, v, nsweeps, nf, local, gamma, smax, hess),
        4 => mcs_wrapper::<4>(py, func, u, v, nsweeps, nf, local, gamma, smax, hess),
        5 => mcs_wrapper::<5>(py, func, u, v, nsweeps, nf, local, gamma, smax, hess),
        6 => mcs_wrapper::<6>(py, func, u, v, nsweeps, nf, local, gamma, smax, hess),
        7 => mcs_wrapper::<7>(py, func, u, v, nsweeps, nf, local, gamma, smax, hess),
        8 => mcs_wrapper::<8>(py, func, u, v, nsweeps, nf, local, gamma, smax, hess),
        9 => mcs_wrapper::<9>(py, func, u, v, nsweeps, nf, local, gamma, smax, hess),
        10 => mcs_wrapper::<10>(py, func, u, v, nsweeps, nf, local, gamma, smax, hess),
        11 => mcs_wrapper::<11>(py, func, u, v, nsweeps, nf, local, gamma, smax, hess),
        12 => mcs_wrapper::<12>(py, func, u, v, nsweeps, nf, local, gamma, smax, hess),
        13 => mcs_wrapper::<13>(py, func, u, v, nsweeps, nf, local, gamma, smax, hess),
        14 => mcs_wrapper::<14>(py, func, u, v, nsweeps, nf, local, gamma, smax, hess),
        15 => mcs_wrapper::<15>(py, func, u, v, nsweeps, nf, local, gamma, smax, hess),
        _ => Err(PyValueError::new_err(format!("N={n} is not supported")))
    }
}


// The generic‐over‐`N` helper that actually does:
// 1) check dimensions,
// 2) stash `func` into TLS,
// 3) call `mcs(rust_callback::<N>, &u_vec, &v_vec, ...)`,
// 4) clear the TLS,
// 5) convert outputs back into Python objects.
fn mcs_wrapper<const N: usize>(
    py: Python,
    func: PyObject,
    u: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    nsweeps: usize,
    nf: usize,
    local: usize,
    gamma: f64,
    smax: usize,
    hess: PyReadonlyArray2<f64>,
) -> PyResult<(PyObject, f64, usize, usize, String)> {
    // 1) Convert numpy arrays to raw slices and check dimensions:
    let u_array = u.as_array();
    let v_array = v.as_array();
    let hess_array = hess.as_array();

    if u_array.len() != N || v_array.len() != N {
        return Err(PyValueError::new_err(format!(
            "Array dimensions don't match N={}. Got u.len()={}, v.len()={}",
            N,
            u_array.len(),
            v_array.len()
        )));
    }
    if hess_array.shape() != [N, N] {
        return Err(PyValueError::new_err(format!(
            "Hessian matrix dimensions don't match {}×{}. Got shape: {:?}",
            N,
            N,
            hess_array.shape()
        )));
    }

    let u_slice = u_array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Failed to get slice from u array"))?;
    let v_slice = v_array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Failed to get slice from v array"))?;
    let hess_slice = hess_array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Failed to get slice from hess array"))?;

    let u_vec = SVector::<f64, N>::from_column_slice(u_slice);
    let v_vec = SVector::<f64, N>::from_column_slice(v_slice);
    let hess_mat = SMatrix::<f64, N, N>::from_column_slice(hess_slice);

    // ————————————————————————————————————————— Stash the Python function —————————————————————————————————————————
    //
    // We clone the `PyObject` under the GIL and put it into thread‐local storage,
    // so that `rust_callback::<N>(…)` can later fetch it.
    set_pyfunc(func.clone_ref(py));

    // 2) Call the original `mcs` with a **function pointer** (`rust_callback::<N>`) instead of a closure.
    let result = mcs(
        rust_callback::<N> as fn(&SVector<f64, N>) -> f64,
        &u_vec,
        &v_vec,
        nsweeps,
        nf,
        local,
        gamma,
        smax,
        &hess_mat,
    );

    // 3) Clear the TLS right away (so we don’t accidentally leak a reference).
    clear_pyfunc();

    // 4) Match on the result from `mcs(…)` and convert back to Python‐friendly types.
    match result {
        Ok((xbest, fbest, _, _, ncall, ncloc, exit_flag)) => {
            // Convert `xbest: SVector<f64, N>` → Vec<f64> → PyArray1 → PyObject
            let xbest_vec: Vec<f64> = xbest.iter().cloned().collect();
            let xbest_py = PyArray1::from_vec(py, xbest_vec).into_pyobject(py)?;

            let exit_flag_str = format!("{:?}", exit_flag);

            Ok((PyObject::from(xbest_py), fbest, ncall, ncloc, exit_flag_str))
        }
        Err(e) => Err(PyValueError::new_err(e)),
    }
}


// Module definition
#[pymodule]
fn rust_mcs_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mcs_py, m)?)?;
    Ok(())
}