# Rust_MCS_Python

A Python wrapper for calling the [Rust_MCS](https://github.com/SergeiGL/Rust_MCS) minimizer from Python 3.

## Features

- **Rust‐powered performance**  
   - Leverages Rust’s speed and memory safety.

- **Seamless Python integration**  
   - Uses [maturin](https://www.maturin.rs/) to build and publish a native Python extension. No manual binding code required.

- **Zero-copy NumPy support**  
   - Pass NumPy arrays into Rust, thanks to the Rust FFI and the `ndarray` feature.

- **Cross-platform**
  - Works on Windows, macOS, and Linux as long as you have a compatible Rust toolchain and Python 3.13.

---

## Prerequisites

Before you begin, ensure your system has:

1. **Git**
2. **Conda** (recommended) or another Python 3.13 environment manager  
   - We assume you’ll create an isolated environment named `rust_mcs_python`.  
   - You can substitute `venv` or any other virtual‐environment tool, but the instructions below use Conda.

3. **Rust toolchain**  
   - Install via [rustup](https://rustup.rs/)

4. **Python 3.13**  
   - We recommend using Conda to manage this version.  
   - The wrapper has only been tested on Python 3.13

---

## Installation

Clone this repository and build the native extension using `maturin`:

```bash
# 1. Clone the repository
git clone https://github.com/SergeiGL/Rust_MCS_Python.git
```
```bash
# 2. Move to the downloaded folder
cd Rust_MCS_Python
```
```bash
# 3. Create and activate a new environment
conda create -n rust_mcs_python python=3.13
conda activate rust_mcs_python
```

```bash
# 4. Install build dependencies
pip install maturin numpy
```
```bash
# 5. Build and install the Python extension (in develop mode)
maturin develop --release
```

Your `rust_mcs_python` conda environment is ready to go!

See Python3 code examples in the `Python_Example` folder (`.ipynb` file)
