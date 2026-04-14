# Parensnet

Parallel Ensemble 

## Installation

Clone sources with recurse-submodules option so that all the dependent 
sources are cloned.

```
git clone --recurse-submodules https://github.com/srirampc/parensnet_rs.git
```

### Dependencies

Rust programming language and the corresponding tools needs be installed with 
rustup as shown in [rust-lang.org](https://rust-lang.org/learn/get-started/) 
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```


`parensnet_rs` depends on the following rust wrapper libraries:

- `rsmpi` requires a compatible MPI-3.1 implementation,
- modified `hdf5-rust` that includes parallel IO 
  (included as a submodule in ext/hdf5-rust dir) is compiled from source 
  -- requires a HDF5 library installation with parallel io, and
- modified `lightgbm3-rs` (included as a submodule in ext/ dir) 
  is compiled from source (ext/ dir). To
  compile `lightgbm3`, cmake version 3.28 or higher is required.

The following C/C++ libraries needs to be installed and be available via 
standard paths:

1. CMake version >= 3.28
2. MPI implementation (mpicc/mpirun) that works with `rsmpi`
3. HDF5 (h5cc/h5pcc) compiled with MPI support,
4. LLM with libclang  (for `bindgen` support). 

### Dependencies Versions 

We have sucessfully tested with the following versions: 

1. CMake version 3.31.9
2. MPI: Open MPI v 5.0.9 and v 5.0.9
3. HDF5 :  1.14.6  compiled with Opend MPI
4. LLVM : 21.1.8 and 18.1.3. **Doesn't work with v22 due to bindgen error.**

### Build 

Souces can be built with cargo

```
cargo b --release
```
