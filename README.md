# Parensnet

Parallel Ensemble Gene Networks for large-scale single cell data. Generate in 
parallel integrated networks from MI, GRNBoost and PIDC based networks.

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

- `sope` an MPI-3 interface library (included as a submodel in ext/sope).
   `sope` is built on top of `rsmpi` with additional utilties and sorting 
   algorithms. Requires a compatible MPI-3.1 implementation,
- a modified `hdf5-rust` that includes parallel IO 
  (included as a submodule in ext/hdf5-rust dir) is compiled from source 
  -- requires a HDF5 library installation with parallel io, and
- a modified `lightgbm3-rs` (included as a submodule in ext/ dir) 
  is compiled from source (ext/ dir). To
  compile `lightgbm3`, cmake version 3.28 or higher is required.
- `mcpnet_rs`, a rust wrapper around MCPNet library to run the kernels 
   available with the MCPNet library. (included as a submodule in ext/ dir)  

The following C/C++ libraries needs to be installed and be available via 
standard paths:

1. CMake version >= 3.29
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
