# Parensnet

Parensnet is a library to build Parallel Ensemble Gene Networks for large-scale
single cell data. Currenly from MI, GRNBoost and PIDC based networks are 
supported.

## Overview

`parensnet_rs` is a Rust library and a small set of MPI-parallel binaries
that build gene-regulatory networks (GRNs) directly from single-cell
expression matrices stored in AnnData (`.h5ad`) format. All workflows are
driven by a single YAML config file and use parallel HDF5 to read inputs
and persist intermediate / final results.

### Library

The library (`parensnet_rs`) is organised into three groups of modules:

- **Network construction workflows**
  - `pucn` — PUC Network pipeline (histograms → MI / SI / LMR → PUC
    redundancy scoring, after Chan et al., 2017).
  - `gbn`  — GRNBoost-style gradient boosting (after Arboreto by 
    Moerman et al., , 2019) on top of LightGBM, with a cross-validation step
    to pick the boosting-round count.
  - `mcpn` — MCPNet B-spline mutual-information kernels (B-spline weight
    construction and pairwise MI evaluation, after Daub et al., 2004).
- **Information-measure kernels**: `hist` (Bayesian-blocks and Knuth
  histograms), `mvim` (mutual / specific information, Williams–Beer
  redundancy, LMR data structures), and `corr` (B-spline MI kernels
  used by `mcpn`).
- **I/O, MPI plumbing, and common utilities**: `anndata` (AnnData
  reader), `h5` (serial + MPI-collective HDF5 I/O), `comm` (MPI world
  wrapper), `util` (pair / block work distributors and shared
  containers), and `types` (numeric / serde marker traits).

See the crate-level docs (`cargo doc --open`) for the detailed API.

### Binaries

Three MPI front-ends ship in `src/bin`, one per construction method.
All three take a single YAML config file as their only positional
argument and are meant to be launched under `mpirun` / `srun`:

| Binary       | Source                  | Workflow                                                                                                     |
| ------------ | ----------------------- | ------------------------------------------------------------------------------------------------------------ |
| `pucgrn_cli` | `src/bin/pucgrn_cli.rs` | Parens-Net (PUC). Dispatches the stages listed in `pucn::WorkflowArgs::mode` through `pucn::execute_workflow`. |
| `gbgrn_cli`  | `src/bin/gbgrn_cli.rs`  | Gradient-boosted GRN. Dispatches on `gbn::RunMode` to either `gbn::run_cross_fold_gbm` (CV-only) or `gbn::infer_gb_network` (full pipeline). |
| `mcpgrn_cli` | `src/bin/mcpgrn_cli.rs` | MCPNet B-spline MI. Parses the `mcpn::WorkflowArgs` and (once enabled) dispatches `mcpn::execute_workflow` over `mcpn::RunMode::{MIBSplineWeights, MIBSpline}`. |

Typical invocation:

```
mpirun -np <N> ./target/release/<binary> <path/to/config.yml>
```

Example configs live under `config/`.

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
