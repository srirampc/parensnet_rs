# Parensnet 

## Installation

Clone sources with recurse-submodules option so that all dependecies 
are cloned.

```
git clone --recurse-submodules https://github.com/srirampc/parensnet_rs.git
```

### Dependencies

When compiling `parensnet_rs`,

- `lightgbm3` (and therefore `LightGBM`) compiled from source (ext/ dir)
- `hdf5-rust` is compiled from source, and
- `rsmpi` requires an MPI implementation.

The following dependecies needs to be installed prior:

1. CMake version > 3.28
2. MPI implementation that works with `rsmpi`
3. HDF5 compiled with MPI summport
4. LLM with libclang  (for `bindgedn` support)

### Build 

Souces can be built with cargo

```
cargo b --release
```
