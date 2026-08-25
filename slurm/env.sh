#!/bin/sh
spack load openmpi hdf5 llvm cmake@3.31.9
LD_LIBRARY_PATH=$(spack location -i hdf5)/lib:$(mpicc -showme:libdirs):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
