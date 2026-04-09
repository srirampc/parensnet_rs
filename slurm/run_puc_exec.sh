#!/bin/bash

print_usage () {
  echo "USAGE:: $0 -c CONFIG_FILE [-d release/debug]"
  exit 0
}

EXE_DIR=release
while getopts "c:p:n:d:l:h" opt; do
  case $opt in
    c) P_CFG=$OPTARG ;;
    d) EXE_DIR=$OPTARG ;;
    l) RLOG=$OPTARG ;;
    h) print_usage ;;
    *) echo "Invalid argument"; exit 1;;
  esac
done

if [ -z "$P_CFG" ]; then 
    echo "Config file not provided; Run with -c";
    print_usage;
fi


if [ -z "$EXE_DIR" ]; then
  EXE_DIR=target
fi

if [ -z "$RLOG" ]; then
  RLOG=info
fi

echo "Running with config $P_CFG , exe: $EXE_DIR"

SRC_DIR=$HOME/dev/parensnet_rs/
P_EXE="$SRC_DIR/target/$EXE_DIR/pucgrn_cli"

# Load module dependencies
if ! command -v spack >/dev/null 2>&1
then
    echo "spack could not be found; assuming default environment"
else
    spack load mvapich hdf5
fi

echo ------------------ENV-------------------
printenv | grep -E -w 'HOME|PWD|USER|PATH'
printenv | grep '^SLURM'
printenv | grep -E 'MPI'
echo ---------------------------------------

HDF5_DIR=/storage/ideas/is-schockalingam6-0/phe/spack/opt/spack/linux-cascadelake/hdf5-1.14.6-ijvczkid4msgmpvss75tjiortvvhgyl4/lib
MPI_DIR=storage/ideas/is-schockalingam6-0/phe/spack/opt/spack/linux-cascadelake/mvapich-4.1-sl5jrxvquujoy7vlwjnqiqiw72rwkyua/lib
export LD_LIBRARY_PATH=$HDF5_DIR:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MPI_DIR:$LD_LIBRARY_PATH
RUST_LOG=$RLOG
export RUST_LOG
echo "mpirun PATH     :: $(which mpirun)"
echo "LD_LIBRARY_PATH :: $LD_LIBRARY_PATH" 
echo "RUST_LOG        :: $RUST_LOG" 
ldd $P_EXE
LD_LIBRARY_PATH=$LD_LIBRARY_PATH $P_EXE --help
#mpirun --help
#mpiexec -np $NP $MPI_ARGS $P_EXE $P_CFG

spack unload mvapich hdf5
