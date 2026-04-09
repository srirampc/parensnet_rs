#!/bin/bash

print_usage () {
  echo "USAGE:: $0 -c CONFIG_FILE -p NUM_PROC [-d release/debug -n PROC_PER_NODE]"
  exit 0
}

EXE_DIR=release
while getopts "c:p:n:d:l:h" opt; do
  case $opt in
    c) P_CFG=$OPTARG ;;
    d) EXE_DIR=$OPTARG ;;
    l) RLOG=$OPTARG ;;
    p) NP=$OPTARG  ;;
    n) NPR=$OPTARG  ;;
    h) print_usage ;;
    *) echo "Invalid argument"; exit 1;;
  esac
done

if [ -z "$P_CFG" ]; then 
    echo "Config file not provided; Run with -c";
    print_usage;
fi

if [ -z "$NP" ]; then 
    echo "No. process not provided; Run with -p";
    print_usage;
fi


if [ -z "$EXE_DIR" ]; then
  EXE_DIR=target
fi

if [ -z "$RLOG" ]; then
  RLOG=info
fi

echo "Running with config $P_CFG and no. processors: $NP, exe: $EXE_DIR"

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

if [ -z "$NPR" ]; then 
MPI_ARGS="--bind-to core"
else
MPI_ARGS="-ppn $NPR"
#MPI_ARGS="--npernode $NPR"
#MPI_ARGS="--map-by node"
#MPI_ARGS=""
fi
HDF5_DIR=$HOME/data/spack/opt/spack/linux-cascadelake/hdf5-1.14.6-ijvczkid4msgmpvss75tjiortvvhgyl4/lib
export LD_LIBRARY_PATH=$MPI_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HDF5_DIR:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
RUST_LOG=$RLOG
export RUST_LOG
# CMD="mpirun -np $NP $MPI_ARGS $P_EXE $P_CFG"
echo "mpirun PATH     :: $(which mpirun)"
echo "LD_LIBRARY_PATH :: $LD_LIBRARY_PATH" 
echo "RUST_LOG        :: $RUST_LOG" 
echo mpiexec -np "$NP" "$MPI_ARGS" "$P_EXE" "$P_CFG"
ldd $P_EXE
$P_EXE --help
unset LUA_PATH
unset LUA_CPATH
#mpirun --help
LD_LIBRARY_PATH=$LD_LIBRARY_PATH mpiexec -np $NP $MPI_ARGS $P_EXE $P_CFG

spack unload mvapich hdf5
