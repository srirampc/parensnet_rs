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
    spack load openmpi hdf5
fi

echo ------------------ENV-------------------
printenv | grep -E -w 'HOME|PWD|USER|PATH'
printenv | grep '^SLURM'
printenv | grep -E 'MPI'
echo ---------------------------------------

if [ -z "$NPR" ]; then 
MPI_ARGS="--bind-to CORE"
else
MPI_ARGS="--map-by ppr:$NPR:node"
fi
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(mpicc -showme:libdirs)"  
export LD_LIBRARY_PATH
RUST_LOG=$RLOG
export RUST_LOG
CMD="mpirun -np $NP $MPI_ARGS $P_EXE $P_CFG"
echo "LD_LIBRARY_PATH :: $LD_LIBRARY_PATH" 
echo "RUST_LOG        :: $RUST_LOG" 
echo "$CMD" 
$CMD

spack unload openmpi hdf5
