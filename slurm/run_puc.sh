#!/bin/bash

EXE_DIR=release
while getopts "c:p:n:d" opt; do
  case $opt in
    c) P_CFG=$OPTARG ;;
    d) EXE_DIR=release ;;
    p) NP=$OPTARG  ;;
    n) NPR=$OPTARG  ;;
    *) echo "Invalid argument"; exit 1;;
  esac
done

if [ -z "$P_CFG" ]; then 
    echo "Config file not provided; Run with -c";
    exit 0;
fi

if [ -z "$NP" ]; then 
    echo "No. process not provided; Run with -p";
    exit 0;
fi

echo "Running with config $P_CFG and no. processors: $NP"

SRC_DIR=$HOME/dev/parensnet_rs/
P_EXE="$SRC_DIR/target/$EXE_DIR/pucgrn_cli"

# Load module dependencies
spack load openmpi hdf5

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
RUST_LOG=info
export RUST_LOG
CMD="mpirun -np $NP $MPI_ARGS $P_EXE $P_CFG"
echo "LD_LIBRARY_PATH :: $LD_LIBRARY_PATH" 
echo "RUST_LOG        :: $RUST_LOG" 
echo "$CMD" 
$CMD

spack unload openmpi hdf5
