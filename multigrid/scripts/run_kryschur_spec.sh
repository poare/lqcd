#!bin/bash

GRID=/Users/patrickoare/libraries/Grid
GRIDINSTALL=/Users/patrickoare/libraries/GridInstall
LQCD=/Users/patrickoare/lqcd

cd ${GRID}/build
make -C examples && make install -C examples

cd ${GRIDINSTALL}/bin

Nm=20
Nk=12
maxIter=10000
Nstop=10

root="ckpoint_EODWF_lat.125"
dirname="${root}_Nm${Nm}_Nk${Nk}"

rf="EvalReSmall"

inFile="/Users/patrickoare/libraries/PETSc-Grid/${root}"
logs="/Users/patrickoare/lqcd/multigrid/logs/${dirname}.out"
outDir="/Users/patrickoare/lqcd/multigrid/spectra/${dirname}"

mkdir -p ${outDir}

./Example_spec_kryschur ${Nm} ${Nk} ${maxIter} ${Nstop} ${inFile} ${outDir} ${rf} > ${logs}

cd ${LQCD}/multigrid/scripts
