#!bin/bash

TIME=$(date +%F_%T)

GRID=/Users/patrickoare/libraries/Grid
GRIDINSTALL=/Users/patrickoare/libraries/GridInstall
LQCD=/Users/patrickoare/lqcd

cd ${GRID}/build
make -C examples && make install -C examples

cd ${GRIDINSTALL}/bin

Nm=100
Nk=50
maxIter=4
Nstop=8

root="ckpoint_EODWF_lat.125"
# runname="Nm${Nm}_Nk${Nk}_${TIME}"
runname="unprec_Nm${Nm}_Nk${Nk}_${TIME}"

homedir=/Users/patrickoare
inFile="${homedir}/libraries/PETSc-Grid/${root}"

logDir="${homedir}/lqcd/multigrid/logs/${root}"
outDir="${homedir}/lqcd/multigrid/spectra/${root}/${runname}"

mkdir -p ${logDir}
mkdir -p ${outDir}

logs="${logDir}/${runname}.log"
rf="EvalReSmall"
# rf="EvalReLarge"

# ./Example_spec_kryschur ${Nm} ${Nk} ${maxIter} ${Nstop} ${inFile} ${outDir} ${rf} > ${logs}
./Example_spec_kryschur ${Nm} ${Nk} ${maxIter} ${Nstop} ${inFile} ${outDir} ${rf}

cd ${LQCD}/multigrid/scripts
