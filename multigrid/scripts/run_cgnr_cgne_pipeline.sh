#!bin/bash

# Runs CGNR-CGNE pipeline with the following steps:
# 1. Compiles GRID with any changes made to Grid/examples/Example_cgnr_cgne.cc
# 2. Runs the executable at GridInstall/bin/Example_cgnr_cgne
# 3. Parses the XML output with lqcd/multigrid/python_scripts/parse_cgnr_cgne_xml.py
# 4. Exits script in the lqcd/multigrid/scripts directory
# Upon completion, one should be able to rerun the Jupyter notebook lqcd/multigrid/python_scripts/plot_cgnr_cgne.ipynb with updated data.

GRID=/Users/patrickoare/libraries/Grid
GRIDINSTALL=/Users/patrickoare/libraries/GridInstall
LQCD=/Users/patrickoare/lqcd

cd ${GRID}/build
make -C examples && make install -C examples

cd ${GRIDINSTALL}/bin
./Example_cgnr_cgne --log Iterative

cd ${LQCD}/multigrid/python_scripts
python3 parse_cgnr_cgne_xml.py

cd ${LQCD}/multigrid/scripts
