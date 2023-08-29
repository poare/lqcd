cd ../python_scripts
for ii in `seq 0 4`
do
    # for jj in `seq 0 3`
    # do
    #     echo "Ensemble ${ii}, fit form ${jj}"
    #     python3 exc_state_fits_single_model.py ${ii} ${jj}
    # done
    echo "Ensemble ${ii}, fit form f6"
    python3 exc_state_fits_single_model.py ${ii} 3
done
cd ../scripts
