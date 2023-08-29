cd ../python_scripts
for ii in `seq 0 4`
do
    echo "Ensemble ${ii}"
    python3 exc_state_fits_all.py ${ii}
done
cd ../scripts
