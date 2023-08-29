cd ../python_scripts
for ii in `seq 0 4`
do
    echo "Ensemble ${ii}, Taylor expanded model."
    python3 exc_state_fits_final.py ${ii}
done
cd ../scripts
