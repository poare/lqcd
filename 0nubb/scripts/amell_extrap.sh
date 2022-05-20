cd ../python_scripts
for ii in `seq 0 1`
do
    python3 amell_extrap.py ${ii}
done
cd ../scripts
