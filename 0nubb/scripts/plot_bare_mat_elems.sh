cd ../python_scripts
for ii in `seq 0 4`
do
    python3 plot_bare_mat_elems.py ${ii}
done
cd ../scripts
