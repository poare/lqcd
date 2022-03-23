cd ../python_scripts
for ii in `seq 0 4`
do
    python3 short_distance_analysis.py ${ii}
done
cd ../scripts
