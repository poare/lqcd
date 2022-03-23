cd ../python_scripts
for ii in `seq 0 4`
do
    python3 current_renorm.py ${ii}
done
cd ../scripts
