rm -f ./filelist
touch ./filelist

for cfg in `seq 1000 10 5000`
do
    echo "${cfg}" >> ./filelist
done
