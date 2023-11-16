# Parameters
EXE=python3
Nc=2
L=8
T=8
# L=16
# T=16
kappa=0.125
beta=2.0
ntraj=500

bstr=${beta//[.]/p}
kstr=${kappa//[.]/p}
parent_dir="/mnt/c/Users/Patrick/Dropbox (MIT)/research/2d_adjoint_qcd/meas/tests"
out_dir="${parent_dir}/wilson_Nc${Nc}_${L}_${T}_b${bstr}_k${kstr}"
out_dir_cold="${out_dir}_cold"
out_dir_hot="${out_dir}_hot"

echo "Output directory: ${out_dir}"
mkdir -p "${out_dir_cold}"
mkdir -p "${out_dir_hot}"

cd ../python_scripts
# $EXE rhmc.py -N "${Nc}" -L "${L}" -T "${T}" -k "${kappa}" -b "${beta}" -o "${out_dir_cold}" -M "${ntraj}"
# $EXE rhmc.py -N "${Nc}" -L "${L}" -T "${T}" -k "${kappa}" -b "${beta}" -o "${out_dir_hot}" -M "${ntraj}" --hot

# for i in `seq 1 3`
for i in `seq 2 3`
do
    $EXE rhmc.py -N "${Nc}" -L "${L}" -T "${T}" -k "${kappa}" -b "${beta}" -o "${out_dir_hot}" -M "${ntraj}" -s "${i}" --hot
done
