# Parameters
EXE=python3
Nc=2
L=4
T=4
kappa=0.125
beta=2.0
ntraj=10

bstr=${beta//[.]/p}
kstr=${kappa//[.]/p}
parent_dir="/Users/theoares/Dropbox (MIT)/research/2d_adjoint_qcd/meas/tests"
out_dir="${parent_dir}/wilson_Nc${Nc}_${L}_${T}_b${bstr}_k${kstr}"
out_dir="${out_dir}_cold"               # for testing

echo "Output directory: ${out_dir}"
mkdir -p "${out_dir}"

cd ../python_scripts
$EXE rhmc.py -N "${Nc}" -L "${L}" -T "${T}" -k "${kappa}" -b "${beta}" -o "${out_dir}" -M "${ntraj}" --cold
