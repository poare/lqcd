include("calc_kernel-SVD.jl")
using .MakeK

if length(ARGS) == 0
    println("Error! Please use input parameter file. like: ")
    println("julia makeK.jl kinput.in")
    exit()
else
    Omega_max,mat_K = make_kernel(ARGS[1])
end

