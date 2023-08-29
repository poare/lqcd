#=
!***********************************************************
! Sparse modeling approach to analytic continuation
!      of imaginary-time Monte Carlo data
!                           3 Apr 2020  E. Itou and Y. Nagai
!***********************************************************
=#


include("./input.jl")
using .Input
include("./smac_lib.jl")
using .Smac_lib
include("../../MakeKernel/calc_kernel-SVD.jl")
using .MakeK

function loadK(NT,Nomega,filename)
    data = readlines(filename)
    Omega_max = parse(Float64,data[1])
    mat_K = zeros(Float64,NT,Nomega)
    count = 1
    for it=1:NT
        for io=1:Nomega
            count += 1
            mat_K[it,io] = parse(Float64,data[count])
        end
    end
    return Omega_max,mat_K
end

function loadG(NT,filename)
    data = readlines(filename)
    vec_G = zeros(Float64,NT)
    count = 0
    for it=1:NT
        count += 1
        u = split(data[count])
        vec_G[it] = parse(Float64,u[3])
    end
    return vec_G
end

function main()
    if length(ARGS) == 0
        println("Error! Please use input parameter file. like: ")
        println("julia main_v2.jl input.in")
        exit()
    else
        input = Inputdata(ARGS[1])
    end

    NT = getInt(input,"NT",16)
    NT_org = getInt(input,"NT_org",16)

    dN = NT_org - NT
    istart = div(dN-1,2) 
    println("$(istart+1) <= tau <= $(NT_org-istart-1) is considered")



    Nomega =getInt(input,"Nomega",3001)
    domega = 2/(Nomega-1)
    inomega = string(Nomega, pad = 6)
    RFILE = "./data-K-Nt"*string(NT_org)*"-Ndomega"*inomega*".dat"
    println("read file ",RFILE)
    if isfile(RFILE) == false
        println("Making the kernel K...")
        if length(ARGS) < 2
            println("Error! Please use input parameter file for making the kernel. like: ")
            println("julia main_v2.jl input.dat kinput.in")
            exit()
        else
            Omega_max,mat_K = make_kernel(ARGS[2])
        end
    else
        println("The kernel K is loaded")
        Omega_max,mat_K = loadK(NT,Nomega,RFILE)
    end
    RFILE = getString(input,"Ctau","test.dat")
    println(RFILE)
    vec_G = loadG(NT,RFILE)
#    println(vec_G)
    #cucl d_norm for sum rule
    d_norm = vec_G[1]/sqrt(domega)
    println("d_norm ", d_norm)
    vec_G .= vec_G/d_norm

    # read SVD of K' 
    RFILE="./data-s-value"*inomega*".dat"
    data = readlines(RFILE)
    data_S = zeros(Float64,NT)
    for it=1:NT
        data_S[it] = parse(Float64,data[it])
    end
#    println(data_S)
    RFILE="./data-matrix-U"*inomega*".dat"
    data = readlines(RFILE)
    U = zeros(Float64,NT,NT)
    count = 0
    for it=1:NT
        for io=1:NT
            count += 1
            U[it,io] = parse(Float64,data[count])
        end
    end
    RFILE="./data-matrix-V"*inomega*".dat"
    data = readlines(RFILE)
    V = zeros(Float64,NT,Nomega)
    count = 0
    for it=1:NT
        for io=1:Nomega
            count += 1
            V[it,io] = parse(Float64,data[count])
        end
    end
    V_t = V'
    println("read file done")
    ii = 0
    for it=1:NT
        ii += 1
        if data_S[it] <= 1e-10
            break
        end
    end
    L = ii
    println("L = $L")

    vec_S,U_mat,V_t_mat = mat_reduced(L,data_S,U,V_t)
    println("mat_reduced done")

    e = ones(Float64,Nomega)
    xi = zeros(Float64,L)
    for il=1:L
        for io=1:Nomega
            xi[il] += V_t_mat[io,il]*e[io]
        end
    end
#    xi = V_t_mat'*e
    for il=1:L
        println(" xi in main \t",il,"\t",xi[il])
    end

    vec_Gout = zeros(Float64,NT)
    xout = zeros(Float64,Nomega)
    println("before smac")
    xout = smac_est(NT,Nomega,L,xout,vec_G,mat_K,xi,U_mat,vec_S,V_t_mat)
    println("end smax")

    WFILE="./data-rho-Nomega"*inomega*".dat"
    fp = open(WFILE,"w")
    for io=1:Nomega
        println(fp,((io-1)*domega-1)*Omega_max/NT_org,"\t",xout[io]*d_norm/sqrt(domega)/NT_org^4/Omega_max)
    end
    close(fp)

    WFILE="./data-C_tau_result-Nomega"*inomega*".dat"
    fp = open(WFILE,"w")
    for it=1:NT
        for io=1:Nomega
            vec_Gout[it] += mat_K[it,io]*xout[io]*d_norm
        end
        println(fp,"it,vec_G \t",it,"\t",vec_Gout[it])
    end
    close(fp)

end

main()