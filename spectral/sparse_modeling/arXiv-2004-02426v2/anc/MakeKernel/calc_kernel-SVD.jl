#=
!***********************************************************
! Sparse modeling approach to analytic continuation
!      of imaginary-time Monte Carlo data
!                           3 Apr 2020  E. Itou and Y. Nagai
!***********************************************************
=#


module MakeK
    include("../SpM/julia/input.jl")
    using .Input
    using LinearAlgebra
    export make_kernel

    function make_kernel(inputfile)

        input = Inputdata(inputfile)
        NT = getInt(input,"NT",16)
        NT_org = getInt(input,"NT_org",16) #full tau
        dN = NT_org - NT
        
        if dN != 0
            if dN % 2 != 1
                println("Error! dN should be odd")
                exit()
            end
        end
        istart = div(dN-1,2) 
        if dN != 0
            println("$(istart+1) <= tau <= $(NT_org-istart-1) is considered")
        else
            println("0 <= tau <= $(NT_org-1) is considered")
        end
#        println("first $(istart+1) taus and last $istart taus are removed to construct the kernel K.")

        Nomega =getInt(input,"Nomega",3001)
        Omega_max = getFloat64(input,"Omega_max",4.0)
        Lambda = NT_org*Omega_max

        inomega = string(Nomega, pad = 6)
        mat_K = zeros(NT,Nomega)
#        K = zeros(NT_org,Nomega)
        

        for itau=1:NT
            for iomega=1:Nomega
                dtau = (itau+istart)/NT_org
                dx = 2*dtau-1
                domega = 2/(Nomega-1)
                omegaP = (iomega-1)*domega-1

                mat_K[itau,iomega] = sqrt(domega)*cosh(Lambda*dx*omegaP*0.5)/cosh(Lambda*omegaP*0.5)
                if mat_K[itau,iomega] > 1
                    println( "K is larger than 1")
                end
            end
        end
        WFILE = "./data-K-Nt"*string(NT_org)*"-Ndomega"*inomega*".dat"
        fp = open(WFILE,"w")
        println(fp,Lambda)
        for itau=1:NT
            for iomega=1:Nomega
                println(fp,mat_K[itau,iomega])
            end
        end
        close(fp)


        u, s, v = svd(mat_K,full =true)
        println("Singular values")

        WFILE="./data-s-value"*inomega*".dat"
        fp = open(WFILE,"w")
        for it=1:NT
            println(fp,s[it])
            println(s[it])
        end
        close(fp)

        #Normalize so that u[1,j]>=0
        for i=1:NT
            if u[1,i] < 0
                u[1:NT,i] = -u[1:NT,i]
                v[1:Nomega,i] = - v[1:Nomega,i]
            end
        end

        WFILE="./data-matrix-U"*inomega*".dat"
        fp = open(WFILE,"w")

        for i=1:NT
            for itau=1:NT
                println(fp,u[i,itau])
            end
        end
        close(fp)

        WFILE="./data-matrix-V"*inomega*".dat"
        fp = open(WFILE,"w")

        for itau=1:NT
            for iomega=1:Nomega
                println(fp,v[iomega,itau])
            end
        end

        
        close(fp)
        return Omega_max,mat_K


    end
end