#=
!***********************************************************
! Sparse modeling approach to analytic continuation
!      of imaginary-time Monte Carlo data
!                            3 Apr 2020  E.Itou and Y. Nagai
!               See Appendix A in arXiv:1702.03056[cond-mat]
!***********************************************************
=#

module Smac_lib
    export mat_reduced,smac_est
    function mat_reduced(L,S_data,U,V_t)
        NT = length(S_data)
        Nomega = size(V_t)[1]
        vec_S = S_data[1:L]
        U_mat = U[1:NT,1:L]
        V_t_mat = V_t[1:Nomega,1:L]
        return vec_S,U_mat,V_t_mat
    end

    struct Param
        NT::Int64
        Nomega::Int64
        L::Int64
        d_mu::Float64
        d_mup::Float64
        lambda::Float64
        U_mat::Array{Float64,2}
        vec_S::Array{Float64,1}
        V_t_mat::Array{Float64,2}
    end

    function smac_est(NT,Nomega,L,xout,vec_G,mat_K,xi,U_mat,vec_S,V_t_mat)
        d_mu = 1.0
        d_mup = 1.0
        fchiold = 0.0
        lambda_est=0.0

        l0=-15.0
        lam0 = 10.0^(l0)
        lambda = lam0
        println("in est lambda $l0 $lam0 $lambda")
        itemax = 10000
        p = Param(NT,Nomega,L,d_mu,d_mup,lambda,U_mat,vec_S,V_t_mat)
        xout = smac_calc(p,vec_G,xi,itemax)
        
        Ctmp = mat_K*xout
        chi0 = sum((vec_G .- Ctmp).^2)
        for it=1:NT
            sa = vec_G[it] - Ctmp[it]
        #    println(sa)
        end
        println("chi0 ",lambda,"\t",chi0)
        log_chi0 = log10(chi0)

        l1=2.0
        lam1 = 10^(l1)
        lambda = lam1
        p = Param(NT,Nomega,L,d_mu,d_mup,lambda,U_mat,vec_S,V_t_mat)
        xout = smac_calc(p,vec_G,xi,itemax)
        Ctmp = mat_K*xout
        chi1 = sum((vec_G .- Ctmp).^2)
        println("chi1 ",lambda,"\t",chi1)
        log_chi1 = log10(chi1)

        b = (log(chi0)-log(chi1))/(log(lam0)-log(lam1))
        a = exp(log(chi0)-b*log(lam0))
        lambda_est = 0.0
        for il=2:100-1
            ll = (il -1)*(l1-l0)/(100-1)+l0
            lambda = 10^(ll)
            p = Param(NT,Nomega,L,d_mu,d_mup,lambda,U_mat,vec_S,V_t_mat)
            xout = smac_calc(p,vec_G,xi,itemax)
            Ctmp = mat_K*xout
            chi1 = sum((vec_G .- Ctmp).^2)  
            println("chi1 \t $lambda \t $chi1")
            fchi= a*lambda^b/chi1
            println("fchi \t $lambda \t $fchi")
            if il > 3
                if fchi > fchiold
                    lambda_est = lambda
                    fchiold = fchi
                    println("update lambda_est")
                end
            else
                fchiold = fchi
            end
        end
        lambda = lambda_est
        itemax = 1000000000
        println("Calc. final smanc, lambda $lambda")
        p = Param(NT,Nomega,L,d_mu,d_mup,lambda,U_mat,vec_S,V_t_mat)
        xout = smac_calc(p,vec_G,xi,itemax)
        return xout

    end

    function smac_calc(p,y,xi,itemax)
        yp = zeros(Float64,p.L)
        styp = zeros(Float64,p.L)
        stxp = zeros(Float64,p.L)
        xp = zeros(Float64,p.L)
        zp = zeros(Float64,p.L)
        up = zeros(Float64,p.L)
        Ctmp1 = zeros(Float64,p.Nomega)

        z =zeros(Float64,p.Nomega)
        u = zeros(Float64,p.Nomega)
        #=
        for il=1:p.L
            for it=1:p.NT
                yp[il] += p.U_mat[it,il]*y[it]
            end
        end
        =#
        yp = p.U_mat'*y
        #=
        for il=1:p.L
            styp[il] = p.vec_S[il]*yp[il]
        end
        =#
        @. styp = p.vec_S*yp
        sums = 0.0

        for ite =1:itemax
            styp,xp,zp,up,z,u,xi = smac_update!(p,styp,xp,zp,up,z,u,xi)
            #=
            for il=1:p.L
                stxp[il] = p.vec_S[il]*xp[il]
            end
            =#
            @. stxp = p.vec_S*xp
            for io=1:p.Nomega
                Ctmp1[io] = 0.0
                for il=1:p.L
                    Ctmp1[io] += p.V_t_mat[io,il]*xp[il]
                end
            end

                
            Ctmp1 = p.V_t_mat*xp
            sums = sum(abs.(z-Ctmp1))
            #println(ite,"\t",sums)
            

            if sums < 1e-8
                #println(sums)
                break
            end
        end

        xout = p.V_t_mat*xp
        #println(sums)

        return xout

    end

    function smac_update!(p,styp,xp,zp,up,z,u,xi)
        #vec_temp = zeros(Float64,L)
        xi1 = zeros(Float64,p.L)
        xi2 = zeros(Float64,p.L)

        vec_temp2=zeros(Float64,p.Nomega)
        xi1_1 = zeros(Float64,p.L)
        xi2_1 = zeros(Float64,p.L)
        vec_temp = zeros(Float64,p.L)

        @. vec_temp2 = z - u
        #=
        for il=1:p.L
            vec_temp3 = 0.0
            for io=1:p.Nomega
                vec_temp3 += p.V_t_mat[io,il]*vec_temp2[io]
            end
            vec_temp[il] = vec_temp3
        end
        =#
        vec_temp3 = p.V_t_mat'*vec_temp2
        vec_temp = vec_temp3[:]

        @. xi1_1 = styp/p.lambda + p.d_mup*(zp - up) + p.d_mu*vec_temp
        @. xi2_1 = xi 

        mat_inv = zeros(Float64,p.L,p.L)
        for i=1:p.L
            mat_inv[i,i] = 1/(p.vec_S[i]^2/p.lambda+ (p.d_mu + p.d_mup))
        end
        xi1 = mat_inv*xi1_1
        xi2 = mat_inv*xi2_1

        #prepare <>V xi1> and <V xi2> in Eq.A6
        V_xi1 = p.V_t_mat*xi1
        V_xi2 = p.V_t_mat*xi2
        sum_V_xi1 = sum(V_xi1)
        sum_V_xi2 = sum(V_xi2)
        d_nu = (1.0-sum_V_xi1)/sum_V_xi2

        #This is x' in  Eq.A5a
        @. xp = xi1 + d_nu*xi2
        vec_xp_up = zeros(Float64,p.L)
        @. vec_xp_up = xp + up
        d_mup_inv = 1.0/p.d_mup
        zp = calc_salpha(p,d_mup_inv,vec_xp_up)
        @. up = up + xp - zp

        #println(up)
        

        #prepare (Vx' + u) in Eq.A5d
        V_xp_u = zeros(Float64,p.Nomega)
        for io=1:p.Nomega
            V_xp_u_tmp = 0.0
            for il2=1:p.L
                V_xp_u_tmp += p.V_t_mat[io,il2]*xp[il2]
            end
            V_xp_u[io] = V_xp_u_tmp +  u[io] 
        end

        #This is z in Eq.A5d
        z = calc_P(p,V_xp_u)
        #prepare Vx' in Eq.A5e
        V_xp = p.V_t_mat*xp

        #println(V_xp)

        @. u = u + V_xp - z



        
        return styp,xp,zp,up,z,u,xi




    end

    function calc_P(p,x)
        vec_Px = zeros(Float64,p.Nomega)
        for io=1:p.Nomega
            vec_Px[io] = max(x[io],0.0)
        end
        return vec_Px
    end

    function calc_salpha(p,alpha,x)
        salpha = zeros(Float64,p.L)
        for il=1:p.L
            if x[il] > alpha
                salpha[il] = x[il] - alpha
            elseif x[il] <= -alpha
                salpha[il] = x[il] + alpha
            else
                salpha[il] = 0.0
            end
        end
        return salpha
    end
end