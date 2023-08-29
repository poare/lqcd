//***********************************************************
// Sparse modeling approach to analytic continuation
//      of imaginary-time Monte Carlo data
//                           3 Apr 2020  E. Itou and Y. Nagai
//               See Appendix A in arXiv:1702.03056[cond-mat]
//***********************************************************

#include "smac_lib.h"

#define PRINT_MAT(X) std::cout << #X << ":\n" << X << std::endl << std::endl

using namespace Eigen;



void calc_salpha(std::vector<double>& salpha,const int L, const double alpha,const std::vector<double>& x){
    //salpha.fill(0.0);
    for (int il = 0; il < L; il++){
        if(x[il] > alpha){
            salpha[il] = x[il] -alpha;
        }
        else if(x[il] <= -alpha){
            salpha[il] = x[il] +alpha;
        }else{
            salpha[il] = 0.0;
        }
    };
}



void calc_P(const int Nomega,std::vector<double>& vec_Px,const std::vector<double>& x){
    for (int io = 0; io < Nomega; io++){
        vec_Px[io] = std::max(x[io],0.0);
    }
}

Smac::Smac(const int NT0,const int Nomega0, const int L0, const Eigen::MatrixXd& mat_K0,const Eigen::MatrixXd& U0,
        const Eigen::MatrixXd& V0, const Eigen::MatrixXd& V_t0,const Eigen::VectorXd& vec_S0
    ){
        NT = NT0;
        Nomega = Nomega0;
        L = L0;
        mat_K = mat_K0;
        U = U0;
        V = V0;
        V_t = V_t0;
        vec_S = vec_S0;
    };


// This is a wrapper for the full calculation. SMAC = Sparse Modeling Analytic Continuation, pass in vectorized
// G and xi = V \vec{e}.
std::vector<double> Smac::smac_est(const std::vector<double>& vec_G, const std::vector<double>& xi){
    std::vector<double> xout(Nomega);
    double d_mu = 1.0;
    double d_mup = 1.0;
    double fchiold = 0.0;
    double fchi = 0.0;
    double lambda_est = 0.0;

    double l0 = -15.0;
    double lam0 = std::pow(10.0, l0);
    double lambda = lam0;
    std::cout << "in est lambda " << l0 << " " << lam0 << " " << lambda << std::endl;
    int itemax = 10000;
    xout = this->smac_calc(vec_G,xi,itemax,lambda,d_mu,d_mup);
    double chi0 = 0.0;

    std::vector<double> Ctmp(NT,0.0);
    for (int it = 0; it < NT; it++){
        Ctmp[it] = 0.0;
        for (int io = 0; io < Nomega; io++){
            //std::cout << mat_K(it,io) << " " << xout[io]<<std::endl;
            Ctmp[it] = Ctmp[it] + mat_K(it,io)*xout[io];
        }
        double sa = vec_G[it]-Ctmp[it];
        //std::cout << vec_G[it] << " " << Ctmp[it] <<std::endl;

        chi0 = chi0 + sa*sa;
    };


    std::cout << "chi0 " << lambda << " " << chi0 << std::endl;
    double log_chi0 = std::log10(chi0);

    double l1 = 2.0;
    double lam1 = std::pow(10.0, l1);
    lambda = lam1;
    xout = this->smac_calc(vec_G,xi,itemax,lambda,d_mu,d_mup);

    double chi1 = 0.0;

    for (int it = 0; it < NT; it++){
        Ctmp[it] = 0.0;
        for (int io = 0; io < Nomega; io++){
            Ctmp[it] = Ctmp[it] + mat_K(it,io)*xout[io];
        }
    };

    // TODO from here on down, what does this do?
    for (int it = 0; it < NT; it++){
        double sa = vec_G[it]-Ctmp[it];
        chi1 = chi1 + sa*sa;
    };
    std::cout << "chi1 " << lambda << " " << chi1 << std::endl;
    double log_chi1 = std::log10(chi1);

    double b = (log(chi0)-log(chi1))/(log(lam0)-log(lam1));
    double a = exp(log(chi0)-b*log(lam0));

    for (int il = 2; il < 100; il++){
        double ll = (il-1)*(l1-l0)/(100-1)+l0;
        lambda = std::pow(10.0, ll);
        xout = this->smac_calc(vec_G,xi,itemax,lambda,d_mu,d_mup);
        chi1 = 0.0;
        for (int it = 0; it < NT; it++){
            Ctmp[it] = 0.0;
            for (int io = 0; io < Nomega; io++){
                Ctmp[it] = Ctmp[it] + mat_K(it,io)*xout[io];
            }
        };

        for (int it = 0; it < NT; it++){
            double sa = vec_G[it]-Ctmp[it];
            chi1 = chi1 + sa*sa;
        };
        std::cout << "chi1 " << lambda << " " << chi1 << std::endl;
        fchi = a*std::pow(lambda,b)/chi1;
        std::cout << "fchi " << lambda << " " << fchi << std::endl;

        if(il > 3){
            if(fchi > fchiold){
                lambda_est = lambda;
                fchiold = fchi;
                std::cout << "update lambda_est "  << std::endl;
            }
        }else{
            fchiold = fchi;
        }
    }
    lambda = lambda_est;
    itemax = 10000;
    // itemax = 1000000;
    // itemax = 1000000000;
    std::cout << "Calc. filal smac, lambda " << lambda << std::endl;
    xout = this->smac_calc(vec_G,xi,itemax,lambda,d_mu,d_mup);

    return xout;
}


std::vector<double> Smac::smac_calc(const std::vector<double>& y,const std::vector<double>& xi,const int itemax,
const double lambda,const double d_mu,const double d_mup){
    std::vector<double> xout(Nomega);
    //VectorXd yp(L);
    //yp.fill(0.0);
    std::vector<double> styp(L,0.0);
    std::vector<double> stxp(L,0.0);
    std::vector<double> xp(L,0.0);

    std::vector<double> zp(L,0.0);

    std::vector<double> up(L,0.0);


    std::vector<double> z(Nomega,0.0);

    std::vector<double> u(Nomega,0.0);

    std::vector<double> yp(L,0.0);


    std::vector<double> Ctmp1(Nomega,0.0);

    for (int il = 0; il < L; il++){    // rotate Green's function y into the IR
        yp[il] = 0.0;
        for (int it = 0; it < NT; it++){
            yp[il] = yp[il] + U(it,il)*y[it];
        }
    }

    for (int il = 0; il < L; il++){
        styp[il] = vec_S(il)*yp[il];
    };
    double sum = 0.0;


    for (int ite = 1; ite < itemax+1; ite++){
        this->smac_update(styp,xp,zp,up,z,u,xi,lambda,d_mu,d_mup);
        for (int il = 0; il < L; il++){
            stxp[il] = vec_S(il)*xp[il];
        };
        sum = 0.0;
        //convert rho' -> rho org.

        for (int io = 0; io < Nomega; io++){
            Ctmp1[io] = 0.0;
            for (int il = 0; il < L; il++){
                Ctmp1[io] = Ctmp1[io] + V_t(io,il)*xp[il];
            }
        }

        // "Convergence" here means that z = x, i.e. z = V^T x', since this is the Lagrange
        // multiplier constraint.
        for (int io = 0; io < Nomega; io++){
            sum = sum + abs(z[io] - Ctmp1[io]);
        };
        //std::cout << ite << "\t" << sum << std::endl;


        //if(ite %10 == 0){
        //    std::cout << ite << "\t" << sum << std::endl;
        //}

        if(sum < 1e-8){

            break;
        }

    };

    for (int io = 0; io < Nomega; io++){
        xout[io] = 0.0;
        for (int il = 0; il < L; il++){
            xout[io] = xout[io] + V_t(io,il)*xp[il];
        }
    }

    //for (int io = 0; io < Nomega; io++){
    //    double xtmp = 0.0;
    //    for (int il = 0; il < L; il++){
    //        xtmp = xtmp + V_t(io,il)*xp[il];
    //    }
    //    xout[io] = xtmp;
        //std::cout << xtmp << std::endl;
    //}
    //std::cout << sum << std::endl;

    return xout;
}



void Smac::smac_update(const std::vector<double> & styp,
    std::vector<double> & xp,std::vector<double> & zp,std::vector<double> & up,
    std::vector<double> & z,std::vector<double> & u,const std::vector<double> & xi,
    const double lambda,const double d_mu,const double d_mup){

    std::vector<double> vec_temp(L, 0.0);
    std::vector<double> xi1(L, 0.0);
    std::vector<double> xi2(L, 0.0);

    std::vector<double> vec_temp2(Nomega, 0.0);
    for (int io = 0; io < Nomega; io++){
        vec_temp2[io] = z[io] - u[io];
    }

    for (int il = 0; il < L; il++){
        double vec_temp3 = 0.0;
        for (int io = 0; io < Nomega; io++){
            vec_temp3 = vec_temp3 + V(il,io)*vec_temp2[io]; // V (\vec z - \vec u)
        }
        vec_temp[il]=vec_temp3;
    }

    std::vector<double> xi1_1(L);
    std::vector<double> xi2_1(L);
    for (int il = 0; il < L; il++){
        xi1_1[il] = styp[il]/lambda +
            d_mup*(zp[il]-up[il]) + d_mu*vec_temp[il];  // mu' (z - u) + mu V^T (z - u)
        xi2_1[il] = xi[il];
    }

    MatrixXd mat_inv(L,L);
    mat_inv.fill(0.0);
    for (int il = 0; il < L; il++){
        mat_inv(il,il) = 1.0/(vec_S(il)*vec_S(il)/lambda + (d_mu + d_mup));
    };

    // First parameter in update: xi1, xi2

    for (int il1 = 0; il1 < L; il1++){
        xi1[il1] = 0.0;
        xi2[il1] = 0.0;
        for (int il2 = 0; il2 < L; il2++){
            xi1[il1] = xi1[il1] + mat_inv(il1,il2)*xi1_1[il2];
            xi2[il1] = xi2[il1] + mat_inv(il1,il2)*xi2_1[il2];
        }
    }

    double sum_V_xi1 = 0.0;
    double sum_V_xi2 = 0.0;

    std::vector<double>  V_xi1(Nomega);
    std::vector<double>  V_xi2(Nomega);


    for (int io = 0; io < Nomega; io++){
        V_xi1[io] = 0.0;
        V_xi2[io] = 0.0;
        for (int il2 = 0; il2 < L; il2++){
            V_xi1[io] = V_xi1[io] + V_t(io,il2)*xi1[il2];
            V_xi2[io] = V_xi2[io] + V_t(io,il2)*xi2[il2];
        }
    }

    // Define nu. Nu doesn't need to be stored once xp is updated.
    for (int io = 0; io < Nomega; io++){
        sum_V_xi1 = sum_V_xi1 + V_xi1[io];
        sum_V_xi2 = sum_V_xi2 + V_xi2[io];
    }
    double d_nu = (1.0 - sum_V_xi1)/sum_V_xi2;

    // Update x'
    for (int il = 0; il < L; il++){
        xp[il]= xi1[il] + d_nu*xi2[il];
    };

    std::vector<double> vec_xp_up(L);

    for (int il = 0; il < L; il++){
        vec_xp_up[il]= xp[il] + up[il];
    };

    double d_mup_inv = 1.0/d_mup;

    // Update z'
    //salpha. Passes zp by reference, so updates its value
    calc_salpha(zp,L,d_mup_inv,vec_xp_up);

    // Update u'
    for (int il = 0; il < L; il++){
        up[il]= up[il]+xp[il]-zp[il];
    };

    std::vector<double>  V_xp_u(Nomega);


    for (int io = 0; io < Nomega; io++){
        double V_xp_u_tmp = 0.0;
        for (int il2 = 0; il2 < L; il2++){
            V_xp_u_tmp = V_xp_u_tmp + V_t(io,il2)*xp[il2];
        }
        V_xp_u[io] = V_xp_u_tmp + u[io];
    }

    // Update z
    calc_P(Nomega,z,V_xp_u);

    std::vector<double>  V_xp(Nomega);

    for (int io = 0; io < Nomega; io++){
        V_xp[io] = 0.0;
        for (int il2 = 0; il2 < L; il2++){
            V_xp[io] = V_xp[io] + V_t(io,il2)*xp[il2];
        }
    }

    // Update u
    for (int io = 0; io < Nomega; io++){
        u[io] = u[io] + V_xp[io] - z[io];
    }


}
