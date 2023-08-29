C***********************************************************
C         < smac_lib.f >
C                        (c) Etsuko Itou 14 Dec 2018
C      See Appendix A in arXiv:1702.03056[cond-mat]
C***********************************************************
c
      subroutine mat_reduced(S_data,U,V_t)
c
      include 'para.h'
C------------Local variables--------------------------------
      real*8    S_data(NT)
      real*8    U(NT,NT)
      real*8    V_t(Nomega,NT)
C------------Main-------------------------------------------
c
        do i = 1, L
        vec_S(i)= S_data(i)
        end do
c
        do it = 1,NT
        do il = 1,L
        U_mat(it,il)= U(it,il)
        end do
        end do
c
        do io = 1, Nomega
        do il = 1, L
        V_t_mat(io,il)=V_t(io,il)
        end do
        end do
C-----------------------------------------------------------
      Return
      END
C***********************************************************
c
      subroutine smac_est(xout,vec_G,mat_K,xi)
c
      include 'para.h'
C------------Local variables--------------------------------
      real*8    vec_G(NT)
      real*8    mat_K(NT,Nomega)
      real*8    xi(L)
      real*8    xout(Nomega)
      real*8    Ctmp(NT)
c
      real*8    lambda_est, fchiold, fchi, a, b
      real*8    d_mu, d_mup, lambda
      real*8    l0, lam0, l1, lam1, ll
      real*8    chi0, chi1, log_chi0, log_chi1
C------------Main-------------------------------------------
C      set (mu,mu')
       d_mu =1.d0
       d_mup=1.d0
c
       fchiold= 0.d0
       lambda_est=0.d0
c
c      set lambda_{min}
       l0= -15.d0
       lam0=10.d0**(l0)
       lambda = lam0
       write(*,*) 'in est lambda',l0,lam0,lambda
       itemax = 10000
c   
c------------
        CALL smac_calc(xout,vec_G,xi,itemax)
c
        chi0 = 0.d0
        do it = 1, NT
        Ctmp(it)=0.d0
        do io = 1, Nomega
        Ctmp(it)=Ctmp(it) + mat_K(it,io)*xout(io)
        end do
        chi0 = chi0 + (vec_G(it)-Ctmp(it))**2.d0
        end do
        write(*,*) 'chi0',lambda,chi0
        log_chi0=dlog10(chi0)
c
c-------
c       set lambda_{max}
        l1 = 2.d0
        lam1 = 10.d0**(l1)
        lambda=lam1     
c
        CALL smac_calc(xout,vec_G,xi,itemax)
c
        chi1 = 0.d0
        do it = 1, NT
        Ctmp(it)=0.d0
        do io = 1, Nomega
        Ctmp(it)=Ctmp(it) + mat_K(it,io)*xout(io)
        end do
        chi1 = chi1 + (vec_G(it)-Ctmp(it))**2.d0
        end do
        write(*,*) 'chi1',lambda,chi1
        log_chi1=dlog10(chi1)
c
        b = (dlog(chi0)-dlog(chi1))/(dlog(lam0)-dlog(lam1))
        a = dexp(dlog(chi0)-b*dlog(lam0))
c-------
c       "100" below = N_{lambda} 
        do il = 2, 100-1
        ll = (il -1)*(l1-l0)/(100-1)+l0
        lambda=10.d0**(ll)
c
        CALL smac_calc(xout,vec_G,xi,itemax)
c
        chi1 = 0.d0
        do it = 1, NT
        Ctmp(it)=0.d0
        do io = 1, Nomega
        Ctmp(it)=Ctmp(it) + mat_K(it,io)*xout(io)
        end do
        chi1 = chi1 + (vec_G(it)-Ctmp(it))**2.d0
        end do
        write(*,*) 'chi1',lambda,chi1
        fchi= a*lambda**b/chi1
        write(*,*) 'fchi',lambda,fchi
c
         if(il.gt.3) then
            if (fchi.gt.fchiold) then
               lambda_est = lambda
               fchiold = fchi
               write(*,*) 'update lambda_est'
            end if
            else
               fchiold = fchi
            end if
c
         end do !--- il loop
c
          lambda = lambda_est
c         set # of ADMM interations
          itemax = 1000
c          itemax = 1000000000
c---- 
          write(*,*) 'Calc. final smac,lambda',lambda
        CALL smac_calc(xout,vec_G,xi,itemax)        
c
C-----------------------------------------------------------
      Return
      END
C***********************************************************
      subroutine smac_calc(xout,y,xi,itemax)

      include 'para.h'
C------------Local variables--------------------------------
      real*8    xout(Nomega), z(Nomega), u(Nomega), Ctmp1(Nomega)
      real*8    y(NT)
      real*8    xi(L),yp(L),styp(L),stxp(L),xp(L),zp(L),up(L)
      real*8    xtmp, sum
C------------Main-------------------------------------------

      do il = 1, L
      yp(il) = 0.d0
      styp(il) = 0.d0
      stxp(il) = 0.d0
      xp(il) = 0.d0
      zp(il) = 0.d0
      up(il) = 0.d0
      end do
c
      do io = 1, Nomega
      z(io) = 0.d0
      u(io) = 0.d0
      end do
c----   prepare U^t * vec_G(tau)
      do il = 1, L
      yp(il)=0.d0
      do it = 1, NT
      yp(il) = yp(il) + U_mat(it,il)*y(it)
      end do
      end do
c----  styp = S^t * U^t * y' (in Eq.A5a) 
       do il = 1, L
       styp (il) = vec_S(il)*yp(il)
       end do
c
       do ite = 1, itemax
       CALL  smac_update(styp,xp,zp,up,z,u,xi)
c         
        do il = 1, L
        stxp(il)= vec_S(il)*xp(il)
        end do
c
        sum = 0.d0
c---   convert rho' -> rho org.
        do io = 1, Nomega
        Ctmp1(io)=0.d0
        do il = 1, L
        Ctmp1(io) = Ctmp1(io) + V_t_mat(io,il)*xp(il)
        end do
        end do

         do io = 1, Nomega
         sum = sum + ABS(z(io)-Ctmp1(io))
         end do
c
       if(sum.lt.1.D-8) go to 100
      end do !--- ite loop
 100   continue
c
      do io = 1, Nomega
      xtmp=0.d0
      do il = 1, L
      xtmp= xtmp +  V_t_mat(io,il)*xp(il)
      end do
      xout(io)= xtmp
      end do
c
C-----------------------------------------------------------
      Return
      END
C***********************************************************
      subroutine smac_update(styp,xp,zp,up,z,u,xi)

      include 'para.h'
C------------Local variables--------------------------------
      real*8    z(Nomega), u(Nomega)
      real*8    xi(L),yp(L),styp(L),stxp(L),xp(L),zp(L),up(L)
      real*8    vec_temp2(Nomega),vec_temp(L),vec_temp3
      real*8    mat_inv(L,L)
      real*8    sum_V_xi1,sum_V_xi2,d_nu,d_mup_inv
      real*8    V_xi1(Nomega),V_xi2(Nomega)
      real*8    V_xp_u(Nomega),V_xp(Nomega),V_xp_u_tmp
      real*8    vec_xp_up(L), xi1(L),xi2(L),xi1_1(L),xi2_1(L)
      real*8    d_mu, d_mup, lambda,d_norm
C------------Main-------------------------------------------
c
      do il = 1, L
      vec_temp(il)=0.d0
      xi1(il)=0.d0
      xi2(il)=0.d0
      end do
c
      do io = 1, Nomega
      vec_temp2(io)=0.d0
      end do
c
      do io = 1, Nomega
      vec_temp2(io)=z(io)-u(io)
      end do
c
      do il = 1,L
      vec_temp3 = 0.d0
      do io = 1, Nomega
      vec_temp3 = vec_temp3 + V_t_mat(io,il)*vec_temp2(io)
      end do
      vec_temp(il)=vec_temp3
      end do
c
c-----   prepare xi1 and xi2 in Eq.A5a
      do il = 1,L
      xi1_1(il)=styp(il)/lambda + d_mup*(zp(il)-up(il))
     &        + d_mu*vec_temp(il)
      xi2_1(il)=xi(il)
      end do
c-----   prepare (S^t S + (mu'+mu))^{-1} in Eq.A5a
      do il1 = 1, L
      do il2 = 1, L
      mat_inv(il1,il2)=0.d0
      end do
      end do
c
      do i = 1, L
      mat_inv(i,i)= 1.d0/(vec_S(i)**2.d0/lambda + (d_mu + d_mup))
      end do
c-----   make xi1 and xi2 in Eq.A5a
      do il1 = 1, L
      xi1(il1)=0.d0
      xi2(il1)=0.d0
      do il2 = 1, L
      xi1(il1)=xi1(il1)+mat_inv(il1,il2)*xi1_1(il2)
      xi2(il1)=xi2(il1)+mat_inv(il1,il2)*xi2_1(il2)
      end do
      end do
c-----   prepare <>V xi1> and <V xi2> in Eq.A6
      sum_V_xi1 = 0.d0
      sum_V_xi2 = 0.d0
c      
      do io = 1, Nomega
      V_xi1(io)=0.d0
      V_xi2(io)=0.d0
      do il2 = 1, L
      V_xi1(io)= V_xi1(io)+ V_t_mat(io,il2)*xi1(il2)
      V_xi2(io)= V_xi2(io)+ V_t_mat(io,il2)*xi2(il2)
      end do
      end do
c
      do io = 1, Nomega
      sum_V_xi1 = sum_V_xi1 + V_xi1(io)
      sum_V_xi2 = sum_V_xi2 + V_xi2(io)
      end do
c-----   prepare nu in Eq.A5a
      d_nu= (1.d0 -sum_V_xi1 )/sum_V_xi2

c     Switch off the sum rule 
c     d_nu= 0.d0
c
c-----   This is x' in  Eq.A5a
      do il = 1, L
      xp(il) = xi1(il) + d_nu* xi2(il)
      end do
c-----   prepare for Eq.A5b
      do il = 1, L
      vec_xp_up(il) = xp(il) + up(il)
      end do
c-----   make z' in Eq.A5b
      d_mup_inv = 1.d0/d_mup
      CALL calc_salpha(zp,d_mup_inv,vec_xp_up)
c-----   This is u' in Eq.A5c
      do il = 1, L
      up(il)= up(il)+xp(il) -zp(il)
      end do
c-----   prepare (Vx' + u) in Eq.A5d
      do io = 1, Nomega
      V_xp_u_tmp=0.d0
      do il2 = 1, L
      V_xp_u_tmp=V_xp_u_tmp + V_t_mat(io,il2)*xp(il2)
      end do
      V_xp_u(io)=V_xp_u_tmp + u(io)
      end do
c-----   This is z in Eq.A5d
       CALL calc_P(z,V_xp_u)
c-----   prepare Vx' in Eq.A5e
      do io = 1, Nomega
      V_xp(io)=0.d0
      do il2 = 1, L
      V_xp(io)=V_xp(io) + V_t_mat(io,il2)*xp(il2)
      end do
      end do
c
      do io = 1, Nomega
      u(io)= u(io)+V_xp(io) -z(io)
      end do
C-----------------------------------------------------------
      Return
      END
C***********************************************************
      subroutine calc_P(vec_Px,x)
c
      include 'para.h'
C------------Local variables--------------------------------
      real*8    vec_Px(Nomega),x(Nomega)
C------------Main-------------------------------------------      
      do io = 1, Nomega 
      vec_Px(io)=max(x(io),0.d0)
      end do
C-----------------------------------------------------------
      Return
      END
C***********************************************************
      subroutine calc_salpha(salpha,alpha,x)
c
      include 'para.h'
C------------Local variables--------------------------------
      real*8    salpha(L), X(L)
      real*8    alpha
C------------Main-------------------------------------------
c      
      do il = 1, L
      salpha(il)=0.d0
      end do
c
      do il = 1, L
c
      if (x(il).gt.alpha) then
      salpha(il)= x(il) - alpha
      elseif(x(il).le.- alpha) then
      salpha(il) = x(il) + alpha
      else
      salpha(il)=0.d0
      end if
c
      end do
c
C-----------------------------------------------------------
      Return
      END
C***************************************************END*****

