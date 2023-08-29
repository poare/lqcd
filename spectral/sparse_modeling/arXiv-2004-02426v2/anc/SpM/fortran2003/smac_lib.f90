!***********************************************************
! Sparse modeling approach to analytic continuation
!      of imaginary-time Monte Carlo data
!                            3 Apr 2020  E.Itou and Y. Nagai
!               See Appendix A in arXiv:1702.03056[cond-mat]
!***********************************************************

!
      subroutine mat_reduced(S_data,U,V_t)
            use para
            implicit none
      !      implicit real*8 (a-h,o-z)
      !      parameter (Nomega_max=20)
      !      parameter (NT_max=60)

            !include 'para.h'

!------------Local variables--------------------------------
            real*8    S_data(NT)
            real*8    U(NT,NT)
            real*8    V_t(Nomega,NT)
            integer::i,it,il,io
!      real*8    vec_S(L)
!      real*8    mat_U(NT,L)
!      real*8    mat_V_t(Nomega,L)
!------------Main-------------------------------------------
!           vec_S = 0d0
            do i = 1, L
            vec_S(i)= S_data(i)
            end do
      !
            do it = 1,NT
            do il = 1,L
            U_mat(it,il)= U(it,il)
            end do
            end do
      !
            
            do io = 1, Nomega
            do il = 1, L
            V_t_mat(io,il)=V_t(io,il)
            end do
            end do
!-----------------------------------------------------------
            Return
      END subroutine
!***********************************************************
!
      subroutine smac_est(xout,vec_G,mat_K,xi)
            use para,only:d_mu, d_mup, lambda,NT,Nomega,L
            implicit none
!            include 'para.h'

!------------Local variables--------------------------------
            real*8    vec_G(NT)
            real*8    mat_K(NT,Nomega)
            real*8    xi(L)
            real*8    xout(Nomega)
            real*8    Ctmp(NT)
      !
            real*8    lambda_est, fchiold, fchi, a, b
            !real*8    d_mu, d_mup, lambda
            real*8    l0, lam0, l1, lam1, ll
            real*8    chi0, chi1, log_chi0, log_chi1
            integer::itemax,it,io,il
!------------Main-------------------------------------------
!
            d_mu =1.d0
            d_mup=1.d0
      !       d_mu=0.01d0
      !       d_mup=0.01d0
            fchiold= 0.d0
            lambda_est=0.d0
      !
      !       l0= -6.d0
            l0= -15.d0
            lam0=10.d0**(l0)
            lambda = lam0
            write(*,*) 'in est lambda',l0,lam0,lambda
            itemax = 10000
!   
!------------
!        do i = 1, L
!        write(*,*) 'xi in est',xi(i)
!        end do
            CALL smac_calc(xout,vec_G,xi,itemax)
      !
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
!
!-------
            l1 = 2.d0
            lam1 = 10.d0**(l1)
            lambda=lam1     
!
            CALL smac_calc(xout,vec_G,xi,itemax)
!
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
!
            b = (dlog(chi0)-dlog(chi1))/(dlog(lam0)-dlog(lam1))
            a = dexp(dlog(chi0)-b*dlog(lam0))
!-------
            do il = 2, 100-1
                  ll = (il -1)*(l1-l0)/(100-1)+l0
            !        write(*,*) 'll=',ll
                  lambda=10.d0**(ll)
            !
                  CALL smac_calc(xout,vec_G,xi,itemax)
!
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
!
                  if(il.gt.3) then
                        if (fchi.gt.fchiold) then
                              lambda_est = lambda
                              fchiold = fchi
                              write(*,*) 'update lambda_est'
                        end if
                  else
                        fchiold = fchi
                  end if
!
            end do !--- il loop
!
            lambda = lambda_est
!          itemax = 10000000
            itemax = 1000000000
!---- 
            write(*,*) 'Calc. final smac,lambda',lambda
            CALL smac_calc(xout,vec_G,xi,itemax)        
!
!-----------------------------------------------------------
            Return
      END subroutine
!***********************************************************
      subroutine smac_calc(xout,y,xi,itemax)
            use para,only:L,Nomega,NT,U_mat,vec_S,V_t_mat
            implicit none
!            include 'para.h'  
      !------------Local variables--------------------------------
            real*8    xout(Nomega), z(Nomega), u(Nomega), Ctmp1(Nomega)
            real*8    y(NT)
            real*8    xi(L),yp(L),styp(L),stxp(L),xp(L),zp(L),up(L)
            real*8    xtmp, sum
            integer::il,io,ite,itemax,it
      !------------Main-------------------------------------------

            do il = 1, L
                  yp(il) = 0.d0
                  styp(il) = 0.d0
                  stxp(il) = 0.d0
                  xp(il) = 0.d0
                  zp(il) = 0.d0
                  up(il) = 0.d0
            end do
!
            do io = 1, Nomega
                  z(io) = 0.d0
                  u(io) = 0.d0
            end do
!----   prepare U^t * vec_G(tau)
            do il = 1, L
                  yp(il)=0.d0
                  do it = 1, NT
                        yp(il) = yp(il) + U_mat(it,il)*y(it)
                  end do
            end do
!----  styp = S^t * U^t * y' (in Eq.A5a) 
            do il = 1, L
            styp (il) = vec_S(il)*yp(il)
            end do
!
            do ite = 1, itemax
      !       write(*,*) 'ite in smac_calc',ite
                  CALL  smac_update(styp,xp,zp,up,z,u,xi)

                  do il = 1, L
                        stxp(il)= vec_S(il)*xp(il)
                  end do
      !
                  sum = 0.d0
            !---   convert rho' -> rho org.
!                  do io = 1, Nomega
!                        Ctmp1(io)=0.d0
!                        do il = 1, L
!                              Ctmp1(io) = Ctmp1(io) + V_t_mat(io,il)*xp(il)
!                        end do
!                  end do
                  Ctmp1 = matmul(V_t_mat(1:Nomega,1:L),xp(1:L))

                  do io = 1, Nomega
                        sum = sum + ABS(z(io)-Ctmp1(io))
                  end do
            !if (mod(ite,10) == 0) then
            !      write(*,*) ite,sum
            !endif
                  if(sum.lt.1.D-8) exit !go to 100
            end do !--- ite loop
! 100   continue
!
            xout = matmul(V_t_mat(1:Nomega,1:L),xp(1:L))
!            do io = 1, Nomega
!                  xtmp=0.d0
!                  do il = 1, L
!                        xtmp= xtmp +  V_t_mat(io,il)*xp(il)
!                  end do
!                  xout(io)= xtmp
!            end do
!
!-----------------------------------------------------------
            Return
      END subroutine
!***********************************************************
      subroutine smac_update(styp,xp,zp,up,z,u,xi)
            use para,only:L,Nomega,lambda,d_mu,d_mup,d_norm,V_t_mat,vec_S
            implicit none
!            include 'para.h'
      !------------Local variables--------------------------------
            real*8    z(Nomega), u(Nomega)
            real*8    xi(L),yp(L),styp(L),stxp(L),xp(L),zp(L),up(L)
            real*8    vec_temp2(Nomega),vec_temp(L),vec_temp3
            real*8    mat_inv(L,L)
            real*8    sum_V_xi1,sum_V_xi2,d_nu,d_mup_inv
            real*8    V_xi1(Nomega),V_xi2(Nomega)
            real*8    V_xp_u(Nomega),V_xp(Nomega),V_xp_u_tmp
            real*8    vec_xp_up(L), xi1(L),xi2(L),xi1_1(L),xi2_1(L)
            !real*8    d_mu, d_mup, lambda,d_norm
            integer::il,io,il1,i,il2
      !------------Main-------------------------------------------
      
!      write(*,*) 'lambda=',lambda
            do il = 1, L
                  vec_temp(il)=0.d0
                  xi1(il)=0.d0
                  xi2(il)=0.d0
            end do

            do io = 1, Nomega
                  vec_temp2(io)=0.d0
            end do
!
            do io = 1, Nomega
                  vec_temp2(io)=z(io)-u(io)
            end do
!      write(*,*) 'vec_temp2,z,u',vec_temp2(1),z(1),u(1)

            do il = 1,L
                  vec_temp3 = 0.d0
                  do io = 1, Nomega
                        vec_temp3 = vec_temp3 + V_t_mat(io,il)*vec_temp2(io)
                  end do
                  vec_temp(il)=vec_temp3
            end do
      !      write(*,*) 'vec_temp', vec_temp(1)
      !-----   prepare xi1 and xi2 in Eq.A5a
            do il = 1,L
                  xi1_1(il)=styp(il)/lambda + d_mup*(zp(il)-up(il)) &
                        + d_mu*vec_temp(il)
                  xi2_1(il)=xi(il)
            end do
      !      write(*,*) 'xi,styp',xi(1),styp(1)
      !      write(*,*) 'xi1,2',xi1_1(1),xi2_1(1)

!-----   prepare (S^t S + (mu'+mu))^{-1} in Eq.A5a
            do il1 = 1, L
                  do il2 = 1, L
                        mat_inv(il1,il2)=0.d0
                  end do
            end do
!
            do i = 1, L
                  mat_inv(i,i)= 1.d0/(vec_S(i)**2.d0/lambda + (d_mu + d_mup))
      !      write(*,*) 'mat_inv',mat_inv(i,i)
            end do
!-----   make xi1 and xi2 in Eq.A5a
            xi1 = matmul(mat_inv(1:L,1:L),xi1_1(1:L))
            xi2 = matmul(mat_inv(1:L,1:L),xi2_1(1:L))
            !do il1 = 1, L
            !      xi1(il1)=0.d0
            !      xi2(il1)=0.d0
            !      do il2 = 1, L
            !            xi1(il1)=xi1(il1)+mat_inv(il1,il2)*xi1_1(il2)
            !            xi2(il1)=xi2(il1)+mat_inv(il1,il2)*xi2_1(il2)
            !      end do
            !end do
!           write(*,*) 'xi1,2',xi1(1),xi2(1)

!-----   prepare <>V xi1> and <V xi2> in Eq.A6
            sum_V_xi1 = 0.d0
            sum_V_xi2 = 0.d0

            V_xi1 = matmul(V_t_mat(1:Nomega,1:L),xi1(1:L))
            V_xi2 = matmul(V_t_mat(1:Nomega,1:L),xi2(1:L))
            
!            do io = 1, Nomega
!                  V_xi1(io)=0.d0
!                  V_xi2(io)=0.d0
!                  do il2 = 1, L
!                        V_xi1(io)= V_xi1(io)+ V_t_mat(io,il2)*xi1(il2)
!                        V_xi2(io)= V_xi2(io)+ V_t_mat(io,il2)*xi2(il2)
!                  end do
!            end do
!
            do io = 1, Nomega
                  sum_V_xi1 = sum_V_xi1 + V_xi1(io)
                  sum_V_xi2 = sum_V_xi2 + V_xi2(io)
            end do
!-----   prepare nu in Eq.A5a
            d_nu= (1.d0 -sum_V_xi1 )/sum_V_xi2
!      d_nu= 0.d0


!-----   This is x' in  Eq.A5a
            do il = 1, L
                  xp(il) = xi1(il) + d_nu* xi2(il)
            end do
!-----   prepare for Eq.A5b
            do il = 1, L
                  vec_xp_up(il) = xp(il) + up(il)
            end do
!-----   make z' in Eq.A5b
            d_mup_inv = 1.d0/d_mup
            CALL calc_salpha(zp,d_mup_inv,vec_xp_up)
      !-----   This is u' in Eq.A5c
            do il = 1, L
                  up(il)= up(il)+xp(il) -zp(il)
            end do
!-----   prepare (Vx' + u) in Eq.A5d
            V_xp_u = matmul(V_t_mat(1:Nomega,1:L),xp(1:L))
!            do io = 1, Nomega
!                  V_xp_u_tmp=0.d0
!                  do il2 = 1, L
!                        V_xp_u_tmp=V_xp_u_tmp + V_t_mat(io,il2)*xp(il2)
!                  end do
!                  V_xp_u(io)=V_xp_u_tmp + u(io)
!            end do
!-----   This is z in Eq.A5d
            CALL calc_P(z,V_xp_u)
!-----   prepare Vx' in Eq.A5e
            V_xp = matmul(V_t_mat(1:Nomega,1:L),xp(1:L))
!            do io = 1, Nomega
!                  V_xp(io)=0.d0
!                  do il2 = 1, L
!                        V_xp(io)=V_xp(io) + V_t_mat(io,il2)*xp(il2)
!                  end do
!            end do
!
            do io = 1, Nomega
                  u(io)= u(io)+V_xp(io) -z(io)
            end do
!-----------------------------------------------------------
            Return
      END subroutine
!***********************************************************
      subroutine calc_P(vec_Px,x)
            use para
            implicit none
!            include 'para.h'

      !------------Local variables--------------------------------
            real*8    vec_Px(Nomega),x(Nomega)
            integer::io
      !------------Main-------------------------------------------
            
            do io = 1, Nomega 
                  vec_Px(io)=max(x(io),0.d0)
            end do
!-----------------------------------------------------------
            Return
      END subroutine
!***********************************************************
      subroutine calc_salpha(salpha,alpha,x)
            use para
            implicit none
!            include 'para.h'

      !------------Local variables--------------------------------
            real*8    salpha(L), X(L)
            real*8    alpha
            integer::il
      !------------Main-------------------------------------------
            
            do il = 1, L
                  salpha(il)=0.d0
            end do
      !
            do il = 1, L
      !
                  if (x(il).gt.alpha) then
                        salpha(il)= x(il) - alpha
                  elseif(x(il).le.- alpha) then
                        salpha(il) = x(il) + alpha
                  else
                        salpha(il)=0.d0
                  end if
            !
            end do
      !
      !-----------------------------------------------------------
            Return
      END subroutine
!***************************************************END*****

