C***********************************************************
C Sparse modeling approach to analytic continuation
C      of imaginary-time Monte Carlo data
C                                   5 Dec 2018    E.Itou
C***********************************************************
      include 'para.h'
C------------Local variables--------------------------------
      character*100 RFILE
      character*100 WFILE
      character*6 inomega
c
      real*8    mat_K(NT,Nomega)
      real*8    U(NT,NT)
      real*8    V_t(Nomega,NT), V(NT,Nomega)
      real*8    data_S(NT)
      real*8    e(Nomega),xi(NT),vec_G(NT),vec_Gout(NT),xout(Nomega)
      real*8    lambda, d_mu, d_mup, data_G,d_norm,temp_G,Omega_max
      real*8    domega
c
C------------Main-------------------------------------------
C
      domega = 2.d0/DBLE(Nomega-1)
c
      inomega= CHAR( ICHAR('0')+MOD(Nomega/100000,10)) 
     &      //CHAR(ICHAR('0')+MOD(Nomega/10000,10))   
     &      //CHAR(ICHAR('0')+MOD(Nomega/1000,10))    
     &      //CHAR(ICHAR('0')+MOD(Nomega/100,10))     
     &      //CHAR(ICHAR('0')+MOD(Nomega/10,10))      
     &      //CHAR(ICHAR('0')+MOD(Nomega,10))    
C---    read K' = K * omega * Delta_omega
       RFILE='./data-K-Nt16-Ndomega'//inomega//'.dat'
c
       OPEN(20,FILE=RFILE,STATUS='UNKNOWN',FORM='FORMATTED')
c
       write(*,*) 'read file',RFILE
c
       read(20,*)  Omega_max
       write(*,*) ' Lambda=omega_max*Ntau=', Omega_max
c
       do it = 1, NT
       do io = 1, Nomega
       read(20, *)  mat_K(it, io)
       end do
       end do
c
       CLOSE(20)
c------------
       RFILE='./samplectau.in'
c
       OPEN(19,FILE=RFILE,STATUS='UNKNOWN',FORM='FORMATTED')
c
       write(*,*) 'read file',RFILE
c
       do it = 1, NT
       read(19, *)  data_dum,data_dum,data_G,data_dum
       vec_G(it)=data_G
       end do
c
       CLOSE(19)
c---   set d_norm for sum rule
c       
       d_norm = vec_G(1)/dsqrt(domega)
       write(*,*) 'd_norm', d_norm    
       do it = 1, NT
       temp_G = vec_G(it)
       vec_G(it)= temp_G/d_norm
       end do
c
c---   read SVD of K'  
       RFILE='./data-s-value'//inomega//'.dat'
c
        OPEN(21,FILE=RFILE,STATUS='UNKNOWN',FORM='FORMATTED')
        do it = 1, NT
        read (21, *) data_S(it)
        end do
        CLOSE(21)
c
       RFILE='./data-matrix-U'//inomega//'.dat'
c
        OPEN(21,FILE=RFILE,STATUS='UNKNOWN',FORM='FORMATTED')
        do it = 1, NT
        do io = 1, NT
        read (21, *) U(it,io)
        end do
        end do
        CLOSE(21)
c
       RFILE='./data-matrix-V'//inomega//'.dat'
c
        OPEN(21,FILE=RFILE,STATUS='UNKNOWN',FORM='FORMATTED')
        do it = 1, NT
        do io = 1, Nomega
        read (21, *) V(it,io)
        end do
        end do
        CLOSE(21)
c
        do it = 1, NT
        do io = 1, Nomega
        V_t(io,it)=V(it,io)
        end do
        end do
c
        write(*,*) 'read file done'
c
c-----------  
        ii = 0
        do it = 1, NT
        ii = ii + 1
        if(data_S(it).le. 1.D-10) go to 200
        end do
 200   continue
c
        L = ii
        write(*,*) 'L=', L
c------------
        CALL mat_reduced(data_S,U,V_t)
c
        write(*,*) 'mat_reduced done' 
c
        do io = 1, Nomega
        e(io)=1.d0
        end do
c
        do il = 1, L
        xi(il)=0.d0
        do io = 1, Nomega
        xi(il) = xi(il) + V_t_mat(io,il)*e(io)
        end do
        write(*,*) 'xi in main',il,xi(il)
        end do
c
        do it = 1, NT
        vec_Gout(it)=0.d0
        end do
c
        do io = 1, Nomega
        xout(io)=0.d0
        end do
c
        write(*,*) 'before smac' 
        CALL smac_est(xout,vec_G,mat_K,xi)
        write(*,*) 'end smac' 
c
c----  write \tilde{rho}' 
       WFILE='./data-rho-Nomega'//inomega//'.dat'

       OPEN(23,FILE=WFILE,STATUS='UNKNOWN',FORM='FORMATTED')
        do io = 1, Nomega
c        write(23,*) io,xout(io)
        write(23,*) (DBLE(io-1)*domega-1.d0)*Omega_max/DBLE(NT_org),
     &    xout(io)*d_norm/dsqrt(domega)/DBLE(NT_org)**4.d0/Omega_max 
        end do
       CLOSE(23)
c----  write output correlator 
       WFILE='./data-C_tau_result-Nomega'//inomega//'.dat'

       OPEN(24,FILE=WFILE,STATUS='UNKNOWN',FORM='FORMATTED')

        do it = 1, NT
        do io = 1, Nomega
        vec_Gout(it)=vec_Gout(it)+mat_K(it,io)*xout(io)*d_norm
        end do 
        write(24,*) 'it,vec_G',it,vec_Gout(it)
        end do
       CLOSE(24)

C-----------------------------------------------------------
      STOP
      END
C***********************************************************
C***************************************************END*****

        subroutine mygetarg(i, argc)
                implicit none
                integer, intent(in) :: i
                character(len=*), intent(out) :: argc
                
                call getarg(0, argc)
                if(argc == "") then
                        call getarg(i + 1, argc)
                else
                        call getarg(i, argc)
                end if
        end subroutine
        
