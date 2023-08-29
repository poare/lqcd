!***********************************************************
! Sparse modeling approach to analytic continuation
!      of imaginary-time Monte Carlo data
!                            3 Apr 2020  E.Itou and Y. Nagai
!***********************************************************



program main
        use readfiles
        use para,only:NT,NT_org,Nomega,U_mat,V_t_mat,vec_S,lambda, d_mu, d_mup, d_norm,L
        implicit none
        !include 'para.h'
        !------------Local variables--------------------------------
        character*100 RFILE
        character*100 WFILE
        character*6 inomega
        real(8),allocatable,dimension(:,:)::mat_K,U,V_t,V
        real(8),allocatable,dimension(:)::data_S,e,xi,vec_G,vec_Gout,xout

        !real*8    mat_K(NT,Nomega)
        !real*8    U(NT,NT)
        !real*8    V_t(Nomega,NT), V(NT,Nomega)
        !real*8    data_S(NT)
        !real*8    e(Nomega),xi(NT),vec_G(NT),vec_Gout(NT),xout(Nomega)
        !      integer   L
        real*8    data_G,temp_G,Omega_max,data_dum
        real*8    domega
        character(:), allocatable::cvalue
        
        integer::it,io,ii,il
        
        character(:), allocatable::arg1

        
!
!------------Main-------------------------------------------
        NT = 10
        Nomega = 2001
        NT_org = 10
        call readarg(arg1)
        call readfromfiles(arg1,"NT",NT)
        call readfromfiles(arg1,"Nomega",Nomega)
        call readfromfiles(arg1,"NT_org",NT_org)
        

        allocate(U_mat(NT,NT))
        allocate(V_t_mat(Nomega,NT))
        allocate(vec_S(NT))

        allocate(mat_K(NT,Nomega))
        allocate(  U(NT,NT))
        allocate(     V_t(Nomega,NT), V(NT,Nomega))
        allocate(     data_S(NT))
        allocate(     e(Nomega),xi(NT),vec_G(NT),vec_Gout(NT),xout(Nomega))


!
        domega = 2.d0/DBLE(Nomega-1)
        !
        inomega= CHAR( ICHAR('0')+MOD(Nomega/100000,10)) &
                //CHAR(ICHAR('0')+MOD(Nomega/10000,10))   &
                //CHAR(ICHAR('0')+MOD(Nomega/1000,10))    &
                //CHAR(ICHAR('0')+MOD(Nomega/100,10))     &
                //CHAR(ICHAR('0')+MOD(Nomega/10,10))      &
                //CHAR(ICHAR('0')+MOD(Nomega,10))    
        !---    read K' = K * omega * Delta_omega
        RFILE='./data-K-Nt16-Ndomega'//inomega//'.dat'

        OPEN(20,FILE=RFILE,STATUS='UNKNOWN',FORM='FORMATTED')

        write(*,*) 'read file',RFILE

        read(20,*)  Omega_max
        write(*,*) ' Lambda=omega_max*Ntau=', Omega_max

        do it = 1, NT
                do io = 1, Nomega
                !       read(20, '(4000e12.4)')  mat_K(it, 1:Nomega)
                        read(20, *)  mat_K(it, io)
                end do
        end do

        CLOSE(20)
        !------------
        !       RFILE='./rescale-EMT-data-corr-T12-beta6.93-Ns64-Nt16.dat'
        !RFILE='./rescale-EMT-reduced-data-T12-beta6.93-Ns64-Nt16.dat'
        call readfromfiles(arg1,"Ctau",cvalue)
        RFILE = trim(ADJUSTL(cvalue))
        !call mygetarg(1, RFILE)
        write(*,*) "readfile = ",RFILE

        OPEN(19,FILE=RFILE,STATUS='UNKNOWN',FORM='FORMATTED')

        write(*,*) 'read file',RFILE

        do it = 1, NT
                read(19, *)  data_dum,data_dum,data_G,data_dum
                vec_G(it)=data_G
        end do

        CLOSE(19)
        !----------- cucl d_norm for sum rule
        !       
        d_norm = vec_G(1)/dsqrt(domega)
        write(*,*) 'd_norm', d_norm    
        do it = 1, NT
                temp_G = vec_G(it)
                vec_G(it)= temp_G/d_norm
        end do
        !-----------

        !---     read SVD of K'  
        RFILE='./data-s-value'//inomega//'.dat'

        OPEN(21,FILE=RFILE,STATUS='UNKNOWN',FORM='FORMATTED')
        do it = 1, NT
                read (21, *) data_S(it)
        end do
        CLOSE(21)
        !
        RFILE='./data-matrix-U'//inomega//'.dat'

        OPEN(21,FILE=RFILE,STATUS='UNKNOWN',FORM='FORMATTED')
        do it = 1, NT
                do io = 1, NT
                        read (21, *) U(it,io)
                end do
        end do
        CLOSE(21)

        RFILE='./data-matrix-V'//inomega//'.dat'

        OPEN(21,FILE=RFILE,STATUS='UNKNOWN',FORM='FORMATTED')
        do it = 1, NT
                do io = 1, Nomega
                        read (21, *) V(it,io)
                end do
        end do
        CLOSE(21)


        do it = 1, NT
                do io = 1, Nomega
                        V_t(io,it)=V(it,io)
                end do
        end do


        write(*,*) 'read file done'

!-----------  
        ii = 0
        do it = 1, NT
                ii = ii + 1
        !        write(*,*) data_S(it)
                if(data_S(it).le. 1.D-10) exit !go to 200
        end do
        !200   continue

        L = ii
        write(*,*) 'L=', L
!------------
        CALL mat_reduced(data_S,U,V_t)
!
        write(*,*) 'mat_reduced done' 

        do io = 1, Nomega
                e(io)=1.d0
        end do
!
        do il = 1, L
                xi(il)=0.d0
                do io = 1, Nomega
                        xi(il) = xi(il) + V_t_mat(io,il)*e(io)
                end do
                write(*,*) 'xi in main',il,xi(il)
        end do
!
        do it = 1, NT
                vec_Gout(it)=0.d0
        end do
!
        do io = 1, Nomega
                xout(io)=0.d0
        end do
!
        write(*,*) 'before smac' 
        CALL smac_est(xout,vec_G,mat_K,xi)
        write(*,*) 'end smac' 
        !
        !---------
        !----   write rho' = rho(omega)/omega
        WFILE='./data-rho-Nomega'//inomega//'.dat'

        OPEN(23,FILE=WFILE,STATUS='UNKNOWN',FORM='FORMATTED')
        do io = 1, Nomega
        !        write(23,*) io,xout(io)
                write(23,*) (DBLE(io-1)*domega-1.d0)*Omega_max/DBLE(NT_org), &
                xout(io)*d_norm/dsqrt(domega)/DBLE(NT_org)**4.d0/Omega_max 
        end do
        CLOSE(23)
        !----  write output correlator sum_omega (K*omega*Delta_omega)*rho/omega
        WFILE='./data-C_tau_result-Nomega'//inomega//'.dat'

        OPEN(24,FILE=WFILE,STATUS='UNKNOWN',FORM='FORMATTED')

        do it = 1, NT
                do io = 1, Nomega
                        vec_Gout(it)=vec_Gout(it)+mat_K(it,io)*xout(io)*d_norm
                end do 
                write(24,*) 'it,vec_G',it,vec_Gout(it)
        end do
        CLOSE(24)

        !-----------------------------------------------------------
        STOP
        
end program main
!***********************************************************
!***************************************************END*****


