module para
    integer::NT,NT_org,Nomega
    integer::L
    real(8),allocatable::U_mat(:,:)
    real(8),allocatable::V_t_mat(:,:)
    real(8),allocatable::vec_S(:)
    real(8)::lambda, d_mu, d_mup, d_norm

!    COMMON /MATU/   U_mat(NT,NT)
!    COMMON /MATV_t/   V_t_mat(Nomega,NT)
!    COMMON /VecS/   vec_S(NT)
!
! ---Simulation parameters---
!    COMMON /param/ lambda, d_mu, d_mup, d_norm
end module