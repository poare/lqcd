C***********************************************************
C  Common variables for Sparse modeling analysis
C 
C                          6 Dec 2018  E.Itou
C-----------------------------------------------------------
      IMPLICIT REAL*8 (A-H,O-Z)
C------------Variables--------------------------------------
       PARAMETER(NT = 16)
       PARAMETER(NT_org = 16)
       PARAMETER(Nomega = 3001)
C 
       COMMON /L/  L
C
C  matrices  variable 
       COMMON /MATU/   U_mat(NT,NT)
       COMMON /MATV_t/   V_t_mat(Nomega,NT)
       COMMON /VecS/   vec_S(NT)
C
C ---Simulation parameters---
       COMMON /param/ lambda, d_mu, d_mup, d_norm
C
C-----------------------------------------------------------
