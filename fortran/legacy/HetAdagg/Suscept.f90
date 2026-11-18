


        Function ChiM(omega,gammaM,omegaM)
         IMPLICIT NONE
         double precision::omega,gammaM,omegaM,T1,T2,BNORM
         double complex::ChiM
! MECHANICAL susceptibility
! chi_M
        BNORM=(GammaM)**2 + (omega-OmegaM)**2
      t1= (GammaM)/BNORM
      t2=(omega-OmegaM)/BNORM
      ChiM=CMPLX(t1,t2)
 
       Return
       End
!-------------------------------------
         Function Chimom(omega,gammaM,OmegaM)
         IMPLICIT NONE
         double precision::omega,gammaM,omegaM,T1,T2,BNORM
         double complex::ChiMom
!   MECHANICAL susceptibility conjg, -omega
! chi_MinusOmega= Chi_mom

! CHI_M^*(-omega)
        BNORM=(GammaM)**2 + (omega+OmegaM)**2
      t1= (GammaM)/BNORM
       t2=(omega+OmegaM)/BNORM
       Chimom=CMPLX(t1,t2)
       Return
       End

!*****************************************
       Function ChiR(omega,kapp2,Detun)
      IMPLICIT NONE
     double precision::omega,Kapp2,Detun,T1,T2,ANORM
    double complex::ChiR
!Chi_R
       ANORM=KAPP2**2+ (omega+Detun)**2
       t1=kapp2/ANORM
       t2= (omega+Detun)/ANORM
       ChiR=CMPLX(t1,t2)
        Return
        End
!------------------------------------------

      Function CHISTMOM(omega,kapp2,Detun)
     IMPLICIT NONE
     double precision::omega,Kapp2,Detun,T1,T2,ANORM
     double complex::ChistMom
! chi_r^*(-omega) = Chi^star (-Omega)=ChistMom
      ANORM=KAPP2**2+ (-omega+Detun)**2
      t1=kapp2/ANORM
      t2= (-omega+Detun)/ANORM
      ChistMom=CMPLX(t1,-t2)

      Return
       End
!******************************************


