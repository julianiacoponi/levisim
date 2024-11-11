
!********************************************************************
!  Generic routine for noise spectra of trap and probe beams
!********************************************************************

  SUBROUTINE ANALYT(Omega,Theta,Detun,Kapp2,AVphot,AVN,Gmat,GMM,Phases,GAMMAMod,OMEGS,PSD,CrossSpec,SHOM1,SHET1)


!************************************************************************
!************************************************************************
            
! dimension N= 3 modes gives 6 equations
!     x,y,z, P_x,Py,P_z =6x Npcle + 2 optical (a,a^\dagg)
! 1 pcle => NT=8; 2 pcle => NT=14.
 ! Neglect z? NT= b_x,b_x^\dagg,b_y,b_y^\dagg x 2 particles + 2opt= NT=6.
!**************************************************************
	IMPLICIT NONE
        integer  ::ii,m,jj,iterm,ipcle,imode,Npcle,Nmode,NT
      include 'Nparticle.h'
    
        double precision:: Detun,GammaM,OmegaM,Kapp2,Theta,Omega
        double precision::omhet,oma, AVphot
        double precision::pi,pi2,SHOM1,SHET1,CrossSpec

       Double precision :: AVN(NT)
       double precision:: Phases(Npcle)
!   coupling to light
         double precision::GMAT(npcle,Nmode)
! direct coupling
         double precision:: GMM(npcle,Nmode)

        double precision:: OMEGS(Npcle,Nmode),GAMMAMod(Npcle,Nmode)
        Double precision:: PSD(Npcle,Nmode)
 
        double complex::Chim,Chimom,ChiR,Chistmom
! Optical susceptibilities
        double complex::ChiOpt, ChiOptMinus
! mechanical susceptibilities
       double complex::Chi(Npcle,Nmode),ChiMinus(Npcle,Nmode)
       double complex:: Xvec(Npcle,Nmode,NT),A1(NT),A1dagg(NT)
        pi=dacos(-1.d0)
        SHOM1=0.d0
        SHET1=0.d0
        
 !----------------------------------------------------------
 !  X=1; Y=2; Z=3 in chi(Npcle,3), chiMinus(Npcle,3) etc
! zero all arrays as a precaution
       
        PSD=0.d0
 
        CrossSpec=0.d0
        Chi=0.d0
        ChiMinus=0.d0
        Xvec=0.d0
        A1=0.d0
        A1dagg=0.d0
!********************************************
! First do mechanical susceptibilities
           do ipcle=1,Npcle
!x,y,z
! If Nmode=2  only x,y
     do imode=1,Nmode
      OmegaM=OMEGS(ipcle,imode)
      gammaM=GammaMod(ipcle,imode)

      chi(ipcle,imode)=ChiM(omega,gammaM,omegaM)
      chiMinus(ipcle,imode)=Chimom(omega,gammaM,OmegaM)
      enddo
            enddo
!******************************************
!  Now do optical susceptibilities
      ChiOpt=ChiR(omega,kapp2,Detun)
      ChiOptMinus=CHISTMOM(omega,kapp2,Detun)
! ********************************************
  

! work out Xvec=  vector of noises (inc optical) for mechanical modes
! And also from the above get optical noise vectors a1 and a1dagg (only one optical mode here)
   
    Call Noisevect(kapp2,GMAT,GMM,GammaMod,Phases,ChiOpt,ChiOptMinus,Chi,ChiMinus,A1,A1dagg,Xvec)

! *******WORK OUT MECHANICAL PSD and cross SPECTRA
      Call PSDNoise(Avphot,AVN,Xvec,PSD,CrossSpec)


!&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
! work out homodyne spectra using obtained vectors

      call Homodyne(AVN,AVPHOT,THETA,A1,A1dagg,SHOM1)
! rescaling of heterodyne for Gx/Gy to get displacement spectra.
 !    Gav=(GMAT(1)+GMAT(2))/2.
!     SHOM1=(SHOM1-1.d0)/abs(CHiOpt-ChiOptMinus)**2/Gav**2/kapp2/2.

! work out heterodyne spectra using obtained vectors-NO FREQUENCY SHIFT

      call Heterodyne(AVN,AVPHOT,THETA,A1,A1dagg,SHET1)
      
100   format(6D16.8)
      RETURN
       END



!&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
!  work out mechanical PSDs: work in progress to add cross correlations
      Subroutine PSDNoise(Avphot,AVN,Xvec,PSD,CrossSpec)

!&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
       IMPLICIT NONE

       integer  ::ii,jj,iterm,ipcle,imode,Npcle,Nmode,NT,icount
        integer:: imode1,imode2,ipcle1,ipcle2
       include 'Nparticle.h'
     double precision :: AvPhot

     Double precision:: CrossSpec
     double complex::  cross
     double precision:: AVN(NT)

    Double precision:: PSD(Npcle,Nmode)
      double complex:: Xvec(Npcle,Nmode,NT)
 ! Xvec contains vectors of :
  ! a_in,a_in^\dagg, b_in,x1,b^\dagg_in,x1,b_in,y1,b^\dagg_in,y1,...b_in,x2, b^\dagg_in,x2...

         PSD=0.d0
         CrossSpec=0.d0

!  Add optical noise contributions ie components with a_in, a_in^\dagger. Single optical mode
         icount=0
        do 10 ipcle=1,Npcle
  do 20 imode=1,Nmode
 icount=icount+1
PSD(ipcle,imode)=PSD(ipcle,imode)+(AVphot+1)*(ABS(Xvec(ipcle,imode,1)))**2+AVphot*(ABS(Xvec(ipcle,imode,2)))**2
    20      enddo
         10    enddo
     if(icount+2.gt.NT) write(6,*)'warning: dimensions in PSD wrong',icount+2,NT
!------------------------------------------------------------------------------------
!  Add MECHANICAL noise contributions ie components with b_in, b_in^\dagger
! note phonon occupancies AVN(3)=AVN(4)=AVNX1; AVN(5)=AVN(6)=AVNY1
! clearly AVN(1)=AVN(2)=0 since that stores optical mode info.
            do 11 ipcle=1,Npcle
        do 21 imode=1,Nmode
    do iterm=2,NT-2,2
 PSD(ipcle,imode)=PSD(ipcle,imode)+(AVN(iterm+1)+1.)*(ABS(Xvec(ipcle,imode,iterm+1)))**2 &
                                   +AVN(iterm+1)*(ABS(Xvec(ipcle,imode,iterm+2)))**2
    enddo 
 21     enddo
 11          enddo
   

!********************************************
! cross correlations for a pair of modes
!  pick two modes
     imode1=1
     imode2=2
     ipcle1=1
     ipcle2=1
!  optical terms
        Cross=(AVphot+1)*Xvec(ipcle1,imode1,1)*CONJG(Xvec(ipcle2,imode2,1))
    Cross=Cross+AVphot*Xvec(ipcle1,imode1,2)*CONJG(Xvec(ipcle2,imode2,2))

do iterm=2,NT-2,2
    Cross=Cross+(AVN(iterm+1)+1.)*Xvec(ipcle1,imode1,iterm+1)*CONJG(Xvec(ipcle2,imode2,iterm+1))
     Cross=Cross+AVN(iterm+1)*Xvec(ipcle1,imode1,iterm+2)*CONJG(Xvec(ipcle2,imode2,iterm+2))
enddo
  CrossSpec=Real(Cross)
! for non-zero they need to have been appropriately populated in Noisevec
!*************************************************
return
   end


!*************************************************
 ! Work out homodyne spectra
!**************************************************
      Subroutine Homodyne(AVN,AVPHOT,THETA,A1,A1dagg,SHOM1)
        IMPLICIT NONE
     integer  ::ii,m,jj,iterm,NT,NTOT,npcle,Nmode
     include 'Nparticle.h'
         double precision:: THETA,SHOM1,AVPHOT
       double complex::XI

       double precision::AVN(NT)
        double complex::XTHET1(NT)
         double complex:: XPM1(NT),XAM1(NT)
      double complex:: A1(NT)
      double complex:: A1dagg(NT)

      NTOT=NT
!  zero arrays
           XPM1=(0.d0,0.d0)
           XAM1=(0.d0,0.d0)

           XTHET1=(0.d0,0.d0)

! i
        XI=CMPLX(0.d0,1.d0)
! Optical quadrature
       do ii=1,Ntot
        XAM1(ii)=A1(ii)+A1dagg(ii)
        XPM1(ii)=XI*(A1(ii)-A1dagg(ii))

        XTHET1(ii)=XPM1(ii)*sin(theta)+XAM1(ii)*cos(theta)


enddo

! Homodyne PSD
        SHOM1=0.d0
       SHOM1=SHOM1+ABS(XTHET1(1))**2*AVPHOT+ABS(XTHET1(2))**2*(AVPHOT+1.d0)

! Xvec contains vectors of :
 ! a_in,a_in^\dagg, b_in,x1,b^\dagg_in,x1,b_in,y1,b^\dagg_in,y1,...b_in,x2, b^\dagg_in,x2...! note phonon occupancies AVN(3)=AVN(4)=AVNX1; AVN(5)=AVN(6)=AVNY1
! clearly AVN(1)=AVN(2)=0 since that stores optical mode info.

        do ii=2,NT-2,2
  
      SHOM1=SHOM1+ABS(XTHET1(ii+1))**2*(AVN(ii+1)+1)+ABS(XTHET1(ii+2))**2*AVN(ii+1)
     enddo

100   format(6D16.8)
RETURN
END




!************************
! Heterodyne
!**************************************************

Subroutine Heterodyne(AVN,AVPHOT,THETA,A1,A1dagg,SHET1)
IMPLICIT NONE
integer  ::ii,m,jj,iterm,NT,npcle,Nmode
include 'Nparticle.h'
 double precision:: THETA,SHOM1,SHET1,AVPHOT
double complex::XI

double precision::AVN(NT)
double complex::XTHET1(NT)
 double complex:: XPM1(NT),XAM1(NT)
double complex:: A1(NT)
double complex:: A1dagg(NT)

XTHET1=(0.d0,0.d0)

! i
XI=CMPLX(0.d0,1.d0)
do ii=1,NT

XTHET1(ii)=A1dagg(ii)
!XTHET1(ii)=A1(ii)
enddo
SHET1=0.d0


! Heterodyne PSD
        SHET1=0.d0
       SHET1=SHET1+ABS(XTHET1(1))**2*AVPHOT+ABS(XTHET1(2))**2*(AVPHOT+1.d0)

! Xvec contains vectors of :
 ! a_in,a_in^\dagg, b_in,x1,b^\dagg_in,x1,b_in,y1,b^\dagg_in,y1,...b_in,x2, b^\dagg_in,x2...! note phonon occupancies AVN(3)=AVN(4)=AVNX1; AVN(5)=AVN(6)=AVNY1
! clearly AVN(1)=AVN(2)=0 since that stores optical mode info.

        do ii=2,NT-2,2
  
      SHET1=SHET1+ABS(XTHET1(ii+1))**2*(AVN(ii+1)+1)+ABS(XTHET1(ii+2))**2*AVN(ii+1)
     enddo

100   format(6D16.8)
RETURN
END
!*************************************************
 ! Work out homodyne spectra
!**************************************************
      Subroutine Homthet(AVN,AVPHOT,THETA,A1,A1dagg,SHOM)
        IMPLICIT NONE
     integer  ::ii,m,jj,iterm,NT,NTOT,npcle,Nmode
     include 'Nparticle.h'
         double precision:: THETA,SHOM,AVPHOT
       double complex::XI,Cthet2p,Cthet2m,Scross

       double precision::AVN(NT)
      double complex:: A1(NT)
      double complex:: A1dagg(NT)

      NTOT=NT
!  zero arrays

           Cthet2p=CMPLX(cos(2*theta),sin(2*theta))
           Cthet2m=CMPLX(cos(2*theta),-sin(2*theta))
! i
        XI=CMPLX(0.d0,1.d0)


! Homodyne PSD square terms
        SHOM=0.d0
!--------------------------------------------
       
       SHOM=SHOM+ABS(A1(1))**2*AVPHOT+ABS(A1(2))**2*(AVPHOT+1.d0)

! Xvec contains vectors of :
 ! a_in,a_in^\dagg, b_in,x1,b^\dagg_in,x1,b_in,y1,b^\dagg_in,y1,...b_in,x2, b^\dagg_in,x2...! note phonon occupancies AVN(3)=AVN(4)=AVNX1; AVN(5)=AVN(6)=AVNY1
! clearly AVN(1)=AVN(2)=0 since that stores optical mode info.

        do ii=2,NT-2,2
  
      SHOM=SHOM+ABS(A1(ii+1))**2*(AVN(ii+1)+1)+ABS(A1(ii+2))**2*AVN(ii+1)
     enddo
!----------

      SHOM=SHOM+ABS(A1dagg(1))**2*AVPHOT+ABS(A1dagg(2))**2*(AVPHOT+1.d0)

! Xvec contains vectors of :
! a_in,a_in^\dagg, b_in,x1,b^\dagg_in,x1,b_in,y1,b^\dagg_in,y1,...b_in,x2, b^\dagg_in,x2...! note phonon occupancies AVN(3)=AVN(4)=AVNX1; AVN(5)=AVN(6)=AVNY1
! clearly AVN(1)=AVN(2)=0 since that stores optical mode info.

 do ii=2,NT-2,2

SHOM=SHOM+ABS(A1dagg(ii+1))**2*(AVN(ii+1)+1)+ABS(A1dagg(ii+2))**2*AVN(ii+1)
enddo
! Homodyne cross terms

        Scross=0.d0
!--------------------------------------------
       
       Scross=Scross+A1(1)*Conjg(A1dagg(1))*AVPHOT+A1(2)*conjg(A1dagg(2))*(AVPHOT+1.d0)

! Xvec contains vectors of :
 ! a_in,a_in^\dagg, b_in,x1,b^\dagg_in,x1,b_in,y1,b^\dagg_in,y1,...b_in,x2, b^\dagg_in,x2...! note phonon occupancies AVN(3)=AVN(4)=AVNX1; AVN(5)=AVN(6)=AVNY1
! clearly AVN(1)=AVN(2)=0 since that stores optical mode info.

        do ii=2,NT-2,2
  
      Scross=Scross+A1(ii+1)*conjg(A1dagg(ii+1))*(AVN(ii+1)+1)+A1(ii+2)*conjg(A1dagg(ii+2))*AVN(ii+1)
     enddo
!----------
!       SHOM=Scross*Cthet2p+Conjg(Scross*Cthet2p)

      SHOM=SHOM+Scross*Cthet2p+Conjg(Scross*Cthet2p)
!----------------------------------------------------------------
100   format(6D16.8)
RETURN
END
