
!******************************************************************
Subroutine Parameters(OMEGS,GMAT,GMM,GammaMod,AVN,Phases,G1,GammaM,OMX,OMY,OMZ,cbath)
	IMPLICIT NONE
        integer  ::ii,m,jj,ipcle,imode,Npcle,Nmode,NT
        include 'Nparticle.h'
    
        double precision:: GammaM,OMX,OMy,OMz,cbath
       
        double precision::pi,pi2
 !----------------------------------------------------- 

        Double precision :: AVN(NT)
         double precision:: Phases(Npcle)
      double precision:: OMEGS(Npcle,Nmode),GammaMod(Npcle,Nmode)
!   coupling to light
         double precision::GMAT(npcle,Nmode)
! direct coupling
         double precision:: GMM(npcle,Nmode,Nmode)
       Double precision :: G1(6)
     pi2=2*dacos(-1.d0)
     AVN=0.d0
     PHASES=0.d0
     OMEGS=0.d0
     GammaMod=0.d0
     GMAT=0.d0
     GMM=0.d0

!   Fill GammaMod
   Do ii=1,Npcle
   do jj=1,Nmode
   GammaMod(ii,jj)=GammaM
   enddo
   enddo
!--------------------------------------
!  Fill Phases
   do ii=1,Npcle
   Phases(ii)=0.d0
   enddo
!------------------------------------------------
 ! G1 =couplings for pcle 1 
 ! G1(1)=gx,G1(2)=g_y, G1(3)=g_z; G1(4)=g_xy; G1(5)=g_yz, G1(6)=g_xz
!-----------------------------------------------------------------
! Fill GMAT
! Particle 1

    GMAT(1,1)=G1(1)
    GMAT(1,2)=G1(2)

!    if(Nmode.gt.2)then
 !   GMAT(1,3)=G1(3)
 !   endif
! Particle 2

  GMAT(1,1)=-40000*pi2
  GMAT(1,2)=-40000*pi2

 GMAT(2,1)=GMAT(1,1)
 GMAT(2,2)=GMAT(1,2)

GMAT(3,1)=GMAT(1,1)
GMAT(3,2)=GMAT(1,2)

!GMAT(1,1)=0.d0
!GMAT(1,2)=0.d0

!  GMAT(2,3)=G1(3)

!Fill GMM
! Particle 1
    GMM(1,1,2)=G1(4)
    GMM(1,2,1)=G1(4)


!    if(Nmode.gt.2) then
!    GMM(1,2,3)=G1(5)
!    GMM(1,3,2)=G1(5)
!    GMM(1,1,3)=G1(6)
 !   GMM(1,3,1)=G1(6)
 !    endif
! Particle 2
   
    GMM(2,1,2)=G1(4)
    GMM(2,2,1)=G1(4)
! Particle 3
    GMM(3,1,2)=G1(4)
    GMM(3,2,1)=G1(4)

 !   if(Nmode.gt.2) then
 !   GMM(2,2,3)=G1(5)
 !   GMM(2,3,2)=G1(5)
 !   GMM(2,1,3)=G1(6)
 !   GMM(2,3,1)=G1(6)
   
!----------------------------------
!  Fill OMEGS
! Particle 1
    OMEGS(1,1)=233000*pi2
    OMEGS(1,2)=229000*pi2
 !   OMEGS(1,3)=Omz
    
! Particle 2
    OMEGS(2,1)=225000*pi2
    OMEGS(2,2)=221000*pi2
!   OMEGS(2,3)=Omz*1.03
! Particle 2
    OMEGS(3,1)=217000*pi2
    OMEGS(3,2)=213000*pi2

!-----------------------------------------
! Fill AVN, phonon occupancies for b_u and b_u^\dagg, u=x1,y1...
! AVN(3)=x1 AVN(4)=x1 AVN(5)=y1 AVN(6)=y1 AVN(7)=z1 AVN(8)=z1
! AVN(9)=x2 AVN(10)=x2 AVN(11)=y2 AVN(12)=y2 AVN(13)=z2 AVN(14)=z2
!cbath=BOLTZ*TBATH/hbar
! Particle 1
    AVN(3)=cbath/OMEGS(1,1)
    AVN(4)=cbath/OMEGS(1,1)
    AVN(5)=cbath/OMEGS(1,2)
    AVN(6)=cbath/OMEGS(1,2)

 !   AVN(7)=cbath/OMEGS(1,3)
 !   AVN(8)=cbath/OMEGS(1,3)
 

   AVN(7)=cbath/OMEGS(2,1)
   AVN(8)=cbath/OMEGS(2,1)
   AVN(9)=cbath/OMEGS(2,2)
  AVN(10)=cbath/OMEGS(2,2)


  AVN(11)=cbath/OMEGS(3,1)
  AVN(12)=cbath/OMEGS(3,1)
  AVN(13)=cbath/OMEGS(3,2)
  AVN(14)=cbath/OMEGS(3,2)
  
    Return
    End

!********************************************************************
