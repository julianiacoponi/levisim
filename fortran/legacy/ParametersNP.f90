
!******************************************************************
Subroutine Parameters(OMEGS,GMAT,GMM,GammaMod,AVN,Phases,G1,GammaM,OMX,OMY,OMZ,cbath)
	IMPLICIT NONE
        integer  ::ii,m,jj,ipcle,imode,Npcle,Nmode,NT,icount
        include 'Nparticle.h'
    
        double precision:: GammaM,cbath,Omx,omy,Omz,GG
       
        double precision::pi,pi2,DelOmega,zero
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
     zero=0.d0
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
! Fill GMAT: here all the same

    GG=-40000.d0
    do ipcle=1,Npcle
    do imode=1,Nmode
    Gmat(ipcle,imode)=GG*pi2
enddo
enddo

!----------------------------------
!  Fill OMEGS
 !   OMEGS(1,1)=350000*pi2
   OMEGS(1,1)=320000*pi2
    DelOmega=3.d3*pi2
    icount=0
    do ipcle=1,Npcle
     do imode=1,Nmode
    
    Omegs(ipcle,imode)=omegs(1,1)-icount*DelOmega
    write(14,*)Omegs(ipcle,imode)/pi2*1.d-3,zero+0.25
    icount=icount+1
     enddo
     enddo
    Omegs(3,1)=5000000*pi2
    Omegs(6,2)=5000000*pi2

!-----------------------------------------
! Fill AVN, phonon occupancies for b_u and b_u^\dagg, u=x1,y1...
! AVN(3)=x1 AVN(4)=x1 AVN(5)=y1 AVN(6)=y1 AVN(7)=z1 AVN(8)=z1
! AVN(9)=x2 AVN(10)=x2 AVN(11)=y2 AVN(12)=y2 AVN(13)=z2 AVN(14)=z2
!cbath=BOLTZ*TBATH/hbar
 !   NT=2*Npcle*Nmode+2
    icount=3
    do ipcle=1,Npcle
     do imode=1,Nmode
    AVN(icount)=cbath/OMEGS(ipcle,imode)
    AVN(icount+1)=cbath/OMEGS(ipcle,imode)
    icount=icount+2
   enddo
   enddo


  
    Return
    End

!********************************************************************
