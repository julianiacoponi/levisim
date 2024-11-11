!*****************************************************************
! Work our vectors of X and optical fields but in terms of noise operators
!*******************************************************************

    Subroutine Noisevect(kapp2,GMAT,GMM,GammaMod,Phase,&
      ChiOpt,ChiOptMinus,Chi,ChiMinus,A1,A1dagg,Xvec)

       IMPLICIT NONE

       integer  ::ii,m,jj,ip1,ip2,ipcle,imod,imode,Npcle,Nmode,NT
       integer  ::ipcle1,ipcle2,imod1,imod2,Nmech,icount1,icount2
        double precision:: ANORM,BNORM,DIF,ph,coef
        double precision:: t1,t2,Sqrtkapp,Sqrtgamm,Gamm,Gamm2,kapp2
       
     

      include 'Nparticle.h'

!**********************************
double precision:: Phase(Npcle)
!   coupling to light
double precision::GMAT(npcle,Nmode)

! direct coupling should be GMM(npcle,Nmode,Nmode)
double precision:: GMM(npcle,Nmode,Nmode)

double precision:: OMEGS(Npcle,Nmode),GAMMAMod(Npcle,Nmode)
Double precision:: PSD(Npcle,Nmode),CrossSpec(Npcle,Nmode)
Double precision :: AVN(NT)
! mechanical susceptibilities
   double complex::ChiOpt, ChiOptMinus
    double complex::Chi(Npcle,Nmode),ChiMinus(Npcle,Nmode)
   double complex::BetMod(Npcle,Nmode),CMmod(Npcle,Nmode),EtaMod(Npcle,Nmode)
     double complex:: Xvec(Npcle,Nmode,NT),N0Mode(Npcle,Nmode,NT)
    double complex::Gcoef(Npcle,Nmode,Npcle,Nmode)
   double complex::R(Npcle,Nmode,Npcle,Nmode)
   double complex::SMAT(Npcle*Nmode,Npcle*Nmode),SMinus(Npcle*Nmode,Npcle*Nmode),Atest(Npcle*Nmode,Npcle*Nmode)
!************************************************

        double complex:: XI,ONE
       double complex::RXY,RYX,RXZ,RZX,RYZ,RZY
       double complex:: CSUM,CNORM,eta0c,etaMpi2c,etaPpi2c,etaMPhi
       double complex :: CPHASE,CPHASEM,CA1,CA1dagg

      double complex, intent(out):: A1(NT)
      double complex, intent(out):: A1dagg(NT)
    
    
       Gamm=GammaMod(1,1)
!&&&&&&&&& zero arrays  &&&&&&&&&&&&&&&&&&
       Gcoef=0.d0
       etaMod=0.d0
       CMmod=0.d0
       BETmod=0.d0
       R=0.d0
       SMAT=0.d0
       SMinus=0.d0
       Xvec=0.d0
       A1=CMPLX(0.d0,0.d0)
      A1dagg=CMPLX(0.d0,0.d0)
!&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
! actually Gamm is gamma/2
       Gamm2=Gamm
! zero arrays
        XVEC=0.d0
      
        N0Mode=CMPLX(0.d0,0.d0)
        R=0.d0

        
! i
        XI=COMPLEX(0.d0,1.d0)
        ONE=COMPLEX(1.d0,0.0d0)
       Sqrtkapp=sqrt(2.d0*KAPP2)
       Sqrtgamm=sqrt(2.d0*Gamm2)
!      SUSCEPTIBILITIES
        eta0c=ChiOpt-ChiOptMinus

        etaMpi2c=XI*(ChiOpt+ChiOptMinus)
        etaPpi2c=-XI*(ChiOpt+ChiOptMinus)
       
      do ipcle=1,Npcle
      do imod=1,Nmode
      etaMod(ipcle,imod)=Chi(ipcle,imod)-ChiMinus(ipcle,imod)
     CMmod(ipcle,imod)=1.d0+GMAT(ipcle,imod)**2*etaMod(ipcle,imod)*eta0c
     BETmod(ipcle,imod)=XI*Sqrtkapp*etaMod(ipcle,imod)*GMAT(ipcle,imod)
     enddo
     enddo

! cavity mediated coupling- Combines direct and indirect coupling
do 10 ipcle1=1,Npcle
do 11 ipcle2=1,Npcle
 if(ipcle1==ipcle2)then
! intraparticle x-y coupling  X1-Y1 and X2-Y2
    Gcoef(ipcle1,1,ipcle1,2)=GMM(ipcle1,1,2)+Xi*eta0c*GMAT(ipcle1,1)*GMAT(ipcle1,2)
! intraparticle y-x coupling  Y1-X1 and Y2-X2
    Gcoef(ipcle1,2,ipcle1,1)=Gcoef(ipcle1,1,ipcle1,2)

              

   elseif(ipcle1.NE.ipcle2)then

! ipcle1 .neq. ipcle2   :  COUPLING BETWEEN TWO PARTICLES
!&&&&&&&&&&&&&&&&&& INCLUDE PHASES&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  !        DIF=0.d0
          DIF=PHASE(ipcle2)-PHASE(ipcle1)
         CPHASE=COMPLEX(COS(DIF),SIN(DIF))
         etaMphi=ChiOpt*CPHASE-ChiOptMinus*CONJG(CPHASE)

!bleeeuw  above.

!    X(ipcle1),X(ipcle2) coupling
    Gcoef(ipcle1,1,ipcle2,1)= Xi*etaMphi*GMat(ipcle1,1)*GMat(ipcle2,1)
!    Y(ipcle1),Y(ipcle2) coupling
    Gcoef(ipcle1,2,ipcle2,2)= Xi*etaMphi*GMat(ipcle1,2)*GMat(ipcle2,2)

!    X(ipcle1),Y(ipcle2) coupling
    Gcoef(ipcle1,1,ipcle2,2)= Xi*etaMphi*GMat(ipcle1,1)*GMat(ipcle2,2)
!    Y(ipcle1),X(ipcle2) coupling
    Gcoef(ipcle1,2,ipcle2,1)= Xi*etaMphi*GMat(ipcle1,2)*GMat(ipcle2,1)
  endif
         
11    enddo
10    enddo

!&&&&&&&&&&&&&&&&&&&&&&&&
! zero-th order X noise vector; weights of a1,a1*,bx,bx*,by,by*,bz,bz*

          ip1=1

!------------------------------------------------------------
    do ipcle=1,Npcle
    do imode=1,Nmode
       N0Mode(ipcle,imode,1)=BETMod(ipcle,imode)*ChiOpt/CMmod(ipcle,imode)
       N0Mode(ipcle,imode,2)=BETMod(ipcle,imode)*ChiOptMinus/CMmod(ipcle,imode)
    enddo
    enddo
      icount1=2
  do ipcle=1,Npcle
  do imode=1,Nmode
       
       N0Mode(ipcle,imode,icount1+1)=Sqrtgamm*Chi(ipcle,imode)/CMmod(ipcle,imode)
       N0Mode(ipcle,imode,icount1+2)=Sqrtgamm*ChiMinus(ipcle,imode)/CMmod(ipcle,imode)
       icount1=icount1+2
   enddo
   enddo


!----------------------------------------------------------------------
! ONE PARTICLE SOLUTION
  icount1=0
 do 30 ip1=1,Npcle
 do 31 imod1=1,Nmode
         icount1=icount1+1
          icount2=0
          do  ip2=1,Npcle
          do  imod2=1,Nmode
           icount2=icount2+1
            R(ip1,imod1,ip2,imod2)=xi*Gcoef(ip1,imod1,ip2,imod2)*etaMod(ip1,imod1)/CMmod(ip1,imod1)

          coef=0.d0
          if(icount1==icount2)coef=1.d0
          SMAT(icount1,icount2)=coef-R(ip1,imod1,ip2,imod2)
! for explicit use below
 !        SMAT(icount1,icount2)=R(ip1,imod1,ip2,imod2)
          enddo
          enddo
31 enddo
30  enddo

       Nmech=Npcle*Nmode

!---------LAPACK INVERTER---------------------------------------
       Call  Sinvert(Nmech,SMAT,Sminus)
!       Call Sinvert_2D2P(Nmech,SMAT,Sminus)
!   Atest=Matmul(SMAT,Sminus)
!   write(6,*)Atest
! Re-pack Sminus into 4d
     R=0.d0
    icount1=0

 do ip1=1,Npcle
  do imod1=1,Nmode
    icount1=icount1+1
     icount2=0
    do ip2=1,Npcle
      do imod2=1,Nmode
        icount2=icount2+1
         R(ip1,imod1,ip2,imod2)=Sminus(icount1,icount2)
      enddo
    enddo
  enddo
 enddo

!---------------------------------------------------------------
 do ii=1,NT
   do ip1=1,Npcle
     do imod1=1,Nmode
       do ip2=1,Npcle
          do imod2=1,Nmode
           XVEC(ip1,imod1,ii)=XVEC(ip1,imod1,ii)+ R(ip1,imod1,ip2,imod2)*N0Mode(ip2,imod2,ii)
          enddo
          enddo
      enddo
    enddo
  enddo
!-------------------------------------------------------------------
! now work out the optical trap field =a1
         CA1=XI*ChiOpt
! now work out the photon field a1dagger
         CA1dagg=-XI*ChiOptMinus
!!
 
         do 111 ipcle=1,Npcle
! phase terms
    ph=PHASE(ipcle)
    CPHASE=COMPLEX(COS(ph),SIN(ph))
  
         do 112 imode=1,Nmode
 !       do 112 imode=1,1
   do  ii=1,NT
      A1(ii)=A1(ii)+CA1*GMat(ipcle,imode)*XVEC(ipcle,imode,ii)*Cphase
       A1dagg(ii)=A1dagg(ii)+CA1dagg*GMat(ipcle,imode)*XVEC(ipcle,imode,ii)*conjg(CPHASE)


enddo
112        enddo
111        enddo
! add shot or incoming noise
! trap beam: add cavity-filtered contribution
        A1(1)=A1(1)+Sqrtkapp*ChiOpt
       A1dagg(2)=A1dagg(2)+Sqrtkapp*ChiOptMinus

! cavity output : add incoming imprecision
! work out a_out=a_in-Sqrtkapp(a)
      do ii=1,NT
      A1(ii)=-A1(ii)*Sqrtkapp
      A1dagg(ii)=-A1dagg(ii)*Sqrtkapp
      enddo
      A1(1)=ONE+A1(1)
     A1dagg(2)=ONE+A1dagg(2)


!&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


    return
    end
!
!*****************************************************
  SUBROUTINE  Sinvert(N,A,Aminus)
   IMPLICIT NONE
   integer  ::N,LDA
   ! Lapack sizes
   integer:: INFO,LWORK

   Integer::IPIV(N)
   DOUBLE COMPLEX ::scale

   DOUBLE COMPLEX :: A(N,N),Aminus(N,N)

   DOUBLE COMPLEX :: WORK(N*N)
     Aminus=0.d0
    
     LWORK=N*N

   call ZGETRF(N,N,A,N,IPIV,info)
 
     call  zgetri(N,A,N,IPIV,WORK,LWORK,INFO)
      If(info.NE.0) write(6,*)'info',info
     Aminus=A
     Return
     End
!****************************************************
