!This code has 1 beam plus 3 dimensions
!**************************************
! Calculates quantum noise spectra.
! USES real mean fields
! works out position and momentum optical quadrature
!&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

! GammaM=mechanical damping due to gas, also noise from fluctuation dissipation.
!
!*********************************************
! NB no intent(in) or intent(out) used- add these later!!
!*********************************************************
!*********************************************
     IMPLICIT NONE
     integer  ::ll,kk,mm,it,im,ii,jj,ithet,Nthet,Nx0,ix0,Ndet,Ndx0
        integer  ::ipcle,imode,Npcle,Nmode,NT
        integer  ::NTOT,NPERIOD,NPTS
!************************************************************************
     double precision::R0,RHO,EPSR,EPSI0,C,hbar,BOLTZ,TEMP,Gravity
     double precision::WK,waist,XL,Finesse,Press,WX,WY,DelFSR
      double precision::PIN1,DET1,X0,Y0,Z0,Theta0,Thetahom,Delx0,XX0,Deldet,DET2PIini,test
      double precision::Delthet,cavphotn
! parameter file with input values
     
        include 'Nparticle.h'
        include 'CSCAVITY.h'
 
     Parameter(Nthet=1,Delx0=0.01d0*1.064d-6,Ndx0=1,Ndet=1,Deldet=150.d3,Thetahom=0.125d0,NPTS=10000)
        double precision::OMx,OMy,OMz
!calculated  parameters
!
double precision::Det2pi,kappnano,kappin,kapp2,kappa,Polaris,thet,THEhom
double precision::XM,GAMMAM,Wkx0,epsTW,epsCAV,ZR,Ed,cbath
       double precision:: XMAX,DEL

        double precision::pi,pi2,OMsweep,SHOM1,AHOM1,SHET1
       double precision::XX1QM,YY1QM,ZZ1QM,XX2QM,YY2QM,ZZ2QM,XX3QM,YY3QM
double precision::XX1QMs,YY1QMs,ZZ1QMs,XX2QMs,YY2QMs,ZZ2QMs,XX3QMs,YY3QMs
double precision:: TPERIOD,AVNX,AVNY,AVNZ,AVPHOT,AVRE,OPTOCOOL,PHONONS
       double precision::COOLX,COOLY,COOLZ,TEMPX,TEMPY,TEMPZ,TBATH,CrossSpec
 
 !----------------------------------------------------- 
        Double precision :: AVN(NT)  
         double precision:: Phases(Npcle)
      double precision:: OMEGS(Npcle,Nmode),GammaMod(Npcle,Nmode),PSD(Npcle,Nmode)
!   coupling to light
         double precision::GMAT(npcle,Nmode)
! direct coupling
         double precision:: GMM(npcle,Nmode,Nmode)
!----------------------------------------------------------
! read parameter file with input values
       DOUBLE PRECISION::G1(6)
      DOUBLE PRECISION::PHON(3),AV(3)
        DOUBLE PRECISION, DIMENSION(NPTS):: SXX1QM,SYY1QM,SXX2QM,SYY2QM,SXX3QM,SYY3QM
      DOUBLE PRECISION, DIMENSION(NPTS)::OMSTOR,Shom
         DOUBLE COMPLEX::XI,ZOM

!
!&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
!***************************************************************
! Here plot optical field amplitudes
     OPEN(8,file="FTanXYZ.dat",status="unknown")
    OPEN(9,file="FTSXY.dat",status="unknown")
     OPEN(10,file="FTanOPT.dat",status="unknown")
    open(12,file="OMEGAS.dat",status="unknown")
   open(13,file="Omegas1.dat",status="unknown")
   open(14,file="Omegas2.dat",status="unknown")
!
   pi=dacos(-1.d0)
   pi2=2.d0*pi
   DET2PIini=DET1*pi2
    TBATH=TEMP
   

!  OPEN LOOP OVER DEtun&&&&&&&&&&&&&&&&&&&&&&&
!     do  10 ithet=1,Nthet
    do  10 ix0=1,Ndx0
     DET2PI=DET2PIini
!     DET2PI=DET2PIini+Deldet*pi*2*(ix0-1)
      write(6,*)'detuning',Det2pi/2/pi
!     thet=theta0+Delthet*(ithet-1)
      thet=theta0
      xx0=Wk*(x0-(ix0-1)*Delx0)
 !      xx0=Wk*x0
      Wkx0=xx0
     thet=thet*pi
      THEhom=Thetahom*pi
      write(6,*)'theta/pi', thet/pi
!******************************************
! WORK OUT  particle 1 parameters
!------------------------------------------------------
! note BEAD routine uses x0 but not theta
  call BEAD_parameters(Wkx0,Polaris,epsTW,epsCAV,ZR,XM,kappin,kappnano,GAMMAM)
  call EQUIL_PAR(thet,Det2pi,Wkx0,Polaris,epsTW,epsCAV,XM,ZR,kappnano,kappin,GammaM,OMX,OMY,OMZ,G1,PHON)
   
!*****************************************
   cbath=BOLTZ*TBATH/hbar
 ! AVNX1=BOLTZ*TBATH/hbar/OMX so cbath is for AVN

   call Parameters(OMEGS,GMAT,GMM,GammaMod,AVN,Phases,G1,GammaM,OMX,OMY,OMZ,cbath)
   write(6,*) ' OMx1              OMy1            OMx2             OMY2    '
!   write(6,120)OMEGS(1,1)/pi2,OMEGS(1,2)/pi2
 write(6,120)OMEGS(1,1)/pi2,OMEGS(1,2)/pi2,OMEGS(2,1)/pi2,OMEGS(2,2)/pi2

!    write(12,120)thet,(abs(Gmat(ii)),ii=1,6)

120 Format(7E14.6)
!***************************************************
! cavity occupancy particle 1
!****************************************************
      Ed=0.5d0*Polaris*epsTW*epsCAV*sin(thet)
!     Ed=Ed/hbar
      cavphotn=Ed*Ed*cos(Wkx0)*cos(Wkx0)/hbar/hbar
      kappa=(kappnano+kappin)
      kapp2=kappa*0.5d0
      write(6,*)'kappa in main',kappa/2/pi,kapp2/2/pi
      cavphotn=Cavphotn/(kapp2**2+Det2pi**2)
      write(6,*)'photons in cavity', cavphotn
!****************************************************
! shot noise
      AVPHOT=0.d0
!***************************************************************
!&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
! open loop over frequency for noise spectra eg S_xx(\omega)=FT(autocorrelation <X(t)X^T(t+tau)>)
            XMAX=2*OMX*1.001
            XMAX=7.d5*pi2
            DEL=XMAX/NPTS
             SXX1QM=0.d0
             SYY1QM=0.d0
            SXX2QM=0.d0
             SYY2QM=0.d0
           SXX3QM=0.d0
            SYY3QM=0.d0

            

             SHOM=0.d0
             do 11 mm=1,NPTS
!           do 11 mm=1,1
!             OMsweep=122000.0*pi2
         OMsweep=-XMAX+2*(mm-1)*DEL

! store frequency for integration
             OMSTOR(mm)=OMsweep
! work out PSD of homodyne
        AHOM1=0.d0
! symmetrised 
       XX1QMs=0.d0
       YY1QMs=0.d0
       ZZ1QMs=0.d0
  XX2QMs=0.d0
  YY2QMs=0.d0
  ZZ2QMs=0.d0
    

! ! work out  for negative frequency for symmetrisation
   CALL ANALYT(-OMsweep,THehom,Det2pi,Kapp2,AVphot,AVN,Gmat,GMM,Phases,GAMMAMod,OMEGS,PSD,CrossSpec,SHOM1,SHET1)
    
 
    XX1QM=PSD(1,1)
    YY1QM=PSD(1,2)
   XX2QM=PSD(2,1)
   YY2QM=PSD(2,2)
   XX3QM=PSD(3,1)
   YY3QM=PSD(3,2)
!   ZZQM=PSD(1,3)


! update homodyne and symm disp.
      AHOM1=AHOM1+SHOM1

! work out same for positive and symmetrise homodyne
  CALL ANALYT(OMsweep,THehom,Det2pi,Kapp2,AVphot,AVN,Gmat,GMM,Phases,GAMMAMod,OMEGS,PSD,CrossSpec,SHOM1,SHET1)

    XX1QM=PSD(1,1)
    YY1QM=PSD(1,2)
   XX2QM=PSD(2,1)
   YY2QM=PSD(2,2)
    XX3QM=PSD(3,1)
   YY3QM=PSD(3,2)

!   ZZQM=PSD(1,3)

  

! update
  
       AHOM1=0.5*(AHOM1+SHOM1)


      XX1QMs=XX1QMs+XX1QM
      YY1QMs=YY1QMs+YY1QM
      XX2QMs=XX2QMs+XX2QM
      YY2QMs=YY2QMs+YY2QM
   XX3QMs=XX3QMs+XX3QM
   YY3QMs=YY3QMs+YY3QM

!       ZZQMs=ZZQMs+ZZQM


omsweep=omsweep/2/pi/1.d3
   if((omsweep.ge.-400.d0).and.(omsweep.LE.400))then
    write(8,101)OMsweep,XX1QM,YY1QM,XX2QM,YY2QM,XX3QM,YY3QM
! write(8,101)OMsweep,XXQMs*0.1+Wkx0,YYQMs*0.1+Wkx0,ZZQMs,XXQMs,YYQMs,ZZQM
! write(8,101)OMsweep/2/pi,XXQMs*1d5+Det2pi/pi2*1.d-3,YYQMs*1d5+Det2pi/pi2*1d-3,ZZQMs,XXQMs,YYQMs,ZZQM
! single value of Sxy

    write(9,101)OMsweep,CrossSpec
! if loop over Sxy
!  write(9,101)OMsweep,XYQMs*0.1+WKx0/pi2,YXQMs+Wkx0/pi2
 write(10,101)OMsweep,Shet1,Ahom1
 endif
101   format(8D13.5)
             SXX1QM(mm)=XX1QM
             SYY1QM(mm)=YY1QM
       SXX2QM(mm)=XX2QM
       SYY2QM(mm)=YY2QM
       SXX3QM(mm)=XX3QM
       SYY3QM(mm)=YY3QM

!SZZQM(mm)=AHOM1
! to find optimal squeezing
              SHOM(mm)=AHOM1
11            enddo

AVRE=(omx+Det2pi)**2+kapp2**2
AVRE=AVRE/4/omx/(-det2pi)
    
write(6,*)'X: back action limited phonons',AVRE
  AVRE=(omy+Det2pi)**2+kapp2**2
  AVRE=AVRE/4/omY/(-det2pi)
  write(6,*)'Y: back action limited phonons',AVRE
   AVRE=(omz+Det2pi)**2+kapp2**2
  AVRE=AVRE/4/omz/(-det2pi)
   write(6,*)'Z: back action limited phonons',AVRE
! integrate and normalise the quantum  noise spectra.

      CALL NORM1(NPTS,OMX,TBATH,GAMMAM,TEMPX,SXX1QM,OMSTOR,AVRE)
! Area AVRE corresponds to 2n+1 so convert to get n
                PHONONS=PHON(1)
     AVRE=0.5d0*(AVRE-1.d0)
     AV(1)=AVRE
      write(6,*)'X1 phonons from formula,  from SXX1 FT'
      write(6,200)PHONONS,AVRE
       CALL NORM1(NPTS,OMY,TBATH,GAMMAM,TEMPY,SYY1QM,OMSTOR,AVRE)
! Area AVRE corresponds to 2n+1 so convert to get n
        PHONONS=PHON(2)
        AVRE=0.5d0*(AVRE-1.d0)
        AV(2)=AVRE
write(6,*)'Y1 phonons from formula,  from SYY1 FT'
        write(6,200)PHONONS,AVRE

CALL NORM1(NPTS,OMX,TBATH,GAMMAM,TEMPX,SXX2QM,OMSTOR,AVRE)
! Area AVRE corresponds to 2n+1 so convert to get n
          PHONONS=PHON(1)
AVRE=0.5d0*(AVRE-1.d0)
AV(1)=AVRE
write(6,*)'X2 phonons from formula,  from SXX2 FT'
write(6,200)PHONONS,AVRE

CALL NORM1(NPTS,OMX,TBATH,GAMMAM,TEMPX,SYY2QM,OMSTOR,AVRE)
! Area AVRE corresponds to 2n+1 so convert to get n
          PHONONS=PHON(1)
AVRE=0.5d0*(AVRE-1.d0)
AV(1)=AVRE
write(6,*)'Y2 phonons from formula,  from SYY2 FT'
write(6,200)PHONONS,AVRE
     write(6,200)PHONONS,AVRE

CALL NORM1(NPTS,OMX,TBATH,GAMMAM,TEMPX,SXX3QM,OMSTOR,AVRE)
! Area AVRE corresponds to 2n+1 so convert to get n
          PHONONS=PHON(1)
AVRE=0.5d0*(AVRE-1.d0)
AV(1)=AVRE
write(6,*)'X3 phonons from formula,  from SXX3 FT'
write(6,200)PHONONS,AVRE

CALL NORM1(NPTS,OMX,TBATH,GAMMAM,TEMPX,SYY3QM,OMSTOR,AVRE)
! Area AVRE corresponds to 2n+1 so convert to get n
          PHONONS=PHON(1)
AVRE=0.5d0*(AVRE-1.d0)
AV(1)=AVRE
write(6,*)'Y3 phonons from formula,  from SYY3 FT'
write(6,200)PHONONS,AVRE
     write(6,200)PHONONS,AVRE

! loop over theta
10    enddo

!***********************************************************************
100   FORMAT(I3,3E14.6,1x,2(E14.6))
200  FORMAT(7E14.6)

    STOP
    END

!********************************************************************
!  Function to evaluate optomechanical cooling formula for single particle1 
!********************************************************************

     FUNCTION optocool(G1,Det1X,KAPP2,OMEGAM,GAMMAM)
       Implicit None
      double precision::G1,Det1X,KAPP2,OMEGAM

       double precision::C1,C2,C3,C4
       double precision::OPTOCOOL,COOL1,COOL2,GAMMAM
       double precision:: hbar,BOLTZ,TBATH
       PARAMETER(BOLTZ=1.4d-23,hbar=1.05d-34,TBATH=300.0)
        COOL1=0.d0


! now work out opto cooling expression 
! Trap beam
       C1=2.*KAPP2*G1*G1
       C2=DET1X


      C3=(C2+omegam)**2+kapp2**2
       C3=1.d0/C3
       C4=(C2-omegam)**2+kapp2**2
       C4=1.d0/C4
       COOL1=-C1*(C3-C4)

        optocool=COOL1
100   format(4D16.6)
   return
    end
!*************************************************************************
! Normalise area under PSDs
!************************************************************************

       SUBROUTINE NORM1(NP,OMEGAM,TBATH,Gamm,TEMP,SXXW,OMSTOR,XRE)

!************************************************************************
!************************************************************************
                
    IMPLICIT NONE
        integer  ::ii,jj,NP
         double precision::omegam

       double precision::PI2,XRE,XIM,COOL,gamm,TEM
      double precision::TEMP2,TEMP,C1,C2,DEL
 
        double precision::TBATH,hbar,BOLTZ
       PARAMETER(hbar=1.05d-34,BOLTZ=1.4d-23)
         DOUBLE PRECISION, DIMENSION(NP):: OMSTOR
        DOUBLE PRECISION:: SXXW(NP)
     

         pi2=2.d0*dacos(-1.d0)
!  integrate the position spectrum of bead
! quick hack - use trapezoidal rule- improve later
        XRE=0.d0
        XIM=0.d0
        DEL=ABS(OMSTOR(2)-OMSTOR(1))
        do ii=1,NP-1
        Tem=0.5d0*(SXXW(ii)+SXXW(ii+1))
         XRE=XRE+TEm
         enddo
         XRE=XRE*DEL/pi2
        TEMP=XRE*hbar*OMEGAM/BOLTZ

 100       format(6E14.6)
        return
         end
!************************
!**************************************************

