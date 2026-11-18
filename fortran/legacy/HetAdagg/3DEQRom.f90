

!*************************************************************************
!*************************************************************************
! subroutine below is provided by user
! and calculates  relevant parameters
  SUBROUTINE BEAD_parameters(WKx0,Polaris,epsTW,epsCAV,ZR,XM,kappin,kappnano,GAMMAM)
! outputs
!************************************************************************
!************************************************************************
                
	IMPLICIT NONE 
        integer  ::ii,m,jj

     integer  ::NPERIOD,NTOT
    double precision::R0,RHO,EPSR,EPSI0,C,hbar,BOLTZ,TEMP,Gravity
    double precision::WK,waist,XL,Finesse,Press,WX,WY,DelFSR
    double precision::PIN1,DET1,X0,Y0,Z0,Theta0

       include 'CSCAVITY.h'
        double precision:: epsTW,epsCAV,ZR,A,WKx0
         double precision::W2,W2M,OMOPT,coeff
        double precision:: KAPPA,XM,kappnano,kappin
        double precision::pi,pi2
        double precision::Polaris,VOL
        double precision:: Gammam
   
! zero eq. initial values
       PI=dacos(-1.d0)
       XM=RHO*4.*pi/3.*R0**3

       Polaris=4.*pi*EPSI0*(EPSR-1.)/(EPSR+2.)*R0**3
 

       OMOPT=C*WK
! note waist is waist radius
      W2=waist**2

! add a factor of 4 here. Not in GALAX1-5 routines!!!
      VOL=XL*Pi*W2/4.d0

! &&&&&&&&&&&&&
! Depth of cavity field. Weak and unimportant for CS case

      A=OMOPT*POLARIS/2./VOL/EPSI0
  write(6,*)'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'

     write(6,*)'depth of cavity field A', A/2/pi
!&&&&&&&&&&&&&&&&&&
!&&&&&&&&&&&&&&&&&&
      KAPPin=pi*c/finesse/XL
 
!
      epsTW=4.*PIN1/(Wx*Wy*pi*c*EPSI0)
      epsTW=sqrt(epsTW)
      epsCAV=hbar*OMOPT/(2.*VOL*Epsi0)
       epsCAV=sqrt(epsCAV)
    

      coeff=WK*Polaris/EPSI0/OMOPT**2/pi
      kappnano=4.*coeff**2*DelFSR*cos(WK*x0)*cos(WK*x0)

     kappa=kappnano+kappin
!&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
      ZR=WX*WY*WK/2.d0
    
 
!
! take usual expression eg Levitated review by Li Geraci etc
! 1 bar= 10^ 5 pascal; Press is in mbar = 10^ 2 pascal
! gamma=16 P/(pi*v*rho*R)
! v=speed of air=500 /s
      GAMMAM=1600.*press/pi
      GAMMAM=GAMMAM/500/RHO/R0
! Fix of Feb.2016 our GAMMAM => GAMMAM/2!!
!gamma/2=8 P/(pi*v*rho*R)
    GAMMAM=GAMMAM/2.d0
  

100   format(4D16.8)

      RETURN
       END

!*************************************************************************
!*************************************************************************
! subroutine below obtains the optomechanical parameters
 SUBROUTINE EQUIL_PAR(thet,Det,Wkx0,Polaris,epsTW,epsCAV,XM,ZR,kappnano,kappin,GammaM,OMX,OMY,OMZ,GMAT,PHON)
            
!************************************************************************
!************************************************************************
     IMPLICIT NONE
     integer  ::ii,m,jj,NT,iwrite

    integer  ::NPERIOD,NTOT
   double precision::R0,RHO,EPSR,EPSI0,C,hbar,BOLTZ,TEMP,Gravity
   double precision::WK,waist,XL,Finesse,Press,WX,WY,DelFSR,Det
   double precision::PIN1,DET1,X0,Y0,Z0,Theta0
    include 'CSCAVITY.h'

double precision:: GX,GY,Gz,GXY,GZX,GYZ,XZPF,YZPF,ZZPF,thet
     double precision::OMx,OMy,OMz, Edip,Ediph,ZR,Nphoton,ALPre,ALPim
    double precision::COOLx,COOLy,COOLz,epsCAV,epsTW,Polaris


     double precision:: DET2pi,GammaM
     double precision:: KAPPA,kapp2,kappnano,kappin,A,XM
     double precision::pi,pi2
     double precision:: WKX0,Wkxx0,Wky0,Wkz0
     double precision::C1,C2,C3,C4
! analytic equil phonon numbers
     DOUBLE PRECISION :: PHON(3)
     DOUBLE PRECISION :: GMAT(6)

     iwrite=6
     PI=dacos(-1.d0)
 !    Det2pi=Det1*2*pi
       Det2pi=Det
! kappanano is from photon scattering losses from nanoparticle, in addition to usual losses
! through mirrors. Not always added in
! in practice kappa of Antonio already includes those. But care when working out input-outpu
! as that \sqrt(kappa) is about mirror allowing leaking out of light.
      kappa=kappnano+kappin
      kapp2=0.5d0*kappa
   
!     note that detunings include zeroth order correction for linearised versions
!   as a first pass work out frequencies with Wk0=0
!      WkX0=WK*X0
!-------------------------------
!FFFFFFFFFFFFFFFFFFFF
      OmX=Polaris*epsTw**2/XM/WX**2
      OmY=Polaris*epsTw**2/XM/Wy**2
      OMz=0.5d0*Polaris*epsTw**2/XM/ZR**2

!  we will use negative Edip
   Edip=-0.5d0*Polaris*epsTW*epsCAV*sin(thet)
   Ediph=Edip/hbar


write(6,*)'EDIPH'
write(6,100)EDIPH
! photon number in cavity
! real part of photon field
!FFFFFFFFFFFFFFFFFFFFFFF
   ALPRe=Det2pi*Ediph*cos(Wkx0)/(kapp2**2+Det2pi**2)
   write(6,*)'ALPRE=',ALPRE
    ALPim=-kapp2*Ediph*cos(Wkx0)/(kapp2**2+Det2pi**2)
!FFFFFFFFFFFFFFFFFFFFFF
   Nphoton=Ediph*Ediph*cos(Wkx0)*cos(Wkx0)
   Nphoton=Nphoton/(kapp2**2+Det2pi**2)
 
!!ADD CS POTENTIAL CORRECTION to frequency squared
    C1=-Edip/XM*2.*ALPRe*cos(Wkx0)
    omx=omx+C1*(WK*WK*sin(thet)*sin(thet)+ 2./WX**2)
   omy=omy+C1*(WK*WK*cos(thet)*cos(thet)+ 2./Wy**2)
!omx=omx+C1*(WK*WK*sin(thet)*sin(thet))
!omy=omy+C1*(WK*WK*cos(thet)*cos(thet))
    omz=omz-2.*Edip/XM*ALPre*(WK-1.d0/ZR)**2*cos(Wkx0)

! ADD CAVITY**2 term: usually small unless many photons
     write(6,*)'number of Photons',Nphoton
     C1=2.*Polaris*epsCAV**2*Nphoton*WK**2*cos(2*Wkx0)/XM
 !    omx=omx +C1*sin(thet)*sin(thet)
 !   omy=omy +C1*cos(thet)*cos(thet)

      omx=sqrt(omx)
      omy=sqrt(omy)
      omz=sqrt(omz)
    write(iwrite,*)'Frequencies WITHOUT optical spring'
   write(iwrite,*)'mech freq Omx/2pi=',omx/2/pi
   write(iwrite,*)'mech freq Omy/2pi=',omy/2/pi
   write(iwrite,*)'mech freq Omz/2pi=',omz/2/pi
 ! This is the end of calculation of the static frequency, corrected by co-trapping.

!&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


! Optomechanical couplings
    XZPF=sqrt(hbar/(2.d0*XM*OMX))
    YZPF=sqrt(hbar/(2.d0*XM*OMY))
    ZZPF=sqrt(hbar/(2.d0*XM*OMZ))

   GX=Ediph*WK*XZPF*sin(thet)*sin(Wkx0)
   GY=Ediph*WK*YZPF*cos(thet)*sin(Wkx0)
   GZ=-Ediph*(WK-1.d0/ZR)*ZZPF*cos(Wkx0)
      write(iwrite,*)'GX , GY, GZ in Hz',GX/2/pi,GY/2/pi,Gz/2/pi
  GXY=Ediph*WK*XZPF*WK*YZPF*ALPre*sin(2.*thet)*cos(Wkx0)
  GZX=2.d0*Ediph*(WK-1.d0/ZR)*ZZPF*WK*XZPF*ALPim*sin(Wkx0)*sin(thet)
  GYZ=2.d0*Ediph*(WK-1.d0/ZR)*ZZPF*WK*YZPF*ALPim*sin(Wkx0)*cos(thet)
   write(iwrite,*)'GXY , GYZ, GZX in Hz',GXY/2/pi,GYZ/2/pi,GZX/2/pi
      GMAT(1)=GX
     GMAT(2)=GY
     GMAT(3)=GZ
     GMAT(4)=GXY
     GMAT(5)=GYZ
     GMAT(6)=GZX
! calculate cooling rates

   
  call  COOL(WKX0,DET2pi,Kappa,GX,GY,GZ,OMx,OMy,OMz,COOLx,COOLy,COOLz,Gammam)

!&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
write(iwrite,*)'X optomechanical cooling rate, mechanical damping,kx0'
! we compare phonon damping with Gamma_tot=2*Gammam
  write(iwrite,100)COOLx,2.*GAMMAM,Wkx0
write(iwrite,*)'Number of phonons: at room temp;at equil'
  C1=boltz*temp/hbar/omX
  C2=2.d0*C1*GAMMAM/(abs(COOLx)+2.*GammaM)
   write(iwrite,100)C1,abs(C2)
PHON(1)=abs(C2)
write(iwrite,*)'X temperature: at room temp;at equil'
write(iwrite,*)Temp,Temp*2.*gammam/(2.*gammam+abs(coolx))


!*****************************************************
!  Y
!*****************************************************
     write(iwrite,*)'y optomechanical cooling rate, mechanical damping,Y0'
! we compare phonon damping with Gamma_tot=2*Gammam
        write(iwrite,100)COOLy,2.*GAMMAM,y0
write(iwrite,*)'Number of phonons: at room temp;at equil'
        C1=boltz*temp/hbar/omy
        C2=2.d0*C1*GAMMAM/(abs(COOLy)+2.*gammam)
         write(iwrite,100)C1,abs(C2)
write(iwrite,*)'Y temperature: at room temp;at equil'
      write(iwrite,*)Temp,Temp*2.*gammam/(2.*gammam+abs(Cooly))
          PHON(2)=abs(C2)
!*****************************************************
!  Z
!**************************************************

    write(iwrite,*)'z optomechanical cooling rate, mechanical damping, Z0'
! we compare phonon damping with Gamma_tot=2*Gammam (gammam is act gamm/2)
    write(iwrite,100)COOLz,2.*GAMMAM,z0
   write(iwrite,*)'Number of phonons: at room temp;at equil'
    C1=boltz*temp/hbar/omz
    C2=2.d0*C1*GAMMAM/(abs(COOLz)+2.*gammam)
     write(iwrite,100)C1,abs(C2)
   write(iwrite,*)'Z temperature: at room temp;at equil'
  write(iwrite,*)Temp,Temp*2.*gammam/(2.*gammam+abs(Coolz))
          PHON(3)=abs(C2)
!**************************************************
! WORK OUR Self Energy and optical spring
       write(iwrite,*)'OPTICAL SPRING EFFECT'
       write(iwrite,*)'============================='
     Call OpticalDamping(WKX0,DET2Pi,Kappa,GX,GY,GZ,OMx,OMy,OMz,COOLx,COOLy,COOLz,Gammam)
       write(iwrite,*)'============================='

100   format(4D16.8)

      RETURN
       END
!***********************************************************


!*************************************************************************
! subroutine below obtains the optomechanical parameters
 SUBROUTINE COOL(WKX0,DETUN,Kappa,GX,GY,GZ,OMx,OMy,OMz,COOLx,COOLy,COOLz,Gammam)

!************************************************************************
! Note that DETUN effectively is
! DEtun+AOPT. It is corrected by zeroth order optical contribution
! in the linearised case
!************************************************************************

     IMPLICIT NONE
     integer  ::ii,m,jj,iwrite
    integer  ::NPERIOD,NTOT
    double precision::R0,RHO,EPSR,EPSI0,C,hbar,BOLTZ,TEMP,Gravity
    double precision::WK,waist,XL,Finesse,Press,WX,WY,DelFSR
    double precision::PIN1,DET1,X0,Y0,Z0,Theta0
    include 'CSCAVITY.h'


     double precision:: DETUN,GammaM,GX,GY,Gz,OMx,OMy,OMz
     double precision:: KAPPA,KAPP2
     double precision::pi,pi2
     double precision:: WKX0
     double precision::C1,C2,C3,C4,COOLx,COOLy,COOLz


     PI=dacos(-1.d0)
     Kapp2=0.5d0*Kappa
! **********************************

      KAPP2=0.5d0*KAPPA
!***********************************
!  X cooling
!**********************************
!  for maximum cooling sin2kx_0=1
!       C1=sin(2.d0*WK0)**2*2.*Kapp2*WK**2*A**2*ASQ*hbar/XM/2.d0/OMX
       C1=GX**2*kappa
       C2=DETUN
!      write(iwrite,*)'shifted detuning=det+A'
!      write(iwrite,100)C2/2.d0/pi

       C3=(C2+omX)**2+kapp2**2
       C3=1.d0/C3
       C4=(C2-omx)**2+kapp2**2
       C4=1.d0/C4
        COOLx=-C1*(C3-C4)


!**************************************************
!***********************************
!  y cooling
!**********************************
       C1=Gy**2*kappa
       C2=DETUN
!      write(iwrite,*)'shifted detuning=det+A'
!      write(iwrite,100)C2/2.d0/pi

       C3=(C2+omY)**2+kapp2**2
       C3=1.d0/C3
       C4=(C2-omY)**2+kapp2**2
       C4=1.d0/C4
        COOLy=-C1*(C3-C4)

!**************************************************
!***********************************
!  z cooling
!**********************************
     C1=Gz**2*kappa
     C2=DETUN
!     write(iwrite,*)'shifted detuning=det+A'
 !     write(iwrite,100)C2/2.d0/pi
! here we neglect opto shift!
     C3=(C2+omz)**2+kapp2**2
     C3=1.d0/C3
     C4=(C2-omz)**2+kapp2**2
     C4=1.d0/C4
     COOLz=-C1*(C3-C4)

!**************************************************

100   format(4D16.8)

      RETURN
       END

!*************************************************************************
! subroutine below obtains the optomechanical parameters
 SUBROUTINE OpticalDamping(WKX0,DETUN,Kappa,GX,GY,GZ,OMx,OMy,OMz,COOLx,COOLy,COOLz,Gammam)

!************************************************************************
! Note that DETUN effectively is
! DEtun+AOPT. It is corrected by zeroth order optical contribution
! in the linearised case
!************************************************************************

     IMPLICIT NONE
     integer  ::ii,m,jj,iwrite
    integer  ::NPERIOD,NTOT
    double precision::R0,RHO,EPSR,EPSI0,C,hbar,BOLTZ,TEMP,Gravity
    double precision::WK,waist,XL,Finesse,Press,WX,WY,DelFSR
    double precision::PIN1,DET1,X0,Y0,Z0,Theta0
    include 'CSCAVITY.h'


     double precision:: DETUN,GammaM,GX,GY,Gz,OMx,OMy,OMz,omega
     double precision:: KAPPA,KAPP2,XM
     double precision::pi,pi2
     double precision:: WKX0,ANORM,BNORM,t1,t2,gamm2
     double precision::C1,C2,C3,C4,COOLx,COOLy,COOLz,SpringX,SpringY
     Double complex::CHIX,CHIY,CHINegstarX,CHINegstarY,CHIMX,CHIMY
     Double complex::SelfEX,SelfEY,XI,C2DX,C2DY

! i
        XI=CMPLX(0.d0,1.d0)
      PI=dacos(-1.d0)
     XM=RHO*4.*pi/3.*R0**3
! we already halved gammam
     gamm2=gammam
     Kapp2=0.5d0*Kappa

! **********************************
!  x mode optical suscept at omega=omx
       omega=Omx
!*****************************************
!Chi_RX
       ANORM=KAPP2**2+ (omega+DETUN)**2
       t1=kapp2/ANORM
       t2= (omega+DETUN)/ANORM
       CHIX=CMPLX(t1,t2)
! chi_r^*(-omega)^X
      ANORM=KAPP2**2+ (-omega+DETUN)**2
      t1=kapp2/ANORM
      t2= (-omega+DETUN)/ANORM
      CHINegstarX=CMPLX(t1,-t2)

!*****************************************
! optical suscept at omega=omy
       omega=Omy
!*****************************************
!Chi_RY
       ANORM=KAPP2**2+ (omega+DETUN)**2
       t1=kapp2/ANORM
       t2= (omega+DETUN)/ANORM
       CHIY=CMPLX(t1,t2)
! chi_r^*(-omega)^Y
      ANORM=KAPP2**2+ (-omega+DETUN)**2
      t1=kapp2/ANORM
      t2= (-omega+DETUN)/ANORM
      CHINegstarY=CMPLX(t1,-t2)

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!******************************************
! mechanical susceptibility of X at the frequency of Y???NOT SURE:CHECK
       omega=Omy
!******************************************
! X MECHANICAL susceptibility
! chi_M X
        BNORM=(Gamm2)**2 + (omega-OMX)**2
      t1= (Gamm2)/BNORM
      t2=(omega-OMX)/BNORM
      CHIMX=CMPLX(t1,t2)
!******************************************
! mechanical susceptibility of Y at the frequency of X???NOT SURE
       omega=Omx
!******************************************
! Y MECHANICAL susceptibility
! chi_M Y
        BNORM=(Gamm2)**2 + (omega-OMY)**2
      t1= (Gamm2)/BNORM
      t2=(omega-OMY)/BNORM
      CHIMY=CMPLX(t1,t2)
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!****************************************************
     SelfEX=-XI*2.*GX**2*(CHIX-CHINegstarX)*XM*Omx
     SelfEY=-XI*2.*GY**2*(CHIY-CHINegstarY)*XM*Omy

! For 2D we can add the additional correction.
     C2DX=1.d0+GY**2*(CHIY-CHINegstarY)*CHIMY
     C2DY=1.d0+GX**2*(CHIX-CHINegstarX)*CHIMX
! LEAVE OUT 2D correction for now
     SelfEX=SelfEX/C2DX
     SelfEY=SelfEY/C2DY
!******************************************


       Coolx=AIMAG(SelfEX)/XM/Omx
      SpringX=Real(SelfEX)/XM
      SpringX=Omx*Omx+SpringX
     SpringX=sqrt(SpringX)

       CoolY=AIMAG(SelfEY)/XM/Omy
        SpringY=Real(SelfEY)/XM
    SpringY=Omy*Omy+SpringY
    SpringY=sqrt(SpringY)
!**************************************************
     write(6,*)'wkx0,omx no spring, omx +spring, omy no spring, omy + spring '
     write(6,100)Wkx0/2/pi,omx/2/pi,SpringX/2/pi,Omy/2/pi,SpringY/2/pi
     write(12,100)WKx0/2/pi,1.e-3*omx/2/pi,1.e-3*SpringX/2/pi,1.e-3*Omy/2/pi,1.e-3*SpringY/2/pi
100   format(5F16.4)

      RETURN
       END
