! input parameters for UCL- Antonio cavity
! NPERIOD=number of periods in prop. (of fastest oscillation: kappa^-1 or om_M)
! NTOT=number of equations so 8=>1 optical mode +3D
! R0= sphere radius
! XL=cavity length 
! Pin1=input power in Watts tweezer beam
!
! Press = air pressure in millibars
! XL=cavity length
! rho=sphere density
! WK=2*pi/lambda=k
! DelFSR = 1 FSR= 14 GHz
! DET1=detuning in KHz trap beam
! Theta0= angle between tweezer polarization and cavity axis. Given as as
! FRACTION of pi so pi/4  is theta0=0.25
! waist is  waist radius
!
 PARAMETER(NPERIOD=10000,NTOT=8,R0=75.d-9, &
            rho=2200.d0,EPSR=2.07d0,Epsi0=8.854d-12, &
    c=3.d8,hbar=1.05d-34,BOLTZ=1.4d-23,TEMP=300.d0,Gravity=9.8d0,&
    WK=5.9d6,waist=40.d-6,WY=1.25*0.85d-6,WX=1.0*0.85d-6,XL=0.96d-2,Finesse=4.d4, &
!    Press=1.e-7,Pin1=0.4d0,DET1=-450.d3,DelFSR=0.d9,theta0=0.25d0,&
     Press=1.e-5,Pin1=0.4d0,DET1=-450.d3,DelFSR=0.d9,theta0=0.25d0,&
! Equilibrium positions 
	    ! node is X0=0.25*lambda
	    Y0=0.0d-6,X0=0.25d0*1.064d-6,Z0=0.d-6)
