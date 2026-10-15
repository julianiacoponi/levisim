
May 2024

codes that compare Linear amplifier full numerics with collective mode formula

gfortran -O2 -llapack NParticle.f90  3DEQRom.f90 ParametersNP.f90 Suscept.f90 PSDhet.f90 NoiseVec2D.f90

equal
gfortran -O2 -llapack NParticle.f90  3DEQRom.f90 ParametersNP.f90 Suscept.f90 PSD.f90 NoiseVec2D.f90

Random
gfortran -O2 -llapack NParticle.f90  3DEQRom.f90 ParametersN8.f90 Suscept.f90 PSD.f90 NoiseVec2D.f90

Pairs
gfortran -O2 -llapack NParticle.f90  3DEQRom.f90 ParametersNpair.f90 Suscept.f90 PSD.f90 NoiseVec2D.f90
----------------------------------------------------------------------
! PSD is S_QQ by linear amplifier
! PSDCBM is S_QQ by formula
! HetCBM is heterodyne by formula
! SHet1 is heterodyne by linear amplifier

! ! work out  for negative frequency for symmetrisation
   CALL ANALYT(-OMsweep,THehom,Det2pi,Kapp2,AVphot,AVN,Gmat,GMM,Phases,GAMMAMod,OMEGS,&
          PSD,PSDCBM,HetCBM,SHOM1,SHET1)


--------------------------------------------------
gfortran -O2 -llapack 2Particle.f90  3DEQRom.f90 Parameters2P.f90 Suscept.f90 PSD.f90 NoiseVec2D.f90

gfortran -O2 2Particle.f90  3DEQRom.f90 Parameters.f90 Suscept.f90 PSD.f90 NoiseVec.f90



analyt
This just does angles & frequencies
gfortran -O2 3DSXYANGLEloop.f90 3DEQRom.f90

This does PSDs
gfortran -O2 3DSXYloopx0.f90 3DEQRom.f90

gfortran -O2 3DSXY.f90 3DEQ.f90
----------------------------------------------------------


New 4D code X1=1,Y1=2,X2=3 Y2=4



----------

(2) work out new 2pcle R and new vectors
use phase info

-----------------
(3) work out BAs from the Rs
 use new ADJ for the 2D case of Nmode=2 ONLY

------------------------------------------
(4) re-write Equil to get g and omega and Gamma.
PARAMETERS:
  Fill up Gmat (new); Fill up GMM (direct)
  Fill up OMEGS, GAMMA, AVN
   Fill up Phases


------------------------------------------


Add/fix Heterodyne




Later
1) Add Z with tilt
2) Add momentum vectors




