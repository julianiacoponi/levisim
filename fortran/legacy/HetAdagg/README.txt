

 Lapack diagonalisation
 gfortran -O2 -llapack NParticle.f90  3DEQRom.f90 ParametersNP.f90 Suscept.f90 PSD.f90 NoiseVec2D.f90


explicit labels of coupling,still Lapack
 gfortran -O2 -llapack  2Particle.f90  3DEQRom.f90 Parameters2P.f90 Suscept.f90 PSD.f90 NoiseExplicit.f90


gfortran -O2 -llapack 2Particle.f90  3DEQRom.f90 Parameters2P.f90 Suscept.f90 PSD.f90 NoiseVecMat.f90 ADJ.f90

------------------------------------------


Add/fix Heterodyne




Later
1) Add Z with tilt
2) Add momentum vectors




