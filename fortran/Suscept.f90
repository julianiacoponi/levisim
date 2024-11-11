! FUNCTION chi_M(&
!     omega, &
!     half_Gamma_M, &
!     Omega_M)
!     ! chi_M(omega)
!     IMPLICIT NONE
!     double precision::omega, half_Gamma_M, Omega_M
!     double precision::chi_real, chi_imag, norm
!     double complex::chi_M
!     ! MECHANICAL susceptibility
!     ! chi_M
!     norm = (half_Gamma_M)**2 + (omega - Omega_M)**2
!     chi_real = (half_Gamma_M) / norm
!     chi_imag = (omega - Omega_M) / norm
!     ChiM = cmplx(chi_real, chi_imag)
!     RETURN
! END

! FUNCTION chi_M_star(omega, half_Gamma_M, Omega_M)
!     ! chi_M*(-omega)
!     IMPLICIT NONE
!     double precision::omega, half_Gamma_M, Omega_M
!     double precision::chi_real, chi_imag, norm
!     double complex::chi_M_star

!     norm = (half_Gamma_M)**2 + (omega + Omega_M)**2
!     chi_real = (half_Gamma_M) / norm
!     chi_imag = (omega + Omega_M) / norm
!     chi_M_star = cmplx(chi_real, chi_imag)
!     RETURN
! END

! FUNCTION chi_C(omega, half_kappa, cavity_detuning)
!     IMPLICIT NONE

!     double precision::omega, half_kappa, cavity_detuning
!     double precision::chi_real, chi_imag, norm
!     double complex::chi_C

!     norm = half_kappa**2 + (omega + cavity_detuning)**2
!     chi_real = half_kappa / norm
!     chi_imag = (omega + cavity_detuning) / norm
!     chi_C = cmplx(chi_real, chi_imag)
!     RETURN
! END

! FUNCTION chi_C_star(omega, half_kappa, cavity_detuning)
!     IMPLICIT NONE
!     double precision::omega, half_kappa, cavity_detuning, T1, T2, norm
!     double complex::chi_C_star
!     ! chi_r^*(-omega) = Chi^star(-Omega) = chi_C_star
!     norm = half_kappa**2 + (-omega + cavity_detuning)**2
!     chi_real = half_kappa / norm
!     chi_imag = (-omega + cavity_detuning) / norm
!     chi_C_star = cmplx(chi_real, -chi_imag)
!     RETURN
! END

FUNCTION CHI_FUNCTION(&
    freq, &
    resonance, &
    damping)

    IMPLICIT NONE

    double precision::freq, resonance, damping
    double precision::chi_real, chi_imag, norm
    double complex::CHI_FUNCTION

    norm = (damping / 2.)**2 + (freq - resonance)**2
    chi_real = (damping / 2.) / norm
    chi_imag = (freq - resonance) / norm
    CHI_FUNCTION = cmplx(chi_real, chi_imag)

    RETURN
END

FUNCTION CHI_STAR_FUNCTION(&
    freq, &
    resonance, &
    damping)

    IMPLICIT NONE

    double precision::freq, resonance, damping
    double precision::chi_real, chi_imag, norm
    double complex::CHI_STAR_FUNCTION

    ! chi*(-freq)
    ! -freq instead of freq
    norm = (damping / 2.)**2 + (-freq - resonance)**2
    chi_real = (damping / 2.) / norm
    chi_imag = (-freq - resonance) / norm

    ! conjugate* by using -chi_imag
    CHI_STAR_FUNCTION = cmplx(chi_real, -chi_imag)

    RETURN
END

