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
    CHI_FUNCTION = cmplx(chi_real, chi_imag, kind=8)

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
    CHI_STAR_FUNCTION = cmplx(chi_real, -chi_imag, kind=8)

    RETURN
END

