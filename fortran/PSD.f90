! Generic routine for noise spectra of trap and probe beams
SUBROUTINE CALCULATE_NOISE_SPECTRA(&
    omega, &
    cavity_detuning, &
    kappa, &
    theta_homodyne, &
    noise_photons, &
    phase_per_tweezer, &
    mech_freq_array, &
    Gamma_M_array, &
    quanta_per_operator, &
    light_couplings_array, &
    mech_couplings_array, &
    PSD_array, &
    PSD_value_CBM, &
    homodyne_value, &
    heterodyne_value)

    IMPLICIT NONE

    integer::num_traps, Cartesian_axes_per_trap, num_operators
    INCLUDE 'Nparticle.h'

    integer::DEBUG
    integer::num_dimensions
    integer::trap, axis

    double precision::pi, pi2, kHz
    double precision::omega
    double precision::theta_homodyne

    double precision::cavity_detuning, kappa
    double precision::Omega_M, Gamma_M
    double precision::homodyne_value, heterodyne_value
    double precision::real_CSD, PSD_value_CBM

    double precision::noise_photons
    double precision::quanta_per_operator(num_operators)
    double precision::phase_per_tweezer(num_traps)

    double precision::light_couplings_array(num_traps, Cartesian_axes_per_trap)
    double precision::mech_couplings_array(num_traps, Cartesian_axes_per_trap)
    double precision::mech_freq_array(num_traps, Cartesian_axes_per_trap)
    double precision::Gamma_M_array(num_traps, Cartesian_axes_per_trap)
    double precision::PSD_array(num_traps, Cartesian_axes_per_trap)

    double complex::CHI_FUNCTION, CHI_STAR_FUNCTION
    double complex::chi_C, chi_C_star

    double complex::chi_M_array(num_traps, Cartesian_axes_per_trap)
    double complex::chi_M_star_array(num_traps, Cartesian_axes_per_trap)

    ! noise vectors: 1 noise per operator
    double complex::X_noise_vector(num_traps, Cartesian_axes_per_trap, num_operators)
    double complex::&
        a_hat(num_operators), &
        a_dag(num_operators), &
        CBM_noise_vector(num_operators), &
        a_hat_CBM(num_operators), &
        a_dag_CBM(num_operators)

    num_dimensions = num_traps * Cartesian_axes_per_trap

    pi = dacos(-1.d0)
    pi2 = 2.d0 * pi
    kHz = 1.d3 * pi2

    homodyne_value = 0.d0
    heterodyne_value = 0.d0

    ! zero all arrays as a precaution
    PSD_array = 0.d0
    real_CSD = 0.d0
    chi_M_star_array = 0.d0
    X_noise_vector = 0.d0

    a_hat = 0.d0
    a_dag = 0.d0
    CBM_noise_vector = 0.d0
    a_hat_CBM = 0.d0

    PSD_value_CBM = 0.d0
    heterodyne_value = 0.d0

    chi_C = CHI_FUNCTION(omega, -cavity_detuning, kappa)
    chi_C_star = CHI_STAR_FUNCTION(omega, -cavity_detuning, kappa)

    DO trap=1, num_traps
        DO axis=1, Cartesian_axes_per_trap
            Omega_M = mech_freq_array(trap, axis)
            Gamma_M = Gamma_M_array(trap, axis)

            chi_M_array(trap, axis) = CHI_FUNCTION(omega, Omega_M, Gamma_M)
            chi_M_star_array(trap, axis) = CHI_STAR_FUNCTION(omega, Omega_M, Gamma_M)
        ENDDO
    ENDDO

    IF ((omega / kHz).GE.297.95.AND.(omega / kHz).LE.298.05) THEN
        WRITE(6, *) '==========================================================='
        WRITE(6, *) 'PSD.f90 DEBUG (before CALCULATE_NOISE_VECTORS)'
        WRITE(6, *) '==========================================================='

        WRITE(6, *) 'omega (kHz)', omega / kHz
        WRITE(6, *) 'chi_C', omega / kHz, chi_C
        WRITE(6, *) 'chi_C_star', omega / kHz, chi_C_star
    ENDIF



    ! work out X_noise_vector = vector of noises (inc optical) for mechanical modes
    ! And also from the above get optical noise vectors a_hat and a_dag (only one optical mode here)
    CALL CALCULATE_NOISE_VECTORS(&
        omega, &
        kappa, &
        light_couplings_array, &
        mech_couplings_array, &
        Gamma_M_array, &
        phase_per_tweezer, &
        chi_C, &
        chi_C_star, &
        chi_M_array, &
        chi_M_star_array, &
        a_hat, &
        a_dag, &
        X_noise_vector, &
        CBM_noise_vector, &
        a_hat_CBM, &
        a_dag_CBM)

    IF ((omega / kHz).GE.297.95.AND.(omega / kHz).LE.298.05) THEN
        WRITE(6, *) '==========================================================='
        WRITE(6, *) 'PSD.f90 DEBUG (after CALCULATE_NOISE_VECTORS)'
        WRITE(6, *) '==========================================================='
        WRITE(6, *) 'omega (kHz)', omega / kHz
        WRITE(6, *) 'a_out_hat', a_hat
        WRITE(6, *) 'a_out_dag', a_dag
        WRITE(6, *) '-----------------------------------------------------------'
        WRITE(6, *) 'X_noise_vector(1, 1, :)', X_noise_vector(1, 1, :)
        WRITE(6, *) 'X_noise_vector(1, 2, :)', X_noise_vector(1, 2, :)
        WRITE(6, *) 'X_noise_vector(2, 1, :)', X_noise_vector(2, 1, :)
        WRITE(6, *) 'X_noise_vector(2, 2, :)', X_noise_vector(2, 2, :)
        WRITE(6, *) 'X_noise_vector(8, 2, :)', X_noise_vector(8, 2, :)
        WRITE(6, *) '-----------------------------------------------------------'
        ! 101 FORMAT(7E14.6)
    ENDIF

    CALL CALCULATE_POWER_SPECTRA(&
        omega, &
        noise_photons, &
        quanta_per_operator, &
        X_noise_vector, &
        PSD_array, &
        a_dag_CBM, &  ! NOTE: this is 'CBM_noise_vector' apparently?
        PSD_value_CBM, &
        real_CSD)

    IF ((omega / kHz).GE.297.95.AND.(omega / kHz).LE.298.05) THEN
        WRITE(6, *) '-----------------------------------------------------------'
        WRITE(6, *) 'PSD_array(1, 1)', PSD_array(1, 1)
        WRITE(6, *) 'PSD_array(1, 2)', PSD_array(1, 2)
        WRITE(6, *) 'PSD_array(2, 1)', PSD_array(2, 1)
        WRITE(6, *) 'PSD_array(2, 2)', PSD_array(2, 2)
        WRITE(6, *) 'PSD_array(8, 2)', PSD_array(8, 2)
        WRITE(6, *) '-----------------------------------------------------------'
    ENDIF

    CALL CALCULATE_HOMODYNE_SPECTRUM_VALUE(&
        omega, &
        quanta_per_operator, &
        noise_photons, &
        theta_homodyne, &
        a_hat, &
        a_dag, &
        homodyne_value)

    CALL CALCULATE_HETERODYNE_SPECTRUM_VALUE(&
        omega, &
        quanta_per_operator, &
        noise_photons, &
        a_hat, &
        a_dag, &
        heterodyne_value)

        IF ((omega / kHz).GE.297.95.AND.(omega / kHz).LE.298.05) THEN
            WRITE(6, *) '-----------------------------------------------------------'
            WRITE(6, *) 'omega (kHz)', omega / kHz
            WRITE(6, *) 'homodyne_value', homodyne_value
            WRITE(6, *) 'theta_homodyne (π)', theta_homodyne / pi
            WRITE(6, *) 'heterodyne_value', heterodyne_value
            WRITE(6, *) '-----------------------------------------------------------'
        ENDIF

    RETURN
END


SUBROUTINE CALCULATE_POWER_SPECTRA(&
    omega, &
    noise_photons, &
    quanta_per_operator, &
    X_noise_vector, &
    PSD_array, &
    CBM_noise_vector, &
    PSD_value_CBM, &
    real_CSD)
    ! work out mechanical PSDs: work in progress to add cross correlations
    IMPLICIT NONE

    integer::ii, trap, axis, num_traps, Cartesian_axes_per_trap, num_operators, icount
    integer::jj, kk, trap_A, trap_B
    INCLUDE 'Nparticle.h'
    double precision::omega
    double precision::noise_photons

    double precision::real_CSD, PSD_value_CBM
    double complex::CSD_value
    double precision::quanta_per_operator(num_operators)

    double precision::PSD_array(num_traps, Cartesian_axes_per_trap)
    double complex::X_noise_vector(num_traps, Cartesian_axes_per_trap, num_operators)
    double complex::CBM_noise_vector(num_operators)

    ! noise vector of:
    ! a_in, a_in^\dagg
    ! b_in,X1, b^\dagg_in,X1
    ! b_in,Y1, b^\dagg_in,Y1
    ! b_in,X2, b^\dagg_in,X2
    ! ...
    ! quanta_per_operator(1) = quanta_per_operator(2) = 0 (shot noise)
    ! quanta_per_operator(3) = quanta_per_operator(4) = X1 phonons
    ! quanta_per_operator(5) = quanta_per_operator(6) = Y1 phonons
    PSD_array = 0.d0
    real_CSD = 0.d0
    PSD_value_CBM = 0.d0
    ! Add optical noise contributions ie components with a_in, a_in^\dagger. Single optical mode
    icount = 0
    DO 10 trap=1, num_traps
        DO 20 axis=1, Cartesian_axes_per_trap
            icount = icount + 1
            PSD_array(trap, axis) = PSD_array(trap, axis) &
                + (noise_photons + 1) * (abs(X_noise_vector(trap, axis, 1)))**2 &
                + noise_photons * (abs(X_noise_vector(trap, axis, 2)))**2
        20 ENDDO
    10 ENDDO

    PSD_value_CBM = PSD_value_CBM &
        + (noise_photons + 1) * (abs(CBM_noise_vector(1)))**2 &
        + (noise_photons) * (abs(CBM_noise_vector(2)))**2

    ! PSD_value_CBM = PSD_value_CBM + (noise_photons + 1) * (abs(CBM_noise_vector(2)))**2
    IF(icount + 2.GT.num_operators) WRITE(6, *) 'warning: dimensions in PSD wrong', icount + 2, num_operators

    ! Add MECHANICAL noise contributions ie components with b_in, b_in^\dagger
    DO 11 trap=1, num_traps
        DO 21 axis=1, Cartesian_axes_per_trap
            DO ii=2, num_operators - 2, 2
                PSD_array(trap, axis) = &
                PSD_array(trap, axis) &
                + (quanta_per_operator(ii + 1) + 1.) * (abs(X_noise_vector(trap, axis, ii + 1)))**2 &
                + quanta_per_operator(ii + 1) * (abs(X_noise_vector(trap, axis, ii + 2)))**2
            ENDDO
        21 ENDDO
    11 ENDDO

    PSD_value_CBM = 0.d0
    DO ii=2, num_operators - 2, 2
        PSD_value_CBM = PSD_value_CBM + (quanta_per_operator(ii + 1) + 1.) * (abs(CBM_noise_vector(ii + 1)))**2 &
        + quanta_per_operator(ii + 1) * (abs(CBM_noise_vector(ii + 2)))**2
    ENDDO

    ! cross correlations for a pair of modes
    ! pick two modes
    jj = 1
    kk = 2
    trap_A = 1
    trap_B = 1
    ! optical terms
    CSD_value = (noise_photons + 1) * X_noise_vector(trap_A, jj, 1) * conjg(X_noise_vector(trap_B, kk, 1))
    CSD_value = CSD_value + noise_photons * X_noise_vector(trap_A, jj, 2) * conjg(X_noise_vector(trap_B, kk, 2))

    DO ii=2, num_operators - 2, 2
        CSD_value = CSD_value &
            + (quanta_per_operator(ii + 1) + 1.) * X_noise_vector(trap_A, jj, ii + 1) * conjg(X_noise_vector(trap_B, kk, ii + 1)) &
            + quanta_per_operator(ii + 1) * X_noise_vector(trap_A, jj, ii + 2) * conjg(X_noise_vector(trap_B, kk, ii + 2))
    ENDDO
    real_CSD = real(CSD_value)
    ! for non-zero they need to have been appropriately populated in Noisevec
    RETURN
END


SUBROUTINE CALCULATE_HOMODYNE_SPECTRUM_VALUE(&
    omega, &
    quanta_per_operator, &
    noise_photons, &
    theta_homodyne, &
    a_hat, &
    a_dag, &
    homodyne_value)

    IMPLICIT NONE

    integer::ii, num_operators, num_traps, Cartesian_axes_per_trap
    INCLUDE 'Nparticle.h'

    double precision::omega, kHz
    double precision::theta_homodyne, homodyne_value, noise_photons
    double complex::imag_num

    double precision::quanta_per_operator(num_operators)
    double complex::homodyne_noise_vector(num_operators)
    double complex::phase_quadrature(num_operators), amplitude_quadrature(num_operators)
    double complex::a_hat(num_operators)
    double complex::a_dag(num_operators)

    kHz = 1.d3 * 2.d0 * dacos(-1.d0)
    ! zero arrays
    imag_num = complex(0.d0, 1.d0)
    phase_quadrature = (0.d0, 0.d0)
    amplitude_quadrature = (0.d0, 0.d0)
    homodyne_noise_vector = (0.d0, 0.d0)

    ! optical quadrature
    DO ii=1, num_operators
        amplitude_quadrature(ii) = a_hat(ii) + a_dag(ii)
        phase_quadrature(ii) = imag_num * (a_hat(ii) - a_dag(ii))
        homodyne_noise_vector(ii) = &
            phase_quadrature(ii) * sin(theta_homodyne) &
            + amplitude_quadrature(ii) * cos(theta_homodyne)
    ENDDO

    IF ((omega / kHz).GE.297.95.AND.(omega / kHz).LE.298.05) THEN
        WRITE(6, *) '-----------------------------------------------------------'
        WRITE(6, *) 'omega (kHz)', omega / kHz
        WRITE(6, *) 'homodyne_noise_vector', homodyne_noise_vector
        WRITE(6, *) '-----------------------------------------------------------'
    ENDIF


    ! homodyne PSD
    homodyne_value = 0.d0
    homodyne_value = homodyne_value &
        + abs(homodyne_noise_vector(1))**2 * noise_photons &
        + abs(homodyne_noise_vector(2))**2 * (noise_photons + 1.d0)

    DO ii=2, num_operators-2, 2
        homodyne_value = homodyne_value &
            + abs(homodyne_noise_vector(ii + 1))**2 * (quanta_per_operator(ii + 1) + 1) &
            + abs(homodyne_noise_vector(ii + 2))**2 * quanta_per_operator(ii + 1)
    ENDDO

    RETURN
END


SUBROUTINE CALCULATE_HETERODYNE_SPECTRUM_VALUE(&
    omega, &
    quanta_per_operator, &
    noise_photons, &
    a_hat, &
    a_dag, &
    heterodyne_value)

    IMPLICIT NONE

    integer::ii, num_operators, num_traps, Cartesian_axes_per_trap
    INCLUDE 'Nparticle.h'


    double precision::omega, kHz
    double precision::heterodyne_value, noise_photons
    double complex::imag_num

    double precision::quanta_per_operator(num_operators)
    double complex::heterodyne_noise_vector(num_operators)
    double complex::a_hat(num_operators)
    double complex::a_dag(num_operators)


    kHz = 1.d3 * 2 * dacos(-1.d0)
    heterodyne_noise_vector = (0.d0, 0.d0)

    imag_num = complex(0.d0, 1.d0)
    DO ii=1, num_operators
        ! heterodyne_noise_vector(ii) = a_dag(ii)
        heterodyne_noise_vector(ii) = a_hat(ii)
    ENDDO

    IF ((omega / kHz).GE.297.95.AND.(omega / kHz).LE.298.05) THEN
        WRITE(6, *) '-----------------------------------------------------------'
        WRITE(6, *) 'omega (kHz)', omega / kHz
        WRITE(6, *) 'heterodyne_noise_vector', heterodyne_noise_vector
        WRITE(6, *) '-----------------------------------------------------------'
    ENDIF

    ! heterodyne PSD
    heterodyne_value = 0.d0
    heterodyne_value = &
        heterodyne_value &
        + abs(heterodyne_noise_vector(1))**2 * (noise_photons + 1.d0) &
        + abs(heterodyne_noise_vector(2))**2 * (noise_photons)

    DO ii=2, num_operators - 2, 2
        heterodyne_value = &
            heterodyne_value &
            + abs(heterodyne_noise_vector(ii + 1))**2 * (quanta_per_operator(ii + 1) + 1) &
            + abs(heterodyne_noise_vector(ii + 2))**2 * quanta_per_operator(ii + 1)
    ENDDO

    RETURN
END


SUBROUTINE CALCULATE_HOMODYNE_SPECTRUM_VALUE_VIA_OPTICAL_OPERATORS_AND_CSD(&
    quanta_per_operator, &
    noise_photons, &
    theta_homodyne, &
    a_hat, &
    a_dag, &
    homodyne_value)
    IMPLICIT NONE
    integer::ii, num_operators, num_traps, Cartesian_axes_per_trap
    INCLUDE 'Nparticle.h'
    double precision::theta_homodyne, homodyne_value, noise_photons
    double complex::imag_num, complex_2_theta, CSD_value

    double precision::quanta_per_operator(num_operators)
    double complex::a_hat(num_operators)
    double complex::a_dag(num_operators)

    complex_2_theta = complex(cos(2 * theta_homodyne), sin(2 * theta_homodyne))
    imag_num = complex(0.d0, 1.d0)

    ! Homodyne PSD square terms
    homodyne_value = 0.d0

    homodyne_value = &
        homodyne_value &
        + abs(a_hat(1))**2 * noise_photons &
        + abs(a_hat(2))**2 * (noise_photons + 1.d0)

    DO ii=2, num_operators - 2, 2
        homodyne_value = &
            homodyne_value &
            + abs(a_hat(ii + 1))**2 * (quanta_per_operator(ii + 1) + 1) &
            + abs(a_hat(ii + 2))**2 * quanta_per_operator(ii + 1)
    ENDDO

    homodyne_value = &
        homodyne_value &
        + abs(a_dag(1))**2 * noise_photons &
        + abs(a_dag(2))**2 * (noise_photons + 1.d0)

    DO ii=2, num_operators - 2, 2
        homodyne_value = &
            homodyne_value &
            + abs(a_dag(ii + 1))**2 * (quanta_per_operator(ii + 1) + 1) &
            + abs(a_dag(ii + 2))**2 * quanta_per_operator(ii + 1)
    ENDDO

    ! Homodyne cross terms
    CSD_value = 0.d0
    CSD_value = &
        CSD_value &
        + a_hat(1) * conjg(a_dag(1)) * noise_photons &
        + a_hat(2) * conjg(a_dag(2)) * (noise_photons + 1.d0)

    DO ii=2, num_operators - 2, 2
        CSD_value = &
            CSD_value &
            + a_hat(ii + 1) * conjg(a_dag(ii + 1)) * (quanta_per_operator(ii + 1) + 1) &
            + a_hat(ii + 2) * conjg(a_dag(ii + 2)) * quanta_per_operator(ii + 1)
    ENDDO
    homodyne_value = &
        homodyne_value &
        + CSD_value * complex_2_theta + conjg(CSD_value * complex_2_theta)

    100 FORMAT(6D16.8)
    RETURN
END
