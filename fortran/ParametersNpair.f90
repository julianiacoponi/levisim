SUBROUTINE FILL_PARAMETER_ARRAYS(&
    Gamma_M, &
    kBT_h, &
    phase_per_trap, &
    mech_freq_array, &
    Gamma_M_array, &
    light_couplings_array, &
    mech_couplings_array, &
    quanta_per_operator)

    IMPLICIT NONE

    integer::num_traps, Cartesian_axes_per_trap, num_operators, num_dimensions
    INCLUDE 'Nparticle.h'

    integer::trap, axis, count

    double precision::Gamma_M, kBT_h
    double precision::pi, pi2, kHz
    double precision::omega_gap

    double precision::quanta_per_operator(num_operators)
    double precision::phase_per_trap(num_traps)
    double precision::mech_freq_array(num_traps, Cartesian_axes_per_trap)
    double precision::Gamma_M_array(num_traps, Cartesian_axes_per_trap)

    double precision::light_couplings_array(num_traps, Cartesian_axes_per_trap)
    double precision::mech_couplings_array(num_traps, Cartesian_axes_per_trap, Cartesian_axes_per_trap)


    pi = dacos(-1.d0)
    pi2 = pi + pi
    kHz = 1.d3 * pi2

    phase_per_trap = 0.d0
    Gamma_M_array = 0.d0
    light_couplings_array = 0.d0

    mech_freq_array = 0.d0
    mech_couplings_array = 0.d0

    mech_freq_array(1, 1) = 312.d0 * kHz
    omega_gap = 15.d0 * kHz
    count = 0
    DO trap=1, num_traps
        phase_per_trap(trap) = 0.d0
        DO axis=1, Cartesian_axes_per_trap

            Gamma_M_array(trap, axis) = Gamma_M
            light_couplings_array(trap, axis) = -40.d0 * kHz

            mech_freq_array(trap, axis) = mech_freq_array(1, 1) - (trap - 1) * omega_gap
            IF(axis == 2) THEN
                mech_freq_array(trap, axis) = mech_freq_array(trap, axis) + omega_gap / 10.d0
            ENDIF
            WRITE(6, *) trap, axis, count, mech_freq_array(trap, axis) / kHz
            count = count + 1

        ENDDO
    ENDDO

    ! Fill quanta_per_operator
    ! phonon occupancies for b_u and b_u^\dagg
    ! u = x1, y1, ...
    ! quanta_per_operator(1) = noise_photons (shot noise)
    ! quanta_per_operator(2) = noise_photons (shot noise)
    ! quanta_per_operator(3) = x1 phonons
    ! quanta_per_operator(4) = x1 phonons
    ! quanta_per_operator(5) = y1 phonons
    ! quanta_per_operator(6) = y1 phonons
    quanta_per_operator = 0.d0
    count = 3
    DO trap=1, num_traps
        DO axis=1, Cartesian_axes_per_trap
            quanta_per_operator(count) = kBT_h / mech_freq_array(trap, axis)
            quanta_per_operator(count + 1) = kBT_h / mech_freq_array(trap, axis)
            count = count + 2
        ENDDO
    ENDDO

    RETURN
END
