! Equally-spaced mechanical frequency ladder.
!
! Ported from the legacy ParametersNP.f90, which still used the pre-rename interface
! (SUBROUTINE Parameters with Npcle/Nmode/NT) and therefore no longer compiled against
! Nparticle.h. The physics is unchanged; only the interface and naming were updated:
!
!     Npcle      -> num_traps                  OMEGS     -> mech_freq_array
!     Nmode      -> Cartesian_axes_per_trap    GMAT      -> light_couplings_array
!     NT         -> num_operators              GMM       -> mech_couplings_array
!     GammaM     -> Gamma_M                    GammaMod  -> Gamma_M_array
!     cbath      -> kBT_h                      AVN       -> quanta_per_operator
!     Phases     -> phase_per_trap
!
! The legacy arguments G1, OMX, OMY and OMZ are not part of FILL_PARAMETER_ARRAYS and were
! unused in the body, so they are dropped.
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
    double precision::omega_first, omega_step, omega_parked

    double precision::quanta_per_operator(num_operators)
    double precision::phase_per_trap(num_traps)
    double precision::mech_freq_array(num_traps, Cartesian_axes_per_trap)
    double precision::Gamma_M_array(num_traps, Cartesian_axes_per_trap)

    double precision::light_couplings_array(num_traps, Cartesian_axes_per_trap)
    double precision::mech_couplings_array(num_traps, Cartesian_axes_per_trap, Cartesian_axes_per_trap)

    pi = dacos(-1.d0)
    pi2 = pi + pi
    kHz = 1.d3 * pi2

    ! the ladder starts here and steps down by omega_step for every mode in turn
    omega_first = 320.d0 * kHz
    omega_step = 3.d0 * kHz

    ! two modes are deliberately parked far above the sweep so they take no part in the
    ! dynamics - this reproduces the legacy `Omegs(3,1)` and `Omegs(6,2)` overrides
    omega_parked = 5.d6 * pi2

    phase_per_trap = 0.d0
    Gamma_M_array = 0.d0
    light_couplings_array = 0.d0
    mech_freq_array = 0.d0
    mech_couplings_array = 0.d0

    count = 0
    DO trap=1, num_traps
        DO axis=1, Cartesian_axes_per_trap

            Gamma_M_array(trap, axis) = Gamma_M
            light_couplings_array(trap, axis) = -40.d0 * kHz

            ! NOTE: count runs over every (trap, axis) pair, so the ladder is uniform
            ! across all modes rather than stepping once per trap
            mech_freq_array(trap, axis) = omega_first - count * omega_step
            WRITE(14, *) mech_freq_array(trap, axis) / kHz, 0.25d0
            count = count + 1

        ENDDO
    ENDDO

    IF(num_traps >= 3) mech_freq_array(3, 1) = omega_parked
    IF(num_traps >= 6 .AND. Cartesian_axes_per_trap >= 2) mech_freq_array(6, 2) = omega_parked

    ! phonon occupancies for b_u and b_u^\dagg, u = x1, y1, ...
    ! entries 1 and 2 are the optical shot-noise quanta and stay zero
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
