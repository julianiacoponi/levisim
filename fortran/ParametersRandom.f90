! Mechanical frequency ladder with pseudo-random jitter on each mode.
!
! Ported from the legacy ParametersN8.f90, which still used the pre-rename interface and
! therefore no longer compiled against Nparticle.h. See ParametersEqual.f90 for the full
! name mapping. The physics is unchanged apart from one deliberate change, noted below.
!
! REPRODUCIBILITY: the legacy version called RANDOM_NUMBER without seeding, so every run
! produced a different frequency layout and no result could be reproduced or regression
! tested. The generator is now seeded from a fixed constant, so runs are repeatable. Set
! LEVISIM_RANDOM_SEED in the environment to vary it deliberately.
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
    integer::seed_size, seed_index
    integer, ALLOCATABLE::seed_array(:)
    character(len=32)::seed_from_environment
    integer::seed_value, read_status

    double precision::Gamma_M, kBT_h
    double precision::pi, pi2, kHz
    double precision::omega_first, omega_step, jitter_fraction

    double precision::quanta_per_operator(num_operators)
    double precision::phase_per_trap(num_traps)
    double precision::mech_freq_array(num_traps, Cartesian_axes_per_trap)
    double precision::Gamma_M_array(num_traps, Cartesian_axes_per_trap)

    double precision::light_couplings_array(num_traps, Cartesian_axes_per_trap)
    double precision::mech_couplings_array(num_traps, Cartesian_axes_per_trap, Cartesian_axes_per_trap)

    pi = dacos(-1.d0)
    pi2 = pi + pi
    kHz = 1.d3 * pi2

    omega_first = 350.d0 * kHz
    omega_step = 7.5d0 * kHz

    ! seed the generator so the layout is reproducible run to run
    seed_value = 20240501
    CALL GET_ENVIRONMENT_VARIABLE('LEVISIM_RANDOM_SEED', seed_from_environment, STATUS=read_status)
    IF(read_status == 0) READ(seed_from_environment, *, IOSTAT=read_status) seed_value
    CALL RANDOM_SEED(SIZE=seed_size)
    ALLOCATE(seed_array(seed_size))
    DO seed_index=1, seed_size
        seed_array(seed_index) = seed_value + 37 * seed_index
    ENDDO
    CALL RANDOM_SEED(PUT=seed_array)
    DEALLOCATE(seed_array)
    WRITE(6, *) 'ParametersRandom.f90: RANDOM_SEED set from', seed_value

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

            CALL RANDOM_NUMBER(jitter_fraction)
            mech_freq_array(trap, axis) = &
                omega_first - count * omega_step + jitter_fraction * omega_step

            ! the legacy code wrote x modes to unit 13 and y modes to unit 14
            IF(axis == 1) WRITE(13, *) mech_freq_array(trap, axis) / kHz, 0.25d0
            IF(axis == 2) WRITE(14, *) mech_freq_array(trap, axis) / kHz, 0.25d0
            count = count + 1

        ENDDO
    ENDDO

    ! phonon occupancies for b_u and b_u^\dagg, u = x1, y1, ...
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
