PARAMETER(&
    ! ---------------------------------------------------------------------------------------------
    ! constants
    vacuum_permittivity=8.854d-12, & ! F/m
    speed_of_light=3.d8, & ! m/s
    hbar=1.05d-34, & ! J/(rad/s)
    kB=1.4d-23, & ! J/K
    ! ---------------------------------------------------------------------------------------------
    ! tweezer
    tweezer_power=0.4d0, & ! Watts
    ! angle between tweezer polarization and cavity axis, as fraction of π
    ! 0.25 => π/4 = 45 degrees
    theta_tweezer_over_pi=0.25d0, &
    tweezer_waist_X=1.0 * 0.85d-6, &
    tweezer_waist_Y=1.25 * 0.85d-6, &
    ! ---------------------------------------------------------------------------------------------
    ! bead
    bead_radius=75.d-9, & ! metres
    bead_density=2200.d0, & ! kg/m3
    bead_permittivity=2.07d0, &
    ! ---------------------------------------------------------------------------------------------
    ! gas
    gas_temperature=300.d0, & ! K
    gas_pressure_mbar=1.e-6, & ! mbar
    ! ---------------------------------------------------------------------------------------------
    ! cavity
    k_cav=5.9d6, & ! 1/m
    cavity_length=0.96d-2, & ! metres
    cavity_waist=40.d-6, & ! metres
    cavity_detuning_kHz=-450.d0, & ! kHz
    cavity_finesse=40.d3, &
    Free_Spectral_Range=14.d9, & ! Hz
    theta_homodyne_over_pi=0.25d0, &
    ! ---------------------------------------------------------------------------------------------
    ! equilibrium positions
    ! node is X0 = 0.25 * lambda
    X0=0.25d0 * 1.064d-6, &
    Y0=0.0d-6, &
    Z0=0.d-6)
