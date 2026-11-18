! Calculates quantum noise spectra.
! uses real mean fields
! works out position and momentum optical quadrature

! Gamma_M = mechanical damping due to gas, also noise from fluctuation dissipation.
! NB no intent(in) or intent(out) used - add these later!!

IMPLICIT NONE
! --------------------------------------------------------------------------------------------------
integer::num_traps, Cartesian_axes_per_trap, num_operators, num_dimensions
INCLUDE 'Nparticle.h'

double precision::pi, hbar, speed_of_light, kB, vacuum_permittivity
double precision::Hz, kHz
double precision::tweezer_power, theta_tweezer_over_pi, tweezer_waist_X, tweezer_waist_Y
double precision::bead_radius, bead_density, bead_permittivity
double precision::gas_temperature, gas_pressure_mbar
double precision::&
    k_cav, cavity_waist, cavity_length, cavity_detuning_kHz, cavity_finesse, &
    Free_Spectral_Range, theta_homodyne_over_pi
double precision::X0, Y0, Z0
INCLUDE 'CSCAVITY.h'

! --------------------------------------------------------------------------------------------------
double precision::cavity_photons
double precision::X0_step
integer::num_omega_samples

PARAMETER(&
    X0_step=0.01d0 * 1.064d-6, &
    num_omega_samples=2000)

double precision::k_X0

double precision::Omega_X, Omega_Y, Omega_Z

! --------------------------------------------------------------------------------------------------
! calculated parameters
double precision::E_tweezer, tweezer_zR, theta_tweezer
double precision::bead_mass, bead_polarisability
double precision::E_cavity, cavity_detuning, kappa_nano, kappa_input, half_kappa, kappa, theta_homodyne
double precision::Gamma_M, half_Gamma_M
double precision::E_drive
double precision::kBT_h

integer::omega_index
double precision::omega_range_value, omega_range_max, omega_step
double precision::homodyne_spectrum_symmetrised, homodyne_value
double precision::heterodyne_value, heterodyne_value_CBM
double precision::PSD_value_CBM
double precision, DIMENSION(num_omega_samples)::&
    omega_range, &
    homodyne_array, &
    X1_PSD_array, &
    X2_PSD_array, &
    X3_PSD_array, &
    Y1_PSD_array, &
    Y2_PSD_array, &
    Y3_PSD_array, &
    Z1_PSD_array, &
    Z2_PSD_array, &
    Z3_PSD_array

double precision::&
    X1_PSD_value, &
    X2_PSD_value, &
    X3_PSD_value, &
    Y1_PSD_value, &
    Y2_PSD_value, &
    Y3_PSD_value, &
    Z1_PSD_value, &
    Z2_PSD_value, &
    Z3_PSD_value

double precision::&
    X1_PSD_value_symmetrised, &
    X2_PSD_value_symmetrised, &
    X3_PSD_value_symmetrised, &
    Y1_PSD_value_symmetrised, &
    Y2_PSD_value_symmetrised, &
    Y3_PSD_value_symmetrised, &
    Z1_PSD_value_symmetrised, &
    Z2_PSD_value_symmetrised, &
    Z3_PSD_value_symmetrised

double precision::noise_photons
double precision::phonons_via_PSD_area, PSD_area
double precision::phonons_via_formula

double precision::mech_freq_array(num_traps, Cartesian_axes_per_trap)
double precision::quanta_per_operator(num_operators)
double precision::phase_per_tweezer(num_traps)
double precision::Gamma_M_array(num_traps, Cartesian_axes_per_trap)
double precision::PSD_array(num_traps, Cartesian_axes_per_trap)

double precision::light_couplings_array(num_traps, Cartesian_axes_per_trap)
double precision::mech_couplings_array(num_traps, Cartesian_axes_per_trap, Cartesian_axes_per_trap)

! read PARAMETER file with input values
double precision::couplings_array(6)

double precision::phonons_from_cooling_formula_X
double precision::phonons_from_cooling_formula_Y
double precision::phonons_from_cooling_formula_Z
double precision::phonons_from_cooling_formula(3)
double precision::phonons_via_area(3)


! Here plot optical field amplitudes
OPEN(8, file="FTanXYZ.dat", status="unknown")
OPEN(9, file="FTCBM.dat", status="unknown")
OPEN(10, file="FTanOPT.dat", status="unknown")
OPEN(12, file="OMEGAS.dat", status="unknown")
OPEN(13, file="Omegas1.dat", status="unknown")
OPEN(14, file="Omegas2.dat", status="unknown")

WRITE(6, *) 'detuning (kHz)', cavity_detuning_kHz

cavity_detuning = cavity_detuning_kHz * kHz
k_X0 = k_cav * X0

theta_tweezer = theta_tweezer_over_pi * pi
WRITE(6, *) 'theta_tweezer / pi', theta_tweezer / pi

theta_homodyne = theta_homodyne_over_pi * pi
WRITE(6, *) 'theta_homodyne / pi', theta_homodyne / pi
WRITE(6, *) 'k_X0', k_X0

! --------------------------------------------------------------------------------------------------
! ONE-TIME PARAMETER CALCULATION
! --------------------------------------------------------------------------------------------------
! works out parameters for trap 1 only
CALL CALCULATE_TRAP_PARAMETERS(&
    E_tweezer, &
    tweezer_zR, &
    bead_mass, &
    bead_polarisability, &
    Gamma_M, &
    half_Gamma_M, &
    E_cavity, &
    kappa_input, &
    kappa_nano)

CALL CALCULATE_EQUILIBRIUM_PARAMETERS(&
    E_tweezer, &
    tweezer_zR, &
    theta_tweezer, &
    bead_polarisability, &
    bead_mass, &
    k_X0, &
    Gamma_M, &
    half_Gamma_M, &
    E_cavity, &
    cavity_detuning, &
    kappa_nano, &
    kappa_input, &
    Omega_X, &
    Omega_Y, &
    Omega_Z, &
    couplings_array, &
    phonons_from_cooling_formula)

! kBT_h is for quanta_per_operator
! X1 phonons = kB * gas_temperature / (hbar * Omega_X1)
kBT_h = kB * gas_temperature / hbar

! these arrays are for all traps
CALL FILL_PARAMETER_ARRAYS(&
    Gamma_M, &
    kBT_h, &
    phase_per_tweezer, &
    mech_freq_array, &
    Gamma_M_array, &
    light_couplings_array, &
    mech_couplings_array, &
    quanta_per_operator)

WRITE(6, *) 'Omega_X1, Omega_Y1, Omega_X2, Omega_Y2 (kHz)'
WRITE(6, 101) mech_freq_array(1, 1) / kHz, mech_freq_array(1, 2) / kHz, mech_freq_array(2, 1) / kHz, mech_freq_array(2, 2) / kHz


kappa = kappa_input + kappa_nano
half_kappa = 0.5d0 * kappa
WRITE(6, *) 'kappa, half_kappa (kHz)', kappa / kHz, half_kappa / kHz

E_drive = 0.5d0 * bead_polarisability * E_tweezer * E_cavity * sin(theta_tweezer)
cavity_photons = (E_drive * cos(k_X0) / hbar)**2 / (half_kappa**2 + cavity_detuning**2)
WRITE(6, *) 'cavity photons', cavity_photons


! --------------------------------------------------------------------------------------------------
! SPECTRUM CALCULATION LOOP OVER FREQUENCY
! --------------------------------------------------------------------------------------------------
! shot noise
noise_photons = 0.d0
! open loop over frequency for noise spectra
! e.g. S_XX(omega) = FT(autocorrelation <X(t).X^T(t+tau)>)
omega_range_max = 5.d5 * Hz
omega_step = omega_range_max / num_omega_samples
WRITE(6, *) 'omega_step (kHz)', omega_step / kHz

X1_PSD_array = 0.d0
X2_PSD_array = 0.d0
X3_PSD_array = 0.d0

Y1_PSD_array = 0.d0
Y2_PSD_array = 0.d0
Y3_PSD_array = 0.d0

homodyne_array = 0.d0
DO omega_index=1, num_omega_samples
    omega_range_value = -omega_range_max + 2 * (omega_index - 1) * omega_step
    ! store frequency for integration
    omega_range(omega_index) = omega_range_value

    ! work out PSD of homodyne
    homodyne_spectrum_symmetrised = 0.d0

    ! symmetrised
    X1_PSD_value_symmetrised = 0.d0
    X2_PSD_value_symmetrised = 0.d0
    X3_PSD_value_symmetrised = 0.d0

    Y1_PSD_value_symmetrised = 0.d0
    Y2_PSD_value_symmetrised = 0.d0
    Y3_PSD_value_symmetrised = 0.d0

    Z1_PSD_value_symmetrised = 0.d0
    Z2_PSD_value_symmetrised = 0.d0
    Z3_PSD_value_symmetrised = 0.d0

    ! PSD_value is S_QQ by linear amplifier
    ! PSD_value_CBM is S_QQ by CBM formula

    ! heterodyne_value is heterodyne by linear amplifier
    ! heterodyne_value_CBM is heterodyne by CBM formula

    ! work out  for negative frequency for symmetrisation

    IF ((omega_range_value / kHz).GE.297.95.AND.(omega_range_value / kHz).LE.298.05) THEN
        WRITE(6, *) '==========================================================='
        WRITE(6, *) 'NParticle.f90 DEBUG (before CALCULATE_NOISE_SPECTRA)'
        WRITE(6, *) '==========================================================='
        WRITE(6, *) 'cavity_detuning (kHz)', cavity_detuning / kHz
        WRITE(6, *) 'kappa (kHz)', kappa / kHz
        WRITE(6, *) 'theta_homodyne (π)', theta_homodyne / pi
        WRITE(6, *) 'noise_photons', noise_photons
        WRITE(6, *) 'phase_per_tweezer', phase_per_tweezer
        WRITE(6, *) 'mech_freq_array', mech_freq_array
        WRITE(6, *) 'Gamma_M_array', Gamma_M_array
        WRITE(6, *) 'quanta_per_operator', quanta_per_operator
        WRITE(6, *) 'light_couplings_array', light_couplings_array
        WRITE(6, *) 'mech_couplings_array', mech_couplings_array
    ENDIF

    CALL CALCULATE_NOISE_SPECTRA(&
        -omega_range_value, &
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

    X1_PSD_value = PSD_array(1, 1)
    X2_PSD_value = PSD_array(2, 1)
    X3_PSD_value = PSD_array(3, 1)

    Y1_PSD_value = PSD_array(1, 2)
    Y2_PSD_value = PSD_array(2, 2)
    Y3_PSD_value = PSD_array(3, 2)

    ! update homodyne and syomega_index disp.
    homodyne_spectrum_symmetrised = homodyne_spectrum_symmetrised + homodyne_value

    ! work out same for positive and symmetrise homodyne
    CALL CALCULATE_NOISE_SPECTRA(&
        omega_range_value, &
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

    ! NOTE: this seems to just override the previous calculation?
    X1_PSD_value = PSD_array(1, 1)
    X2_PSD_value = PSD_array(2, 1)
    X3_PSD_value = PSD_array(3, 1)

    Y1_PSD_value = PSD_array(1, 2)
    Y2_PSD_value = PSD_array(2, 2)
    Y3_PSD_value = PSD_array(3, 2)

    homodyne_spectrum_symmetrised = 0.5 * (homodyne_spectrum_symmetrised + homodyne_value)

    X1_PSD_value_symmetrised = X1_PSD_value_symmetrised + X1_PSD_value
    X2_PSD_value_symmetrised = X2_PSD_value_symmetrised + X2_PSD_value
    X3_PSD_value_symmetrised = X3_PSD_value_symmetrised + X3_PSD_value

    Y1_PSD_value_symmetrised = Y1_PSD_value_symmetrised + Y1_PSD_value
    Y2_PSD_value_symmetrised = Y2_PSD_value_symmetrised + Y2_PSD_value
    Y3_PSD_value_symmetrised = Y3_PSD_value_symmetrised + Y3_PSD_value

    omega_range_value = omega_range_value / kHz
    IF((omega_range_value.GE.0.d0).AND.(omega_range_value.LE.400.d0)) THEN
        ! 8 => FTanXYZ.dat
        WRITE(8, 101) omega_range_value, X1_PSD_value, Y1_PSD_value, X2_PSD_value, Y2_PSD_value, X3_PSD_value, Y3_PSD_value
        ! formula
        ! 9 => FTCBM.dat
        WRITE(9, 101) omega_range_value, heterodyne_value_CBM, PSD_value_CBM
        ! 10 => FTanOPT.dat
        WRITE(10, 101) omega_range_value, heterodyne_value, homodyne_spectrum_symmetrised
    ENDIF

    ! NOTE: this is only taking the positive frequency calculation?
    X1_PSD_array(omega_index) = X1_PSD_value
    X2_PSD_array(omega_index) = X2_PSD_value
    X3_PSD_array(omega_index) = X3_PSD_value

    Y1_PSD_array(omega_index) = Y1_PSD_value
    Y2_PSD_array(omega_index) = Y2_PSD_value
    Y3_PSD_array(omega_index) = Y3_PSD_value

    homodyne_array(omega_index) = homodyne_spectrum_symmetrised
ENDDO

! --------------------------------------------------------------------------------------------------
! PHONON CALCULATIONS
! --------------------------------------------------------------------------------------------------
phonons_via_formula = (Omega_X + cavity_detuning)**2 + half_kappa**2
phonons_via_formula = phonons_via_formula / 4 / Omega_X / (-cavity_detuning)
WRITE(6, *) 'X: back action limited phonons', phonons_via_formula

phonons_via_formula = (Omega_Y + cavity_detuning)**2 + half_kappa**2
phonons_via_formula = phonons_via_formula / 4 / Omega_Y / (-cavity_detuning)
WRITE(6, *) 'Y: back action limited phonons', phonons_via_formula

phonons_via_formula = (Omega_Z + cavity_detuning)**2 + half_kappa**2
phonons_via_formula = phonons_via_formula / 4 / Omega_Z / (-cavity_detuning)
WRITE(6, *) 'Z: back action limited phonons', phonons_via_formula

! TEMP = area * hbar * Omega_M / kB
CALL AREA_UNDER_PSD(&
    num_omega_samples, &
    omega_range, &
    X1_PSD_array, &
    PSD_area)

phonons_from_cooling_formula_X = phonons_from_cooling_formula(1)
! Area phonons_via_PSD_area corresponds to 2n+1 so convert to get n
phonons_via_PSD_area = 0.5d0 * (PSD_area - 1.d0)
WRITE(6, *) 'X1 phonons from formula, from SXX1 FT'
WRITE(6, 101) phonons_from_cooling_formula_X, phonons_via_PSD_area

CALL AREA_UNDER_PSD(&
    num_omega_samples, &
    omega_range, &
    Y1_PSD_array, &
    PSD_area)
phonons_from_cooling_formula_Y = phonons_from_cooling_formula(2)
phonons_via_PSD_area = 0.5d0 * (PSD_area - 1.d0)
WRITE(6, *) 'Y1 phonons from formula, from SYY1 FT'
WRITE(6, 101) phonons_from_cooling_formula_Y, phonons_via_PSD_area

101 FORMAT(7E14.7)

STOP
END

FUNCTION OPTOMECHANICAL_COOLING_RATE(&
    Omega_M, &
    cavity_detuning, &
    kappa, &
    light_coupling)
    ! """
    ! Function to evaluate optomechanical cooling formula for single particle
    ! """

    IMPLICIT NONE

    double precision::Omega_M
    double precision::cavity_detuning, kappa, half_kappa
    double precision::light_coupling
    double precision::C_plus, C_minus
    double precision::OPTOMECHANICAL_COOLING_RATE

    half_kappa = kappa / 2.

    C_plus = (cavity_detuning + Omega_M)**2 + half_kappa**2
    C_plus = 1.d0 / C_plus
    C_minus = (cavity_detuning - Omega_M)**2 + half_kappa**2
    C_minus = 1.d0 / C_minus

    OPTOMECHANICAL_COOLING_RATE = -kappa * light_coupling**2 * (C_plus - C_minus)

    RETURN
END

SUBROUTINE AREA_UNDER_PSD(&
    num_points, &
    omega_range, &
    PSD_array, &
    area)
    ! """
    ! Normalise area under PSDs
    ! """

    IMPLICIT NONE

    integer::ii, num_points

    double precision::pi, Hz
    double precision::omega_step, trapeze_base, area
    double precision::PSD_array(num_points)
    double precision, DIMENSION(num_points)::omega_range

    pi = dacos(-1.d0)
    Hz = pi + pi

    ! integrate the position spectrum of bead
    ! quick hack - use trapezoidal rule - improve later
    omega_step = abs(omega_range(2) - omega_range(1))
    trapeze_base = omega_step / Hz
    area = 0.d0

    DO ii=1, num_points - 1
        area = area + 0.5d0 * (PSD_array(ii) + PSD_array(ii + 1))
    ENDDO
    area = area * trapeze_base

    RETURN
END

