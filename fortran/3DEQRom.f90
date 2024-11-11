SUBROUTINE CALCULATE_TRAP_PARAMETERS(&
    E_tweezer, &
    tweezer_zR, &
    bead_mass, &
    bead_polarisability, &
    Gamma_M, &
    half_Gamma_M, &
    E_cavity, &
    kappa_input, &
    kappa_nano)

    IMPLICIT NONE

    double precision::vacuum_permittivity, speed_of_light, hbar, kB
    double precision::tweezer_power, theta_tweezer_over_pi, tweezer_waist_X, tweezer_waist_Y
    double precision::bead_radius, bead_density, bead_permittivity
    double precision::gas_temperature, gas_pressure_mbar
    double precision::&
        k_cav, cavity_waist, cavity_length, cavity_detuning_kHz, cavity_finesse, &
        Free_Spectral_Range, theta_homodyne_over_pi
    double precision::X0, Y0, Z0
    INCLUDE 'CSCAVITY.h'

    double precision::pi, pi2, Hz, kHz
    double precision::bead_mass, bead_polarisability
    double precision::E_tweezer, tweezer_zR
    double precision::E_cavity, cavity_field_depth, cavity_volume
    double precision::omega_cav, coeff
    double precision::kappa, kappa_nano, kappa_input
    double precision::Gamma_M, half_Gamma_M

    pi = dacos(-1.d0)
    pi2 = 2.d0 * pi
    Hz = pi2
    kHz = Hz * 1.d3

    ! ---------------------------------------------------------------------------------------------
    ! tweezer
    E_tweezer = sqrt(4.* tweezer_power / (tweezer_waist_X * tweezer_waist_Y * pi * speed_of_light * vacuum_permittivity))
    tweezer_zR = tweezer_waist_X * tweezer_waist_Y * k_cav / 2.d0

    ! ---------------------------------------------------------------------------------------------
    ! bead
    bead_mass = bead_density * 4. * pi / 3. * bead_radius**3
    bead_polarisability = 4. * pi * vacuum_permittivity * (bead_permittivity - 1.) / (bead_permittivity + 2.) * bead_radius**3

    ! ---------------------------------------------------------------------------------------------
    ! gas
    ! take usual expression eg Levitated review by Li Geraci etc
    ! 1 bar = 10^ 5 pascal; gas_pressure_mbar is in mbar = 10^ 2 pascal
    ! gamma = 16 P/(pi*v*bead_density*R)
    ! v = speed of air=500 m/s
    Gamma_M = 1600. * gas_pressure_mbar / pi
    Gamma_M = Gamma_M / 500 / bead_density / bead_radius
    ! Fix of Feb.2016 our Gamma_M => Gamma_M/2!!
    ! gamma / 2 = 8*P / (pi * v * bead_density * R)
    half_Gamma_M = Gamma_M / 2.d0

    WRITE(6, *) 'gas_pressure_mbar (Hz)', gas_pressure_mbar
    WRITE(6, *) 'Gamma_M (Hz)', Gamma_M / Hz
    WRITE(6, *) 'half_Gamma_M (Hz)', half_Gamma_M / Hz

    ! ---------------------------------------------------------------------------------------------
    ! cavity
    ! add a factor of 4 here. Not in GALAX1-5 routines!!!
    cavity_volume = cavity_length * pi * cavity_waist**2 / 4.d0
    ! Depth of cavity field. Weak and unimportant for CS case
    omega_cav = speed_of_light * k_cav
    cavity_field_depth = omega_cav * bead_polarisability / 2. / cavity_volume / vacuum_permittivity
    WRITE(6, *) '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
    WRITE(6, *) 'depth of cavity field A', cavity_field_depth / pi2
    E_cavity = sqrt(hbar * omega_cav / (2. * cavity_volume * vacuum_permittivity))

    kappa_input = pi * speed_of_light / cavity_finesse / cavity_length
    coeff = k_cav * bead_polarisability / vacuum_permittivity / omega_cav**2 / pi
    kappa_nano = 4. * coeff**2 * Free_Spectral_Range * cos(k_cav * X0)**2
    kappa = kappa_nano + kappa_input

    WRITE(6, *) 'kappa_nano (Hz)', kappa_nano / Hz
    WRITE(6, *) 'kappa_input (Hz)', kappa_input / Hz
    WRITE(6, *) 'kappa (Hz)', kappa / Hz


    100 FORMAT(4D16.8)

    RETURN
END


SUBROUTINE CALCULATE_EQUILIBRIUM_PARAMETERS(&
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

    IMPLICIT NONE

    double precision::vacuum_permittivity, speed_of_light, hbar, kB
    double precision::tweezer_power, theta_tweezer_over_pi, tweezer_waist_X, tweezer_waist_Y
    double precision::bead_radius, bead_density, bead_permittivity
    double precision::gas_temperature, gas_pressure_mbar
    double precision::&
        k_cav, cavity_waist, cavity_length, cavity_detuning_kHz, cavity_finesse, &
        Free_Spectral_Range, theta_homodyne_over_pi
    double precision::X0, Y0, Z0
    INCLUDE 'CSCAVITY.h'

    integer::iwrite
    double precision::pi, pi2, Hz, kHz

    double precision::E_tweezer, tweezer_zR, theta_tweezer
    double precision::bead_polarisability, bead_mass, k_X0
    double precision::Gamma_M, half_Gamma_M
    double precision::E_cavity, kappa, half_kappa, kappa_nano, kappa_input, cavity_detuning

    ! combined physics outputs
    double precision::E_drive, omega_drive
    double precision::num_photons, alpha_real, alpha_imag
    double precision::Omega_X, Omega_Y, Omega_Z

    ! couplings
    double precision::GX, GY, GZ
    double precision::GXY, GZX, GYZ
    double precision::X_zpf, Y_zpf, Z_zpf
    double precision::cooling_rate_X, cooling_rate_Y, cooling_rate_Z

    ! useful intermediate variables
    double precision::CS_coeff
    double precision::C1, C2

    double precision::phonons_from_cooling_formula(3)
    double precision::couplings_array(6)

    iwrite = 6
    pi = dacos(-1.d0)
    pi2 = 2.d0 * pi
    Hz = pi2
    kHz = Hz * 1.d3


    ! kappa_nano is from photon scattering losses from nanoparticle, in addition to usual losses
    ! through mirrors. Not always added in
    ! in practice kappa of Antonio already includes those. But care when working out input-output
    ! as that sqrt(kappa) is about mirror allowing leaking out of light.
    kappa = kappa_nano + kappa_input
    half_kappa = 0.5d0 * kappa

    ! we will use negative omega_drive
    E_drive = -0.5d0 * bead_polarisability * E_tweezer * E_cavity * sin(theta_tweezer)
    omega_drive = E_drive / hbar
    WRITE(6, *) 'E_drive (J)'
    WRITE(6, 100) E_drive
    WRITE(6, *) 'omega_drive'
    WRITE(6, 100) omega_drive
    WRITE(6, *) 'omega_drive (Hz)'
    WRITE(6, 100) omega_drive / Hz

    ! photon field in cavity
    ! real part of photon field

    alpha_real = cavity_detuning * omega_drive * cos(k_X0) / (half_kappa**2 + cavity_detuning**2)
    WRITE(6, *) 'alpha_real = ', alpha_real
    alpha_imag = -half_kappa * omega_drive * cos(k_X0) / (half_kappa**2 + cavity_detuning**2)
    WRITE(6, *) 'alpha_imag = ', alpha_imag
    num_photons = (omega_drive * cos(k_X0))**2 / (half_kappa**2 + cavity_detuning**2)
    WRITE(6, *) 'number of photons', num_photons

    ! ADD CAVITY**2 term: usually small unless many photons
    ! C1 = 2. * bead_polarisability * E_cavity**2 * num_photons * k_cav**2 * cos(2 * k_X0) / bead_mass

    ! note that detunings include zeroth order correction for linearised versions
    ! as a first pass work out frequencies with Wk0 = 0

    Omega_X = bead_polarisability * E_tweezer**2 / bead_mass / tweezer_waist_X**2
    Omega_Y = bead_polarisability * E_tweezer**2 / bead_mass / tweezer_waist_Y**2
    Omega_Z = bead_polarisability * E_tweezer**2 / bead_mass / (2. * tweezer_zR**2)

    WRITE(6, *) 'Omega_X (Hz)'
    WRITE(6, 100) sqrt(Omega_X) / Hz
    WRITE(6, *) 'Omega_Y (Hz)'
    WRITE(6, 100) sqrt(Omega_Y) / Hz
    WRITE(6, *) 'Omega_Z (Hz)'
    WRITE(6, 100) sqrt(Omega_Z) / Hz


    ! ADD CS POTENTIAL CORRECTION to frequency squared
    CS_coeff = -E_drive / bead_mass * 2. * alpha_real * cos(k_X0)
    WRITE(6, *) 'CS_coeff'
    WRITE(6, 100) CS_coeff

    Omega_X = Omega_X + CS_coeff * ((k_cav * sin(theta_tweezer))**2 + (2. / tweezer_waist_X**2))
    Omega_Y = Omega_Y + CS_coeff * ((k_cav * cos(theta_tweezer))**2 + (2. / tweezer_waist_Y**2))
    Omega_Z = Omega_Z - 2. * E_drive / bead_mass * alpha_real * (k_cav - 1.d0 / tweezer_zR)**2 * cos(k_X0)
    ! Omega_Z = Omega_Z - CS_coeff * (k_cav - 1.d0 / tweezer_zR)**2

    Omega_X = sqrt(Omega_X)
    Omega_Y = sqrt(Omega_Y)
    Omega_Z = sqrt(Omega_Z)
    WRITE(iwrite, *) 'Frequencies WITHOUT optical spring'
    WRITE(iwrite, *) 'mech freq Omega_X (Hz) = ', Omega_X / Hz
    WRITE(iwrite, *) 'mech freq Omega_Y (Hz) = ', Omega_Y / Hz
    WRITE(iwrite, *) 'mech freq Omega_Z (Hz) = ', Omega_Z / Hz

    ! This is the end of calculation of the static frequency, corrected by co-trapping.

    ! Optomechanical couplings
    X_zpf = sqrt(hbar / (2.d0 * bead_mass * Omega_X))
    Y_zpf = sqrt(hbar / (2.d0 * bead_mass * Omega_Y))
    Z_zpf = sqrt(hbar / (2.d0 * bead_mass * Omega_Z))

    GX = omega_drive * k_cav * X_zpf * sin(theta_tweezer) * sin(k_X0)
    GY = omega_drive * k_cav * Y_zpf * cos(theta_tweezer) * sin(k_X0)
    GZ = -omega_drive * (k_cav - 1.d0 / tweezer_zR) * Z_zpf * cos(k_X0)
    WRITE(iwrite, *) 'GX, GY, GZ (Hz)', GX / Hz, GY / Hz, GZ / Hz

    GXY = omega_drive * k_cav * X_zpf * k_cav * Y_zpf * alpha_real * sin(2. * theta_tweezer) * cos(k_X0)
    GZX = 2.d0 * omega_drive * (k_cav - 1.d0 / tweezer_zR) * Z_zpf * k_cav * X_zpf * alpha_imag * sin(k_X0) * sin(theta_tweezer)
    GYZ = 2.d0 * omega_drive * (k_cav - 1.d0 / tweezer_zR) * Z_zpf * k_cav * Y_zpf * alpha_imag * sin(k_X0) * cos(theta_tweezer)
    WRITE(iwrite, *) 'GXY, GYZ, GZX (Hz)', GXY / Hz, GYZ / Hz, GZX / Hz

    couplings_array(1) = GX
    couplings_array(2) = GY
    couplings_array(3) = GZ
    couplings_array(4) = GXY
    couplings_array(5) = GYZ
    couplings_array(6) = GZX

    CALL CALCULATE_COOLING_RATES(&
        cavity_detuning, &
        kappa, &
        GX, &
        GY, &
        GZ, &
        Omega_X, &
        Omega_Y, &
        Omega_Z, &
        cooling_rate_X, &
        cooling_rate_Y, &
        cooling_rate_Z)

    WRITE(iwrite, *) 'X optomechanical cooling rate, mechanical damping, kX0'
    WRITE(iwrite, 100) cooling_rate_X, Gamma_M, k_X0
    WRITE(iwrite, *) 'Number of phonons: at room gas_temperature; at equil'
    C1 = kB * gas_temperature / hbar / Omega_X
    C2 = C1 * Gamma_M / (abs(cooling_rate_X) + Gamma_M)
    WRITE(iwrite, 100) C1, abs(C2)
    phonons_from_cooling_formula(1) = abs(C2)
    WRITE(iwrite,*) 'X gas_temperature: at room gas_temperature;at equil'
    WRITE(iwrite,*) gas_temperature, gas_temperature * Gamma_M / (Gamma_M + abs(cooling_rate_X))

    ! Y
    WRITE(iwrite, *) 'y optomechanical cooling rate, mechanical damping, Y0'
    WRITE(iwrite, 100) cooling_rate_Y, Gamma_M, y0
    WRITE(iwrite,*) 'Number of phonons: at room gas_temperature;at equil'
    C1 = kB * gas_temperature / hbar / Omega_Y
    C2 = C1 * Gamma_M / (abs(cooling_rate_Y) + Gamma_M)
    WRITE(iwrite, 100) C1, abs(C2)
    WRITE(iwrite, *) 'Y gas_temperature: at room gas_temperature; at equil'
    WRITE(iwrite, *) gas_temperature, gas_temperature * Gamma_M / (Gamma_M + abs(cooling_rate_Y))
    phonons_from_cooling_formula(2) = abs(C2)

    ! Z
    WRITE(iwrite, *) 'z optomechanical cooling rate, mechanical damping, Z0'
    WRITE(iwrite, 100) cooling_rate_Z, Gamma_M, z0
    WRITE(iwrite, *) 'Number of phonons: at room gas_temperature; at equil'
    C1 = kB * gas_temperature / hbar / Omega_Z
    C2 = C1 * Gamma_M / (abs(cooling_rate_Z) + Gamma_M)
    WRITE(iwrite, 100) C1, abs(C2)
    WRITE(iwrite, *) 'Z gas_temperature: at room gas_temperature; at equil'
    WRITE(iwrite, *) gas_temperature, gas_temperature * Gamma_M / (Gamma_M + abs(cooling_rate_Z))
    phonons_from_cooling_formula(3) = abs(C2)

    ! work out self energy and optical spring
    WRITE(iwrite, *) 'OPTICAL SPRING EFFECT'
    WRITE(iwrite, *) '============================='
    CALL CALCULATE_OPTICAL_DAMPING(&
        bead_mass, &
        k_X0, &
        Gamma_M, &
        half_Gamma_M, &
        cavity_detuning, &
        kappa, &
        GX, &
        GY, &
        GZ, &
        Omega_X, &
        Omega_Y, &
        Omega_Z, &
        cooling_rate_X, &
        cooling_rate_Y, &
        cooling_rate_Z)
    WRITE(iwrite, *) '============================='

    100 FORMAT(4D16.8)

    RETURN
END

SUBROUTINE CALCULATE_COOLING_RATES(&
    cavity_detuning, &
    kappa, &
    GX, &
    GY, &
    GZ, &
    Omega_X, &
    Omega_Y, &
    Omega_Z, &
    cooling_rate_X, &
    cooling_rate_Y, &
    cooling_rate_Z)

    ! Note that cavity_detuning effectively is
    ! cavity_detuning + AOPT. It is corrected by zeroth order optical contribution
    ! in the linearised case

    IMPLICIT NONE

    double precision::vacuum_permittivity, speed_of_light, hbar, kB
    double precision::tweezer_power, theta_tweezer_over_pi, tweezer_waist_X, tweezer_waist_Y
    double precision::bead_radius, bead_density, bead_permittivity
    double precision::gas_temperature, gas_pressure_mbar
    double precision::&
        k_cav, cavity_waist, cavity_length, cavity_detuning_kHz, cavity_finesse, &
        Free_Spectral_Range, theta_homodyne_over_pi
    double precision::X0, Y0, Z0
    INCLUDE 'CSCAVITY.h'

    double precision::pi, pi2, Hz, kHz
    double precision::kappa, half_kappa, cavity_detuning

    double precision::GX, GY, GZ
    double precision::Omega_X, Omega_Y, Omega_Z
    double precision::C_plus, C_minus
    double precision::cooling_rate_X, cooling_rate_Y, cooling_rate_Z

    pi = dacos(-1.d0)
    pi2 = 2.d0 * pi
    Hz = pi2
    kHz = Hz * 1.d3

    half_kappa = 0.5d0 * kappa

    ! X cooling
    ! for maimag_nummum cooling sin2kx_0 = 1
    ! C1 = sin(2.d0 * WK0)**2 * 2. * half_kappa * k_cav**2 * A**2 * ASQ * hbar / bead_mass / 2.d0 / Omega_X
    C_plus = (cavity_detuning + Omega_X)**2 + half_kappa**2
    C_plus = 1.d0 / C_plus
    C_minus = (cavity_detuning - Omega_X)**2 + half_kappa**2
    C_minus = 1.d0 / C_minus
    cooling_rate_X = -GX**2 * kappa * (C_plus - C_minus)

    ! Y cooling
    C_plus = (cavity_detuning + Omega_Y)**2 + half_kappa**2
    C_plus = 1.d0 / C_plus
    C_minus = (cavity_detuning - Omega_Y)**2 + half_kappa**2
    C_minus = 1.d0 / C_minus
    cooling_rate_Y = -GY**2 * kappa * (C_plus - C_minus)

    ! Z cooling
    ! here we neglect opto shift!
    C_plus = (cavity_detuning + Omega_Z)**2 + half_kappa**2
    C_plus = 1.d0 / C_plus
    C_minus = (cavity_detuning - Omega_Z)**2 + half_kappa**2
    C_minus = 1.d0 / C_minus
    cooling_rate_Z = -GZ**2 * kappa * (C_plus - C_minus)

    100 FORMAT(4D16.8)

    RETURN
END


SUBROUTINE CALCULATE_OPTICAL_DAMPING(&
    bead_mass, &
    k_X0, &
    Gamma_M, &
    half_Gamma_M, &
    cavity_detuning, &
    kappa, &
    GX, &
    GY, &
    GZ, &
    Omega_X, &
    Omega_Y, &
    Omega_Z, &
    cooling_rate_X, &
    cooling_rate_Y, &
    cooling_rate_Z)

    ! Note that cavity_detuning effectively is
    ! cavity_detuning + AOPT. It is corrected by zeroth order optical contribution
    ! in the linearised case

    IMPLICIT NONE

    double precision::vacuum_permittivity, speed_of_light, hbar, kB
    double precision::tweezer_power, theta_tweezer_over_pi, tweezer_waist_X, tweezer_waist_Y
    double precision::bead_radius, bead_density, bead_permittivity
    double precision::gas_temperature, gas_pressure_mbar
    double precision::&
        k_cav, cavity_waist, cavity_length, cavity_detuning_kHz, cavity_finesse, &
        Free_Spectral_Range, theta_homodyne_over_pi
    double precision::X0, Y0, Z0
    INCLUDE 'CSCAVITY.h'

    double precision::cavity_detuning
    double precision::bead_mass, k_X0
    double precision::Gamma_M, half_Gamma_M
    double precision::GX, GY, GZ
    double precision::Omega_X, Omega_Y, Omega_Z
    double precision::omega
    double precision::kappa, half_kappa

    double complex::imag_num
    double precision::pi, pi2, Hz, kHz

    double precision::norm
    double precision::chi_real, chi_imag
    double precision::cooling_rate_X, cooling_rate_Y, cooling_rate_Z

    double complex::chi_C_X, chi_C_star_X, chi_M_X
    double complex::chi_C_Y, chi_C_star_Y, chi_M_Y
    double complex::self_energy_X, self_energy_Y
    double complex::correction_2D_X, correction_2D_Y
    double precision::spring_X, spring_Y

    imag_num = cmplx(0.d0, 1.d0)
    pi = dacos(-1.d0)
    pi2 = 2.d0 * pi
    Hz = pi2
    kHz = Hz * 1.d3


    half_kappa = 0.5d0 * kappa

    ! optical suscept at omega = Omega_X
    omega = Omega_X
    ! chi_C(Omega_X)
    norm = half_kappa**2 + (omega + cavity_detuning)**2
    chi_real = half_kappa / norm
    chi_imag = (omega + cavity_detuning) / norm
    chi_C_X = cmplx(chi_real, chi_imag)
    ! chi_C*(-Omega_X)
    norm = half_kappa**2 + (-omega + cavity_detuning)**2
    chi_real = half_kappa / norm
    chi_imag = (-omega + cavity_detuning) / norm
    chi_C_star_X = cmplx(chi_real, -chi_imag)

    ! optical suscept at omega = Omega_Y
    omega = Omega_Y
    ! chi_C(Omega_Y)
    norm = half_kappa**2 + (omega + cavity_detuning)**2
    chi_real = half_kappa / norm
    chi_imag = (omega + cavity_detuning) / norm
    chi_C_Y = cmplx(chi_real, chi_imag)
    ! chi_C*(-Omega_Y)
    norm = half_kappa**2 + (-omega + cavity_detuning)**2
    chi_real = half_kappa / norm
    chi_imag = (-omega + cavity_detuning) / norm
    chi_C_star_Y = cmplx(chi_real, -chi_imag)

    ! mechanical susceptibility of X at the frequency of Y??? NOT SURE: CHECK!!!
    ! chi_X(Omega_Y)
    omega = Omega_Y
    norm = (half_Gamma_M)**2 + (omega - Omega_X)**2
    chi_real = (half_Gamma_M) / norm
    chi_imag = (omega - Omega_X) / norm
    chi_M_X = cmplx(chi_real, chi_imag)

    ! mechanical susceptibility of Y at the frequency of X??? NOT SURE: CHECK!!!
    ! chi_X(Omega_Y)
    omega = Omega_X
    norm = (half_Gamma_M)**2 + (omega - Omega_Y)**2
    chi_real = (half_Gamma_M) / norm
    chi_imag = (omega - Omega_Y) / norm
    chi_M_Y = cmplx(chi_real, chi_imag)

    self_energy_X = -imag_num * 2. * GX**2 * (chi_C_X - chi_C_star_X) * bead_mass * Omega_X
    self_energy_Y = -imag_num * 2. * GY**2 * (chi_C_Y - chi_C_star_Y) * bead_mass * Omega_Y

    ! For 2D we can add the additional correction
    ! NOTE: are these X,Y the wrong way around?
    correction_2D_X = 1.d0 + GY**2 * (chi_C_Y - chi_C_star_Y) * chi_M_Y
    correction_2D_Y = 1.d0 + GX**2 * (chi_C_X - chi_C_star_X) * chi_M_X
    ! LEAVE OUT 2D correction for now
    self_energy_X = self_energy_X / correction_2D_X
    self_energy_Y = self_energy_Y / correction_2D_Y

    cooling_rate_X = aimag(self_energy_X) / bead_mass / Omega_X
    spring_X = real(self_energy_X) / bead_mass
    spring_X = Omega_X**2 + spring_X
    spring_X = sqrt(spring_X)

    cooling_rate_Y = aimag(self_energy_Y) / bead_mass / Omega_Y
    spring_Y = real(self_energy_Y) / bead_mass
    spring_Y = Omega_Y**2 + spring_Y
    spring_Y = sqrt(spring_Y)

    WRITE(6, *) 'k_X0, cooling_rate_X, Omega_X no spring, Omega_X + spring (kHz)'
    WRITE(6, 100) k_X0 / Hz, cooling_rate_X / kHz, Omega_X / kHz, spring_X / kHz
    WRITE(6, *) 'k_X0, cooling_rate_Y, Omega_Y no spring, Omega_Y + spring (kHz)'
    WRITE(6, 100) k_X0 / Hz, cooling_rate_Y / kHz, Omega_Y / kHz, spring_Y / kHz

    WRITE(12, 100) k_X0 / Hz, Omega_X / kHz, spring_X / kHz, Omega_Y / kHz, spring_Y / kHz
    100 FORMAT(5F16.4)

    RETURN
END
