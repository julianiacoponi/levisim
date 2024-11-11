! Work our vectors of X and optical fields but in terms of noise operators
SUBROUTINE CALCULATE_NOISE_VECTORS(&
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

    IMPLICIT NONE

    integer::num_traps, Cartesian_axes_per_trap, num_operators
    INCLUDE 'Nparticle.h'

    double precision::omega, pi, pi2, Hz, kHz

    integer::num_dimensions
    integer::ii, jj, kk
    integer::trap, axis
    integer::trap_A, trap_B
    integer::count_1, count_2

    double precision::phi_AB, trap_phase, coeff
    double precision::sqrt_kappa, sqrt_Gamma_M, Gamma_M, half_Gamma_M, kappa

    double precision::phase_per_tweezer(num_traps)

    ! arrays with dimensions of (#traps x #axes)
    double precision::&
        light_couplings_array(num_traps, Cartesian_axes_per_trap), &
        Gamma_M_array(num_traps, Cartesian_axes_per_trap)
    double complex::&
        eta_M_array(num_traps, Cartesian_axes_per_trap), &
        chi_M_array(num_traps, Cartesian_axes_per_trap), &
        chi_M_star_array(num_traps, Cartesian_axes_per_trap), &
        Beta_M(num_traps, Cartesian_axes_per_trap), &
        M_norm_array(num_traps, Cartesian_axes_per_trap)

    ! mechanical susceptibilities
    double complex::chi_C, chi_C_star, M_norm_CBM, eta_M_total, Beta_M_CBM, eta_phi_AB


    ! noise vectors
    double complex::X_noise_vector(num_traps, Cartesian_axes_per_trap, num_operators)
    double complex::X1D_noise_vector(num_traps, Cartesian_axes_per_trap, num_operators)

    double precision::mech_couplings_array(num_traps, Cartesian_axes_per_trap, Cartesian_axes_per_trap)
    double complex::G_coeff(num_traps, Cartesian_axes_per_trap, num_traps, Cartesian_axes_per_trap)
    double complex::Rjk_array(num_traps, Cartesian_axes_per_trap, num_traps, Cartesian_axes_per_trap)
    double complex::hybridisation_solution_array(num_traps, Cartesian_axes_per_trap, num_traps, Cartesian_axes_per_trap)

    ! hybridisation matrix = inv(derangement matrix)
    double complex::derangement_matrix(num_traps * Cartesian_axes_per_trap, num_traps * Cartesian_axes_per_trap)
    double complex::hybridisation_matrix(num_traps * Cartesian_axes_per_trap, num_traps * Cartesian_axes_per_trap)
    double complex::DEBUG_test_of_matrix_inversion(num_traps * Cartesian_axes_per_trap, num_traps * Cartesian_axes_per_trap)

    double complex::imag_num, one
    double complex::eta_0, exp_iphi_AB, i_chi_C, i_chi_C_star

    double complex, INTENT(out)::a_hat(num_operators), a_dag(num_operators)
    double complex, INTENT(out)::a_hat_CBM(num_operators), a_dag_CBM(num_operators)
    double complex, INTENT(out)::CBM_noise_vector(num_operators)

    pi = dacos(-1.d0)
    pi2 = 2 * pi
    Hz = pi2
    kHz = 1.d3 * Hz

    num_dimensions = num_traps * Cartesian_axes_per_trap
    Gamma_M = Gamma_M_array(1, 1)

    ! zero arrays
    G_coeff = 0.d0
    eta_M_array = 0.d0
    M_norm_array = 0.d0
    Beta_M = 0.d0
    hybridisation_solution_array = 0.d0
    derangement_matrix = 0.d0
    hybridisation_matrix = 0.d0
    X_noise_vector = 0.d0
    a_hat = complex(0.d0, 0.d0)
    a_dag = complex(0.d0, 0.d0)
    a_hat_CBM = complex(0.d0, 0.d0)
    a_dag_CBM = complex(0.d0, 0.d0)
    CBM_noise_vector = complex(0.d0, 0.d0)

    half_Gamma_M = Gamma_M / 2.
    ! zero arrays
    X_noise_vector = 0.d0

    X1D_noise_vector = complex(0.d0, 0.d0)
    hybridisation_solution_array = 0.d0

    ! i
    imag_num = complex(0.d0, 1.d0)
    one = complex(1.d0, 0.0d0)
    sqrt_kappa = sqrt(kappa)
    sqrt_Gamma_M = sqrt(Gamma_M)

    ! SUSCEPTIBILITIES
    eta_0 = chi_C - chi_C_star

    ! TOTAL SUSCEPTIBILITY
    eta_M_total = 0.d0
    DO trap=1, num_traps
        DO axis=1, Cartesian_axes_per_trap
            eta_M_array(trap, axis) = chi_M_array(trap, axis) - chi_M_star_array(trap, axis)
            eta_M_total = eta_M_total + eta_M_array(trap, axis)
        ENDDO
    ENDDO

    ! BA norm term for CBM. Assume all g same = light_couplings_array(1, 1)
    M_norm_CBM = 1.d0 + light_couplings_array(1, 1)**2 * eta_M_total * eta_0
    Beta_M_CBM = imag_num * sqrt_kappa * eta_M_total * light_couplings_array(1, 1)
    DO trap=1, num_traps
        DO axis=1, Cartesian_axes_per_trap
            M_norm_array(trap, axis) = 1.d0 + light_couplings_array(trap, axis)**2 * eta_M_array(trap, axis) * eta_0
            Beta_M(trap, axis) = imag_num * sqrt_kappa * eta_M_array(trap, axis) * light_couplings_array(trap, axis)
        ENDDO
    ENDDO

    IF ((omega / kHz).GE.297.95.AND.(omega / kHz).LE.298.05) THEN
        WRITE(6, *) '==========================================================='
        WRITE(6, *) 'NoiseVec2D.f90 DEBUG'
        WRITE(6, *) '==========================================================='
        WRITE(6, *) 'omega (kHz)', omega / kHz
        WRITE(6, *) 'kappa (kHz)', kappa / kHz
        WRITE(6, *) 'phase_per_tweezer', phase_per_tweezer
        WRITE(6, *) 'light_couplings_array (kHz)', light_couplings_array / kHz
        WRITE(6, *) 'eta_0', eta_0
        WRITE(6, *) 'eta_M_array', eta_M_array
        WRITE(6, *) 'eta_M_total', eta_M_total

        ! WRITE(6, *) 'M_norm_CBM', M_norm_CBM
        ! WRITE(6, *) 'Beta_M_CBM', Beta_M_CBM

        WRITE(6, *) 'M_norm_array', M_norm_array
        WRITE(6, *) 'Beta_M', Beta_M
    ENDIF


    ! cavity mediated coupling- Combines direct and indirect coupling
    DO 10 trap_A=1, num_traps
        DO 11 trap_B=1, num_traps
            IF(trap_A == trap_B) THEN
                    ! intraparticle x-y coupling  X1-Y1 and X2-Y2
                    G_coeff(trap_A, 1, trap_A, 2) = &
                        mech_couplings_array(trap_A, 1, 2) &
                        + imag_num * eta_0 * light_couplings_array(trap_A, 1) * light_couplings_array(trap_A, 2)
                    ! intraparticle y-x coupling  Y1-X1 and Y2-X2
                    G_coeff(trap_A, 2, trap_A, 1) = G_coeff(trap_A, 1, trap_A, 2)

            ELSEIF(trap_A.NE.trap_B) THEN
                ! trap_A .neq. trap_B: COUPLING BETWEEN TWO PARTICLES
                ! INCLUDE PHASES
                ! phi_AB = 0.d0
                phi_AB = phase_per_tweezer(trap_B) - phase_per_tweezer(trap_A)
                exp_iphi_AB = complex(cos(phi_AB), sin(phi_AB))
                eta_phi_AB = chi_C * exp_iphi_AB - chi_C_star * conjg(exp_iphi_AB)

                ! X(trap_A), X(trap_B) coupling
                G_coeff(trap_A, 1, trap_B, 1) = &
                    imag_num * eta_phi_AB * light_couplings_array(trap_A, 1) * light_couplings_array(trap_B, 1)
                ! Y(trap_A), Y(trap_B) coupling
                G_coeff(trap_A, 2, trap_B, 2) = &
                    imag_num * eta_phi_AB * light_couplings_array(trap_A, 2) * light_couplings_array(trap_B, 2)
                ! X(trap_A), Y(trap_B) coupling
                G_coeff(trap_A, 1, trap_B, 2) = &
                    imag_num * eta_phi_AB * light_couplings_array(trap_A, 1) * light_couplings_array(trap_B, 2)
                ! Y(trap_A), X(trap_B) coupling
                G_coeff(trap_A, 2, trap_B, 1) = &
                    imag_num * eta_phi_AB * light_couplings_array(trap_A, 2) * light_couplings_array(trap_B, 1)
            ENDIF

        11 ENDDO
    10 ENDDO

    ! zero-th order X noise vector
    ! weights of a_hat, a_hat*, bx, bx*, by, by*, bz, bz*
    trap_A = 1
    DO trap=1, num_traps
        DO axis=1, Cartesian_axes_per_trap
            X1D_noise_vector(trap, axis, 1) = Beta_M(trap, axis) * chi_C / M_norm_array(trap, axis)
            X1D_noise_vector(trap, axis, 2) = Beta_M(trap, axis) * chi_C_star / M_norm_array(trap, axis)
        ENDDO
    ENDDO

    CBM_noise_vector(1) = Beta_M_CBM * chi_C / M_norm_CBM
    CBM_noise_vector(2) = Beta_M_CBM * chi_C_star / M_norm_CBM
    count_1 = 2
    DO trap=1, num_traps
        DO axis=1, Cartesian_axes_per_trap

            X1D_noise_vector(trap, axis, count_1 + 1) = sqrt_Gamma_M * chi_M_array(trap, axis) / M_norm_array(trap, axis)
            X1D_noise_vector(trap, axis, count_1 + 2) = sqrt_Gamma_M * chi_M_star_array(trap, axis) / M_norm_array(trap, axis)

            CBM_noise_vector(count_1 + 1) = sqrt_Gamma_M * chi_M_array(trap, axis) / M_norm_CBM
            CBM_noise_vector(count_1 + 2) = sqrt_Gamma_M * chi_M_star_array(trap, axis) / M_norm_CBM
            count_1 = count_1 + 2

        ENDDO
    ENDDO



    ! one PARTICLE SOLUTION
    count_1 = 0
    DO 30 trap_A=1, num_traps
        DO 31 jj=1, Cartesian_axes_per_trap
            count_1 = count_1 + 1
            count_2 = 0
            DO trap_B=1, num_traps
                DO kk=1, Cartesian_axes_per_trap
                    count_2 = count_2 + 1
                    Rjk_array(trap_A, jj, trap_B, kk) = &
                        imag_num * G_coeff(trap_A, jj, trap_B, kk) * eta_M_array(trap_A, jj) / M_norm_array(trap_A, jj)
                    coeff = 0.d0
                    IF(count_1 == count_2) coeff = 1.d0
                    derangement_matrix(count_1, count_2) = coeff - Rjk_array(trap_A, jj, trap_B, kk)
                    ! for explicit use below
                    ! derangement_matrix(count_1, count_2) = hybridisation_solution_array(trap_A, jj, trap_B, kk)
                ENDDO
            ENDDO
        31 ENDDO
    30 ENDDO

    IF ((omega / kHz).GE.297.95.AND.(omega / kHz).LE.298.05) THEN
        WRITE(6, *) 'omega (kHz)', omega / kHz
        WRITE(6, *) 'R_x1_x1 = Rjk_array(1, 1, 1, 1)', Rjk_array(1, 1, 1, 1)
        WRITE(6, *) 'R_x1_y1 = Rjk_array(1, 1, 1, 2)', Rjk_array(1, 1, 1, 2)
        WRITE(6, *) 'R_x1_x2 = Rjk_array(1, 1, 2, 1)', Rjk_array(1, 1, 2, 1)
        WRITE(6, *) 'R_y1_y2 = Rjk_array(1, 1, 1, 1)', Rjk_array(1, 2, 2, 2)
        WRITE(6, *) 'X1D_noise_vector(1, 1, :)', X1D_noise_vector(1, 1, :)
        WRITE(6, *) 'X1D_noise_vector(1, 2, :)', X1D_noise_vector(1, 2, :)
        WRITE(6, *) 'X1D_noise_vector(2, 1, :)', X1D_noise_vector(2, 1, :)
        WRITE(6, *) 'X1D_noise_vector(2, 2, :)', X1D_noise_vector(2, 2, :)
    ENDIF



    ! LAPACK INVERTER
    CALL INVERT_NxN_MATRIX(&
        num_dimensions, &
        derangement_matrix, &
        hybridisation_matrix)
    ! DEBUG_test_of_matrix_inversion = matmul(derangement_matrix, hybridisation_matrix)
    ! WRITE(6, *) DEBUG_test_of_matrix_inversion

    ! Re-pack hybridisation_matrix into 4d
    hybridisation_solution_array = 0.d0
    count_1 = 0

    DO trap_A=1, num_traps
        DO jj=1, Cartesian_axes_per_trap
            count_1 = count_1 + 1
            count_2 = 0
            DO trap_B=1, num_traps
                DO kk=1, Cartesian_axes_per_trap
                    count_2 = count_2 + 1
                    hybridisation_solution_array(trap_A, jj, trap_B, kk) = hybridisation_matrix(count_1, count_2)
                ENDDO
            ENDDO
        ENDDO
    ENDDO

    DO ii=1, num_operators
        DO trap_A=1, num_traps
            DO jj=1, Cartesian_axes_per_trap
                DO trap_B=1, num_traps
                    DO kk=1, Cartesian_axes_per_trap
                        X_noise_vector(trap_A, jj, ii) = &
                            X_noise_vector(trap_A, jj, ii) &
                            + hybridisation_solution_array(trap_A, jj, trap_B, kk) * X1D_noise_vector(trap_B, kk, ii)
                    ENDDO
                ENDDO
            ENDDO
        ENDDO
    ENDDO

    ! now work out the optical trap field = a_hat
    i_chi_C = imag_num * chi_C

    ! now work out the photon field = a_dagger
    i_chi_C_star = -imag_num * chi_C_star

    DO 111 trap=1, num_traps
        ! phase terms
        trap_phase = phase_per_tweezer(trap)
        exp_iphi_AB = complex(cos(trap_phase), sin(trap_phase))
        DO 112 axis=1, Cartesian_axes_per_trap
            DO  ii=1, num_operators
                a_hat(ii) = a_hat(ii) + i_chi_C * light_couplings_array(trap, axis) * X_noise_vector(trap, axis, ii)
                a_dag(ii) = a_dag(ii) + i_chi_C_star * light_couplings_array(trap, axis) * X_noise_vector(trap, axis, ii)
                ! spectrum of collective mode
                a_hat_CBM(ii) = i_chi_C * light_couplings_array(1, 1) * CBM_noise_vector(ii)
                a_dag_CBM(ii) = i_chi_C_star * light_couplings_array(1, 1) * CBM_noise_vector(ii)
            ENDDO
        112 ENDDO
    111 ENDDO

    ! add shot or incoming noise
    ! trap beam: add cavity-filtered contribution
    a_hat(1) = a_hat(1) + sqrt_kappa * chi_C
    a_dag(2) = a_dag(2) + sqrt_kappa * chi_C_star
    a_hat_CBM(1) = a_hat_CBM(1) + sqrt_kappa * chi_C
    a_dag_CBM(2) = a_dag_CBM(2) + sqrt_kappa * chi_C_star

    ! cavity output: add incoming imprecision
    ! work out a_out = a_in - sqrt_kappa(a)
    DO ii=1, num_operators
        a_hat(ii) = -a_hat(ii) * sqrt_kappa
        a_dag(ii) = -a_dag(ii) * sqrt_kappa
        a_hat_CBM(ii) = -a_hat_CBM(ii) * sqrt_kappa
        a_dag_CBM(ii) = -a_dag_CBM(ii) * sqrt_kappa
    ENDDO

    a_hat(1) = one + a_hat(1)
    a_dag(2) = one + a_dag(2)
    a_hat_CBM(1) = one + a_hat_CBM(1)
    a_dag_CBM(2) = one + a_dag_CBM(2)

    RETURN
END

SUBROUTINE INVERT_NxN_MATRIX(&
    N, &
    matrix, &
    inverse)

    IMPLICIT NONE

    integer::N, INFO, LWORK, pivot_indices(N)
    double complex::matrix(N, N), inverse(N, N), WORK(N * N)
    ! TODO: understand LWORK and WORK better?

    LWORK = N * N
    inverse = 0.d0

    ! https://netlib.org/lapack/explore-html/db/d04/group__getrf_ga85d0f9ae367fd540a745f558a77717db.html
    !  ZGETRF computes an LU factorization of a general M-by-N matrix A
    !  using partial pivoting with row interchanges.
    !  The factorization has the form
    !     A = P * L * U
    ! - P is a permutation matrix
    ! - L is lower triangular with unit diagonal elements (lower trapezoidal if m > n)
    ! - U is upper triangular (upper trapezoidal if m < n).
    CALL ZGETRF(&
        N, & ! [in] num rows
        N, & ! [in] num columns
        ! [in, out] matrix is COMPLEX*16 array, dimension (LDA,N)
        ! On entry, the M-by-N matrix to be factored.
        ! On exit, the factors L and U from the factorization
        ! A = P*L*U; the unit diagonal elements of L are not stored.
        matrix, &
        N, & ! [in] LDA = leading dimension of matrix A
        pivot_indices, & ! [out] row #i of the matrix is interchanged with row #pivot_indices(i)
        ! [out] INFO is INTEGER
        ! = 0:  successful exit
        ! < 0:  if INFO = -i, the i-th argument had an illegal value
        ! > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
        !       has been completed, but the factor U is exactly
        !       singular, and division by zero will occur if it is used
        !       to solve a system of equations.
        INFO)

    ! https://netlib.org/lapack/explore-html/da/d28/group__getri_ga563600017476dbf36fb663c3f9f61b8a.html
    ! ZGETRI computes the inverse of a matrix using the LU factorization
    ! computed by ZGETRF.
    ! This method inverts U and then computes inv(A) by solving the system
    ! inv(A)*L = inv(U) for inv(A).
    CALL ZGETRI(&
        N, & ! [in] matrix order
        matrix, & ! [in, out] matrix to be factored
        N, & ! [in] LDA = leading dimension of matrix A
        pivot_indices, & ! [out] row #i of the matrix is interchanged with row #pivot_indices(i)
        WORK, & ! [out] WORK(1) = optimal LWORK if INFO=0 (successful inversion)
        ! [in] LWORK is INTEGER
        ! The dimension of the array WORK.  LWORK >= max(1,N).
        ! For optimal performance LWORK >= N*NB, where NB is
        ! the optimal blocksize returned by ILAENV.

        ! If LWORK = -1, then a workspace query is assumed; the routine
        ! only calculates the optimal size of the WORK array, returns
        ! this value as the first entry of the WORK array, and no error
        ! message related to LWORK is issued by XERBLA.
        LWORK, &
        ! [out] INFO is INTEGER
        ! = 0:  successful exit
        ! < 0:  if INFO = -i, the i-th argument had an illegal value
        ! > 0:  if INFO = i, U(i,i) is exactly zero; the matrix is
        !       singular and its inverse could not be computed.
        INFO)

    IF(INFO.NE.0) WRITE(6, *) 'Failed to invert matrix! INFO = ', INFO

    inverse = matrix

    RETURN
END
