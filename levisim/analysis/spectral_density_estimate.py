import scipy.signal

import numpy as np

from scipy.optimize import curve_fit
from scipy.integrate import simpson

from levisim.core.units import (
    nm,
    ms,
    Hz, kHz,
)


# TODO: what do momentum spectra look like... or radial spectra?!
# TODO: write unit tests
class SpectralDensityEstimate:
    """
    Estimates S_AB, i.e. the spectral density:
    - PSD (Power Spectral Density), if B=A, e.g. Sxx, Syy etc.
    - CSD (Cross-correlation Spectral Density), if B≠A, e.g. Sxy etc.
    """

    properties = [
        'Axes',
        'Type',
        'One Sided',

        'Num. Segments',
        'Num. Segments to Average',

        'Segment Width (Samples/Segment)',
        'Length of Freq. Range (# of Freq. bins)',

        'Segment Time, 1/Δf (ms)',
        'Frequency Step, Δf (Hz)',

        'Averaging time (ms)',
        'Measurement bandwidth (Hz)',

        'Ω (kHz)',
        # 'Γ (kHz)',

        'Temperature, via Area (K)',
        'Variance, σ² via Area (nm²)',
        'St.Dev., σ, via Area (nm)',

        'Window',

        'simulation',
        'id(simulation)',

        'self',
        'id(self)',
    ]

    def __init__(
        self,

        # this will either be from Simulation, Experiment or a TracePair
        source,

        # a string of the two axes to get the spectrum of, e.g. 'xx' for Sxx, 'xy' for Sxy
        # if the axes are more than one character, can be a tuple e.g. ('x-axis', 'y-axis')
        AB,

        # into what number of segments should the time traces be divided, such that each segment
        # is then averaged?
        # the larger this number, the larger the frequency bin, hence: the "smoother" the spectra
        num_segments=None,

        num_samples_per_segment=None,

        desired_Δf=None,


        # NOTE: One-sided vs. Two-sided spectra
        # Bendat, Piersol - Random Data Analysis and Measurement Procedures - Chapter 5

        # Spectra calculated directly from FFTs are 'two-sided'
        # i.e. with (classically symmetrical!) positive and negative frequency parts
        # 'One-sided' spectra keep only the positive half, keeping the same area by doubling its value

        # Two-sided example:
        # Tsaturyan, Barg, Polzik, Schliesser -- Ultracoherent nanomechanical resonators via soft clamping and dissipation dilution
        # http://dx.doi.org/10.1038/nnano.2017.101, Eq.10
        # Two-sided assumes Sth = 2m kbT (Ω / Q) = 2m kbT Γ
        # where Q = Ω/Γ is the quality factor of the damped harmonic oscillations
        # It is the default for np.fft.fft
        # Stems from analytic/physical resutls relating to Langevin force noises
        # https://en.wikipedia.org/wiki/Langevin_equation

        # One-sided:
        # Liang et al. -- Yoctonewton force detection based on optically levitated oscillator
        # https://doi.org/10.1016/j.fmre.2022.09.021, Eq. 4
        # One-sided spectra double the spectral floor, therefore Sth = 4m kbT Γ
        # It is used a lot in signal processing as it throws away the (sometimes confusing) concept of negative frequencies
        # Hence it's more practical for measurement and simulations
        one_sided=True,

        **kwargs,
    ):
        self.debug = kwargs.get('debug', False)
        self.source = source
        self.one_sided = one_sided

        A, B = AB
        self.AB = AB
        self.A, self.B = source.traces[A], source.traces[B]

        if A == B:
            self.type = 'PSD'
        else:
            self.type = 'CSD'

        self.define_averaging_variables(num_segments, num_samples_per_segment, desired_Δf)
        self.halfway_index = int(self.num_samples_per_segment // 2)

        # include this to be able to use a subset of the segments
        self.num_segments_to_average = kwargs.get('num_segments_to_average', self.num_segments)
        self.averaging_time = self.num_segments_to_average * self.segment_time
        self.measurement_bandwidth = 1 / self.averaging_time

        # TODO: figure out best window?
        # from Jon's code
        self.window = scipy.signal.windows.blackmanharris(int(self.num_samples_per_segment)) / ( 287/800 ) / np.sqrt(2)
        self.window = 1

        # initialise the arrays to sum each data segment's periodogram
        segment_SDs = np.zeros(self.num_samples_per_segment, dtype=complex)

        # initialise arrays to track each segments FFTs, and times
        self.fft_per_segment_A = np.zeros([self.num_segments, self.num_samples_per_segment], dtype=complex)
        self.fft_per_segment_B = np.zeros([self.num_segments, self.num_samples_per_segment], dtype=complex)

        time_array = self.source.time_array
        self.time_arrays_per_segment = np.zeros([self.num_segments, self.num_samples_per_segment])

        # the length of SDs_per_segment depends on whether the final spectral density we want
        # is one-sided or two-sided
        final_num_samples_per_segment = self.halfway_index if self.one_sided else self.num_samples_per_segment
        self.SDs_per_segment = np.zeros([self.num_segments, final_num_samples_per_segment], dtype=complex)

        # keep track of the covariance matrices for each segment
        # i.e. a 2x2 matrix of [[xx, xy], [yx, yy]] correlations
        if self.type != 'PSD':
            self.covariance_matrices_per_segment = np.zeros([self.num_segments, 2, 2])
            self.norm_covariance_matrices_per_segment = np.zeros([self.num_segments, 2, 2])

        start_index = 0
        for segment_index in range(self.num_segments_to_average):
            end_index = (segment_index + 1) * self.num_samples_per_segment

            segment_A = self.A.values[start_index:end_index] * self.window
            segment_B = self.B.values[start_index:end_index] * self.window

            if self.type != 'PSD':
                self.covariance_matrices_per_segment[segment_index] = np.cov(np.array([segment_A, segment_B]))
                self.norm_covariance_matrices_per_segment[segment_index] = np.corrcoef(np.array([segment_A, segment_B]))

            # the default for numpy's fft applies the 1/N factor to the inverse transform only
            # N = num_samples_per_segment
            segment_fft_A = np.fft.fft(segment_A)  # * Δt
            segment_fft_B = np.fft.fft(segment_B)  # * Δt
            # apply the factor of Δt outside the loop, for efficiency
            # 2010 - Bendat, Piersol - Random Data Analysis and Measurement Procedures - Eq. 11.36,38, p368

            # self.fft_per_segment_A.append(segment_fft_A)
            # self.fft_per_segment_B.append(segment_fft_B)
            self.fft_per_segment_A[segment_index] = segment_fft_A
            self.fft_per_segment_B[segment_index] = segment_fft_B

            segment_SD = np.conj(segment_fft_A) * segment_fft_B
            segment_SDs += segment_SD

            self.time_arrays_per_segment[segment_index] = time_array[start_index:end_index]
            if self.one_sided:
                # self.SDs_per_segment.append(2 * segment_SD[:self.halfway_index])
                self.SDs_per_segment[segment_index] = 2 * segment_SD[:self.halfway_index]
            else:
                # self.SDs_per_segment.append(segment_SD)
                self.SDs_per_segment[segment_index] = segment_SD

            start_index = end_index

        # 2010 - Bendat, Piersol - Random Data Analysis and Measurement Procedures - Eq. 11.98, p386
        # normalisation equivalent to
        # 1 / (num_segments * segment_time)
        # 1 / (num_samples * Δt)
        # 1 /  source.time_length

        # if num_segments_to_average != num_segments, the above need a further factor of
        # 1 / (num_segments_to_average / num_segments)
        # i.e. convert num_segments to num_segments_to_average
        # i.e. convert source.time_length to self.averaging_time

        # then, apply the factor of Δt - for each FFT, so Δt**2 ! - outside the loop, for efficiency
        normalisation = (source.Δt**2 / self.averaging_time)
        # overall, equivalent to
        # Δt / num_samples (if num_segments_to_average = num_segments)
        # i.e. doesn't depend on the number of segments!
        # (if num_segments_to_average ≠ num_segments, then it does!)

        # generate the frequency range for the selected num_samples_per_segment and Δt, units are Hz!
        # for n = int(num_samples_per_segment // 2)
        # [0, Δf, 2Δf, ... (n-1)Δf, nΔf, -nΔf, -(n-1)Δf, ..., -2Δf, -Δf], len = 2n
        # i.e. zero first, then positive, then most negative, toward zero again (weird right?)
        fftfreqs = np.fft.fftfreq(self.num_samples_per_segment, d=source.Δt)

        # adapted from the np.fft.fftfreq docstring:
        # m = num_samples_per_segment
        # fftfreqs = [0, 1, ...,   m/2-1,     -m/2, ..., -1] / (m*Δt)   if m is even
        # fftfreqs = [0, 1, ..., (m-1)/2, -(m-1)/2, ..., -1] / (m*Δt)   if m is odd

        fft_spectrum = two_sided_spectrum = normalisation * segment_SDs

        # NOTE: Random Data (Bendat, Piersol) bottom of p387 describes how the 0 + Nyquist frequencies suffer from errors and are usually discarded!
        # TODO: store freq_range + values in a 2D array, so that they are linked?
        if self.one_sided:
            # throw away the second half of the arrays fft and fftfreq return which represent negative frequency values
            # [0, Δf, 2Δf, ... nΔf], len = n
            # i.e. 0 first, then positive
            self.freq_range = fftfreqs[:self.halfway_index]
            # one-sided spectra simply double the values of the two-sided spectra
            self.values = 2 * two_sided_spectrum[:self.halfway_index]
        else:
            # reorder the frequencies to be:
            # [ -nΔf, -(n-1)Δf, ..., -2Δf, -Δf, 0, Δf, 2Δf, ... (n-1)Δf, nΔf], len = 2n
            # negative first, 0 in the middle, then positive
            self.freq_range = np.concatenate((fftfreqs[self.halfway_index:], fftfreqs[:self.halfway_index]))

            # do the same reordering for the values to match up
            self.values = np.concatenate((fft_spectrum[self.halfway_index:], fft_spectrum[:self.halfway_index]))

        # self.Afit, self.Ωfit, self.Γfit = 0, 0, 0
        # self.Afit_sd, self.Γfit_sd, self.Ωfit_sd = 0, 0, 0
        if self.type == 'PSD':
            fit_PSD_kwargs = kwargs.get('fit_PSD_kwargs', {})
            self.fit_PSD(**fit_PSD_kwargs)

    @property
    def info(self):
        """ Prints info on the spectral density estimate """

        fit_str = ''
        if self.type == 'PSD':
            fit_str = (
                f'\n    Afit = {self.Afit / Hz**2:.3e} ± {self.Afit_sd / Hz**2:.3e} Hz²'
                f'\n    Ωfit = {self.Ωfit / kHz:.3f} ± {self.Ωfit_sd / kHz:.3f} kHz'
                f'\n    Γfit = {self.Γfit / kHz:.3f} ± {self.Γfit_sd / kHz:.3f} kHz'
                f'\n'
                f'\n    temperature_via_area = {self.temperature_via_area:.3f} K'
                f'\n'
            )

        print(
            f'SpectralDensityEstimate(id={id(self)}'
            f'\n    S_AB = S_{self.AB}'
            f'\n    type = {self.type}'
            f'\n    one_sided = {self.one_sided}'
            f'\n'
            f'\n    num_segments = {self.num_segments}'
            f'\n    num_segments_to_average = {self.num_segments_to_average}'
            f'\n'
            f'\n    num_samples_per_segment = {self.num_samples_per_segment}'
            f'\n    len(freq_range) = {len(self.freq_range)}'
            f'\n'
            f'\n    segment_time = {self.segment_time / ms:.3f} ms'
            f'\n    Δf = {self.Δf:.3f} Hz'
            f'\n'
            f'\n    averaging_time = {self.averaging_time / ms:.3f} ms'
            f'\n    measurement_bandwidth = {self.measurement_bandwidth:.3f} Hz'
            f'\n    window = {self.window}'
            f'\n'
            f'\n    Var = Area = {self.area / nm**2:.3f} nm²'
            f'\n    σ = √Area = {np.sqrt(self.area) / nm:.3f} nm'
            f'\n'
            f'{fit_str}'
            f'\n    type(source)={type(self.source)}'
            f'\n    id(source)={id(self.source)}'
            f'\n)'
        )
        return

    def define_averaging_variables(self, num_segments, num_samples_per_segment, desired_Δf):
        """
        Given one of the three:
            - num_segments (to divide the time trace into)
            or
            - num_samples_per_segment (how many time step samples of the time trace do we want to have per segment?)
            or
            - desired_Δf (how big a frequency bin size do we want?)
        the other two can be derived.
        (NOTE: I do not do any zero-padding of the data!)

        In particular, the frequency bin size can be seen as:
        Δf = frequency step size for each segmented PSD estimate
        Δf = 1 / segment_time
        equivalent to
        Δf = num_segments / time_length
        Δf = sample_rate / num_samples_per_segment (where sample_rate = 1/Δt)
        Δf = the index-to-frequency(Hz) conversion factor, in Hz

        Also, in particular:
        segment_time = time length of each segment, equivalent to:
        segment_time = 1 / Δf
        segment_time = num_samples_per_segment * Δt
        segment_time = time_length / num_segments
        segment_time = the frequency(Hz)-to-index conversion factor, in seconds
        """

        # ensure only one of these 3 args is given
        self.desired_Δf = desired_Δf
        assert np.sum([bool(num_segments), bool(num_samples_per_segment), bool(desired_Δf)]) == 1, f'Only 1 of these 3 kwargs should be specified! {num_segments=}, {num_samples_per_segment=}, {desired_Δf=}'

        # NOTE: my use of int() throughout means the floor is taken when perfect ratios aren't acheived
        if num_segments is not None:
            self.num_segments = int(num_segments)
            self.num_samples_per_segment = int(self.source.num_samples // num_segments)
            self.Δf = 1 / self.segment_time

        elif num_samples_per_segment is not None:
            self.num_samples_per_segment = int(num_samples_per_segment)
            self.num_segments = int(self.source.num_samples // num_samples_per_segment)
            self.Δf = 1 / self.segment_time

        elif desired_Δf is not None:
            assert desired_Δf >= self.source.min_Δf, 'Frequency bin size cannot be smaller than 1 / time_length!'
            self.num_segments = int(self.source.time_length * desired_Δf)
            self.num_samples_per_segment = int(self.source.num_samples // self.num_segments)

            # NOTE: since I am not implementing any zero-padding, the frequency bin size is limited
            # by integer ratios of the number of samples.
            # thus, the Δf kwarg is more of a 'suggestion', hence qualifying it with 'desired'
            self.Δf = 1 / self.segment_time

    @property
    def segment_time(self):
        """ Time length of a segment of the time trace that a one-shot spectral density estimate is made from """
        return self.source.time_length / self.num_segments

    @property
    def Ω(self):
        """
        Estimates the resonant frequency (rad/s)
        by finding the numerical peak of the spectrum
        """
        if self.type != 'PSD':
            return None
        # ignore the first point which, at least for Szz, seems to be very high...
        ignore = 1
        vals = self.values[ignore:]
        return np.abs(self.freq_range[vals.argmax() + ignore] * Hz)

    @property
    def Γ(self):
        """
        Estimates the full-width half maximum (rad/s)
        by calculating the difference between the two values closest to half the spectral peak value
        """
        if self.type != 'PSD':
            return None

        # ignore the first point which, at least for Szz, seems to be very high...
        ignore = 1
        vals = self.values[ignore:]
        halfmax = max(vals) / 2

        # this should pick out the indices of the values closest to Ωj ± Γj/2
        self.Γ_indices, = np.where(np.isclose(
            vals,
            halfmax,
            atol=halfmax / 5,
        ))
        self.Γ_indices += ignore

        # hence, the linewidth can be estiamted as the largest of the diffs of these values
        try:
            return max(np.diff(self.freq_range[self.Γ_indices])) * Hz
        except ValueError:
            print(f'ValueError: Γ_indices array may be empty? => len(Γ_indices)={len(self.Γ_indices)}')
            return None

    @property
    def area(self):
        """ Area under the curve of the values across the frequency range """
        return np.trapz(np.real(self.values), self.freq_range)

    @property
    def temperature_via_area(self):
        """
        NOTE: Bendat, Piersol -- Random Data - Analysis and Measurement Procedures, p147, Eq 5.125
        Area under PSD = Var(x) =〈x²〉
        〈x²〉= (1/2π) ∫ Sxx(Ω)dΩ
        〈x²〉= ∫ Sxx(f)df = ∫ Gxx(f)df
        For two-sided Sxx, the limits are -∞ to +∞
        For one-sided Gxx = 2 * Sxx, the limits are 0 to +∞

        Equipartition theorem
        ½mΩ²〈x²〉 = ½kb*T
        Therefore expecting T = (mΩ² / kb) * Area

        NOTE: since 〈z〉≠ 0, Var(z) =〈z²〉- 〈z〉², so the z-trace needs to be adjusted for it to
        give a sensible answer for the temperature_via_area

        """
        if self.type != 'PSD':
            return None

        axis = self.A.axis
        # equipartition_factor = self.source.physics.bead.m * self.source.physics.Ωj[axis]**2 / kB
        equipartition_factor = self.source.physics.bead.m * self.Ωfit**2 / kB
        return equipartition_factor * (self.area)

    def frequency_to_index(
        self,
        lower_bound_Hz,
        upper_bound_Hz,
    ):
        """ Converts a lower and upper frequency bound (in Hz) into the indices of the arrays of the spectral estimate """
        assert lower_bound_Hz < upper_bound_Hz

        # the maximum frequency that can be estimated (for a given sample rate = 1/Δt) is the Nyquist frequency = sample_rate / 2
        f_Nyq = self.source.Nyquist_frequency

        # the lower bound, if negative, can be equal to the Nyquist frequency
        assert -f_Nyq <= lower_bound_Hz < f_Nyq
        assert -f_Nyq < upper_bound_Hz < f_Nyq

        # Δf converts index-to-frequency(Hz)
        # 1/Δf converts frequency(Hz)-to-index

        # by using int( // ), any value below the Nyquist freq. (even by a tiny amount)
        # will be taken to the correct 'floor' for the indexing to work
        lower_index =  int(lower_bound_Hz // self.Δf)
        upper_index = int(upper_bound_Hz // self.Δf)

        if self.one_sided:
            # for one-sided spectra, I throw away the negative frequencies, leaving:
            # freq_range = [0, Δf, 2Δf, ... nΔf], len = n = num_samples_per_segment // 2
            assert lower_bound_Hz >= 0
            assert upper_bound_Hz >= 0
            return lower_index, upper_index

        # for two-sided spectra, I reorder the frequencies to be:
        # freq_range = [ -nΔf, -(n-1)Δf, ..., -2Δf, -Δf, 0, Δf, 2Δf, ... (n-1)Δf, nΔf], len = 2n = num_samples_per_segment
        # so, if the bounds are negative, then the indices will be negative too
        # so adding this to the halfway index will give the desired index
        return self.halfway_index + lower_index, self.halfway_index + upper_index

    def get_plot_points(
        self,
        lower_bound_Hz,
        upper_bound_Hz,
    ):
        """ Returns the portions of the arrays given by the lower and upper frequency bounds """
        lb, ub = self.frequency_to_index(lower_bound_Hz, upper_bound_Hz)
        freqs = self.freq_range[lb:ub]
        values = self.values[lb:ub]
        return freqs, values

    def fit_PSD(
        self,
        # starting guesses for the fit function
        A0=None,
        Γ0=None,
        Ω0=None,
        # the range of Hz values in which to fit
        range=(1_000, 500_000),
        # error tolerance on the function evaluations
        ftol=1e-10,
        # maximum number of function evaluations
        maxfev=1e9,
        debug=False,
    ):
        """ Fits for the PSDs, all done in Hertz """
        if self.type != 'PSD':
            return None

        def responsefit_1_peak(ω, A, Γ, Ω):
            """ Function to fitting 1 response curve """
            return A * Γ / ((Ω**2 - ω**2)**2 + (Γ*ω)**2)

        self.fit_range = range

        if A0 is None:
            # A = kb*T / (π*m) for a displacement PSD
            A0 = self.source.physics.A / Hz**2
        if Γ0 is None:
            Γ0 = self.source.physics.Γj[self.A.axis] / Hz
        if Ω0 is None:
            Ω0 = self.Ω / Hz

        p0_Hz = (A0, Γ0, Ω0)

        freqs, PSD_vals = self.get_plot_points(*range)
        if debug:
            print(f'fit_PSD: range={range}')
            print(f'fit_PSD: min(freqs)={min(freqs)}')
            print(f'fit_PSD: max(freqs)={max(freqs)}')
            print(f'fit_PSD: A0={A0:.3f}')
            print(f'fit_PSD: Γ0={Γ0:.3f} Hz')
            print(f'fit_PSD: Ω0={Ω0:.3f} Hz')

        self.fitting_func = responsefit_1_peak
        self.fit_params, self.fit_params_covariance_matrix = curve_fit(
            self.fitting_func,
            freqs,
            PSD_vals,
            p0=p0_Hz,
            ftol=ftol,
            maxfev=int(maxfev),
        )

        # fit params are all in Hz, since the PSDs curves x-axis values are in Hz
        A, Γ, Ω = self.fit_params

        # just like we convert FWHM from Hz to rad/s, need to convert A from V²Hz² to V²(rad/s)²
        self.Afit = A * Hz**2
        self.Γfit = Γ * Hz
        self.Ωfit = Ω * Hz

        # the diagonal elements of the covariance matrix are the variances of the fit parameters
        self.fit_params_st_devs = np.sqrt(np.diag(self.fit_params_covariance_matrix))
        self.Afit_sd, self.Γfit_sd, self.Ωfit_sd = self.fit_params_st_devs
