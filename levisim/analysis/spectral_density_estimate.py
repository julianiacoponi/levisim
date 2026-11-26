"""
Spectral density estimates (PSDs and CSDs) of recorded time traces, by Bartlett's method
"""

import numpy as np

from scipy.optimize import curve_fit
from scipy.integrate import simpson, trapezoid

from levisim.core.constants import kB
from levisim.core.units import (
    nm,
    ms,
    Hz, kHz,
)
from levisim.core.logger import logger


class SpectralDensityEstimate:
    """
    Estimates S_AB, i.e. the spectral density:

    - PSD (Power Spectral Density), if B=A, e.g. Sxx, Syy etc.
    - CSD (Cross-correlation Spectral Density), if B≠A, e.g. Sxy etc.

    Bartlett's method: the traces are cut into `num_segments` non-overlapping segments, each is
    (optionally) windowed and Fourier transformed, and the periodograms X*_i(f) Y_i(f) are averaged.
    Normalised so that, for the default rectangular window, the two-sided estimate obeys Parseval
    exactly: Σ S(f_k) Δf = 〈a b〉 over the averaged samples. Units: [A][B]/Hz, e.g. m²/Hz.

    Any trace in source.traces can be used, e.g. 'x', 'y', 'z', the momenta 'px', 'py', 'pz'
    (Simulation(store_momenta=True)) or the radial 'r' = √(x² + y²) (Simulation(store_radial=True));
    pass multi-character axes as a tuple, e.g. ('px', 'px'). A PSD of 'x', 'y' or 'z' is fitted
    automatically (fit_PSD) when the source has a Trap.
    NOTE: momentum spectra follow from the position ones, Spp(ω) = m²ω² Sxx(ω) (since p = m dx/dt);
    the radial trace r(t) is a non-linear function of x, y, so its spectrum mixes them (difference and
    sum frequencies Ωx ± Ωy, and near-DC power from 〈r〉 fluctuations) and has no single Lorentzian form.

    Ensembles: the periodograms can also be averaged over independent members of an ensemble
    (Bartlett's method over members × segments). Either pass a list of sources (e.g. seeded
    FreeSpaceSimulation runs of the same length and Δt), or a source with trajectories (a
    CavitySimulation) and `ensemble=True` (every trajectory) or `ensemble=[k0, k1, ...]` (those
    trajectories). The estimate's values are then the ensemble average, averaging_time is the total
    averaged time (members × segments × segment time), and the per-segment arrays
    (fft_per_segment_A/B, SDs_per_segment, covariance matrices) gain a leading ensemble axis of
    length num_ensemble_members. Without an ensemble nothing changes.

    Keyword arguments: window [None => rectangular], see the window property; fit_PSD_kwargs;
    num_segments_to_average; subtract_segment_mean [False] (subtract each segment's mean before
    windowing); keep_segments [True] (False skips storing the per-segment arrays, which hold
    members × samples complex numbers each); fit [True]; debug.
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

        # None: one source; True (every trajectory of the source) or a sequence of trajectory
        # indices: an ensemble average (see the class docstring). A list of sources is an ensemble too.
        ensemble=None,

        **kwargs,
    ):
        self.debug = kwargs.get('debug', False)
        members = self._ensemble_members(source, ensemble)
        self.is_ensemble = members is not None
        if members is None:
            members = [source]
        else:
            source = members[0]
        self.source = source
        self.members = members
        self.num_ensemble_members = len(members)
        self.one_sided = one_sided

        A, B = AB
        self.AB = AB
        self.A, self.B = source.traces[A], source.traces[B]
        trace_pairs = [(self.A, self.B)] + [(m.traces[A], m.traces[B]) for m in members[1:]]

        if A == B:
            self.type = 'PSD'
        else:
            self.type = 'CSD'

        self.define_averaging_variables(num_segments, num_samples_per_segment, desired_Δf)
        self.halfway_index = int(self.num_samples_per_segment // 2)

        # include this to be able to use a subset of the segments
        self.num_segments_to_average = kwargs.get('num_segments_to_average', self.num_segments)
        self.averaging_time = self.num_segments_to_average * self.segment_time
        if self.is_ensemble:
            # every member contributes num_segments_to_average periodograms
            self.averaging_time = self.num_ensemble_members * self.averaging_time
        self.measurement_bandwidth = 1 / self.averaging_time

        # NOTE: answer to "figure out best window?": there is no single best one - a window trades
        # leakage (dynamic range far from a peak) against resolution and variance. The rectangular
        # window (default, unchanged) has the best resolution and is unbiased for white noise, but its
        # -13 dB, 1/f² sidelobes leak a sharp, high-Q peak into the wings; Hann or Blackman-Harris
        # (-92 dB sidelobes) are better for measuring a weak floor next to a strong, narrow peak.
        # `window` takes any scipy.signal.get_window spec (e.g. 'hann', 'blackmanharris') or an array.
        # It is normalised to unit mean square, w / √〈w²〉, which preserves the level (and area) of
        # broadband PSDs. Jon's Blackman-Harris normalisation, 1/((287/800)√2), is this to within 0.1%:
        # 287/800 = 0.35875 is the window's mean (a0), and 0.35875·√2 = 0.50735 vs √〈w²〉 = 0.50784
        # (4096-sample periodic window).
        self.window_spec = kwargs.get('window', None)
        self.window = self.make_window(self.window_spec, self.num_samples_per_segment)

        # initialise the arrays to sum each data segment's periodogram
        segment_SDs = np.zeros(self.num_samples_per_segment, dtype=complex)
        subtract_segment_mean = kwargs.get('subtract_segment_mean', False)
        keep_segments = kwargs.get('keep_segments', True)
        K = self.num_ensemble_members

        # the length of SDs_per_segment depends on whether the final spectral density we want
        # is one-sided or two-sided
        final_num_samples_per_segment = self.halfway_index if self.one_sided else self.num_samples_per_segment
        if keep_segments:
            # initialise arrays to track each segments FFTs, and times (one row of segments per member)
            fft_A = np.zeros([K, self.num_segments, self.num_samples_per_segment], dtype=complex)
            fft_B = np.zeros([K, self.num_segments, self.num_samples_per_segment], dtype=complex)
            SDs = np.zeros([K, self.num_segments, final_num_samples_per_segment], dtype=complex)
            time_array = self.source.time_array
            self.time_arrays_per_segment = np.zeros([self.num_segments, self.num_samples_per_segment])
            # keep track of the covariance matrices for each segment
            # i.e. a 2x2 matrix of [[xx, xy], [yx, yy]] correlations
            if self.type != 'PSD':
                covariances = np.zeros([K, self.num_segments, 2, 2])
                norm_covariances = np.zeros([K, self.num_segments, 2, 2])

        for member, (trace_A, trace_B) in enumerate(trace_pairs):
            start_index = 0
            for segment_index in range(self.num_segments_to_average):
                end_index = (segment_index + 1) * self.num_samples_per_segment

                segment_A = trace_A.values[start_index:end_index]
                segment_B = trace_B.values[start_index:end_index]
                if subtract_segment_mean:
                    segment_A = segment_A - segment_A.mean()
                    segment_B = segment_B - segment_B.mean()
                segment_A = segment_A * self.window
                segment_B = segment_B * self.window

                if keep_segments and self.type != 'PSD':
                    covariances[member, segment_index] = np.cov(np.array([segment_A, segment_B]))
                    norm_covariances[member, segment_index] = np.corrcoef(np.array([segment_A, segment_B]))

                # the default for numpy's fft applies the 1/N factor to the inverse transform only
                # N = num_samples_per_segment
                segment_fft_A = np.fft.fft(segment_A)  # * Δt
                # a PSD transforms the same segment twice: reuse it (np.fft.fft is deterministic)
                segment_fft_B = segment_fft_A if trace_B is trace_A else np.fft.fft(segment_B)  # * Δt
                # apply the factor of Δt outside the loop, for efficiency
                # 2010 - Bendat, Piersol - Random Data Analysis and Measurement Procedures - Eq. 11.36,38, p368

                segment_SD = np.conj(segment_fft_A) * segment_fft_B
                segment_SDs += segment_SD

                if keep_segments:
                    fft_A[member, segment_index] = segment_fft_A
                    fft_B[member, segment_index] = segment_fft_B
                    if member == 0:
                        self.time_arrays_per_segment[segment_index] = time_array[start_index:end_index]
                    if self.one_sided:
                        SDs[member, segment_index] = 2 * segment_SD[:self.halfway_index]
                        SDs[member, segment_index, 0] = segment_SD[0]
                    else:
                        SDs[member, segment_index] = segment_SD

                start_index = end_index

        if keep_segments:
            # without an ensemble the per-segment arrays keep their single-source shapes
            pick = (lambda a: a) if self.is_ensemble else (lambda a: a[0])
            self.fft_per_segment_A = pick(fft_A)
            self.fft_per_segment_B = pick(fft_B)
            self.SDs_per_segment = pick(SDs)
            if self.type != 'PSD':
                self.covariance_matrices_per_segment = pick(covariances)
                self.norm_covariance_matrices_per_segment = pick(norm_covariances)
        else:
            self.fft_per_segment_A = self.fft_per_segment_B = self.SDs_per_segment = None
            self.time_arrays_per_segment = None

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
        # the two-sided values in np.fft.fftfreq order (f = 0 first), before any reordering
        self.fft_order_values = two_sided_spectrum

        # NOTE: Random Data (Bendat, Piersol) bottom of p387 describes how the 0 + Nyquist frequencies suffer from errors and are usually discarded!
        # (the frequencies and values are also available together as the 2D array self.spectrum)
        if self.one_sided:
            # throw away the second half of the arrays fft and fftfreq return which represent negative frequency values
            # [0, Δf, 2Δf, ... nΔf], len = n
            # i.e. 0 first, then positive
            self.freq_range = fftfreqs[:self.halfway_index]
            # one-sided spectra double the values of the two-sided spectra at 0 < f < f_Nyq
            # NOTE: the f = 0 bin has no negative-frequency partner, so it is *not* doubled
            # (Bendat & Piersol, Random Data, bottom of p387); earlier versions doubled it too
            self.values = 2 * two_sided_spectrum[:self.halfway_index]
            self.values[0] = two_sided_spectrum[0]
        else:
            # reorder the frequencies to be:
            # [ -nΔf, -(n-1)Δf, ..., -2Δf, -Δf, 0, Δf, 2Δf, ... (n-1)Δf, nΔf], len = 2n
            # negative first, 0 in the middle, then positive
            self.freq_range = np.concatenate((fftfreqs[self.halfway_index:], fftfreqs[:self.halfway_index]))

            # do the same reordering for the values to match up
            self.values = np.concatenate((fft_spectrum[self.halfway_index:], fft_spectrum[:self.halfway_index]))

        # self.Afit, self.Ωfit, self.Γfit = 0, 0, 0
        # self.Afit_sd, self.Γfit_sd, self.Ωfit_sd = 0, 0, 0
        # fit the PSD automatically for the position axes of a source that has a Trap
        trap = getattr(source, 'trap', None)
        if self.type == 'PSD' and trap is not None and self.A.axis in trap.Γj and kwargs.get('fit', True):
            fit_PSD_kwargs = kwargs.get('fit_PSD_kwargs', {})
            self.fit_PSD(**fit_PSD_kwargs)

    @staticmethod
    def _ensemble_members(source, ensemble):
        """The sources of an ensemble estimate, or None for a single source"""
        if isinstance(source, (list, tuple)):
            if ensemble is not None:
                raise ValueError('pass either a list of sources or ensemble=..., not both')
            members = list(source)
        elif ensemble is None or ensemble is False:
            return None
        else:
            if not hasattr(source, 'trajectory') or not hasattr(source, 'num_trajectories'):
                raise ValueError(
                    f'ensemble={ensemble!r} needs a source with trajectories (e.g. a CavitySimulation); '
                    'for independent runs pass a list of sources instead'
                )
            indices = range(source.num_trajectories) if ensemble is True else [int(k) for k in ensemble]
            members = [source.trajectory(k) for k in indices]
        if not members:
            raise ValueError('an ensemble needs at least one member')
        first = members[0]
        for member in members[1:]:
            if member.Δt != first.Δt or member.num_samples != first.num_samples:
                raise ValueError('every member of an ensemble needs the same Δt and num_samples')
        return members

    @staticmethod
    def make_window(window, num_samples):
        """
        Segment window, normalised to unit mean square so that broadband PSD levels are preserved
        window: None or 'rectangular' => 1 (no window); a scipy.signal.get_window spec, e.g. 'hann',
        'blackmanharris', ('tukey', 0.25); or an array of length num_samples
        [·]
        """
        if window is None or (isinstance(window, str) and window in ('rectangular', 'boxcar')):
            return 1
        if isinstance(window, (str, tuple)):
            # imported here, not at the top: scipy.signal takes about half a second to import
            import scipy.signal
            w = scipy.signal.get_window(window, int(num_samples), fftbins=True)
        else:
            w = np.asarray(window, dtype=float)
            assert len(w) == num_samples, f'window has {len(w)} samples, segments have {num_samples}'
        return w / np.sqrt(np.mean(w**2))

    @property
    def spectrum(self):
        """
        The frequencies and values as one 2D array, shape (len(freq_range), 2): [[f, S(f)], ...]
        (complex dtype, since a CSD is complex)
        [Hz, units²/Hz]
        """
        return np.column_stack((self.freq_range, self.values))

    @property
    def info(self):
        """
        Prints info on the spectral density estimate
        """
        fit_str = ''
        if self.type == 'PSD':
            fit_str = (
                f'\n    Afit = {self.Afit:.3e} ± {self.Afit_sd:.3e} m²/s²'
                f'\n    Ωfit = {self.Ωfit / kHz:.3f} ± {self.Ωfit_sd / kHz:.3f} kHz'
                f'\n    Γfit = {self.Γfit / kHz:.3f} ± {self.Γfit_sd / kHz:.3f} kHz'
                f'\n'
                f'\n    temperature_via_area = {self.temperature_via_area:.3f} K'
                f'\n    temperature_via_fit = {self.temperature_via_fit:.3f} K'
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
        Given one of the three::

            - num_segments (to divide the time trace into)
            or
            - num_samples_per_segment (how many time step samples of the time trace do we want to have per segment?)
            or
            - desired_Δf (how big a frequency bin size do we want?)

        the other two can be derived.
        (NOTE: I do not do any zero-padding of the data!)

        In particular, the frequency bin size can be seen as::

            Δf = frequency step size for each segmented PSD estimate
            Δf = 1 / segment_time
            equivalent to
            Δf = num_segments / time_length
            Δf = sample_rate / num_samples_per_segment (where sample_rate = 1 / Δt)
            Δf = the index-to-frequency(Hz) conversion factor, in Hertz

        Also, in particular::

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
            logger.warning(
                'Linewidth estimate failed: no pair of points found around the half maximum'
                ' (len(Γ_indices) = {}), so returning None',
                len(self.Γ_indices),
            )
            return None

    @property
    def area(self):
        """
        Area under the curve of the values across the frequency range (trapezoidal rule)
        NOTE: np.trapz was removed in NumPy 2.4, which made this (and temperature_via_area) raise;
        scipy.integrate.trapezoid is the same algorithm.
        [units of the trace]²
        """
        return trapezoid(np.real(self.values), self.freq_range)

    @property
    def temperature_via_area(self):
        """
        NOTE: Bendat, Piersol -- Random Data - Analysis and Measurement Procedures, p147, Eq 5.125 ::

            Area under PSD = Var(x) =〈x²〉
            〈x²〉= (1/2π) ∫ Sxx(Ω)dΩ
            〈x²〉= ∫ Sxx(f)df = ∫ Gxx(f)df
            For two-sided Sxx, the limits are -∞ to +∞
            For one-sided Gxx = 2 * Sxx, the limits are 0 to +∞

        Equipartition theorem ::

            ½mΩ²〈x²〉 = ½kb*T

        Therefore expecting T = (mΩ² / kb) * Area

        NOTE: since 〈z〉≠ 0, Var(z) =〈z²〉- 〈z〉², so the z-trace needs to be adjusted for it to
        give a sensible answer for the temperature_via_area

        """
        if self.type != 'PSD':
            return None

        axis = self.A.axis
        # equipartition_factor = self.source.trap.bead.m * self.source.trap.Ωj[axis]**2 / kB
        equipartition_factor = self.source.trap.bead.m * self.Ωfit**2 / kB
        return equipartition_factor * (self.area)

    def frequency_to_index(
        self,
        lower_bound_Hz,
        upper_bound_Hz,
    ):
        """
        Converts a lower and upper frequency bound (in Hz) into the indices of the arrays of the spectral estimate
        """
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
        # freq_range = [-nΔf, -(n-1)Δf, ..., -2Δf, -Δf, 0, Δf, 2Δf, ... (n-1)Δf, nΔf], len = 2n = num_samples_per_segment
        # so, if the bounds are negative, then the indices will be negative too
        # so adding this to the halfway index will give the desired index
        return self.halfway_index + lower_index, self.halfway_index + upper_index

    def get_plot_points(
        self,
        lower_bound_Hz,
        upper_bound_Hz,
    ):
        """
        Returns the portions of the arrays given by the lower and upper frequency bounds
        """
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
        fit_range=(1_000, 500_000),
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

        self.fit_range = fit_range

        sidedness = 2 if self.one_sided else 1
        if A0 is None:
            # amplitude in Hz variables: A_Hz = 2 kB T_eff / m / (2π)³, doubled if one-sided (see Trap.A_Hz)
            trap = self.source.trap
            A0 = sidedness * trap.A_Hz * trap.T_eff[self.A.axis] / trap.gas.T
        if Γ0 is None:
            Γ0 = self.source.trap.Γj[self.A.axis] / Hz
        if Ω0 is None:
            Ω0 = self.Ω / Hz

        p0_Hz = (A0, Γ0, Ω0)

        freqs, PSD_vals = self.get_plot_points(*fit_range)
        # a PSD is real; the values are stored as complex (shared with the CSD code path)
        PSD_vals = np.real(PSD_vals)
        if debug:
            logger.debug(
                'fit_PSD: range={}, freqs from {} to {} Hz, starting values A0={:.3f}, Γ0={:.3f} Hz, Ω0={:.3f} Hz',
                fit_range, min(freqs), max(freqs), A0, Γ0, Ω0,
            )

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
        A_fit, Γ, Ω = self.fit_params

        # convert the amplitude to angular variables like Γ and Ω. The Lorentzian Γ/((Ω²-ω²)²+Γ²ω²)
        # has dimensions time³, so this needs (2π)³ (earlier versions used (2π)², giving A/2π; see Trap.A_rad).
        # Afit is then directly comparable to 2 kB T_eff / m (Trap.A at T_eff), ×2 if one-sided.
        self.Afit = A_fit * Hz**3
        self.Γfit = Γ * Hz
        self.Ωfit = Ω * Hz

        # the diagonal elements of the covariance matrix are the variances of the fit parameters
        self.fit_params_st_devs = np.sqrt(np.diag(self.fit_params_covariance_matrix))
        self.Afit_sd, self.Γfit_sd, self.Ωfit_sd = self.fit_params_st_devs
        self.Afit_sd = self.Afit_sd * Hz**3
        self.Γfit_sd = self.Γfit_sd * Hz
        self.Ωfit_sd = self.Ωfit_sd * Hz

    @property
    def temperature_via_fit(self):
        """
        Effective temperature from the fitted PSD amplitude

        Afit = 2 kB T_eff / m (two-sided), 4 kB T_eff / m (one-sided)
        [K]
        """
        if self.type != 'PSD':
            return None
        sidedness = 2 if self.one_sided else 1
        return self.source.trap.bead.m * self.Afit / (2 * sidedness * kB)
