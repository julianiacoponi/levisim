"""
Tests of the data/utility helpers used with Simulation and Experiment: filters, kick lists,
kick cross-checks, Trace time arrays and Experiment file loading
"""
import numpy as np
import pytest
import scipy.signal
from scipy.io import savemat

from levisim.core.utils import apply_filter, cross_check_kick_indices, generate_kick_list
from levisim.data.experiment import Experiment
from levisim.data.trace import Trace


def test_sos_filter_matches_transfer_function_filtfilt():
    rng = np.random.default_rng(0)
    fs = 1e6
    signal = rng.standard_normal(200_000)
    nyq = fs / 2
    for filter_type, bandwidth in (('butter', 20e3), ('iirpeak', 5e3)):
        new = apply_filter(signal, nyq, 150e3, filter_type, bandwidth, plot=False)
        if filter_type == 'butter':
            b, a = scipy.signal.butter(5, [(150e3 - bandwidth/2)/nyq, (150e3 + bandwidth/2)/nyq], btype='band')
        else:
            b, a = scipy.signal.iirpeak(150e3, 150e3 / bandwidth, fs=fs)
        old = scipy.signal.filtfilt(b, a, signal)
        interior = slice(5_000, -5_000)
        rel = np.max(np.abs(new[interior] - old[interior])) / np.max(np.abs(old[interior]))
        assert rel < 1e-8


def test_generate_kick_list_broadcasts_scalars():
    kicks = generate_kick_list(num_kicks=3, starts=[0.0, 1e-3, 2e-3], durations=1e-7, strengths=np.float64(5), angles=0)
    assert [k.start for k in kicks] == [0.0, 1e-3, 2e-3]
    assert all(k.duration == 1e-7 and k.K == 5 and k.θ == 0 for k in kicks)
    with pytest.raises(AssertionError):
        generate_kick_list(num_kicks=2, starts=[0.0], durations=1e-7, strengths=1, angles=0)
    assert generate_kick_list() == []


def test_cross_check_kick_indices():
    assert list(cross_check_kick_indices([100, 500, 900], [98, 503, 400], tolerance=5)) == [100, 500]


def test_trace_time_array_matches_values():
    trace = Trace(np.zeros(1000), 0.0, 1e-3, 1e-6, 'x', 'm', None)
    assert len(trace.time_array) == trace.num_samples
    assert trace.time_array[1] == pytest.approx(1e-6)


def test_experiment_loads_mat_and_bin(tmp_path):
    rng = np.random.default_rng(1)
    n = 4096
    channels = {c: rng.standard_normal((n, 1)) for c in 'ABD'}
    mat = tmp_path / 'run.mat'
    savemat(mat, {'Length': np.array([[n]]), 'Tinterval': np.array([[1e-6]]), **channels})
    exp = Experiment(str(mat))
    assert set(exp.traces) == {'x', 'y', 'kick'}
    assert np.allclose(exp.traces['x'].values + exp.traces['x'].mean, channels['B'][:, 0])

    data = rng.integers(-1000, 1000, size=4 * n, dtype=np.int16)
    binfile = tmp_path / 'run.bin'
    data.tofile(binfile)
    (tmp_path / 'streaming_parameter.log').write_text('x;1000000;4;0;0\n')
    exp = Experiment(str(binfile))
    assert exp.num_samples == n
    assert np.array_equal(exp.traces['kick'].values, data[3::4])
    with pytest.raises(ValueError):
        Experiment(str(tmp_path / 'run.txt'))
