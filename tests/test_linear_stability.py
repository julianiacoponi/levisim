"""
The linear-stability report of MultiTrap: drift_eigenvalues, max_growth_rate, is_stable,
least_stable_mode, unstable_modes, and the construction WARNING built from them
"""
import numpy as np
import pytest
from loguru import logger

from levisim.core.units import degrees, kHz
from levisim.multiphysics.covariance import steady_state_covariance
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.shelf import DEFAULT_cavity_kwargs, DEFAULT_gas_kwargs, OFF_NODE_toy_params_pairs


def default_2x2(φ2=None, **kwargs):
    if φ2 is not None:
        kwargs['φn_dict'] = {2: φ2}
    return MultiTrap(
        num_traps=2, Cartesian_axes_per_trap=2, gas_kwargs=DEFAULT_gas_kwargs, cavity_kwargs=DEFAULT_cavity_kwargs,
        ω_range=np.array([0.0]), **kwargs,
    )


def same_eigenvalues(a, b, tol):
    """a and b hold the same eigenvalues to within tol (in any order; a pair λ, λ* with equal real
    parts can come out in either order)"""
    return len(a) == len(b) and all(np.min(np.abs(b - λ)) <= tol for λ in a) and all(np.min(np.abs(a - λ)) <= tol for λ in b)


def test_built_in_phases_give_one_growing_mode():
    mtrap = default_2x2()
    assert not mtrap.is_stable
    assert mtrap.max_growth_rate == pytest.approx(2677.5, rel=1e-4)
    [(Ω, rate)] = mtrap.unstable_modes
    assert Ω / kHz == pytest.approx(134.92, abs=0.01)
    # the two members of a conjugate pair agree only to round-off, and either can come first
    assert rate == pytest.approx(mtrap.max_growth_rate, rel=1e-9)
    assert mtrap.least_stable_mode == pytest.approx((Ω, rate), rel=1e-9)
    with pytest.raises(Exception):
        steady_state_covariance(mtrap)


def test_thirty_degrees_is_stable():
    mtrap = default_2x2(30*degrees)
    assert mtrap.is_stable and mtrap.unstable_modes == []
    assert mtrap.max_growth_rate == pytest.approx(-294.0, rel=1e-3)
    assert mtrap.least_stable_mode[0] / kHz == pytest.approx(136.53, abs=0.01)


def test_eigenvalues_are_those_of_the_drift_matrix_sorted_and_in_conjugate_pairs():
    mtrap = default_2x2()
    λ = mtrap.drift_eigenvalues
    assert len(λ) == mtrap.num_operators
    assert np.all(np.diff(λ.real) <= 0)
    tol = 1e-9 * np.abs(λ).max()
    assert same_eigenvalues(λ, np.linalg.eigvals(mtrap.drift_matrix), tol)
    # every eigenvalue has its conjugate partner (a mode and its creation-operator image)
    assert same_eigenvalues(λ, np.conjugate(λ), tol)


@pytest.mark.parametrize('φ2_deg, stable', [(54.3, True), (54.5, False), (125.5, False), (125.7, True)])
def test_phase_window_edges(φ2_deg, stable):
    assert default_2x2(φ2_deg*degrees).is_stable is stable


def test_phase_window_repeats_every_180_degrees():
    for φ2 in (10, 60, 90, 150):
        a = default_2x2(φ2*degrees).drift_eigenvalues
        b = default_2x2((φ2 + 180)*degrees).drift_eigenvalues
        assert same_eigenvalues(a, b, 1e-9 * np.abs(a).max())


def test_static_instability_is_one_real_mode():
    mtrap = MultiTrap(
        num_traps=1, Cartesian_axes_per_trap=3, toy_params=OFF_NODE_toy_params_pairs,
        gas_kwargs=DEFAULT_gas_kwargs, cavity_kwargs=DEFAULT_cavity_kwargs, ω_range=np.array([0.0]),
    )
    [(Ω, rate)] = mtrap.unstable_modes
    assert Ω == 0.0 and rate > 1e5


def test_warning_and_info_use_the_report():
    messages = []
    sink = logger.add(messages.append, level='WARNING', format='{message}')
    try:
        unstable = default_2x2()
        stable = default_2x2(30*degrees)
    finally:
        logger.remove(sink)
    [message] = messages
    assert f'{len([λ for λ in unstable.drift_eigenvalues if λ.real > unstable.stability_tolerance])} of the 10' in message
    assert '134.92 kHz' in message and '+2678 /s' in message
    assert 'linearly_stable = False' in unstable.info_str
    assert 'unstable mode at 134.922 kHz grows at Re λ = +2678 /s' in unstable.info_str
    assert 'linearly_stable = True' in stable.info_str and 'unstable mode' not in stable.info_str


def test_photon_recoil_does_not_change_the_report():
    on, off = default_2x2(), default_2x2(photon_recoil=False)
    assert np.array_equal(on.drift_eigenvalues, off.drift_eigenvalues)
    assert on.unstable_modes == off.unstable_modes
