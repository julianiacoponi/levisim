"""
The second-order expansion of the Richards-Wolf focal field (levisim.physics.focal_expansion)

It is checked against finite differences of FocussingOptics' full focal field, and for the
symmetry statements the two-tone trap model is built on: light is softer along its polarisation,
and two orthogonal polarisations interfere only in an x-y term equal to the anisotropy.
"""
import numpy as np
import pytest

from levisim.core.constants import π
from levisim.physics.focal_expansion import FocalExpansion, polarisation_matrix
from levisim.physics.focussing_optics import FocussingOptics


def _expansion(NA, apodisation='levisim'):
    reference = FocussingOptics(NA=NA)
    return FocalExpansion(FocussingOptics(NA=NA, w0=reference.w0 / reference.f0), apodisation=apodisation)


@pytest.mark.parametrize('NA', [0.5, 0.77, 0.9])
@pytest.mark.parametrize('ψ1, ψ2', [(0.0, 0.0), (π/2, π/2), (0.0, π/2), (π/4, π/4), (π/6, 2*π/3)])
def test_closed_forms_match_the_full_focal_field(NA, ψ1, ψ2):
    fe = _expansion(NA)
    analytic, numeric = fe.interference_hessian(ψ1, ψ2), fe.numerical_hessian(ψ1, ψ2)
    assert np.abs(analytic - numeric).max() < 2e-3 * np.abs(analytic).max()


@pytest.mark.parametrize('apodisation', ['levisim', 'novotny'])
@pytest.mark.parametrize('NA', [0.4, 0.6, 0.8, 0.95])
def test_the_focus_is_softer_along_the_polarisation(NA, apodisation):
    κ = _expansion(NA, apodisation).curvatures
    assert 0 < κ['par'] < κ['perp']
    assert 0 < κ['z'] < κ['par']


def test_orthogonal_polarisations_interfere_only_in_xy():
    fe = _expansion(0.77)
    H = fe.interference_hessian(0.0, π/2)
    assert np.allclose(np.diag(H), 0.0, atol=1e-15)
    assert np.allclose(H[:2, 2], 0.0) and np.allclose(H[2, :2], 0.0)
    # E^(x)·E^(y)* ≈ k²(κ_⊥ - κ_∥)·xy, whose Hessian has xy entry κ_⊥ - κ_∥
    assert H[0, 1] == pytest.approx(fe.anisotropy, rel=1e-12)
    assert fe.anisotropy > 0


def test_rotating_the_polarisation_rotates_the_trap():
    fe = _expansion(0.77)
    H0 = fe.intensity_hessian(0.0)[:2, :2]
    for ψ in (0.3, 1.1, 2.0):
        R = np.array([[np.cos(ψ), -np.sin(ψ)], [np.sin(ψ), np.cos(ψ)]])
        assert np.allclose(fe.intensity_hessian(ψ)[:2, :2], R @ H0 @ R.T, atol=1e-14)


def test_polarisation_matrix():
    assert np.allclose(polarisation_matrix(0.0), np.diag([-1.0, 1.0]))
    assert np.allclose(polarisation_matrix(π/2), [[0.0, -1.0], [-1.0, 0.0]])
