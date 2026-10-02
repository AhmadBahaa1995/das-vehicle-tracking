import numpy as np
import pytest

from dastrack.fk import fk_candidates, representative_ridge, slant_stack

FS, DX = 250.0, 2.0419


def moving_streak(v_mps, nt=2501, nx=121, width_ch=2.0):
    """A Gaussian streak crossing the patch centre at velocity v (v > 0: toward higher channel)."""
    t = (np.arange(nt) - nt // 2) / FS
    x = np.arange(nx)
    centre = nx // 2 + v_mps * t / DX
    return np.exp(-0.5 * ((x[None, :] - centre[:, None]) / width_ch) ** 2) \
        * (1 + 0.5 * np.cos(2 * np.pi * 3 * t))[:, None]


@pytest.mark.parametrize("v_true", [-12.0, -7.0, 7.0, 12.0])
def test_slant_stack_recovers_signed_velocity(v_true):
    v_ax, score, *_ = slant_stack(moving_streak(v_true), FS, DX, v_max_mps=27)
    cands, contrast = fk_candidates(v_ax, score)
    v_best = cands[0][0]
    assert np.sign(v_best) == np.sign(v_true)
    assert abs(v_best - v_true) < 0.15 * abs(v_true)
    assert contrast > 1.5


@pytest.mark.parametrize("v_true", [-9.0, 9.0])
def test_energy_lies_on_f_equals_minus_vk(v_true):
    patch = moving_streak(v_true)
    _, _, fk_db, freqs, wvn = slant_stack(patch, FS, DX, v_max_mps=27)
    power = 10 ** (fk_db / 10)
    F, K = np.meshgrid(freqs, wvn, indexing='ij')
    upper = np.abs(F) > 0.2
    same_sign = power[upper & (F > 0) & (K > 0)].sum()
    opposite = power[upper & (F > 0) & (K < 0)].sum()
    # f = -v k: for v > 0 the f > 0 energy is at k < 0, and vice versa
    assert (opposite > same_sign) == (v_true > 0)

    pf, pk = representative_ridge(patch, FS, DX, v_true)
    assert pf > 0
    assert pf == pytest.approx(-v_true * pk, rel=1e-6)


def test_small_patch_returns_none():
    assert slant_stack(np.zeros((5, 5)), FS, DX)[0] is None
