# -*- coding: utf-8 -*-
r"""The Wiener deconvolution against its closed forms and its own error bars.

The estimator is checked three ways: against the covariance form of the
filter, :math:`\mathbf C_E \mathbf T^\dagger (\mathbf T \mathbf C_E \mathbf
T^\dagger + \mathbf C_N)^{-1}`, written independently of the code; in its
limits (no noise, no prior, no signal); and by drawing fields and noise from
the priors, where the mean squared error must equal the variance it reports.
"""

import numpy as np
import pytest
import scipy.fft as sf

from grand.basis.validate import GRANDlibWarning
from grand.recon.deconvolution import Response, WienerDeconvolution
from grand.recon.deconvolution.base import time_variance

N_FREQ = 64


def random_complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def random_response(rng, n_freq=N_FREQ, **geometry):
    return Response(transfer=random_complex(rng, (n_freq, 3, 3)),
                    noise_psd=rng.uniform(0.5, 2.0, (n_freq, 3)), **geometry)


def covariance_form(response, signal_cov, voltage_f):
    r"""The filter written as :math:`C_E T^\dagger (T C_E T^\dagger + C_N)^{-1} V`."""
    T = response.transfer
    Th = T.conj().swapaxes(1, 2)
    C_N = np.einsum("fk,kl->fkl", response.noise_psd, np.eye(3))
    W = signal_cov @ Th @ np.linalg.inv(T @ signal_cov @ Th + C_N)
    return np.einsum("fjk,fk->fj", W, voltage_f)


@pytest.fixture
def rng():
    return np.random.default_rng(1)


def test_scalar_formula(rng):
    h = random_complex(rng, N_FREQ)
    noise, signal = 0.7, 3.0
    response = Response(transfer=np.einsum("f,jk->fjk", h, np.eye(3)),
                        noise_psd=np.full((N_FREQ, 3), noise))
    voltage_f = random_complex(rng, (N_FREQ, 3))
    result = WienerDeconvolution(signal, mode="full3d").deconvolve_spectrum(voltage_f, response)
    expected = (h.conj() * signal / (np.abs(h) ** 2 * signal + noise))[:, np.newaxis] * voltage_f
    np.testing.assert_allclose(result.efield, expected, rtol=1e-12)
    np.testing.assert_allclose(result.variance[:, 0], signal * noise / (np.abs(h) ** 2 * signal + noise))


@pytest.mark.parametrize("mode", ["full3d", "transverse", "polarized"])
def test_matches_covariance_form(rng, mode):
    p = np.array([0.3, -0.5, 0.8])
    n = np.cross(p, [1.0, 0.0, 0.0])
    response = random_response(rng, polarization=p, line_of_sight=n)
    signal_psd = rng.uniform(0.5, 5.0, N_FREQ)
    voltage_f = random_complex(rng, (N_FREQ, 3))

    wiener = WienerDeconvolution(signal_psd, mode=mode)
    basis = wiener.basis(response)
    signal_cov = np.einsum("f,jk->fjk", signal_psd, basis @ basis.T)
    result = wiener.deconvolve_spectrum(voltage_f, response)

    np.testing.assert_allclose(result.efield, covariance_form(response, signal_cov, voltage_f),
                               rtol=1e-10, atol=1e-12)
    assert result.valid.all()


def test_transverse_has_no_component_along_line_of_sight(rng):
    n = np.array([1.0, 2.0, -0.5])
    response = random_response(rng, line_of_sight=n)
    result = WienerDeconvolution(2.0, mode="transverse").deconvolve_spectrum(
        random_complex(rng, (N_FREQ, 3)), response)
    np.testing.assert_allclose(result.efield @ response.line_of_sight, 0, atol=1e-12)


def test_no_prior_is_the_exact_inverse(rng):
    response = random_response(rng)
    efield_f = random_complex(rng, (N_FREQ, 3))
    voltage_f = np.einsum("fkj,fj->fk", response.transfer, efield_f)
    result = WienerDeconvolution(np.inf, mode="full3d").deconvolve_spectrum(voltage_f, response)
    np.testing.assert_allclose(result.efield, efield_f, rtol=1e-9)


def test_polarized_recovers_a_noiseless_field(rng):
    p = np.array([0.0, 0.6, 0.8])
    response = random_response(rng, polarization=p)
    amplitude = random_complex(rng, N_FREQ)
    voltage_f = np.einsum("fkj,j,f->fk", response.transfer, p, amplitude)
    result = WienerDeconvolution(np.inf).deconvolve_spectrum(voltage_f, response)
    np.testing.assert_allclose(result.efield, np.outer(amplitude, p), rtol=1e-9, atol=1e-12)


def test_vanishing_prior_gives_zero(rng):
    response = random_response(rng, polarization=[1, 0, 0])
    result = WienerDeconvolution(1e-12).deconvolve_spectrum(random_complex(rng, (N_FREQ, 3)),
                                                            response)
    assert np.max(np.abs(result.efield)) < 1e-9


def test_bins_without_response_are_left_out(rng):
    response = random_response(rng, polarization=[0, 1, 0])
    response.transfer[:10] = 0
    voltage_f = random_complex(rng, (N_FREQ, 3))

    without_prior = WienerDeconvolution(np.inf).deconvolve_spectrum(voltage_f, response)
    assert not without_prior.valid[:10].any() and without_prior.valid[10:].all()
    assert np.all(without_prior.efield[:10] == 0) and np.all(without_prior.variance[:10] == 0)

    # With a prior every bin is defined: the estimate falls to zero, the
    # variance to the prior
    with_prior = WienerDeconvolution(2.0).deconvolve_spectrum(voltage_f, response)
    assert with_prior.valid.all()
    np.testing.assert_allclose(with_prior.efield[:10], 0)
    np.testing.assert_allclose(with_prior.variance[:10, 1], 2.0)


def test_a_large_prior_keeps_every_bin(rng):
    # 1/P_e far below the largest eigenvalue over the bins is no reason to drop
    # a bin: the prior defines it, here where the response vanishes as well
    response = random_response(rng, polarization=[0, 1, 0])
    response.transfer[:10] = 0
    result = WienerDeconvolution(1e15).deconvolve_spectrum(random_complex(rng, (N_FREQ, 3)),
                                                           response)
    assert result.valid.all()
    np.testing.assert_allclose(result.variance[:10, 1], 1e15)


def test_a_per_frequency_prior_may_be_partly_infinite(rng):
    response = random_response(rng, polarization=[0, 1, 0])
    response.transfer[:10] = 0
    signal_psd = np.full(N_FREQ, np.inf)
    signal_psd[:5] = 2.0
    result = WienerDeconvolution(signal_psd).deconvolve_spectrum(random_complex(rng, (N_FREQ, 3)),
                                                                 response)
    assert result.valid[:5].all() and not result.valid[5:10].any() and result.valid[10:].all()
    np.testing.assert_allclose(result.variance[:5, 1], 2.0)


def test_frequencies_out_of_band_are_left_out(rng):
    response = random_response(rng, polarization=[0, 1, 0])
    band = np.arange(N_FREQ) >= 10
    noise_psd = response.noise_psd.copy()
    noise_psd[~band] = 0                # not read there
    restricted = Response(response.transfer, noise_psd, polarization=[0, 1, 0], band=band)
    voltage_f = random_complex(rng, (N_FREQ, 3))

    for signal_psd in (2.0, np.inf, "auto"):
        result = WienerDeconvolution(signal_psd).deconvolve_spectrum(voltage_f, restricted)
        np.testing.assert_array_equal(result.valid, band)
        assert np.all(result.efield[~band] == 0) and np.all(result.variance[~band] == 0)
        assert np.all(result.variance[band, 1] > 0)

    # In the band, the same answer as without the restriction
    np.testing.assert_allclose(
        WienerDeconvolution(2.0).deconvolve_spectrum(voltage_f, restricted).efield[band],
        WienerDeconvolution(2.0).deconvolve_spectrum(voltage_f, response).efield[band])


def test_noise_must_be_positive_in_the_band_only(rng):
    transfer = random_complex(rng, (N_FREQ, 3, 3))
    noise_psd = np.ones((N_FREQ, 3))
    noise_psd[:10, 1] = 0
    with pytest.raises(ValueError, match="noise_psd"):
        Response(transfer, noise_psd)
    Response(transfer, noise_psd, band=np.arange(N_FREQ) >= 10)
    with pytest.raises(ValueError, match="band"):
        Response(transfer, np.ones((N_FREQ, 3)), band=np.ones(N_FREQ - 1, dtype=bool))


@pytest.mark.parametrize("mode", ["transverse", "polarized"])
@pytest.mark.parametrize("level", [0.5, 30.0])
def test_the_flat_level_is_recovered(rng, mode, level):
    draws = 40000
    p = np.array([0.6, 0.0, 0.8])
    n = np.array([0.8, 0.0, -0.6])
    transfer = random_complex(rng, (draws, 3, 3))
    noise_psd = rng.uniform(0.5, 2.0, (draws, 3))
    response = Response(transfer, noise_psd, polarization=p, line_of_sight=n)
    wiener = WienerDeconvolution("auto", mode=mode)
    basis = wiener.basis(response)
    amplitude = np.sqrt(level / 2) * random_complex(rng, (draws, basis.shape[1]))
    voltage_f = np.einsum("fkj,fj->fk", transfer, amplitude @ basis.T)
    voltage_f = voltage_f + np.sqrt(noise_psd / 2) * random_complex(rng, (draws, 3))
    assert wiener.estimate_flat_level(voltage_f, response) == pytest.approx(level, rel=0.05)


def test_the_flat_level_of_pure_noise_stays_finite(rng):
    noise_psd = np.ones((N_FREQ, 3))
    response = Response(random_complex(rng, (N_FREQ, 3, 3)), noise_psd, polarization=[1, 0, 0])
    wiener = WienerDeconvolution("auto")
    # Below the expected noise power the excess is negative: the level is then
    # the uncertainty of the excess, sqrt(3 N_FREQ) over the information
    gain = response.transfer @ wiener.basis(response)
    expected = np.sqrt(3 * N_FREQ) / np.sum(np.abs(gain) ** 2)
    assert wiener.estimate_flat_level(np.zeros((N_FREQ, 3)), response) == pytest.approx(expected)
    # So the error bar stays one of a real prior, not a false certainty of zero
    result = wiener.deconvolve_spectrum(np.zeros((N_FREQ, 3)), response)
    information = np.sum(np.abs(gain) ** 2, axis=(1, 2))
    np.testing.assert_allclose(result.variance[:, 0], 1 / (information + 1 / expected))


def test_the_flat_level_without_response_is_no_prior(rng):
    response = Response(np.zeros((N_FREQ, 3, 3)), np.ones((N_FREQ, 3)), polarization=[1, 0, 0])
    assert WienerDeconvolution("auto").estimate_flat_level(np.ones((N_FREQ, 3)), response) == np.inf


def test_nothing_to_estimate_says_so(rng):
    response = random_response(rng, polarization=[0, 1, 0])
    response.transfer[:] = 0
    with pytest.warns(GRANDlibWarning, match="no frequency can be estimated"):
        result = WienerDeconvolution(np.inf).deconvolve_spectrum(np.zeros((N_FREQ, 3)), response)
    assert not result.valid.any()


@pytest.mark.parametrize("mode", ["full3d", "polarized"])
def test_variance_is_the_mean_squared_error(rng, mode):
    # Many bins sharing one response: each is an independent draw of field and
    # noise from the priors, so the error averaged over bins must equal the
    # posterior variance the filter reports.
    draws = 40000
    p = np.array([0.6, 0.0, 0.8])
    transfer = random_complex(rng, (3, 3))
    noise_psd = np.array([0.5, 1.0, 2.0])
    response = Response(transfer=np.broadcast_to(transfer, (draws, 3, 3)),
                        noise_psd=np.broadcast_to(noise_psd, (draws, 3)), polarization=p)
    wiener = WienerDeconvolution(3.0, mode=mode)
    basis = wiener.basis(response)

    amplitude = np.sqrt(3.0 / 2) * random_complex(rng, (draws, basis.shape[1]))
    efield_f = amplitude @ basis.T
    noise = np.sqrt(noise_psd / 2) * random_complex(rng, (draws, 3))
    voltage_f = efield_f @ transfer.T + noise

    result = wiener.deconvolve_spectrum(voltage_f, response)
    mse = np.mean(np.abs(result.efield - efield_f) ** 2, axis=0)
    np.testing.assert_allclose(mse, result.variance[0], rtol=0.03, atol=1e-12)


@pytest.mark.parametrize("n_fft", [256, 257])
def test_time_variance_of_white_noise(n_fft):
    # White noise of variance s2 has E|X_k|^2 = n_fft * s2 in every bin
    s2 = 2.5
    np.testing.assert_allclose(time_variance(np.full(n_fft // 2 + 1, n_fft * s2), n_fft), s2)


@pytest.mark.parametrize("n_fft", [None, 400])
def test_time_domain_round_trip(rng, n_fft):
    n_samples = 300
    n_freq = (n_fft or n_samples) // 2 + 1
    t = np.arange(n_samples)
    pulse = np.exp(-0.5 * ((t - 120) / 4.0) ** 2) * np.sin(0.3 * t)
    p = np.array([0.2, 0.9, -0.4])
    p /= np.linalg.norm(p)
    efield = np.outer(p, pulse)

    # The response of a real filter: a real mixing of the components and a
    # delay of 5 samples, real at 0 and at the Nyquist frequency.  A complex
    # constant there would lose the imaginary part to irfft.
    delay = np.exp(-2j * np.pi * 5 * np.arange(n_freq) / (n_fft or n_samples))
    response = Response(transfer=np.einsum("f,jk->fjk", delay, rng.normal(size=(3, 3))),
                        noise_psd=np.ones((n_freq, 3)), polarization=p)
    voltage_f = np.einsum("fkj,jf->kf", response.transfer, sf.rfft(efield, n=n_fft, axis=1))
    voltage = sf.irfft(voltage_f, n=n_fft or n_samples, axis=1)[:, :n_samples]

    recovered, sigma = WienerDeconvolution(np.inf).deconvolve(voltage, response, n_fft=n_fft)
    np.testing.assert_allclose(recovered, efield, atol=1e-10)
    assert sigma.shape == (3,) and np.all(sigma > 0)


def test_refuses_bad_input(rng):
    response = random_response(rng)
    with pytest.raises(ValueError, match="polarization"):
        WienerDeconvolution(1.0).deconvolve_spectrum(np.zeros((N_FREQ, 3)), response)
    with pytest.raises(ValueError, match="line_of_sight"):
        WienerDeconvolution(1.0, mode="transverse").deconvolve_spectrum(np.zeros((N_FREQ, 3)),
                                                                        response)
    with pytest.raises(ValueError, match="signal_psd"):
        WienerDeconvolution(0.0)
    with pytest.raises(ValueError, match="signal_psd"):
        WienerDeconvolution("flat")
    with pytest.raises(ValueError, match="mode"):
        WienerDeconvolution(1.0, mode="2d")
    with pytest.raises(ValueError, match="signal_psd"):
        WienerDeconvolution(np.ones(N_FREQ - 1), mode="full3d").deconvolve_spectrum(
            np.zeros((N_FREQ, 3)), response)
    with pytest.raises(ValueError, match="noise_psd"):
        Response(transfer=response.transfer, noise_psd=np.zeros((N_FREQ, 3)))
    with pytest.raises(ValueError, match="frequencies"):
        WienerDeconvolution(1.0, mode="full3d").deconvolve(np.zeros((3, 100)), response)
