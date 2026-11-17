# -*- coding: utf-8 -*-
r"""The noise models against realizations of the noise, and through a whole shower.

The models are compared with the mean of many draws, and then with the noise
``Efield2Voltage`` adds to the ZHAireS shower committed in ``sim2root/Common``.
On that shower the filter's error bars must be the error it actually makes: the
one check that needs the noise model, the transfer matrix and the propagation of
the variance all to be right.
"""

import pathlib

import numpy as np
import pytest
import scipy.fft as sf
import scipy.signal as ss

import grand
from grand import Efield2Voltage
from grand.recon.deconvolution import (DetectorResponse, GalacticNoise, MeasuredNoise, WhiteNoise,
                                       WienerDeconvolution)
from grand.recon.deconvolution.base import time_variance
from grand.sim.noise.galaxy import galactic_noise, galactic_noise_psd

SAMPLE = (pathlib.Path(grand.__file__).parents[1]
          / "sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")
LST = 18.0


@pytest.fixture
def rng():
    return np.random.default_rng(1)


@pytest.mark.parametrize("n_samples", [1000, 1001])
def test_the_galactic_psd_is_what_is_drawn_from(n_samples):
    # 500 MHz: for an even length, the Nyquist bin falls in the band of the tables
    freqs_mhz = sf.rfftfreq(n_samples, 1 / 500.0)
    draws = galactic_noise(LST, n_samples, freqs_mhz, nb_ant=4000, seed=3)
    psd = galactic_noise_psd(LST, n_samples, freqs_mhz)
    assert psd.shape == (freqs_mhz.size, 3)
    in_band = psd[:, 0] > 0
    mean = np.mean(np.abs(draws) ** 2, axis=0).T
    np.testing.assert_allclose(mean[in_band].mean(axis=0) / psd[in_band].mean(axis=0), 1, atol=0.01)
    if n_samples % 2 == 0:
        np.testing.assert_allclose(mean[-1] / psd[-1], 1, atol=0.1)
    assert np.all(psd[~in_band] == 0)


def test_the_galactic_psd_of_part_of_the_axis():
    # A band alone is not a complete grid: no self-conjugate bin to double
    freqs_mhz = np.arange(30.0, 251.0)
    psd = galactic_noise_psd(LST, 2048, freqs_mhz)
    draws = galactic_noise(LST, 2048, freqs_mhz, nb_ant=4000, seed=3)
    np.testing.assert_allclose(np.mean(np.abs(draws[:, :, 100]) ** 2, axis=0) / psd[100], 1,
                               atol=0.1)
    with pytest.raises(ValueError, match="f_lst"):
        galactic_noise_psd(24.0, 2048, freqs_mhz)
    with pytest.raises(ValueError, match="du_type"):
        galactic_noise_psd(LST, 2048, freqs_mhz, du_type="Horizon")


def test_the_galactic_model_goes_through_the_rf_chain():
    n_fft = 1024
    freqs_mhz = sf.rfftfreq(n_fft, 1 / 500.0)
    rf_tf = np.exp(-freqs_mhz / 100.0) * np.array([[1.0], [2.0], [3.0j]])
    model = GalacticNoise(LST)
    psd = model.psd(freqs_mhz, n_fft, rf_tf)
    np.testing.assert_allclose(psd[:, 1], 4 * galactic_noise_psd(LST, n_fft, freqs_mhz)[:, 1]
                               * np.exp(-freqs_mhz / 50.0))
    in_band = psd[:, 0] > 0
    assert 29 < freqs_mhz[in_band][0] < 31 and 249 < freqs_mhz[in_band][-1] <= 250

    # A new length is a new normalisation, not a stale table
    longer = sf.rfftfreq(2 * n_fft, 1 / 500.0)
    assert model.psd(longer, 2 * n_fft, np.ones((3, longer.size))).shape == (longer.size, 3)
    np.testing.assert_allclose(model.psd(freqs_mhz, n_fft, rf_tf), psd)


def test_white_noise_is_its_periodogram(rng):
    n_fft = 512
    sigma = np.array([1.0, 2.0, 3.0])
    traces = rng.normal(size=(2000, 3, n_fft)) * sigma[:, np.newaxis]
    freqs_mhz = sf.rfftfreq(n_fft, 1 / 500.0)
    psd = WhiteNoise(sigma).psd(freqs_mhz, n_fft, None)
    mean = np.mean(np.abs(sf.rfft(traces, axis=2)) ** 2, axis=0).T
    np.testing.assert_allclose(mean[1:-1].mean(axis=0) / psd[0], 1, atol=0.01)
    np.testing.assert_allclose(WhiteNoise(2.0).psd(freqs_mhz, n_fft, None), n_fft * 4.0)
    for bad in (0.0, [1.0, 2.0], [1.0, -2.0, 3.0]):
        with pytest.raises(ValueError, match="sigma"):
            WhiteNoise(bad)


def test_measured_noise_of_white_noise(rng):
    sigma = np.array([1.0, 2.0, 3.0])
    traces = rng.normal(size=(50, 3, 4096)) * sigma[:, np.newaxis]
    n_fft = 2048
    freqs_mhz = sf.rfftfreq(n_fft, 2e-3)
    psd = MeasuredNoise(traces, 2.0, n_segment=256).psd(freqs_mhz, n_fft, None)
    np.testing.assert_allclose(psd[10:-10].mean(axis=0) / (n_fft * sigma ** 2), 1, atol=0.01)


def test_measured_noise_follows_the_shape_of_the_spectrum(rng):
    # A band-pass filter: the estimate has to recover its known response
    taps = ss.firwin(101, [50, 150], pass_zero=False, fs=500.0)
    traces = ss.lfilter(taps, 1.0, rng.normal(size=(40, 3, 8192)), axis=2)[:, :, 200:]
    n_fft = 1024
    freqs_mhz = sf.rfftfreq(n_fft, 2e-3)
    psd = MeasuredNoise(traces, 2.0, n_segment=512).psd(freqs_mhz, n_fft, None)
    expected = n_fft * np.abs(ss.freqz(taps, 1.0, worN=freqs_mhz, fs=500.0)[1]) ** 2
    bins = (freqs_mhz > 60) & (freqs_mhz < 140)
    np.testing.assert_allclose(psd[bins, 0] / expected[bins], 1, atol=0.1)
    assert psd[freqs_mhz > 200, 0].max() < 1e-3 * psd[bins, 0].mean()


def test_measured_noise_of_one_trace_and_beyond_its_nyquist(rng):
    model = MeasuredNoise(rng.normal(size=(3, 1024)), 2.0)   # 500 MHz
    freqs_mhz = sf.rfftfreq(1024, 0.5e-3)                    # 2 GHz
    psd = model.psd(freqs_mhz, 1024, None)
    assert np.all(psd[freqs_mhz > 250] == 0) and np.all(psd[(freqs_mhz > 1) & (freqs_mhz < 250)] > 0)


def test_measured_noise_refuses_bad_input(rng):
    with pytest.raises(ValueError, match="traces"):
        MeasuredNoise(rng.normal(size=(4, 100)), 2.0)
    with pytest.raises(ValueError, match="n_segment"):
        MeasuredNoise(rng.normal(size=(3, 100)), 2.0, n_segment=200)
    with pytest.raises(ValueError, match="t_bin_ns"):
        MeasuredNoise(rng.normal(size=(3, 100)), 0.0)


def simulate(directory, noise):
    sim = Efield2Voltage(str(SAMPLE), "voltage.root", output_directory=str(directory), seed=1,
                         efield_level=0)
    sim.params.update(add_noise=noise, add_rf_chain=True, lst=LST)
    sim.compute_voltage_event(0)
    sim.final_resample()
    return sim


@pytest.fixture(scope="module")
def shower(tmp_path_factory):
    directory = tmp_path_factory.mktemp("noise")
    clean, noisy = simulate(directory, False), simulate(directory, True)
    detector = DetectorResponse(antenna_model=clean.ant_model)
    model = GalacticNoise(LST)
    responses = [detector.response(clean.du_pos[du], clean.evt_shower, clean.freqs_mhz, model)
                 for du in range(clean.nb_du)]
    return clean, noisy, detector, responses


def visible_field(clean, du, response):
    r"""The field the antenna sees, in the time domain: transverse, in the band."""
    n = response.line_of_sight
    field_f = sf.rfft(clean.traces[du].astype(float), n=clean.fft_size, axis=1).T
    field_f = (field_f @ (np.eye(3) - np.outer(n, n))) * response.band[:, np.newaxis]
    return sf.irfft(field_f.T, n=clean.fft_size, axis=1)


def recovered(wiener, noisy, du, response):
    result = wiener.deconvolve_spectrum(noisy.vout_f[du].T, response)
    return (sf.irfft(result.efield.T, n=noisy.fft_size, axis=1),
            np.sqrt(time_variance(result.variance, noisy.fft_size)))


def test_the_band_is_where_the_tables_and_the_chain_are(shower):
    clean, _, _, responses = shower
    band = responses[0].band
    assert 30 <= clean.freqs_mhz[band][0] < 31
    assert 249 < clean.freqs_mhz[band][-1] < 250      # the chain is zero at 250 MHz
    np.testing.assert_array_equal(np.flatnonzero(band), np.arange(*np.flatnonzero(band)[[0, -1]]
                                                                  + [0, 1]))


def test_the_model_is_the_noise_of_the_simulation(shower):
    clean, noisy, _, responses = shower
    noise_f = noisy.vout_f - clean.vout_f
    band = responses[0].band
    ratio = (np.mean(np.abs(noise_f) ** 2, axis=0).T[band] / responses[0].noise_psd[band])
    np.testing.assert_allclose(ratio.mean(axis=0), 1, atol=0.02)


def test_the_measured_noise_finds_the_same_level(shower):
    clean, noisy, detector, responses = shower
    model = MeasuredNoise(noisy.vout - clean.vout, clean.dt_ns[0], n_segment=512)
    measured = detector.response(clean.du_pos[0], clean.evt_shower, clean.freqs_mhz, model)
    # Away from the edges of the band, where the chain varies fast within a bin of the estimate
    bins = (clean.freqs_mhz > 50) & (clean.freqs_mhz < 200)
    ratio = measured.noise_psd[bins] / responses[0].noise_psd[bins]
    np.testing.assert_allclose(np.median(ratio, axis=0), 1, atol=0.05)


def test_error_bars_without_prior_are_the_error_made(shower):
    clean, noisy, _, responses = shower
    wiener = WienerDeconvolution(np.inf, mode="transverse")
    ratios = []
    for du, response in enumerate(responses):
        efield, sigma = recovered(wiener, noisy, du, response)
        error = (efield - visible_field(clean, du, response)).std(axis=1)
        ratios.append(sigma / error)
    # Each from thousands of correlated samples, dominated by the band's edges
    np.testing.assert_allclose(np.median(ratios, axis=0), 1, atol=0.05)


def test_no_prior_leaves_the_peak_of_a_strong_trace_alone(shower):
    clean, noisy, _, responses = shower
    wiener = WienerDeconvolution(np.inf, mode="transverse")
    du = int(np.argmax(np.abs(clean.vout).max(axis=(1, 2))))
    truth = np.linalg.norm(visible_field(clean, du, responses[du]), axis=0)
    efield, _ = recovered(wiener, noisy, du, responses[du])
    peak = truth.argmax()
    near = np.linalg.norm(efield, axis=0)[peak - 40:peak + 40].max()
    assert near == pytest.approx(truth[peak], rel=0.1)


def test_the_flat_level_lowers_the_error_on_every_unit(shower):
    clean, noisy, _, responses = shower
    inverse = WienerDeconvolution(np.inf, mode="transverse")
    auto = WienerDeconvolution("auto", mode="transverse")
    for du, response in enumerate(responses):
        truth = visible_field(clean, du, response)
        without, _ = recovered(inverse, noisy, du, response)
        with_level, _ = recovered(auto, noisy, du, response)
        assert np.linalg.norm(with_level - truth) < np.linalg.norm(without - truth)


def test_the_flat_level_shrinks_the_peak(shower):
    # Not what one would want, but what this prior does: see WienerDeconvolution.
    clean, noisy, _, responses = shower
    auto = WienerDeconvolution("auto", mode="transverse")
    du = int(np.argmax(np.abs(clean.vout).max(axis=(1, 2))))
    truth = np.linalg.norm(visible_field(clean, du, responses[du]), axis=0)
    efield, _ = recovered(auto, noisy, du, responses[du])
    assert np.linalg.norm(efield, axis=0).max() < truth.max()


def test_white_noise_in_the_voltage_is_estimated_everywhere_the_antenna_responds(shower):
    clean, _, detector, responses = shower
    response = detector.response(clean.du_pos[0], clean.evt_shower, clean.freqs_mhz,
                                 WhiteNoise(100.0))
    assert np.count_nonzero(response.band) >= np.count_nonzero(responses[0].band)
    assert np.all(response.noise_psd[response.band] == clean.fft_size * 100.0 ** 2)
