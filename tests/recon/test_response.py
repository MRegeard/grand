# -*- coding: utf-8 -*-
r"""The transfer matrix against the simulation it has to undo, on a real shower.

The ZHAireS shower committed in ``sim2root/Common`` goes through
``Efield2Voltage`` without noise.  The matrix must reproduce what the simulation
applied, and the Wiener filter without a prior must then bring the field back:
all of it the antenna can see, that is its part in the band of the tables and
transverse to the line of sight.

The tolerances are limited by the simulation's single precision: about 1e-7 on
the spectra, and 1.1e-5 of the peak on the worst unit in the time domain.
"""

import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.fft as sf

import grand
from grand import Efield2Voltage, LTP
from grand.basis.validate import GRANDlibWarning
from grand.geo.coordinates import ECEF
from grand.recon.deconvolution import DetectorResponse, Response, WienerDeconvolution
from grand.sim.detector.process_ant import AntennaProcessing, antenna_frame

SAMPLE = (pathlib.Path(grand.__file__).parents[1]
          / "sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")


def simulate(directory, **params):
    r"""Returns the simulation of the sample's first event, without noise."""
    sim = Efield2Voltage(str(SAMPLE), "voltage.root", output_directory=str(directory), seed=1,
                         efield_level=0)
    sim.params.update(add_noise=False, add_rf_chain=True, **params)
    sim.compute_voltage_event(0)
    sim.final_resample()
    return sim


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    return simulate(tmp_path_factory.mktemp("response"))


@pytest.fixture(scope="module")
def detector(sim):
    return DetectorResponse(antenna_model=sim.ant_model)


def field_spectrum(sim, du):
    return sf.rfft(sim.traces[du].astype(float), n=sim.fft_size, axis=1).T


def visible_part(efield_f, response, valid):
    r"""The field the antenna sees: transverse, in the valid bins."""
    n = response.line_of_sight
    return (efield_f @ (np.eye(3) - np.outer(n, n))) * valid[:, np.newaxis]


def test_the_matrix_is_what_the_simulation_applies(sim, detector):
    for du in range(sim.nb_du):
        response = detector.response(sim.du_pos[du], sim.evt_shower, sim.freqs_mhz, 1.0)
        voltage_f = np.einsum("fkj,fj->fk", response.transfer, field_spectrum(sim, du))
        expected = sim.vout_f[du].T
        assert np.abs(voltage_f - expected).max() < 1e-6 * np.abs(expected).max()


def test_without_rf_chain_it_is_the_open_circuit_voltage(sim):
    detector = DetectorResponse(rf_chain=None, antenna_model=sim.ant_model)
    response = detector.response(sim.du_pos[0], sim.evt_shower, sim.freqs_mhz, 1.0)
    voltage_f = np.einsum("fkj,fj->fk", response.transfer, field_spectrum(sim, 0))
    expected = sim.voc_f[0].T
    assert np.abs(voltage_f - expected).max() < 1e-6 * np.abs(expected).max()


def test_the_antenna_does_not_see_along_the_line_of_sight(sim, detector):
    band = (sim.freqs_mhz > 50) & (sim.freqs_mhz < 200)
    for du in (0, sim.nb_du - 1):
        transfer, line_of_sight = detector.transfer_matrix(sim.du_pos[du], sim.evt_shower,
                                                           sim.freqs_mhz)
        singular = np.linalg.svd(transfer[band], compute_uv=False)
        assert np.all(singular[:, 2] < 1e-12 * singular[:, 0])
        assert np.all(singular[:, 1] > 0.1 * singular[:, 0])
        assert np.abs(transfer[band] @ line_of_sight).max() < 1e-12 * singular[:, 0].max()


def test_the_line_of_sight_goes_from_the_source_to_the_unit(sim, detector):
    du = 0
    _, line_of_sight = detector.transfer_matrix(sim.du_pos[du], sim.evt_shower, sim.freqs_mhz)
    shower = sim.evt_shower
    location = antenna_frame(sim.du_pos[du], shower.grand_ref_frame).location
    maximum = ECEF(shower.maximum)
    path = np.ravel(shower.frame.basis @ np.array([np.ravel(location.x - maximum.x),
                                                   np.ravel(location.y - maximum.y),
                                                   np.ravel(location.z - maximum.z)]))
    np.testing.assert_allclose(line_of_sight, path / np.linalg.norm(path), atol=1e-6)


def test_round_trip_in_time(sim, detector):
    wiener = WienerDeconvolution(np.inf, mode="transverse")
    for du in range(sim.nb_du):
        response = detector.response(sim.du_pos[du], sim.evt_shower, sim.freqs_mhz, 1.0)
        valid = wiener.deconvolve_spectrum(sim.vout_f[du].T, response).valid
        expected = sf.irfft(visible_part(field_spectrum(sim, du), response, valid).T,
                            n=sim.fft_size, axis=1)
        efield, sigma = wiener.deconvolve(sim.vout[du], response, n_fft=sim.fft_size)
        assert np.abs(efield - expected[:, :efield.shape[1]]).max() < 1e-4 * np.abs(expected).max()
        assert np.all(np.isfinite(sigma))


def test_the_recovered_band_is_the_band_of_the_tables(sim, detector):
    response = detector.response(sim.du_pos[0], sim.evt_shower, sim.freqs_mhz, 1.0)
    valid = WienerDeconvolution(np.inf, mode="transverse").deconvolve_spectrum(
        sim.vout_f[0].T, response).valid
    assert sim.freqs_mhz[valid][0] == pytest.approx(30, abs=sim.freqs_mhz[1])
    assert sim.freqs_mhz[valid][-1] == pytest.approx(250, abs=sim.freqs_mhz[1])


def test_round_trip_at_the_rate_of_the_detector(sim, detector, tmp_path):
    # 2 GHz to 500 MHz: the frequency axis stops at the top of the band
    low = simulate(tmp_path, resample_to_mhz=500)
    du = int(np.argmax(np.abs(sim.traces).max(axis=(1, 2))))
    n_fft = low.vout.shape[2]
    freqs_mhz = sf.rfftfreq(n_fft, 1 / 500.0)
    response = detector.response(low.du_pos[du], low.evt_shower, freqs_mhz, 1.0)
    wiener = WienerDeconvolution(np.inf, mode="transverse")
    efield, _ = wiener.deconvolve(low.vout[du], response, n_fft=n_fft)

    full = detector.response(sim.du_pos[du], sim.evt_shower, sim.freqs_mhz, 1.0)
    valid = wiener.deconvolve_spectrum(sim.vout_f[du].T, full).valid
    expected = sf.irfft(visible_part(field_spectrum(sim, du), full, valid).T,
                        n=sim.fft_size, axis=1)[:, ::4]
    assert np.abs(efield - expected).max() < 1e-3 * np.abs(expected).max()


def test_round_trip_of_a_polarized_field(tmp_path):
    # Three arms for one unknown, on a field that is truly linearly polarized
    sim = Efield2Voltage(str(SAMPLE), "voltage.root", output_directory=str(tmp_path), seed=1,
                         efield_level=0)
    sim.params.update(add_noise=False, add_rf_chain=True)
    sim.get_event(0)
    detector = DetectorResponse(antenna_model=sim.ant_model)
    t = np.arange(sim.sig_size)
    pulse = 100 * np.exp(-0.5 * ((t - 3000) / 8.0) ** 2) * np.sin(0.3 * t)

    for du in (0, sim.nb_du - 1):
        _, line_of_sight = detector.transfer_matrix(sim.du_pos[du], sim.evt_shower, sim.freqs_mhz)
        p = np.cross(line_of_sight, [0.3, 0.5, 0.8])
        p /= np.linalg.norm(p)
        sim.traces[du] = np.outer(p, pulse)
        sim.compute_voltage_du(du)

        response = detector.response(sim.du_pos[du], sim.evt_shower, sim.freqs_mhz, 1.0,
                                     polarization=p)
        wiener = WienerDeconvolution(np.inf)
        result = wiener.deconvolve_spectrum(sim.vout_f[du].T, response)
        expected = sf.irfft(sf.rfft(np.outer(p, pulse), n=sim.fft_size, axis=1)
                            * result.valid, n=sim.fft_size, axis=1)
        efield, _ = wiener.deconvolve(sim.vout[du], response, n_fft=sim.fft_size)
        assert np.abs(efield - expected).max() < 1e-4 * np.abs(expected).max()


def test_the_full_field_cannot_be_inverted_without_a_prior(sim, detector):
    response = detector.response(sim.du_pos[0], sim.evt_shower, sim.freqs_mhz, 1.0)
    with pytest.warns(GRANDlibWarning, match="no frequency can be estimated"):
        result = WienerDeconvolution(np.inf, mode="full3d").deconvolve_spectrum(
            sim.vout_f[0].T, response)
    assert not result.valid.any()

    # A finite prior defines it: the same field, and the prior along the line of sight
    signal_psd = 1e8
    wide = WienerDeconvolution(signal_psd, mode="full3d").deconvolve_spectrum(sim.vout_f[0].T,
                                                                              response)
    band = (sim.freqs_mhz > 50) & (sim.freqs_mhz < 200)
    assert wide.valid[band].all()
    expected = visible_part(field_spectrum(sim, 0), response, band)
    assert np.abs(wide.efield - expected)[band].max() < 1e-3 * np.abs(expected).max()
    np.testing.assert_allclose(wide.variance[band].sum(axis=1), signal_psd, rtol=1e-3)


def test_a_source_below_the_horizon_gives_no_response(sim, detector):
    frame = sim.evt_shower.frame
    below = SimpleNamespace(grand_ref_frame=sim.evt_shower.grand_ref_frame, frame=frame,
                            maximum=LTP(x=0.0, y=0.0, z=-5000.0, frame=frame))
    response = detector.response(sim.du_pos[0], below, sim.freqs_mhz, 1.0)
    assert isinstance(response, Response)
    assert np.all(response.transfer == 0)


def test_the_simulations_interpolation_is_left_alone(sim, detector):
    held = AntennaProcessing.pre_cpt
    freqs_mhz = sf.rfftfreq(1024, 1 / 500.0)
    detector.transfer_matrix(sim.du_pos[0], sim.evt_shower, freqs_mhz)
    assert AntennaProcessing.pre_cpt is held


def test_the_frequency_axis_is_prepared_once(sim):
    detector = DetectorResponse(antenna_model=sim.ant_model)
    freqs_mhz = sf.rfftfreq(1024, 1 / 500.0)
    detector.transfer_matrix(sim.du_pos[0], sim.evt_shower, freqs_mhz)
    rf_tf = detector._rf_tf
    detector.transfer_matrix(sim.du_pos[1], sim.evt_shower, freqs_mhz.copy())
    assert detector._rf_tf is rf_tf
    detector.transfer_matrix(sim.du_pos[1], sim.evt_shower, sf.rfftfreq(2048, 1 / 500.0))
    assert detector._rf_tf.shape[1] == 1025


def test_refuses_bad_input(sim, detector):
    with pytest.raises(ValueError, match="rf_chain"):
        DetectorResponse(rf_chain="unknown", antenna_model=sim.ant_model)
    with pytest.raises(ValueError, match="du_position"):
        detector.transfer_matrix([0.0, 0.0], sim.evt_shower, sim.freqs_mhz)
    with pytest.raises(ValueError, match="du_position"):
        detector.transfer_matrix([0.0, np.nan, 0.0], sim.evt_shower, sim.freqs_mhz)
    with pytest.raises(ValueError, match="starting at 0 MHz"):
        detector.transfer_matrix(sim.du_pos[0], sim.evt_shower, sim.freqs_mhz[1:])
