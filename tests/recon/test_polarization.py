# -*- coding: utf-8 -*-
r"""The expected polarization against its closed forms and a simulated shower.

The geometry is checked by hand on simple directions, then against the field of
the ZHAireS shower committed in ``sim2root/Common``, which knows nothing of this
code: any wrong sign or frame in the azimuth or the inclination shows there.
"""

import pathlib
from functools import partial

import numpy as np
import pytest
import scipy.fft as sf

import grand
from grand import Efield2Voltage
from grand.recon.deconvolution import (DetectorResponse, GalacticNoise, WienerDeconvolution,
                                       geomagnetic_polarization, polarization_angle)
from grand.recon.deconvolution.base import time_variance
from grand.recon.deconvolution.polarization import geomagnetic_direction, shower_axis

SAMPLE = (pathlib.Path(grand.__file__).parents[1]
          / "sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")
LST = 18.0
#: Inclination at the GP300 site used by DC2
INCLINATION = 61.6


@pytest.mark.parametrize("zenith, azimuth, expected", [
    (0.0, 0.0, (0, 0, -1)),      # from straight above, going down
    (90.0, 0.0, (-1, 0, 0)),     # from the north, going south
    (90.0, 90.0, (0, -1, 0)),    # from the west, going east
    (90.0, 180.0, (1, 0, 0)),    # from the south, going north
])
def test_the_shower_travels_away_from_where_it_comes_from(zenith, azimuth, expected):
    np.testing.assert_allclose(shower_axis(zenith, azimuth), expected, atol=1e-12)


@pytest.mark.parametrize("inclination, expected", [
    (0.0, (1, 0, 0)),            # along magnetic north
    (90.0, (0, 0, -1)),          # straight down
    (-90.0, (0, 0, 1)),          # straight up, in the south
])
def test_the_geomagnetic_field_points_north_and_down(inclination, expected):
    np.testing.assert_allclose(geomagnetic_direction(inclination), expected, atol=1e-12)


def test_a_shower_from_the_north_is_polarized_by_hand():
    # v to the south, B to the north and down: v x B is (-x) x (x cos I - z sin I),
    # which is -y sin I, then normalized; so along -y, to the east
    v = shower_axis(90.0, 0.0)
    np.testing.assert_allclose(geomagnetic_polarization(v, 90.0, 0.0, INCLINATION), [0, -1, 0],
                               atol=1e-12)


def test_the_polarization_is_a_unit_vector_transverse_to_the_line_of_sight():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = rng.normal(size=3)
        p = geomagnetic_polarization(n, rng.uniform(0, 88), rng.uniform(0, 360), INCLINATION)
        assert np.linalg.norm(p) == pytest.approx(1)
        assert p @ n == pytest.approx(0, abs=1e-12)


def test_on_the_axis_the_polarization_is_along_v_cross_b():
    v = shower_axis(70.0, 30.0)
    b = geomagnetic_direction(INCLINATION)
    p = geomagnetic_polarization(v, 70.0, 30.0, INCLINATION)
    assert p @ b == pytest.approx(0, abs=1e-12)
    assert p @ v == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(p, np.cross(v, b) / np.linalg.norm(np.cross(v, b)))


@pytest.mark.parametrize("zenith", [40.0, 60.0, 80.0])
@pytest.mark.parametrize("azimuth", [10.0, 100.0, 170.0, 200.0, 310.0])
def test_the_angle_on_the_axis_has_a_closed_form(zenith, azimuth):
    v = shower_axis(zenith, azimuth)
    b = geomagnetic_direction(INCLINATION)
    cos_alpha = (np.cos(np.deg2rad(INCLINATION)) * np.sin(np.deg2rad(azimuth))
                 / np.linalg.norm(np.cross(v, b)))
    p = geomagnetic_polarization(v, zenith, azimuth, INCLINATION)
    assert polarization_angle(p, v) == pytest.approx(np.degrees(np.arccos(cos_alpha)))


def largest_difference_from_the_formula_without_norm(zenith):
    azimuths = np.arange(5.0, 360.0, 10.0)
    formula = np.degrees(np.arccos(np.cos(np.deg2rad(INCLINATION)) * np.sin(np.deg2rad(azimuths))))
    vector = [polarization_angle(geomagnetic_polarization(shower_axis(zenith, phi), zenith, phi,
                                                          INCLINATION), shower_axis(zenith, phi))
              for phi in azimuths]
    return np.max(np.abs((np.array(vector) - formula + 90.0) % 180.0 - 90.0))


def test_the_formula_without_norm_holds_for_inclined_showers_only():
    assert largest_difference_from_the_formula_without_norm(85.0) < 3.0
    assert largest_difference_from_the_formula_without_norm(40.0) > 20.0


def test_the_angle_is_that_of_a_direction():
    n = shower_axis(70.0, 30.0)
    p = geomagnetic_polarization(n, 70.0, 30.0, INCLINATION)
    assert polarization_angle(-p, n) == pytest.approx(polarization_angle(p, n))
    assert 0 <= polarization_angle(p, n) < 180


def test_the_angle_is_measured_from_theta_toward_phi():
    # From straight north at the horizon: u_theta is down, u_phi is to the west
    n = -np.array([1.0, 0.0, 0.0])
    assert polarization_angle([0, 0, -1], n) == pytest.approx(0)
    assert polarization_angle([0, 1, 0], n) == pytest.approx(90)


def test_refuses_what_has_no_direction():
    # Along the field, which goes north and down: from the south, at 61.6 deg elevation
    v = shower_axis(28.4, 180.0)
    assert v @ geomagnetic_direction(INCLINATION) == pytest.approx(1)
    with pytest.raises(ValueError, match="along the geomagnetic field"):
        geomagnetic_polarization(v + [0, 0.1, 0], 28.4, 180.0, INCLINATION)

    v = shower_axis(70.0, 30.0)
    v_cross_b = np.cross(v, geomagnetic_direction(INCLINATION))
    with pytest.raises(ValueError, match="line of sight is along"):
        geomagnetic_polarization(v_cross_b, 70.0, 30.0, INCLINATION)

    with pytest.raises(ValueError, match="line_of_sight"):
        geomagnetic_polarization([0.0, 0.0, 0.0], 70.0, 30.0, INCLINATION)
    with pytest.raises(ValueError, match="line_of_sight"):
        polarization_angle([1.0, 0.0, 0.0], [0.0, 0.0])


def simulate(directory, noise):
    sim = Efield2Voltage(str(SAMPLE), "voltage.root", output_directory=str(directory), seed=1,
                         efield_level=0)
    sim.params.update(add_noise=noise, add_rf_chain=True, lst=LST)
    sim.compute_voltage_event(0)
    sim.final_resample()
    return sim


@pytest.fixture(scope="module")
def shower(tmp_path_factory):
    directory = tmp_path_factory.mktemp("polarization")
    clean, noisy = simulate(directory, False), simulate(directory, True)
    detector = DetectorResponse(antenna_model=clean.ant_model)
    polarization = partial(geomagnetic_polarization, zenith=clean.shower.zenith,
                           azimuth=clean.shower.azimuth,
                           inclination=float(clean.shower.magnetic_field[0]))
    model = GalacticNoise(LST)
    responses = [detector.response(clean.du_pos[du], clean.evt_shower, clean.freqs_mhz, model,
                                   polarization=polarization)
                 for du in range(clean.nb_du)]
    fields = []
    for du, response in enumerate(responses):
        n = response.line_of_sight
        field_f = sf.rfft(clean.traces[du].astype(float), n=clean.fft_size, axis=1).T
        field_f = (field_f @ (np.eye(3) - np.outer(n, n))) * response.band[:, np.newaxis]
        fields.append(sf.irfft(field_f.T, n=clean.fft_size, axis=1))
    return clean, noisy, responses, np.array(fields)


def strong_units(fields):
    r"""The units above 100 µV/m in the band, which the model is claimed to describe."""
    return np.flatnonzero(np.abs(fields).max(axis=(1, 2)) > 100)


def test_the_model_is_the_polarization_of_the_simulated_field(shower):
    _, _, responses, fields = shower
    strong = strong_units(fields)
    assert strong.size >= 10
    for du in strong:
        # The field's main direction, whatever the model says
        measured = np.linalg.svd(fields[du], full_matrices=False)[0][:, 0]
        angle = np.degrees(np.arccos(min(abs(measured @ responses[du].polarization), 1.0)))
        assert angle < 3.0


def test_the_angle_agrees_with_the_simulated_field(shower):
    _, _, responses, fields = shower
    for du in strong_units(fields):
        n = responses[du].line_of_sight
        measured = np.linalg.svd(fields[du], full_matrices=False)[0][:, 0]
        difference = polarization_angle(measured, n) - polarization_angle(responses[du].polarization, n)
        assert abs((difference + 90.0) % 180.0 - 90.0) < 3.0


def test_a_callable_polarization_gets_the_line_of_sight(shower):
    clean, _, responses, _ = shower
    detector = DetectorResponse(antenna_model=clean.ant_model)
    seen = []

    def polarization(line_of_sight):
        seen.append(line_of_sight)
        return [0.0, 0.0, 2.0]

    response = detector.response(clean.du_pos[0], clean.evt_shower, clean.freqs_mhz, 1.0,
                                 polarization=polarization)
    np.testing.assert_allclose(seen[0], responses[0].line_of_sight)
    np.testing.assert_allclose(response.polarization, [0, 0, 1])


def reconstruct(wiener, noisy, du, response):
    result = wiener.deconvolve_spectrum(noisy.vout_f[du].T, response)
    return (sf.irfft(result.efield.T, n=noisy.fft_size, axis=1),
            np.sqrt(time_variance(result.variance, noisy.fft_size)))


@pytest.mark.parametrize("signal_psd", [np.inf, "auto"])
def test_the_constraint_lowers_the_error_on_strong_units(shower, signal_psd):
    _, noisy, responses, fields = shower
    free = WienerDeconvolution(signal_psd, mode="transverse")
    constrained = WienerDeconvolution(signal_psd, mode="polarized")
    for du in strong_units(fields):
        with_free, _ = reconstruct(free, noisy, du, responses[du])
        with_constraint, _ = reconstruct(constrained, noisy, du, responses[du])
        assert (np.linalg.norm(with_constraint - fields[du])
                < np.linalg.norm(with_free - fields[du]))


def test_the_constraint_brings_the_peak_closer_with_a_prior(shower):
    _, noisy, responses, fields = shower
    free = WienerDeconvolution("auto", mode="transverse")
    constrained = WienerDeconvolution("auto", mode="polarized")
    du = int(np.argmax(np.abs(fields).max(axis=(1, 2))))
    truth = np.linalg.norm(fields[du], axis=0).max()
    peak_free = np.linalg.norm(reconstruct(free, noisy, du, responses[du])[0], axis=0).max()
    peak = np.linalg.norm(reconstruct(constrained, noisy, du, responses[du])[0], axis=0).max()
    assert abs(peak - truth) < abs(peak_free - truth)


def test_error_bars_along_the_polarization_are_the_error_made(shower):
    _, noisy, responses, fields = shower
    wiener = WienerDeconvolution(np.inf, mode="polarized")
    ratios = []
    for du in strong_units(fields):
        p = responses[du].polarization
        efield, sigma = reconstruct(wiener, noisy, du, responses[du])
        ratios.append(np.linalg.norm(sigma) / (p @ (efield - fields[du])).std())
    np.testing.assert_allclose(np.median(ratios), 1, atol=0.05)
