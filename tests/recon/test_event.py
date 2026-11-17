# -*- coding: utf-8 -*-
r"""Deconvolving AOI events read from files, against the simulation that wrote them.

The ZHAireS shower committed in ``sim2root/Common`` goes through ``Efield2Voltage``,
to a voltage file next to its efield, run and shower files, and comes back through
``EventList``.  What was simulated is known, so the whole path can be checked: the
files, the matching of units, the geometry, the frequency axis and the method.
"""

import os
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.fft as sf

import grand
from grand import Efield2Voltage
from grand.aoi import EventList
from grand.aoi.antenna import Antenna
from grand.aoi.timetrace import Efield, Voltage
from grand.dataio import DataDirectory
from grand.geo.coordinates import GRANDCS, LTP, Geodetic
from grand.recon.deconvolution import (Deconvolution, DeconvolvedSpectrum, DetectorResponse,
                                       EventDeconvolution, EventDeconvolver, GalacticNoise,
                                       WienerDeconvolution, geomagnetic_polarization,
                                       polarization_angle)

SAMPLE = (pathlib.Path(grand.__file__).parents[1]
          / "sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")
LST = 18.0


def simulate(directory, noise):
    r"""Writes the voltage of the sample's first event; returns the simulation and the event."""
    for path in SAMPLE.glob("*_L0_*.root"):
        if not path.name.startswith("voltage"):
            os.symlink(path, directory / path.name)
    sim = Efield2Voltage(str(SAMPLE), "voltage_1618-13790_L0_0000.root",
                         output_directory=str(directory), seed=1, efield_level=0)
    sim.params.update(add_noise=noise, add_rf_chain=True, lst=LST)
    sim.compute_voltage(event_idx=0)
    return sim, next(iter(EventList(str(directory))))


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return simulate(tmp_path_factory.mktemp("clean"), False)


@pytest.fixture(scope="module")
def noisy(tmp_path_factory):
    return simulate(tmp_path_factory.mktemp("noisy"), True)


def visible_fields(sim, detector, response_noise=1.0):
    r"""What the antennas can see of the simulated field, for each unit."""
    fields = []
    for du in range(sim.nb_du):
        response = detector.response(sim.du_pos[du], sim.evt_shower, sim.freqs_mhz, response_noise)
        n = response.line_of_sight
        field_f = sf.rfft(sim.traces[du].astype(float), n=sim.fft_size, axis=1).T
        field_f = (field_f @ (np.eye(3) - np.outer(n, n))) * response.band[:, np.newaxis]
        fields.append(sf.irfft(field_f.T, n=sim.fft_size, axis=1))
    return np.array(fields)


def deconvolver_for(sim, method, noise=1.0, **kwargs):
    return EventDeconvolver(method, noise, shower="simshower", antenna_model=sim.ant_model, **kwargs)


def traces_of(result):
    return np.array([np.asarray(field.trace) for field in result.efields])


def test_the_event_comes_back_as_it_was_simulated(clean):
    sim, event = clean
    result = deconvolver_for(sim, WienerDeconvolution(np.inf, mode="transverse")).deconvolve_event(event)
    assert isinstance(result, EventDeconvolution)
    np.testing.assert_array_equal(result.du_id, sim.du_id)
    expected = visible_fields(sim, DetectorResponse(antenna_model=sim.ant_model))
    fields = traces_of(result)
    peak = np.abs(expected).max(axis=(1, 2))
    assert np.all(np.abs(fields - expected).max(axis=(1, 2)) < 1e-3 * peak)


def test_the_geometry_is_that_of_the_simulation(clean):
    sim, event = clean
    deconvolver = deconvolver_for(sim, WienerDeconvolution(np.inf, mode="transverse"))
    geometry = deconvolver.geometry(event)
    np.testing.assert_allclose(np.ravel(geometry.maximum), np.ravel(sim.evt_shower.maximum))
    np.testing.assert_allclose(geometry.frame.basis, sim.evt_shower.frame.basis)
    np.testing.assert_allclose(np.ravel(geometry.frame.location),
                               np.ravel(sim.evt_shower.frame.location))
    detector = DetectorResponse(antenna_model=sim.ant_model)
    transfer, line_of_sight = detector.transfer_matrix(sim.du_pos[0], sim.evt_shower, sim.freqs_mhz)
    transfer_event, line_of_sight_event = detector.transfer_matrix(sim.du_pos[0], geometry,
                                                                   sim.freqs_mhz)
    np.testing.assert_allclose(transfer_event, transfer)
    np.testing.assert_allclose(line_of_sight_event, line_of_sight)


def test_the_units_are_matched_by_identifier_not_by_order(clean):
    sim, event = clean
    shuffled = SimpleNamespace(voltages=event.voltages, antennas=list(reversed(event.antennas)),
                               _all_antennas=None, simshower=event.simshower, shower=None,
                               antennas_origin=event.antennas_origin)
    method = WienerDeconvolution(np.inf, mode="transverse")
    np.testing.assert_allclose(
        traces_of(deconvolver_for(sim, method).deconvolve_event(shuffled)),
        traces_of(deconvolver_for(sim, method).deconvolve_event(event)))


def test_the_sampling_comes_from_the_voltage(clean):
    # The event's efields say 2 ns when they were sampled at 0.5 ns: a voltage read
    # with such a bin would get the wrong frequency axis, and a wrong field
    sim, event = clean
    assert event.voltages[0].t_bin_size == 0.5
    result = deconvolver_for(sim, WienerDeconvolution(np.inf, mode="transverse")).deconvolve_event(event)
    assert result.band_mhz[0, 0] == pytest.approx(30, abs=0.3)
    assert result.band_mhz[0, 1] == pytest.approx(250, abs=0.3)


def test_a_folder_of_several_levels_is_not_read_with_the_wrong_sampling():
    # A run tree at 2 ns from L1, and voltages at 0.5 ns from L0: the field came out
    # about 300 times too large, with no other sign
    event = next(iter(EventList(str(SAMPLE))))
    assert event.trun.analysis_level == 1 and event.tvoltage.analysis_level == 0
    assert event.voltages[0].t_bin_size == 2      # what the AOI says, wrongly
    method = WienerDeconvolution(np.inf, mode="transverse")
    deconvolver = EventDeconvolver(method, 1.0, shower="simshower")
    with pytest.raises(ValueError, match="analysis_level=0"):
        deconvolver.deconvolve_event(event)


def test_one_level_or_a_given_sampling_is_accepted():
    method = WienerDeconvolution(np.inf, mode="transverse")
    deconvolver = EventDeconvolver(method, 1.0, shower="simshower")
    event = next(iter(EventList(DataDirectory(str(SAMPLE), analysis_level=0))))
    assert event.voltages[0].t_bin_size == 0.5
    by_level = deconvolver.deconvolve_event(event)

    event = next(iter(EventList(str(SAMPLE))))
    given = EventDeconvolver(method, 1.0, shower="simshower", t_bin_ns=0.5,
                             antenna_model=deconvolver.detector.antenna_model)
    by_value = given.deconvolve_event(event)
    assert by_value.efields[0].t_bin_size == 0.5
    np.testing.assert_allclose(by_value.band_mhz, by_level.band_mhz)
    np.testing.assert_allclose(np.asarray(by_value.efields[0].trace),
                               np.asarray(by_level.efields[0].trace))
    with pytest.raises(ValueError, match="t_bin_ns"):
        EventDeconvolver(method, 1.0, t_bin_ns=0.0, antenna_model=deconvolver.detector.antenna_model)


def test_the_fields_carry_the_time_axis_of_the_voltages(clean):
    sim, event = clean
    result = deconvolver_for(sim, WienerDeconvolution(np.inf, mode="transverse")).deconvolve_event(event)
    assert len(result.efields) == len(event.voltages)
    for field, voltage in zip(result.efields, event.voltages):
        assert isinstance(field, Efield)
        assert field.du_id == voltage.du_id
        assert field.t0 == voltage.t0 and field.trigger_time == voltage.trigger_time
        assert field.t_bin_size == voltage.t_bin_size and field.n_points == voltage.n_points
        np.testing.assert_array_equal(field.t_vector, voltage.t_vector)
        assert field.t_vector is not voltage.t_vector
        assert np.asarray(field.trace).shape == np.asarray(voltage.trace).shape


def test_the_event_is_left_as_it_was(clean):
    sim, event = clean
    efields = event.efields
    before = np.array(np.asarray(event.voltages[0].trace))
    deconvolver_for(sim, WienerDeconvolution(np.inf, mode="transverse")).deconvolve_event(event)
    assert event.efields is efields
    np.testing.assert_array_equal(np.asarray(event.voltages[0].trace), before)


def test_the_polarized_mode_recovers_the_strong_units(clean):
    sim, event = clean
    detector = DetectorResponse(antenna_model=sim.ant_model)
    expected = visible_fields(sim, detector)
    result = deconvolver_for(sim, WienerDeconvolution(np.inf, mode="polarized")).deconvolve_event(event)
    fields = traces_of(result)
    for du in np.flatnonzero(np.abs(expected).max(axis=(1, 2)) > 100):
        truth, recovered = (np.linalg.norm(x, axis=0).max() for x in (expected[du], fields[du]))
        # Only the part along p comes back, and p is within 2 deg of the field
        assert recovered == pytest.approx(truth, rel=0.05)


def test_the_expected_polarization_is_reported(clean):
    sim, event = clean
    deconvolver = deconvolver_for(sim, WienerDeconvolution(np.inf, mode="transverse"))
    result = deconvolver.deconvolve_event(event)
    detector = DetectorResponse(antenna_model=sim.ant_model)
    shower = event.simshower
    for du in (0, sim.nb_du - 1):
        _, n = detector.transfer_matrix(sim.du_pos[du], sim.evt_shower, sim.freqs_mhz)
        p = geomagnetic_polarization(n, shower.zenith, shower.azimuth, shower.magnetic_field[0])
        np.testing.assert_allclose(result.polarization[du], p, atol=1e-6)
        assert result.polarization_angle[du] == pytest.approx(polarization_angle(p, n), abs=1e-3)
    assert result.efields[0].eta == 0     # another angle, not filled


def without_magnetic_field(event):
    shower = SimpleNamespace(**{name: getattr(event.simshower, name) for name in (
        "energy_primary", "zenith", "azimuth", "primary_type", "Xmaxpos", "origin_geoid",
        "core_ground_pos")}, magnetic_field=np.zeros(3))
    return SimpleNamespace(voltages=event.voltages[:2], antennas=event.antennas, _all_antennas=None,
                           simshower=shower, shower=None, antennas_origin=event.antennas_origin)


def test_without_a_magnetic_field_there_is_no_expected_polarization(clean):
    sim, event = clean
    blank = without_magnetic_field(event)
    result = deconvolver_for(sim, WienerDeconvolution(np.inf, mode="transverse")).deconvolve_event(blank)
    assert np.all(np.isnan(result.polarization)) and np.all(np.isnan(result.polarization_angle))
    with pytest.raises(ValueError, match="needs the response's polarization"):
        deconvolver_for(sim, WienerDeconvolution(np.inf)).deconvolve_event(blank)


def test_an_inclination_given_replaces_the_one_of_the_shower(clean):
    sim, event = clean
    method = WienerDeconvolution(np.inf)
    given = deconvolver_for(sim, method, inclination=float(event.simshower.magnetic_field[0]))
    blank = given.deconvolve_event(without_magnetic_field(event))
    reference = deconvolver_for(sim, method).deconvolve_event(event)
    np.testing.assert_allclose(blank.polarization[:2], reference.polarization[:2])
    other = deconvolver_for(sim, method, inclination=-30.0).deconvolve_event(event)
    assert np.abs(other.polarization[0] - reference.polarization[0]).max() > 0.1


def test_positions_with_an_origin_of_their_own(clean):
    # As Event.fill_antennas gives for GP300, from GPS: the same place in another frame
    sim, event = clean
    deconvolver = deconvolver_for(sim, WienerDeconvolution(np.inf, mode="transverse"))
    geometry = deconvolver.geometry(event)
    origin = np.ravel(event.origin_geoid) + [0.03, 0.03, 0.0]    # about 4 km away
    other = Geodetic(latitude=origin[0], longitude=origin[1], height=origin[2])
    antennas = []
    for antenna in event.antennas[:2]:
        x, y, z = np.ravel(antenna.position)
        place = Geodetic(LTP(x=x, y=y, z=z, frame=geometry.grand_ref_frame))
        moved = Antenna(id=antenna.id)
        moved.position = np.ravel(GRANDCS(place, location=other))
        antennas.append(moved)
    assert np.linalg.norm(np.ravel(antennas[0].position) - np.ravel(event.antennas[0].position)) > 1000
    moved_event = SimpleNamespace(voltages=event.voltages[:2], antennas=antennas, _all_antennas=None,
                                  simshower=event.simshower, shower=None,
                                  antennas_origin=tuple(origin))
    first = deconvolver.deconvolve_event(moved_event)
    assert first.du_id.tolist() == [event.voltages[0].du_id, event.voltages[1].du_id]
    reference = SimpleNamespace(voltages=event.voltages[:2], antennas=event.antennas,
                                _all_antennas=None, simshower=event.simshower, shower=None,
                                antennas_origin=event.antennas_origin)
    np.testing.assert_allclose(first.polarization, deconvolver.deconvolve_event(reference).polarization,
                               atol=1e-3)


def test_padding_keeps_the_time_axis(clean):
    sim, event = clean
    method = WienerDeconvolution(np.inf, mode="transverse")
    padded = deconvolver_for(sim, method, padding_factor=1.5).deconvolve_event(event)
    assert np.asarray(padded.efields[0].trace).shape == np.asarray(event.voltages[0].trace).shape
    strongest = int(np.argmax([np.abs(np.asarray(v.trace)).max() for v in event.voltages]))
    plain = deconvolver_for(sim, method).deconvolve_event(event)
    a, b = (np.asarray(result.efields[strongest].trace) for result in (padded, plain))
    assert np.abs(a - b).max() < 0.05 * np.abs(b).max()


def test_error_bars_on_a_noisy_event_are_the_error_made(noisy):
    sim, event = noisy
    method = WienerDeconvolution(np.inf, mode="transverse")
    result = deconvolver_for(sim, method, noise=GalacticNoise(LST)).deconvolve_event(event)
    detector = DetectorResponse(antenna_model=sim.ant_model)
    errors = (traces_of(result) - visible_fields(sim, detector)).std(axis=2)
    np.testing.assert_allclose(np.median(result.sigma / errors, axis=0), 1, atol=0.05)


def test_any_method_can_be_used(clean):
    class Nothing(Deconvolution):
        def deconvolve_spectrum(self, voltage_f, response):
            return DeconvolvedSpectrum(np.zeros_like(voltage_f), np.ones(voltage_f.shape, dtype=float),
                                       np.ones(voltage_f.shape[0], dtype=bool))

    sim, event = clean
    result = deconvolver_for(sim, Nothing()).deconvolve_event(event)
    assert np.all(traces_of(result) == 0)
    assert np.all(result.sigma > 0)


def test_refuses_what_it_cannot_use(clean):
    sim, event = clean
    method = WienerDeconvolution(np.inf, mode="transverse")
    with pytest.raises(TypeError, match="Deconvolution"):
        EventDeconvolver("wiener", 1.0, antenna_model=sim.ant_model)
    with pytest.raises(ValueError, match="shower"):
        EventDeconvolver(method, 1.0, shower="truth", antenna_model=sim.ant_model)
    with pytest.raises(ValueError, match="padding_factor"):
        deconvolver_for(sim, method, padding_factor=0.5)
    # The event of an L0 folder has only the simulated shower, and says so
    with pytest.raises(ValueError, match="choose it with EventDeconvolver.shower='simshower'"):
        EventDeconvolver(method, 1.0, antenna_model=sim.ant_model).deconvolve_event(event)

    deconvolver = deconvolver_for(sim, method)
    base = dict(antennas=event.antennas, _all_antennas=None, simshower=event.simshower, shower=None,
                antennas_origin=event.antennas_origin)
    with pytest.raises(ValueError, match="no voltages"):
        deconvolver.deconvolve_event(SimpleNamespace(voltages=None, **base))
    unknown = Voltage(du_id=999999)
    unknown.trace = np.zeros((3, 64))
    unknown.t_bin_size = 0.5
    with pytest.raises(ValueError, match="no antenna for the voltage of unit 999999"):
        deconvolver.deconvolve_event(SimpleNamespace(voltages=[unknown], **base))
    no_xmax = SimpleNamespace(Xmaxpos=np.zeros(3))
    with pytest.raises(ValueError, match="no usable Xmax"):
        deconvolver.geometry(SimpleNamespace(simshower=no_xmax, shower=None))


def test_an_event_without_units_gives_an_empty_result(clean):
    sim, event = clean
    empty = SimpleNamespace(voltages=[], antennas=[], _all_antennas=None, simshower=event.simshower,
                            shower=None, antennas_origin=event.antennas_origin)
    result = deconvolver_for(sim, WienerDeconvolution(np.inf, mode="transverse")).deconvolve_event(empty)
    assert result.efields == [] and result.sigma.shape == (0, 3) and result.du_id.size == 0


def test_a_result_outlives_the_event_it_came_from(tmp_path):
    # EventList fills the same Event again for the next one
    for path in SAMPLE.glob("*_L0_*.root"):
        if not path.name.startswith("voltage"):
            os.symlink(path, tmp_path / path.name)
    sim = Efield2Voltage(str(SAMPLE), "voltage_1618-13790_L0_0000.root",
                         output_directory=str(tmp_path), seed=1, efield_level=0)
    sim.params.update(add_noise=False, add_rf_chain=True)
    sim.compute_voltage()
    deconvolver = deconvolver_for(sim, WienerDeconvolution(np.inf, mode="transverse"))

    results, kept, numbers = [], [], []
    for event in EventList(str(tmp_path)):
        results.append(deconvolver.deconvolve_event(event))
        kept.append(traces_of(results[-1]).copy())
        numbers.append(event.event_number)
    assert len(numbers) == 2 and numbers[0] != numbers[1]
    for result, traces in zip(results, kept):
        np.testing.assert_array_equal(traces_of(result), traces)
    assert results[0].du_id.tolist() != results[1].du_id.tolist()
