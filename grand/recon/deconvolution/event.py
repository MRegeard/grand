r"""Deconvolution of a whole AOI event, with any method.

The only part of this package that knows :mod:`grand.aoi`: it takes the voltages,
antennas and shower of an :class:`~grand.aoi.event.Event`, builds each unit's
:class:`~grand.recon.deconvolution.base.Response` and returns the fields.  It
writes nothing, and leaves the event as it is.
"""

from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import numpy as np

from grand.aoi.timetrace import Efield
from grand.basis import validate as _validate
from grand.basis.signal import get_fastest_size_fft
from grand.geo.coordinates import GRANDCS, Geodetic
from grand.sim.efield2voltage import _MIN_XMAX_DISTANCE_M
from grand.sim.shower.gen_shower import ShowerEvent
from .base import Deconvolution
from .polarization import geomagnetic_polarization, polarization_angle
from .response import DetectorResponse


@dataclass
class EventDeconvolution:
    r"""The fields recovered from one event, unit by unit, in the order of its voltages.

    Parameters
    ----------
    efields : list of grand.aoi.timetrace.Efield
        The fields, in µV/m, in the components of the shower frame: north, west
        and up, from magnetic north.  They hold what the antenna can see, that is
        their part in `band_mhz`, and transverse to the line of sight.  Their
        ``eta`` is not filled: it is a different angle, see `polarization_angle`.
    du_id : ndarray, shape (n_du,)
        Identifier of each unit.
    sigma : ndarray, shape (n_du, 3)
        Standard deviation of each component, in µV/m, the same at every sample.
    polarization : ndarray, shape (n_du, 3)
        Expected direction of the field, see
        :func:`~grand.recon.deconvolution.polarization.geomagnetic_polarization`;
        NaN when the geomagnetic field is not known.
    polarization_angle : ndarray, shape (n_du,)
        Its angle in the tangential frame, in degrees; NaN likewise.  This is the
        expected angle, not one measured on the field.
    band_mhz : ndarray, shape (n_du, 2)
        Lowest and highest frequency estimated, in MHz; NaN if there are none.
    """

    efields: list
    du_id: np.ndarray
    sigma: np.ndarray
    polarization: np.ndarray
    polarization_angle: np.ndarray
    band_mhz: np.ndarray


class EventDeconvolver:
    r"""Applies a deconvolution method to every unit of an event.

    Parameters
    ----------
    method : Deconvolution
        The method, with its settings.
    noise : NoiseModel, float or ndarray
        The noise in the voltage, see
        :meth:`~grand.recon.deconvolution.response.DetectorResponse.response`.
    du_type : str, optional
        Antenna model.
    rf_chain : str, RF chain or None, optional
        The chain the voltages went through, see
        :class:`~grand.recon.deconvolution.response.DetectorResponse`.
    shower : {"shower", "simshower"}, optional
        Which shower of the event gives the geometry: the reconstructed one or the
        simulated one.
    inclination : float, optional
        Inclination of the geomagnetic field, in degrees, for the expected
        polarization; from the shower by default.
    padding_factor : float, optional
        Zero-padding of the traces before the transform, at least 1.
    antenna_model : AntennaModel, optional
        Already loaded antenna tables, which replace `du_type`.
    t_bin_ns : float, optional
        Sampling interval of the voltages, in ns; by default, what the event says,
        which comes from the run tree.  Giving it allows voltages and a run tree of
        different analysis levels, which is refused otherwise.

    Examples
    --------
    Recover the field of a simulated event, assuming it is polarized along
    :math:`\vec v \times \vec B`, with no prior on its level::

        from grand.aoi import EventList
        from grand.dataio import DataDirectory
        from grand.recon.deconvolution import (EventDeconvolver, GalacticNoise,
                                               WienerDeconvolution)

        deconvolver = EventDeconvolver(WienerDeconvolution(float("inf"), mode="polarized"),
                                       GalacticNoise(lst=18.0), shower="simshower")
        for event in EventList(DataDirectory(folder, analysis_level=0)):
            result = deconvolver.deconvolve_event(event)
            print(event.event_number, result.du_id, result.sigma)

    The voltages must be in µV.  ``TRawVoltage`` traces, in ADC counts, are not.
    They must also come from the models given here: the file committed in
    ``sim2root/Common``, from 2024, differs from today's chain by a factor of 0.45 to
    3.7 depending on the band and the arm, and does not deconvolve back to its field.
    Without ``analysis_level``, a folder holding several levels gives the run tree of
    the highest one, whose sampling may not be that of the voltages.
    """

    def __init__(self, method, noise, du_type="GP300", rf_chain="rf_chain", shower="shower",
                 inclination=None, padding_factor=1.0, antenna_model=None, t_bin_ns=None):
        where = "EventDeconvolver"
        if not isinstance(method, Deconvolution):
            raise TypeError(_validate.message(
                where, "'method' must be a Deconvolution, got %s" % type(method).__name__))
        _validate.one_of(shower, ("shower", "simshower"), "shower", where)
        if not _validate.as_real(padding_factor, "padding_factor", where) >= 1:
            raise ValueError(_validate.message(
                where, "'padding_factor' must be at least 1, got %s" % padding_factor))
        self.method = method
        self.noise = noise
        self.detector = DetectorResponse(du_type, rf_chain, antenna_model)
        self.shower = shower
        self.inclination = inclination
        self.padding_factor = padding_factor
        if t_bin_ns is not None:
            _validate.positive(_validate.as_real(t_bin_ns, "t_bin_ns", where), "t_bin_ns", where, "ns")
        self.t_bin_ns = t_bin_ns

    def geometry(self, event):
        r"""The geometry of an event, as the simulation builds it.

        Parameters
        ----------
        event : Event
            The event.

        Returns
        -------
        ShowerEvent

        Raises
        ------
        ValueError
            If the event lacks the chosen shower, or its shower maximum.
        """
        where = "EventDeconvolver.geometry"
        shower = getattr(event, self.shower)
        if shower is None:
            other = "simshower" if self.shower == "shower" else "shower"
            raise ValueError(_validate.message(
                where, "the event has no '%s'%s" % (self.shower, (
                    "; it has a '%s': choose it with EventDeconvolver(shower='%s')" % (other, other)
                    if getattr(event, other) is not None else ""))))
        maximum = np.ravel(np.asarray(shower.Xmaxpos, dtype=float))
        if not np.all(np.isfinite(maximum)) or np.linalg.norm(maximum) < _MIN_XMAX_DISTANCE_M:
            raise ValueError(_validate.message(
                where, "the '%s' of the event has no usable Xmax position (%s); the antenna "
                "response needs its direction" % (self.shower, maximum.tolist())))

        geometry = ShowerEvent()
        geometry.origin_geoid = np.ravel(np.asarray(shower.origin_geoid, dtype=float)).tolist()
        geometry.load_root(SimpleNamespace(
            energy_primary=shower.energy_primary, zenith=shower.zenith, azimuth=shower.azimuth,
            primary_type=shower.primary_type, xmax_pos_shc=maximum,
            shower_core_pos=np.ravel(np.asarray(shower.core_ground_pos, dtype=float))))
        # Positions computed from GPS have an origin of their own (Event.antennas_origin)
        if event.antennas_origin is not None and not np.allclose(event.antennas_origin,
                                                                 geometry.origin_geoid):
            latitude, longitude, height = event.antennas_origin
            origin = Geodetic(latitude=latitude, longitude=longitude, height=height)
            geometry.grand_ref_frame = GRANDCS(arg=origin, location=origin)
        return geometry

    def _inclination(self, event):
        r"""The inclination to use, or None when neither the user nor the shower gives it.

        Parameters
        ----------
        event : Event
            The event.

        Returns
        -------
        float or None
        """
        if self.inclination is not None:
            return self.inclination
        field = np.ravel(np.asarray(getattr(event, self.shower).magnetic_field, dtype=float))
        return float(field[0]) if field.size == 3 and np.any(field != 0) else None

    def _check_sampling(self, event):
        r"""Refuses voltages whose sampling comes from a run tree of another level.

        ``Voltage.t_bin_size`` is read from the run tree.  With a wrong one the
        frequency axis is wrong, and so is the field, by orders of magnitude and
        with no other sign.

        Parameters
        ----------
        event : Event
            The event.
        """
        run, voltage = getattr(event, "trun", None), getattr(event, "tvoltage", None)
        if self.t_bin_ns is not None or run is None or voltage is None:
            return
        if run.analysis_level != voltage.analysis_level:
            raise ValueError(_validate.message(
                "EventDeconvolver.deconvolve_event", "the voltages are of analysis level %d and "
                "the run tree, which gives their sampling (%s ns), of level %d. Read one level, "
                "with EventList(DataDirectory(folder, analysis_level=%d)), or give the sampling "
                "with EventDeconvolver(t_bin_ns=...)" % (
                    voltage.analysis_level, event.voltages[0].t_bin_size if event.voltages else "?",
                    run.analysis_level, voltage.analysis_level)))

    def _position(self, event, du_id):
        r"""The position of a unit, in the array frame.

        Parameters
        ----------
        event : Event
            The event.
        du_id : int
            Identifier of the unit.

        Returns
        -------
        ndarray, shape (3,)
        """
        for antennas in (event.antennas or [], (event._all_antennas or {}).values()):
            for antenna in antennas:
                if antenna.id == du_id:
                    return np.ravel(np.asarray(antenna.position, dtype=float))
        raise ValueError(_validate.message(
            "EventDeconvolver", "the event has no antenna for the voltage of unit %s, so its "
            "direction is unknown; the positions come from the run tree or from GPS" % du_id))

    def deconvolve_event(self, event):
        r"""Recovers the field of every unit of an event that has a voltage.

        Parameters
        ----------
        event : Event
            The event.  It is not changed, and the result stays valid after an
            :class:`~grand.aoi.event_list.EventList` has moved on to the next one.

        Returns
        -------
        EventDeconvolution
        """
        if event.voltages is None:
            raise ValueError(_validate.message(
                "EventDeconvolver.deconvolve_event", "the event has no voltages"))
        self._check_sampling(event)
        geometry = self.geometry(event)
        inclination = self._inclination(event)
        polarization = None
        if inclination is not None:
            polarization = partial(geomagnetic_polarization, zenith=geometry.zenith,
                                   azimuth=geometry.azimuth, inclination=inclination)

        n_du = len(event.voltages)
        result = EventDeconvolution(
            efields=[], du_id=np.empty(n_du, dtype=int), sigma=np.empty((n_du, 3)),
            polarization=np.full((n_du, 3), np.nan), polarization_angle=np.full(n_du, np.nan),
            band_mhz=np.full((n_du, 2), np.nan))
        for i, voltage in enumerate(event.voltages):
            trace = np.asarray(voltage.trace, dtype=float)
            t_bin_ns = voltage.t_bin_size if self.t_bin_ns is None else self.t_bin_ns
            n_fft, freqs_mhz = get_fastest_size_fft(trace.shape[1], 1e3 / t_bin_ns,
                                                    self.padding_factor)
            response = self.detector.response(self._position(event, voltage.du_id), geometry,
                                              freqs_mhz, self.noise, polarization=polarization,
                                              n_fft=n_fft)
            efield, sigma = self.method.deconvolve(trace, response, n_fft=n_fft)

            result.efields.append(_as_efield(efield, voltage, t_bin_ns))
            result.du_id[i] = voltage.du_id
            result.sigma[i] = sigma
            if polarization is not None:
                result.polarization[i] = response.polarization
                result.polarization_angle[i] = polarization_angle(response.polarization,
                                                                  response.line_of_sight)
            if response.band.any():
                result.band_mhz[i] = freqs_mhz[response.band][[0, -1]]
        return result


def _as_efield(trace, voltage, t_bin_ns):
    r"""Returns a field on the time axis of a voltage.

    Parameters
    ----------
    trace : ndarray, shape (3, n_samples)
        The field.
    voltage : grand.aoi.timetrace.Voltage
        The voltage it was recovered from.
    t_bin_ns : float
        The sampling interval used, in ns.

    Returns
    -------
    grand.aoi.timetrace.Efield
    """
    efield = Efield()
    efield.n_points = trace.shape[1]
    efield.time_step = voltage.time_step
    efield.t0 = voltage.t0
    efield.trigger_time = voltage.trigger_time
    efield.t_bin_size = t_bin_ns
    efield.du_id = voltage.du_id
    efield.t_vector = np.array(voltage.t_vector)
    efield.trace = trace
    return efield
