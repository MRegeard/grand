r"""The transfer matrix of a detection unit, built from the simulation's own operators.

:math:`T_{kj}(\nu) = RF_k(\nu)\,L_{kj}(\theta, \phi, \nu)`, with the effective
length :math:`L` and the RF chain :math:`RF` evaluated by the code
:class:`~grand.sim.efield2voltage.Efield2Voltage` uses, so that a deconvolution
undoes what a simulation does and nothing else.

The effective length only has components along :math:`\vec e_\theta` and
:math:`\vec e_\phi`, so **the matrix has rank 2**: :math:`\mathbf T\vec n = 0`,
with :math:`\vec n` the line of sight.  The component of the field along it
leaves no trace in the voltage and no method can recover it.
"""

import numpy as np

from grand.basis import validate as _validate
from grand.sim.detector.antenna_model import AntennaModel
from grand.sim.detector.process_ant import AntennaProcessing, PreComputeInterpol, antenna_frame
from grand.sim.detector.rf_chain import RFChain, RFChainNut, RFChain_gaa
from .base import Response
from .noise import NoiseModel

#: The chains by the name :class:`~grand.sim.efield2voltage.Efield2Voltage` gives them
RF_CHAINS = {"rf_chain": RFChain, "rf_chainnut": RFChainNut, "rf_chaingaa": RFChain_gaa}


class DetectorResponse:
    r"""Builds the :class:`~grand.recon.deconvolution.base.Response` of each unit.

    Holds what does not change from unit to unit: the antenna tables, and the
    RF chain and interpolation for the last frequency axis asked for.

    Parameters
    ----------
    du_type : str, optional
        Antenna model, as for :class:`~grand.sim.detector.antenna_model.AntennaModel`.
    rf_chain : str, RF chain or None, optional
        One of ``"rf_chain"`` (the default of the simulation), ``"rf_chainnut"``
        or ``"rf_chaingaa"``; an object with ``compute_for_freqs`` and
        ``get_tf``, for another gain say; or None for the open-circuit
        voltage.  It must be the chain the voltages went through.
    antenna_model : AntennaModel, optional
        Already loaded tables, which are slow to load; they replace `du_type`.
    """

    def __init__(self, du_type="GP300", rf_chain="rf_chain", antenna_model=None):
        if isinstance(rf_chain, str):
            _validate.one_of(rf_chain, tuple(RF_CHAINS), "rf_chain", "DetectorResponse")
            rf_chain = RF_CHAINS[rf_chain]()
        self.rf_chain = rf_chain
        self.antenna_model = AntennaModel(du_type) if antenna_model is None else antenna_model
        self._freqs_mhz = None
        self._interpolation = None
        self._rf_tf = None

    def _set_frequencies(self, freqs_mhz):
        r"""Prepares the RF chain and the interpolation for a frequency axis, if new.

        Parameters
        ----------
        freqs_mhz : ndarray, shape (n_freq,)
            rFFT axis of the traces, in MHz.
        """
        if self._freqs_mhz is not None and np.array_equal(freqs_mhz, self._freqs_mhz):
            return
        # Held here, not by AntennaProcessing.init_interpolation: that sets a
        # class attribute a simulation in progress relies on.
        interpolation = PreComputeInterpol()
        interpolation.init_linear_interpol(self.antenna_model.leff_sn.frequency * 1e-6, freqs_mhz)
        if self.rf_chain is None:
            rf_tf = np.ones((3, freqs_mhz.size))
        else:
            self.rf_chain.compute_for_freqs(freqs_mhz)
            rf_tf = np.array(self.rf_chain.get_tf())
            if hasattr(self.rf_chain, "release_arrays"):
                self.rf_chain.release_arrays()
        self._freqs_mhz, self._interpolation, self._rf_tf = freqs_mhz, interpolation, rf_tf

    def transfer_matrix(self, du_position, shower, freqs_mhz):
        r"""Transfer matrix of one unit, and its line of sight to the shower maximum.

        Parameters
        ----------
        du_position : array_like, shape (3,)
            Position of the unit, in m, in the array frame (``TRun.du_xyz``).
        shower : ShowerEvent
            Gives the frame of the field (``frame``), the array frame and the
            source (``maximum``), after ``load_root``.
        freqs_mhz : ndarray, shape (n_freq,)
            rFFT axis of the traces, in MHz, starting at 0.

        Returns
        -------
        transfer : ndarray, complex, shape (n_freq, 3, 3)
            Voltage of arm ``k`` (SN, EW, UP) for a unit field along component
            ``j`` of ``shower.frame``.  Zero outside the tables' band, and at
            every frequency for a source below the antenna's horizon.
        line_of_sight : ndarray, shape (3,)
            Unit vector from the source to the unit, in ``shower.frame``.
        """
        where = "DetectorResponse.transfer_matrix"
        du_position = _validate.as_array(du_position, "du_position", where, shape=(3,), finite=True)
        freqs_mhz = _validate.as_array(freqs_mhz, "freqs_mhz", where, ndim=1, min_length=2,
                                       finite=True)
        self._set_frequencies(freqs_mhz)

        frame = antenna_frame(du_position, shower.grand_ref_frame)
        model = self.antenna_model
        arms = [AntennaProcessing(model_leff=leff, pos=frame)
                for leff in (model.leff_sn, model.leff_ew, model.leff_z)]
        theta, phi = arms[0].source_direction(shower.maximum)

        leff = np.empty((3, 3, freqs_mhz.size), dtype=complex)
        for k, arm in enumerate(arms):
            arm.set_out_freq_mhz(freqs_mhz)
            arm.pre_cpt = self._interpolation
            leff[k] = arm.leff_in_direction(theta, phi, shower.frame)
        transfer = np.moveaxis(self._rf_tf[:, np.newaxis, :] * leff, 2, 0)

        theta_rad = np.deg2rad(float(np.ravel(theta)[0]))
        phi_rad = np.deg2rad(float(np.ravel(phi)[0]))
        to_source = np.array([np.sin(theta_rad) * np.cos(phi_rad),
                              np.sin(theta_rad) * np.sin(phi_rad),
                              np.cos(theta_rad)])
        return transfer, -arms[0].antenna_to_frame(to_source, shower.frame)

    def response(self, du_position, shower, freqs_mhz, noise, polarization=None, n_fft=None):
        r"""Everything a deconvolution method needs for one unit.

        Its band is where the antenna responds and the noise is known: nothing
        can be learnt about the field elsewhere, so nothing is estimated, and
        the error bars of a deconvolution are those of the field in that band.

        Parameters
        ----------
        du_position, shower, freqs_mhz
            As for :meth:`transfer_matrix`.
        noise : NoiseModel, float or ndarray, shape (n_freq, 3)
            A model of the noise, or its PSD in the voltage, per rFFT bin.
        polarization : array_like, shape (3,), or callable, optional
            Polarization of the field, in the components of ``shower.frame``; or a
            function giving it from the unit's line of sight, as
            :func:`~grand.recon.deconvolution.polarization.geomagnetic_polarization`
            with its other arguments set.
        n_fft : int, optional
            Length of the transform `freqs_mhz` belongs to, which a noise model
            needs.  Even by default, ``2 * (n_freq - 1)``; an odd one has to be
            given.

        Returns
        -------
        Response
        """
        transfer, line_of_sight = self.transfer_matrix(du_position, shower, freqs_mhz)
        n_freq = transfer.shape[0]
        if isinstance(noise, NoiseModel):
            n_fft = 2 * (n_freq - 1) if n_fft is None else n_fft
            noise_psd = noise.psd(self._freqs_mhz, n_fft, self._rf_tf)
        else:
            noise_psd = np.broadcast_to(noise, (n_freq, 3))
        band = np.any(transfer != 0, axis=(1, 2)) & np.all(noise_psd > 0, axis=1)
        if callable(polarization):
            polarization = polarization(line_of_sight)
        return Response(transfer=transfer, noise_psd=noise_psd, polarization=polarization,
                        line_of_sight=line_of_sight, band=band)
