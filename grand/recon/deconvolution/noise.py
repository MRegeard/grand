r"""Models of the noise power spectrum in the voltage of a detection unit.

A model gives the PSD of the noise on each arm in the normalisation of
:mod:`~grand.recon.deconvolution.base`, at the point of the chain where the
voltages were taken.  Arms are taken as uncorrelated.  Where a model has nothing
to say it returns zero, and those frequencies are not estimated: the result is
restricted to the model's band.
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy.fft as sf
import scipy.signal as ss

from grand.basis import validate as _validate
from grand.sim.noise.galaxy import galactic_noise_psd


class NoiseModel(ABC):
    r"""A model of the noise PSD in the voltage."""

    @abstractmethod
    def psd(self, freqs_mhz, n_fft, rf_tf):
        r"""The noise PSD on a frequency axis.

        Parameters
        ----------
        freqs_mhz : ndarray, shape (n_freq,)
            rFFT axis of the traces, in MHz.
        n_fft : int
            Length of the transform the axis belongs to.
        rf_tf : ndarray, complex, shape (3, n_freq)
            Transfer function of the RF chain, for a noise entering before it.

        Returns
        -------
        ndarray, shape (n_freq, 3)
            PSD per arm, in µV²; zero where the model does not apply.
        """


class WhiteNoise(NoiseModel):
    r"""Noise of the same power at every frequency, in the voltage.

    Parameters
    ----------
    sigma : float or array_like, shape (3,)
        RMS of the noise in the time domain, in µV: one value, or one per arm.
    """

    def __init__(self, sigma):
        sigma = _validate.as_array(sigma, "sigma", "WhiteNoise", finite=True)
        if sigma.shape not in ((), (3,)) or np.any(sigma <= 0):
            raise ValueError(_validate.message(
                "WhiteNoise", "'sigma' must be one positive value or one per arm, got %s"
                % (sigma.tolist(),)))
        self.sigma = sigma

    def psd(self, freqs_mhz, n_fft, rf_tf):
        r"""The noise PSD, see :meth:`NoiseModel.psd`.

        Parameters
        ----------
        freqs_mhz : ndarray, shape (n_freq,)
            rFFT axis of the traces, in MHz.
        n_fft : int
            Length of the transform.
        rf_tf : ndarray
            Not used: this noise is given as seen in the voltage.

        Returns
        -------
        ndarray, shape (n_freq, 3)
        """
        return np.broadcast_to(n_fft * self.sigma ** 2, (freqs_mhz.size, 3)).copy()


class GalacticNoise(NoiseModel):
    r"""Galactic noise through the RF chain: :math:`|RF|^2 P_{gal}`.

    The model of :func:`~grand.sim.noise.galaxy.galactic_noise`, without its
    random draw.  It is defined in the 30--250 MHz of its tables, so the
    deconvolution is restricted to that band.  It holds no electronic noise,
    like the simulation.

    Parameters
    ----------
    lst : float
        Local sidereal time, in hours.
    du_type : str, optional
        Antenna model: ``"GP300"``, ``"GP300_nec"`` or ``"GP300_mat"``.
    """

    def __init__(self, lst, du_type="GP300"):
        self.lst = lst
        self.du_type = du_type
        self._axis = None
        self._open_circuit = None

    def psd(self, freqs_mhz, n_fft, rf_tf):
        r"""The noise PSD, see :meth:`NoiseModel.psd`.

        Parameters
        ----------
        freqs_mhz : ndarray, shape (n_freq,)
            rFFT axis of the traces, in MHz.
        n_fft : int
            Length of the transform.
        rf_tf : ndarray, complex, shape (3, n_freq)
            Transfer function of the RF chain.

        Returns
        -------
        ndarray, shape (n_freq, 3)
            Zero outside the band of the tables, and where the chain is zero.
        """
        # The tables are read once per axis, not for each unit
        if (self._axis is None or self._axis[0] != n_fft
                or not np.array_equal(self._axis[1], freqs_mhz)):
            self._open_circuit = galactic_noise_psd(self.lst, n_fft, freqs_mhz, self.du_type)
            self._axis = (n_fft, freqs_mhz)
        return self._open_circuit * np.abs(rf_tf.T) ** 2


class MeasuredNoise(NoiseModel):
    r"""Noise PSD estimated from traces that hold no signal (Welch's method).

    Take them at the same point of the chain as the traces to deconvolve:
    forced triggers, or the part of a trace away from the pulse.

    Parameters
    ----------
    traces : array_like, shape (3, n_samples) or (n_traces, 3, n_samples)
        Noise voltages, in µV.
    t_bin_ns : float
        Sampling interval, in ns.
    n_segment : int, optional
        Length of the Hann-windowed, half-overlapping segments.  It sets the
        resolution of the estimate, and has to be no longer than the traces;
        the whole trace by default.  Shorter gives a smoother estimate, longer
        resolves narrow lines.
    """

    def __init__(self, traces, t_bin_ns, n_segment=None):
        where = "MeasuredNoise"
        traces = _validate.as_array(traces, "traces", where, finite=True)
        if traces.ndim == 2:
            traces = traces[np.newaxis]
        if traces.ndim != 3 or traces.shape[1] != 3:
            raise ValueError(_validate.message(
                where, "'traces' must have shape (3, n_samples) or (n_traces, 3, n_samples), "
                "got %s" % (traces.shape,)))
        t_bin_ns = _validate.as_real(t_bin_ns, "t_bin_ns", where)
        _validate.positive(t_bin_ns, "t_bin_ns", where, "ns")
        n_samples = traces.shape[2]
        n_segment = n_samples if n_segment is None else n_segment
        n_segment = _validate.as_integer(n_segment, "n_segment", where, minimum=2,
                                         maximum=n_samples)
        # Scaled per sample, not per Hz: white noise of variance s2 gives s2
        window = ss.get_window("hann", n_segment)
        step = n_segment // 2
        starts = range(0, n_samples - n_segment + 1, step)
        power = 0.0
        for start in starts:
            segment = traces[:, :, start:start + n_segment]
            segment = segment - segment.mean(axis=2, keepdims=True)
            power = power + np.abs(sf.rfft(segment * window, axis=2)) ** 2
        self.spectrum = np.mean(power, axis=0).T / (len(starts) * np.sum(window ** 2))
        self.freqs_mhz = sf.rfftfreq(n_segment, t_bin_ns * 1e-3)

    def psd(self, freqs_mhz, n_fft, rf_tf):
        r"""The noise PSD, see :meth:`NoiseModel.psd`.

        Parameters
        ----------
        freqs_mhz : ndarray, shape (n_freq,)
            rFFT axis of the traces, in MHz.
        n_fft : int
            Length of the transform.
        rf_tf : ndarray
            Not used: the noise was measured in the voltage.

        Returns
        -------
        ndarray, shape (n_freq, 3)
            Zero above the frequencies the noise traces reach.
        """
        return n_fft * np.stack([np.interp(freqs_mhz, self.freqs_mhz, self.spectrum[:, arm],
                                           right=0.0) for arm in range(3)], axis=1)
