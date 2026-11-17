r"""What every deconvolution method shares: its inputs, its output, its interface.

The measurement model of one detection unit, at each frequency :math:`\nu`, is

.. math::

    \vec V(\nu) = \mathbf T(\nu)\,\vec E(\nu) + \vec N(\nu)

with :math:`\vec V` the voltages of the three arms (SN, EW, UP), :math:`\vec E`
the three Cartesian components of the field, :math:`\mathbf T` the 3x3 transfer
matrix (effective length times RF chain) and :math:`\vec N` the noise.

Conventions
-----------
* Spectra are ``scipy.fft.rfft`` coefficients (``norm="backward"``) of the
  time traces, frequency first: a voltage spectrum has shape ``(n_freq, 3)``.
* A power spectral density (PSD) is the expected squared modulus of those
  coefficients, :math:`\mathbb E|X_k|^2`, per bin -- the normalisation
  :func:`grand.sim.noise.galaxy.galactic_noise` uses.  With voltages in µV and
  fields in µV/m, a noise PSD is in µV² and a signal PSD in (µV/m)².
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import scipy.fft as sf

from grand.basis import validate as _validate


@dataclass
class Response:
    r"""What a method needs to know about one detection unit.

    Parameters
    ----------
    transfer : ndarray, complex, shape (n_freq, 3, 3)
        Transfer matrix :math:`T_{kj}(\nu)`: voltage of arm ``k`` for a unit
        field along component ``j``, on the rFFT axis of the trace.
    noise_psd : ndarray, shape (n_freq, 3)
        PSD of the noise on each arm, as seen in the voltage.  Arms are taken
        as uncorrelated; it must be positive at every arm in `band`, and is not
        read outside it.
    polarization : ndarray, shape (3,), optional
        Direction of the field's linear polarization, in the components of
        `transfer`.  Normalised on construction.
    line_of_sight : ndarray, shape (3,), optional
        Direction the wave travels along, in the same components.  Normalised
        on construction.
    band : ndarray, bool, shape (n_freq,), optional
        Frequencies to estimate; a method leaves the others out.  All of them
        by default.
    """

    transfer: np.ndarray
    noise_psd: np.ndarray
    polarization: np.ndarray = None
    line_of_sight: np.ndarray = None
    band: np.ndarray = None

    def __post_init__(self):
        r"""Checks the shapes and values of the arrays."""
        where = "Response"
        self.transfer = _validate.as_array(self.transfer, "transfer", where, shape=(None, 3, 3),
                                           finite=True, dtype=complex)
        self.noise_psd = _validate.as_array(self.noise_psd, "noise_psd", where,
                                            shape=(self.n_freq, 3), finite=True)
        if self.band is None:
            self.band = np.ones(self.n_freq, dtype=bool)
        self.band = _validate.as_array(self.band, "band", where, shape=(self.n_freq,), dtype=bool)
        if np.any(self.noise_psd[self.band] <= 0):
            raise ValueError(_validate.message(
                where, "'noise_psd' must be positive at every arm of every frequency in 'band'; "
                "a zero noise makes the estimate infinitely confident"))
        for name in ("polarization", "line_of_sight"):
            value = getattr(self, name)
            if value is None:
                continue
            value = _validate.as_array(value, name, where, shape=(3,), finite=True)
            norm = np.linalg.norm(value)
            if norm == 0:
                raise ValueError(_validate.message(where, "'%s' must not be the zero vector" % name))
            setattr(self, name, value / norm)

    @property
    def n_freq(self):
        """Number of frequency bins.

        Returns
        -------
        int
        """
        return self.transfer.shape[0]


@dataclass
class DeconvolvedSpectrum:
    r"""The field a method recovered, in the frequency domain.

    Parameters
    ----------
    efield : ndarray, complex, shape (n_freq, 3)
        Estimated field spectrum, in the components of the transfer matrix.
    variance : ndarray, shape (n_freq, 3)
        Variance of the estimate of each component in each bin, in the PSD
        normalisation of the module.  Zero where `valid` is False.
    valid : ndarray, bool, shape (n_freq,)
        Bins where the field could be estimated.  Elsewhere `efield` is set to
        zero: the result is band-limited to the valid bins.
    """

    efield: np.ndarray
    variance: np.ndarray
    valid: np.ndarray


class Deconvolution(ABC):
    r"""A method recovering the electric field of one unit from its voltages.

    A subclass implements :meth:`deconvolve_spectrum`; :meth:`deconvolve` then
    handles the time domain.  The method's own settings (priors, mode, ...) go
    into the subclass's constructor, so that the same object can process
    every unit of every event.
    """

    @abstractmethod
    def deconvolve_spectrum(self, voltage_f, response):
        r"""Recovers the field spectrum from the voltage spectrum of one unit.

        Parameters
        ----------
        voltage_f : ndarray, complex, shape (n_freq, 3)
            rFFT of the voltage of the three arms.
        response : Response
            The unit's transfer matrix and noise, on the same frequency axis.

        Returns
        -------
        DeconvolvedSpectrum
        """

    def deconvolve(self, voltage, response, n_fft=None):
        r"""Recovers the field time trace from the voltage time trace of one unit.

        Parameters
        ----------
        voltage : ndarray, shape (3, n_samples)
            Voltage of the three arms, in µV.  The time-major layout of
            :attr:`grand.aoi.Voltage.trace`.
        response : Response
            The unit's response on the rFFT axis of `n_fft` samples, that is
            with ``n_fft // 2 + 1`` frequencies.
        n_fft : int, optional
            Transform length; the trace is zero-padded to it.  The trace
            length by default.

        Returns
        -------
        efield : ndarray, shape (3, n_samples)
            The field, on the time axis of `voltage`.
        sigma : ndarray, shape (3,)
            Standard deviation of each component, the same at every sample.
        """
        where = "%s.deconvolve" % type(self).__name__
        voltage = _validate.as_array(voltage, "voltage", where, shape=(3, None), finite=True)
        n_samples = voltage.shape[1]
        if n_fft is None:
            n_fft = n_samples
        n_fft = _validate.as_integer(n_fft, "n_fft", where, minimum=n_samples)
        if not isinstance(response, Response):
            raise TypeError(_validate.message(where, "'response' must be a Response, got %s"
                                              % type(response).__name__))
        if response.n_freq != n_fft // 2 + 1:
            raise ValueError(_validate.message(
                where, "the response has %d frequencies; an rFFT of %d samples has %d"
                % (response.n_freq, n_fft, n_fft // 2 + 1)))

        voltage_f = sf.rfft(voltage, n=n_fft, axis=1).T
        result = self.deconvolve_spectrum(voltage_f, response)
        efield = sf.irfft(result.efield.T, n=n_fft, axis=1)[:, :n_samples]
        return efield, np.sqrt(time_variance(result.variance, n_fft))


def time_variance(variance_f, n_fft):
    r"""Variance in the time domain of a trace whose rFFT bins have `variance_f`.

    With independent errors in the bins, every sample of the inverse transform
    has the same variance,

    .. math::

        \sigma_t^2 = \frac{1}{N^2}\Big(v_0 + 2\sum_{0<k<N/2} v_k + v_{N/2}\Big)

    where the Nyquist term exists only for even `n_fft`: the other bins stand
    for themselves and their negative-frequency mirror.

    Parameters
    ----------
    variance_f : ndarray, shape (n_fft // 2 + 1, ...)
        Variance per bin, frequency first.
    n_fft : int
        Length of the transform.

    Returns
    -------
    ndarray, shape variance_f.shape[1:]
    """
    weights = np.full(variance_f.shape[0], 2.0)
    weights[0] = 1.0
    if n_fft % 2 == 0:
        weights[-1] = 1.0
    return np.tensordot(weights, variance_f, axes=1) / n_fft**2
