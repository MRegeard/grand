r"""Wiener deconvolution: the linear estimate of the field with the least mean square error.

For one unit, with signal covariance :math:`\mathbf C_E` and noise covariance
:math:`\mathbf C_N`, the Wiener estimate and its covariance are

.. math::

    \hat{\vec E} = \left[\mathbf C_E^{-1} + \mathbf T^\dagger \mathbf C_N^{-1}\mathbf T\right]^{-1}
                   \mathbf T^\dagger \mathbf C_N^{-1}\vec V,
    \qquad
    \mathbf C_{\hat E} = \left[\mathbf C_E^{-1} + \mathbf T^\dagger \mathbf C_N^{-1}\mathbf T\right]^{-1}

All three modes write the field as :math:`\vec E = \mathbf B\,\vec a`, with
:math:`\mathbf B` a fixed 3 x m basis and an isotropic prior
:math:`\mathbf C_a = P_e\,\mathbf I_m` on the m amplitudes:

``"full3d"``
    :math:`\mathbf B = \mathbf I_3`: no assumption on the polarization.  An
    antenna does not see the component along the line of sight (see
    :mod:`~grand.recon.deconvolution.response`), so that one stays at zero,
    with the variance of the prior, and this mode needs a finite prior.
``"transverse"``
    :math:`\mathbf B` spans the plane normal to the line of sight: a
    transverse wave, polarization free.
``"polarized"``
    :math:`\mathbf B = \vec p`: a field linearly polarized along a known
    direction, as an air shower's.  The three arms then combine into one
    scalar estimate per frequency.

The estimate is then solved for :math:`\vec a`, in an m x m system with
:math:`\mathbf G = \mathbf T\mathbf B`, so a rank-deficient :math:`\mathbf C_E`
causes no inverse of a singular matrix.
"""

import numpy as np

from grand.basis import validate as _validate
from .base import Deconvolution, DeconvolvedSpectrum, Response

MODES = ("full3d", "transverse", "polarized")


class WienerDeconvolution(Deconvolution):
    r"""Wiener deconvolution of one unit, with a prior on the polarization.

    Parameters
    ----------
    signal_psd : float, ndarray, shape (n_freq,), or "auto"
        PSD :math:`P_e` of the field amplitude along each basis direction, in
        (µV/m)² per rFFT bin.  ``numpy.inf`` drops the prior: the estimate is
        then the noise-weighted least-squares inverse, unbiased but unbounded
        where the response vanishes; such bins are left out (see `rcond`).
        ``"auto"`` estimates one flat level for each trace, see
        :meth:`estimate_flat_level`.  A field's power varies by orders of
        magnitude from unit to unit, so no single number suits an event.
    mode : {"polarized", "transverse", "full3d"}, optional
        Prior on the polarization, see the module.  ``"polarized"`` needs
        :attr:`Response.polarization`, ``"transverse"`` needs
        :attr:`Response.line_of_sight`.
    rcond : float, optional
        Only without a prior: bins where the smallest eigenvalue of
        :math:`\mathbf G^\dagger \mathbf C_N^{-1} \mathbf G` is below `rcond`
        times the largest one over all bins are not estimated.  This leaves
        out the frequencies outside the band of the response.

    Examples
    --------
    One arm along x, unit response, a signal PSD four times the noise: the
    filter keeps 4/5 of the voltage.

    >>> import numpy as np
    >>> from grand.recon.deconvolution import Response, WienerDeconvolution
    >>> n_freq = 5
    >>> response = Response(transfer=np.broadcast_to(np.eye(3), (n_freq, 3, 3)),
    ...                     noise_psd=np.ones((n_freq, 3)), polarization=[1, 0, 0])
    >>> wiener = WienerDeconvolution(signal_psd=4.0)
    >>> voltage_f = np.tile([1.0, 0.0, 0.0], (n_freq, 1))
    >>> wiener.deconvolve_spectrum(voltage_f, response).efield[0].real
    array([0.8, 0. , 0. ])

    Notes
    -----
    The filter minimises the squared error over the whole trace, and treats the
    signal as stationary: it does not know the field is a short pulse.  A finite
    prior therefore lowers the peak, the more so as the signal-to-noise ratio
    falls, whereas no prior leaves it unbiased, at the price of a larger error.
    On the ZHAireS event in ``sim2root/Common``, ``"auto"`` lowers the error by
    4 to 90 times, and the peak by about 20% above a voltage signal-to-noise
    ratio of 15, and by about 90% between 1 and 15.
    """

    def __init__(self, signal_psd, mode="polarized", rcond=1e-12):
        where = "WienerDeconvolution"
        _validate.one_of(mode, MODES, "mode", where)
        if isinstance(signal_psd, str):
            _validate.one_of(signal_psd, ("auto",), "signal_psd", where)
        else:
            signal_psd = _validate.as_array(signal_psd, "signal_psd", where)
            if signal_psd.ndim > 1:
                raise ValueError(_validate.message(
                    where, "'signal_psd' must be a number or one value per frequency, got shape %s"
                    % (signal_psd.shape,)))
            if np.any(np.isnan(signal_psd)) or np.any(signal_psd <= 0):
                raise ValueError(_validate.message(
                    where, "'signal_psd' must be positive (numpy.inf for no prior); a zero signal "
                    "PSD would give a zero field everywhere"))
        self.signal_psd = signal_psd
        self.mode = mode
        self.rcond = _validate.as_real(rcond, "rcond", where)

    def basis(self, response):
        r"""The 3 x m basis :math:`\mathbf B` the field is expanded on, for this mode.

        Parameters
        ----------
        response : Response
            Supplies the polarization or the line of sight.

        Returns
        -------
        ndarray, shape (3, m)
            Orthonormal columns.
        """
        where = "WienerDeconvolution"
        if self.mode == "full3d":
            return np.eye(3)
        if self.mode == "polarized":
            if response.polarization is None:
                raise ValueError(_validate.message(
                    where, "mode 'polarized' needs the response's polarization"))
            return response.polarization[:, np.newaxis]
        if response.line_of_sight is None:
            raise ValueError(_validate.message(
                where, "mode 'transverse' needs the response's line_of_sight"))
        # The last two right-singular vectors of n span the plane normal to it
        return np.linalg.svd(response.line_of_sight[np.newaxis, :])[2][1:].T

    def estimate_flat_level(self, voltage_f, response):
        r"""The flat :math:`P_e` of a trace, from its power in excess of the noise.

        With a flat :math:`P_e`, :math:`\mathbb E[\vec V^\dagger \mathbf C_N^{-1} \vec V]
        = P_e \,\mathrm{tr}(\mathbf G^\dagger \mathbf C_N^{-1} \mathbf G) + 3` in each
        bin, which gives :math:`P_e` by summing over the band.  The result is kept
        no lower than the statistical uncertainty of that sum: a trace that holds
        only noise is then shrunk strongly, with a finite error bar.

        Parameters
        ----------
        voltage_f : ndarray, complex, shape (n_freq, 3)
            rFFT of the voltage of the three arms.
        response : Response
            The unit's transfer matrix, noise and geometry.

        Returns
        -------
        float
            In (µV/m)² per rFFT bin; ``numpy.inf``, that is no prior, if the
            response is zero over the whole band.
        """
        band = response.band
        noise_psd = response.noise_psd[band]
        gain = response.transfer[band] @ self.basis(response)
        information = np.sum(np.abs(gain) ** 2 / noise_psd[:, :, np.newaxis])
        if information <= 0:
            return np.inf
        n_terms = 3 * np.count_nonzero(band)
        excess = np.sum(np.abs(voltage_f[band]) ** 2 / noise_psd) - n_terms
        return max(excess, np.sqrt(n_terms)) / information

    def deconvolve_spectrum(self, voltage_f, response):
        r"""Wiener estimate of the field spectrum of one unit.

        Parameters
        ----------
        voltage_f : ndarray, complex, shape (n_freq, 3)
            rFFT of the voltage of the three arms.
        response : Response
            The unit's transfer matrix, noise and geometry.

        Returns
        -------
        DeconvolvedSpectrum
        """
        where = "WienerDeconvolution.deconvolve_spectrum"
        n_freq = response.n_freq
        voltage_f = _validate.as_array(voltage_f, "voltage_f", where, shape=(n_freq, 3),
                                       finite=True, dtype=complex)
        if isinstance(self.signal_psd, str):
            signal_psd = np.full(n_freq, self.estimate_flat_level(voltage_f, response))
        elif self.signal_psd.ndim == 1 and self.signal_psd.size != n_freq:
            raise ValueError(_validate.message(
                where, "'signal_psd' has %d values for %d frequencies"
                % (self.signal_psd.size, n_freq)))
        else:
            signal_psd = np.broadcast_to(self.signal_psd, n_freq)

        basis = self.basis(response)
        m = basis.shape[1]
        gain = response.transfer @ basis                                  # (n_freq, 3, m)
        band = response.band
        noise_psd = np.where(band[:, np.newaxis], response.noise_psd, 1.0)  # not read outside
        weighted = gain.conj().swapaxes(1, 2) / noise_psd[:, np.newaxis, :]
        projected = np.einsum("fmk,fk->fm", weighted, voltage_f)          # G^† C_N^-1 V

        # Diagonalising G^† C_N^-1 G keeps its null direction exact: the
        # inverse there is the prior, whatever the range of the eigenvalues.
        eigenvalues, vectors = np.linalg.eigh(weighted @ gain)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        # A prior defines every bin; without one, the data have to
        largest = np.max(eigenvalues[band], initial=0.0)
        valid = band & (np.isfinite(signal_psd) | (eigenvalues[:, 0] > self.rcond * largest))
        if not valid.any():
            _validate.warn(where, "no frequency can be estimated in mode '%s', the field is zero: "
                           "the response vanishes, or it has rank 2 and the mode has no prior "
                           "(use 'transverse' or a finite signal_psd)" % self.mode)

        covariance = np.zeros((n_freq, m, m), dtype=complex)
        inverse = 1.0 / (eigenvalues[valid] + 1.0 / signal_psd[valid, np.newaxis])
        covariance[valid] = (vectors[valid] * inverse[:, np.newaxis, :]
                             @ vectors[valid].conj().swapaxes(1, 2))
        amplitude = np.einsum("fmn,fn->fm", covariance, projected)

        efield = amplitude @ basis.T
        variance = np.einsum("km,fmn,kn->fk", basis, covariance, basis).real
        return DeconvolvedSpectrum(efield=efield, variance=variance, valid=valid)
