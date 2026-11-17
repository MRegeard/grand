"""Deconvolution of the antenna response: from voltage back to electric field.

Every method derives from :class:`Deconvolution` and works on plain arrays, one
detection unit at a time.  It needs the unit's :class:`Response` (transfer matrix
and noise PSD) on the rFFT frequency axis of the trace, which
:class:`DetectorResponse` builds from the simulation's antenna and RF-chain
models.  :class:`EventDeconvolver` applies a method to a whole AOI event.
"""

from .base import Deconvolution, DeconvolvedSpectrum, Response
from .event import EventDeconvolution, EventDeconvolver
from .noise import GalacticNoise, MeasuredNoise, NoiseModel, WhiteNoise
from .polarization import geomagnetic_polarization, polarization_angle
from .response import DetectorResponse
from .wiener import WienerDeconvolution

__all__ = ["Deconvolution", "DeconvolvedSpectrum", "DetectorResponse", "EventDeconvolution",
           "EventDeconvolver", "GalacticNoise", "MeasuredNoise", "NoiseModel", "Response", "WhiteNoise", "WienerDeconvolution",
           "geomagnetic_polarization", "polarization_angle"]
