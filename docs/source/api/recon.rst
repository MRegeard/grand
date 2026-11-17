Deconvolution
=============

:mod:`grand.recon.deconvolution` recovers the electric field at each detection
unit from its voltages.  A method derives from
:class:`~grand.recon.deconvolution.base.Deconvolution`;
:class:`~grand.recon.deconvolution.wiener.WienerDeconvolution` is the only one so
far.  :class:`~grand.recon.deconvolution.event.EventDeconvolver` applies a method
to a whole AOI event.  It returns the fields without writing them.

.. note::

   The tests check it on the ZHAireS shower in ``sim2root/Common``, simulated
   with :class:`~grand.sim.efield2voltage.Efield2Voltage` from the same antenna
   and RF-chain models.  It has not been tested on other geometries or on measured
   data, whose ADC counts have to be converted to µV first.

What comes back is limited by the antenna.  It is the part of the field in the band
of the tables and the noise model, 30--250 MHz for
:class:`~grand.recon.deconvolution.noise.GalacticNoise`.  It is also transverse to
the line of sight: the arms do not see the component along it.

.. autosummary::

   grand.recon.deconvolution.base
   grand.recon.deconvolution.wiener
   grand.recon.deconvolution.event
   grand.recon.deconvolution.response
   grand.recon.deconvolution.noise
   grand.recon.deconvolution.polarization

``grand.recon.deconvolution.base``
----------------------------------

.. automodule:: grand.recon.deconvolution.base
   :members:

``grand.recon.deconvolution.wiener``
------------------------------------

.. automodule:: grand.recon.deconvolution.wiener
   :members:

``grand.recon.deconvolution.event``
-----------------------------------

.. automodule:: grand.recon.deconvolution.event
   :members:

``grand.recon.deconvolution.response``
--------------------------------------

.. automodule:: grand.recon.deconvolution.response
   :members:

``grand.recon.deconvolution.noise``
-----------------------------------

.. automodule:: grand.recon.deconvolution.noise
   :members:

``grand.recon.deconvolution.polarization``
------------------------------------------

.. automodule:: grand.recon.deconvolution.polarization
   :members:
