r"""The direction of the electric field of an air shower, from its geometry.

Geomagnetic emission polarizes the field along :math:`\vec v \times \vec B`, with
:math:`\vec v` the direction the shower travels.  The wave reaches a unit along
the line of sight :math:`\vec n` from the shower maximum, and is transverse to
it, so the direction expected at that unit is

.. math::

    \vec p \propto (\mathbf I_3 - \vec n\vec n^T)\,(\vec v \times \vec B)

This leaves out the charge-excess emission, which is radial around the shower
axis.  On the ZHAireS event in ``sim2root/Common``, :math:`\vec p` is within 1.9
degrees of the simulated field on the 18 units above 100 µV/m in 30--250 MHz, and
up to 78 degrees off on the weakest.

Vectors are in the frame of the shower, ``ShowerEvent.frame``: north, west and up,
from magnetic north.
"""

import numpy as np

from grand.basis import validate as _validate


def shower_axis(zenith, azimuth):
    r"""The unit vector along which a shower travels.

    Parameters
    ----------
    zenith, azimuth : float
        Direction the shower comes from, in degrees, the azimuth from north
        toward west, as ``TShower`` holds them.

    Returns
    -------
    ndarray, shape (3,)
    """
    where = "shower_axis"
    theta = np.deg2rad(_validate.as_real(zenith, "zenith", where))
    phi = np.deg2rad(_validate.as_real(azimuth, "azimuth", where))
    return -np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def geomagnetic_direction(inclination):
    r"""The unit vector along the geomagnetic field.

    The frame is measured from magnetic north, so the declination does not enter.

    Parameters
    ----------
    inclination : float
        Inclination in degrees, positive when the field points down, as the first
        value of ``TShower.magnetic_field``.

    Returns
    -------
    ndarray, shape (3,)
    """
    inclination = np.deg2rad(_validate.as_real(inclination, "inclination", "geomagnetic_direction"))
    return np.array([np.cos(inclination), 0.0, -np.sin(inclination)])


def geomagnetic_polarization(line_of_sight, zenith, azimuth, inclination):
    r"""The unit vector :math:`\vec p` expected at a unit, see the module.

    Parameters
    ----------
    line_of_sight : array_like, shape (3,)
        Direction from the shower maximum to the unit.
    zenith, azimuth : float
        Direction the shower comes from, in degrees, see :func:`shower_axis`.
    inclination : float
        Inclination of the geomagnetic field, in degrees.

    Returns
    -------
    ndarray, shape (3,)
        Along the projection of :math:`\vec v \times \vec B`.

    Raises
    ------
    ValueError
        If the shower travels along the geomagnetic field, where this emission
        vanishes, or the unit sees the wave along :math:`\vec v \times \vec B`.
    """
    where = "geomagnetic_polarization"
    n = _unit(line_of_sight, "line_of_sight", where)
    v_cross_b = np.cross(shower_axis(zenith, azimuth), geomagnetic_direction(inclination))
    if np.linalg.norm(v_cross_b) < 1e-6:
        raise ValueError(_validate.message(
            where, "the shower travels along the geomagnetic field: the geomagnetic emission "
            "vanishes and gives no direction"))
    projected = v_cross_b - n * (n @ v_cross_b)
    if np.linalg.norm(projected) < 1e-6:
        raise ValueError(_validate.message(
            where, "the line of sight is along v x B: nothing of the geomagnetic emission "
            "is transverse to it"))
    return projected / np.linalg.norm(projected)


def polarization_angle(polarization, line_of_sight):
    r"""The polarization angle :math:`\alpha`, in the tangential frame of the source.

    :math:`\vec p = \cos\alpha\,\vec u_\theta + \sin\alpha\,\vec u_\phi`, with the
    unit vectors of the spherical coordinates of the source's direction,
    :math:`-\vec n`.  For :math:`\vec n` along the shower axis, this is

    .. math::

        \cos\alpha = \frac{\cos\beta \sin\phi}{\lVert \vec v \times \vec B \rVert}

    Without the denominator, which depends on the zenith angle, this is the formula
    of J.-M. Colley's 2025 slides: good for inclined showers with a geomagnetic angle
    near 90 degrees, as the ones it was checked on, and not otherwise.

    Parameters
    ----------
    polarization : array_like, shape (3,)
        Direction of the field.
    line_of_sight : array_like, shape (3,)
        Direction from the source to the unit.

    Returns
    -------
    float
        In degrees, in ``[0, 180)``: a direction has no sign.
    """
    where = "polarization_angle"
    p = _unit(polarization, "polarization", where)
    to_source = -_unit(line_of_sight, "line_of_sight", where)
    theta = np.arccos(np.clip(to_source[2], -1.0, 1.0))
    phi = np.arctan2(to_source[1], to_source[0])
    u_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    u_phi = np.array([-np.sin(phi), np.cos(phi), 0.0])
    return float(np.degrees(np.arctan2(p @ u_phi, p @ u_theta)) % 180.0)


def _unit(vector, name, where):
    r"""Returns `vector` normalized, after checking it.

    Parameters
    ----------
    vector : array_like, shape (3,)
        The vector.
    name, where : str
        Its name and the function concerned, for the message.

    Returns
    -------
    ndarray, shape (3,)
    """
    vector = _validate.as_array(vector, name, where, shape=(3,), finite=True)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError(_validate.message(where, "'%s' must not be the zero vector" % name))
    return vector / norm
