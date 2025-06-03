# This file is part of sedprep
#
# Copyright (C) 2020 Helmholtz Centre Potsdam
# GFZ German Research Centre for Geosciences, Potsdam, Germany
# (https://www.gfz-potsdam.de)
#
# sedprep is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# sedprep is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import torch
import scipy
import requests
import io
from warnings import warn

from pymagglobal.utils import lmax2N
# from pymagglobal.utils import dsh_basis as _dsh_basis

from sedprep.constants import rad2deg, field_params, dtype, REARTH


def nez2dif(n, e, z, f_shallow=1, cal_fac=1, be=torch):
    """
    Transform the magnetic field components north, east and vertically
    downward to declination, inclination and intensity. This is similar to a
    transformation from Cartesian to spherical coordinates.

    Parameters
    ----------
    n : numpy.ndarray
        Pointing towards north.
    e : numpy.ndarray
        Pointing towards east.
    z : numpy.ndarray
        Pointing radially downwards.

    Returns
    -------
    numpy.ndarray
        Declination.
    numpy.ndarray
        Inclination.
    numpy.ndarray
        Intensity.

    Example
    -------
    >>> nez2dif(1, 0, 0, be=np)
    (0.0, 0.0, 1.0)
    >>> nez2dif(*np.array([1, 0, 2]), f_shallow=0.5, be=np)
    (0.0, 45.0, 2.23606797749979)
    >>> nez2dif(*torch.tensor([1, 0, 0]), be=torch)
    (tensor(0.), tensor(0.), tensor(1.))

    """
    return (
        be.arctan2(e, n) * rad2deg,
        # torch.arcsin(_z / torch.sqrt(n**2 + e**2 + _z**2)) * rad2deg,
        be.arctan2(f_shallow * z, be.sqrt(n**2 + e**2)) * rad2deg,
        be.sqrt(n**2 + e**2 + z**2) / cal_fac,
    )


def grad_d(b):
    """
    Calculate the gradient of the declination.

    Parameters
    ----------
    b : torch.Tensor
        The input tensor with shape (N, 3) representing N geomagnetic field
        components north, east and vertically.

    Returns
    -------
    torch.Tensor
        The gradient of the declination, with the same shape as the input 'b'.

    Examples
    --------
    >>> b = torch.tensor([[1.0, 2.0, 3.0]])
    >>> grad_d(b)
    tensor([[-22.9183,  11.4592,   0.0000]])
    >>> b = torch.tensor([[1.0, 2.0, 3.0], [5, 6, 7]])
    >>> grad_d(b)
    tensor([[-22.9183,  11.4592,   0.0000],
            [ -5.6357,   4.6964,   0.0000]])
    """
    res = torch.zeros_like(b, device=b.device)
    res[:, 0] = -b[:, 1]
    res[:, 1] = b[:, 0]
    res /= (b[:, 0] ** 2 + b[:, 1] ** 2)[:, None]
    return res * rad2deg


def grad_i(b, f_shallow=1):
    """
    Calculate the gradient of the inclination.

    Parameters
    ----------
    b : torch.Tensor
        The input tensor with shape (N, 3) representing N geomagnetic field
        components north, east and vertically.
    f_shallow : float, optional
        A scaling factor for the inclination shallow effect.
        It has to be a number between 0 (complete shallowing)
        and 1 (no shallowing). Default is 1.0.

    Returns
    -------
    torch.Tensor
        The gradient of the inclination, with the same shape as the input 'b'.

    Examples
    --------
    >>> b = torch.tensor([[1.0, 2.0, 3.0], [5, 6, 7]])
    >>> grad_i(b)
    tensor([[ -5.4907, -10.9815,   9.1512],
            [ -2.3342,  -2.8010,   4.0681]])
    >>> grad_i(b, f_shallow=0.5)
    tensor([[ -5.3014, -10.6028,   8.8357],
            [ -1.7526,  -2.1031,   3.0546]])
    """
    res = torch.zeros_like(b, device=b.device)
    F_H_sq = b[:, 0] ** 2 + b[:, 1] ** 2
    F_f = F_H_sq + f_shallow**2 * b[:, 2] ** 2
    res[:, 0] = -b[:, 2] * b[:, 0]
    res[:, 1] = -b[:, 2] * b[:, 1]
    res[:, 2] = F_H_sq
    res /= (torch.sqrt(F_H_sq) * F_f)[:, None]
    return f_shallow * res * rad2deg


def grad_f(b, cal_fac=1):
    """
    Calculate the gradient of the intensity.

    Parameters
    ----------
    b : torch.Tensor
        The input tensor with shape (N, 3) representing N geomagnetic field
        components north, east and vertically.

    Returns
    -------
    torch.Tensor
        The gradient of the intensity, with the same shape as the input 'b'.

    Examples
    --------
    >>> b = torch.tensor([[1.0, 2.0, 3.0], [5, 6, 7]])
    >>> grad_f(b)
    tensor([[0.2673, 0.5345, 0.8018],
            [0.4767, 0.5721, 0.6674]])
    """
    res = b / torch.sqrt(b[:, 0] ** 2 + b[:, 1] ** 2 + b[:, 2] ** 2)[:, None]
    return res / cal_fac


def lif(bs):
    """
    Returns the parameterized lock-in function for given parameters b_1 to b_4.

    Parameters
    ----------
    bs : numpy.ndarray, list or torch.tensor
        The four or two parameters that characterize the lock-in function

    Returns
    -------
    scipy.interpolate.BSpline
        The parameterized lock-in function as a BSpline

    Example
    -------
    >>> lock_in_func = lif([10, 10, 0, 10])
    >>> lock_in_func(15)
    array(0.05)
    >>> lock_in_func = lif(np.array([10, 10, 0, 10]))
    >>> lock_in_func(0)
    array(0.)
    >>> lock_in_func = lif(torch.tensor([10, 10, 0, 10]))
    >>> lock_in_func(20)
    array(0.1)

    """
    if len(bs) == 2:
        bs = [bs[0], bs[1] / 2, bs[1], bs[1] / 2]
    if torch.is_tensor(bs):
        bs = bs.cpu().detach().numpy()
    pars = np.cumsum(bs)
    knots = np.concatenate(([-2, -1, 0], pars, [pars[3] + 1], [pars[3] + 2]))
    vals = np.zeros(7)
    vals[3] = 1
    vals[4] = 1
    norm = (pars[3] + pars[2] - pars[1] - pars[0]) / 2
    if norm != 0:
        vals /= norm
    # linear interpolation of lock-in function
    lockin_function = scipy.interpolate.BSpline(knots, vals, 1)
    return lockin_function


def lif_antid(bs, use_spline=False, be=torch):
    """
    Returns the antiderivative of the parameterized lock-in function
    for given parameters b_1 to b_4.

    Parameters
    ----------
    bs : numpy.ndarray, list or torch.tensor
        The four or two parameters that characterize the lock-in function
    use_spline : bool, optional
        If True, returns a spline antiderivative. If False, returns a function.
        Default is False.
    be : torch or numpy, optional
        The library used for calculations. Default is the torch module.

    Returns
    -------
    function or scipy.interpolate.BSpline
        If `use_spline` is False, it returns a Python function that computes
        the antiderivative.
        If `use_spline` is True, it returns a spline antiderivative object.

    See Also
    --------
    lif : Lock-in function.

    Example
    -------
    >>> lock_in_func_antid = lif_antid([10, 10, 0, 10])
    >>> lock_in_func_antid(16)
    tensor(0.1800)
    >>> lock_in_func_antid = lif_antid([10, 10, 0, 10], use_spline=True)
    >>> lock_in_func_antid(16)
    array(0.18)
    >>> lock_in_func_antid = lif_antid(torch.tensor([10, 10, 0, 10]))
    >>> lock_in_func_antid(16)
    tensor(0.1800)
    >>> lock_in_func_antid = lif_antid(torch.tensor([10, 10, 0, 10]))
    >>> lock_in_func_antid((torch.linspace(8, 30, 8)))
    tensor([0.0000, 0.0065, 0.0918, 0.2759, 0.5555, 0.8024, 0.9506, 1.0000])

    """
    # if len(bs) == 2:
    #     bs_tmp = bs.clone()
    #     bs = [bs_tmp[0], bs_tmp[1]/2, bs_tmp[1], bs_tmp[1]/2]
    if isinstance(bs, list):
        bs = np.array(bs) if be == np else torch.tensor(bs)
    if use_spline:
        return lif(bs).antiderivative()
    else:
        bs = be.cumsum(bs, axis=0, dtype=dtype if be == torch else None)
        b0 = bs[0]
        b1 = bs[1] if len(bs) == 4 else (bs[0] + bs[1]) / 2
        b2 = bs[2] if len(bs) == 4 else (3 * bs[1] - bs[0]) / 2
        b3 = bs[3] if len(bs) == 4 else 2 * bs[1] - bs[0]
        beta = 2 / (b3 + b2 - b1 - b0)

        def f(x):
            if isinstance(x, (int, float)):
                x = np.array(x) if be == np else torch.tensor(x, dtype=dtype)
            cond1 = (b0 < x) & (x <= b1)
            cond2 = (b1 < x) & (x <= b2)
            cond3 = (b2 < x) & (x <= b3)
            result = be.zeros_like(x, dtype=dtype if be == torch else None)
            result[cond1] = beta * (x[cond1] - b0) ** 2 / (2 * (b1 - b0))
            result[cond2] = beta * (x[cond2] - (b1 + b0) / 2)
            result[cond3] = beta * (
                -(x[cond3] ** 2 / 2 - b3 * x[cond3] + b2**2 / 2) / (b3 - b2)
                - (b1 + b0) / 2
            )
            result[x > b3] = 1
            return result

        return f


def compute_hlid(bs):
    """
    Compute the half lock-in depth of a lock-in function
    characterized by the four parameters in `bs`.

    Parameters
    ----------
    bs : numpy.ndarray or list
        An array-like object containing four parameters (b_1, b_2, b_3, b_4)
        that characterize the lock-in function.

    Returns
    -------
    float
        The half lock-in depth of the lock-in function, which is the value of x
        such that the antiderivative of the lock-in function reaches 0.5.

    Example
    -------
    >>> bs = [10.0, 20.0, 15.0, 30.0]
    >>> compute_hlid(bs)
    40.0
    >>> bs = [5.0, 15.0, 10.0, 25.0]
    >>> compute_hlid(bs)
    27.5
    """
    if len(bs) == 2:
        return np.sum(bs) if bs[1] > 0 else 0
    lif_ant = lif(bs).antiderivative()
    bs_cs = np.cumsum(bs)
    return scipy.optimize.root_scalar(
        lambda x: lif_ant(x) - 0.5,
        x0=bs_cs[2] - bs_cs[1],
        bracket=[bs_cs[0], bs_cs[3]],
    ).root


def kalmag(mean_path, cov_path):
    """
    Returns the mean and covariance matrix in the right format.

    Parameters
    ----------
    mean_path : str
        Path to the mean (can also be a link including https)
    cov_path : str
        Path to the covariance (can also be a link including https)

    Returns
    -------
    tupe[numpy.ndarray, numpy.ndarray]
        Mean and covariance in the right format

    Notes
    -----
    We recommend using the Kalmag model [1]_

    References
    ----------
    .. [1] Julien, B., Matthias, H., Saynisch-Wagner, J. et al.
        Kalmag: a high spatio-temporal model of the geomagnetic field.
        Earth Planets Space 74, 139 (2022).
        https://doi.org/10.1186/s40623-022-01692-5
    """
    if "https:" in mean_path:
        r_mean = requests.get(mean_path, stream=True)
        r_cov = requests.get(cov_path, stream=True)
        mean_path = io.BytesIO(r_mean.raw.read())
        cov_path = io.BytesIO(r_cov.raw.read())
    n_coeffs = lmax2N(field_params["lmax"])
    kalmag_mean = np.genfromtxt(mean_path)
    kalmag_cov = np.genfromtxt(cov_path)
    inds = np.concatenate(
        (
            np.arange(0, n_coeffs),
            int(len(kalmag_mean) / 2) + np.arange(0, n_coeffs),
        )
    )
    kalmag_mean = kalmag_mean[inds] / 1000
    kalmag_cov = kalmag_cov[inds[None, :], inds[:, None]] / 1000 / 1000
    return kalmag_mean, kalmag_cov


def angles(vec):
    return np.arctan2(vec[1], vec[0]), np.pi / 2 - np.arctan2(
        vec[2], np.sqrt(vec[0] ** 2 + vec[1] ** 2)
    )


def rot_z(ang):
    return np.array(
        [
            [np.cos(ang), np.sin(ang), 0],
            [-np.sin(ang), np.cos(ang), 0],
            [0, 0, 1],
        ]
    )


def rot_y(ang):
    return np.array(
        [
            [np.cos(ang), 0, np.sin(ang)],
            [0, 1, 0],
            [-np.sin(ang), 0, np.cos(ang)],
        ]
    )


def rotator(vec):
    vec = np.asarray(vec)
    p, t = angles(vec)
    return np.dot(rot_y(t).T, rot_z(p))


def sample_Fisher(n, mu=(0, 0, 1), kappa=20):
    """Generate samples from the Fisher distribution

    Parameters:
    -----------
    n : int
        The number of samples to be generated
    mu : array-like of length 3, optional
        A vector pointing towards the center of the distribution.
        Its length is ignored.
    kappa : float, optional
        The concentration parameter.

    Returns:
    --------
        numpy array of shape (3, n) containing the sampled vectors

    Reference:
    ----------
    [1]: W. Jakob, "Numerically stable sampling of the von Mises
         Fisher distribution on S^2 (and other tricks)",
         http://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf,
         2015
    """
    if kappa <= 0:
        raise ValueError(
            f"The concentration parameter has to be positive, "
            f"but kappa={kappa} was given.\n"
            f"For kappa=0 use a uniform sampler on the sphere."
        )
    trafo_mat = rotator(mu)

    # sample from the uniform circle, V in [1]
    angles = scipy.stats.uniform.rvs(scale=2 * np.pi, size=n)
    vs = np.array([np.cos(angles), np.sin(angles)])

    # sample W in [1] via inverse cdf sampling
    def inv_cdf(x):
        return 1 + np.log(x + (1 - x) * np.exp(-2 * kappa)) / kappa

    unis = scipy.stats.uniform.rvs(size=n)
    ws = inv_cdf(unis)
    ret = np.sqrt(1 - ws**2) * vs
    res = np.einsum("i...,ij->j...", np.array([ret[0], ret[1], ws]), trafo_mat)
    if n == 1:
        return res.flatten()
    else:
        return res


def t2ocx(tf, ts, be=np):
    """See Bouligand 2016, below eq. (10)"""
    c = 0.5 * (1 / tf + 1 / ts)
    x = 0.5 * (1 / tf - 1 / ts)
    # o = np.sqrt(c**2 - x**2)
    o = 1 / be.sqrt(tf * ts)
    return o, c, x


def _fix_z(z):
    """Hotfix for single inputs."""
    try:
        z.shape[1]
        return z
    except IndexError:
        return np.atleast_2d(z).T


# def dsh_basis(lmax, z, R=REARTH):
#     z = _fix_z(z)
#     return _dsh_basis(lmax, z, R=R)


try:
    from _csedprep import _dspharm

    def dsh_basis(lmax, z, R=REARTH):
        # If you ever encounter a problem with this version, check that your
        # z is fortran order!
        _z = _fix_z(z)[0:3]
        if not np.isfortran(_z):
            _z = np.asfortranarray(_z)
        return _dspharm(int(lmax), _z, R)

except ImportError:
    warn(
        "c++-accelerated version could not be loaded. Falling back to "
        "pymagglobal.",
        UserWarning,
    )
    from pymagglobal.utils import dsh_basis as _dsh_basis

    def dsh_basis(lmax, z, R=REARTH):
        z = _fix_z(z)
        return _dsh_basis(lmax, z, R=R)

finally:
    dsh_basis.__doc__ = """
        Evaluate the magnetic field basis functions (derivatives of spherical
        harmonics).

        Parameters
        ----------
        lmax : int
            The maximum spherical harmonics degree
        z : array
            The points at which to evaluate the basis functions, given as

            * z[0, :] contains the colatitude in degrees.
            * z[1, :] contains the longitude in degrees.
            * z[2, :] contains the radius in kilometers.

            Any additional columns are ignored.
        R : float, optional
            The reference radius in kilometers. Default is REARTH=6371.2 km.

        Returns
        -------
        array
            An array of shape (lmax*(lmax+2), 3*z.shape[1]), containing the
            basis functions.
        """
