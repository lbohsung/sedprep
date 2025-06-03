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
from tqdm import tqdm

from pymagglobal.utils import scaling, lmax2N, i2lm_l, grad_d, grad_i, grad_f

from sedprep import data_handling
from sedprep.utils import lif, dsh_basis, t2ocx, nez2dif
from sedprep.constants import REARTH
from sedprep.constants import field_params as fp
import torch


def poe(inp):
    """Point of expansion used for deconvolution"""
    poe = [fp["gamma"], 0, 0]
    poe_lmax = i2lm_l(len(poe) - 1)
    poe_field = dsh_basis(poe_lmax, inp)
    return poe @ poe_field


def NEZ_kernel(d1, d2, alphas, taus, base, derivative=False):
    """Covariance function used for deconvolution"""
    dd = np.abs(d1 - d2)
    cov = alphas * (1 + dd / taus) * np.exp(-dd / taus)
    if derivative:
        cov = alphas * 1 / taus**2
    return base.T * cov @ base


def deconvolve(data, bs_DI=None, bs_F=None, quiet=True):
    """
    Performs a Gaussian Process deconvolution for inclination and declination
    using the lock-in functions characterized by the parameters `bs`.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data containing declination (D), inclination (I),
        and associated uncertainties (dD and dI).
    bs_DI : numpy.ndarray or list
        numpy.ndarras of shape (4,) or list of length 4 containing
        the four parameters that characterize the lock-in function
        for the directions.
    bs_F : numpy.ndarray or list
        numpy.ndarras of shape (4,) or list of length 4 containing
        the four parameters that characterize the lock-in function
        for the intensities.
    quiet : bool, optional
        If True, disables progress bar. Default is True.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - mean_D: Estimated mean of the deconvolved declination.
        - cov_D: Estimated covariance of the deconvolved declination.
        - mean_I: Estimated mean of the deconvolved inclination.
        - cov_I: Estimated covariance of the deconvolved inclination.
        - mean_F: Estimated mean of the deconvolved intensity.
        - cov_F: Estimated covariance of the deconvolved intensity.

    Example
    -------
    >>> import pandas as pd
    >>> data = pd.DataFrame({"depth": [34.0, 29.2, 25.0],
    ...                      "t": [1628.901, 1679.669, 1724.09],
    ...                      "dt": [300, 300, 300],
    ...                      "lat": 60.151,
    ...                      "lon": 13.055,
    ...                      "D": [-10.067, -37.988, -33.412],
    ...                      "I": [66.016, 66.090, 59.699],
    ...                      "dD": [1.791, 5.289, 2.161],
    ...                      "dI": [0.478, 2.075, 0.478],
    ...                      "type": "sediments"})
    >>> bs = np.array([10, 10, 0, 10])
    >>> mean_D, cov_D, mean_I, cov_I = deconvolve(data, bs_DI=bs)
    >>> print(mean_D)
    [-4.09211145 -6.00544398 -8.47237724]
    >>> print(cov_D)
    [[425.68123873 349.72930172 259.23230587]
     [349.72930172 317.84582714 251.89248699]
     [259.23230587 251.89248699 220.63552318]]
    >>> print(mean_I)
    [70.08638594 69.29138313 68.44060611]
    >>> print(cov_I)
    [[37.41646712 30.73750016 22.78189817]
     [30.73750016 27.93500644 22.13987968]
     [22.78189817 22.13987968 19.39734344]]
    """

    if (bs_DI is None) & (bs_F is None):
        raise ValueError("bs_DI and bs_F can not be None at the same time")
    for key, val in fp.items():
        if torch.is_tensor(val):
            fp[key] = fp[key].detach().cpu().numpy()
    lmax = fp["lmax"]
    n_coeffs = lmax2N(lmax)
    _t2d = data_handling.t2d(data)
    if "tau_dip_slow" in fp.keys():
        omega_dip, _, _ = t2ocx(
            (
                fp["tau_dip"]
                if isinstance(fp["tau_dip"], float)
                else fp["tau_dip"][0]
            ),
            fp["tau_dip_slow"],
            be=np,
        )
        omega_dip_d = 1 / np.mean(_t2d(data.t - 1 / omega_dip) - _t2d(data.t))

    if isinstance(fp["tau_dip"], float):
        alphas = fp["alpha_wodip"] ** 2 * np.ones(n_coeffs)
        alphas[0:3] = fp["alpha_dip"] ** 2
        alphas *= scaling(fp["R"], REARTH, lmax) ** 2

        tau_dip_d = np.mean(_t2d(data.t - fp["tau_dip"]) - _t2d(data.t))
        tau_wodip_d = np.mean(_t2d(data.t - fp["tau_wodip"]) - _t2d(data.t))
        taus = np.array(
            [tau_wodip_d / float(i2lm_l(i)) for i in range(n_coeffs)],
        )
        if fp["axial"]:
            taus[0] = (
                tau_dip_d
                if "tau_dip_slow" not in fp.keys()
                else 1 / omega_dip_d
            )
            taus[1:3] = 1 / 2 * tau_wodip_d
        else:
            taus[0:3] = (
                tau_dip_d
                if "tau_dip_slow" not in fp.keys()
                else 1 / omega_dip_d
            )
    else:
        if fp["axial"]:
            alphas = np.hstack([fp["alpha_dip"], fp["alpha_wodip"]]) ** 2

            tau_dip_d = [
                np.mean(_t2d(data.t - td) - _t2d(data.t))
                for td in fp["tau_dip"]
            ]
            tau_dip_d = [
                np.mean(_t2d(data.t - tw) - _t2d(data.t))
                for tw in fp["tau_wodip"]
            ]
            taus = np.hstack([1 / omega_dip_d, tau_dip_d[1:3], tau_dip_d])
        else:
            print("Not implemented yet")

    inp = np.array([90 - data.lat[0], data.lon[0], REARTH])

    base = dsh_basis(fp["lmax"], inp)

    n_obs = len(data)
    depths = list(data.depth)

    _dtBB = NEZ_kernel(0, 0, alphas, taus, base, derivative=True)

    d_llim = _t2d(data["t"] + data["dt"])
    d_ulim = _t2d(data["t"] - data["dt"])
    d = _t2d(data["t"])
    d_uerr = abs(d - d_ulim)
    d_lerr = abs(d - d_llim)
    d_err = np.maximum(d_lerr, d_uerr)

    poe_field = poe(inp)
    if bs_DI is not None:
        grad_D = grad_d(*poe_field.reshape(1, 3).T)
        grad_I = grad_i(*poe_field.reshape(1, 3).T)

        max_lid_DI = (
            np.sum(bs_DI) if len(bs_DI) == 4 else 2 * bs_DI[1] + bs_DI[0]
        )
        lif_DI_ant = lif(bs_DI).antiderivative()
        n_int_DI = max(10, int((max_lid_DI - bs_DI[0])))
        int_width_DI = (max_lid_DI - bs_DI[0]) / n_int_DI
        K_x_D, K_y_D, K_xy_D, K_x_I, K_y_I, K_xy_I = [
            np.zeros((n_obs, n_obs)) for _ in range(6)
        ]

        z_DI = bs_DI[0] + np.arange(n_int_DI + 1) * int_width_DI
        # Integrate lock-in function for each interval
        lif_DI_vals = np.array(
            [
                (
                    lif_DI_ant(bs_DI[0] + (k + 1) * int_width_DI)
                    - lif_DI_ant(z_DI[k])
                )
                for k in range(n_int_DI + 1)
            ]
        )
        # Double integration of lock-in function for each interval
        liflif_DI_vals = lif_DI_vals[:, None] * lif_DI_vals
        # Approximate integrals using Riemann sums
        for i in tqdm(range(n_obs), disable=quiet):
            for j in range(n_obs):
                if j >= i:
                    # K_x approximation
                    NEZ_cov = NEZ_kernel(
                        depths[j], depths[i], alphas, taus, base
                    )
                    K_x_D[i, j] = grad_D @ NEZ_cov @ grad_D.T
                    K_x_I[i, j] = grad_I @ NEZ_cov @ grad_I.T
                    # K_y approximation
                    sum_y_D, sum_y_I = 0, 0
                    for m in range(n_int_DI):
                        sum_y_inner_D, sum_y_inner_I = 0, 0
                        for n in range(n_int_DI):
                            NEZ_cov = (
                                NEZ_kernel(
                                    depths[j] - z_DI[m],
                                    depths[i] - z_DI[n],
                                    alphas,
                                    taus,
                                    base,
                                )
                                * liflif_DI_vals[m, n]
                            )
                            sum_y_inner_D += grad_D @ NEZ_cov @ grad_D.T
                            sum_y_inner_I += grad_I @ NEZ_cov @ grad_I.T
                        sum_y_D += sum_y_inner_D
                        sum_y_I += sum_y_inner_I
                    K_y_D[i, j] = sum_y_D
                    K_y_I[i, j] = sum_y_I
                # K_xy approximation
                sum_xy_D, sum_xy_I = 0, 0
                for m in range(n_int_DI):
                    NEZ_cov = (
                        NEZ_kernel(
                            depths[j] - z_DI[m], depths[i], alphas, taus, base
                        )
                        * lif_DI_vals[m]
                    )
                    sum_xy_D += grad_D @ NEZ_cov @ grad_D.T
                    sum_xy_I += grad_I @ NEZ_cov @ grad_I.T
                K_xy_D[i, j] = sum_xy_D
                K_xy_I[i, j] = sum_xy_I
        K_y_D = K_y_D + K_y_D.T - np.diag(np.diag(K_y_D))
        K_y_I = K_y_I + K_y_I.T - np.diag(np.diag(K_y_I))
        K_x_D = K_x_D + K_x_D.T - np.diag(np.diag(K_x_D))
        K_x_I = K_x_I + K_x_I.T - np.diag(np.diag(K_x_I))

        nigp_corr_D = grad_D @ _dtBB @ grad_D.T
        nigp_corr_I = grad_I @ _dtBB @ grad_I.T

        inv_D = np.linalg.inv(
            K_y_D + np.diag(data.dD**2) + np.diag(nigp_corr_D * d_err)
        )
        inv_I = np.linalg.inv(
            K_y_I + np.diag(data.dI**2) + np.diag(nigp_corr_I * d_err)
        )
        mean_D = K_xy_D @ inv_D @ data.D
        mean_I = data_handling.I_dip(data.lat[0]) + K_xy_I @ inv_I @ (
            data.I - data_handling.I_dip(data.lat[0])
        )
        cov_D = K_x_D - K_xy_D @ inv_D @ K_xy_D.T
        cov_I = K_x_I - K_xy_I @ inv_I @ K_xy_I.T
        return mean_D, cov_D, mean_I, cov_I

    if bs_F is not None:
        grad_F = grad_f(*poe_field.reshape(1, 3).T)
        max_lid_F = np.sum(bs_F) if len(bs_F) == 4 else 2 * bs_F[1] + bs_F[0]
        lif_F_ant = lif(bs_F).antiderivative()
        n_int_F = max(5, int((max_lid_F - bs_F[0])))
        int_width_F = (max_lid_F - bs_F[0]) / n_int_F
        K_x_F, K_y_F, K_xy_F = [np.zeros((n_obs, n_obs)) for _ in range(3)]

        z_F = bs_F[0] + np.arange(n_int_F + 1) * int_width_F
        # Integrate lock-in function for each interval
        lif_F_vals = np.array(
            [
                (
                    lif_F_ant(bs_F[0] + (k + 1) * int_width_F)
                    - lif_F_ant(z_F[k])
                )
                for k in range(n_int_F + 1)
            ]
        )
        # Double integration of lock-in function for each interval
        liflif_F_vals = lif_F_vals[:, None] * lif_F_vals
        # Approximate integrals using Riemann sums
        for i in tqdm(range(n_obs), disable=quiet):
            for j in range(n_obs):
                if j >= i:
                    # K_x approximation
                    NEZ_cov = NEZ_kernel(
                        depths[j], depths[i], alphas, taus, base
                    )
                    K_x_F[i, j] = grad_F @ NEZ_cov @ grad_F.T
                    # K_y approximation
                    sum_y_F = 0
                    for m in range(n_int_F):
                        sum_y_inner_F = 0
                        for n in range(n_int_F):
                            NEZ_cov = (
                                NEZ_kernel(
                                    depths[j] - z_F[m],
                                    depths[i] - z_F[n],
                                    alphas,
                                    taus,
                                    base,
                                )
                                * liflif_F_vals[m, n]
                            )
                            sum_y_inner_F += grad_F @ NEZ_cov @ grad_F.T
                        sum_y_F += sum_y_inner_F
                    K_y_F[i, j] = sum_y_F
                # K_xy approximation
                sum_xy_F = 0
                for m in range(n_int_F):
                    NEZ_cov = (
                        NEZ_kernel(
                            depths[j] - z_F[m], depths[i], alphas, taus, base
                        )
                        * lif_F_vals[m]
                    )
                    sum_xy_F += grad_F @ NEZ_cov @ grad_F.T
                K_xy_F[i, j] = sum_xy_F
        K_y_F = K_y_F + K_y_F.T - np.diag(np.diag(K_y_F))
        K_x_F = K_x_F + K_x_F.T - np.diag(np.diag(K_x_F))

        nigp_corr_F = grad_F @ _dtBB @ grad_F.T

        inv_F = np.linalg.inv(
            K_y_F + np.diag(data.dF**2) + np.diag(nigp_corr_F * d_err)
        )
        prior_cal_fac = float(nez2dif(*(fp["gamma"] * base[0, :]), be=np)[2])
        mean_F = prior_cal_fac + K_xy_F @ inv_F @ (data.F - prior_cal_fac)
        cov_F = K_x_F - K_xy_F @ inv_F @ K_xy_F.T
        return mean_F, cov_F
