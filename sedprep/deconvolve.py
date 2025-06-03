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
import pandas as pd
from tqdm import tqdm

from pymagglobal.utils import scaling, lmax2N, i2lm_l, grad_d, grad_i, grad_f

from data_handling import t2d, I_dip
from utils import lif, dsh_basis, t2ocx, nez2dif
from constants import REARTH
from constants import field_params as fp


def poe(inp):
    """Point of expansion used for deconvolution"""
    poe = [fp["gamma"], 0, 0]
    poe_lmax = i2lm_l(len(poe) - 1)
    poe_field = dsh_basis(poe_lmax, inp)
    return poe @ poe_field


def get_cov(dd, alphas, taus):
    return alphas * (1 + dd / taus) * np.exp(-dd / taus)


def deconvolve(data, bs_DI=None, bs_F=None):
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
    """

    if (bs_DI is None) & (bs_F is None):
        raise ValueError("bs_DI and bs_F can not be None at the same time")
    lmax = fp["lmax"]
    n_coeffs = lmax2N(lmax)
    _t2d = t2d(data)

    if "omega_dip" in fp.keys():
        alphas = fp["alpha_wodip"]**2 * np.ones(n_coeffs)
        alphas[0:3] = fp["alpha_dip"]**2
        alphas *= scaling(fp["R"], REARTH, lmax) ** 2

        omega_dip_d = 1 / np.mean(_t2d(data.t - 1 / omega_dip) - _t2d(data.t))
        # chi_dip_d = np.sqrt(fp["xi_dip"]**2 + omega_dip_d**2)
        tau_wodip_d = np.mean(_t2d(data.t - fp["tau_wodip"]) - _t2d(data.t))
        taus = np.array(
            [tau_wodip_d / float(i2lm_l(i)) for i in range(n_coeffs)],
        )
        if fp["axial"]:
            taus[0] = 1 / omega_dip_d
            taus[1:3] = 1 / 2 * tau_wodip_d
        else:
            taus[0:3] = 1 / omega_dip_d

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
    depths = np.array(data.depth)
    i_indices, j_indices = np.triu_indices(n_obs)

    if "dt" in data.columns:
        _dtBB = base.T * (alphas / taus**2) @ base
        d_llim = _t2d(data["t"] + data["dt"])
        d_ulim = _t2d(data["t"] - data["dt"])
        d = _t2d(data["t"])
        d_uerr = abs(d - d_ulim)
        d_lerr = abs(d - d_llim)
        d_err = np.maximum(d_lerr, d_uerr)

    poe_field = poe(inp)
    grad_D = grad_d(*poe_field.reshape(1, 3).T)
    grad_I = grad_i(*poe_field.reshape(1, 3).T)

    max_lid_DI = (
        np.sum(bs_DI) if len(bs_DI) == 4 else 2 * bs_DI[1] + bs_DI[0]
    )
    lif_DI_ant = lif(bs_DI).antiderivative()
    n_int_DI = max(10, int((max_lid_DI - bs_DI[0])))
    int_width_DI = (max_lid_DI - bs_DI[0]) / n_int_DI

    z_DI = bs_DI[0] + np.arange(n_int_DI + 1) * int_width_DI
    lif_DI_vals = lif_DI_ant(z_DI[1:]) - lif_DI_ant(z_DI[:-1])
    # Double integration of lock-in function for each interval
    liflif_DI_vals = np.outer(lif_DI_vals, lif_DI_vals)

    # depths = depths.astype(np.float32)
    # z_DI = z_DI.astype(np.float32)
    # lif_DI_vals = lif_DI_vals.astype(np.float32)
    # liflif_DI_vals = liflif_DI_vals.astype(np.float32)
    # alphas = alphas.astype(np.float32)
    # taus = taus.astype(np.float32)
    # base = base.astype(np.float32)
    # grad_D = grad_D.astype(np.float32)
    # grad_I = grad_I.astype(np.float32)

    K_x_D, K_x_I, K_y_D, K_y_I = [np.zeros((n_obs, n_obs)) for _ in range(4)]

    # chunk_size = 1000
    # for i_start in range(0, n_obs, chunk_size):
    #     i_end = min(i_start + chunk_size, n_obs)
    #     for j_start in range(0, n_obs, chunk_size):
    #         j_end = min(j_start + chunk_size, n_obs)
    #         # Compute indices for the block
    #         block_i_indices = np.arange(i_start, i_end)
    #         block_j_indices = np.arange(j_start, j_end)
    #         dd_block = np.abs(depths[block_i_indices][:, None] - depths[block_j_indices][None, :])
    #         cov_block = get_cov(dd_block[..., None], alphas[None, :], taus[None, :])
    #         NEZ_cov_block = np.einsum("ik,kp,kq->ipq", cov_block, base, base)
    #         K_x_D_block = np.einsum("p,ipq,q->i", grad_D, NEZ_cov_block, grad_D)
    #         # Assign the block to the full matrix
    #         K_x_D[block_i_indices[:, None], block_j_indices] = K_x_D_block.reshape(len(block_i_indices), len(block_j_indices))

    dd = np.abs(depths[i_indices] - depths[j_indices])
    cov = get_cov(dd[..., None], alphas[None, :], taus[None, :])
    NEZ_cov = np.einsum("ik,kp,kq->ipq", cov, base, base)
    K_x_D_upper = np.einsum("p,ipq,q->i", grad_D, NEZ_cov, grad_D)
    K_x_I_upper = np.einsum("p,ipq,q->i", grad_I, NEZ_cov, grad_I)
    K_x_D[i_indices, j_indices] = K_x_D_upper
    K_x_I[i_indices, j_indices] = K_x_I_upper
    K_x_D[j_indices, i_indices] = K_x_D_upper
    K_x_I[j_indices, i_indices] = K_x_I_upper
    print(K_x_D.shape)

    i_indices, j_indices = np.triu_indices(n_obs)
    delta_jm = depths[j_indices][:, None] - z_DI[:-1][None, :]
    delta_in = depths[i_indices][:, None] - z_DI[:-1][None, :]
    dd_mn = np.abs(delta_jm[:, :, None] - delta_in[:, None, :])
    cov_mn = get_cov(
        dd_mn[..., None],
        alphas[None, None, None, :],
        taus[None, None, None, :],
    )
    NEZ_cov_mn = np.einsum("imnk,ka,kb->imnab", cov_mn, base, base)
    liflif_DI_vals_expanded = liflif_DI_vals[None, :, :, None, None]
    NEZ_cov_mn_weighted = NEZ_cov_mn * liflif_DI_vals_expanded
    NEZ_cov_sum = NEZ_cov_mn_weighted.sum(axis=(1, 2))
    K_y_D_upper = np.einsum("p,ipq,q->i", grad_D, NEZ_cov_sum, grad_D)
    K_y_I_upper = np.einsum("p,ipq,q->i", grad_I, NEZ_cov_sum, grad_I)
    K_y_D[i_indices, j_indices] = K_y_D_upper
    K_y_I[i_indices, j_indices] = K_y_I_upper
    K_y_D[j_indices, i_indices] = K_y_D_upper
    K_y_I[j_indices, i_indices] = K_y_I_upper

    # K_xy approximation
    dd_m = np.abs(
        (depths[:, None] - z_DI[:-1][None, :])[None, :, :]
        - depths[:, None, None]
    )
    cov_m = get_cov(
        dd_m[..., None],
        alphas[None, None, None, :],
        taus[None, None, None, :],
    )
    NEZ_cov_m = np.einsum("ijmk,ka,kb->ijmab", cov_m, base, base)
    lif_DI_vals_expanded = lif_DI_vals[None, None, :, None, None]
    NEZ_cov_weighted = NEZ_cov_m * lif_DI_vals_expanded
    NEZ_cov_sum = NEZ_cov_weighted.sum(axis=2)
    K_xy_D = np.einsum("a,ijab,b->ij", grad_D, NEZ_cov_sum, grad_D)
    K_xy_I = np.einsum("a,ijab,b->ij", grad_I, NEZ_cov_sum, grad_I)

    if "dt" in data.columns:
        nigp_corr_D = grad_D @ _dtBB @ grad_D.T
        nigp_corr_I = grad_I @ _dtBB @ grad_I.T
        inv_D = np.linalg.inv(
            K_y_D + np.diag(data.dD**2) + np.diag(nigp_corr_D * d_err)
        )
        inv_I = np.linalg.inv(
            K_y_I + np.diag(data.dI**2) + np.diag(nigp_corr_I * d_err)
        )
    else:
        inv_D = np.linalg.inv(K_y_D + np.diag(data.dD**2))
        inv_I = np.linalg.inv(K_y_I + np.diag(data.dI**2))
    mean_D = K_xy_D @ inv_D @ data.D
    mean_I = I_dip(data.lat[0]) + K_xy_I @ inv_I @ (
        data.I - I_dip(data.lat[0])
    )
    cov_D = K_x_D - K_xy_D @ inv_D @ K_xy_D.T
    cov_I = K_x_I - K_xy_I @ inv_I @ K_xy_I.T
    return mean_D, cov_D, mean_I, cov_I


# core_id = "P2"
# sed_data = pd.read_csv(f"../dat/sed_data/{core_id}_prepared.csv")
# sed_data["F"] = np.nan
# sed_data["dF"] = np.nan
# sed_data = sed_data[:24]
# mean_D, cov_D, mean_I, cov_I = deconvolve(sed_data, [5, 1, 2, 1])
# print(mean_D)