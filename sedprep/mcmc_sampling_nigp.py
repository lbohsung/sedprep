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
import pymc as pm
from pymc.sampling import jax as pmj
from pytensor import tensor as pt
from utils import interp, interp_adm, grad_dif
from pymagglobal.utils import lmax2N

from prior import generate_prior
from constants import (
    acc_shape,
    mem_alpha,
    mem_beta,
    tDir,
    tInt,
    alpha_nu,
    beta_nu,
    mcmc_params,
    clip_nigp_D,
    clip_nigp_I,
    clip_nigp_F,
)
from constants import field_params as fp

from data_handling import Data


def mcmc_sampling(
    arch_data,
    sed_data_list,
    t_min=-12000,
    t_max=2000,
    step=50,
    path_results="../results/result.nc",
):
    n_coeffs = lmax2N(fp["lmax"])
    knots = np.arange(t_min, t_max + step, step)
    prior_mean, prior_chol, n_ref, ref_coeffs, prior_chol_d = generate_prior(
        knots, **fp
    )

    arch_data = arch_data[arch_data.dt <= 100]
    arch_data = arch_data[(arch_data.t >= t_min) & (arch_data.t <= t_max)]
    arch_data.reset_index(inplace=True, drop=True)
    arch = Data(arch_data)
    sed_dict = {
        sed_data["core_id"]: Data(sed_data["mag_data"], sed_data["adm_data"])
        for sed_data in sed_data_list
    }
    base_tensor_a = pt.as_tensor(arch.base.transpose(1, 0, 2))

    mcModel = pm.Model()
    with mcModel:
        nus = pm.Gamma("nu", alpha=alpha_nu, beta=beta_nu, size=3)
        # coefficients at model knots
        g_cent = pm.Normal(
            "g_cent", mu=0, sigma=1, size=(n_coeffs, len(knots) - n_ref)
        )
        gs_at = pt.batched_dot(prior_chol, g_cent) + prior_mean
        gs_at_d = pt.batched_dot(prior_chol_d, g_cent)

        gs_at_knots = pm.Deterministic(
            "gs_at_knots", pt.horizontal_stack(gs_at, pt.as_tensor(ref_coeffs))
        )
        gs_at_knots_d = pm.Deterministic(
            "gs_at_knots_d",
            pt.horizontal_stack(gs_at_d, pt.as_tensor(0 * ref_coeffs)),
        )
        # -------------------------------------------------------------------------
        # archeo data
        # t_cent = pm.Normal("t_cent", mu=0, sigma=1, size=arch.n)
        # gs_a = interp(arch.t - t_cent * np.sqrt(arch.dt), knots, gs_at_knots.T)
        gs_a = interp(arch.t, knots, gs_at_knots.T)
        gs_a_d = interp(arch.t, knots, gs_at_knots_d.T)
        nez_tensor_a = pt.batched_dot(gs_a, base_tensor_a)
        nez_tensor_a_d = pt.batched_dot(gs_a_d, base_tensor_a)

        _f_a = pt.sqrt(pt.sum(pt.square(nez_tensor_a), axis=1))
        _i_a = pt.rad2deg(pt.arcsin(nez_tensor_a[:, 2] / _f_a))

        grad_d, grad_i, grad_f = grad_dif(nez_tensor_a)
        _d_a_d = pt.sum(grad_d * nez_tensor_a_d, axis=1)
        _i_a_d = pt.sum(grad_i * nez_tensor_a_d, axis=1)
        _f_a_d = pt.sum(grad_f * nez_tensor_a_d, axis=1)
        nigp_d = pt.clip(
            arch.dt[arch.idx_D] * _d_a_d[arch.idx_D] ** 2,
            0,
            clip_nigp_D**2,
        )
        nigp_i = pt.clip(
            arch.dt[arch.idx_I] * _i_a_d[arch.idx_I] ** 2,
            0,
            clip_nigp_I**2,
        )
        nigp_f = pt.clip(
            arch.dt[arch.idx_F] * _f_a_d[arch.idx_F] ** 2,
            0,
            clip_nigp_F**2,
        )
        # print(np.mean(np.sqrt((arch.dt[arch.idx_F] * _f_a_d[arch.idx_F] ** 2).eval())))
        # print(np.mean(np.sqrt(nigp_f.eval())))
        # print(np.mean(np.sqrt(arch.errs_F)))
        rI_a = pm.Deterministic(
            "rI_a",
            (_i_a[arch.idx_I] - arch.out_I)
            / np.sqrt(arch.errs_I + nigp_i + tDir**2),
        )

        d_a = pt.rad2deg(
            pt.arctan2(
                nez_tensor_a[arch.idx_D, 1], nez_tensor_a[arch.idx_D, 0]
            )
        )
        _rD_a = d_a - arch.out_D
        rD_a = pm.Deterministic(
            "rD_a",
            (_rD_a - 360 * (_rD_a > 180) + 360 * (-180 > _rD_a))
            / pt.sqrt(
                arch.errs_D
                + nigp_d
                + pt.clip(tDir / pt.cos(pt.deg2rad(_i_a[arch.idx_D])), -30, 30)
                ** 2
            ),
        )

        rF_a = pm.Deterministic(
            "rF_a",
            (_f_a[arch.idx_F] - arch.out_F)
            / np.sqrt(arch.errs_F + nigp_f + tInt**2),
        )

        rD_obs_a = pm.StudentT(
            "d_obs_a",
            nu=1 + nus[0],
            mu=rD_a,
            sigma=1.0,
            observed=np.zeros(len(arch.idx_D)),
        )

        rI_obs_a = pm.StudentT(
            "i_obs_a",
            nu=1 + nus[1],
            mu=rI_a,
            sigma=1.0,
            observed=np.zeros(len(arch.idx_I)),
        )

        rF_obs_a = pm.StudentT(
            "f_obs_a",
            nu=1 + nus[2],
            mu=rF_a,
            sigma=1.0,
            observed=np.zeros(len(arch.idx_F)),
        )
        # -------------------------------------------------------------------------
        # sediment data - age depth model
        # for name, sed in sed_dict.items():
        #     lvls_dict = sed.levels_dict
        #     n_lvls = lvls_dict["n_levels"]
        #     K_fine = lvls_dict[f"{n_lvls}"]["nK"]
        #     acc_shape_adj = (
        #         acc_shape * (n_lvls - 1) / lvls_dict["multi_parent_adj"]
        #     )
        #     age0_cent = pm.HalfFlat(f"age0_cent_{name}", size=(1,))
        #     age0 = pm.Deterministic(f"age0_{name}", sed.age0_bound + age0_cent)
        #     # age0 = [sed.age0_bound]
        #     alpha_0 = pm.Gamma(
        #         f"alphas_level_1_{name}",
        #         alpha=acc_shape_adj,
        #         beta=acc_shape_adj / sed.acc_mean,
        #         size=(1,),
        #     )
        #     _alpha_list = [alpha_0]
        #     for it in range(2, n_lvls + 1):
        #         lvl_data = lvls_dict[f"{it}"]
        #         parent_mean = (
        #             lvl_data["wts1"] * _alpha_list[it - 2][lvl_data["parent1"]]
        #             + lvl_data["wts2"]
        #             * _alpha_list[it - 2][lvl_data["parent2"]]
        #         )
        #         alpha = pm.Gamma(
        #             f"alphas_level_{it}_{name}",
        #             alpha=acc_shape_adj,
        #             beta=acc_shape_adj / parent_mean,
        #             size=lvl_data["nK"],
        #         )
        #         _alpha_list.append(alpha)

        # R = pm.Beta(f"R_{name}", alpha=mem_alpha, beta=mem_beta, size=1)
        # w = R**sed.delta_c
        # _x = pm.math.concatenate(([_alpha_list[-1][0]], _alpha_list[-1]))

        # idx = np.arange(K_fine)
        # x = pm.Deterministic(f"x_{name}", w * _x[idx] + (1 - w) * _x[idx + 1])
        # c_ages = pm.Deterministic(
        #     f"c_ages_{name}",
        #     pm.math.concatenate(
        #         (pt.as_tensor(age0), age0 + pt.cumsum(x) * sed.delta_c),
        #     ),
        # )
        # mod_age = c_ages[sed.which_c] + x[sed.which_c] * (
        #     sed.adm_depth - sed.c_depth_top[sed.which_c]
        # )
        # rAge = (mod_age - sed.adm_t) / sed.adm_dt

        # obs_age = pm.StudentT(
        #     f"obs_age_{name}",
        #     nu=6,
        #     mu=rAge,
        #     sigma=1.0,
        #     observed=np.zeros(len(sed.adm_t)),
        # )
        # # -------------------------------------------------------------------------
        # # sediment data - integration
        # # translate knots to depth
        # # NOTE the hack with the minus sign, this is because for interp1d
        # # the x points have to be ascending
        # # this also makes applying the ordering constraint easier
        # knots_d = interp_adm(
        #     -knots, -(1950 - c_ages), sed.modelled_depths, alpha_0
        # )
        # # get weights
        # a_s = pm.Uniform(
        #     f"lock_in_{name}",
        #     lower=1e-6,
        #     upper=100.0,
        #     size=4,
        #     initval=np.random.uniform(low=1e-6, high=5, size=4),
        # )

        # b_s = pt.cumsum(a_s, axis=0)

        # width_bound = pm.math.log(
        #     pm.math.sigmoid(0.3 * (b_s[0] - b_s[3] + 50))
        # )
        # pm.Potential(f"width_bound_{name}", width_bound)

        # d_io = sed.depth[None, :] - knots_d[:, None]

        # beta = 2 / (b_s[3] + b_s[2] - b_s[1] - b_s[0])

        # # F_01 = 0.
        # F_12 = beta * (d_io - b_s[0]) ** 2 / 2 / (b_s[1] - b_s[0])
        # F_23 = beta * (d_io - (b_s[1] + b_s[0]) / 2)
        # F_34 = beta * (
        #     -(d_io**2 / 2 - b_s[3] * d_io + b_s[2] ** 2 / 2)
        #     / (b_s[3] - b_s[2])
        #     - (b_s[1] + b_s[0]) / 2
        # )
        # F_4x = 1

        # # ind_01 = d_io <= b_s[0]
        # ind_12 = (b_s[0] < d_io) * (d_io <= b_s[1])
        # ind_23 = (b_s[1] < d_io) * (d_io <= b_s[2])
        # ind_34 = (b_s[2] < d_io) * (d_io <= b_s[3])
        # ind_4x = b_s[3] < d_io

        # ints = ind_12 * F_12 + ind_23 * F_23 + ind_34 * F_34 + ind_4x * F_4x

        # weights = ints[1:] - ints[:-1]
        # # Normalize, i.e. increase the weight of the present day field for
        # # sediments that see "the future" due to incomplete lockin
        # # XXX there might be a more efficient way to do this
        # today_comp = np.zeros(len(knots) - 1)
        # today_comp[-1] = 1
        # weights += (1 - weights.sum(axis=0))[None, :] * today_comp[:, None]

        # # integrate (Riemann sum)
        # gs_s = (gs_at_knots[:, :-1] @ weights).T

        # nez_tensor_s = gs_s @ pt.as_tensor(sed.base)

        # _h_s = pt.sqrt(pt.sum(pt.square(nez_tensor_s[:, 0:2]), axis=1))

        # if 0 < len(sed.idx_I):
        #     f_shallow = pm.TruncatedNormal(
        #         f"f_shallow_{name}", mu=1.0, sigma=0.6, lower=0.0, upper=1.0
        #     )
        #     _i_s_I = pt.rad2deg(
        #         pt.arctan(
        #             f_shallow * nez_tensor_s[sed.idx_I, 2] / _h_s[sed.idx_I]
        #         ),
        #     )
        #     rI_s = pm.Deterministic(
        #         f"rI_sed_{name}",
        #         (_i_s_I - sed.out_I) / np.sqrt(sed.errs_I + tDir**2),
        #     )
        #     rI_obs_s = pm.StudentT(
        #         f"i_obs_sed_{name}",
        #         nu=4,
        #         mu=rI_s,
        #         sigma=1.0,
        #         observed=np.zeros(len(sed.idx_I)),
        #     )

        # if 0 < len(sed.idx_D):
        #     d_s = pt.rad2deg(
        #         pt.arctan2(
        #             nez_tensor_s[sed.idx_D, 1], nez_tensor_s[sed.idx_D, 0]
        #         )
        #     )
        #     for sc_name, (arr_o, mu_o) in sed.subcores.items():
        #         _offset = pm.Normal(f"offset_{sc_name}", mu=mu_o, sigma=90)
        #         d_s += _offset * arr_o

        #     _i_s_D = pt.rad2deg(
        #         pt.arctan2(nez_tensor_s[sed.idx_D, 2], _h_s[sed.idx_D])
        #     )
        #     _rD_s = d_s[sed.idx_D] - sed.out_D
        #     rD_s = pm.Deterministic(
        #         f"rD_sed_{name}",
        #         (_rD_s - 360 * (_rD_s > 180) + 360 * (-180 > _rD_s))
        #         / pt.sqrt(
        #             sed.errs_D
        #             + pt.clip(tDir / pt.cos(pt.deg2rad(_i_s_D)), -30, 30) ** 2
        #         ),
        #     )
        #     rD_obs_s = pm.StudentT(
        #         f"d_obs_sed_{name}",
        #         nu=4,
        #         mu=rD_s,
        #         sigma=1.0,
        #         observed=np.zeros(len(sed.idx_D)),
        #     )

        # if 0 < len(sed.idx_F):
        #     _f_s = pt.sqrt(pt.sum(pt.square(nez_tensor_s[sed.idx_F]), axis=1))
        #     F_mean = sed.out_F.mean()
        #     f_calib = pm.TruncatedNormal(
        #         f"f_calib_{name}",
        #         # See Merril 1998, Eq. 3.4.5
        #         mu=np.abs(fp["gamma"])
        #         * np.sqrt(1 + 3 * np.cos(sed.colat) ** 2)
        #         / F_mean,
        #         sigma=20.0 / F_mean,
        #         lower=1e-6,
        #     )
        #     rF_s = pm.Deterministic(
        #         f"rF_sed_{name}",
        #         (_f_s / f_calib - sed.out_F)
        #         / pt.sqrt(sed.errs_F + (tInt / f_calib) ** 2),
        #     )
        #     rF_obs_s = pm.StudentT(
        #         f"f_obs_sed_{name}",
        #         nu=4,
        #         mu=rF_s,
        #         sigma=1.0,
        #         observed=np.zeros(len(sed.idx_F)),
        #     )

    with mcModel:
        idata = pmj.sample_numpyro_nuts(
            mcmc_params["n_samps"],
            tune=mcmc_params["n_warmup"],
            progressbar=True,
            # cores=1,
            chains=mcmc_params["n_chains"],
            target_accept=mcmc_params["target_accept"],
            postprocessing_backend="cpu",
            nuts_kwargs={
                # 'dense_mass': [('betas'), ('NEZ_residual'), ('t_cent')],
                # 'init_strategy': init_to_median,
            },
        )
    idata.to_netcdf(f"{path_results}")
