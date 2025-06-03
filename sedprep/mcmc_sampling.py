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
from lock_in_function import lock_in_function
from data_handling import Data
from adm import BaconAdm as AgeDepthModel


def mcmc_sampling(
    arch_data,
    sed_data_list,
    t_min=-12000,
    t_max=2000,
    step=50,
    lif_type="4p",
    path_results="../results/result.nc",
):
    n_coeffs = lmax2N(fp["lmax"])
    knots = np.arange(t_min, t_max + step, step)
    prior_mean, prior_chol, n_ref, ref_coeffs, cov_d = generate_prior(
        knots, **fp
    )

    arch_data = arch_data[arch_data.dt <= 200]
    arch_data = arch_data[(arch_data.t >= t_min) & (arch_data.t <= t_max)]
    arch_data.reset_index(inplace=True, drop=True)
    arch = Data(arch_data)
    sed_dict = {
        sed_data["core_id"]: Data(sed_data["mag_data"], sed_data["adm_data"])
        for sed_data in sed_data_list
    }
    base_tensor_a = pt.as_tensor(arch.base.transpose(1, 0, 2))
    ref_coeffs_tensor = pt.as_tensor(ref_coeffs)
    _dtBB_p = np.einsum(
        "ij, j, jk -> ik", arch.base[:, 0, :].T, cov_d, arch.base[:, 0, :]
    )

    mcModel = pm.Model()
    with mcModel:
        nus = pm.Gamma("nu", alpha=alpha_nu, beta=beta_nu, size=3)
        # coefficients at model knots
        g_cent = pm.Normal(
            "g_cent", mu=0, sigma=1, size=(n_coeffs, len(knots) - n_ref)
        )
        gs_at = pt.batched_dot(prior_chol, g_cent) + prior_mean

        gs_at_knots = pm.Deterministic(
            "gs_at_knots", pt.horizontal_stack(gs_at, ref_coeffs_tensor)
        )
        # -------------------------------------------------------------------------
        # archeo data
        # t_cent = pm.Normal("t_cent", mu=0, sigma=1, size=arch.n)
        # gs_a = interp(arch.t - t_cent * np.sqrt(arch.dt), knots, gs_at_knots.T)
        gs_a = interp(arch.t, knots, gs_at_knots.T)
        nez_tensor_a = pt.batched_dot(gs_a, base_tensor_a)

        _f_a = pt.sqrt(pt.sum(pt.square(nez_tensor_a), axis=1))
        _i_a = pt.rad2deg(pt.arcsin(nez_tensor_a[:, 2] / _f_a))

        grad_d, grad_i, grad_f = grad_dif(nez_tensor_a)
        # Equivalent to pt.diagonal((grad_d @ _dtBB_p) @ grad_d.T))
        _d_a_d = pt.sum((grad_d @ _dtBB_p) * grad_d, axis=1)
        _i_a_d = pt.sum((grad_i @ _dtBB_p) * grad_i, axis=1)
        _f_a_d = pt.sum((grad_f @ _dtBB_p) * grad_f, axis=1)

        nigp_d = pt.clip(
            arch.dt[arch.idx_D] * _d_a_d[arch.idx_D], 0, clip_nigp_D**2
        )
        nigp_i = pt.clip(
            arch.dt[arch.idx_I] * _i_a_d[arch.idx_I], 0, clip_nigp_I**2
        )
        nigp_f = pt.clip(
            arch.dt[arch.idx_F] * _f_a_d[arch.idx_F], 0, clip_nigp_F**2
        )
        rI_a = pm.Deterministic(
            "rI_a",
            (_i_a[arch.idx_I] - arch.out_I)
            # / pt.sqrt(arch.errs_I + tDir**2),
            / pt.sqrt(arch.errs_I + nigp_i + tDir**2),
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
            # / pt.sqrt(arch.errs_F + tInt**2),
            / pt.sqrt(arch.errs_F + nigp_f + tInt**2),
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
        for name, sed in sed_dict.items():
            adm = AgeDepthModel(name, sed)
            mod_age = adm.get_ages(sed.adm_data["depth"].values)
            rAge = (mod_age - (1950 - sed.adm_data["t"])) / sed.adm_data["dt"]
            obs_age = pm.Normal(
                f"obs_age_{name}",
                # nu=6,
                mu=rAge,
                sigma=1.0,
                observed=np.zeros(len(sed.adm_data)),
            )
            # -------------------------------------------------------------------------
            # sediment data - integration
            knots_d = adm.get_depths(knots)

            # get weights
            # a_s = pm.Uniform(
            #     f"lock_in_{name}",
            #     lower=1e-6,
            #     upper=100.0,
            #     size=4,
            #     initval=np.random.uniform(low=1e-6, high=5, size=4),
            # )
            # b_s = pt.cumsum(a_s, axis=0)
            b_s = lock_in_function(lif_type)(name)

            width = b_s[3] - b_s[0]
            width_bound = pm.math.log(pm.math.sigmoid(-0.3 * (width - 50)))
            pm.Potential(f"width_bound_{name}", width_bound)

            d_io = sed.depth[None, :] - knots_d[:, None]

            beta = 2 / (b_s[3] + b_s[2] - b_s[1] - b_s[0])

            F_12 = beta * (d_io - b_s[0]) ** 2 / 2 / (b_s[1] - b_s[0])
            F_23 = beta * (d_io - (b_s[1] + b_s[0]) / 2)
            F_34 = beta * (
                -(d_io**2 / 2 - b_s[3] * d_io + b_s[2] ** 2 / 2)
                / (b_s[3] - b_s[2])
                - (b_s[1] + b_s[0]) / 2
            )
            F_4x = 1

            ind_12 = (b_s[0] < d_io) * (d_io <= b_s[1])
            ind_23 = (b_s[1] < d_io) * (d_io <= b_s[2])
            ind_34 = (b_s[2] < d_io) * (d_io <= b_s[3])
            ind_4x = b_s[3] < d_io

            ints = (
                ind_12 * F_12 + ind_23 * F_23 + ind_34 * F_34 + ind_4x * F_4x
            )

            weights = ints[1:] - ints[:-1]
            # Normalize, i.e. increase the weight of the present day field for
            # sediments that see "the future" due to incomplete lockin
            # XXX there might be a more efficient way to do this
            today_comp = np.zeros(len(knots) - 1)
            today_comp[-1] = 1
            weights += (1 - weights.sum(axis=0))[None, :] * today_comp[:, None]

            weight_bound = pm.math.sum(
                pm.math.log(pm.math.sigmoid(30 * (0.7 - weights[-1])))
            )
            pm.Potential(f"weight_bound_{name}", weight_bound)

            # integrate (Riemann sum)
            gs_s = (gs_at_knots[:, :-1] @ weights).T

            nez_tensor_s = gs_s @ pt.as_tensor(sed.base)

            _h_s = pt.sqrt(pt.sum(pt.square(nez_tensor_s[:, 0:2]), axis=1))

            if 0 < len(sed.idx_I):
                f_shallow = pm.TruncatedNormal(
                    f"f_shallow_{name}",
                    mu=1.0,
                    sigma=0.4,
                    lower=0.001,
                    upper=1.0,
                )
                _i_s_I = pt.rad2deg(
                    pt.arctan(
                        f_shallow
                        * nez_tensor_s[sed.idx_I, 2]
                        / _h_s[sed.idx_I]
                    ),
                )
                rI_s = pm.Deterministic(
                    f"rI_sed_{name}",
                    (_i_s_I - sed.out_I) / pt.sqrt(sed.errs_I + tDir**2),
                )
                rI_obs_s = pm.StudentT(
                    f"i_obs_sed_{name}",
                    nu=4,
                    mu=rI_s,
                    sigma=1.0,
                    observed=np.zeros(len(sed.idx_I)),
                )

            if 0 < len(sed.idx_D):
                d_s = pt.rad2deg(
                    pt.arctan2(nez_tensor_s[:, 1], nez_tensor_s[:, 0])
                )
                for sc_name, (arr_o, mu_o) in sed.subcores.items():
                    _offset = pm.Normal(
                        f"offset_{sc_name}",
                        mu=mu_o,
                        sigma=30,
                        initval=np.random.normal(mu_o, 2),
                    )
                    d_s += _offset * arr_o

                _i_s_D = pt.rad2deg(
                    pt.arctan2(nez_tensor_s[sed.idx_D, 2], _h_s[sed.idx_D])
                )
                pm.Deterministic(f"d_s_{name}", d_s)
                _rD_s = d_s[sed.idx_D] - sed.out_D
                rD_s = pm.Deterministic(
                    f"rD_sed_{name}",
                    (_rD_s - 360 * (_rD_s > 180) + 360 * (-180 > _rD_s))
                    / pt.sqrt(
                        sed.errs_D
                        + pt.clip(tDir / pt.cos(pt.deg2rad(_i_s_D)), -30, 30)
                        ** 2
                    ),
                )
                rD_obs_s = pm.StudentT(
                    f"d_obs_sed_{name}",
                    nu=4,
                    mu=rD_s,
                    sigma=1.0,
                    observed=np.zeros(len(sed.idx_D)),
                )

            if 0 < len(sed.idx_F):
                _f_s = pt.sqrt(
                    pt.sum(pt.square(nez_tensor_s[sed.idx_F]), axis=1)
                )
                F_mean = sed.out_F.mean()
                f_calib = pm.TruncatedNormal(
                    f"f_calib_{name}",
                    # See Merril 1998, Eq. 3.4.5
                    mu=np.abs(fp["gamma"])
                    * np.sqrt(1 + 3 * np.cos(sed.colat) ** 2)
                    / F_mean,
                    sigma=20.0 / F_mean,
                    lower=1e-6,
                )
                rF_s = pm.Deterministic(
                    f"rF_sed_{name}",
                    (_f_s / f_calib - sed.out_F)
                    / pt.sqrt(sed.errs_F + (tInt / f_calib) ** 2),
                )
                rF_obs_s = pm.StudentT(
                    f"f_obs_sed_{name}",
                    nu=4,
                    mu=rF_s,
                    sigma=1.0,
                    observed=np.zeros(len(sed.idx_F)),
                )

    with mcModel:
        idata = pmj.sample_numpyro_nuts(
            mcmc_params["n_samps"],
            tune=mcmc_params["n_warmup"],
            progressbar=True,
            chains=mcmc_params["n_chains"],
            target_accept=mcmc_params["target_accept"],
            postprocessing_backend="cpu",
        )
        idata.observed_data["knots"] = knots
    idata.to_netcdf(f"{path_results}")
