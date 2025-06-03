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
import scipy
import os
import csv
import torch

# import numdifftools as nd
# import sys

from sedprep.kalman_filter import Filter
from sedprep.data_handling import I_dip, chunk_data
from sedprep.constants import field_params, device, dtype
from sedprep.dlib_wrap import dlib_opt
from sedprep.utils import nez2dif


class Optimizer:
    def __init__(
        self,
        sed_data,
        arch_data,
        lif_params=4,
        prior_mean=None,
        prior_cov=None,
        delta_t=40,
        start=2000,
        end=-6000,
        adm_d2t=None,
    ):
        self.lif_params = lif_params
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.delta_t = delta_t
        self.subs = sed_data.subs.unique()
        self.cdat_sed, self.cdat_arch = chunk_data(
            sed_data,
            arch_data,
            lmax=field_params["lmax"],
            delta_t=delta_t,
            start=start,
            end=end,
            adm_d2t=adm_d2t,
        )
        self.prior_mean_offsets = {}
        self.prior_std_offsets = {}
        for sub_name, sub_data in sed_data.groupby("subs"):
            self.prior_mean_offsets[sub_name] = np.mean(sub_data.D)
            self.prior_std_offsets[sub_name] = np.std(sub_data.D)
        prior_f_shallow = np.tan(np.deg2rad(sed_data["I"])) / np.tan(
            np.deg2rad(I_dip(sed_data.lat))
        )
        self.prior_mean_f_shallow = np.clip(np.mean(prior_f_shallow), 0, 1)
        self.prior_std_f_shallow = np.std(prior_f_shallow)
        prior_cal_fac = float(
            nez2dif(*(field_params["gamma"] * self.cdat_sed.base[0, :]))[2]
        )
        self.prior_mean_cal_fac = np.mean(prior_cal_fac / sed_data["F"])
        self.prior_std_cal_fac = np.std(prior_cal_fac / sed_data["F"])

    def compute_log_ml_value(
        self, bs_DI=None, bs_F=None, offsets=None, f_shallow=None, cal_fac=None
    ):
        filt = Filter(
            **field_params,
            bs_DI=bs_DI,
            bs_F=bs_F,
            offsets=offsets,
            f_shallow=f_shallow,
            cal_fac=cal_fac,
        )
        ret = -filt.forward(
            self.cdat_sed,
            self.cdat_arch,
            self.prior_mean,
            self.prior_cov,
            quiet=False,
        )
        res = ret.detach().cpu().numpy()
        print(f"Log-ml value for given parameters is: {res}")
        return res

    def f_opt(
        self,
        x,
        optimize_bs_DI,
        optimize_bs_F,
        optimize_offsets,
        optimize_f_shallow,
        optimize_cal_fac,
        max_lid,
        fixed_bs_DI=None,
        fixed_bs_F=None,
        fixed_offsets=None,
        fixed_f_shallow=None,
        fixed_cal_fac=None,
        grad=False,
        hessian=False,
    ):
        if torch.is_tensor(x):
            inp = x.clone().requires_grad_(True)
        else:
            inp = (
                torch.tensor(np.array(x), device=device, dtype=dtype)
                .clone()
                .detach()
                .requires_grad_(grad)
            )
        bs_DI, bs_F, offsets, f_shallow, cal_fac = self.split_input(
            inp,
            optimize_bs_DI,
            optimize_bs_F,
            optimize_offsets,
            optimize_f_shallow,
            optimize_cal_fac,
            fixed_bs_DI,
            fixed_bs_F,
            fixed_offsets,
            fixed_f_shallow,
            fixed_cal_fac,
        )
        width_prior = 0
        if optimize_bs_DI:
            # 0.3 * (-(bs4 - bs1) + 40)
            # width_prior = -torch.log(torch.sigmoid(0.3 * (bs_DI[0] - torch.sum(bs_DI) + 40)))
            s_DI = (
                sum(bs_DI)
                if self.lif_params == 4
                else sum([bs_DI[0], bs_DI[1] / 2, bs_DI[1], bs_DI[1] / 2])
            )
            width_prior += -(torch.log(torch.sigmoid(0.2 * (bs_DI[0] - s_DI + 50))) * 30)
        if optimize_bs_F:
            s_F = (
                sum(bs_F)
                if optimize_bs_F and self.lif_params == 4
                else sum([bs_F[0], bs_F[1] / 2, bs_F[1], bs_F[1] / 2])
            )
            width_prior += -(torch.log(torch.sigmoid(0.2 * (bs_F[0] - s_F + 50))) * 30)
        if not grad:
            if optimize_bs_DI:
                if max_lid < s_DI:
                    return 1e16
            if optimize_bs_F:
                if max_lid < s_F:
                    return 1e16
        filt = Filter(
            **field_params,
            bs_DI=bs_DI,
            bs_F=bs_F,
            offsets=offsets,
            f_shallow=f_shallow,
            cal_fac=cal_fac,
        )
        ret = -filt.forward(
            self.cdat_sed, self.cdat_arch, self.prior_mean, self.prior_cov
        )
        # if optimize_bs_DI:
        #     ret += -torch.log(torch.sigmoid(-torch.sum(bs_DI) + max_lid))
        # if optimize_bs_F:
        #     ret += -torch.log(torch.sigmoid(-torch.sum(bs_F) + max_lid))
        ret += width_prior
        if hessian:
            return ret
        if grad:
            ret.backward()
            return (
                ret.detach().cpu().numpy(),
                inp.grad.detach().cpu().numpy(),
            )
        else:
            return ret.detach().cpu().numpy()

    def optimize_with_dlib(
        self,
        optimize_bs_DI=False,
        optimize_bs_F=False,
        optimize_offsets=False,
        optimize_f_shallow=False,
        optimize_cal_fac=False,
        fixed_bs_DI=None,
        fixed_bs_F=None,
        fixed_offsets=None,
        fixed_f_shallow=None,
        fixed_cal_fac=None,
        bounds=None,
        max_lid=100,
        max_feval=3500,
        rtol=1e-8,
        max_opt=70,
        n_rand=500,
        quiet=False,
    ):
        if not any(
            [
                optimize_bs_DI,
                optimize_bs_F,
                optimize_offsets,
                optimize_f_shallow,
                optimize_cal_fac,
            ]
        ):
            print(
                "Specify at least one parameter for optimization."
                "No estimation is executed."
                "Log-marginal likelihood for default parameters is"
            )
            return self.compute_log_ml_value()
        if bounds is None:
            bounds = self._update_bounds(
                optimize_bs_DI,
                optimize_bs_F,
                optimize_offsets,
                optimize_f_shallow,
                optimize_cal_fac,
                max_lid,
            )
        res = dlib_opt(
            func=self.f_opt,
            bounds=bounds,
            args=(
                optimize_bs_DI,
                optimize_bs_F,
                optimize_offsets,
                optimize_f_shallow,
                optimize_cal_fac,
                max_lid,
                fixed_bs_DI,
                fixed_bs_F,
                fixed_offsets,
                fixed_f_shallow,
                fixed_cal_fac,
            ),
            max_feval=max_feval,
            rtol=rtol,
            max_opt=max_opt,
            n_rand=n_rand,
            progress=1 - quiet,
        )
        return res

    def optimize_with_scipy(
        self,
        x0,
        optimize_bs_DI=False,
        optimize_bs_F=False,
        optimize_offsets=False,
        optimize_f_shallow=False,
        optimize_cal_fac=False,
        fixed_bs_DI=None,
        fixed_bs_F=None,
        fixed_offsets=None,
        fixed_f_shallow=None,
        fixed_cal_fac=None,
        bounds=None,
        max_lid=100,
        method="SLSQP",
        grad=False,
        options=None,
        quiet=False,
    ):
        # it can happen that scipy tries values ignoring constraints
        if optimize_bs_DI + optimize_bs_F == 2:
            if self.lif_params == 4:
                constraint = [
                    {"type": "ineq", "fun": lambda x: max_lid - sum(x[:4])},
                    {"type": "ineq", "fun": lambda x: max_lid - sum(x[4:8])},
                ]
            if self.lif_params == 2:
                constraint = [
                    {
                        "type": "ineq",
                        "fun": lambda x: max_lid
                        - sum([x[0], x[1] / 2, x[1], x[1] / 2]),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda x: max_lid
                        - sum([x[2], x[3] / 2, x[3], x[3] / 2]),
                    },
                ]
        elif optimize_bs_DI + optimize_bs_F == 1:
            constraint = {
                "type": "ineq",
                "fun": lambda x: max_lid - sum(x[: self.lif_params]),
            }
        else:
            constraint = ()
        if bounds is None:
            bounds = self._update_bounds(
                optimize_bs_DI,
                optimize_bs_F,
                optimize_offsets,
                optimize_f_shallow,
                optimize_cal_fac,
                max_lid,
            )
        with tqdm(total=options["maxiter"], disable=quiet) as pbar:
            res = scipy.optimize.minimize(
                self.f_opt,
                x0,
                bounds=bounds,
                args=(
                    optimize_bs_DI,
                    optimize_bs_F,
                    optimize_offsets,
                    optimize_f_shallow,
                    optimize_cal_fac,
                    max_lid,
                    fixed_bs_DI,
                    fixed_bs_F,
                    fixed_offsets,
                    fixed_f_shallow,
                    fixed_cal_fac,
                    grad,
                ),
                method=method,
                jac=True if grad else None,
                options=options,
                constraints=constraint,
                # tol=1e-11,
                callback=lambda _: pbar.update(),
            )
        return res

    def _update_bounds(
        self,
        optimize_bs_DI,
        optimize_bs_F,
        optimize_offsets,
        optimize_f_shallow,
        optimize_cal_fac,
        max_lid,
    ):
        bounds = list()
        if optimize_bs_DI:
            [bounds.append([0.0, max_lid]) for _ in range(self.lif_params)]
        if optimize_bs_F:
            [bounds.append([0.0, max_lid]) for _ in range(self.lif_params)]
        if optimize_offsets:
            for _ in self.subs:
                bounds.append([-180, 180])
        if optimize_f_shallow:
            bounds.append([0.01, 1])
        if optimize_cal_fac:
            bounds.append(
                [
                    max(
                        self.prior_mean_cal_fac - 3 * self.prior_std_cal_fac, 0
                    ),
                    self.prior_mean_cal_fac + 3 * self.prior_std_cal_fac,
                ]
            )
        bounds = np.array(bounds)
        return bounds

    def split_input(
        self,
        inp,
        optimize_bs_DI,
        optimize_bs_F,
        optimize_offsets,
        optimize_f_shallow,
        optimize_cal_fac,
        fixed_bs_DI=None,
        fixed_bs_F=None,
        fixed_offsets=None,
        fixed_f_shallow=None,
        fixed_cal_fac=None,
    ):
        fixed_params = [
            [fixed_bs_DI for _ in range(self.lif_params)],
            [fixed_bs_F for _ in range(self.lif_params)],
            [fixed_offsets for _ in range(len(self.subs))],
            [fixed_f_shallow],
            [fixed_cal_fac],
        ]
        opt_params_dupl = [
            [optimize_bs_DI for _ in range(self.lif_params)],
            [optimize_bs_F for _ in range(self.lif_params)],
            [optimize_offsets for _ in range(len(self.subs))],
            [optimize_f_shallow],
            [optimize_cal_fac],
        ]
        idx_inp = 0
        result = []
        for o1, f1 in zip(opt_params_dupl, fixed_params):
            if any(o1):
                result.append(inp[idx_inp: idx_inp + len(o1)])
                idx_inp += len(o1)
            else:
                result.append(f1[idx_inp: idx_inp + len(o1)])
        bs_DI = result[0] if optimize_bs_DI else None
        bs_F = result[1] if optimize_bs_F else None
        offsets = (
            (
                {self.subs[i]: result[2][i] for i in range(len(self.subs))}
                if len(self.subs) > 1
                else result[2]
            )
            if optimize_offsets
            else fixed_offsets
        )
        f_shallow = result[3][0] if optimize_f_shallow else fixed_f_shallow
        cal_fac = result[4][0] if optimize_cal_fac else fixed_cal_fac
        return bs_DI, bs_F, offsets, f_shallow, cal_fac

    def write_results(
        self,
        fname_results,
        bs_DI,
        bs_F,
        offsets,
        f_shallow,
        cal_fac,
        optimizer_output,
        optimizer_args,
        max_lid,
        delta_t,
        optimizer,
    ):
        new_row = [
            bs_DI,
            bs_F,
            offsets,
            f_shallow,
            cal_fac,
            optimizer_output.fun,
            optimizer_output,
            optimizer_args,
            max_lid,
            delta_t,
            optimizer,
        ]
        # Check if the CSV file exists
        if os.path.exists(fname_results):
            # Append new data to the CSV file
            with open(fname_results, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(new_row)
        else:
            # Create CSV file with header row and append new data
            header_row = [
                "bs_DI",
                "bs_F",
                "offsets",
                "f_shallow",
                "cal_fac",
                "log_ml",
                "optimizer_output",
                "optimizer_args",
                "max_lid",
                "delta_t",
                "optimizer",
            ]
            with open(fname_results, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header_row)
                writer.writerow(new_row)

    # def compute_hessian(
    #     self,
    #     bs_DI=None,
    #     bs_F=None,
    #     offsets=None,
    #     f_shallow=None,
    #     cal_fac=None,
    #     fixed_bs_DI=None,
    #     fixed_bs_F=None,
    #     fixed_offsets=None,
    #     fixed_f_shallow=None,
    #     fixed_cal_fac=None,
    #     approximate=False,
    # ):
    #     if isinstance(offsets, dict):
    #         offsets = list(offsets.values())
    #     comb_vals = []
    #     for p in [bs_DI, bs_F, offsets, f_shallow, cal_fac]:
    #         if p is None:
    #             continue
    #         comb_vals += p if isinstance(p, list) else [p]

    #     def wrapper(x):
    #         print(".", end="")
    #         sys.stdout.flush()
    #         if approximate:
    #             if bs_DI is not None:
    #                 x[: len(bs_DI)] = np.clip(x[: len(bs_DI)], 0, None)
    #             elif bs_F is not None:
    #                 x[: len(bs_F)] = np.clip(x[: len(bs_F)], 0, None)
    #             elif bs_DI is not None and bs_F is not None:
    #                 x[: len(bs_DI) + len(bs_F)] = np.clip(
    #                     x[: len(bs_DI) + len(bs_F)], 0, None
    #                 )
    #         return self.f_opt(
    #             x,
    #             optimize_bs_DI=bs_DI is not None,
    #             optimize_bs_F=bs_F is not None,
    #             optimize_offsets=offsets is not None,
    #             optimize_f_shallow=f_shallow is not None,
    #             optimize_cal_fac=cal_fac is not None,
    #             max_lid=1e16,
    #             fixed_bs_DI=fixed_bs_DI,
    #             fixed_bs_F=fixed_bs_F,
    #             fixed_offsets=fixed_offsets,
    #             fixed_f_shallow=fixed_f_shallow,
    #             fixed_cal_fac=fixed_cal_fac,
    #             grad=False,
    #             hessian=1 - approximate,
    #         )

    #     if approximate:
    #         H = nd.Hessian(wrapper)(comb_vals)
    #     else:
    #         H = torch.autograd.functional.hessian(
    #             wrapper,
    #             torch.tensor(comb_vals, dtype=dtype, device=device),
    #         )
    #     return H
