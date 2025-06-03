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
from tqdm import tqdm
from pymagglobal.utils import lmax2N, i2lm_l, scaling

from sedprep.utils import nez2dif, lif_antid, grad_d, grad_i, grad_f, t2ocx
from sedprep.data_handling import max_lit, adjust_declination
from sedprep.constants import (
    REARTH,
    device,
    dtype,
    clip_nigp_D,
    clip_nigp_I,
    clip_nigp_F,
    trunc_dir,
    trunc_int,
    clip_trunc_D,
    use_dip,
)

print(device)


class Filter:
    def __init__(
        self,
        lmax,
        R,
        gamma,
        alpha_dip,
        tau_dip,
        alpha_wodip,
        tau_wodip,
        tau_dip_slow=None,
        axial=False,
        bs_DI=None,
        bs_F=None,
        offsets=None,
        f_shallow=None,
        cal_fac=None,
    ):
        self.lmax = int(lmax)
        self.R = R
        self.gamma = gamma
        self.alpha_dip = alpha_dip
        self.tau_dip = tau_dip
        self.tau_dip_slow = tau_dip_slow
        self.alpha_wodip = alpha_wodip
        self.tau_wodip = tau_wodip
        self.offsets = offsets if offsets is not None else 0
        self.f_shallow = f_shallow if f_shallow is not None else 1
        self.cal_fac = cal_fac if cal_fac is not None else 1
        self.n_coeffs = lmax2N(self.lmax)
        self.with_lock_in_DI = bs_DI is not None
        self.with_lock_in_F = bs_F is not None
        if (bs_DI is not None) and (len(bs_DI) == 4):
            self.with_lock_in_DI = np.any(list(bs_DI[1:]) != [0.0, 0.0, 0.0])
        if (bs_F is not None) and (len(bs_F) == 4):
            self.with_lock_in_F = np.any(list(bs_F[1:]) != [0.0, 0.0, 0.0])
        if (bs_DI is not None) and (len(bs_DI) == 2):
            self.with_lock_in_DI = bs_DI[1] != 0.0
        if (bs_F is not None) and (len(bs_F) == 2):
            self.with_lock_in_F = bs_F[1] != 0.0
        self.with_lock_in = self.with_lock_in_DI | self.with_lock_in_F
        self.max_lid_DI, self.max_lid_F = 0, 0
        if self.with_lock_in_DI:
            if not torch.is_tensor(bs_DI):
                bs_DI = torch.tensor(bs_DI, device=device, dtype=dtype)
            self.max_lid_DI = torch.round(torch.sum(bs_DI) + 0.5)
            if (bs_DI is not None) and (len(bs_DI) == 2):
                self.max_lid_DI = torch.round(2*bs_DI[1] + bs_DI[0] + 0.5)
            self.lif_antid_DI = lif_antid(bs_DI)
        if self.with_lock_in_F:
            if not torch.is_tensor(bs_F):
                bs_F = torch.tensor(bs_F, device=device, dtype=dtype)
            self.max_lid_F = torch.round(torch.sum(bs_F) + 0.5)
            if (bs_F is not None) and (len(bs_F) == 2):
                self.max_lid_F = torch.round(2*bs_F[1] + bs_F[0] + 0.5)
            self.lif_antid_F = lif_antid(bs_F)

        if tau_dip_slow is not None:
            self.omega_dip, self.chi_dip, self.xi_dip = t2ocx(
                torch.tensor(tau_dip, device=device, dtype=dtype)
                if isinstance(tau_dip, float)
                else tau_dip[0],
                torch.tensor(tau_dip_slow, device=device, dtype=dtype),
                be=torch,
            )
        self.axial = axial

        # initiate prior mean and prio covariance
        if isinstance(self.tau_dip, float):
            diag = self.alpha_wodip**2 * torch.ones(
                self.n_coeffs, device=device, dtype=dtype
            )
            diag[0:3] = self.alpha_dip**2
            diag *= torch.from_numpy(
                scaling(self.R, REARTH, self.lmax) ** 2
            ).to(device=device, dtype=dtype)

            self.taus = torch.tensor(
                [
                    self.tau_wodip / float(i2lm_l(i))
                    for i in range(self.n_coeffs)
                ],
                device=device,
                dtype=dtype,
            )
            if self.axial:
                self.taus[0] = (
                    self.tau_dip
                    if tau_dip_slow is None
                    else 1 / self.omega_dip
                )
                self.taus[1:3] = 1 / 2 * self.tau_wodip
            else:
                self.taus[0:3] = (
                    self.tau_dip
                    if tau_dip_slow is None
                    else 1 / self.omega_dip
                )
        else:
            if self.axial:
                diag = torch.hstack([self.alpha_dip, self.alpha_wodip]) ** 2
                self.taus = torch.hstack(
                    [1 / self.omega_dip, self.tau_dip[1:3], self.tau_wodip]
                )
            else:
                print("Not implemented yet")
        self.prior_cov = torch.diag(torch.hstack((diag, diag / self.taus**2)))
        self.prior_mean = torch.zeros(
            2 * self.n_coeffs, device=device, dtype=dtype
        )
        self.prior_mean[0] = self.gamma

    def F(self, delta_t):
        ns = torch.arange(self.n_coeffs, device=device, dtype=int)
        F = torch.zeros(
            (2 * self.n_coeffs, 2 * self.n_coeffs), device=device, dtype=dtype
        )
        frac = delta_t / self.taus
        exp = torch.exp(-torch.abs(frac))
        F[ns, ns] = (1 + torch.abs(frac)) * exp
        F[ns, ns + self.n_coeffs] = delta_t * exp
        F[ns + self.n_coeffs, ns] = -frac / self.taus * exp
        F[ns + self.n_coeffs, ns + self.n_coeffs] = (1 - torch.abs(frac)) * exp
        if self.tau_dip_slow is not None:
            dips = torch.arange(
                1 if self.axial else 3, device=device, dtype=int
            )
            F[dips, dips] = (
                0.5
                / self.xi_dip
                * (
                    (self.xi_dip + self.chi_dip)
                    * torch.exp((self.xi_dip - self.chi_dip) * abs(delta_t))
                    + (self.xi_dip - self.chi_dip)
                    * torch.exp(-(self.xi_dip + self.chi_dip) * abs(delta_t))
                )
            )
            F_01 = (
                torch.exp(-self.chi_dip * abs(delta_t))
                * torch.sinh(self.xi_dip * delta_t)
                / self.xi_dip
            )
            F[dips, dips + self.n_coeffs] = F_01
            F[dips + self.n_coeffs, dips] = -self.omega_dip**2 * F_01
            F[dips + self.n_coeffs, dips + self.n_coeffs] = (
                0.5
                / self.xi_dip
                * (
                    (self.xi_dip - self.chi_dip)
                    * torch.exp((self.xi_dip - self.chi_dip) * abs(delta_t))
                    + (self.xi_dip + self.chi_dip)
                    * torch.exp(-(self.xi_dip + self.chi_dip) * abs(delta_t))
                )
            )
        return F

    def _f_vals(self, lif_antid, intervals):
        f_vals = lif_antid(intervals[1:]) - lif_antid(intervals[:-1])
        # normalize numerically for consistency
        if torch.any(f_vals < -1e-4):
            raise ValueError(
                "Approximating the lock-in function went wrong."
            )
        f_vals[f_vals <= 0] = 0.0
        norm = torch.sum(f_vals)
        if torch.abs(norm - 1) > 1e-6:
            raise ValueError(
                "Approximating the lock-in function went wrong. "
                "The norm of the lock-in function is not one."
            )
        f_vals /= norm
        return f_vals

    def forward(
        self,
        chunked_sed_data,
        chunked_arch_data,
        mean=None,
        cov=None,
        quiet=True,
    ):
        if chunked_arch_data.delta_t != chunked_sed_data.delta_t:
            raise ValueError(
                "chunked_sed_data and chunked_arch_data must have same delta_t"
            )
        else:
            delta_t = chunked_arch_data.delta_t
        self.cal_fac = self.cal_fac / chunked_sed_data.axial_cal_fac
        self.base_s = chunked_sed_data.base
        self.base_a = chunked_arch_data.base
        self._dtBB = torch.einsum(
            "ij, j, jk -> ik",
            self.base_s.T,
            torch.diag(self.prior_cov)[self.n_coeffs:],
            self.base_s,
        )
        if mean is None:
            mean = self.prior_mean
        else:
            mean = torch.from_numpy(mean).to(device=device, dtype=dtype)
        if cov is None:
            cov = self.prior_cov
        else:
            cov = torch.from_numpy(cov).to(device=device, dtype=dtype)
        self.step = delta_t
        self.t2d = chunked_sed_data.t2d
        if chunked_sed_data.n_DI == 0:
            self.max_lid_DI = 0
            self.with_lock_in_DI = False
        if chunked_sed_data.n_F == 0:
            self.max_lid_F = 0
            self.with_lock_in_F = False
        max_lid = max(self.max_lid_DI, self.max_lid_F)
        self.max_lit = max_lit(chunked_sed_data.acc_rates, max_lid, self.step)
        self.int_knots = np.arange(0, self.max_lit + self.step, self.step)
        self.int_knots += chunked_arch_data.start
        # integration intervals with adaptive spacing
        self.intervals = np.linspace(
            self.int_knots.min() - self.step / 2.0,
            self.int_knots.max() + self.step / 2.0,
            len(self.int_knots) + 1,
        )
        self.n_kn = len(self.int_knots)
        # initial set of means and covariances, one for each integration point
        int_means = mean.repeat(self.n_kn, 1)
        int_covs = torch.zeros(
            (self.n_kn, 2 * self.n_coeffs, self.n_kn, 2 * self.n_coeffs),
            device=device,
            dtype=dtype,
        )
        for it in range(self.n_kn):
            for jt in range(self.n_kn):
                pwr = jt - it
                if pwr <= 0:
                    int_covs[it, :, jt, :] = (
                        cov @ self.F(-abs(pwr) * self.step).T
                    )
                else:
                    int_covs[it, :, jt, :] = (
                        self.F(-abs(pwr) * self.step) @ cov
                    )
        log_ml = 0
        for chunk_s, chunk_a in tqdm(
            zip(chunked_sed_data.chunks, chunked_arch_data.chunks),
            total=len(chunked_sed_data.chunks),
            disable=quiet,
        ):
            # calculate the forecast for the first entry of the means
            F = self.F(-delta_t)
            mean = self.prior_mean + F @ (int_means[0] - self.prior_mean)
            cov = (
                F @ (int_covs[0, :, 0, :] - self.prior_cov) @ F.T
                + self.prior_cov
            )
            # shift the knots
            self.int_knots -= delta_t
            self.intervals -= delta_t
            # shift the means...
            int_means[1:] = int_means[:-1].clone()
            # ...and update the first entry using the forward model
            int_means[0] = mean.clone()
            # shift covariances
            int_covs[1:, :, 1:, :] = int_covs[:-1, :, :-1, :].clone()
            # and update the rest
            int_covs[0, :, 1:, :] = torch.einsum(
                "ij, jkl -> ikl", F, int_covs[0, :, :-1, :]
            )
            int_covs[1:, :, 0, :] = torch.einsum(
                "ijk, kl -> ijl", int_covs[:-1, :, 0, :], F.T
            )
            int_covs[0, :, 0, :] = cov.clone()
            # prediction step
            if chunk_s.n + chunk_a.n:
                # torch.cuda.empty_cache()
                int_means, int_covs, res = self.correct(
                    int_means,
                    int_covs,
                    chunk_s,
                    chunk_a,
                )
                log_ml = log_ml + res
        return log_ml

    def correct(self, int_means, int_covs, chunk_s, chunk_a):
        n_kn_coeffs = self.n_kn * 2 * self.n_coeffs
        H_s = torch.zeros(
            (3, self.n_kn, self.n_coeffs), device=device, dtype=dtype
        )
        if chunk_s.n:
            mean_s_DI = int_means[0, : self.n_coeffs]
            mean_s_F = int_means[0, : self.n_coeffs]
            if self.with_lock_in:
                intervals_d = self.t2d(self.intervals)
                intervals_d -= self.t2d(self.int_knots[0])
                intervals_d *= -1
                intervals_d = torch.from_numpy(intervals_d).to(
                    device=device, dtype=dtype
                )
                if self.with_lock_in_DI:
                    f_vals_DI = self._f_vals(self.lif_antid_DI, intervals_d)
                    # integrate the mean (Riemann sum)
                    mean_s_DI = torch.sum(
                        int_means[:, : self.n_coeffs] * f_vals_DI[:, None],
                        axis=0,
                    )
                if self.with_lock_in_F:
                    f_vals_F = self._f_vals(self.lif_antid_F, intervals_d)
                    # integrate the mean (Riemann sum)
                    mean_s_F = torch.sum(
                        int_means[:, : self.n_coeffs] * f_vals_F[:, None],
                        axis=0,
                    )
            mu_NEZ_s_DI = mean_s_DI @ self.base_s
            mu_NEZ_s_F = mean_s_F @ self.base_s
            grad_D_s = grad_d(mu_NEZ_s_DI[None, :])
            grad_I_s = grad_i(mu_NEZ_s_DI[None, :], f_shallow=self.f_shallow)
            grad_F_s = grad_f(mu_NEZ_s_F[None, :], cal_fac=self.cal_fac)
            grad_DI_s = torch.vstack((grad_D_s, grad_I_s))
            H_s[:2, :, :] = (grad_DI_s @ self.base_s.T)[:, None, :]
            H_s[2, :, :] = (grad_F_s @ self.base_s.T)[:, None, :]
            if self.with_lock_in_DI:
                H_s[:2, :, :] *= f_vals_DI[None, :, None]
            if self.with_lock_in_F:
                H_s[2, :, :] *= f_vals_F[:, None]
        mu_NEZ_a = torch.einsum(
            "i, ijk->jk",
            int_means[0, : self.n_coeffs],
            self.base_a[:, chunk_a.index],
        )
        grad_D_a = grad_d(mu_NEZ_a[chunk_a.idx_D_rel])
        grad_I_a = grad_i(mu_NEZ_a[chunk_a.idx_I_rel])
        grad_F_a = grad_f(mu_NEZ_a[chunk_a.idx_F_rel])
        grad_DIF = torch.vstack((grad_D_a, grad_I_a, grad_F_a))
        H_a = torch.einsum(
            "ijk,jk->ij",
            self.base_a[
                :,
                np.concatenate(
                    (chunk_a.idx_D_abs, chunk_a.idx_I_abs, chunk_a.idx_F_abs)
                ),
            ],
            grad_DIF,
        ).T
        H = torch.zeros(
            (chunk_a.n_DIF + chunk_s.n_DIF, *int_means.size()),
            device=device,
            dtype=dtype,
        )
        H[: chunk_a.n_DIF, 0, : self.n_coeffs] = H_a

        # 0th order term
        mu_D_a, mu_I_a, mu_F_a = nez2dif(*mu_NEZ_a.T)
        mu_DIF = torch.hstack(
            (
                mu_D_a[chunk_a.idx_D_rel],
                mu_I_a[chunk_a.idx_I_rel],
                mu_F_a[chunk_a.idx_F_rel],
            )
        )
        errs_T = chunk_a.errs_T
        errs = chunk_a.errs
        outputs = chunk_a.outputs
        if chunk_s.n:
            H[chunk_a.n_DIF:, :, : self.n_coeffs] = torch.vstack(
                [
                    H_s[0, :, :].repeat(chunk_s.idx_D_rel.size, 1, 1),
                    H_s[1, :, :].repeat(chunk_s.idx_I_rel.size, 1, 1),
                    H_s[2, :, :].repeat(chunk_s.idx_F_rel.size, 1, 1),
                ]
            )
            mu_D_s, mu_I_s, _ = nez2dif(*mu_NEZ_s_DI, f_shallow=self.f_shallow)
            _, _, mu_F_s = nez2dif(*mu_NEZ_s_F, cal_fac=self.cal_fac)
            mu_D_s = [
                mu_D_s + self.offsets[sub]
                if isinstance(self.offsets, dict)
                else mu_D_s + self.offsets
                for sub in chunk_s.subs
            ]
            mu_DIF = torch.hstack(
                (
                    mu_DIF,
                    *mu_D_s,
                    mu_I_s.repeat(chunk_s.idx_I_rel.size),
                    mu_F_s.repeat(chunk_s.idx_F_rel.size),
                )
            )
            grad_DIF = torch.vstack(
                (
                    grad_DIF,
                    grad_D_s.repeat(chunk_s.idx_D_rel.size, 1),
                    grad_I_s.repeat(chunk_s.idx_I_rel.size, 1),
                    grad_F_s.repeat(chunk_s.idx_F_rel.size, 1),
                )
            )
            errs_T = torch.concatenate((errs_T, chunk_s.errs_T))
            errs = torch.concatenate((errs, chunk_s.errs))
            outputs = torch.concatenate((outputs, chunk_s.outputs))

        # calculate NIGP correction
        nigp_corr = torch.einsum("ij,jk,ik->i", grad_DIF, self._dtBB, grad_DIF)
        nigp_corr = nigp_corr * errs_T
        clip_values = torch.ones_like(nigp_corr)
        cv1 = clip_values.clone()
        cv1[: chunk_a.n_D] *= clip_nigp_D**2
        cv2 = cv1.clone()
        cv2[chunk_a.n_D: chunk_a.n_DI] *= clip_nigp_I**2
        cv3 = cv2.clone()
        cv3[chunk_a.n_DI: chunk_a.n_DIF] *= clip_nigp_F**2
        if chunk_s.n:
            n_DIF_D = chunk_a.n_DIF + chunk_s.n_D
            n_DIF_DI = chunk_a.n_DIF + chunk_s.n_DI
            n_DIF_DIF = chunk_a.n_DIF + chunk_s.n_DIF
            cv4 = cv3.clone()
            cv4[chunk_a.n_DIF: n_DIF_D] *= clip_nigp_D**2
            cv5 = cv4.clone()
            cv5[n_DIF_D:n_DIF_DI] *= clip_nigp_I**2
            cv6 = cv5.clone()
            cv6[n_DIF_DI:n_DIF_DIF] *= clip_nigp_F**2
            nigp_corr = torch.clamp(nigp_corr.clone(), max=cv6)
        else:
            nigp_corr = torch.clamp(nigp_corr.clone(), max=cv3)

        if use_dip:
            I_a = torch.arctan(2 * torch.tan(torch.deg2rad(chunk_a.lat)))[
                chunk_a.idx_D_rel
            ]
            if chunk_s.n:
                I_s = torch.arctan(2 * torch.tan(torch.deg2rad(chunk_s.lat)))
        else:
            I_a = torch.deg2rad(mu_I_a[chunk_a.idx_D_rel])
            if chunk_s.n:
                I_s = torch.deg2rad(mu_I_s)

        # truncation errors
        trunc_errs_temp = torch.zeros_like(nigp_corr)
        trunc_errs = trunc_errs_temp.clone()
        trunc_dI = (57.3 / 140) * trunc_dir
        trunc_dD_a = (trunc_dI / torch.cos(I_a)).clip(None, clip_trunc_D)
        trunc_errs[: chunk_a.n_D] = trunc_dD_a**2
        trunc_errs[chunk_a.n_D: chunk_a.n_DI] = trunc_dI**2
        trunc_errs[chunk_a.n_DI: chunk_a.n_DIF] = trunc_int**2
        if chunk_s.n:
            trunc_dD_s = (trunc_dI / torch.cos(I_s)).clip(None, clip_trunc_D)
            if len(chunk_s.idx_D_rel):
                trunc_errs[chunk_a.n_DIF: n_DIF_D] = trunc_dD_s**2
            trunc_errs[n_DIF_D:n_DIF_DI] = trunc_dI**2
            trunc_errs[n_DIF_DI:n_DIF_DIF] = trunc_int**2

        H_stacked = H.reshape(-1, n_kn_coeffs)
        cov_stacked = int_covs.reshape(n_kn_coeffs, -1)
        R = torch.diag(errs + nigp_corr + trunc_errs)
        ker = H_stacked @ cov_stacked @ H_stacked.T + R
        C = torch.linalg.cholesky(ker)
        Q = torch.linalg.solve_triangular(C, H_stacked, upper=False)
        Q = torch.linalg.solve_triangular(C.T, Q, upper=True)
        K = cov_stacked @ Q.T

        df = outputs - mu_DIF
        # Consider periodicity of declination values
        df[: chunk_a.n_D] = adjust_declination(df[: chunk_a.n_D])
        if chunk_s.n_D:
            n = chunk_a.n_DIF
            n2 = n + chunk_s.n_D
            df[n:n2] = adjust_declination(df[n:n2])

        # we can not replace the second
        # matrix multiplication due to precision errors
        # cov = cov - K @ cov_HT.T
        # Therefore we also need the Josephson form below...
        J = torch.eye(n_kn_coeffs, device=device, dtype=dtype) - K @ H_stacked
        mean_new = int_means + K.reshape(self.n_kn, 2*self.n_coeffs, -1) @ df
        cov_new = (J @ cov_stacked @ J.T + K @ R @ K.T).reshape(
            self.n_kn, 2 * self.n_coeffs, self.n_kn, 2 * self.n_coeffs
        )

        v = torch.linalg.solve_triangular(C, df.unsqueeze(-1), upper=False)
        misfit = (v**2).sum()
        logdet = -2 * torch.sum(torch.log(torch.diag(C)))

        return mean_new, cov_new, - misfit + logdet

    def sample_prior(self, ts, n_samps=10, quiet=True):
        """Returns an ensemble of n_samps samples from the prior.

        Parameters
        ----------
        ts : array
            The knot points at which the prior samples are evaluated
        n_samps : int, optional
            The number of samples
        quiet : bool, optional
            If False, show progress using tqdm.

        Returns
        -------
        ens: array
            An array storing the ensemble along the first dimension, i.e.
            each element ens[i] is a sample from the posterior.
        """
        n = len(ts)
        mean = self.prior_mean.cpu().detach().numpy()
        cov = self.prior_cov.cpu().detach().numpy()

        ens = np.zeros((n_samps, self.prior_mean.shape[0], n))

        ens[:, :, -1] = np.random.multivariate_normal(
            mean=mean,
            cov=cov,
            size=n_samps,
        )
        ens[:, :, -1] -= np.mean(ens[:, :, -1], axis=0)
        ens[:, :, -1] += mean

        for it in tqdm(range(n - 1), disable=quiet):
            k = n - (it + 1)
            k1 = n - (it + 2)
            delta_t = ts[k1] - ts[k]
            delta_t = -delta_t

            F = self.F(delta_t)
            G = np.linalg.multi_dot((cov, F.T, np.linalg.inv(cov)))

            noise_cov = cov - np.linalg.multi_dot((G, self.prior_cov, G.T))

            ens[:, :, k1] = (
                mean[None, :]
                + np.einsum(
                    "...ij, ...j -> ...i", G, ens[:, :, k] - mean[None, :]
                )
                + np.random.multivariate_normal(
                    mean=np.zeros(noise_cov.shape[0]),
                    cov=noise_cov,
                    size=n_samps,
                )
            )

            ens[:, :, k1] -= np.mean(ens[:, :, k1], axis=0)
            ens[:, :, k1] += mean

        return ens
