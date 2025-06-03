import numpy as np
import pymc as pm

from pytensor import tensor as pt

from utils import sqe_kernel, interp_adm

from hamstr_utils import (
    get_acc_mean,
    get_K_factor,
    get_brks_half_offset,
    get_levels_dict,
)

sigma_c = 2 / 3

class BaconAdm:
    def __init__(self, name, sed):
        self.name = name
        adm_data = sed.adm_data
        self.dd = adm_data.loc[1, "depth"] - adm_data.loc[0, "depth"]

        bottom_depth = max(max(adm_data["depth"]), max(sed.depth))
        self.D = int(round(bottom_depth / self.dd + 0.5) + 1)
        depths = np.arange(self.D) * self.dd

        # don't go into the future (age -70 = year 2020)
        age0_bound = -70
        age0 = 1950 - adm_data["t"][adm_data["depth"] == 0].item()
        dage0 = adm_data["dt"][adm_data["depth"] == 0].item()
        # self.theta = pm.TruncatedNormal(
        #     f"theta_{name}", mu=age0, sigma=dage0, lower=age0_bound
        # )
        self.theta = pm.Normal(f"theta_{name}", mu=age0, sigma=dage0)
        self.acc_mean = get_acc_mean(
            1950 - adm_data["t"].values, adm_data["depth"].values
        )

        mu_c = (np.log(self.acc_mean) - sigma_c**2 / 2) * np.ones_like(depths)
        cov_c = sigma_c**2 * sqe_kernel(depths, tau=100 / self.acc_mean)
        L_c = np.linalg.cholesky(cov_c + 1e-6 * np.eye(cov_c.shape[0]))
        c_cent = pm.Normal(f"c_cent_{name}", mu=0.0, sigma=1.0, size=self.D)
        c = mu_c + L_c @ c_cent

        self.a = pm.Deterministic(f"a_{name}", pt.exp(c))

        a_mean = pm.math.mean(self.a)
        a_mean_bound = (
            -0.5 * ((self.acc_mean - a_mean) / (2 * self.acc_mean)) ** 2
        )
        pm.Potential(f"a_mean_bound_{name}", a_mean_bound)

        self.a_sums = pt.cumsum(self.a, axis=0)

    def get_ages(self, depths):
        z = pt.as_tensor(depths)
        ind = (depths // self.dd).astype(int)
        return pm.Deterministic(
            f"t_{self.name}",
            (
                self.theta
                + self.a_sums[ind] * pt.as_tensor(self.dd)
                + self.a[ind] * (z - pt.as_tensor(self.dd) * (ind + 1))
            ),
        )

    def get_depths(self, ages):
        t_i = pm.math.concatenate(
            [[self.theta], self.theta + self.a_sums * self.dd]
        )
        d_i = np.arange(self.D + 1) * self.dd
        return interp_adm(-ages, -(1950 - t_i), d_i, self.acc_mean)


acc_shape = 5
mem_mean = 0.5
mem_strength = 10

mem_alpha = mem_strength * mem_mean
mem_beta = mem_strength * (1 - mem_mean)


class HamstrAdm:
    def __init__(self, name, sed):
        self.name = name
        adm_data = sed.adm_data
        self.adm_data_dd = adm_data.loc[1, "depth"] - adm_data.loc[0, "depth"]
        top_depth = 0
        bottom_depth = max(max(adm_data["depth"].values), max(sed.depth))

        K_fine_1 = int(bottom_depth - top_depth)

        median_depth_diff = np.median(
            np.diff(np.sort(np.unique(adm_data["depth"].values)))
        )
        K_fine_2 = np.round(16 * K_fine_1 / median_depth_diff)
        K_fine = int(min(K_fine_1, K_fine_2, 900))
        K_factor = get_K_factor(K_fine)
        brks = get_brks_half_offset(K_fine, K_factor)

        levels_dict = get_levels_dict(brks)

        n_lvls = levels_dict["n_levels"]
        K_fine = levels_dict[f"{n_lvls}"]["nK"]

        acc_shape_adj = (
            acc_shape * (n_lvls - 1) / levels_dict["multi_parent_adj"]
        )

        delta_c = (bottom_depth - top_depth) / K_fine
        self.c_depth_bottom = [
            delta_c * c + top_depth for c in list(range(1, K_fine + 1))
        ]
        self.c_depth_top = np.concatenate(
            [[top_depth], self.c_depth_bottom[: K_fine - 1]]
        )
        self.modelled_depths = np.concatenate(
            [[self.c_depth_top[0]], self.c_depth_bottom]
        )

        self.acc_mean = get_acc_mean(
            1950 - adm_data["t"].values, adm_data["depth"].values
        )

        # don't go into the future (age -70 = year 2020)
        age0_bound = -70
        age0 = 1950 - adm_data["t"][adm_data["depth"] == 0].item()
        dage0 = adm_data["dt"][adm_data["depth"] == 0].item()

        age0 = pm.Uniform(
            f"age0_{name}",
            lower=max(age0_bound, age0 - dage0),
            upper=age0 + dage0,
            size=(1,),
        )

        self.alpha_0 = pt.as_tensor([self.acc_mean])
        lower = min(2, self.acc_mean / 3.5)
        upper = self.acc_mean * 3.5

        _alpha_list = [self.alpha_0]
        for it in range(2, n_lvls + 1):
            lvl_data = levels_dict[f"{it}"]
            parent_mean = (
                lvl_data["wts1"] * _alpha_list[it - 2][lvl_data["parent1"]]
                + lvl_data["wts2"] * _alpha_list[it - 2][lvl_data["parent2"]]
            )

            alpha = pm.Gamma(
                f"alphas_level_{it}_{name}",
                alpha=acc_shape_adj,
                beta=acc_shape_adj / parent_mean,
                initval=np.clip(
                    np.random.normal(
                        self.acc_mean,
                        self.acc_mean / np.sqrt(acc_shape_adj) / 5,
                        size=lvl_data["nK"],
                    ),
                    lower,
                    upper,
                ),
                size=lvl_data["nK"],
            )
            alpha_mean = pm.math.mean(alpha)
            alpha_mean_bound = (
                -0.5
                * ((self.acc_mean - alpha_mean) / (self.acc_mean / 2)) ** 2
            )
            pm.Potential(f"alpha_mean_bound_{it}_{name}", alpha_mean_bound)
            _alpha_list.append(alpha)

        R = pm.Beta(
            f"R_{name}",
            alpha=mem_alpha,
            beta=mem_beta,
            initval=np.random.normal(0.5, 0.1, size=1),
            size=1,
        )
        # R = 0.5
        w = R**delta_c
        _x = pm.math.concatenate(([_alpha_list[-1][0]], _alpha_list[-1]))

        idx = np.arange(K_fine)
        self.x = pm.Deterministic(
            f"x_{name}", w * _x[idx] + (1 - w) * _x[idx + 1]
        )

        self.c_ages = pm.Deterministic(
            f"c_ages_{name}",
            pm.math.concatenate(
                (pt.as_tensor(age0), age0 + pt.cumsum(self.x) * delta_c)
            ),
        )

    def get_ages(self, depths):
        which_c = [
            np.argmax((self.c_depth_bottom < d) * (self.c_depth_bottom - d))
            for d in depths
        ]
        return self.c_ages[which_c] + self.x[which_c] * (
            depths - self.c_depth_top[which_c]
        )

    def get_depths(self, ages):
        # translate ages to depth
        # NOTE the hack with the minus sign, this is because for interp_adm
        # the x points have to be ascending
        # this also makes applying the ordering constraint easier
        return interp_adm(
            -ages, -(1950 - self.c_ages), self.modelled_depths, self.acc_mean
        )
