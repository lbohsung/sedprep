import numpy as np
import pandas as pd
import sys
import arviz as az
from tqdm import tqdm
from data_handling import Data, unshallow_inc, offset_dec
from deconvolve_old import deconvolve
import pymc as pm
from adm import BaconAdm as AgeDepthModel
from scipy.interpolate import interp1d

use_mean = True

core_id = sys.argv[1:][0]

sed_data = pd.read_csv(f"../dat/sed_data/{core_id}.csv")
adm_data = pd.read_csv(f"../dat/sed_data/{core_id}_adm_data.csv")
# sed_data = pd.read_csv(
#     f"../../paleomag_mcmc/dat/prepared_data/{core_id}.csv"
# )
# adm_data = pd.read_csv(
#     f"../../paleomag_mcmc/dat/prepared_adm_data/{core_id}_adm_data.csv"
# )
# adm_data.rename(
#     columns={"depth [cm]": "depth", "t [age BP]": "t", "dt [yrs.]": "dt"},
#     inplace=True,
# )
# adm_data["t"] = 1950 - adm_data["t"]
sed_data["F"] = np.nan
sed_data["dF"] = np.nan
sed_data = sed_data.dropna(subset=["D", "I"], how="all").reset_index(drop=True)
sed = Data(sed_data, adm_data)

with pm.Model() as _:
    adm = AgeDepthModel(core_id, sed)

total_samples = 2000
n_samples = 500
step_size = max(1, total_samples // n_samples)
n_data = len(sed_data)

posterior = az.from_netcdf(f"../results/{core_id}_result.nc").posterior
a_samps = posterior[f"a_{core_id}"].values[:, ::step_size].reshape(-1, adm.D).T
sum_a_samps = np.cumsum(a_samps, axis=0)
theta_samps = posterior[f"theta_{core_id}"].values[:, ::step_size].flatten()
z = np.linspace(0, adm.D * adm.dd - 0.1, 1001)
ind = (z // adm.dd).astype(int)
t_samps = (
    theta_samps
    + sum_a_samps[ind] * adm.dd
    + a_samps[ind] * (z - adm.dd * (ind + 1))[:, None]
)
a_s_samps = posterior[f"lock_in_{core_id}"].values.reshape(-1, 4)[::step_size]
bs_samps = np.cumsum(a_s_samps.T, axis=0)
f_shallow_samps = posterior[f"f_shallow_{core_id}"].values.reshape(-1)[
    ::step_size
]
offset_samps = {}
for name in sed.subcores:
    offset_samps[name] = posterior[f"offset_{name}"].values.reshape(-1)[
        ::step_size
    ]

if use_mean:
    offsets_mean = {name: offset_samps[name].mean() for name in sed.subcores}
    bs_mean = bs_samps.mean(axis=1)
    sed_data_temp = sed_data.copy()
    d2t_mean = interp1d(
        z, 1950 - t_samps.mean(axis=1), kind="linear", fill_value="extrapolate"
    )
    d2t_std = interp1d(
        z, t_samps.std(axis=1), kind="linear", fill_value="extrapolate"
    )
    sed_data_temp["t"] = d2t_mean(sed_data.depth)
    sed_data_temp["dt"] = d2t_std(sed_data.depth)
    sed_data_temp["I"] = unshallow_inc(sed_data_temp.I, f_shallow_samps.mean())
    sed_data_temp = offset_dec(sed_data_temp, offsets_mean)
    mean_D, cov_D, mean_I, cov_I = deconvolve(sed_data_temp, bs_DI=bs_mean)
else:
    means_D, vars_D, means_I, vars_I = [
        np.zeros((n_samples, n_data)) for _ in range(4)
    ]
    for j in tqdm(range(n_samples)):
        sed_data_temp = sed_data.copy()
        t_samp = t_samps[:, j]
        bs = bs_samps[:, j]
        f_shallow = f_shallow_samps[j]
        offsets = {}
        for name in sed.subcores:
            offsets[name] = offset_samps[name][j]

        d2t = interp1d(
            z, 1950 - t_samp, kind="linear", fill_value="extrapolate"
        )
        sed_data_temp["t"] = d2t(sed_data_temp["depth"])
        sed_data_temp["I"] = unshallow_inc(sed_data_temp.I, f_shallow)
        sed_data_temp = offset_dec(sed_data_temp, offsets)

        mean_D, cov_D, mean_I, cov_I = deconvolve(sed_data_temp, bs_DI=bs)

        means_D[j, :] = mean_D
        vars_D[j, :] = np.diag(cov_D)
        means_I[j, :] = mean_I
        vars_I[j, :] = np.diag(cov_I)

    # Aggregate results
    mean_D = np.mean(means_D, axis=0)
    var_D = np.mean(vars_D, axis=0) + np.var(means_D, axis=0, ddof=1)
    mean_I = np.mean(means_I, axis=0)
    var_I = np.mean(vars_I, axis=0) + np.var(means_I, axis=0, ddof=1)

name = "mean" if use_mean else "all"
sed_data.loc[sed.idx_D, "D"] = mean_D
sed_data.loc[sed.idx_D, "dD"] = np.sqrt(np.diag(cov_D)) if use_mean else np.sqrt(var_D)
sed_data.loc[sed.idx_I, "I"] = mean_I
sed_data.loc[sed.idx_I, "dI"] = np.sqrt(np.diag(cov_I)) if use_mean else np.sqrt(var_I)
sed_data.to_csv(
    f"../results/preprocessed_data/{core_id}_{name}.csv", index=False
)

