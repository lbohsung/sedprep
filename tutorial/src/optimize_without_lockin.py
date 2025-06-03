import numpy as np
import pandas as pd
import sys

from sedprep.utils import kalmag
from sedprep.optimizer import Optimizer

# import requests
# import io
# from sedprep.data_handling import read_arch_data

sed_data_name = sys.argv[1]

fname_results = f"../results/{sed_data_name}.csv"

estimate_DI = True
estimate_F = True
estimate_DIF = False

delta_t = 40

max_lid = 100
delta_t = 40
lif_params = 2

dlib_args = {
    "max_feval": 3500,
    "rtol": 1e-8,
    "max_opt": 70,
    "n_rand": 300,
}
scipy_args = {
    "options": {"maxiter": 3500, "maxfun": 3500},
    "grad": True,
    "method": "SLSQP",  # Nelder-Mead, SLSQP
}

# pre = "https://nextcloud.gfz-potsdam.de/s/"
# rej_response = requests.get(f"{pre}WLxDTddq663zFLP/download")
# rej_response.raise_for_status()
# with np.load(io.BytesIO(rej_response.content), allow_pickle=True) as fh:
#     to_reject = fh["to_reject"]
# data_response = requests.get(f"{pre}r6YxrrABRJjideS/download")
# arch_data = read_arch_data(io.BytesIO(data_response.content), to_reject)
arch_data = pd.read_csv("../dat/real_arch_data.csv")

mean_path = "../dat/MF0_6371_Y2000M0D0H0M0S0.dat"
cov_path = "../dat/CovarianceMF0_6371_Y2000M0D0H0M0S0.dat"
prior_mean, prior_cov = kalmag(mean_path, cov_path)


if estimate_DI:
    sed_data = pd.read_csv(f"../dat/{sed_data_name}_prepared.csv")
    sed_data["F"] = np.nan
    sed_data["dF"] = np.nan

    optimizer = Optimizer(
        sed_data,
        arch_data,
        lif_params=lif_params,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        delta_t=delta_t,
        end=min(min(sed_data.t), -6000),
    )
    x0 = list(optimizer.prior_mean_offsets.values()) + [
        optimizer.prior_mean_f_shallow
    ]
    print(
        f"Use scipy's {scipy_args['method']} to optimize the following values"
    )
    print(x0)
    opt = optimizer.optimize_with_scipy(
        x0=x0,
        optimize_bs_DI=False,
        optimize_bs_F=False,
        optimize_offsets=True,
        optimize_f_shallow=True,
        optimize_cal_fac=False,
        **scipy_args,
    )
    print(opt)
    bs_DI, bs_F, offsets, f_shallow, cal_fac = optimizer.split_input(
        opt.x,
        optimize_bs_DI=False,
        optimize_bs_F=False,
        optimize_offsets=True,
        optimize_f_shallow=True,
        optimize_cal_fac=False,
    )
    optimizer.write_results(
        fname_results,
        bs_DI=None,
        bs_F=None,
        offsets=offsets,
        f_shallow=f_shallow,
        cal_fac=None,
        optimizer_output=opt,
        optimizer_args=scipy_args,
        max_lid=None,
        delta_t=optimizer.delta_t,
        optimizer="scipy",
    )


if estimate_F:
    sed_data = pd.read_csv(f"../dat/{sed_data_name}_prepared.csv")
    sed_data["D"] = np.nan
    sed_data["dD"] = np.nan
    sed_data["I"] = np.nan
    sed_data["dI"] = np.nan

    optimizer = Optimizer(
        sed_data,
        arch_data,
        lif_params=lif_params,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        delta_t=delta_t,
        end=min(min(sed_data.t), -6000),
    )
    x0 = [optimizer.prior_mean_cal_fac]
    print(
        f"Use scipy's {scipy_args['method']} to optimize the following values"
    )
    print(x0)
    opt = optimizer.optimize_with_scipy(
        x0=x0,
        optimize_bs_DI=False,
        optimize_bs_F=False,
        optimize_offsets=False,
        optimize_f_shallow=False,
        optimize_cal_fac=True,
        **scipy_args,
    )
    print(opt)
    optimizer.write_results(
        fname_results,
        bs_DI=None,
        bs_F=None,
        offsets=None,
        f_shallow=None,
        cal_fac=opt.x,
        optimizer_output=opt,
        optimizer_args=scipy_args,
        max_lid=None,
        delta_t=optimizer.delta_t,
        optimizer="scipy",
    )
