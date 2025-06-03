import numpy as np
import pandas as pd
import scipy
import sys

import ast
import json

from sedprep.utils import kalmag
from sedprep.optimizer import Optimizer

# import requests
# import io
# from sedprep.data_handling import read_arch_data

sed_data_name = sys.argv[1]

synth_testing = True
folder = "synth_data" if synth_testing else "real_data"
fname_results_DI = f"../results/{folder}/{sed_data_name}_DI.csv"
fname_results_F = f"../results/{folder}/{sed_data_name}_F.csv"
fname_results_DIF = f"../results/{folder}/{sed_data_name}.csv"

hessian = True
estimate_DI = False
estimate_F = False
estimate_DIF = False
estimate_all_with_dlib = False

max_lid = 100
delta_t = 40
lif_params = 4

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

if synth_testing:
    arch_data = pd.read_csv(f"../dat/{folder}/arch.csv")
else:
    # pre = "https://nextcloud.gfz-potsdam.de/s/"
    # rej_response = requests.get(f"{pre}WLxDTddq663zFLP/download")
    # rej_response.raise_for_status()
    # with np.load(io.BytesIO(rej_response.content), allow_pickle=True) as fh:
    #     to_reject = fh["to_reject"]
    # data_response = requests.get(f"{pre}r6YxrrABRJjideS/download")
    # arch_data = read_arch_data(io.BytesIO(data_response.content), to_reject)
    arch_data = pd.read_csv("../dat/real_data/real_arch_data.csv")


mean_path = "https://nextcloud.gfz-potsdam.de/s/exaT4iPjnbq2xzo/download"
cov_path = "https://nextcloud.gfz-potsdam.de/s/NcLAi6yM2mp9WDA/download"
if synth_testing:
    knots, dt = np.linspace(
        min(arch_data["t"]), max(arch_data["t"]), 2000, retstep=True
    )
    knots = np.concatenate(
        (
            [min(knots) - 2],
            [min(knots) - 1],
            knots,
            [max(knots) + 1],
            [max(knots) + 2],
        )
    )
    prior_sample = np.load("../dat/prior_sample.npy")
    vals = np.insert(
        prior_sample.T,
        [prior_sample.T.shape[0], 0],
        [prior_sample.T[-1], prior_sample.T[0]],
        axis=0,
    )
    prior_sample_spline = scipy.interpolate.BSpline(knots, vals, 1)
    prior_mean = prior_sample_spline(2000)
    _, prior_cov = kalmag(mean_path, cov_path)
else:
    prior_mean, prior_cov = kalmag(mean_path, cov_path)


if hessian:
    sed_data = pd.read_csv(f"../dat/{folder}/{sed_data_name}.csv")
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
    bs_DI, bs_F, offsets, f_shallow, cal_fac = optimizer.split_input(
        [8.963, 13.1, 0.0, 5.913, 29.13, -6.213, 0.6426],
        optimize_bs_DI=True,
        optimize_bs_F=False,
        optimize_offsets=True,
        optimize_f_shallow=True,
        optimize_cal_fac=False,
    )
    h = optimizer.compute_hessian(bs_DI, bs_F, offsets, f_shallow, cal_fac)
    print(h)

if estimate_DI:
    sed_data = pd.read_csv(f"../dat/{folder}/{sed_data_name}.csv")
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
    print(
        "dlib optimizer is determining "
        "a global optimum for the lock-in function parameters\n"
    )
    if fname_results_DI is not None:
        print(f"The results are stored under {fname_results_DI}.")
    global_bs_DI_opt = optimizer.optimize_with_dlib(
        optimize_bs_DI=True,
        optimize_bs_F=False,
        optimize_offsets=False,
        optimize_f_shallow=False,
        optimize_cal_fac=False,
        fixed_bs_DI=None,
        fixed_bs_F=None,
        fixed_offsets=optimizer.prior_mean_offsets,
        fixed_f_shallow=optimizer.prior_mean_f_shallow,
        fixed_cal_fac=None,
        max_lid=max_lid,
        **dlib_args,
    )
    if fname_results_DI is not None:
        optimizer.write_results(
            fname_results_DI,
            bs_DI=[b for b in global_bs_DI_opt.x[0:lif_params]],
            bs_F=None,
            offsets=optimizer.prior_mean_offsets,
            f_shallow=optimizer.prior_mean_f_shallow,
            cal_fac=None,
            optimizer_output=global_bs_DI_opt,
            optimizer_args=dlib_args,
            max_lid=max_lid,
            delta_t=optimizer.delta_t,
            optimizer="dlib",
        )
    x0 = (
        list(global_bs_DI_opt.x)
        + list(optimizer.prior_mean_offsets.values())
        + [optimizer.prior_mean_f_shallow]
    )
    print(f"Scipy's {scipy_args['method']} is optimizing the following values")
    print(x0)
    polished_opt_DI = optimizer.optimize_with_scipy(
        x0=x0,
        optimize_bs_DI=True,
        optimize_bs_F=False,
        optimize_offsets=True,
        optimize_f_shallow=True,
        optimize_cal_fac=False,
        max_lid=max_lid,
        **scipy_args,
    )
    print(polished_opt_DI)
    bs_DI, bs_F, offsets, f_shallow, cal_fac = optimizer.split_input(
        polished_opt_DI.x,
        optimize_bs_DI=True,
        optimize_bs_F=False,
        optimize_offsets=True,
        optimize_f_shallow=True,
        optimize_cal_fac=False,
    )
    optimizer.write_results(
        fname_results_DI,
        bs_DI=[b for b in bs_DI],
        bs_F=bs_F,
        offsets=offsets,
        f_shallow=f_shallow,
        cal_fac=cal_fac,
        optimizer_output=polished_opt_DI,
        optimizer_args=scipy_args,
        max_lid=max_lid,
        delta_t=optimizer.delta_t,
        optimizer="scipy",
    )

if estimate_F:
    sed_data = pd.read_csv(f"../dat/{folder}/{sed_data_name}.csv")
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
    global_bs_F_opt = optimizer.optimize_with_dlib(
        optimize_bs_DI=False,
        optimize_bs_F=True,
        optimize_offsets=False,
        optimize_f_shallow=False,
        optimize_cal_fac=False,
        fixed_bs_DI=None,
        fixed_bs_F=None,
        fixed_offsets=None,
        fixed_f_shallow=None,
        fixed_cal_fac=optimizer.prior_mean_cal_fac,
        max_lid=max_lid,
        **dlib_args,
    )
    if fname_results_F is not None:
        optimizer.write_results(
            fname_results_F,
            bs_DI=None,
            bs_F=[b for b in global_bs_F_opt.x[0:lif_params]],
            offsets=None,
            f_shallow=None,
            cal_fac=optimizer.prior_mean_cal_fac,
            optimizer_output=global_bs_F_opt,
            optimizer_args=dlib_args,
            max_lid=max_lid,
            delta_t=optimizer.delta_t,
            optimizer="dlib",
        )
    x0 = list(global_bs_F_opt.x) + [optimizer.prior_mean_cal_fac]
    print(
        f"Use scipy's {scipy_args['method']} to optimize the following values"
    )
    print(x0)
    polished_opt_F = optimizer.optimize_with_scipy(
        x0=x0,
        optimize_bs_DI=False,
        optimize_bs_F=True,
        optimize_offsets=False,
        optimize_f_shallow=False,
        optimize_cal_fac=True,
        max_lid=max_lid,
        **scipy_args,
    )
    print(polished_opt_F)
    bs_DI, bs_F, offsets, f_shallow, cal_fac = optimizer.split_input(
        polished_opt_F.x,
        optimize_bs_DI=False,
        optimize_bs_F=True,
        optimize_offsets=False,
        optimize_f_shallow=False,
        optimize_cal_fac=True,
    )
    optimizer.write_results(
        fname_results_F,
        bs_DI=None,
        bs_F=[b for b in bs_F],
        offsets=offsets,
        f_shallow=f_shallow,
        cal_fac=cal_fac,
        optimizer_output=polished_opt_F,
        optimizer_args=scipy_args,
        max_lid=max_lid,
        delta_t=optimizer.delta_t,
        optimizer="scipy",
    )

if estimate_DIF:
    sed_data = pd.read_csv(f"../dat/{folder}/{sed_data_name}.csv")

    optimizer = Optimizer(
        sed_data,
        arch_data,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        delta_t=delta_t,
        end=min(min(sed_data.t), -6000),
    )
    res_DI = pd.read_csv(f"../results/{folder}/{sed_data_name}_DI.csv")
    res_F = pd.read_csv(f"../results/{folder}/{sed_data_name}_F.csv")
    best_est_bs_DI = ast.literal_eval(
        res_DI[res_DI.optimizer == "scipy"].bs_DI.values[0]
    )
    best_est_bs_F = ast.literal_eval(
        res_F[res_F.optimizer == "scipy"].bs_F.values[0]
    )
    best_est_offsets = json.loads(
        res_DI[res_DI.optimizer == "scipy"].offsets.values[0].replace("'", '"')
    )
    best_est_f_shallow = res_DI[res_DI.optimizer == "scipy"].f_shallow.values[
        0
    ]
    best_est_cal_fac = res_F[res_F.optimizer == "scipy"].cal_fac.values[0]
    x0 = (
        best_est_bs_DI
        + best_est_bs_F
        + list(best_est_offsets.values())
        + [best_est_f_shallow]
        + [best_est_cal_fac]
    )
    print(
        f"Use scipy's {scipy_args['method']} to optimize the following values"
    )
    print(x0)
    polished_opt = optimizer.optimize_with_scipy(
        x0=x0,
        optimize_bs_DI=True,
        optimize_bs_F=True,
        optimize_offsets=True,
        optimize_f_shallow=True,
        optimize_cal_fac=True,
        max_lid=max_lid,
        **scipy_args,
    )
    print(polished_opt)
    bs_DI, bs_F, offsets, f_shallow, cal_fac = optimizer.split_input(
        polished_opt.x,
        optimize_bs_DI=True,
        optimize_bs_F=True,
        optimize_offsets=True,
        optimize_f_shallow=True,
        optimize_cal_fac=True,
    )
    optimizer.write_results(
        fname_results_DIF,
        bs_DI=[b for b in bs_DI],
        bs_F=[b for b in bs_F],
        offsets=offsets,
        f_shallow=f_shallow,
        cal_fac=cal_fac,
        optimizer_output=polished_opt,
        optimizer_args=scipy_args,
        max_lid=max_lid,
        delta_t=optimizer.delta_t,
        optimizer="scipy",
    )


if estimate_all_with_dlib:
    sed_data = pd.read_csv(f"../dat/{folder}/{sed_data_name}.csv")
    sed_data["F"] = np.nan
    sed_data["dF"] = np.nan

    optimizer = Optimizer(
        sed_data,
        arch_data,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        delta_t=delta_t,
        end=min(min(sed_data.t), -6000),
    )
    global_bs_DI_opt = optimizer.optimize_with_dlib(
        optimize_bs_DI=False,
        optimize_bs_F=False,
        optimize_offsets=True,
        optimize_f_shallow=True,
        optimize_cal_fac=False,
        fixed_bs_DI=None,
        fixed_bs_F=None,
        fixed_offsets=None,
        fixed_f_shallow=None,
        fixed_cal_fac=None,
        max_lid=max_lid,
        **dlib_args,
    )
    print(global_bs_DI_opt)
    bs_DI, bs_F, offsets, f_shallow, cal_fac = optimizer.split_input(
        global_bs_DI_opt.x,
        optimize_bs_DI=False,
        optimize_bs_F=False,
        optimize_offsets=True,
        optimize_f_shallow=True,
        optimize_cal_fac=False,
    )
    optimizer.write_results(
        fname_results_DI,
        bs_DI=None,
        bs_F=None,
        offsets=offsets,
        f_shallow=f_shallow,
        cal_fac=None,
        optimizer_output=global_bs_DI_opt,
        optimizer_args=dlib_args,
        max_lid=max_lid,
        delta_t=optimizer.delta_t,
        optimizer="scipy",
    )
