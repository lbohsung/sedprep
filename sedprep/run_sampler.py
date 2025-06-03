import numpy as np
import pandas as pd
import arviz as az
import sys
from scipy.interpolate import interp1d
from mcmc_sampling import mcmc_sampling

# core_ids = ["U1305"]
core_ids = sys.argv[1:]

if core_ids[0] == "all":
    core_ids = [
        "305A5",
        "BIW95-4",
    ]

for core_id in core_ids:
    print(core_id)
    path_arch = "../dat/arch_data_prepared.csv"
    path_sed = "../dat/sed_data"
    path_results = f"../results/{core_id}_result.nc"
    path_summary = f"../results/{core_id}_summary.csv"

    arch_data = pd.read_csv(path_arch)
    mag_data = pd.read_csv(f"{path_sed}/{core_id}.csv")
    adm_data = pd.read_csv(f"{path_sed}/{core_id}_adm_data.csv")
    # mag_data = pd.read_csv(
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

    mag_data["F"] = np.nan
    mag_data["dF"] = np.nan
    prior_d2t = interp1d(
        adm_data["depth"],
        adm_data["t"],
        kind="linear",
        fill_value="extrapolate",
    )
    prior_d2dt = interp1d(
        adm_data["depth"],
        adm_data["dt"],
        kind="linear",
        fill_value="extrapolate",
    )
    depth_max = max(mag_data["depth"])
    min_t = prior_d2t(depth_max) - 2 * prior_d2dt(depth_max)
    print(min_t)

    mcmc_sampling(
        arch_data,
        [{"core_id": core_id, "mag_data": mag_data, "adm_data": adm_data}],
        t_min=min(min_t, -8000),
        t_max=2000,
        step=50,
        lif_type="4p",
        path_results=path_results,
    )
    print(f"Sampling finished. Results can be found under {path_results}")

    summary = az.summary(az.from_netcdf(path_results))
    summary.to_csv(path_summary)

    print(f"Summary generated. Results can be found under {path_summary}")
