import pandas as pd
import numpy as np
from sedprep.constants import na
from sedprep.data_handling import _calc_filehash
import pathlib

path = pathlib.Path(__file__).parent.resolve()


def compute_RPI(row):
    if row["RPIprefID"] in ("102", "105", "107", "111", "108;113", "102;103"):
        if pd.notna(row["NRM_ARM"]):
            return str(row["NRM_ARM"]) + "," + str(row["Sigma[NRM_ARM]"])
        if pd.notna(row["NRM_IRM"]):
            return str(row["NRM_IRM"]) + "," + str(row["Sigma[NRM_IRM]"])
        if pd.notna(row["NRM_k"]):
            return str(row["NRM_k"]) + "," + str(row["Sigma[NRM_k]"])
        else:
            return str(np.NaN) + "," + str(np.NaN)
    elif row["RPIprefID"] in ("103", "112", "103;113", "106;113"):
        if pd.notna(row["NRM_IRM"]):
            return str(row["NRM_IRM"]) + "," + str(row["Sigma[NRM_IRM]"])
        elif pd.notna(row["NRM_ARM"]):
            return str(row["NRM_ARM"]) + "," + str(row["Sigma[NRM_ARM]"])
        elif pd.notna(row["NRM_k"]):
            return str(row["NRM_k"]) + "," + str(row["Sigma[NRM_k]"])
        else:
            return str(np.NaN) + "," + str(np.NaN)
    elif row["RPIprefID"] == "104":
        if pd.notna(row["NRM_k"]):
            return str(row["NRM_k"]) + "," + str(row["Sigma[NRM_k]"])
        elif pd.notna(row["NRM_ARM"]):
            return str(row["NRM_ARM"]) + "," + str(row["Sigma[NRM_ARM]"])
        elif pd.notna(row["NRM_IRM"]):
            return str(row["NRM_IRM"]) + "," + str(row["Sigma[NRM_IRM]"])
        else:
            return str(np.NaN) + "," + str(np.NaN)
    else:
        if pd.notna(row["NRM_ARM"]):
            return str(row["NRM_ARM"]) + "," + str(row["Sigma[NRM_ARM]"])
        elif pd.notna(row["NRM_IRM"]):
            return str(row["NRM_IRM"]) + "," + str(row["Sigma[NRM_IRM]"])
        elif pd.notna(row["NRM_k"]):
            return str(row["NRM_k"]) + "," + str(row["Sigma[NRM_k]"])
        else:
            return str(np.NaN) + "," + str(np.NaN)


def read_geomagia_data(core):
    fname = path.joinpath(f"../dat/{core}_raw_geomagia.csv")
    df = pd.read_csv(
        fname,
        skiprows=1,
        na_values=na,
        usecols=[
            "LocationCode",
            "Lat[deg.]",
            "Lon[deg.]",
            "CoreID",
            "NRM_ARM",
            "NRM_k",
            "NRM_IRM",
            "RPIprefID",
            "Sigma[NRM_ARM]",
            "Sigma[NRM_IRM]",
            "Sigma[NRM_k]",
            "Age[yr.BP]",
            "CoreDepth[cm]",
            "CompDepth[cm]",
            "IncRaw[deg.]",
            "DecRaw[deg.]",
            "MAD[deg.]",
            "DecAdj[deg.]",
            "IncAdj[deg.]",
            "UID",
        ],
    )
    df = df[
        (df["DecRaw[deg.]"].notna())
        | (df["IncRaw[deg.]"].notna())
        | (df["DecAdj[deg.]"].notna())
        | (df["IncAdj[deg.]"].notna())
    ].reset_index(drop=True)
    df["D"] = df.apply(
        lambda row: (
            row["DecAdj[deg.]"]
            if pd.notna(row["DecAdj[deg.]"])
            else row["DecRaw[deg.]"]
        ),
        axis=1,
    )
    df["I"] = df.apply(
        lambda row: (
            row["IncAdj[deg.]"]
            if pd.notna(row["IncAdj[deg.]"])
            else row["IncRaw[deg.]"]
        ),
        axis=1,
    )
    if len(df) == 0:
        print("No data available!")
    df[["F", "dF"]] = df.apply(lambda row: compute_RPI(row), axis=1).str.split(
        ",", expand=True
    )
    df.F = df.F.apply(lambda row: np.NaN if row == "nan" else float(row))
    df.dF = df.dF.apply(lambda row: np.NaN if row == "nan" else float(row))
    df["depth"] = (
        df["CompDepth[cm]"]
        if np.any(~np.isnan(df["CompDepth[cm]"]))
        else df["CoreDepth[cm]"]
    )
    df = df[df["depth"].notna()]
    # change BP to calendar years
    df["t_old"] = 1950 - df["Age[yr.BP]"]
    df.rename(
        columns={
            "MAD[deg.]": "MAD",
            "Lat[deg.]": "lat",
            "Lon[deg.]": "lon",
            "CoreID": "subs",
            "UID": "UID",
        },
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    df["UID"] = df.index
    df["FID"] = _calc_filehash(fname)
    return df


def remove_outliers(data, remove_uids_DI, remove_uids_F):
    df = data.copy()
    df.loc[df["UID"].isin(remove_uids_DI), ["D", "I"]] = np.nan
    df.loc[df["UID"].isin(remove_uids_F), ["F"]] = np.nan
    removed_rows_DI = data[data["UID"].isin(remove_uids_DI)]
    removed_rows_F = data[data["UID"].isin(remove_uids_DI)]
    return df, removed_rows_DI, removed_rows_F


def evaluate_adm(idata, depths, target_samps=1000):
    """
    For an array of depths, use a set of age-depth model samples to calculate
    mean and standard deviation of the corresponding ages.

    Parameters
    ----------
    idata : arviz.InferenceData
        The age-depth model samples as returned by sample_adm.
    depths : array-like of float
        The depths at which the age-depth model should be evaluated.
    target_samps : int, optional
        The number of samples to be used from the chain. Thinning will be
        applied to the chain to match this number.

    Returns
    -------
    array
        The mean of the ages corresponding to depths.
    array
        The standard deviation of the ages corresponding to depths.
    """
    thin = (len(idata.posterior.chain) * len(idata.posterior.draw)) \
        // target_samps
    dd = np.array(idata.adm_pars['dd']).flatten().item()
    D = np.array(idata.adm_pars['D']).flatten().item()
    a_samps = np.array(idata.posterior["a"])
    a_samps = a_samps[:, ::thin].reshape(-1, D).T
    theta_samps = np.array(idata.posterior["theta"])
    theta_samps = theta_samps[:, ::thin].flatten()
    sum_a_samps = np.cumsum(a_samps, axis=0)
    ind_at = (depths // dd).astype(int)
    t_at = theta_samps \
        - sum_a_samps[ind_at] * dd \
        - a_samps[ind_at] * (depths - dd * (ind_at+1))[:, None]
    t_mean = np.mean(t_at, axis=1)
    t_std = np.std(t_at, axis=1)
    return t_mean, t_std
