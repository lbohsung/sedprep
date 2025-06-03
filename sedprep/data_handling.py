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

import pandas as pd
import numpy as np

from scipy.interpolate import interp1d
import hashlib
from warnings import warn

from pymagglobal.utils import lmax2N

from hamstr_utils import (
    get_acc_mean,
    get_K_factor,
    get_brks_half_offset,
    get_levels_dict,
)

from constants import field_params as fp, mad_to_alpha_factors, REARTH
from utils import dsh_basis

# import deconvolve


def adjust_declination(dec):
    """
    Map the magnetic declination (D) values to the range (-180, 180].

    Parameters
    ----------
    dec : pandas.DataFrame.Series, list or numpy.array
        Magnetic declination (D) values.

    Returns
    -------
    pandas.DataFrame.Series, list or numpy.array
        Mapped declination values in the range (-180, 180].
    """
    if isinstance(dec, pd.Series):
        while np.max(dec) > 180:
            dec.where(dec <= 180, dec - 360, inplace=True)
        while np.min(dec) <= -180:
            dec.where(dec > -180, dec + 360, inplace=True)
    elif isinstance(dec, list):
        while np.max(dec) > 180:
            dec = [d if d <= 180 else d - 360 for d in dec]
        while np.min(dec) <= -180:
            dec = [d if d > -180 else d + 360 for d in dec]
    else:
        raise TypeError(
            "Input must be a pandas DataFrame Series," " list or numpy array."
        )
    return dec


def mad_to_alpha95(MAD, demag_steps=4):
    """
    Convert the Maximum Angular Deviation (MAD) into alpha95 values
    using the method described in https://doi.org/10.1093/gji/ggv451

    Parameters
    ----------
    MAD : pandas.DataFrame.Series, list or numpy.array
        The MAD values.
    demag_steps: int
        The number of demagnetization steps.

    Returns
    --------
    pandas.DataFrame.Series, list or numpy.array
        The alpha95 values, which represents the semi-angle of the
        95% confidence cone in Fisherian statistics.
    """
    alpha95 = mad_to_alpha_factors[str(demag_steps)] * MAD
    return alpha95


def alpha95_to_dD_dI(alpha95, inc, lat):
    """
    Convert the alpha95 parameter to declination (dD) and inclination (dI)
    uncertainties using a constant scaling factor of (57.3 / 140).

    Parameters
    ----------
    alpha95 : pandas.DataFrame.Series
        alpha95 values, representing
        the semi-angle of the 95% confidence cone in Fisherian statistics.
    inc : pandas.DataFrame.Series
        The inclination values in degrees [-90, 90].

    Returns
    --------
    tuple of pandas.DataFrame.Series
        dD and dI, representing the uncertainties in
        declination and inclination, respectively.
    """
    dI = (57.3 / 140) * alpha95
    # dD = dI / np.cos(np.deg2rad(inc))
    dD = np.where(
        inc.notna(),
        dI / np.cos(np.deg2rad(inc)),
        dI / np.cos(np.arctan(2 * np.tan(np.deg2rad(lat)))),
    )
    return dD, dI


def offset_dec(data, offsets):
    """
    Convert relative declinations of sub-sections into absolute declinations
    using individual offset.

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame containing declinations (D).
    offset : dictionary
        The offsets for the individual sub-sections as a dictionary.
        Keys correspond to sub-section names and values to offstes.

    Returns
    -------
    tuple
        A tuple containing two lists or floats, dD and dI, representing the
        uncertainties in declination and inclination, respectively.
    """
    data = data.copy()
    data.subs = data.subs.astype(str)
    if isinstance(offsets, (float, int, list)):
        data["D"] -= offsets
    else:
        for sub, group in data.groupby("subs"):
            data.loc[group.index, "D"] -= offsets[sub]
    data["D"] = adjust_declination(data["D"])
    return data


def I_dip(lat):
    """
    Compute the prior dipole inclination.

    Parameters
    ----------
    lat : pandas.DataFrame.Series, list or numpy.array
        latitude

    Returns
    --------
    tuple of pandas.DataFrame.Series, list or numpy.array
        prior dipole inclination
    """
    return np.rad2deg(np.arctan(2 * np.tan(np.deg2rad(lat))))


def shallow_inc(inc, f_shallow):
    """
    Transform inclination into shallower inclinations.

    Parameters
    ----------
    inc : pandas.DataFrame.Series, list or numpy.array
        The inclination values in degrees [-90, 90].
    f_shallow : float
        Shallowing factor

    Returns
    --------
    tuple of pandas.DataFrame.Series, list or numpy.array
        Shallowed inclinations
    """
    return np.rad2deg(np.arctan(f_shallow * np.tan(np.deg2rad(inc))))


def unshallow_inc(inc, f_shallow):
    """
    Transform shallowed inclination to inclinations.

    Parameters
    ----------
    inc : pandas.DataFrame.Series, list or numpy.array
        The inclination values in degrees [-90, 90].
    f_shallow : float
        Shallowing factor

    Returns
    --------
    tuple of pandas.DataFrame.Series, list or numpy.array
        Unshallowed inclinations
    """
    return np.rad2deg(np.arctan(np.tan(np.deg2rad(inc)) / f_shallow))


def t2d(data):
    """
    Compute an age-depth model for sediment data.

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame containing sediment data with columns
        't' (ages) and 'depth' (depths).

    Returns
    -------
    scipy.interpolate.interp1d
        A piecewise linear interpolation representing the age-depth model.
    """
    # Check for monotonicity
    if not (data.t.is_monotonic_decreasing or data.t.is_monotonic_increasing):
        raise ValueError("Ages must be strictly monotone.")
    if not (
        data.depth.is_monotonic_decreasing
        or data.depth.is_monotonic_increasing
    ):
        raise ValueError("Depths must be strictly monotone.")
    return interp1d(
        data.t, data.depth, kind="linear", fill_value="extrapolate"
    )


def d2t(data):
    """Inverse function to `t2d`. See `t2d` for more information."""
    # Check for monotonicity
    if not (data.t.is_monotonic_decreasing or data.t.is_monotonic_increasing):
        raise ValueError("Ages must be strictly monotone.")
    if not (
        data.depth.is_monotonic_decreasing
        or data.depth.is_monotonic_increasing
    ):
        raise ValueError("Depths must be strictly monotone.")
    return interp1d(
        data.depth, data.t, kind="linear", fill_value="extrapolate"
    )


def _calc_filehash(fname):
    """Helper routine to calculate sha256 hash for a file."""
    sha256_hash = hashlib.sha256()
    try:
        fh = open(fname, "rb")
    except (TypeError, OSError):
        fh = fname
    # Read and update hash string value in blocks of 4K
    for byte_block in iter(lambda: fh.read(4096), b""):
        sha256_hash.update(byte_block)
    # go back to initial position
    # (deal with the case that a stream was passed)
    fh.seek(0)
    return sha256_hash.digest()


def add_identifier(fname):
    """
    Add columns with unique identifiers to data frame.

    Parameters
    ----------
    fname : str
        File name of the data frame file.
    """
    data = pd.read_csv(fname)
    data["UID"] = data.index
    data["FID"] = _calc_filehash(fname)
    data.to_csv(fname, index=False)


# def clean_data(
#     data, bs_DI=None, bs_F=None, offsets=None, f_shallow=None, cal_fac=None
# ):
#     """
#     Clean and preprocess the given data.
#     This function allows for optional preprocessing steps, including
#     unshallowing inclination, applying offsets to subsections in declination,
#     and deconvolving declination and inclination. It returns a cleaned and
#     processed data DataFrame.

#     Parameters
#     ----------
#     data : pandas.DataFrame
#         The input data containing declination (D), inclination (I),
#         and associated uncertainties (dD and dI).

#     bs_DI : numpy.ndarray or list, optional
#         An array of shape (4,) or a list with 4 entries representing the
#         directional lock-in function parameters b_1 to b_4 used
#         for deconvolution.
#         If provided, deconvolution of declination and inclination is performed.
#         Default is None.

#     offsets : float or dictionary
#         If Float: Number representing the offset for the whole core
#         If dictionary: Keys correspond to subsection names and values to the
#         associated offsets applied to the subsections declinations.
#         Default is None.

#     f_shallow : float, optional
#         A scaling factor for unshallowing inclination data.
#         If provided, unshallowing is performed.
#         Default is None.

#     Returns
#     -------
#     pd.DataFrame
#         A cleaned and processed data DataFrame with updated declination (D)
#         and inclination (I) columns and associated updated uncertainties.

#     Example
#     -------
#     >>> data = pd.DataFrame({"depth": [34.0, 29.2, 25.0],
#     ...                      "t": [1628.901, 1679.669, 1724.09],
#     ...                      "dt": [300, 300, 300],
#     ...                      "lat": 60.151,
#     ...                      "lon": 13.055,
#     ...                      "D": [-10.067, -37.988, -33.412],
#     ...                      "I": [66.016, 66.090, 59.699],
#     ...                      "F": [58.55, 58.57, 58.66],
#     ...                      "dD": [1.791, 5.289, 2.161],
#     ...                      "dI": [0.478, 2.075, 0.478],
#     ...                      "dF": [13.13, 13.13, 12.1],
#     ...                      "type": "sediments",
#     ...                      "subs": ["A1", "A2", "A2"]})
#     >>> bs_DI = np.array([10, 10, 0, 10])
#     >>> offsets = -10
#     >>> clean_data(data, bs_DI=bs_DI, offsets=offsets, f_shallow=0.5)[
#     ...     ["D", "I", "F", "dD", "dI", "dF", "subs"]
#     ... ]
#               D          I      F         dD        dI     dF subs
#     0  2.483714  77.060772  58.55  20.709634  6.135548  13.13   A1
#     1  1.458709  77.317501  58.57  18.596219  5.678083  13.13   A2
#     2 -0.231032  77.449891  58.66  15.010178  4.430105  12.10   A2
#     >>> offsets = {"A1": 5, "A2": -10}
#     >>> clean_data(data, bs_DI=bs_DI, offsets=offsets, f_shallow=0.5)[
#     ...     ["D", "I", "F", "dD", "dI", "dF", "subs"]
#     ... ]
#                D          I      F         dD        dI     dF subs
#     0  -9.259935  77.060772  58.55  20.709634  6.135548  13.13   A1
#     1 -11.039868  77.317501  58.57  18.596219  5.678083  13.13   A2
#     2 -12.905582  77.449891  58.66  15.010178  4.430105  12.10   A2
#     """
#     data = data.copy()
#     if f_shallow is not None:
#         data["I"] = unshallow_inc(data.I, f_shallow)
#     if offsets is not None:
#         data = offset_dec(data, offsets)
#     if bs_DI is not None:
#         if sum(bs_DI) > 1:
#             data_sub = data.dropna(
#                 subset=["D", "I"], how="all"
#             ).reset_index(drop=True)
#             mean_D, cov_D, mean_I, cov_I = deconvolve.deconvolve(
#                 data_sub, bs_DI=bs_DI, quiet=False
#             )
#             data_sub["D"] = mean_D
#             data_sub["dD"] = np.sqrt(data_sub["dD"] ** 2 + np.diag(cov_D))
#             data_sub["I"] = mean_I
#             data_sub["dI"] = np.sqrt(data_sub["dI"] ** 2 + np.diag(cov_I))
#             data = data_sub[["t", "D", "I", "dD", "dI"]].merge(
#                 data.drop(["D", "I", "dD", "dI"], axis=1), how="right", on="t"
#             )
#     return data


def read_arch_data(fname, rejection_lists=None, update_mex=True):
    """Read a file produced by the GEOMAGIA database and return a
    `pandas.Dataframe`

    Parameters
    ----------
    fname : str
        Path to a file containing paleomagnetic data. It will be assumed
        to point to a file from GEOMAGIA.
    rejection_lists : array or list of arrays, optional
        An array (or list of arrays) of types {'D', 'I', 'F'}, indices, and
        file hashes, which uniquely identify records to be dropped from the
        data. (Outlier rejection). Default is None.
    update_mex : bool, optional
        Whether to update some Mexico data according to personal communication
        with Ahmed Nasser Mahgoub Ahmed (2022).

    Returns
    -------
    DataFrame
        A dataframe containing the data. The following keys are included:
            t : the measurement dates in yrs
            dt : the dating uncertainties in yrs
            rad : the radius of the measurement location in km
            colat : the colatitude of the measurement location in deg
            lat : the latitude of the measurement location in deg
            lon : the longitude of the measurement location in deg
            D : declination measurement in deg
            dD : error of the declination measurement in deg
            I : incination measurement in deg
            dI : error of the inlcination measurement in deg
            F : intensity measurement in uT
            dF : error of the intensity measurement in uT
            UID : unique ID identifying each record in a dataset
            FID : sha256 hash of the file the record stems from
    """
    # Calculate unique hash for file
    filehash = _calc_filehash(fname)

    # Missing values are indicated by either one of
    na = ("9999", "999", "999.9", "nan", "-999", "-9999")
    # Read data as DataFrame
    dat = pd.read_csv(
        fname,
        usecols=[
            "Age[yr.AD]",
            "Sigma-ve[yr.]",
            "Sigma+ve[yr.]",
            "Ba[microT]",
            "SigmaBa[microT]",
            "Dec[deg.]",
            "Inc[deg.]",
            "Alpha95[deg.]",
            "SiteLat[deg.]",
            "SiteLon[deg.]",
            "UID",
        ],
        na_values={
            "Sigma-ve[yr.]": (-1),
            "Sigma+ve[yr.]": (-1),
            "Ba[microT]": na,
            "SigmaBa[microT]": na,
            "Dec[deg.]": na,
            "Inc[deg.]": na,
            "Alpha95[deg.]": na,
        },
        header=1,
        sep=",",
        skipinitialspace=True,
    )
    # Rename columns
    ren_dict = {
        "Age[yr.AD]": "t",
        "Sigma-ve[yr.]": "dt_lo",
        "Sigma+ve[yr.]": "dt_up",
        "Ba[microT]": "F",
        "SigmaBa[microT]": "dF",
        "Dec[deg.]": "D",
        "Inc[deg.]": "I",
        "SiteLat[deg.]": "lat",
        "SiteLon[deg.]": "lon",
    }

    dat.rename(ren_dict, inplace=True, axis="columns")
    dat.dropna(subset=["lat", "lon", "t"], inplace=True)

    dat["FID"] = filehash

    if rejection_lists is not None:
        if not isinstance(rejection_lists, list):
            rejection_lists = [rejection_lists]
        for to_reject in rejection_lists:
            n_rej = 0
            for tp, uid, fid in to_reject[:, :3]:
                idx = dat.query(f"UID == {uid} and FID == {fid}").index
                # prevent checking the dataframe columns for keys that don't
                # exist (if FID does agree, everything should be there)
                if len(idx):
                    dat.loc[idx, [tp]] = np.nan
                    n_rej += len(idx)
            print(f"Rejected {n_rej} outliers.")

    dat.dropna(subset=["D", "I", "F"], inplace=True, how="all")
    # Map declination to [-180:180] degrees
    dat["D"] = dat["D"].where(dat["D"] <= 180, dat["D"] - 360)
    dat["lon"] = dat["lon"].where(dat["lon"] > 0, other=dat["lon"] + 360)

    # Fill missing intensity errors with 8.25 uT
    dat["dF"] = dat["dF"].where(
        dat["F"].isna() | dat["dF"].notna(), other=8.25
    )

    # Standard deviation for inclination
    dat["dI"] = dat["Alpha95[deg.]"]  # Just a copy
    # Fill missing inclination errors with 4.5 degree
    dat["dI"] = dat["dI"].where(dat["I"].isna() | dat["dI"].notna(), other=4.5)
    dat["dI"] *= 57.3 / 140.0

    # Standard deviation for declination
    dat["dD"] = dat["Alpha95[deg.]"]  # Just a copy

    # Find records of only Declination, since this causes trouble in the error
    # calculation
    cond = dat["D"].notna() & dat["I"].isna()
    # Get the corresponding indices
    ind = dat.where(cond).dropna(how="all").index
    # If there are indices in the array, throw a warning.
    if ind.size != 0:
        warn(
            f"Records with indices {ind.values} contain declination, but not"
            f" inclination! The errors need special treatment!\n"
            f"To be able to use the provided data, these"
            f" records have been dropped from the output.",
            UserWarning,
        )

    dat.drop(dat.where(cond).dropna(how="all").index, inplace=True)

    # Fill missing declination errors with 4.5 degree
    dat["dD"] = dat["dD"].where(dat["D"].isna() | dat["dD"].notna(), other=4.5)
    dat["dD"] *= 57.3 / 140.0 / np.cos(np.deg2rad(dat["I"]))
    # Add radius and colatitude (for convenience)
    dat["rad"] = REARTH
    dat["colat"] = 90 - dat["lat"]  # colatitude in [deg]

    dat["dt"] = np.max([dat["dt_lo"], dat["dt_up"]], axis=0)
    dat["dt"] = dat["dt"].where(dat["dt"].notna(), other=100)
    dat["dt"] = dat["dt"].where(dat["dt"] > 10, other=10)

    # SigmaAgeID == 6 means that the error is reported as two standard devs.
    try:
        dat["dt"].where(dat["SigmaAgeID"] != 6, other=dat["dt"] / 2.0)
    except KeyError:
        # SigmaAgeID is not included in the abriged GEOMAGIA form. In this case
        # we have to take dt as given...
        pass

    # Update and remove Mexico data
    if update_mex:
        rem_ids = [11237, 2773, 6891, 13149]
        for rem in rem_ids:
            dat.drop(
                index=dat.query(f"UID == {rem}").index,
                inplace=True,
            )

        update_df = pd.DataFrame(
            data={
                "UID": [
                    13153,
                    2768,
                    2769,
                    11967,
                    6893,
                    11966,
                    2770,
                    6892,
                    13086,
                    13118,
                    11992,
                ],
                "upd_t": [
                    -7550,
                    -8523,
                    -7450,
                    -10000,
                    -10000,
                    -5707,
                    1250,
                    1250,
                    8,
                    8,
                    1545,
                ],
                "upd_dt": [
                    422,
                    800,
                    270,
                    338,
                    338,
                    184,
                    5,
                    5,
                    62,
                    62,
                    94,
                ],
            },
        )
        for _, row in update_df.iterrows():
            idx = dat.query(f"UID == {row['UID']}").index
            dat.loc[idx, "t"] = row["upd_t"]
            dat.loc[idx, "dt"] = row["upd_dt"]

    dat.reset_index(inplace=True)
    dat["type"] = "arch_data"

    # Return the relevant columns
    return dat[
        [
            "t",
            "dt",
            "rad",
            "colat",
            "lat",
            "lon",
            "D",
            "dD",
            "I",
            "dI",
            "F",
            "dF",
            "type",
            "UID",
            "FID",
        ]
    ]


class Data:
    def __init__(self, data, adm_data=None):
        arch = "depth" not in data.columns
        df = data.copy()
        df = df.dropna(subset=["D", "I", "F"], how="all")
        if not arch:
            df = df.sort_values(by="depth", ascending=True)
            self.depth = df.depth.values
            self.colat = 90 - df.lat.values[0]
        df = df.reset_index(drop=True)
        self.n = len(df)
        self.idx_D = np.asarray(df[~np.isnan(df.D)].index)
        self.idx_I = np.asarray(df[~np.isnan(df.I)].index)
        self.idx_F = np.asarray(df[~np.isnan(df.F)].index)
        if arch:
            inp = [90 - df.lat, df.lon, np.full(len(df), REARTH)]
        else:
            df["D"] = adjust_declination(df["D"])
            subcores = {}
            # print(df.groupby("subs").D.mean())
            for name in np.unique(df.subs.astype(str)):
                _df = df.iloc[self.idx_D].reset_index(drop=True)
                _df = _df[_df.subs == name]
                mu = _df["D"].mean()
                vals = np.zeros(len(df["D"]), dtype=bool)
                vals[_df.index] = True
                subcores[name] = (vals.copy(), mu)
            self.subcores = subcores

            adm_data = adm_data.sort_values(by="depth", ascending=True)
            adm_data = adm_data.reset_index(drop=True)
            self.adm_data = adm_data
            # self.adm_depth = adm_data["depth"].values
            # self.adm_t = 1950 - adm_data["t"].values
            # self.adm_dt = adm_data["dt"].values
            # top_depth = 0
            # bottom_depth = max(max(self.adm_depth), max(self.depth))
            # K_fine_1 = int(bottom_depth - top_depth)
            # median_depth_diff = np.median(
            #     np.diff(np.sort(np.unique(self.adm_depth)))
            # )
            # K_fine_2 = np.round(16 * K_fine_1 / median_depth_diff)
            # K_fine = int(min(K_fine_1, K_fine_2, 900))
            # K_factor = get_K_factor(K_fine)
            # brks = get_brks_half_offset(K_fine, K_factor)
            # self.levels_dict = get_levels_dict(brks)
            # n_lvls = self.levels_dict["n_levels"]
            # K_fine = self.levels_dict[f"{n_lvls}"]["nK"]
            # self.delta_c = (bottom_depth - top_depth) / K_fine
            # c_depth_bottom = [
            #     self.delta_c * c + top_depth
            #     for c in list(range(1, K_fine + 1))
            # ]
            # self.c_depth_top = np.concatenate(
            #     [[top_depth], c_depth_bottom[: K_fine - 1]]
            # )
            # self.modelled_depths = np.concatenate(
            #     [[self.c_depth_top[0]], c_depth_bottom]
            # )
            # self.which_c = [
            #     np.argmax((c_depth_bottom < d) * (c_depth_bottom - d))
            #     for d in self.adm_depth
            # ]
            # self.acc_mean = get_acc_mean(self.adm_t, self.adm_depth)
            # self.age0_bound = -70
            # self.age0 = 1950 - adm_data["t"][adm_data["depth"] == 0].item()
            # self.dage0 = adm_data["dt"][adm_data["depth"] == 0].item()
            inp = [90 - df.lat[0], df.lon[0], REARTH]
        self.base = dsh_basis(fp["lmax"], np.array(inp))
        if arch:
            self.base = self.base.reshape(lmax2N(fp["lmax"]), len(df), 3)
            self.t = df.t.values
            self.dt = df.dt.values**2
        self.out_D = df.loc[self.idx_D, "D"].to_numpy()
        self.out_I = df.loc[self.idx_I, "I"].to_numpy()
        self.out_F = df.loc[self.idx_F, "F"].to_numpy()
        self.errs_D = df.loc[self.idx_D, "dD"].to_numpy() ** 2
        self.errs_I = df.loc[self.idx_I, "dI"].to_numpy() ** 2
        self.errs_F = df.loc[self.idx_F, "dF"].to_numpy() ** 2
