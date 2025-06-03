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
import torch
from scipy.interpolate import interp1d
import hashlib
from warnings import warn

from pymagglobal.utils import lmax2N

from sedprep.constants import (
    field_params,
    mad_to_alpha_factors,
    REARTH,
    device,
    dtype,
)
from sedprep.utils import dsh_basis, nez2dif
from sedprep import deconvolve


def adjust_declination(dec):
    """
    Map the magnetic declination (D) values to the range (-180, 180].

    Parameters
    ----------
    dec : pandas.DataFrame.Series, list, numpy.array or torch.tensor
        Magnetic declination (D) values.

    Returns
    -------
    pandas.DataFrame.Series, list, numpy.array or torch.tensor
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
    elif torch.is_tensor(dec):
        if dec.numel() > 0:
            while torch.max(dec) > 180:
                dec = torch.where(dec <= 180, dec, dec - 360)
            while torch.min(dec) <= -180:
                dec = torch.where(dec > -180, dec, dec + 360)
        else:
            return dec
    else:
        raise TypeError(
            "Input must be a pandas DataFrame Series,"
            " list, numpy array or torch tensor."
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


def max_lit(acc_rates, max_lid, delta_t):
    """
    Calculate the maximal lock-in time corresponding to the passed maximal
    lock-in depth, using the derivative of the age-depth model.

    Parameters
    ----------
    t2d : scipy.interpolate.CubicHermiteSpline
        The age-depth model represented as a piecewise linear interpolation.
    max_lid : float
        The maximal lock-in depth.
    delta_t : float
        The time step used in the Kalman filter.

    Returns
    -------
    float
        The maximal lock-in time, calculated based on the age-depth model,
        the specified maximal lock-in depth, and the time step.
    """
    if torch.is_tensor(max_lid):
        max_lid = max_lid.cpu().detach().numpy()
    max_lit = max_lid / np.min(np.abs(acc_rates))
    max_lit = np.round(max_lit / delta_t + 0.5)
    max_lit *= delta_t * 1.0
    return max_lit


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


def clean_data(
    data, bs_DI=None, bs_F=None, offsets=None, f_shallow=None, cal_fac=None
):
    """
    Clean and preprocess the given data.
    This function allows for optional preprocessing steps, including
    unshallowing inclination, applying offsets to subsections in declination,
    and deconvolving declination and inclination. It returns a cleaned and
    processed data DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data containing declination (D), inclination (I),
        and associated uncertainties (dD and dI).

    bs_DI : numpy.ndarray or list, optional
        An array of shape (4,) or a list with 4 entries representing the
        directional lock-in function parameters b_1 to b_4 used
        for deconvolution.
        If provided, deconvolution of declination and inclination is performed.
        Default is None.

    bs_F : numpy.ndarray or list, optional
        An array of shape (4,) or a list with 4 entries representing the
        intensity lock-in function parameters b_1 to b_4 used
        for deconvolution.
        If provided, deconvolution of declination and inclination is performed.
        Default is None.

    offsets : float or dictionary
        If Float: Number representing the offset for the whole core
        If dictionary: Keys correspond to subsection names and values to the
        associated offsets applied to the subsections declinations.
        Default is None.

    f_shallow : float, optional
        A scaling factor for unshallowing inclination data.
        If provided, unshallowing is performed.
        Default is None.

    Returns
    -------
    pd.DataFrame
        A cleaned and processed data DataFrame with updated declination (D)
        and inclination (I) columns and associated updated uncertainties.

    Example
    -------
    >>> data = pd.DataFrame({"depth": [34.0, 29.2, 25.0],
    ...                      "t": [1628.901, 1679.669, 1724.09],
    ...                      "dt": [300, 300, 300],
    ...                      "lat": 60.151,
    ...                      "lon": 13.055,
    ...                      "D": [-10.067, -37.988, -33.412],
    ...                      "I": [66.016, 66.090, 59.699],
    ...                      "F": [58.55, 58.57, 58.66],
    ...                      "dD": [1.791, 5.289, 2.161],
    ...                      "dI": [0.478, 2.075, 0.478],
    ...                      "dF": [13.13, 13.13, 12.1],
    ...                      "type": "sediments",
    ...                      "subs": ["A1", "A2", "A2"]})
    >>> bs_DI = np.array([10, 10, 0, 10])
    >>> offsets = -10
    >>> clean_data(data, bs_DI=bs_DI, offsets=offsets, f_shallow=0.5)[
    ...     ["D", "I", "F", "dD", "dI", "dF", "subs"]
    ... ]
              D          I      F         dD        dI     dF subs
    0  2.483714  77.060772  58.55  20.709634  6.135548  13.13   A1
    1  1.458709  77.317501  58.57  18.596219  5.678083  13.13   A2
    2 -0.231032  77.449891  58.66  15.010178  4.430105  12.10   A2
    >>> offsets = {"A1": 5, "A2": -10}
    >>> clean_data(data, bs_DI=bs_DI, offsets=offsets, f_shallow=0.5)[
    ...     ["D", "I", "F", "dD", "dI", "dF", "subs"]
    ... ]
               D          I      F         dD        dI     dF subs
    0  -9.259935  77.060772  58.55  20.709634  6.135548  13.13   A1
    1 -11.039868  77.317501  58.57  18.596219  5.678083  13.13   A2
    2 -12.905582  77.449891  58.66  15.010178  4.430105  12.10   A2
    """
    data = data.copy()
    if f_shallow is not None:
        data["I"] = unshallow_inc(data.I, f_shallow)
    if offsets is not None:
        data = offset_dec(data, offsets)
    if bs_DI is not None:
        if sum(bs_DI) > 1:
            data_sub = data.dropna(
                subset=["D", "I"], how="all"
            ).reset_index(drop=True)
            mean_D, cov_D, mean_I, cov_I = deconvolve.deconvolve(
                data_sub, bs_DI=bs_DI, quiet=False
            )
            data_sub["D"] = mean_D
            data_sub["dD"] = np.sqrt(data_sub["dD"] ** 2 + np.diag(cov_D))
            data_sub["I"] = mean_I
            data_sub["dI"] = np.sqrt(data_sub["dI"] ** 2 + np.diag(cov_I))
            data = data_sub[["t", "D", "I", "dD", "dI"]].merge(
                data.drop(["D", "I", "dD", "dI"], axis=1), how="right", on="t"
            )
    if cal_fac is not None:
        data["F"] *= cal_fac
        data["dF"] *= cal_fac
    if bs_F is not None:
        if sum(bs_F) > 1:
            data_sub = data.dropna(subset=["F"], how="all").reset_index(drop=True)
            mean_F, cov_F = deconvolve.deconvolve(data_sub, bs_F=bs_F, quiet=False)
            data_sub["F"] = mean_F
            data_sub["dF"] = np.sqrt(data_sub["dF"] ** 2 + np.diag(cov_F))
            data = data_sub[["t", "F", "dF"]].merge(
                data.drop(["F", "dF"], axis=1), how="right", on="t"
            )
    return data


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
    dat["dI"] = dat["dI"].where(
        dat["I"].isna() | dat["dI"].notna(), other=4.5
    )
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
    dat["dD"] = dat["dD"].where(
        dat["D"].isna() | dat["dD"].notna(), other=4.5
    )
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


class Chunk:
    def __init__(self, chunk, arch):
        self.chunk = chunk.copy()
        self.lat = torch.tensor(chunk.lat.values, device=device, dtype=dtype)
        self.n = len(chunk)
        self.idx_D_abs = chunk[~np.isnan(chunk.D)].index
        self.idx_I_abs = chunk[~np.isnan(chunk.I)].index
        self.idx_F_abs = chunk[~np.isnan(chunk.F)].index
        self.idx_D_rel = self.idx_D_abs - np.min(chunk.index)
        self.idx_I_rel = self.idx_I_abs - np.min(chunk.index)
        self.idx_F_rel = self.idx_F_abs - np.min(chunk.index)
        self.n_D = self.idx_D_abs.size
        self.n_DI = self.n_D + self.idx_I_abs.size
        self.n_DIF = self.n_DI + self.idx_F_abs.size
        std_T = np.concatenate(
            (
                chunk["dt"].loc[self.idx_D_abs],
                chunk["dt"].loc[self.idx_I_abs],
                chunk["dt"].loc[self.idx_F_abs],
            )
        )
        std_err = np.concatenate(
            (
                chunk["dD"].loc[self.idx_D_abs],
                chunk["dI"].loc[self.idx_I_abs],
                chunk["dF"].loc[self.idx_F_abs],
            )
        )
        outputs = np.concatenate(
            (
                chunk["D"].loc[self.idx_D_abs],
                chunk["I"].loc[self.idx_I_abs],
                chunk["F"].loc[self.idx_F_abs],
            )
        )
        if arch:
            self.index = chunk.index
        else:
            self.subs = chunk.dropna(subset="D").subs
        self.errs_T = torch.from_numpy(std_T**2).to(device=device, dtype=dtype)
        self.errs = torch.from_numpy(std_err**2).to(device=device, dtype=dtype)
        self.outputs = torch.from_numpy(outputs).to(device=device, dtype=dtype)


class ChunkedData:
    def __init__(self, fname, lmax, delta_t, start=2000, end=-6000, adm_d2t=None):
        if isinstance(fname, pd.DataFrame):
            data = fname.copy()
        else:
            raise ValueError("Need to pass a DataFrame!")
        arch = data.type[0] == "arch_data"
        data.sort_values("t", ascending=False, inplace=True)
        data.dropna(subset=["D", "I", "F"], inplace=True, how="all")
        data["D"] = adjust_declination(data["D"])
        data.reset_index(drop=True, inplace=True)
        if arch:
            inp = [90 - data.lat, data.lon, np.full(len(data), REARTH)]
        else:
            inp = [90 - data.lat[0], data.lon[0], REARTH]
            if adm_d2t is not None:
                # Check for monotonicity
                if not (
                    np.all(np.diff(adm_d2t.x) > 0)
                    or np.all(np.diff(adm_d2t.x) < 0)
                ):
                    raise ValueError("Depths must be strictly monotone.")
                if not (
                    np.all(np.diff(adm_d2t.y) > 0)
                    or np.all(np.diff(adm_d2t.y) < 0)
                ):
                    raise ValueError("Ages must be strictly monotone.")
                idx = adm_d2t.x < max(data.depth)
                self.t2d = interp1d(
                    adm_d2t.y[idx], adm_d2t.x[idx], kind='linear', fill_value="extrapolate"
                )
            else:
                self.t2d = t2d(data)
            self.acc_rates = np.diff(self.t2d.y) / np.diff(self.t2d.x)
            self.n_DI = max(data[["D", "I"]].count())
            self.n_F = data["F"].count()
        self.base = torch.from_numpy(dsh_basis(lmax, np.array(inp))).to(
            device=device, dtype=dtype
        )

        self.delta_t = delta_t
        self.start = start
        if arch:
            self.base = self.base.reshape(lmax2N(lmax), len(data), 3)
        else:
            # calibrate RPI with prior calibration factor
            # derived from axial dipole assumption
            # to bring them in the same range as archeo intensities
            self.axial_cal_fac = (
                1
                if np.isnan(np.mean(data["F"]))
                else float(
                    nez2dif(*(field_params["gamma"] * self.base[0, :]))[2]
                    / np.mean(data["F"])
                )
            )
            data["F"] *= self.axial_cal_fac
            data["dF"] *= self.axial_cal_fac
        self.chunks = []
        while start > end:
            chunk = data[(data.t < start) & (data.t >= start - delta_t)]
            self.chunks.append(Chunk(chunk, arch))
            start -= delta_t


def chunk_data(
    sed_data, arch_data, lmax, delta_t, start=2000, end=-6000, adm_d2t=None
):
    cdat_sed = ChunkedData(sed_data, lmax, delta_t, start, end, adm_d2t=adm_d2t)
    cdat_arch = ChunkedData(arch_data, lmax, delta_t, start, end)
    return cdat_sed, cdat_arch


# def t2d_old(data):
#     """
#     Compute an age-depth model for sediment data.

#     Parameters
#     ----------
#     data : pandas.DataFrame
#         A DataFrame containing sediment data with columns
#         't' (ages) and 'depth' (depths).

#     Returns
#     -------
#     scipy.interpolate.CubicHermiteSpline
#         A CubicHermiteSpline object representing the age-depth model.
#     """
#     # Check if sediment data exists in the input DataFrame
#     idx_sed = data.query('type!="arch_data"').index
#     for i in range(1, 4):
#         if len(idx_sed):
#             # Extract ages and depths from the DataFrame and process them
#             ads = np.copy(
#                 data.iloc[idx_sed]
#                 .sort_values("depth")[["t", "depth"]]
#                 .to_numpy()
#                 .T,
#             )
#             # therefore flip the unique (i.e. sorted) result
#             ads = np.flip(np.unique(ads, axis=1), axis=1)
#             ages = ads[0]
#             depths = ads[1]
#             # Perform linear regression to find a linear age-depth model
#             ads = np.unique(ads, axis=1)
#             X = np.vstack((np.ones_like(ads[1]), ads[1]))
#             # a, m = ads[0] @ X.T @ np.linalg.inv(X @ X.T)
#             idx_u = int((len(ads[0]) * (i-1)/i))
#             X_u = X[:, idx_u:]
#             a_u, m_u = ads[0][idx_u:] @ X_u.T @ np.linalg.inv(X_u @ X_u.T)
#             idx_l = int((len(ads[0]) * 1/i))
#             X_l = X[:, :idx_l]
#             a_l, m_l = ads[0][:idx_l] @ X_l.T @ np.linalg.inv(X_l @ X_l.T)
#             # Insert boundary points for extrapolation
#             depths = np.insert(depths, 0, np.array(-1000))[::-1]
#             ages = np.insert(ages, 0, np.array(a_u - m_u * 1000))[::-1]
#             depths = np.insert(depths, 0, np.array(1e8))
#             ages = np.insert(ages, 0, np.array(a_l + m_l * 1e8))
#         else:
#             # If no sediment data exists, create a simple linear adm
#             def _t2d(t):
#                 return -0.05 * (t - 1950)

#             ages = np.flip(np.unique(data["t"].to_numpy()))
#             depths = _t2d(ages)
#             # Insert boundary points for extrapolation
#             depths = np.insert(depths, 0, np.array(-1000))[::-1]
#             ages = np.insert(ages, 0, np.array(21950.0))[::-1]
#             depths = np.insert(depths, 0, np.array(1e8))
#             ages = np.insert(ages, 0, np.array(-199998050.0))
#         # Calculate the slope (m) at each depth segment
#         deltas = [
#             0
#             if depths[k + 1] == depths[k]
#             else (depths[k + 1] - depths[k]) / (ages[k + 1] - ages[k])
#             for k in range(len(depths) - 1)
#         ]
#         ms = [
#             deltas[0]
#             if k == 0
#             else (
#                 (deltas[k - 1] + deltas[k]) / 2
#                 if np.sign(deltas[k - 1]) == np.sign(deltas[k])
#                 else 0
#             )
#             for k in range(len(depths) - 1)
#         ]
#         ms.append(deltas[-1])
#         for k in range(len(depths) - 1):
#             if deltas[k] == 0:
#                 ms[k] = 0
#                 ms[k + 1] = 0
#         # To prevent overshoot and ensure monotonicity, at least one of the
#         # following three conditions must be met:
#         alphas = np.array([m / d for m, d in zip(ms[:-1], deltas)])
#         betas = np.array([m / d for m, d in zip(ms[1:], deltas)])
#         ps = 2 * alphas + betas - 3
#         qs = alphas + 2 * betas - 3
#         phis = alphas - ps**2 / (3 * (alphas + betas - 2))
#         phis = phis[~np.isnan(phis)]
#         phis = phis[~np.isinf(phis)]
#         cond1 = np.all(phis > 0)
#         cond2 = np.all(ps <= 0)
#         cond3 = np.all(qs <= 0)
#         if not np.any([cond1, cond2, cond3]):
#             warn(
#                 "Age-depth model is not monotone.\n"
#                 "None of the conditions described in "
#                 "https://en.wikipedia.org/wiki/Monotone_cubic_interpolation "
#                 "are met.\n"
#                 "A new age-depth model with an adjusted slope of the "
#                 "extrapolation is calculated."
#             )
#         else:
#             break
#     # Return the age-depth model as a Cubic Hermite Spline
#     return CubicHermiteSpline(ages, depths, ms)


# def d2t_old(data):
#     """Inverse function to `t2d`. See `t2d` for more information."""
#     ads = np.copy(
#         data.sort_values("depth")[["t", "depth"]].to_numpy().T,
#     )
#     for i in range(1, 4):
#         ads = np.unique(ads, axis=1)
#         ages = ads[0]
#         depths = ads[1]
#         X = np.vstack((np.ones_like(ads[1]), ads[1]))
#         # a, m = ads[0] @ X.T @ np.linalg.inv(X @ X.T)
#         idx_u = int((len(ads[0]) * (i-1)/i))
#         X_u = X[:, idx_u:]
#         a_u, m_u = ads[0][idx_u:] @ X_u.T @ np.linalg.inv(X_u @ X_u.T)
#         idx_l = int((len(ads[0]) * 1/i))
#         X_l = X[:, :idx_l]
#         a_l, m_l = ads[0][:idx_l] @ X_l.T @ np.linalg.inv(X_l @ X_l.T)
#         depths = np.append(depths, np.array(-1000))[::-1]
#         ages = np.append(ages, np.array(a_u - m_u * 1000))[::-1]
#         depths = np.append(depths, np.array(1e8))
#         ages = np.append(ages, np.array(a_l + m_l * 1e8))

#         deltas = [
#             0
#             if ages[k + 1] == ages[k]
#             else (ages[k + 1] - ages[k]) / (depths[k + 1] - depths[k])
#             for k in range(len(ages) - 1)
#         ]
#         ms = [
#             deltas[0]
#             if k == 0
#             else (
#                 (deltas[k - 1] + deltas[k]) / 2
#                 if np.sign(deltas[k - 1]) == np.sign(deltas[k])
#                 else 0
#             )
#             for k in range(len(depths) - 1)
#         ]
#         ms.append(deltas[-1])
#         for k in range(len(ages) - 1):
#             if deltas[k] == 0:
#                 ms[k] = 0
#                 ms[k + 1] = 0
#         # To prevent overshoot and ensure monotonicity, at least one of the
#         # following three conditions must be met:
#         alphas = np.array([m / d for m, d in zip(ms[:-1], deltas)])
#         betas = np.array([m / d for m, d in zip(ms[1:], deltas)])
#         ps = 2 * alphas + betas - 3
#         qs = alphas + 2 * betas - 3
#         phis = alphas - ps**2 / (3 * (alphas + betas - 2))
#         phis = phis[~np.isnan(phis)]
#         phis = phis[~np.isinf(phis)]
#         cond1 = np.all(phis > 0)
#         cond2 = np.all(ps <= 0)
#         cond3 = np.all(qs <= 0)
#         if not np.any([cond1, cond2, cond3]):
#             warn(
#                 "Age-depth model is not monotone.\n"
#                 "None of the conditions described in "
#                 "https://en.wikipedia.org/wiki/Monotone_cubic_interpolation "
#                 "are met.\n"
#                 "A new age-depth model with an adjusted slope of the "
#                 "extrapolation is calculated."
#             )
#         else:
#             break
#     return CubicHermiteSpline(depths, ages, ms)
