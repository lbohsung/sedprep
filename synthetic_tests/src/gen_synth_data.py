import numpy as np
import pandas as pd
from scipy import stats, interpolate, integrate
from tqdm import tqdm

from pymagglobal.utils import lmax2N

from sedprep.utils import nez2dif, lif, sample_Fisher, dsh_basis
from sedprep.constants import field_params, REARTH
from sedprep.data_handling import shallow_inc


def add_noise(NEZ, ddec=None, dinc=None, dint=None):
    """
    Add noise to the observed NEZ components (north, east, center).

    Parameters
    ----------
    NEZ : numpy.ndarray
        Observed NEZ components.
    ddec : numpy.ndarray or None, optional
        Array of declination uncertainties.
    dinc : numpy.ndarray or None, optional
        Array of inclination uncertainties.
    dint : numpy.ndarray or None, optional
        Array of intensity uncertainties.

    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Arrays of observed DIF (Declination, Inclination, Intensity)
        and their respective uncertainties (dD, dI, dF).
    """
    n = NEZ.shape[0]
    ddec = np.full(n, np.nan) if ddec is None else ddec
    dinc = np.full(n, np.nan) if dinc is None else dinc
    dint = np.full(n, np.nan) if dint is None else dint
    D, I, F = (np.empty(n), np.empty(n), np.empty(n))
    dD, dI, dF = (np.empty_like(D), np.empty_like(I), np.empty_like(F))
    dint_0 = 8.25
    kappa_0 = 650
    for it, (mu, dd, di, df) in enumerate(zip(NEZ, ddec, dinc, dint)):
        if np.isnan(di) or di == 0:
            kappa = kappa_0
            dI[it] = 57.3 / np.sqrt(kappa)
        else:
            kappa = (57.3 / di) ** 2
            dI[it] = di
        samp = sample_Fisher(1, mu=mu, kappa=kappa)
        D[it] = np.rad2deg(np.arctan2(samp[1], samp[0]))
        I[it] = np.rad2deg(
            np.arctan2(samp[2], np.sqrt(samp[0] ** 2 + samp[1] ** 2))
        )
        dD[it] = (
            dI[it] / np.cos(np.deg2rad(I[it]))
            if np.isnan(dd) or dd == 0
            else dd
        )
        dF[it] = dint_0 if (np.isnan(df) or df <= 0) else df
        f = nez2dif(*mu, be=np)[2]
        b = f / dF[it] ** 2
        a = f * b
        try:
            F[it] = stats.gamma.rvs(a=a, scale=1.0 / b)
        except ValueError as e:
            print(e)
            print(a, b)
            print(dF[it])
    DIF = np.array([D, I, F]).T
    return DIF, dD, dI, dF


def gen_data(
    t,
    lat,
    lon,
    field_spline,
    dt,
    ddec=None,
    dinc=None,
    dint=None,
    depth=None,
    with_noise=True,
):
    """
    Generate synthetic data for given times and locations.

    Parameters
    ----------
    t : numpy.ndarray
        Time array.
    lat : numpy.ndarray or float
        Latitude or array of latitudes.
    lon : numpy.ndarray or float
        Longitude or array of longitudes.
    field_spline : scipy.interpolate.BSpline
        Interpolating spline for the geomagnetic field.
    dt : numpy.ndarray
        Time uncertainties.
    ddec : numpy.ndarray or None, optional
        Array of declination uncertainties.
    dinc : numpy.ndarray or None, optional
        Array of inclination uncertainties.
    dint : numpy.ndarray or None, optional
        Array of intensity uncertainties.
    depth : numpy.ndarray or None, optional
        Depth array.
    with_noise : bool, optional
        Add noise to the generated data using Fisherian statistics,
        by default True.

    Returns
    -------
    pandas.DataFrame, numpy.ndarray
        DataFrame containing generated synthetic data and
        an array containing the NEZ components.
    """
    n = len(t)
    z = REARTH * np.ones((4, n))
    z[0, :] = 90 - lat
    z[1, :] = lon
    z[3, :] = t

    NEZ = np.array(
        [
            dsh_basis(field_params["lmax"], z[:, i]).T
            @ field_spline(z[3, i])[: lmax2N(field_params["lmax"])]
            for i in range(n)
        ]
    )
    if with_noise:
        DIF, dD, dI, dF = add_noise(NEZ, ddec, dinc, dint)
    else:
        DIF = np.array([nez2dif(*nez, be=np) for nez in NEZ])
        dD, dI, dF = 0, 0, 0

    data = pd.DataFrame(
        data={
            "t": t,
            "dt": dt,
            "lat": lat,
            "lon": lon,
            "D": DIF[:, 0],
            "I": DIF[:, 1],
            "F": DIF[:, 2],
            "dD": dD,
            "dI": dI,
            "dF": dF,
        }
    )
    if depth is not None:
        data["depth"] = depth
        data["type"] = "sediments"
    else:
        data["type"] = "arch_data"
    return data, NEZ


def distort_data(
    data_clean,
    NEZ_clean,
    ddec=None,
    dinc=None,
    dint=None,
    bs_DI=[0, 0, 0, 0],
    bs_F=[0, 0, 0, 0],
    offsets={"A1": 0, "A2": 0},
    f_shallow=1,
    cal_fac=1,
    subsections={"A1": np.arange(0, 70, 1), "A2": np.arange(70, 119, 1)},
):
    """
    This function can be used to transform absolute declinations and
    intensities of a clean synthetic data set into relative values and to
    distort the data simulating inclination shallowing and pDRM effects.

    Parameters
    ----------
    data_clean : pandas.DataFrame
        Clean synthetic data.
    NEZ_clean : numpy.ndarray
        Clean synthetic NEZ components.
    ddec : numpy.ndarray or None, optional
        Array of declination uncertainties.
    dinc : numpy.ndarray or None, optional
        Array of inclination uncertainties.
    dint : numpy.ndarray or None, optional
        Array of intensity uncertainties.
    offsets : dict, optional
        Dictionary of declination offsets for different sub-sections.
    f_shallow : float, optional
        Inclination shallowing factor used to simulate inclination shallowing.
    cal_fac : dict, optional
        Dictionary of calibration factors for intensities of different
        sub-sections.
    bs : list, optional
        List of the four parameters used to generate the lock-in function.
        This lock-in function is applied to the components to simulate
        offset and smoothing associated with the pDRM process.
    subsections : dict, optional
        Dictionary of subsections of the core sample. Keys containing the
        sub-section names and values lists of indices.

    Returns
    -------
    pandas.DataFrame
        Distorted data.
    """
    lif_DI = lif(bs_DI)
    lif_F = lif(bs_F)
    for key, sub in subsections.items():
        data_clean.loc[sub, "subs"] = key
    # apply lock-in function to simulate distortion associated to pDRM process
    if (np.sum(bs_DI) > 0) | (np.sum(bs_F) > 0):
        depth_rev = list(data_clean.iloc[::-1]["depth"])
        n = len(depth_rev)
        knots = np.concatenate(
            (
                [min(depth_rev) - 2],
                [min(depth_rev) - 1],
                depth_rev,
                [max(depth_rev) + 1],
                [max(depth_rev) + 2],
            )
        )
    if np.sum(bs_DI) > 0:
        NEZ_obs = np.zeros((n, 3))
        for j in tqdm(range(3)):
            vals = np.flip(NEZ_clean[:, j])
            vals = np.insert(vals, [n, 0], [vals[-1], vals[0]])
            ref_spline = interpolate.BSpline(knots, vals, 1)
            o = np.zeros(n)
            for i in range(n):
                o[i] = integrate.quad(
                    lambda z: ref_spline(depth_rev[i] - z) * lif_DI(z),
                    bs_DI[0],
                    np.sum(bs_DI),
                )[0]
            NEZ_obs[:, j] = np.flip(o)
        DIF_DI, dD, dI, _ = add_noise(NEZ_obs, ddec, dinc, dint)
    else:
        DIF_DI, dD, dI, _ = add_noise(NEZ_clean, ddec, dinc, dint)
    if np.sum(bs_F) > 0:
        NEZ_obs = np.zeros((n, 3))
        for j in tqdm(range(3)):
            vals = np.flip(NEZ_clean[:, j])
            vals = np.insert(vals, [n, 0], [vals[-1], vals[0]])
            ref_spline = interpolate.BSpline(knots, vals, 1)
            o = np.zeros(n)
            for i in range(n):
                o[i] = integrate.quad(
                    lambda z: ref_spline(depth_rev[i] - z) * lif_F(z),
                    bs_F[0],
                    np.sum(bs_F),
                )[0]
            NEZ_obs[:, j] = np.flip(o)
        DIF_F, _, _, dF = add_noise(NEZ_obs, ddec, dinc, dint)
    else:
        DIF_F, _, _, dF = add_noise(NEZ_clean, ddec, dinc, dint)
    data = pd.DataFrame(
        data={
            "t": data_clean.t,
            "dt": data_clean.dt,
            "depth": data_clean.depth,
            "lat": data_clean.lat,
            "lon": data_clean.lon,
            "D": DIF_DI[:, 0],
            "I": DIF_DI[:, 1],
            "F": DIF_F[:, 2],
            "dD": dD,
            "dI": dI,
            "dF": dF,
            "subs": data_clean.subs,
            "type": "sediments",
        }
    )
    for sub, group in data.groupby("subs"):
        data.loc[group.index, "D"] += offsets[sub]
    data["I"] = shallow_inc(data.I, f_shallow)
    data["F"] /= cal_fac
    data["dF"] /= cal_fac
    return data
