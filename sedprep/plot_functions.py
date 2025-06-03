import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature

# import pandas as pd
# from matplotlib.ticker import FormatStrFormatter
import pathlib
import pymagglobal
from pymagglobal.utils import i2lm_l

from data_handling import t2d, d2t
from constants import REARTH
from utils import dsh_basis, nez2dif

path = pathlib.Path(__file__).parent.resolve()


def numerate_plots(ax, text):
    ax.text(
        0,
        1,
        text,
        color="black",
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="center",
        transform=ax.transAxes,
        bbox=dict(
            facecolor="#DCE6F2",
            edgecolor="#004378",
            linewidth=1.3,
            pad=0.35,
            boxstyle="round,rounding_size=0.25",
        ),
    )


def plot_DIF(
    core,
    axD=None,
    axI=None,
    axF=None,
    axF_twin=None,
    time_or_depth="time",
    plot_second_x_axis=True,
    xerr=False,
    yerr=False,
    err_alpha=0.4,
    distinguish_subs=True,
    legend=True,
    subs_cmap="tab20",
    color="C0",
    pymagglobal_model_name=None,
    model_file=None,
    show_uids=False,
    limit_y_to_data=True,
    dpi=300,
):
    if axD is None:
        fig, (axD, axI, axF) = plt.subplots(
            3, 1, figsize=(10, 6), sharex=True, dpi=dpi
        )
        fig.align_ylabels()
        fig.tight_layout()
    axs, comps = [], []
    for ax, comp in zip([axD, axI, axF], ["D", "I", "F"]):
        if ax is not None:
            axs.append(ax)
            comps.append(comp)
    d_t = "t" if time_or_depth == "time" else "depth"
    _t2d = t2d(core)
    _d2t = d2t(core)
    _xerr = core["dt"] if (d_t == "t") & (xerr) else None
    if xerr and d_t == "depth":
        d_llim = _t2d(core["t"] + core["dt"])
        d = _t2d(core["t"])
        d_ulim = _t2d(core["t"] - core["dt"])
        d_uerr = abs(d - d_ulim)
        d_lerr = abs(d - d_llim)
        _xerr = np.array((d_lerr, d_uerr))
    for comp, ax in zip(comps, axs):
        if distinguish_subs:
            cmap = plt.get_cmap(subs_cmap)
            norm = plt.Normalize(vmin=0, vmax=len(core["subs"].unique()))
            for idx, (sub_id, sub) in enumerate(core.groupby("subs")):
                _, _, bars = ax.errorbar(
                    sub[d_t],
                    sub[comp],
                    xerr=(
                        (
                            _xerr[sub.index]
                            if d_t == "t"
                            else _xerr[:, sub.index]
                        )
                        if xerr
                        else None
                    ),
                    yerr=sub["d" + comp] if yerr else None,
                    color=cmap(norm(idx)),
                    fmt=".",
                    ls="",
                    label=f"{sub_id}",
                )
                [bar.set_alpha(err_alpha) for bar in bars]
        else:
            _, _, bars = ax.errorbar(
                core[d_t],
                core[comp],
                xerr=_xerr,
                yerr=core["d" + comp] if yerr else None,
                color=color,
                fmt=".",
                ls="",
                label="sediment data",
            )
            [bar.set_alpha(err_alpha) for bar in bars]
        ax.set_ylabel(
            "Declination"
            if comp == "D"
            else ("Inclination" if comp == "I" else "RPI")
        )
        if limit_y_to_data:
            y_lim_D = axD.get_ylim()
            y_lim_I = axI.get_ylim()
        if show_uids:
            for i, v in zip(core.index, core.UID):
                ax.text(core[d_t][i], core[comp][i], str(v))
    if pymagglobal_model_name is not None:
        if pymagglobal_model_name not in pymagglobal.built_in_models().keys():
            raise ValueError(
                "pymagglobal_model_name has to be one of the following:\n"
                f"{list(pymagglobal.built_in_models().keys())}"
            )
        model = pymagglobal.Model(pymagglobal_model_name)
        model_knots = np.linspace(
            max(min(core["t"]), model.t_min),
            min(max(core["t"]), model.t_max),
            1000,
        )
        loc = (core["lat"][0], core["lon"][0])
        d, i, f = pymagglobal.local_curve(model_knots, loc, model)
        if axF is not None:
            axF_twin = axF.twinx() if axF_twin is None else axF_twin
            for comp, ax in zip([d, i, f/1000], [axD, axI, axF_twin]):
                ax.plot(
                    model_knots if d_t == "t" else _t2d(model_knots),
                    comp,
                    ls="--",
                    c="black",
                    label=pymagglobal_model_name,
                )
            axF_twin.set_ylabel("Intensity [uT]")
        else:
            for comp, ax in zip([d, i], [axD, axI]):
                ax.plot(
                    model_knots if d_t == "t" else _t2d(model_knots),
                    comp,
                    ls="-",
                    c="gray",
                    label=pymagglobal_model_name,
                )
    if model_file is not None:
        model_knots = model_file["knots"]
        indices = np.where(
            (model_knots >= min(core.t)) & (model_knots <= max(core.t))
        )[0]
        model_knots = model_knots[indices]
        model_coeffs = model_file["samples"]
        model_coeffs = model_coeffs.transpose(1, 0, 2)[
            : model_coeffs.shape[1] // 2
        ]
        model_coeffs = model_coeffs[:, :, indices]
        loc = np.array((90 - core.lat[0], core.lon[0], REARTH))
        base = dsh_basis(i2lm_l(model_coeffs.shape[0] - 1), loc)
        nez = np.einsum("ij, ikl->jkl", base, model_coeffs)
        d, i, f = nez2dif(*nez, be=np)
        if axF is not None:
            axF_twin = axF.twinx() if axF_twin is None else axF_twin
            model_x = model_knots if d_t == "t" else _t2d(model_knots)
            for comp, ax in zip([d, i, f], [axD, axI, axF_twin]):
                ax.plot(
                    model_x,
                    np.mean(comp, axis=0),
                    c="gray",
                    label="ArchKalmag14k.r",
                )
                ax.plot(
                    model_x,
                    comp.T,
                    c="gray",
                    alpha=0.05,
                )
            axF_twin.set_ylabel("Intensity [uT]")
        else:
            for comp, ax in zip([d, i], [axD, axI]):
                ax.plot(
                    model_knots if d_t == "t" else _t2d(model_knots),
                    np.mean(comp, axis=0),
                    c="gray",
                    label="ArchKalmag14k.r",
                )
                ax.plot(
                    model_knots if d_t == "t" else _t2d(model_knots),
                    comp.T,
                    c="gray",
                    alpha=0.05,
                )
    if legend:
        axs[-1].legend(
            bbox_to_anchor=(0.5, -0.6),
            loc="center",
            ncol=8,
        )
    axs[-1].set_xlabel(
        "Absolute time [years]" if d_t == "t" else "Absolute depth [cm]"
    )
    if plot_second_x_axis:
        axD_xax2 = axs[0].secondary_xaxis(
            "top",
            functions=(_t2d, _d2t) if d_t == "t" else (_d2t, _t2d),
        )
        axD_xax2.set_xlabel(
            "Absolute depth [cm]" if d_t == "t" else "Absolute time [years]"
        )
    if limit_y_to_data:
        axD.set_ylim((y_lim_D[0], y_lim_D[1]))
        axI.set_ylim(y_lim_I)


def plot_map(lat, lon, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.axis("off")
    left, bottom, width, height = ax.get_position().bounds
    global_ax = fig.add_axes(
        (left-0.03, bottom-0.08, width+0.03, height),
        rasterized=True,
        projection=ccrs.Mollweide()
    )
    global_ax.set_global()
    global_ax.add_feature(cfeature.LAND, zorder=0, color="lightgray")
    global_ax.scatter(
        lon,
        lat,
        marker="*",
        c="C3",
        rasterized=True,
        transform=ccrs.PlateCarree()
    )



# def _log_prob_t(t, dat, calibration_curve, a=3, b=4):
#     # Generalized Student's t-distribution, as proposed by Christen and Perez
#     # (2009)
#     # XXX:not normalized
#     mu = np.interp(
#         t,
#         1950 - calibration_curve["CAL BP"].values,
#         calibration_curve["14C age"].values,
#     )

#     sig = np.interp(
#         t,
#         1950 - calibration_curve["CAL BP"].values,
#         calibration_curve["Sigma 14C"].values,
#     )

#     sig = np.sqrt(dat["Sigma 14C"] ** 2 + sig**2)
#     df = mu - dat["14C age"]
#     return -(a + 0.5) * np.log(b + 0.5 * (df / sig) ** 2)


# def _get_curve(dat, calibration_curve, thresh=1e-3, func=_log_prob_t):
#     _t = 1950 - calibration_curve["CAL BP"].values
#     prob = np.exp(func(_t, dat, calibration_curve=calibration_curve))
#     prob /= np.sum(prob)
#     inds = np.argwhere(thresh * prob.max() <= prob).flatten()

#     return _t[min(inds): max(inds)], prob[min(inds): max(inds)]


# def plot_adm(ax, adm_idata, rc_data, path_to_cal_curves_folder):
#     cal_curves = []
#     for curve_file in ["intcal20", "marine20", "shcal20"]:
#         cal_curves.append(
#             pd.read_csv(
#                 f"{path_to_cal_curves_folder}/{curve_file}.14c",
#                 header=11,
#                 sep=",",
#                 names=[
#                     "CAL BP",
#                     "14C age",
#                     "Sigma 14C",
#                     "Delta 14C",
#                     "Sigma Delta 14C",
#                 ],
#             )
#         )
#     dd = np.array(adm_idata.adm_pars["dd"]).flatten().item()
#     D = np.array(adm_idata.adm_pars["D"]).flatten().item()
#     thin = (
#         len(adm_idata.posterior.chain) * len(adm_idata.posterior.draw)
#     ) // 1000
#     a_samps = np.array(adm_idata.posterior["a"])
#     a_samps = a_samps[:, ::thin].reshape(-1, D).T
#     theta_samps = np.array(adm_idata.posterior["theta"])
#     theta_samps = theta_samps[:, ::thin].flatten()
#     sum_a_samps = np.cumsum(a_samps, axis=0)
#     z = np.linspace(0, D * dd - 0.1, 1001)
#     ind = (z // dd).astype(int)
#     t_samps = (
#         theta_samps
#         - sum_a_samps[ind] * dd
#         - a_samps[ind] * (z - dd * (ind + 1))[:, None]
#     )
#     ax.plot(t_samps, z, zorder=0, color="grey", alpha=0.02)
#     ax.plot(t_samps.mean(axis=1), z, zorder=2, color="C3")
#     idx_nh = rc_data.query('calib == "northern"').index.to_numpy()
#     idx_marine = rc_data.query('calib == "marine"').index.to_numpy()
#     idx_sh = rc_data.query('calib == "southern"').index.to_numpy()
#     for curve, color, idx in zip(
#         cal_curves, ["C0", "C1", "C2"], [idx_nh, idx_marine, idx_sh]
#     ):
#         for _, row in rc_data.iloc[idx].iterrows():
#             _t, _prob = _get_curve(row, calibration_curve=curve)
#             _prob /= _prob.max()
#             ax.fill_between(
#                 _t,
#                 row["depth"] - 15 * _prob,
#                 row["depth"] + 15 * _prob,
#                 alpha=0.5,
#                 color=color,
#                 zorder=2,
#             )

#     ax.set_xlabel("Absolute time [years]")
#     ax.set_ylabel("Absolute depth [cm]")
#     ax.invert_yaxis()
#     for label in ax.get_yticklabels():
#         label.set_rotation(90)
#         label.set_va("center")
#     ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
