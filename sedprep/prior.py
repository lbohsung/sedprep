from pymagglobal.utils import lmax2N, scaling, i2lm_l
import numpy as np
from utils import t2ocx
from constants import REARTH


def get_taus_and_alphas(
    lmax,
    R,
    alpha_dip,
    alpha_wodip,
    tau_wodip,
    omega_dip,
    xi_dip,
    axial=False,
):
    n_coeffs = lmax2N(lmax)

    if omega_dip is not None:
        chi_dip = np.sqrt(xi_dip**2 + omega_dip**2)
        alphas = alpha_wodip**2 * np.ones(n_coeffs)
        alphas[0:3] = alpha_dip**2
        alphas *= scaling(R, REARTH, lmax) ** 2

        taus = np.array(
            [tau_wodip / float(i2lm_l(i)) for i in range(n_coeffs)]
        )
        if axial:
            taus[0] = 1 / omega_dip
            taus[1:3] = 1 / 2 * tau_wodip
        else:
            taus[0:3] = 1 / omega_dip
        return taus, alphas, chi_dip


def generate_prior(
    knots,
    lmax,
    R,
    gamma,
    alpha_dip,
    alpha_wodip,
    tau_wodip,
    omega_dip=None,
    xi_dip=None,
    axial=False,
):
    n_coeffs = lmax2N(lmax)
    taus, diag, chi_dip = get_taus_and_alphas(
        lmax,
        R,
        alpha_dip,
        alpha_wodip,
        tau_wodip,
        omega_dip,
        xi_dip,
        axial,
    )
    mean = np.zeros((n_coeffs, len(knots)))
    mean[0] = gamma

    dt = np.abs(knots[:, None] - knots[None, :])
    dip = 3
    itau = np.diag(1.0 / taus)
    frac = dt[:, None, :, None] * itau[None, :, None, :]
    frac = np.abs(frac)

    # Set the number of knots to be replaced by the reference model
    n_ref = 3

    _Kalmag = np.genfromtxt("../dat/Kalmag_CORE_MEAN_Radius_6371.2.txt")
    # Ref coeffs should be of shape (n_coeffs, n_ref)
    ref_coeffs = _Kalmag[1 : n_coeffs + 1] / 1000

    cov = (1 + frac) * np.exp(-frac) * np.diag(diag)[None, :, None, :]
    # cov_d = (1 - frac) * np.exp(-frac) * np.diag(diag/taus**2)[None, :, None, :]
    # first axis are times, second axis are coeffs
    cov = np.diagonal(cov, axis1=1, axis2=3).copy()
    # cov_d = np.diagonal(cov_d, axis1=1, axis2=3).copy()
    # dipole with different matrix:
    cov[:, :, :dip] = (
        diag[None, None, :dip]
        * 0.5
        / xi_dip
        * (
            (xi_dip + chi_dip) * np.exp((xi_dip - chi_dip) * dt)
            + (xi_dip - chi_dip) * np.exp(-(xi_dip + chi_dip) * dt)
        )[:, :, None]
    )
    # cov_d[:, :, :dip] = (diag/taus**2)[None, None, :dip] * 0.5 / xi_dip * (
    #     (xi_dip - chi_dip) * np.exp((xi_dip - chi_dip)*dt)
    #     + (xi_dip + chi_dip) * np.exp(-(xi_dip + chi_dip)*dt)
    # )[:, :, None]
    chol = np.zeros((n_coeffs, len(knots) - n_ref, len(knots) - n_ref))
    # chol_d = np.zeros((n_coeffs, len(knots)-n_ref, len(knots)-n_ref))
    for it in range(n_coeffs):
        _cov = np.copy(cov[: len(knots) - n_ref, : len(knots) - n_ref, it])
        # _cov_d = np.copy(cov_d[:len(knots)-n_ref, :len(knots)-n_ref, it])
        _cor = cov[:, len(knots) - n_ref :, it]
        # _cor_d = cov_d[:, len(knots)-n_ref:, it]
        _icov = np.linalg.inv(
            cov[len(knots) - n_ref :, len(knots) - n_ref :, it]
        )
        # _icov_d = np.linalg.inv(cov_d[len(knots)-n_ref:, len(knots)-n_ref:, it])
        _cov -= (
            _cor[: len(knots) - n_ref] @ _icov @ _cor[: len(knots) - n_ref].T
        )
        # _cov_d -= _cor_d[:len(knots)-n_ref] @ _icov_d @ _cor_d[:len(knots)-n_ref].T
        chol[it, :, :] = np.linalg.cholesky(_cov)
        # chol_d[it, :, :] = np.linalg.cholesky(_cov_d)
        mean[it] += (
            _cor @ _icov @ (ref_coeffs[it] - mean[it, len(knots) - n_ref :])
        )

    # overwrite for usage in pyMC model
    mean = mean[:, : len(knots) - n_ref]
    # return mean, chol, n_ref, ref_coeffs, chol_d, diag/taus**2
    return mean, chol, n_ref, ref_coeffs, diag / taus**2
