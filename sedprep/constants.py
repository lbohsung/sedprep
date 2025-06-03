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

pi = 3.141592653589793
rad2deg = 180 / pi

REARTH = 6371.2

na = ("9999", "999", "999.9", "nan", "-999", "-999.", " -9999")

# factors to convert MAD values into alpha95 values
# Reference: https://doi.org/10.1093/gji/ggv451
mad_to_alpha_factors = {
    "3": 7.69,
    "4": 3.90,
    "5": 3.18,
    "6": 2.88,
    "7": 2.71,
    "8": 2.63,
    "9": 2.57,
    "10": 2.54,
    "11": 2.51,
    "12": 2.48,
    "13": 2.46,
    "14": 2.44,
    "15": 2.43,
    "16": 2.43,
    "100": 2.37,
}

default_demag_steps = 4

default_dD = 4.5
default_dI = 4.5
default_alpha95 = 7

archkalmag_field_params_1p = {
    "lmax": 5,
    "R": 2800,  # km
    "gamma": -32.5,  # uT
    "alpha_dip": 35.96,  # uT
    "tau_dip": 166.0,  # yrs
    "alpha_wodip": 93.5,  # uT
    "tau_wodip": 482.0,  # yrs
    "axial": False,
}

# archkalmag_field_params_2p = {
#     "lmax": 5,
#     "R": 2800,  # km
#     "gamma": -32.8,  # uT
#     "alpha_dip": 39.6,  # uT
#     "tau_dip": 23.7,  # yrs
#     "tau_dip_slow": 986,  # yrs
#     "alpha_wodip": 94.5,  # uT
#     "tau_wodip": 514.0,  # yrs
#     "axial": False,
# }


archkalmag_field_params_2p = {
    "lmax": 5,
    "R": 2800,  # km
    "gamma": -38.96,  # uT
    "alpha_dip": 54.2,  # uT
    "omega_dip": 1 / 285.72,  # yrs
    "xi_dip": 1 / 94.82,  # yrs
    "alpha_wodip": 88.44,  # uT
    "tau_wodip": 694.3,  # yrs
    "axial": False,
}


pfm9k_field_params = {
    "lmax": 5,
    "R": 6371.2,
    "gamma": -32.5,
    "alpha_dip": [10, 3.5, 3.5],
    "tau_dip": [x / (2 * pi) for x in [433, 200, 200]],
    "tau_dip_slow": 50_000 / (2 * pi),
    "alpha_wodip": 5 * [1.765] + 7 * [1.011] + 9 * [0.455] + 11 * [0.177],
    "tau_wodip": 5 * [133] + 7 * [174] + 9 * [138] + 11 * [95],
    "axial": True,
}

field_params = archkalmag_field_params_2p

clip_nigp_D = 30  # declination clipping for nigp value in deg.
clip_nigp_I = 15  # inclination clipping for nigp value in deg.
clip_nigp_F = 3   # intensity clipping for nigp value in uT

# MCMC parameters
# -----------------------------------------------------------------------------
# Other parameters
tDir = 3.4 * 57.3 / 140  # Directional truncation error in Deg.
# tDir = 1.4                    # smaller value after investigation
tInt = 2.0  # Intensity truncation error in uT
alpha_nu = 2.0
beta_nu = 0.1
# -----------------------------------------------------------------------------
# Age-depth model parameters
acc_shape = 8
mem_mean = 0.5
mem_strength = 10
default_acc_mean = 20
mem_alpha = mem_strength * mem_mean
mem_beta = mem_strength * (1 - mem_mean)

# -----------------------------------------------------------------------------
# MCMC parameters
mcmc_params = {
    "n_samps": 500,
    "n_warmup": 1000,
    "n_chains": 4,
    "target_accept": 0.9,
}
