# This file is part of sedprep
#
# Copyright (C) 2024 Helmholtz Centre Potsdam
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import unittest
from pathlib import Path
import pandas as pd
import torch

from sedprep.constants import (
    archkalmag_field_params_2p,
    archkalmag_field_params_1p,
    pfm9k_field_params,
)
from sedprep.kalman_filter import Filter
from sedprep.data_handling import chunk_data

path = Path(__file__).parent

global inp

arch_data = pd.read_csv(path / "dat/arch.csv")
sed_data = pd.read_csv(path / "dat/sed_sweden_triangle.csv")

sed_data["F"] = np.nan
sed_data["dF"] = np.nan

bs_DI = [10.0, 10.0, 0.0, 10.0]
bs_F = None
offsets = {"A1": 30, "A2": -10}
f_shallow = 0.6
cal_fac = None
inp = torch.tensor(bs_DI + list(offsets.values()) + [f_shallow])

pars = torch.tensor(
    [0, 2, 0, 5] + [20, -8] + [0.4], dtype=torch.float64, requires_grad=True
)


def ml(x, prior_field_params=archkalmag_field_params_2p):
    cdat_sed, cdat_arch = chunk_data(
        sed_data,
        arch_data,
        lmax=prior_field_params["lmax"],
        delta_t=40,
        start=2000,
        end=1000,
    )
    filt = Filter(
        **prior_field_params,
        bs_DI=x[0:4],
        offsets={
            sed_data.subs.unique()[i]: x[4 + i]
            for i in range(len(sed_data.subs.unique()))
        },
        f_shallow=x[-1],
    )
    return -filt.forward(cdat_sed, cdat_arch)


class Test_filter(unittest.TestCase):
    def test_grad(self):
        """Test gradient against finite differences"""
        self.assertTrue(torch.autograd.gradcheck(ml, pars))

    def test_filter_with_archkalmag1p_prior(self):
        """Test filter with archkalmag14k prior with one parameter"""
        self.assertEqual(
            ml(
                inp,
                archkalmag_field_params_1p,
            ),
            torch.tensor(54922.0527362098728190176188945770263671875),
        )

    def test_filter_with_archkalmag2p_prior(self):
        """Test filter with archkalmag14k prior with two parameters"""
        self.assertEqual(
            ml(
                inp,
                archkalmag_field_params_2p,
            ),
            torch.tensor(54905.9634191300501697696745395660400390625),
        )

    def test_filter_with_pfm_prior(self):
        """Test filter with pfm prior"""
        self.assertEqual(
            ml(
                inp,
                pfm9k_field_params,
            ),
            torch.tensor(61544.7528796828482882119715213775634765625),
        )


if __name__ == "__main__":
    unittest.main()
