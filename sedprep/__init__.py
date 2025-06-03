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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
`sedprep` is a python package for a Bayesian
paleomagnetic sediment record preprocessing.
"""

import warnings

# Monkey-patch the line away from warnings, as it is rather irritating.
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = (
    lambda msg, cat, fname, lineno, line=None: formatwarning_orig(
        msg, cat, fname, lineno, line=""
    )
)

__version__ = "0.1.0"
__author__ = ["Bohsung, L.", "Schanner, M. A."]
