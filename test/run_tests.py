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


import sys
from pathlib import PurePath

import unittest
import doctest

from sedprep import utils, data_handling, deconvolve

# Fetch relative path
tests_path = PurePath(__file__).parent
# By convention the TestLoader discovers TestCases in test_*.py
suite = unittest.TestLoader().discover(tests_path)

# Add all doc-tests manually
suite.addTest(doctest.DocTestSuite(utils))
suite.addTest(doctest.DocTestSuite(data_handling))
suite.addTest(doctest.DocTestSuite(deconvolve))

if __name__ == "__main__":
    # Set up a test-runner
    runner = unittest.TextTestRunner(verbosity=2)
    # Collect test results
    result = runner.run(suite)
    # If not successful return 1
    sys.exit(not result.wasSuccessful())
