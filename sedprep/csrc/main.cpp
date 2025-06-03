// This file is part of sedprep
//
// Copyright (C) 2021 Helmholtz Centre Potsdam
// GFZ German Research Centre for Geosciences, Potsdam, Germany
// (https://www.gfz-potsdam.de)
//
// sedprep is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// sedprep is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.


#define _USE_MATH_DEFINES
#include <pybind11/pybind11.h>
#include "dspharm.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(_csedprep, m) {
    m.def("_dspharm", &dspharm, "");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
