# This file is part of sedprep
#
# Copyright (C) 2021 Helmholtz Centre Potsdam
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

import os
import codecs
import setuptools

from pybind11.setup_helpers import Pybind11Extension, build_ext


# https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


def get_author(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__author__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find author string.")


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

name = "sedprep"
version = get_version("sedprep/__init__.py")
author = get_author("sedprep/__init__.py")
description = """Fully Bayesian paleomagnetic sediment record preprocessing"""
copyright = (
    "2024 Helmholtz Centre Potsdam GFZ, "
    + "German Research Centre for Geosciences, Potsdam, Germany"
)


ext_modules = [
    Pybind11Extension(
        "_csedprep",
        ["sedprep/csrc/main.cpp"],
        define_macros=[("VERSION_INFO", version)],
          extra_compile_args=[
            "-march=native",
            "-O3",
            "-fopenmp"
          ],
          libraries=["gomp"],
    ),
]




setuptools.setup(
    name=name,
    version=version,
    author=author,
    author_email="lbohsung@gfz-potsdam.de",
    packages=["sedprep"],
    license="GPL v3",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy>=1.18",
        "scipy>=1.5.4",
        "matplotlib>=2.2.5",
        "dlib>=19.21",
        "requests",
        "arviz",
        "pandas",
        "torch",
        "tqdm",
        "pymagglobal",
    ],
    ext_modules=ext_modules,
    data_files=[("sedprep", ["sedprep/csrc/dspharm.h"])],
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    extras_require={"tests": ["orthopoly>=0.9", "packaging"]},
)
