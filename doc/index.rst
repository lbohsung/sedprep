sedprep: Palaeomagnetic Sediment Record Preprocessing
=======================

.. important::

    This documentation was generated on |today|.
    :code:`sedprep` is still under heavy development and interfaces may change without notice until version 1.0.0 is released.

The `sedprep` package is a python software designed to preprocess palaeomagnetic sediment records for the geomagnetism community. It employs Bayesian modeling techniques based on Gaussian processes to estimate crucial parameters. These parameters can be used to enhance the quality of sediment data by addressing various distortions and transformations.

Key Features
------------

The `sedprep` package offers a range of functionalities, including:

1. **Declination Offset Estimation**: Estimate offset parameters to transform relative declinations to absolute declinations. Either for a complete record or for sub-sections.

2. **Shallowing Factor Estimation**: Estimate shallowing factor to correct for inclination shallowing.

3. **Relative palaeointensity (RPI) Calibration Factor Estimation**: Estimate the calibration factor to transform relative palaeointensity into absolute values.

4. **Lock-In Function Parameter Estimation**: Estimate lock-in function parameters associated to distortions caused by post-depositional detrital remanent magnetization (pDRM) effects. There are two classes of lock-in function, a four-parameter class which leads to more flexible lock-in functions but also higher computation costs and a two-parameter class which can be used if computational resources are small.

5. **Sediment Data Preprocessing**: After estimating the necessary parameters, `sedprep` provides a function to preprocess sediment data, ensuring that your data is corrected for known distortions and ready for scientific analysis.

Publications
------------

If you use :code:`sedprep`, please cite the corresponding papers

.. admonition:: Citation

  | **Bohsung**, L.; **Schanner**, M. A.; **Korte**, M.; **Holschneider**, M. (2024)
  | Estimating Post-Depositional Detrital Remanent Magnetization (pDRM) Effects
  | for Several Lacustrine and Marine Sediment Records
  | Using a Flexible Lock-In Function Approach
  | *JGR Solid Earth* (129)
  | e2024JB028864
  |
  | |doi-shield2|

  | **Bohsung**, L.; **Schanner**, M. A.; **Korte**, M.; **Holschneider**, M. (2023)
  | Estimating post-Depositional Detrital Remanent Magnetization (pDRM) Effects:
  | A Flexible Lock-In Function Approach
  | *JGR Solid Earth* (128)
  | e2023JB027373
  |
  | |doi-shield|


Installation
============

.. note::
  It is recommended to install :code:`sedprep` in a virtual conda environment.

.. important::

  :code:`sedprep` depends on `pymagglobal <https://sec23.git-pages.gfz-potsdam.de/korte/pymagglobal>`_. You have to install it, before installing :code:`sedprep`.

:code:`sedprep` is NOT distributed via the PyPI registry. For the installation follow the steps below

1. Install this theme:

   .. code-block:: bash 

        git clone git@git.gfz-potsdam.de:sec23/korte/sedprep.git

2. Navigate in the root folder of the cloned repository.

3. Run the following code

   .. code-block:: bash 

        pip install .


Testing
=======

To test your :code:`sedprep` installation, run
```
python tests/run_tests.py
```
from :code:`sedprep`.

Contact
-------
| Lukas Bohsung
| Helmholtz Centre Potsdam German Research Centre for Geoscienes GFZ
| Section 2.3: Geomagnetism
| Telegrafenberg
| 14473 Potsdam, Germany
| `lbohsung@gfz-potsdam.de <lbohsung@gfz-potsdam.de>`__

License
-------
GNU General Public License, Version 3, 29 June 2007

Copyright (C) 2024 Helmholtz Centre Potsdam GFZ, German Research Centre for Geosciences, Potsdam, Germany

:code:`sedprep` is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

:code:`sedprep` is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
   

.. |doi-shield| image:: https://img.shields.io/badge/DOI-10.1029%2F2021JB02316-blue.svg
		      :target: https://doi.org/10.1029/2023JB027373

.. |doi-shield2| image:: https://img.shields.io/badge/DOI-10.1029%2F2021JB02316-blue.svg
                  :target: https://doi.org/10.1029/2024JB028864


.. toctree::
    :caption: Table of contents
    :maxdepth: 2
    :hidden:

    introduction
    tutorial/tutorial
    synthetic_tests/src/synthetic_tests

.. toctree::
    :caption: Source
    :titlesonly:
    :maxdepth: 2
    :hidden:

    Repository <https://git.gfz-potsdam.de/sec23/korte/sedprep/>
