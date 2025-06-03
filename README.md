# sedprep: Palaeomagnetic Sediment Record Preprocessing

The `sedprep` package is a python software designed to preprocess palaeomagnetic sediment records for the geomagnetism community. It employs Bayesian modeling techniques based on Gaussian processes to estimate crucial parameters. These parameters can be used to enhance the quality of sediment data by addressing various distortions and transformations.  
Visit our [Website](https://sedprep-lbohsung-32dc33f26a4e1fb1d4e6d8eb5e6997665b41735a809c14.git-pages.gfz-potsdam.de) for a tutorial and additional information.

## Key Features

The `sedprep` package offers a range of functionalities, including:

1. **Declination Offset Estimation**: Estimate offset parameters to transform relative declinations to absolute declinations. Either for a complete record or for sub-sections.

2. **Shallowing Factor Estimation**: Estimate shallowing factor to correct for inclination shallowing.

3. **Relative palaeointensity (RPI) Calibration Factor Estimation**: Estimate the calibration factor to transform relative palaeointensity into absolute values.

4. **Lock-In Function Parameter Estimation**: Estimate lock-in function parameters associated to distortions caused by post-depositional detrital remanent magnetization (pDRM) effects. There are two classes of lock-in function, a four-parameter class which leads to more flexible lock-in functions but also higher computation costs and a two-parameter class which can be used if computational resources are small.

5. **Sediment Data Preprocessing**: After estimating the necessary parameters, `sedprep` provides a function to preprocess sediment data, ensuring that your data is corrected for known distortions and ready for scientific analysis.

## Citation
If you use :code:`sedprep`, please cite the corresponding papers

  | **Bohsung**, L.; **Schanner**, M. A.; **Korte**, M.; **Holschneider**, M. (2023)
  | Estimating post-Depositional Detrital Remanent Magnetization (pDRM) Effects:
  | A Flexible Lock-In Function Approach
  | *JGR Solid Earth* (128)
  | e2023JB027373 
[![DOI](https://img.shields.io/badge/DOI-10.1029%2F2023JB027373-blue.svg)](https://doi.org/10.1029/2023JB027373)

  | **Bohsung**, L.; **Schanner**, M. A.; **Korte**, M.; **Holschneider**, M. (2024)
  | Estimating Post-Depositional Detrital Remanent Magnetization (pDRM) Effects for Several Lacustrine and Marine Sediment Records
  | Using a Flexible Lock-In Function Approach
  | *JGR Solid Earth* (129)
  | e2024JB028864 
[![DOI](https://img.shields.io/badge/DOI-10.1029%2F2023JB027373-blue.svg)](https://doi.org/10.1029/2024JB028864)


## Installation

It is recommended to install `sedprep` in a virtual conda environment.
> **Note:** sedprep depends on [pymagglobal](https://sec23.git-pages.gfz-potsdam.de/korte/pymagglobal). You have to install it before running the install command.

1. Clone this Repository:

    ```
    git clone git@git.gfz-potsdam.de:sec23/korte/sedprep.git
    ```

2. Navigate in the root folder of the cloned repository.

3. Run the following code
    ```
    pip install .
    ```

## Testing

To test your `sedprep` installation, run
```
python tests/run_tests.py
```
from `<sedprep>`.

## Contact
Lukas Bohsung  
Helmholtz Centre Potsdam German Research Centre for Geoscienes GFZ  
Section 2.3: Geomagnetism  
Telegrafenberg  
14473 Potsdam, Germany  
[lbohsung@gfz-potsdam.de](mailto:lbohsung@gfz-potsdam.de)  

## License
GNU General Public License, Version 3, 29 June 2007

Copyright (C) 2024 Helmholtz Centre Potsdam -  GFZ German Research Centre for Geosciences, Potsdam, Germany (https://www.gfz-potsdam.de)

sedprep is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

sedprep is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Data files are licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## References

<a id="1">[1]</a> 
Schanner, M., Korte, M., & Holschneider, M. (2022). ArchKalmag14k: A Kalman‐filter based global geomagnetic model for the Holocene. Journal of Geophysical Research: Solid Earth, 127(2), e2021JB023166.

`sedprep` uses `numpy`, `scipy`, `matplotlib`, `pandas`, `pybind11`, `dlib`, `tqdm`, and `pymagglobal`:

<a id="3">[3]</a> 
[\[scipy\]](https://www.scipy.org/) Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland,  
Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson,   
Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett,  
Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson,   
Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng,  
Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman,
Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald,  
Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt,  
and SciPy 1.0 Contributors (2020)  
"SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python".  
Nature Methods, in press.  
https://doi.org/10.1038/s41592-019-0686-2

[\[matplotlib\]](https://matplotlib.org/)  J. D. Hunter (2007)  
"Matplotlib: A 2D Graphics Environment",  
Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95  
https://doi.org/10.1109/MCSE.2007.55

[\[pandas\]](https://pandas.pydata.org/)  Wes McKinney (2010)   
"[Data structures for statistical computing in python]( http://conference.scipy.org/proceedings/scipy2010/pdfs/mckinney.pdf)",  
Proceedings of the 9th Python in Science Conference, Volume 445

<a id="2">[2]</a> 
[\[dlib\]](http://dlib.net/) King, D.E. (2009)  
"Dlib-ml:  A machine learning toolkit"  

[\[tqdm\]](https://tqdm.github.io/) Casper da Costa-Luis, Stephen Karl Larroque, Kyle Altendorf, Hadrien Mary, richardsheridan, Mikhail Korobov, Noam Yorav-Raphael, Ivan Ivanov, Marcel Bargull, Nishant Rodrigues, Guangshuo CHEN, Antony Lee, Charles Newey, James, Joshua Coales, Martin Zugnoni, Matthew D. Pagel, mjstevens777, Mikhail Dektyarev, ... Max Nordlund (2021)  
"tqdm: A fast, Extensible Progress Bar for Python and CLI"  
Zenodo.  
https://doi.org/10.5281/zenodo.5517697  

[\[pymagglobal\]](https://sec23.git-pages.gfz-potsdam.de/korte/pymagglobal) Schanner, M. A.; Mauerberger, S.; Korte, M. (2020)  
"pymagglobal - Python interface for global geomagnetic field models"  
GFZ Data Services.  
https://doi.org/10.5880/GFZ.2.3.2020.005  
