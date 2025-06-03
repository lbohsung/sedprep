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

import numpy as np
from scipy.optimize import OptimizeResult
from tqdm import tqdm
from dlib import global_function_search, function_spec


def dlib_opt(func, bounds, args=(), max_feval=1000, rtol=1e-5, atol=1e-14,
             max_opt=None, deterministic=False, n_rand=50, progress=False):
    '''Find the global minimum of a function, by wrapping dlib's LIPO-TR
    [dlib]_, [King]_ function maximization algorithm in a style similar to
    `scipy.optimize` functions. Note that LIPO-TR doesn't converge in any
    mathematical sense. The algorithm will simpyly exhaust the `max_feval`
    function calls and report the best value found. "Convergence" can be
    declared, when the optimum found by LIPO-TR doesn't change for several
    iteartions. Use the `max_opt` parameter for this.

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : array-like
        Bounds for variables. They should be specified in a way, that
        `bounds[:, 0]` can be cast to a list of lower bounds, and
        `bounds[:, 1]` to upper bounds respectively.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    max_feval : int, optional
        The maximal number of function evaluations for LIPO-TR. Additional
        function evaluations may be performed by the polishing.
    rtol : float, optional
        Relative tolerance between two optima to be recognized as the same
        optimum.
    atol : float, optional
        Absolute tolerance between two optima to be recognized as the same
        optimum.
    max_opt : None or int, optional
        The numer of equal returns by LIPO-TR to declare convergence. Note
        that LIPO-TR per se doesn't converge in any mathematical sense and this
        is pure heuristics. If None, exhaust the `max_feval` function calls.
    n_rand : int, optional
        Number of initial random function evaluations.
    deterministic : bool, optional
        Wether to randomize LIPO-TR. Usually it uses a fixed seed and
        consecutive runs will therefore evaluate the same random points, i.e.
        the algorithm behaves deterministically.
    quiet : bool, optional
        Whether to output the diagnostics of the polishing stage.
    progress : bool, optional
        Whether to show a progressbar during the LIPO-TR run. If max_opt is
        set, the progressbar will stop abruptly once convergence is reached.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: `x` the solution array, `success` a
        Boolean flag indicating if the optimizer exited successfully and
        `fun`, the value at the optimum. If `polish` was employed, `nfev`
        contains the number of function evaluations by LIPO-TR + the number
        by the polishing.

    References
    ----------
    .. [dlib] : King, D.E., "Dlib-ml:  A machine learning toolkit", Journal of
                Machine Learning Research, vol. 10, pp. 1755-1758, 2009.
    .. [King] : King, D.E., "`A global optimization algorithm worth using
                <http://blog.dlib.net/2017/12/a-global-optimization-
                algorithm-worth.html>`_", accessed 2020-07-07
    '''
    # check if args are tuple, as this works for dlib but breaks the polishing,
    # which is annoying as the error will come only after dlib took its time...
    if not isinstance(args, tuple):
        raise TypeError("'args' have to be passed as a tuple.")

    # set up dlib
    dlib_bounds = function_spec(list(bounds[:, 0]), list(bounds[:, 1]))
    find_max = global_function_search(dlib_bounds)
    if not deterministic:
        find_max.set_seed(np.random.randint(1e6))
    # initial function call
    init = find_max.get_next_x()
    init.set(-func(np.array(init.x), *args))

    # random phase
    nxts = np.empty(n_rand, dtype=object)
    for it in range(n_rand):
        nxts[it] = find_max.get_next_x()

    if progress:
        enum = tqdm(range(n_rand), total=n_rand+max_feval, leave=False)
    else:
        enum = range(n_rand)
    for it in enum:
        nxts[it].set(-func(np.array(nxts[it].x), *args))
    # get best value from random phase
    _, f_opt, _ = find_max.get_best_function_eval()

    # set up alternating phase
    last_f_opt = -1e16
    n_feval = n_rand + 1
    opt_count = 0
    success = True

    if progress:
        enum = tqdm(
            range(max_feval),
            initial=n_rand,
            total=n_rand+max_feval,
        )
    else:
        enum = range(max_feval)
    for _ in enum:
        # get next value
        nxt = find_max.get_next_x()
        # evaluate
        f = -func(np.array(nxt.x), *args)
        nxt.set(f)
        # count function evaluations
        n_feval += 1

        _, f_opt, _ = find_max.get_best_function_eval()
        if max_opt is not None:
            # check if a change occured (within tolerance) and count
            if abs((last_f_opt - f_opt)) < rtol * abs(f_opt) + atol:
                opt_count += 1
            else:
                opt_count = 0
            if 2*max_opt < opt_count:
                break
        # update optimum, if necessary
        if last_f_opt < f_opt:
            last_f_opt = f_opt
    # get optimum
    x_opt, f_opt, _ = find_max.get_best_function_eval()
    x_opt = np.array(x_opt)
    # set up return value
    res = OptimizeResult(x=x_opt,
                         fun=-f_opt,
                         success=success,
                         nfev=n_feval,
                         nfev_lipo=n_feval)
    return res
