import numpy as np

import statsmodels.api as sm
from statsmodels.robust.norms import HuberT

from scipy.optimize import minimize_scalar

from constants import default_acc_mean

# from pytensor import tensor as pt


def get_K_factor(K_fine):
    """Get the default K_factor

    Parameters
    ----------
    K_fine : int
        The number of sections at the highest resolution
    """

    def bar(x, y):
        return abs(y - x**x)

    result = minimize_scalar(
        bar,
        bounds=(1, 10 + np.log10(K_fine)),
        args=(K_fine,),
        method="bounded",
    )

    return np.ceil(result.x)


def get_brks_half_offset(K_fine, K_factor):
    """Get the overlapping breaks structure"""
    db_fine = 1 / K_fine
    db = db_fine
    brks = [np.arange(0, 1 + db, db)]
    n_br = len(brks[0])
    n_sec = n_br - 1
    newbrks = brks[0]

    while n_sec > 3:
        strt = np.min(brks[-1])
        # end = np.max(brks[-1])
        n_new = np.int64(np.ceil((n_sec + 1) / K_factor))
        l_new = n_new * K_factor
        l_old = n_sec
        d_new_old = l_new - l_old

        if d_new_old % 2 == 0:
            new_strt = strt - db * (d_new_old - 1) / 2
        else:
            new_strt = strt - db * (d_new_old) / 2
        newbrks = np.array(
            [new_strt + it * db * K_factor for it in range(n_new + 1)]
        )

        brks.append(newbrks)
        db = K_factor * db
        n_br = len(newbrks)
        n_sec = n_br - 1

    brks.append([newbrks[0], newbrks[-1]])
    brks = list(reversed(brks))

    return brks


def get_wts(a, b):
    """Get weights for parent sections

    Parameters
    ----------
    a : array-like
        The parent breaks
    b : array-like
        The child breaks

    Returns
    -------
    array
        The weights
    """
    intvls = [b[i - 1 : i + 1] for i in range(1, len(b))]
    gaps = [a[(a >= x[0]) & (a <= x[1])] for x in intvls]
    wts = []

    for i in range(len(intvls)):
        combined = np.unique(np.sort(np.concatenate([gaps[i], intvls[i]])))
        diffs = np.diff(combined)
        if len(diffs) == 0:
            diffs = [1]
        range_val = np.ptp(combined)
        wts_i = diffs / range_val if range_val != 0 else diffs
        wts.append(wts_i if len(wts_i) > 1 else np.repeat(wts_i, 2))

    wts = np.hstack([wt.reshape(-1, 1) for wt in wts])

    return wts


def get_levels_dict(brks):
    return_dict = {
        "1": {},
    }

    for i in range(1, len(brks)):
        pa = np.digitize(brks[i][:-1], brks[i - 1], right=False)
        pb = np.digitize(brks[i][1:], brks[i - 1], right=True)
        wts = get_wts(np.array(brks[i - 1]), np.array(brks[i]))
        wts /= np.sum(wts, axis=0)[None, :]
        return_dict[f"{i+1}"] = {
            "nK": len(brks[i]) - 1,
            "brks": brks[i],
            "parent1": pa.astype(int) - 1,
            "parent2": pb.astype(int) - 1,
            "wts1": wts[0],
            "wts2": wts[1],
        }

    _parent_diff = []
    for level in range(2, len(brks) + 1):
        _parent_diff.append(
            return_dict[f"{level}"]["parent1"]
            - return_dict[f"{level}"]["parent2"]
        )
    _parent_diff = np.concatenate(_parent_diff)

    return_dict["multi_parent_adj"] = np.abs(_parent_diff).mean() + 1
    return_dict["n_levels"] = len(brks)

    return return_dict

def get_acc_mean(age, depth):
    X = sm.add_constant(depth)
    model_result = sm.RLM(age, X, M=HuberT()).fit()

    # siginf equivalent, see https://stackoverflow.com/a/56974893
    digits = 2 - np.int64(np.ceil(np.log10(np.abs(model_result.params[1]))))
    acc_mean = np.round(model_result.params[1], digits)
    acc_mean = default_acc_mean if acc_mean <= 0 else acc_mean

    # Root Mean Square Error (RMSE) of residuals
    # residuals = age - model_result.predict(X)
    # rmse = np.sqrt(np.sum(residuals ** 2) / len(age))

    # sigma = model_result.bse[0]
    # age0 = (
    #     model_result.predict([1, max(depth)])[0]
    #     + np.random.normal(0, sigma, 1)[0]
    # )

    return acc_mean
