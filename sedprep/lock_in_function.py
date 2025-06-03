import numpy as np

import pymc as pm
from pytensor import tensor as pt


def lock_in_function(type="4p"):
    def four_parameter_function(name):
        a_s_cent = pm.Uniform(
            f"lock_in_{name}_centered",
            lower=1e-6,
            upper=1,
            size=4,
            initval=np.random.uniform(low=0.01, high=0.05, size=4),
        )
        a_s = pm.Deterministic(f"lock_in_{name}", 100 * a_s_cent)
        return pt.cumsum(a_s, axis=0)

    def two_parameter_function(name):
        a_s_cent_0 = pm.Uniform(
            f"lock_in_shift_{name}_centered",
            lower=1e-6,
            upper=1,
            size=1,
            initval=np.random.uniform(low=0.01, high=0.05, size=1),
        )
        a_s_cent_1 = pm.Uniform(
            f"lock_in_width_{name}_centered",
            lower=1e-6,
            upper=1,
            size=1,
            initval=np.random.uniform(low=0.01, high=0.05, size=1),
        )
        a_s = pm.Deterministic(
            f"lock_in_{name}", pt.concatenate([100 * a_s_cent_0, 30 * a_s_cent_1])
        )
        return (
            a_s[0],
            a_s[0] + a_s[1] / 2,
            a_s[0] + 1.5 * a_s[1],
            a_s[0] + 2 * a_s[1],
        )
    
    if type=="4p":
        return four_parameter_function
    if type=="2p":
        return two_parameter_function
    else:
        raise ValueError(
            "Valid inputs are \'2p\' or \'4p\' to use two-parameter "
            "or four-parameter lock-in function respectively"
        )