from collections import namedtuple
from dataclasses import dataclass
from random import uniform, gauss
from numpy import inf
from typing import Callable

PARAM_LIST = (
    "moinslog10K",
    "n",
    "lambda_s",
    "rhos_cs"
)

Param = namedtuple("Parametres", PARAM_LIST)


def cst(x): return 1.


@dataclass
class Prior:
    range: tuple
    sigma: float
    density: Callable[[float], float] = cst

    def perturb(self, val):
        new_val = val + gauss(0, self.sigma)
        while new_val > self.range[1]:
            new_val -= self.range[1] - self.range[0]
        while new_val < self.range[0]:
            new_val += self.range[1] - self.range[0]
        return new_val

    def sample(self):
        if self.range[0] > -inf and self.range[1] < inf:
            return uniform(*self.range)
        else:
            return 1  # better choices possible : arctan(uniform)


class ParamsPriors:
    def __init__(self, priors):
        self.prior_list = priors

    def sample_params(self):
        return Param(*(prior.sample() for prior in self.prior_list))

    def perturb(self, param):
        return Param(*(prior.perturb(val) for prior, val in zip(self.prior_list, param)))

    def __iter__(self):
        return self.prior_list.__iter__()

    def __getitem__(self, key):
        return self.prior_list[key]


if __name__ == '__main__':
    import numpy as np
    def reciprocal(x): return 1/x
    priors = {
        "moinslog10K": ((1.5, 6.), .01),  # (intervalle, sigma)
        "n": ((.01, .25), .01),
        "lambda_s": ((1, 5), .1),
        "rhos_cs": ((1e6, 1e7), 1e5),
        "sigma_temp": ((0, np.inf), 2, reciprocal)
    }
    priors = ParamsPriors(
        [Prior(*args) for args in (priors[lbl] for lbl in PARAM_LIST)]
    )

    init_param = priors.sample_params()
    print(priors.priors)
    print(init_param)
    print(init_param.sigma_temp)
