from dataclasses import dataclass
from typing import Sequence

from .params import Param
from .layers import Layer


@dataclass
class StateOld:
    '''deprecated version, no stratification'''
    params: Param
    energy: float
    ratio_accept: float
    sigma_temp = float

@dataclass
class State:
    Layers: Sequence[Layer]
    energy: float
    ratio_accept: float
    sigma_temp = float