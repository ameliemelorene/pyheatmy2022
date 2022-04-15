# param par défaut dans gen_test.py
DEFAULT_dH = [1, 24*3600, 10]
DEFAULT_T_riv = [30, 24*3600, 20]
DEFAULT_T_aq = [30, 24*3600, 12]
DEFAULT_time_step = 15 # 15mn

# valeur absurdre par défaut
CODE = 959595

N_SENSORS_SHAFT = 4

from .core import Column
from .params import Param
from .checker import ComputationOrderException
from .layers import layersListCreator
from .gen_test import Time_series