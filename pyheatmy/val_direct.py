import numpy as np
from typing import Sequence

from pyheatmy.core import Column
from pyheatmy import LAMBDA_W, RHO_W, C_W, DEFAULT_period, DEFAULT_time_step

class valdirect:
    "Compute the value of temperature based on the numerical solution."    
    def __init__( 
        self,
        # from column
        z_range : np.array, # z_solve()
        id_sensors : list,  # get_id_sensors()
        time_window
        # from user
        dH : float,
        time_window : tuple,
        # from layer : self.param
        moinslog10K: float, 
        n:float, 
        lambda_s : float,
        rhos_cs : float,
    ):
        # utils.py + layers.py
        
        self._lambda_m = (n * (LAMBDA_W) ** 0.5 + (1.0 - n) * (lambda_s) ** 0.5) ** 2
        self._rho_c_m = n * RHO_W * C_W + (1 - n) * rhos_cs
        self._n_cell = len(z_range)

        self.period = DEFAULT_period*24*3600 # période du signal en secondes (s)
        self.time_step = DEFAULT_time_step*60 # pas de temps en secondes (s)
        self.time_range = 3000
        self._dH = dH
        self._times = [t for t, _ in dH_measures]

        self.moinslog10K = moinslog10K
        self.lambda_m = lambda_m
        self.rho_c_m = rho_c_m
        self.rho_c_w = rho_c_w

        self.K = 10 ** - self.moinslog10K
        self.kappa = self.lambda_m/self.rho_c_m
        self.alpha = self.rho_c_w/self.rho_c_m*self.K

        self.analy_temp_general = None
        self.analy_temp_cond = None

        self._a = None #ou CODE_a
        self._b = None
        self.loc = None

    @classmethod
    def from_dict(cls, val_dir_dict):
        return cls(**val_dir_dict)

    def compute_a(self, nb_cells, period):
        dz = self._real_z[-1] / nb_cells
        v_t = -self.alpha*self._dH/dz
        self._a = (np.sqrt((np.sqrt(v_t**4 + (8*np.pi*self.kappa/period)**2 + v_t**2))/2) - v_t)/(2*self.kappa)

    def compute_b(self, nb_cells, period): 
        dz = self._real_z[-1] / nb_cells
        v_t = -self.alpha*self._dH/dz
        self._b = np.sqrt((np.sqrt(v_t**4 + (8*np.pi*self.kappa/period)**2 - v_t**2))/2)/(2*self.kappa)
    
    def compute_temp_general(self, nb_cells, period, theta_amp, theta_mu):
        theta_amp = 1
        theta_mu = 1
        dz = self._real_z[-1] / nb_cells
        all_dt = np.array([(self._times[j+1] - self._times[j]).total_seconds()
                            for j in range(len(self._times) - 1)])
        self._z_solve = dz/2 + np.array([k*dz for k in range(nb_cells)])

        a = self.compute_a(self, nb_cells, period)
        b = self.compute_b(self, nb_cells, period)
        self.analy_temp_general = theta_mu + theta_amp*np.exp(-a*self._z_solve)*np.cos(2*np.pi*all_dt/period - b*self._z_solve)

    def compute_temp_cond(self, nb_cells, period, theta_amp, theta_mu):
        theta_amp = 1
        theta_mu = 1
        dz = self._real_z[-1] / nb_cells
        all_dt = np.array([(self._times[j+1] - self._times[j]).total_seconds()
                            for j in range(len(self._times) - 1)])
        self._z_solve = dz/2 + np.array([k*dz for k in range(nb_cells)])

        a_cond = np.sqrt(np.pi/(self.kappa*period))
        self.analy_temp_cond = theta_mu + theta_amp*np.exp(-a_cond*self._z_solve)*np.cos(2*np.pi*all_dt/period - a_cond*self._z_solve)

    def get_loc(self, nb_cells):
        dz = self._real_z[-1] / nb_cells
        self.loc = int(self._real_z/dz)