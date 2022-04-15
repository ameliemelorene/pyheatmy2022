import numpy as np
from typing import Sequence

from pyheatmy.core import Column

class valdirect(Column):
    "Compute the value of temperature based on the numerical solution."    
    def __init__(self,
        depth_sensors: Sequence[float],
        offset: float, 
        dH_measures: list, 
        moinslog10K: float, 
        lambda_m: float, 
        rho_c_m: float, 
        rho_c_w: float):
        
        self.depth_sensors = depth_sensors
        self.offset = offset

        self._real_z = np.array([0] + depth_sensors) + offset
        self._real_z[0] -= offset
        self._dH = np.array([d for _, (d, _) in dH_measures])
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
