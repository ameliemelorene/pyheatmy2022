import numpy as np
from datetime import datetime, timedelta
from .params import Param, ParamsPriors, Prior, PARAM_LIST
from .state import State
from .checker import checker
from .core import Column
from .utils import C_W, RHO_W, LAMBDA_W, compute_H, compute_T, compute_H_stratified, compute_T_stratified
from .layers import Layer, getListParameters, sortLayersList


class Time_series:  # on simule un tableau de mesures
    def __init__(
        self,
        param_time_dates: list, # liste [date début, date fin, pas de temps (constant)], format des dates tuple datetime ou none
        param_dH_signal: list, # liste [amplitude (m), période (s), offset (m)] pour un variation sinusoïdale
        param_T_riv_signal: list,  # liste [amplitude (°C), période (en seconde), offset (°C)] pour un variation sinusoïdale
        param_T_aq_signal: list,  # liste [amplitude (°C), période (en seconde), offset (°C)] pour un variation sinusoïdale
        sigma_meas_P: float,  # (m) écart type de l'incertitude sur les valeurs de pression capteur
        sigma_meas_T: float,  # (°C) écart type de l'incertitude sur les valeurs de température capteur
    ):
       # ! Pour l'instant on suppose que les temps matchent
        self._dates = None
        # le tableau d'observation des charges utilisable dans colonne
        self._dH = None
        # récupère la liste de température observée de la rivière (au cours du temps)
        self._T_riv = None
        # récupère la liste de température observée de l'aquifère (au cours du temps)
        self._T_aq = None
        # le tableau d'observation des températures utilisable dans colonne
        self._T_vir = None
        

    
    def _generate_dates_series(self, n_len_times = 2000):
        if self.param_time_dates == None :
            self._dates = np.array([datetime.fromtimestamp(15*k) for k in range(n_len_times)])
        else :
            dt, end, step = datetime(*self.param_time_dates[0]), datetime(*self.param_time_dates[1]), timedelta(seconds=self.param_time_dates[2])
            times_vir1 = []
            while dt < end:
                times_vir1.append(dt) # pas encore un objet datetime
                dt += step
            times_vir1 = np.array(times_vir1) # TODO : mettre au format datetime 
            self._dates = np.array(times_vir1)
    
    def _generate_dH_series(self): # renvoie un signal sinusoïdal de différence de charge
        if self._dates == None :
            self._dH = None
        else :
            t_range = np.arange(len(self._dates))*self.param_time_dates[2]
            dH_signal = self.param_dH_signal[0]*np.cos(2*np.pi*t_range/param_dH_signal[1]) + param_dH_signal[2]
            self.dH = list(zip(self._dates,list(zip(dH_signal, self._T_riv))))

    def _generate_Temp_riv_series(self): # renvoie un signal sinusoïdal de temperature rivière
        if self._dates == None :
            self._dH = None
        else :
            t_range = np.arange(len(self._dates))*self.param_time_dates[2]
            dH_signal = self.param_T_riv_signal[0]*np.cos(2*np.pi*t_range/param_T_riv_signal[1]) + param_T_riv_signal[2]

    def _generate_Temp_aq_series(self): # renvoie un signal sinusoïdal de temperature aquifère
            if self._dates == None :
                self._dH = None
            else :
                t_range = np.arange(len(self._dates))*self.param_time_dates[2]
                dH_signal = self.param_T_riv_signal[0]*np.cos(2*np.pi*t_range/param_T_riv_signal[1]) + param_T_riv_signal[2]
