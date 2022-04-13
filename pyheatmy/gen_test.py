import numpy as np
from datetime import datetime, timedelta
from .params import Param, ParamsPriors, Prior, PARAM_LIST
from .checker import checker
from .core import Column


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

    @classmethod
    def from_dict(cls, time_series_dict):
        return cls(**time_series_dict)

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
            self._T_riv = None
        else :
            t_range = np.arange(len(self._dates))*self.param_time_dates[2]
            self._T_riv = self.param_T_riv_signal[0]*np.cos(2*np.pi*t_range/param_T_riv_signal[1]) + param_T_riv_signal[2]

    def _generate_Temp_aq_series(self): # renvoie un signal sinusoïdal de temperature aquifère
        if self._dates == None :
            self._T_aq = None
        else :
            t_range = np.arange(len(self._dates))*self.param_time_dates[2]
            self._T_aq = self.param_T_aq_signal[0]*np.cos(2*np.pi*t_range/param_T_aq_signal[1]) + param_T_aq_signal[2]

    def _generate_Temp_series(self, n_sens_vir=5): # en argument n_sens_vir le nb de capteur (2 aux frontières et 3 inutiles à 0)
        if self._dates == None :
            self._T_vir = None
        else :
            T_vir = np.zeros((len(self._dates),n_sens_vir)) # le tableau qui accueille des données de températures de forçage
            T_vir[:,0] = self._generate_Temp_riv_series()._T_riv
            T_vir[:,n_sens_vir-1] = self._generate_Temp_aq_series()._T_aq
            self._T_vir = list(zip(self._dates, T_vir))
