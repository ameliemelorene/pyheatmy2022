from datetime import datetime, timedelta
from .params import Param, ParamsPriors, Prior, PARAM_LIST
from .checker import checker
from .core import Column
from pyheatmy import DEFAULT_dH, DEFAULT_T_riv, DEFAULT_T_aq, DEFAULT_time_step, N_SENSORS_SHAFT, CODE

import numpy as np
from numpy.random import normal

class Time_series:  # on simule un tableau de mesures
    def __init__(
        self,
        param_time_dates: list = [None,None,DEFAULT_time_step], # liste [date début, date fin, pas de temps (constant)], format des dates tuple datetime ou none
        param_dH_signal: list = DEFAULT_dH, # liste [amplitude (m), période (s), offset (m)] pour un variation sinusoïdale
        param_T_riv_signal: list = DEFAULT_T_riv,  # liste [amplitude (°C), période (en seconde), offset (°C)] pour un variation sinusoïdale
        param_T_aq_signal: list = DEFAULT_T_aq,  # liste [amplitude (°C), période (en seconde), offset (°C)] pour un variation sinusoïdale
        sigma_meas_P: float = None,  # (m) écart type de l'incertitude sur les valeurs de pression capteur
        sigma_meas_T: float = None,  # (°C) écart type de l'incertitude sur les valeurs de température capteur
    ):
        
        # on définit les attribut paramètres
        self._param_dates = param_time_dates
        self._param_dH = param_dH_signal
        self._param_T_riv = param_T_riv_signal
        self._param_T_aq = param_T_aq_signal
        self._sigma_P = sigma_meas_P
        self._sigma_T = sigma_meas_T

        self._dates = np.array([None])
        # le tableau d'observation des charges utilisable dans colonne
        self._dH = np.array([None])
        self._dH_perturb = np.array([None]) # avec perturbation
        # récupère la liste de température observée de la rivière (au cours du temps)
        self._T_riv = np.array([None])
        self._T_riv_perturb = np.array([None]) # avec perturbation
        # le tableau d'observation de la pression et de la température rivière
        self._T_riv_dH_measures = None
        self._T_riv_dH_measures_perturb = None
        # récupère la liste de température observée de l'aquifère (au cours du temps)
        self._T_aq = np.array([None])
        self._T_aq_perturb = np.array([None]) # avec perturbation
        # le tableau d'observation des températures utilisable dans colonne
        self._T_Shaft = np.array([None])
        self._T_Shaft_perturb = np.array([None]) # avec perturbation

        self._T_Shaft_measures = None
        self._T_Shaft_perturb_measures = None # avec perturbation


    @classmethod
    def from_dict(cls, time_series_dict):
        return cls(**time_series_dict)

    def _generate_dates_series(self, n_len_times = 2000, t_step = DEFAULT_time_step):
        if self._param_dates[0] == None :
            self._dates = np.array([datetime.fromtimestamp(t_step*k) for k in range(n_len_times)])
        else :
            dt, end, step = datetime(*self._param_dates[0]), datetime(*self._param_dates[1]), timedelta(seconds=self._param_dates[2])
            times_vir1 = []
            while dt < end:
                times_vir1.append(dt)
                dt += step
            self._dates = np.array(times_vir1)
    
    def _generate_dH_series(self):
        t_range = np.arange(len(self._dates))*self._param_dates[2]
        self._dH = self._param_dH[0]*np.sin(2*np.pi*t_range/self._param_dH[1]) + self._param_dH[2]

    def _generate_Temp_riv_series(self): # renvoie un signal sinusoïdal de temperature rivière
        if self._dates.any() == None :
            self._generate_dates_series()

        t_range = np.arange(len(self._dates))*self._param_dates[2]
        self._T_riv = self._param_T_riv[0]*np.sin(2*np.pi*t_range/self._param_T_riv[1]) + self._param_T_riv[2]

    def _generate_Temp_aq_series(self): # renvoie un signal sinusoïdal de temperature aquifère
        if self._dates.any() == None :
            self._generate_dates_series()

        t_range = np.arange(len(self._dates))*self._param_dates[2]
        self._T_aq = self._param_T_aq[0]*np.sin(2*np.pi*t_range/self._param_T_aq[1]) + self._param_T_aq[2]

    def _generate_T_riv_dH_series(self): # renvoie un signal sinusoïdal de différence de charge
        if self._dates.any() == None :
            self._generate_dates_series()
            
        if self._dH.any() == None :
            self._generate_dH_series()

        if self._T_riv.any() == None :
            self._generate_Temp_riv_series()

        self._T_river_dH_measures = list(zip(self._dates,list(zip(self._dH, self._T_riv))))

    def _generate_Shaft_Temp_series(self, n_sens_vir=N_SENSORS_SHAFT): # en argument n_sens_vir le nb de capteur (2 aux frontières et 3 inutiles à 0)
        # initialisation
        if self._dates.any() == None :
            self._generate_dates_series()
        self._T_Shaft = np.zeros((len(self._dates),n_sens_vir)) # le tableau qui accueille des données de températures de forçage
        
        if self._T_aq.any() == None :
            self._generate_Temp_aq_series()
        self._T_Shaft[:,n_sens_vir-1] = self._T_aq

        T_init = self._T_aq[0]
        self._T_Shaft[0,:-1]=T_init*np.ones(n_sens_vir-1) # pour les conditions initiales au niveau des capteurs T1, T2, T3, ... avant Taq
        self._T_Shaft[1:,]
        self._T_Shaft_measures = list(zip(self._dates, self._T_Shaft))
    
    def _generate_perturb_Shaft_Temp_series(self, n_sens_vir=N_SENSORS_SHAFT):
        if self._dates.any() == None :
            self._generate_dates_series()
        n_t = len(self._dates)

        if self._T_Shaft.any() == None :
            self._generate_Shaft_Temp_series()

        self._T_Shaft_perturb = np.zeros((n_t,n_sens_vir))
        self._T_Shaft_perturb = self._T_Shaft + normal(0,self._sigma_T, (n_t,n_sens_vir)) # perturbe toutes les températures
        self._T_Shaft_perturb_measures = list(zip(self._dates, self._T_Shaft_perturb)) # peut renvoyer un format T_measure pour un objet colonne

    def _generate_perturb_T_riv_dH_series(self):
        if self._dates.any() == None :
            self._generate_dates_series()
        n_t = len(self._dates)

        if self._T_riv.any() == None :
            self._generate_Temp_riv_series()
        self._T_riv_perturb = self._T_riv + normal(0,self._sigma_T, n_t)

        if self._dH.any() == None :
            self._generate_dH_series()
        self._dH_perturb = self._dH + normal(0,self._sigma_P, n_t)

        self._T_riv_dH_perturb = list(zip(self._dates,list(zip(self._dH_perturb, self._T_riv_perturb))))
    
    def _measures_column_one_layer(self, column, layer, nb_cell):
        column._compute_solve_transi_one_layer(layer, nb_cell)
        id_sensors = column.get_id_sensors()
        for i in range(id_sensors):
            self._T_Shaft[1:,i] = column._temp[1:,id_sensors[i]] # maj les températures émulée des capteurs à partir de t>0 (ie t=1)
        self._generate_Shaft_Temp_series()
    






