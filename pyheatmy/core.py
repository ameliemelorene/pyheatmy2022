from multiprocessing.sharedctypes import Value
from typing import List, Sequence, Union
from random import random, choice
from operator import attrgetter
from numbers import Number
import sys

import numpy as np
from tqdm import trange
from scipy.interpolate import lagrange

from .params import  Param, ParamsPriors, Prior, PARAM_LIST
from .state import State, StateOld
from .checker import checker

from .utils import C_W, RHO_W, LAMBDA_W, compute_H, compute_T, compute_H_stratified, compute_T_stratified
from .layers import Layer, getListParameters, sortLayersList, AllPriors, LayerPriors


class Column:  # colonne de sédiments verticale entre le lit de la rivière et l'aquifère
    def __init__(
        self,
        river_bed: float,  # profondeur de la colonne en mètres
        # profondeur des capteurs de températures en mètres
        depth_sensors: Sequence[float],
        offset: float,  # correspond au décalage du capteur de température par rapport au lit de la rivière
        # liste contenant un tuple avec la date, la charge et la température au sommet de la colonne
        dH_measures: list,
        T_measures: list,  # liste contenant un tuple avec la date et la température aux points de mesure de longueur le nombre de temps mesuré
        sigma_meas_P: float,  # écart type de l'incertitude sur les valeurs de pression capteur
        sigma_meas_T: float,  # écart type de l'incertitude sur les valeurs de température capteur
    ):
        # ! Pour l'instant on suppose que les temps matchent
        self._times = [t for t, _ in dH_measures]
        # récupère la liste des charges de la riviière (au cours du temps)
        self._dH = np.array([d for _, (d, _) in dH_measures])
        # récupère la liste de température de la rivière (au cours du temps)
        self._T_riv = np.array([t for _, (_, t) in dH_measures])
        # récupère la liste de température de l'aquifère (au cours du temps)
        self._T_aq = np.array([t[-1] - 1 for _, t in T_measures])
        # récupère la liste de températures des capteurs (au cours du temps)
        self._T_measures = np.array([t[:-1] for _, t in T_measures])

        # décale d'un offset les positions des capteurs de température (aussi riviere)
        self._real_z = np.array([0] + depth_sensors) + offset
        # enlève l'offset sur la mesure de température rivière car cette mesure est prise dans le capteur pression
        self._real_z[0] -= offset

        self.depth_sensors = depth_sensors
        self.offset = offset

        self._layersList = None

        self._z_solve = None
        self._id_sensors = None
        self._temps = None
        self._H_res = None
        self._flows = None

        self._states = None
        self._initial_energies = None
        self._quantiles_temps = None
        self._quantiles_flows = None

    @classmethod
    def from_dict(cls, col_dict):
        return cls(**col_dict)

    def _check_layers(self, layersList):
        self._layersList = sortLayersList(layersList)

        if len(self._layersList) == 0:
            raise ValueError("Your list of layers is empty.")

        if self._layersList[-1].zLow != self._real_z[-1]:
            raise ValueError(
                "Last layer does not match the end of the column.")

    def _compute_solve_transi_one_layer(self, layer, nb_cells, verbose=True):
        dz = self._real_z[-1] / nb_cells
        self._z_solve = dz/2 + np.array([k*dz for k in range(nb_cells)])

        self._id_sensors = [np.argmin(np.abs(z - self._z_solve))
                            for z in self._real_z[1:-1]]

        all_dt = np.array([(self._times[j+1] - self._times[j]).total_seconds()
                           for j in range(len(self._times) - 1)])
        isdtconstant = np.all(all_dt == all_dt[0])

        H_init = self._dH[0] - self._dH[0] * self._z_solve / self._real_z[-1]
        H_aq = np.zeros(len(self._times))
        H_riv = self._dH

        lagr = lagrange(
            self._real_z, [self._T_riv[0], *
                           self._T_measures[0], self._T_aq[0]]
        )

        T_init = lagr(self._z_solve)
        T_riv = self._T_riv
        T_aq = self._T_aq

        moinslog10K, n, lambda_s, rhos_cs = layer.params

        if verbose:
            print("--- Compute Solve Transi ---",
                  f"One layer : moinslog10K = {moinslog10K}, n = {n}, lambda_s = {lambda_s}, rhos_cs = {rhos_cs}", sep="\n")

        heigth = abs(self._real_z[-1] - self._real_z[0])
        Ss = n / heigth

        H_res = compute_H(moinslog10K, Ss, all_dt,
                          isdtconstant, dz, H_init, H_riv, H_aq)

        T_res = compute_T(
            moinslog10K, n, lambda_s, rhos_cs, all_dt, dz, H_res, H_riv, H_aq, T_init, T_riv, T_aq
        )

        self._temps = T_res
        self._H_res = H_res

        nablaH = np.zeros((nb_cells, len(self._times)), np.float32)

        nablaH[0, :] = 2*(H_res[1, :] - H_riv)/(3*dz)

        for i in range(1, nb_cells - 1):
            nablaH[i, :] = (H_res[i+1, :] - H_res[i-1, :])/(2*dz)

        nablaH[nb_cells - 1, :] = 2*(H_aq - H_res[nb_cells - 2, :])/(3*dz)

        K = 10 ** - moinslog10K
        self._flows = -K * nablaH

        if verbose:
            print("Done.")

    def _compute_solve_transi_multiple_layers(self, layersList, nb_cells, verbose):
        dz = self._real_z[-1] / nb_cells
        self._z_solve = dz/2 + np.array([k*dz for k in range(nb_cells)])

        self._id_sensors = [np.argmin(np.abs(z - self._z_solve))
                            for z in self._real_z[1:-1]]

        all_dt = np.array([(self._times[j+1] - self._times[j]).total_seconds()
                           for j in range(len(self._times) - 1)])
        isdtconstant = np.all(all_dt == all_dt[0])

        H_init = self._dH[0] - self._dH[0] * self._z_solve / self._real_z[-1]
        H_aq = np.zeros(len(self._times))
        H_riv = self._dH

        lagr = lagrange(
            self._real_z, [self._T_riv[0], *
                           self._T_measures[0], self._T_aq[0]]
        )

        T_init = lagr(self._z_solve)
        T_riv = self._T_riv
        T_aq = self._T_aq

        moinslog10K_list, n_list, lambda_s_list, rhos_cs_list = getListParameters(
            layersList, nb_cells)

        heigth = abs(self._real_z[-1] - self._real_z[0])
        Ss_list = n_list / heigth

        if verbose:
            print("--- Compute Solve Transi ---")
            for layer in layersList:
                print(layer)

        H_res = compute_H_stratified(
            moinslog10K_list, Ss_list, all_dt, isdtconstant, dz, H_init, H_riv, H_aq)

        T_res = compute_T_stratified(moinslog10K_list, n_list, lambda_s_list,
                                     rhos_cs_list, all_dt, dz, H_res, H_riv, H_aq, T_init, T_riv, T_aq)

        self._temps = T_res
        self._H_res = H_res

        nablaH = np.zeros((nb_cells, len(self._times)), np.float32)

        nablaH[0, :] = 2*(H_res[1, :] - H_riv)/(3*dz)

        for i in range(1, nb_cells - 1):
            nablaH[i, :] = (H_res[i+1, :] - H_res[i-1, :])/(2*dz)

        nablaH[nb_cells - 1, :] = 2*(H_aq - H_res[nb_cells - 2, :])/(3*dz)

        K_list = 10 ** - moinslog10K_list

        flows = np.zeros((nb_cells, len(self._times)), np.float32)

        for i in range(nb_cells):
            flows[i, :] = - K_list[i]*nablaH[i, :]

        self._flows = flows

        if verbose:
            print("Done.")

    @checker
    def compute_solve_transi(self, layersList: Union[tuple, Sequence[Layer]], nb_cells: int, verbose=True):

        # List of layers or tuple ?
        if isinstance(layersList, tuple):
            layer = [Layer("Layer 1", self._real_z[-1],
                           layersList[0], layersList[1], layersList[2], layersList[3])]
            self.compute_solve_transi(layer, nb_cells, verbose)

        else:
            # Checking the layers are well defined
            self._check_layers(layersList)

            if len(self._layersList) == 1:
                self._compute_solve_transi_one_layer(
                    self._layersList[0], nb_cells, verbose)

            else:
                self._compute_solve_transi_multiple_layers(
                    self._layersList, nb_cells, verbose)

    @ compute_solve_transi.needed
    def get_id_sensors(self):
        return self._id_sensors

    @ compute_solve_transi.needed
    def get_RMSE(self):

        # Number of sensors (except boundary conditions : river and aquifer)
        nb_sensors = len(self._T_measures[0])

        # Number of times for which we have measures
        nb_times = len(self._T_measures)

        # Array of RMSE for each sensor
        list_RMSE = np.array([np.sqrt(np.sum((self.temps_solve[id, :] - temps_obs)**2) / nb_times)
                             for id, temps_obs in zip(self.get_id_sensors(), self._T_measures.T)])

        # Total RMSE
        total_RMSE = np.sqrt(np.sum(list_RMSE**2) / nb_sensors)

        return np.append(list_RMSE, total_RMSE)

    @ compute_solve_transi.needed
    def get_depths_solve(self):
        return self._z_solve

    depths_solve = property(get_depths_solve)

    def get_times_solve(self):
        return self._times

    times_solve = property(get_times_solve)

    @ compute_solve_transi.needed
    def get_temps_solve(self, z=None):
        if z is None:
            return self._temps
        z_ind = np.argmin(np.abs(self.depths_solve - z))
        return self._temps[z_ind, :]

    temps_solve = property(get_temps_solve)

    @ compute_solve_transi.needed
    def get_advec_flows_solve(self):
        return (
            RHO_W
            * C_W
            * self._flows
            * self.temps_solve
        )

    advec_flows_solve = property(get_advec_flows_solve)

    @ compute_solve_transi.needed
    def get_conduc_flows_solve(self):
        dz = self._z_solve[1] - self._z_solve[0]
        nb_cells = len(self._z_solve)

        _, n_list, lambda_s_list, _ = getListParameters(
            self._layersList, nb_cells)

        lambda_m_list = (
            n_list * (LAMBDA_W) ** 0.5
            + (1.0 - n_list) * (lambda_s_list) ** 0.5
        ) ** 2

        nablaT = np.zeros((nb_cells, len(self._times)), np.float32)

        nablaT[0, :] = 2*(self._temps[1, :] - self._T_riv)/(3*dz)

        for i in range(1, nb_cells - 1):
            nablaT[i, :] = (self._temps[i+1, :] - self._temps[i-1, :])/(2*dz)

        nablaT[nb_cells - 1, :] = 2 * \
            (self._T_aq - self._temps[nb_cells - 2, :])/(3*dz)

        conduc_flows = np.zeros((nb_cells, len(self._times)), np.float32)

        for i in range(nb_cells):
            conduc_flows[i, :] = lambda_m_list[i] * nablaT[i, :]

        return conduc_flows

    conduc_flows_solve = property(get_conduc_flows_solve)

    @ compute_solve_transi.needed
    def get_flows_solve(self, z=None):
        if z is None:
            return self._flows
        z_ind = np.argmin(np.abs(self.depths_solve - z))
        return self._flows[z_ind, :]

    flows_solve = property(get_flows_solve)

    
    def _compute_mcmc_deprecated(
        self,
        nb_iter: int,
        priors: dict,
        nb_cells: int,
        quantile: Union[float, Sequence[float]] = (0.05, 0.5, 0.95),
        verbose =True,
        sigma_temp_prior : Prior = None
        ):

        if isinstance(quantile, Number):
            quantile = [quantile]

        priors = ParamsPriors(
            [Prior(*args) for args in (priors[lbl]
                                       for lbl in PARAM_LIST)]  # usefull for optionnal arguments
        )

        ind_ref = [
            np.argmin(
                np.abs(
                    z - np.linspace(self._real_z[0], self._real_z[-1], nb_cells))
            )
            for z in self._real_z[1:-1]
        ]

        temp_ref = self._T_measures[:, :].T

        def compute_energy(temp: np.array, sigma_obs: float = 1):
            # norm = sum(np.linalg.norm(x-y) for x,y in zip(temp,temp_ref))
            norm = np.sum(np.linalg.norm(temp - temp_ref, axis=-1))
            return 0.5 * (norm / sigma_obs) ** 2

        def compute_acceptance(actual_energy: float, prev_energy: float, actual_sigma: float, prev_sigma: float, sigma_distrib):
            return  (prev_sigma/actual_sigma)**3*sigma_distrib(actual_sigma)/(sigma_distrib(prev_sigma))*np.exp((prev_energy - actual_energy))


        if verbose:
            print(
                "--- Compute Mcmc ---",
                "Priors :",
                *(f"    {prior}" for prior in priors),
                f"Number of cells : {nb_cells}",
                f"Number of iterations : {nb_iter}",
                "Launch Mcmc",
                sep="\n",
            )

        self._states = list()

        nb_z = np.linspace(self._real_z[0], self._real_z[-1], nb_cells).size
        _temps = np.zeros((nb_iter + 1, nb_z, len(self._times)), np.float32)
        _flows = np.zeros((nb_iter + 1, nb_z, len(self._times)), np.float32)

        for _ in trange(1000, desc="Init Mcmc ", file=sys.stdout):
            init_param = priors.sample()
            init_sigma_temp = sigma_temp_prior.sample()
            self.compute_solve_transi(init_param, nb_cells, verbose=False)

            self._states.append(
                StateOld(
                    params=init_param,
                    energy=compute_energy(
                        self.temps_solve[ind_ref, :], sigma_obs=init_sigma_temp),
                    ratio_accept=1,
                    sigma_temp  = init_sigma_temp
                )
            )

        self._initial_energies = [state.energy for state in self._states]
        self._states = [min(self._states, key=attrgetter("energy"))]

        _temps[0] = self.temps_solve
        _flows[0] = self.flows_solve

        for _ in trange(nb_iter, desc="Mcmc Computation ", file=sys.stdout):
            params = priors.perturb(self._states[-1].params)
            current_sigma_temp = sigma_temp_prior.perturb(self._states[-1].sigma_temp)
            self.compute_solve_transi(params, nb_cells, verbose=False)
            energy = compute_energy(
                self.temps_solve[ind_ref, :], sigma_obs = current_sigma_temp)
            ratio_accept = compute_acceptance(
                energy, self._states[-1].energy, current_sigma_temp, self._states[-1].sigma_temp, sigma_temp_prior.distrib)
            if random() < ratio_accept:
                self._states.append(
                    StateOld(
                        params=params,
                        energy=energy,
                        ratio_accept=ratio_accept,
                        sigma_temp = current_sigma_temp
                    )
                )
                _temps[_] = self.temps_solve
                _flows[_] = self.flows_solve
            else:
                self._states.append(self._states[-1])
                self._states[-1].ratio_accept = ratio_accept
                _temps[_] = _temps[_ - 1]
                _flows[_] = _flows[_ - 1]
        self.compute_solve_transi.reset()

        if verbose:
            print("Mcmc Done.\n Start quantiles computation")

        self._quantiles_temps = {
            quant: res
            for quant, res in zip(quantile, np.quantile(_temps, quantile, axis=0))
        }
        self._quantiles_flows = {
            quant: res
            for quant, res in zip(quantile, np.quantile(_flows, quantile, axis=0))
        }
        if verbose:
            print("Quantiles Done.")

    @checker
    def compute_mcmc(
        self,
        nb_iter: int,
        all_priors: Union[AllPriors, Sequence[ Union[LayerPriors,
                            Sequence[Union[str, float, Sequence[ Union[Prior, dict]]]]]]],
        nb_cells: int,
        quantile: Union[float, Sequence[float]] = (0.05, 0.5, 0.95),
        verbose=True,
        incertitudes = True,
        sigma_temp_prior : Prior = Prior((0.01, np.inf), 2, lambda x : 1/x)
    ):
        if isinstance(quantile, Number):
            quantile = [quantile]

        if not isinstance(all_priors, AllPriors) :
            all_priors = AllPriors(
                [LayerPriors(*args) for args in (layer for layer in all_priors)])

        ind_ref = [
            np.argmin(
                np.abs(
                    z - np.linspace(self._real_z[0], self._real_z[-1], nb_cells))
            )
            for z in self._real_z[1:-1]
        ]

        temp_ref = self._T_measures[:, :].T

        if incertitudes:
            def compute_energy(temp: np.array, sigma_obs: float = 1):
                # norm = sum(np.linalg.norm(x-y) for x,y in zip(temp,temp_ref))
                norm = np.sum(np.linalg.norm(temp - temp_ref, axis=-1))
                return 0.5 * (norm / sigma_obs) ** 2

            def compute_acceptance(actual_energy: float, prev_energy: float, actual_sigma: float, prev_sigma: float, sigma_distrib):
                return (prev_sigma/actual_sigma)**3*sigma_distrib(actual_sigma)/(sigma_distrib(prev_sigma))*np.exp(prev_energy - actual_energy)   # 3 -> np.size(self._T_measures), plus facile à calculer reparamétrisation du problème
        
        else:
            def compute_energy(temp: np.array, sigma_obs: float = 1):
                # norm = sum(np.linalg.norm(x-y) for x,y in zip(temp,temp_ref))
                norm = np.sum(np.linalg.norm(temp - temp_ref, axis=-1))
                return 0.5 * (norm) ** 2

            def compute_acceptance(actual_energy: float, prev_energy: float, actual_sigma: float, prev_sigma: float, sigma_distrib):
                return np.exp(prev_energy - actual_energy) 
    
        if verbose:
            print(
                "--- Compute Mcmc ---",
                "Priors :",
                *(f"    {prior}" for prior in all_priors),
                f"Number of cells : {nb_cells}",
                f"Number of iterations : {nb_iter}",
                "Launch Mcmc",
                sep="\n",
            )

        self._states = list()

        nb_z = np.linspace(self._real_z[0], self._real_z[-1], nb_cells).size
        _temps = np.zeros((nb_iter + 1, nb_z, len(self._times)), np.float32)
        _flows = np.zeros((nb_iter + 1, nb_z, len(self._times)), np.float32)

        for _ in trange(1000, desc="Init Mcmc ", file=sys.stdout):
            init_layers =  all_priors.sample()
            init_sigma_temp = sigma_temp_prior.sample()
            self.compute_solve_transi(init_layers, nb_cells, verbose=False)

            self._states.append(
                State(
                    layers=init_layers,
                    energy=compute_energy(
                        self.temps_solve[ind_ref, :], sigma_obs=init_sigma_temp),
                    ratio_accept=1,
                    sigma_temp  = init_sigma_temp
                )
            )

        self._initial_energies = [state.energy for state in self._states]
        self._states = [min(self._states, key=attrgetter("energy"))]

        _temps[0] = self.temps_solve
        _flows[0] = self.flows_solve

        for _ in trange(nb_iter, desc="Mcmc Computation ", file=sys.stdout):
            current_layers = all_priors.perturb(self._states[-1].layers)
            current_sigma_temp = sigma_temp_prior.perturb(self._states[-1].sigma_temp)
            self.compute_solve_transi(current_layers, nb_cells, verbose=False)
            energy = compute_energy(
                self.temps_solve[ind_ref, :], sigma_obs = current_sigma_temp)
            ratio_accept = compute_acceptance(
                energy, self._states[-1].energy, current_sigma_temp, self._states[-1].sigma_temp, sigma_temp_prior.density)
            if random() < ratio_accept:
                self._states.append(
                    State(
                        layers=current_layers,
                        energy=energy,
                        ratio_accept=ratio_accept,
                        sigma_temp = current_sigma_temp
                    )
                )
                _temps[_] = self.temps_solve
                _flows[_] = self.flows_solve
            else:
                self._states.append(self._states[-1])
                self._states[-1].ratio_accept = ratio_accept
                _temps[_] = _temps[_ - 1]
                _flows[_] = _flows[_ - 1]
        self.compute_solve_transi.reset()

        if verbose:
            print("Mcmc Done.\n Start quantiles computation")

        self._quantiles_temps = {
            quant: res
            for quant, res in zip(quantile, np.quantile(_temps, quantile, axis=0))
        }
        self._quantiles_flows = {
            quant: res
            for quant, res in zip(quantile, np.quantile(_flows, quantile, axis=0))
        }
        if verbose:
            print("Quantiles Done.")

    @ compute_mcmc.needed
    def get_depths_mcmc(self):
        return self._real_z  # plus cohérent que de renvoyer le time

    depths_mcmc = property(get_depths_mcmc)

    @ compute_mcmc.needed
    def get_times_mcmc(self):
        return self._times

    times_mcmc = property(get_times_mcmc)

    @ compute_mcmc.needed
    def sample_param(self):
        return choice([s.params for s in self._states])

    @ compute_mcmc.needed
    def get_best_param(self):
        """return the params that minimize the energy"""
        return min(self._states, key=attrgetter("energy")).params

    @ compute_mcmc.needed
    def get_all_params(self):
        return [s.params for s in self._states]

    all_params = property(get_all_params)

    @ compute_mcmc.needed
    def get_all_moinslog10K(self):
        return [s.params.moinslog10K for s in self._states]

    all_moinslog10K = property(get_all_moinslog10K)

    @ compute_mcmc.needed
    def get_all_n(self):
        return [s.params.n for s in self._states]

    all_n = property(get_all_n)

    @ compute_mcmc.needed
    def get_all_lambda_s(self):
        return [s.params.lambda_s for s in self._states]

    all_lambda_s = property(get_all_lambda_s)

    @ compute_mcmc.needed
    def get_all_rhos_cs(self):
        return [s.params.rhos_cs for s in self._states]

    all_rhos_cs = property(get_all_rhos_cs)

    @compute_mcmc.needed
    def get_all_sigma(self):
        return [s.sigma_temp for s in self._states]

    all_sigma = property(get_all_sigma)

    @ compute_mcmc.needed
    def get_all_energy(self):
        return self._initial_energies + [s.energy for s in self._states]

    all_energy = property(get_all_energy)

    @ compute_mcmc.needed
    def get_all_acceptance_ratio(self):
        return [s.ratio_accept for s in self._states]

    all_acceptance_ratio = property(get_all_acceptance_ratio)

    @ compute_mcmc.needed
    def get_temps_quantile(self, quantile):
        return self._quantiles_temps[quantile]

    @ compute_mcmc.needed
    def get_flows_quantile(self, quantile):
        return self._quantiles_flows[quantile]
