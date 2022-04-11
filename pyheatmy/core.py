from typing import Sequence, Union
from random import random, choice
from operator import attrgetter
from numbers import Number
import sys

import numpy as np
from tqdm import trange
from scipy.interpolate import lagrange

from .params import Param, ParamsPriors, Prior
from .state import State
from .checker import checker
from .utils import C_W, RHO_W, LAMBDA_W, PARAM_LIST, compute_H, compute_T


class Column:#colonne de sédiments verticale entre le lit de la rivière et l'aquifère
    def __init__(
        self,
        river_bed: float,#profondeur de la colonne en mètres
        depth_sensors: Sequence[float],#profondeur des capteurs de températures en mètres
        offset: float,#correspond au décalage du capteur de température par rapport au lit de la rivière
        dH_measures: list,#liste contenant un tuple avec la date, la charge et la température au sommet de la colonne
        T_measures: list,#liste contenant un tuple avec la date et la température aux points de mesure de longueur le nombre de temps mesuré
        sigma_meas_P: float,#écart type de l'incertitude sur les valeurs de pression capteur
        sigma_meas_T: float,#écart type de l'incertitude sur les valeurs de température capteur
    ):
        # ! Pour l'instant on suppose que les temps matchent
        self._times = [t for t, _ in dH_measures]#récupère la liste des temps
        self._dH = np.array([d for _, (d, _) in dH_measures])#récupère le tableau des charges au niveau de la riviière (=sommet de la colonne) (au cours du temps)
        self._T_riv = np.array([t for _, (_, t) in dH_measures])#récupère le tableau de température de la rivière (=sommet de la colonne) (au cours du temps)
        self._T_aq = np.array([t[-1] - 1 for _, t in T_measures])#récupère le tableau de température de l'aquifère (au cours du temps)
        self._T_measures = np.array([t[:-1] for _, t in T_measures])#récupère le tableau de températures des capteurs (au cours du temps)

        self._real_z = np.array([0] + depth_sensors) + offset #décale d'un offset les positions des capteurs de température (aussi riviere)
        self._real_z[0] -= offset #enlève l'offset sur la mesure de température rivière car cette mesure est prise dans le capteur pression
      
        self.depth_sensors = depth_sensors
        self.offset = offset

        self._param = None#les paramètres moinslog10K,n,lambda_s,rhos_cs
        self._z_solve = None#le tableau contenant la profondeur du milieu des cellules
        self._temps = None#le tableau contenant les températures à tout temps et à toute profondeur (lignes : températures) (colonnes : temps)
        self._H_res = None#le tableau contenant les charges à tout temps et à toute profondeur (lignes : charges) (colonnes : temps)
        self._flows = None#le tableau contenant le débit spécifique à tout temps et à toute profondeur (lignes : débit) (colonnes : temps)

        self._states = None# liste contenant des objets de classe état et de longueur le nombre d'états acceptés par la MCMC (<=nb_iter), passe à un moment par une longueur de 1000 pendant l'initialisation de MCMC
        self._quantiles_temps = None#dictionnaire indexé par les quantiles (0.05,0.5,0.95) à qui on a associe un array de deux dimensions : dimension 1 les profondeurs, dimension 2 : liste des valeurs de températures associées au quantile, de longueur les temps de mesure 
        self._quantiles_flows = None#dictionnaire indexé par les quantiles (0.05,0.5,0.95) à qui on a associe un array de deux dimensions : dimension 1 les profondeurs, dimension 2 : liste des valeurs de débits spécifiques associés au quantile, de longueur les temps de mesure

    @classmethod
    def from_dict(cls, col_dict):
        return cls(**col_dict)

    @checker
    def compute_solve_transi(self, param: tuple, nb_cells: int, verbose=True):#résout les calculs de température au cours du temps en régime transitoire
        """s'applique à la colonne, paramètres contient moinslog10K,n,lambda_s,rhos_cs,nb_cells est le nombre de cellules pour la discrétisation en profondeur,verbose affiche les textes explicatifs"""
        if not isinstance(param, Param):#si param n'est pas de type Param
            param = Param(*param)#crée param de classe Param à partir des valeurs données dans param en entrée
        self._param = param

        if verbose:
            print("--- Compute Solve Transi ---", self._param, sep="\n")

        dz = self._real_z[-1] / nb_cells  #profondeur d'une cellule

        self._z_solve = dz/2 + np.array([k*dz for k in range(nb_cells)])

        K = 10 ** -param.moinslog10K
        heigth = abs(self._real_z[-1] - self._real_z[0])
        Ss = param.n / heigth # l'emmagasinement spécifique = porosité sur la hauteur

        all_dt = np.array([(self._times[j+1] - self._times[j]).total_seconds()
                           for j in range(len(self._times) - 1)])#le tableau des pas de temps (dépend des données d'entrée)

        isdtconstant = np.all(all_dt == all_dt[0])

        H_init = np.linspace(self._dH[0], 0, nb_cells)#initialise (t=0) les charges des cellules en prenant 0 comme référence au niveau de l'aquifère
        H_aq = np.zeros(len(self._times))#fixe toutes les charges de l'aquifère à 0 (à tout temps)
        H_riv = self._dH#self.dH contient déjà les charges de la rivière à tout temps, stocke juste dans une variable locale

        H_res = compute_H(K, Ss, all_dt, isdtconstant, dz, H_init, H_riv, H_aq)#calcule toutes les charges à tout temps et à toute profondeur

        # temps[0] = np.linspace(self._T_riv[0], self._T_aq[0], nb_cells)
        lagr = lagrange(
            self._real_z, [self._T_riv[0], *self._T_measures[0], self._T_aq[0]]
        )#crée le polynome interpolateur de lagrange faisant coincider les températures connues à la profondeur réelle

        T_init = lagr(self._z_solve)#crée les températures initiales (t=0) sur toutes les profondeurs (milieu des cellules)
        T_riv = self._T_riv
        T_aq = self._T_aq

        T_res = compute_T(
            param.moinslog10K, param.n, param.lambda_s, param.rhos_cs, all_dt, dz, H_res, H_riv, H_aq, T_init, T_riv, T_aq
        )#calcule toutes les températures à tout temps et à toute profondeur

        self._temps = T_res
        self._H_res = H_res#stocke les résultats

        nablaH = np.zeros((nb_cells, len(self._times)), np.float32)#création d'un tableau du gradient de la charge selon la profondeur, calculé à tout temps

        nablaH[0, :] = 2*(H_res[1, :] - H_riv)/(3*dz)

        for i in range(1, nb_cells - 1):
            nablaH[i, :] = (H_res[i+1, :] - H_res[i-1, :])/(2*dz)

        nablaH[nb_cells - 1, :] = 2*(H_aq - H_res[nb_cells - 2, :])/(3*dz)

        self._flows = -K * nablaH#calcul du débit spécifique

        if verbose:
            print("Done.")

    @compute_solve_transi.needed#erreur si pas déjà éxécuté compute_solve_transi, sinon l'attribut pas encore affecté à une valeur
    def get_depths_solve(self):
        return self._z_solve

    depths_solve = property(get_depths_solve)
#récupération de l'attribut _z_solve 

    def get_times_solve(self):
        return self._times

    times_solve = property(get_times_solve)
#récupération de l'attribut _times

    @compute_solve_transi.needed#erreur si pas déjà éxécuté compute_solve_transi, sinon l'attribut pas encore affecté à une valeur
    def get_temps_solve(self, z=None):
        if z is None:
            return self._temps
        z_ind = np.argmin(np.abs(self.depths_solve - z))
        return self._temps[z_ind, :]

    temps_solve = property(get_temps_solve)
#récupération des températures au cours du temps à toutes les profondeurs (par défaut) ou bien à une profondeur donnée

    @compute_solve_transi.needed#erreur si pas déjà éxécuté compute_solve_transi, sinon l'attribut pas encore affecté à une valeur
    def get_advec_flows_solve(self):
        return (
            RHO_W
            * C_W
            * self._flows
            * self.temps_solve
        )

    advec_flows_solve = property(get_advec_flows_solve)
#récupération des flux advectifs = masse volumnique*capacité calorifique*débit spécifique*température

    @compute_solve_transi.needed#erreur si pas déjà éxécuté compute_solve_transi, sinon l'attribut pas encore affecté à une valeur
    def get_conduc_flows_solve(self):
        lambda_m = (
            self._param.n * (LAMBDA_W) ** 0.5
            + (1.0 - self._param.n) * (self._param.lambda_s) ** 0.5
        ) ** 2 #conductivité thermique du milieu poreux équivalent

        dz = self._z_solve[1] - self._z_solve[0]#pas en profondeur
        nb_cells = len(self._z_solve)
        #création du gradient de température
        nablaT = np.zeros((nb_cells, len(self._times)), np.float32)

        nablaT[0, :] = 2*(self._temps[1, :] - self._T_riv)/(3*dz)

        for i in range(1, nb_cells - 1):
            nablaT[i, :] = (self._temps[i+1, :] - self._temps[i-1, :])/(2*dz)

        nablaT[nb_cells - 1, :] = 2 * \
            (self._T_aq - self._temps[nb_cells - 2, :])/(3*dz)

        return lambda_m * nablaT

    conduc_flows_solve = property(get_conduc_flows_solve)
#récupération des flux conductifs = conductivité*gradient(T)

    @compute_solve_transi.needed#erreur si pas déjà éxécuté compute_solve_transi, sinon l'attribut pas encore affecté à une valeur
    def get_flows_solve(self, z=None):
        if z is None:
            return self._flows#par défaut, retourne le tableau des débits spécifiques
        z_ind = np.argmin(np.abs(self.depths_solve - z))
        return self._flows[z_ind, :]#sinon ne les retourne que pour la profondeur choisie

    flows_solve = property(get_flows_solve)
#récupération des débits spécifiques au cours du temps à toutes les profondeurs (par défaut) ou bien à une profondeur donnée

    @checker
    def compute_mcmc(
        self,#la colonne
        nb_iter: int, 
        priors: dict, #dictionnaire défini dans params.py, contentant écart type et range si on considère une distribution uniforme, contenant aussi fonction de répartition sinon
        nb_cells: int, #le nombre de cellules de la colonne
        quantile: Union[float, Sequence[float]] = (0.05, 0.5, 0.95), #les quantiles pour l'affichage de stats sur les valeurs de température
        verbose=True, #affiche texte explicatifs ou non
    ):
        if isinstance(quantile, Number):#si quantile est de type nombre, le transforme en liste, vérifier les histoires de type avec Union[float, Sequence[float]] tout de même
            quantile = [quantile]

        priors = ParamsPriors(
            [Prior((a, b), c) for (a, b), c in (priors[lbl]
                                                for lbl in PARAM_LIST)]
        );#va rechanger avec les modifs de valentin, je recommenterai après avoir rajouté les modifs de valentin à mon main

        ind_ref = [
            np.argmin(
                np.abs(
                    z - np.linspace(self._real_z[0], self._real_z[-1], nb_cells))
            )#renvoie la position de la celulle dont le milieu est le plus proche de la position du capteur de température 
            for z in self._real_z[1:-1]#pour les emplacements des 4 capteurs de température
        ]
#liste des indices des cellules contenant les capteur de température : exigence de l'IHM
        temp_ref = self._T_measures[:, :].T#prend la transposée pour avoir en ligne les points de mesures et en colonnes les temps (par souci de cohérence avec les tableaux de résultats de la simulation)

        def compute_energy(temp: np.array, sigma_obs: float = 1):#sigma vaut 1 quand pas d'incertitude sur valeur température
            # norm = sum(np.linalg.norm(x-y) for x,y in zip(temp,temp_ref))
            norm = np.sum(np.linalg.norm(temp - temp_ref, axis=-1))
            return 0.5 * (norm / sigma_obs) ** 2
            #énergie definit par 1/2sigma²||T-Tref||²+ln(T), ici on ne cherche pas à minimiser le terme en ln car il est constant
            #l'énergie se stabilise quand la chaîne de Markov rentre en régime stationnaire

        def compute_acceptance(actual_energy: float, prev_energy: float):
            return min(1, np.exp((prev_energy - actual_energy) / len(self._times) ** 1))
            # probabilité d'acceptation      


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

        nb_z = np.linspace(self._real_z[0], self._real_z[-1], nb_cells).size#nombre de profondeurs (=nb.cells)#pas sur de comprendre l'intérêt
        _temps = np.zeros((nb_iter + 1, nb_z, len(self._times)), np.float32)#tableau tri-dimensionnel de températures des différentes profondeurs en fonction du temps à chaque étape de MCMC
        _flows = np.zeros((nb_iter + 1, nb_z, len(self._times)), np.float32)#tableau tri-dimensionnel de débits spécifiques des différentes profondeurs en fonction du temps à chaque étape de MCMC

#crée un état initial avec des valeurs de températures au niveau des 4 capteurs
        for _ in trange(1000, desc="Init Mcmc ", file=sys.stdout):#initialisation des tableaux de résultats de la MCMC, 1000 nombre arbitraire du TP
            init_param = priors.sample_params()#modifié car pas que des lois uniformes maintenant
            self.compute_solve_transi(init_param, nb_cells, verbose=False)
            #fait tourner 1000 fois le modèle direct avec les paramètres initiaux
            self._states.append(
                State(
                    params=init_param,
                    energy=compute_energy(self.temps_solve[ind_ref, :]),
                    ratio_accept=1,
                )
            )
        #self._states de longueur 1000 à la fin de la boucle for

        self._states = [min(self._states, key=attrgetter("energy"))]#self._states de longueur 1 qu'on prend en état initial, celui qui minimise l'énergie (état stationnaire) de la simulation

        _temps[0] = self.temps_solve
        _flows[0] = self.flows_solve
        #initalise les températures et les débits spécifiques avec les valeurs obtenues au bout de la dernière simulation de l'état initial

        #implémentation de la MCMC sur le nombre d'itérations souhaitées
        for _ in trange(nb_iter, desc="Mcmc Computation ", file=sys.stdout):
            params = priors.perturb(self._states[-1].params)#perturbe les paramètres précédents tout en respectant le prior
            self.compute_solve_transi(params, nb_cells, verbose=False)
            energy = compute_energy(self.temps_solve[ind_ref, :])#calcule les énergies au niveau des capteurs de températures, pourquoi?
            ratio_accept = compute_acceptance(energy, self._states[-1].energy)#calcul de la probabilité d'acceptation
            if random() < ratio_accept:
                self._states.append(
                    State(
                        params=params,
                        energy=energy,
                        ratio_accept=ratio_accept,
                    )
                )#si décide de conserver le nouvel état (cf cours sur MCMC), on rajoute notre nouvel état au tableau des états
                _temps[_] = self.temps_solve
                _flows[_] = self.flows_solve
                #on ajoute dans ce cas les nouvelles températures et débits spécifiques à nos tableaux de résultats
            else:
                self._states.append(self._states[-1])
                self._states[-1].ratio_accept = ratio_accept
                _temps[_] = _temps[_ - 1]
                _flows[_] = _flows[_ - 1]
                #si refusé, on recopie l'état précédent, en changeant la probabilité d'acceptation. On conserve les valeurs de températures et débits spécifiques précédents
        self.compute_solve_transi.reset()#restaure les valeurs par défaut de compute_solve_transi

        if verbose:
            print("Mcmc Done.\n Start quantiles computation")

#création des deux dictionnaires de quantile
        self._quantiles_temps = {
            quant: res
            for quant, res in zip(quantile, np.quantile(_temps, quantile, axis=0))
        }#axis=0 fait calculer en 'moyennant' sur les étapes de la MCMC
        self._quantiles_flows = {
            quant: res
            for quant, res in zip(quantile, np.quantile(_flows, quantile, axis=0))
        }#axis=0 fait calculer en 'moyennant' sur les étapes de la MCMC
        if verbose:
            print("Quantiles Done.")

    @compute_mcmc.needed#erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    def get_depths_mcmc(self):
        return self._z_solve#avant il y avait écrit return self._times, corrigé en self.

    depths_mcmc = property(get_depths_mcmc)

    @compute_mcmc.needed#erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    def get_times_mcmc(self):
        return self._times

    times_mcmc = property(get_times_mcmc)

    @compute_mcmc.needed#erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    def sample_param(self):
        return choice([s.params for s in self._states])#retourne aléatoirement un des couples de paramètres parlesquels est passé la MCMC

    @compute_mcmc.needed#erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    def get_best_param(self):
        """return the params that minimize the energy"""
        return min(self._states, key=attrgetter("energy")).params#retourne le couple de paramètres minimisant l'énergie par lequels est passé la MCMC

    @compute_mcmc.needed#erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    def get_all_params(self):
        return [s.params for s in self._states]#retourne tous les couples de paramètres par lesquels est passé la MCMC

    all_params = property(get_all_params)

    @compute_mcmc.needed#erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    def get_all_moinslog10K(self):
        return [s.params.moinslog10K for s in self._states]#retourne toutes les valeurs de moinslog10K (K : perméabilité) par lesquels est passé la MCMC

    all_moinslog10K = property(get_all_moinslog10K)

    @compute_mcmc.needed#erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    def get_all_n(self):
        return [s.params.n for s in self._states]#retourne toutes les valeurs de n (n : porosité) par lesquels est passé la MCMC

    all_n = property(get_all_n)

    @compute_mcmc.needed#erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    def get_all_lambda_s(self):
        return [s.params.lambda_s for s in self._states]#retourne toutes les valeurs de lambda_s (lambda_s : conductivité thermique du solide) par lesquels est passé la MCMC

    all_lambda_s = property(get_all_lambda_s)

    @compute_mcmc.needed#erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    def get_all_rhos_cs(self):
        return [s.params.rhos_cs for s in self._states]#retourne toutes les valeurs de rho_cs (rho_cs : produite de la densité par la capacité calorifique spécifique du solide) par lesquels est passé la MCMC

    all_rhos_cs = property(get_all_rhos_cs)

    @compute_mcmc.needed#erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    def get_all_energy(self):
        return [s.energy for s in self._states]#retourne toutes les valeurs des énergies par lesquels est passé la MCMC

    all_energy = property(get_all_energy)

    @compute_mcmc.needed
    def get_all_acceptance_ratio(self):#retourne toutes les valeurs de probabilité d'acceptation par lesquels est passé la MCMC
        return [s.ratio_accept for s in self._states]

    all_acceptance_ratio = property(get_all_acceptance_ratio)

    @compute_mcmc.needed#erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    def get_temps_quantile(self, quantile):
        return self._quantiles_temps[quantile]
        #retourne les valeurs des températures en fonction du temps selon le quantile demandé

    @compute_mcmc.needed#erreur si pas déjà éxécuté compute_mcmc, sinon l'attribut pas encore affecté à une valeur
    def get_flows_quantile(self, quantile):
        return self._quantiles_flows[quantile]
        #retourne les valeurs des débits spécifiques en fonction du temps selon le quantile demandé