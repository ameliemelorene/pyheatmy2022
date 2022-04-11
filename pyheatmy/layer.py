import numpy as np
from .params import Param

class Layer:
    def __init__(self, name: str, zHigh: float, zLow: float, k: float, n: float, lbda: float, rho: float):
        self.name = name
        self.zHigh = zHigh
        self.zLow = zLow
        self.params = Param(k, n, lbda, rho)

def layersListCreator(layersListInput):
    layersList = list()
    for name, zHigh, zLow, k, n, lbda, rho in layersListInput:
        layersList.append(Layer(name, zHigh, zLow, k, n, lbda, rho))
    return layersList

def getListParameters(layersList, nbCells: int):
    dz = layersList[-1].zLow / nbCells
    currentAltitude = dz/2
    listParameters = list()
    for layer in layersList:
        while currentAltitude < layer.zLow:
            listParameters.append((layer.params.moinslog10K, layer.params.n, layer.params.lambda_s, layer.params.rhos_cs))
            currentAltitude += dz
    listParameters = np.array(listParameters)
    return listParameters[:, 0], listParameters[:, 1], listParameters[:, 2], listParameters[:, 3]