import numpy as np
from .params import Param


class Layer(object):
    def __new__(cls, name: str, zHigh: float, zLow: float, moinslog10K: float, n: float, lambda_s: float, rhos_cs: float):
        if zHigh < zLow:
            return object.__new__(cls)
        else:
            raise ValueError("zHigh must be less than zLow")

    def __init__(self, name: str, zHigh: float, zLow: float, moinslog10K: float, n: float, lambda_s: float, rhos_cs: float):
        self.name = name
        self.zHigh = zHigh
        self.zLow = zLow
        self.params = Param(moinslog10K, n, lambda_s, rhos_cs)

    def __repr__(self) -> str:
        return self.name + f" : {self.zHigh} m - {self.zLow} m. " + self.params.__repr__()


def layersListCreator(layersListInput):
    layersList = list()
    for name, zHigh, zLow, moinslog10K, n, lambda_s, rhos_cs in layersListInput:
        layersList.append(
            Layer(name, zHigh, zLow, moinslog10K, n, lambda_s, rhos_cs))
    return layersList


def sortLayersList(layersList):
    """
    Return a sorted list of layers (sorted by zHigh)
    """
    return sorted(layersList, key=lambda x: x.zHigh)


def getListParameters(layersList, nbCells: int):
    dz = layersList[-1].zLow / nbCells
    currentAltitude = dz/2
    listParameters = list()
    for layer in layersList:
        while currentAltitude < layer.zLow:
            listParameters.append(
                [layer.params.moinslog10K, layer.params.n, layer.params.lambda_s, layer.params.rhos_cs])
            currentAltitude += dz
    listParameters = np.array(listParameters)
    return listParameters[:, 0], listParameters[:, 1], listParameters[:, 2], listParameters[:, 3]
