# PyHeatMy2022

This version of pyheatmy2 develops on the first PyHeatMy software developped by Mathis Bourdin & Youri Tchouboukoff in 2021 under a MIT license (https://github.com/mathisbrdn/pyheatmy/). In 2022, it is developped in the frame of a learn by doing class of MINES Paris, PSL University under the supervision of Nicolas Flipo and Thomas Romary.

The software calculates water and heat fluxes in a 1D riverbed. It is based on a finite volume formulation of the diffusivity and heat transport equations in porous media. The boundary conditions are hydraulic heads and temperatures at the edge of the column. The software also performs a bayesian inversion of the physical and thermal properties of the equivalent medium. It is based on temperature time series at various elevations below the riverbed. The inversion is based on a Monte Carlo Markov Chain approac and also provides uncertainties on temperatures at the measurment locations.

We do not guarantee any reliability on the resulting values of the calculations, however the data format will remain constant during their full implementation. Think of this code as a template that will remain persistent when the features are reliable in their results.

## Installation :

```sh
pip install -r requirements.txt
pip install -e .
```

## Examples :

In the repository there is a demo file in notebook format : [demo.ipynb]blob/master/demo.ipynb).


## Please note :

This library implements in addition to the api of the MOLONARI project property for most results.
We advise you not to use them if you want to use another pyheat library easily.
Each property has its own public getter in *get_nameproperty* format.

To ensure consistent results, a checker is added where the user cannot call results if he has not executed the corresponding methods. In particular, if you run a Bayesian inversion the results of the transient model must be recalculated.

***

## License

Eclipse Public Licence v2.0

