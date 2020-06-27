# Gaussian-Process-Scenario-Propagation-with-GPy
Propagation of uncertainty using Gaussian process can be a challenging task. In this work, scenarios are sampled
from a Gaussian process, a more natural way to compute the moments in future instances (See [1]). GP packages like GPy [2] are great for automating Gaussian process regression, however they lack in  uncertainty propagation. 
This repo contain GPy implementation of Scenario-based uncertainty propagation using GPy, applied in a photo-production of phycocyanin synthesized by cyanobacterium Arthrospira platensis.

[1] J. Umlauft, T. Beckers and S. Hirche, Scenario-based Optimal Control for Gaussian Process State Space Models, 2018 European Control Conference (ECC), Limassol, 2018, pp. 1386-1392, doi: 10.23919/ECC.2018.8550458.

[2] GPy, GPy: A Gaussian process framework in python, 2018, http://github.com/SheffieldML/GPy
