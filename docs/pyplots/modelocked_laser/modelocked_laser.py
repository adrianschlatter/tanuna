# -*- coding: utf-8 -*-
"""
Demonstrating mode-locked laser.

@author: Adrian Schlatter
"""
# Code Snippet 1 - Start %%%%%
from tanuna.examples.laser import NdYVO4Laser
import numpy as np

Ppump = 0.1
NdYVO4 = NdYVO4Laser(Ppump)

# ODE solving
# =============================================================================

Psteady, gsteady = NdYVO4.steadystate()
t = np.arange(6000) * NdYVO4.TR * 5
P, g = np.zeros(t.shape), np.zeros(t.shape)

for i in range(len(t)):
    P[i], g[i] = NdYVO4.integrate(t[i])

# Code Snippet 1 - End %%%%%
