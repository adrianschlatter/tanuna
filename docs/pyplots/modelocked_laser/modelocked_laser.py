# -*- coding: utf-8 -*-
"""
Demonstrating mode-locked laser.

@author: Adrian Schlatter
"""
# Code Snippet 1 - Start %%%%%
from tanuna.examples.laser import NdYVO4Laser
from tanuna.sources import SourceConstant
from tanuna import connect
import numpy as np

Ppump = 0.1
NdYVO4 = NdYVO4Laser(Ppump)
pump = SourceConstant(y=np.matrix(Ppump))
pumped_NdYVO4 = connect(NdYVO4, pump)

# ODE solving
# =============================================================================

Psteady, gsteady = NdYVO4.steadystate(Ppump)
t = np.arange(35000) * NdYVO4.TR

Pout, state = pumped_NdYVO4(t, return_state=True, method='DOP853')
P, g = state

# Code Snippet 1 - End %%%%%
