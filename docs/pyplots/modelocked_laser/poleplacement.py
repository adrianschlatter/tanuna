# -*- coding: utf-8 -*-
"""
Stabilization of a mode-locked laser using pole placement.

@author: Adrian Schlatter
"""
# Code Snippet 1 - Start %%%%%
from tanuna.examples.laser import NdYVO4Laser
import numpy as np
import tanuna as dyn

# Setup laser
# =============================================================================

Ppump = 0.1
NdYVO4 = NdYVO4Laser(Ppump)


# Linearized
# =============================================================================

M, system = NdYVO4.approximateLTI()
# Add state outputs:
A, B, C, D = system.ABCD
Toc = NdYVO4.Toc
C = np.matrix([[0, Toc],
               [1, 0],
               [0, 1]])
D = np.matrix(np.zeros((3, 1)))
system = dyn.CT_LTI_System(A, B, C, D)
# Code Snippet 1 - End %%%%%

# Control
# =============================================================================

# Code Snippet 2 - Start %%%%%
# Where we want the poles to be:
gamma = 0.05
nu = -1.
# => poles will be at -gamma * w0 +/- sqrt(nu) * w0 = -0.05 * w0 +/- i * w0


# We assume that the controller is not calibrated perfectly.
#   a) The assumed pump power is factor rPp from real pump power
#   b) The implemented feedback is factors rkr, rk1, rk2 from calculated values
# Note how large we choose the errors!
rPp = 1.5
rkr = 0.8
rk1 = 0.8
rk2 = 0.8

# Calculate and apply feedback:
NdYVO4.PP = rPp * NdYVO4.PP
kr = rkr * (gamma**2 - nu)
k1 = rk1 * 2. * (gamma - NdYVO4.zeta()) / NdYVO4.rho()
k2 = rk2 * (gamma**2 - nu - 1.) / NdYVO4.rho()

stateoutput = np.matrix([[1, 0, 0]])
K = np.matrix([[0, k1, k2]])
L = np.vstack([stateoutput, K])
summing = np.matrix([kr, -1])
stabilized = L * system * summing
stabilized = dyn.connect(stabilized, stabilized, Gout=(1,), Hin=(1,))
# Code Snippet 2 - End %%%%%
