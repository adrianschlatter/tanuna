# -*- coding: utf-8 -*-
"""
Stabilization of a mode-locked laser using pole placement.

@author: Adrian Schlatter
"""
# Code Snippet 1 - Start %%%%%
from tanuna.examples.laser import NdYVO4Laser
from tanuna.sources import SourceConstant
import numpy as np
import tanuna as dyn

# Setup laser
# =============================================================================

Ppump = 0.1
NdYVO4 = NdYVO4Laser(Ppump=0.)


# Linearized
# =============================================================================

M, system_lin = NdYVO4.approximateLTI(Ppump)
# Add state outputs:
A, B, C, D = system_lin.ABCD
Toc = NdYVO4.Toc
C = np.matrix([[0, Toc],
               [1, 0],
               [0, 1]])
D = np.matrix(np.zeros((3, 1)))
system_lin = dyn.CT_LTI_System(A, B, C, D)
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
Ppump_assumed = rPp * Ppump
kr = rkr * (gamma**2 - nu)
k1 = rk1 * 2 * (gamma - NdYVO4.zeta(Ppump_assumed)) / NdYVO4.rho(Ppump_assumed)
k2 = rk2 * (gamma**2 - nu - 1.) / NdYVO4.rho(Ppump_assumed)

stateoutput = np.matrix([[1, 0, 0]])
K = np.matrix([[0, k1, k2]])
L = np.vstack([stateoutput, K])
summing = np.matrix([kr, -1])
stabilized_lin = L * system_lin * summing
stabilized_lin = dyn.feedback(stabilized_lin, Gout=(1,), Gin=(1,))
# Code Snippet 2 - End %%%%%


# Code Snippet 3 - Start %%%%%
class StateOutputLaser(NdYVO4Laser):
    """Same as NdYVO4Laser but outputs (Pout, g, P) instead of only Pout."""

    def __init__(self, Ppump=0.):
        super().__init__(Ppump=Ppump)
        self.shape = (3, 1)

    def g(self, t, s, u):
        """
        This is the output function of the CT_System and returns the
        output power of the laser. Despite its name, is *not* related
        to the laser's gain!
        """
        P, g = s
        return np.matrix([self.Toc * P, P, g])


so_NdYVO4 = StateOutputLaser(Ppump=0.0)
so_NdYVO4.s = np.matrix([[0.1, 0]]).T  # "noise photons"

P0, g0 = so_NdYVO4.steadystate((Ppump))

y0 = np.matrix([[0], [-P0], [-g0]])
stabilized = so_NdYVO4.offset_outputs(y0)
stabilized = stabilized.offset_inputs(Ppump)
Maugmented = np.eye(3)
Maugmented[1:, 1:] = M
stabilized = L * Maugmented * stabilized * summing
stabilized = dyn.feedback(stabilized, Gout=(1,), Gin=(1,))
stabilized = dyn.connect(stabilized, SourceConstant(y=np.matrix(0.)))
system = dyn.connect(so_NdYVO4, SourceConstant(y=np.matrix(Ppump)))
# Code Snippet 3 - End %%%%%
