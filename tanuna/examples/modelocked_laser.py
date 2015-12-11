# -*- coding: utf-8 -*-
"""
[ Describe here what this script / module does ]

@author: Adrian Schlatter
"""

import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import ode
from scipy.optimize import brentq
import constants as cn


class Laser(ode):

    def __init__(self, loss, TR, tauL, eta, EsatL, DR, EsatA, PP0, g0=0.):
        self.loss = loss
        self.TR = TR
        self.g0 = g0
        self.tauL = tauL
        self.eta = eta
        self.EsatL = EsatL
        self.DR = DR
        self.EsatA = EsatA
        self.PP0 = PP0
        self.PP = PP0
        super().__init__(self.f, jac=self.grad_f)

#        P0 = self.Psteady
        P0 = 10.
#        g0 = loss + self.qP(TR*P0)
        g0 = 0.08
        t0 = 0.
        self.set_initial_value([P0, g0], t0)
        self.set_integrator('dopri5')

    def qP(self, EP):
        S = EP / self.EsatA
        return(self.DR / S * (1. - np.exp(-S)))

    def dqP_dEP(self, EP):
        EsatA = self.EsatA
        S = EP / EsatA
        return(self.DR / EP * (np.exp(-S) - 1 / S + np.exp(-S) / S))

    def Pdot(self, P, g):
        P = np.array(P)
        g = np.array(g)

        EP = P * self.TR
        return(np.where(P > 0, (g - self.loss - self.qP(EP)) / self.TR * P,
                        np.zeros(P.shape)))

    def gdot(self, P, g):
        spontaneous = (self.g0 - g) / self.tauL
        stimulated = -P * g / self.EsatL
        pump = self.eta * self.PP / self.EsatL
        return(spontaneous + stimulated + pump)

    def f(self, t, s):
        print('f called')
        P, g = s
        sdot = [self.Pdot(P, g), self.gdot(P, g)]
        return(sdot)

    def grad_f(self, t, s):
        print('grad_f called')
        P, g = s
        loss = self.loss
        TR = self.TR
        EP = P * TR
        qP = self.qP(EP)
        EsatL = self.EsatL
        tauL = self.tauL

        dfP_dP = (g - loss - qP) / TR - EP * self.dqP_dEP(EP)
        dfP_dg = P / TR
        dfg_dP = g / EsatL
        dfg_dg = -1 / tauL - P / EsatL

        return([[dfP_dP, dfP_dg], [dfg_dP, dfg_dg]])

    @property
    def Psteady(self):
        EsatL = self.EsatL
        TR = self.TR
        tauL = self.tauL
        PP = self.PP
        loss = self.loss
        DR = self.DR

        # Solve Pss = -EsatL/tauL + eta*PP / (loss + qp(Pss*TR))
        # 1. determine boundaries for Pss:
        threshold = EsatL / tauL
        PssLow = eta * PP / (loss + DR) - threshold
        PssHigh = eta * PP / loss - threshold
#        PssLow = 1e-6
#        PssHigh = PP / loss

        # 2. Find root:
        Pss = brentq(lambda P: -P - EsatL / tauL + eta * PP /
                     (loss + self.qP(P*TR)), PssLow, PssHigh)
        return(Pss)

    @property
    def gsteady(self):
        return(self.loss + self.qP(self.TR * self.Psteady))

    @property
    def stable(self):
        tauL, EsatL, TR = self.tauL, self.EsatL, self.TR
        P = self.Psteady
        epsilon = 1 / tauL + P / EsatL
        alpha = -P * self.dqP_dEP(P * TR)
        return(epsilon > alpha)


# Laser Parameters:
# =============================================================================

PP0 = 0.5

tauL = 90e-6
TR = 10e-9
FsatA = 60e-6/1e-4
wA = 140e-6
DR = 1.7e-2
loss = 9e-2+1.3e-2
wavelength = 1064e-9
sigmaEm = 114e-20*1e-4
wL = 62e-6
eta = 0.6

# Derived Values:
# =============================================================================

nuL = cn.c / wavelength
EsatL = np.pi*wL**2*cn.h*nuL / (2*sigmaEm)
EsatA = FsatA*np.pi*wA**2

# Initialize Laser:
# =============================================================================

NdYVO4 = Laser(loss, TR, tauL, eta, EsatL, DR, EsatA, PP0)

# Streamplot
# =============================================================================
pl.close('all')

Psteady = NdYVO4.Psteady
gsteady = NdYVO4.gsteady
x = np.linspace(0., 2 * Psteady)
y = np.linspace(0., 0.2)
X, Y = np.meshgrid(x, y)
U, V = NdYVO4.f(0, [X, Y])

pl.figure()
pl.title('State Propagation')
pl.xlabel('Intra-Cavity Power (W)')
pl.ylabel('Gain (1)')
pl.streamplot(x, y, U, V)
pl.plot([Psteady], [gsteady], r'ko')
pl.xlim([x[0], x[-1]])
pl.ylim([y[0], y[-1]])

# State Propagation
# =============================================================================

t = np.arange(100) * TR * 10
P, g = np.zeros(t.shape), np.zeros(t.shape)

for i in range(len(t)):
    P[i], g[i] = NdYVO4.integrate(t[i])
    print(NdYVO4.successful())
#    if not NdYVO4.successful():
#        break

pl.plot(P, g, r'r.-')
#print(NdYVO4.successful())


# Linearized
# =============================================================================

alpha = -Psteady * NdYVO4.dqP_dEP(TR * Psteady)
beta = Psteady / TR
gamma = gsteady / EsatL
epsilon = 1 / tauL + Psteady / EsatL
rho = eta / EsatL

A = np.matrix([[alpha, beta], [-gamma, -epsilon]])
B = np.matrix([[0], [rho]])

omg0 = np.sqrt(beta * gamma + alpha * epsilon)
M = np.matrix([[alpha / omg0, beta/omg0], [1, 0]])
