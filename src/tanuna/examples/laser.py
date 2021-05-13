# -*- coding: utf-8 -*-
"""
Model of a passively mode-locked laser.

@author: Adrian Schlatter
"""

# ignore warning 'line break after binary operator'
# as line break *before* binary operator *also* creates a warning ...
# flake8: noqa: W504

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import ode
from ..root import CT_LTI_System


class Laser(ode):
    """A class to simulate lasers with a (slow) saturable absorber in the
    cavity. While it is intended for mode-locked lasers, it may also be useful
    for Q-switched lasers."""

    def __init__(self, loss, TR, tauL, etaP, EsatL, DR, EsatA, Toc, PP0,
                 P0=None, g0=None):
        self.loss = loss
        self.TR = TR
        self.tauL = tauL
        self.etaP = etaP
        self.EsatL = EsatL
        self.DR = DR
        self.EsatA = EsatA
        self.Toc = Toc
        self.PP0 = PP0
        self.PP = PP0
        if P0 is None:
            P0 = self.Psteady()
        if g0 is None:
            g0 = self.gsteady()
        self.g0 = g0

        super().__init__(self.f, jac=self.grad_f)

        t0 = 0.
        self.set_initial_value([P0, g0], t0)
        self.set_integrator('dopri5')

    def qP(self, EP):
        S = EP / self.EsatA
        res = np.where(S == 0,
                       self.DR,
                       self.DR / S * (1. - np.exp(-S)))
        if res.shape == (1,):
            res = res[0]
        return(res)

    def dqP_dEP(self, EP):
        EsatA = self.EsatA
        S = EP / EsatA
        if S == 0:
            return(self.DR / self.EsatA)
        else:
            return(self.DR / EP * (np.exp(-S) - 1. / S + np.exp(-S) / S))

    def Pdot(self, P, g):
        P = np.array(P)
        g = np.array(g)

        EP = P * self.TR
        return(np.where(P > 0, (g - self.loss - self.qP(EP)) / self.TR * P,
                        np.zeros(P.shape)))

    def gdot(self, P, g):
        spontaneous = (self.g0 - g) / self.tauL
        stimulated = -P * g / self.EsatL
        pump = self.etaP * self.PP / self.EsatL
        return(spontaneous + stimulated + pump)

    def f(self, t, s):
        P, g = s
        sdot = [self.Pdot(P, g), self.gdot(P, g)]
        return(sdot)

    def grad_f(self, t, s):
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
    def pumpThreshold(self):
        """Pump power threshold, i.e., pump power needed to start lasing"""
        EsatL, tauL, etaP = self.EsatL, self.tauL, self.etaP
        loss, DR = self.loss, self.DR
        return(EsatL / tauL * (loss + DR) / etaP)

    def steadystate(self, Ppump=None):
        """Steady state (Psteady, gsteady) given pump power Ppump"""

        if Ppump is None:
            Ppump = self.PP

        EsatL, TR, tauL = self.EsatL, self.TR, self.tauL
        loss, DR, etaP = self.loss, self.DR, self.etaP

        PPthreshold = self.pumpThreshold
        if Ppump <= PPthreshold:
            Psteady = 0.
            gsteady = etaP * Ppump * tauL / EsatL
        else:
            # 1. determine boundaries for Psteady:
            # 2. Apply root-finder (brentq) given boundaries

            offs = EsatL / tauL
            # assume non-linear losses (qP(EP)) = 0:
            upperBound = -offs + Ppump * etaP / loss
            # assume max. non-linear losses (qP(EP) = DR):
            lowerBound = -offs + Ppump * etaP / (loss + DR)

            Psteady = brentq(lambda P: -P - EsatL / tauL + etaP * Ppump /
                             (loss + self.qP(P * TR)), lowerBound, upperBound)
            gsteady = loss + self.qP(Psteady * TR)

        return(Psteady, gsteady)

    def Psteady(self, Ppump=None):
        """Steady-state intracavity power given pump power Ppump"""
        return(self.steadystate(Ppump)[0])

    def gsteady(self, Ppump=None):
        """Steady-state gain given pump power Ppump"""
        return(self.steadystate(Ppump)[1])

    def w0(self, Ppump=None):
        """Returns natural angular frequency of disturbances around steady
        state. Steady state is determined from pump power Ppump."""

        EsatL, TR, tauL = self.EsatL, self.TR, self.tauL
        Pst, gst = self.steadystate(Ppump)
        r = Pst / EsatL

        w0 = np.sqrt(r * gst / TR +
                     Pst * self.dqP_dEP(Pst * TR) * (1. / tauL + r))
        return(w0)

    def alpha(self, Ppump=None):
        """Damping rate of relaxation oscillations (negative real part of
        poles). The nice thing about alpha is that it is also correct below
        the lasing threshold (where it is equal to 1 / tauL)."""

        EsatL, TR, tauL = self.EsatL, self.TR, self.tauL
        Pst, gst = self.steadystate(Ppump)
        a = (1. / tauL + Pst * (self.dqP_dEP(Pst * TR) + 1. / EsatL))
        return(a)

    def zeta(self, Ppump=None):
        """Damping ratio of relaxation oscillations."""
        return(self.alpha(Ppump) / 2. / self.w0(Ppump))

    def rho(self, Ppump=None):
        """Internal slope efficiency at pump power Ppump"""

        etaP, EsatL, TR = self.etaP, self.EsatL, self.TR
        return(self.Psteady(Ppump) * etaP / (EsatL * TR * self.w0(Ppump)**2))

    @property
    def stable(self):
        """Return true if laser is stable (i.e. no Q-switching)"""
        return(self.zeta > 0)

    def approximateLTI(self, Ppump=None):
        """Linearizes the state-equations around the steady state corresponding
        to a pump power Ppump and returns a CT_LTI_System."""

        w0 = self.w0(Ppump)
        zeta = self.zeta(Ppump)
        rho = self.rho(Ppump)
        Toc = self.Toc
        Pst = self.Psteady(Ppump)
        TR = self.TR
        dqPdEP = self.dqP_dEP(self.TR * Pst)

        M = np.matrix([[-Pst * dqPdEP / w0, Pst / TR / w0],
                       [1, 0]])
        A = np.matrix([[-2. * w0 * zeta, -w0],
                       [w0, 0.]])
        B = np.matrix([[w0 * rho],
                       [0.]])
        C = np.matrix([[0., Toc]])
        D = np.matrix([[0.]])
        return(M, CT_LTI_System(A, B, C, D))


class NdYVO4Laser(Laser):
    """An pre-configured example of a passively mode-locked 100 MHz Nd:YVO4
    Laser"""

    def __init__(self, Ppump):
        tauL = 90e-6
        TR = 10e-9
        FsatA = 60e-6 / 1e-4
        wA = 140e-6
        DR = 1.7e-2
        loss = 9e-2 + 1.3e-2
        wavelength = 1064e-9
        sigmaEm = 114e-20 * 1e-4
        wL = 62e-6
        etaP = 808. / 1064.
        Toc = 8.7e-2
        c = 3e8
        h = 6.626e-34
        nuL = c / wavelength
        EsatL = np.pi * wL**2 * h * nuL / (2 * sigmaEm)
        EsatA = FsatA * np.pi * wA**2

        Laser.__init__(self, loss, TR, tauL, etaP, EsatL, DR, EsatA, Toc,
                       Ppump, P0=None, g0=None)
