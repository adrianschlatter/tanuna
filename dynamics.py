# -*- coding: utf-8 -*-
"""
[ Describe here what this script / module does ]

@author: Adrian Schlatter
"""

import numpy as np
import scipy.signal as ss


class ApproximationError(Exception):
    pass


class CT_System():
    """
    Describes a continuous-time system with dynamics described by ordinary
    differential equations.

        s:          Internal state (vector) of the system
        s0:         Initial state of the system
        u:          External input (vector)

        f(t, s, u): Dynamics of the system (ds/dt = f(t, s, u))
        s0:         Initial state
        c(s):       Function that maps state s to output y = c(s) + d(u)
        d(u):       Function describing direct term y = c(s) + d(u)

    It is solved by simply calling it with an argument t. t is
    either a float or array-like. In the latter case, the system is solved for
    all the times t in the array.
    """

    def __init__(self, f, s0, c, d):
        pass

    def __call__(self, t):
        pass

    def steadyStates(self, u0, t):
        """Returns a list of tuples (s_i, stability_i) with:

        - s_i:          A steady-state at time t, i.e. f(t, s_i, u0) = 0
        - stability_i:  True if this steady-state is stable, false otherwise
        """
        pass

    def observable(self, t):
        """Returns whether the system is observable at time t (i.e. its
        internal state is determinable from inputs u and outputs y)."""

        pass

    def reachable(self, t):
        """Returns whether the system is reachable at time t (i.e. all states
        are reachable by providing an appropriate input u(t))."""

        pass

    def tangentLTI(self, s0, u0, t):
        """
        Approximates the OrdinarySystem at time t near state s0 and input u0
        by an LTISystem (linear, time-invariant system).
        Raises ApproximationError if the system can not be linearized.
        """

        pass


class CT_LTI_System(CT_System):
    """Linear, time-invariant system"""

    pass


def Thetaphi(b, a):
    """Translate filter-coefficient arrays b and a to Theta, phi
    representation:

    phi(B)*y_t = Theta(B)*x_t

    Theta, phi = Thetaphi(b, a) are the coefficient of the back-shift-operator
    polynomials (index i belongs to B^i)"""

    phi = np.array(a)
    if len(phi) > 1:
        phi[1:] = -phi[1:]
    Theta = np.array(b)
    return [Theta, phi]


def ba(Theta, phi):
    """Translate backshift-operator polynomials Theta and phi to filter
    coefficient array b, a.

    a[0]*y[t] = a[1]*y[t-1] + ... + a[n]*y[t-n] + b[0]*x[t] + ... + b[m]*x[t-m]
    """
    # XXX these b and a are not compatible with scipy.lfilter. Appararently,
    # scipy.lfilter  expects Theta and phi

    # Thetaphi() is its own inverse:
    return(Thetaphi(Theta, phi))


def differenceEquation(b, a):
    """Takes filter coefficient arrays b and a and returns string with
    difference equation using powers of B, where B the backshift operator."""

    Theta, phi = Thetaphi(b, a)
    s = '('
    for i in range(len(phi)):
        s += '%.2f B^%d+' % (phi[i], i)
    s = s[:-1] + ')*y_t = ('
    for i in range(len(Theta)):
        s += '%.2f B^%d+' % (Theta[i], i)
    s = s[:-1]+')*x_t'
    return s


class StateSpaceFilter():
    """Implements the discrete linear, time-variant system with input vector
    u[t], internal state vector x[t], and output vector y[t]:

        x[t+1] = A[t]*x[t] + B[t]*u[t]
        y[t]   = C*x[t] + D*u[t]

    where
        A[t]: state matrices
        B[t]: input matrices
        C[t]: output matrices
        D[t]: feedthrough matrices

    The system is initialized with state vector x[0] = X0.
    """

    def __init__(self, At, Bt, Ct, Dt, X0):
        self.At = At
        self.Bt = Bt
        self.Ct = Ct
        self.Dt = Dt
        self.X = X0
        self.t = 0

    def update(self, U):
        U.shape = (-1, 1)
        t = min(self.t, len(self.At))
        self.X = np.dot(self.At[t], self.X) + np.dot(self.Bt[t], U)
        self.t += 1
        return np.dot(self.Ct[t], self.X) + np.dot(self.Dt[t], U)

    def feed(self, Ut):
        return np.concatenate([self.update(U) for U in Ut.T]).T


class DT_LTI_System(object):
    """Discrete-time linear time-invariant system."""

    def __init__(self, A, B, C, D, x0):
        pass

    @classmethod
    def fromTransferFunction(Theta, phi):
        """Initialize DiscreteLTI instance from transfer-function coefficients
        'Theta' and 'phi'."""
        pass

    def __repr__(self):
        pass

    def stable(self):
        """Returns true if the system is strictly stable"""
        pass

    def observable(self):
        """Returns true if the system is observable"""
        pass

    def controllable(self):
        """Returns true if the system is observable"""
        pass

    def tf(self):
        """Returns the transfer function (b, a) where 'b' are the coefficients
        of the nominator polynomial and 'a' are the coefficients of the
        denominator polynomial."""
        pass

    def proper(self):
        """Returns true if the system's transfer function is strictly proper,
        i.e. the degree of the numerator is less than the degree of the
        denominator."""
        pass

    def __add__(self, right):
        pass

    def __radd__(self, left):
        pass

    def __rsub__(self, left):
        pass

    def __mul__(self, right):
        pass

    def __rmul__(self, left):
        pass

    def __iadd__(self, right):
        pass

    def __isub__(self, right):
        pass

    def __imul__(self, right):
        pass

    def __idiv__(self, right):
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as pl

    # "short-circuit" filter (output = input):
    filter0 = [np.ones(1.), np.ones(1.)]
    # XXX low-pass?
    filter1 = [np.array([6./50.]), np.array([1, -44./50.])]

    Nsamples = 50
    A, B, C, D = ss.tf2ss(*filter1)
    At = A.reshape((1,) + A.shape).repeat(Nsamples, axis=0)
    Bt = B.reshape((1,) + B.shape).repeat(Nsamples, axis=0)
    Ct = C.reshape((1,) + C.shape).repeat(Nsamples, axis=0)
    Dt = D.reshape((1,) + D.shape).repeat(Nsamples, axis=0)

    sys = StateSpaceFilter(At, Bt, Ct, Dt, np.zeros((1, 1)))
    Yt = sys.feed(np.ones((1, Nsamples)))

    pl.figure()
    pl.plot(np.ones(Nsamples), 'ko-')
    pl.plot(Yt.T, 'bo-')
