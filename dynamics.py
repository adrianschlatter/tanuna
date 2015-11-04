# -*- coding: utf-8 -*-
"""
pyDynamics provides tools to work with dynamic systems. This includes

* continuous- and discrete-time systems
* linear and non-linear systems
* time-independent and time-varying systems
* Single-Input Single-Output (SISO) and Multiple-Input Multiple-Output (MISO)
  systems


@author: Adrian Schlatter
"""

import numpy as np
import scipy.signal as signal
from scipy.linalg import expm
from time import sleep


class ApproximationError(Exception):
    pass


class MatrixError(Exception):
    pass


def determinant(M):
    """Calculates the determinant of a square matrix M"""

    if len(M.shape) != 2:
        raise MatrixError('M must have exactly 2 dimensions')
    if M.shape[0] != M.shape[1]:
        raise MatrixError('M is expected to be square')

    for row in range(M.shape[0]):
        # lower-triangulars == 0?
        try:
            i = np.arange(0, row)[M[row, :row] != 0][0]
            v_i = np.array(M[:, i].reshape(-1, 1))
            w_i = np.array(v_i)
            v_i[row:, 0] = 0.
            w_i[:row, 0] = 0.

            M1 = np.hstack((M[:, :i], v_i, M[:, i+1:]))
            M2 = np.hstack((M[:, :i], w_i, M[:, i+1:]))
#            print('Column splitting')
            return(determinant(M1) + determinant(M2))
        except IndexError:
            pass    # all 0

        # diagonal != 0?
        if M[row, row] == 0:
            try:
                col2 = np.arange(row+1, M.shape[0])[M[row, row+1:] != 0][0]
                Msw = np.hstack((M[:, :row],
                                 M[:, col2:col2+1],
                                 M[:, row+1:col2],
                                 M[:, row:row+1],
                                 M[:, col2+1:]))
                return(-determinant(Msw))
            except:
                return(0)
    return(M.diagonal().prod())


class CT_System():
    """
    Describes a continuous-time system with dynamics described by ordinary
    differential equations.

        s:          Internal state (vector) of the system
        s0:         Initial state of the system
        u:          External input (vector)

        f(t, s, u): Dynamics of the system (ds/dt = f(t, s, u))
        g(t, s, u): Function that maps state s to output y = g(t, s, u)

    It is solved by simply calling it with an argument t. t is either a float
    or array-like. In the latter case, the system is solved for all the times
    t in the array.
    """

    def __init__(self, f, g, s0):
        pass

    def __call__(self, t):
        pass

    def steadyStates(self, u0, t):
        """Returns a list of tuples (s_i, stability_i) with:

        - s_i:          A steady-state at time t, i.e. f(t, s_i, u0) = 0
        - stability_i:  True if this steady-state is stable, false otherwise
        """
        raise NotImplementedError

    def observable(self, t):
        """Returns whether the system is observable at time t (i.e. its
        internal state is determinable from inputs u and outputs y)."""
        raise NotImplementedError

    def reachable(self, t):
        """Returns whether the system is reachable at time t (i.e. all states
        are reachable by providing an appropriate input u(t))."""
        raise NotImplementedError

    def tangentLTI(self, s0, u0, t):
        """
        Approximates the CT_System at time t near state s0 and input u0
        by an LTISystem (linear, time-invariant system).
        Raises ApproximationError if the system can not be linearized.
        """

        raise NotImplementedError


class CT_LTI_System(CT_System):
    """Linear, time-invariant system"""

    def __init__(self, A, B, C, D, x0=None):
        A, B, C, D = map(np.asmatrix, (A, B, C, D))
        self._A, self._B, self._C, self._D = A, B, C, D
        if x0 is None:
            self.x = np.matrix(np.zeros((A.shape[0], 1)))
        else:
            self.x = x0

    @property
    def order(self):
        """The order of the system"""
        return(self._A.shape[0])

    @property
    def links(self):
        """Number of inputs and outputs"""
        return(self._D.T.shape)

    @property
    def eigenValues(self):
        """Eigenvalues of the state matrix"""
        return(np.linalg.eigvals(self._A))

    @property
    def stable(self):
        return(np.all(self.eigenValues.real < 0))

    @property
    def Wo(self):
        """Observability matrix"""
        W = np.matrix(np.zeros((0, self._C.shape[1])))
        for n in range(self.order):
            W = np.vstack((W, self._C * self._A**n))
        return(W)

    @property
    def Wr(self):
        """Reachability matrix"""
        W = np.matrix(np.zeros((self._B.shape[0], 0)))
        for n in range(self.order):
            W = np.hstack((W, self._A**n * self._B))
        return(W)

    @property
    def reachable(self):
        """Returns True if the system is reachable."""
        return(np.linalg.matrix_rank(self.Wr) == self.order)

    @property
    def observable(self):
        """Returns True if the system is observable."""
        return(np.linalg.matrix_rank(self.Wo) == self.order)

    def _tResponse(self):
        """Automatically determines appropriate time axis for step- and
        impulse-response plotting"""

        tau = np.abs(1. / self.eigenValues.real)
        f = self.eigenValues.imag / (2 * np.pi)
        period = np.abs(1. / f[f != 0])
        timescales = np.concatenate([tau, period])
        dt = timescales.min() / 20.
        T = tau.max() * 10.
        return(np.arange(0., T, dt))

    def stepResponse(self, t=None):
        """
        Step Response
        +++++++++++++

        Returns (t, ystep), where

            ystep :  Step response
            t     :  Corresponding array of times

        t is either provided as an argument to this function or determined
        automatically.
        """

        if t is None:
            t = self._tResponse()

        A, B, C, D = self._A, self._B, self._C, self._D
        steady = D - C * A.I * B
        y = [C * A.I * expm(A * ti) * B + steady for ti in t]
        return((t, np.array(y).reshape(-1, self.links[1])))

    def impulseResponse(self, t=None):
        """
        Impulse Response
        +++++++++++++

        Returns (t, yimpulse), where

            yimpulse :  Impulse response (*without* direct term D)
            t        :  Corresponding array of times

        t is either provided as an argument to this function or determined
        automatically.
        """

        if t is None:
            t = self._tResponse()

        A, B, C = self._A, self._B, self._C
        y = [C * expm(A * ti) * B for ti in t]
        return((t, np.array(y).reshape(-1, self.links[1])))

    def freqResponse(self, t=None):
        """Frequency response"""
        # see Feedback System, page 153
        raise NotImplementedError

    @property
    def tf(self):
        """
        Transfer Function
        +++++++++++++++++

        Transfer-function representation (b, a) of the system. Returns
        numerator (b) and denominator (a) coefficients.

                 b[0] * s**n + ... + b[n] * s**0
        G(s) =  ---------------------------------
                 a[0] * s**m + ... + a[m] * s**0
        """
        A, B, C, D = self._A, self._B, self._C, self._D
        a = np.poly(A)
        tfs = [[]]
        for outp in range(self.links[1]):
            tfs.append([])
            for inp in range(self.links[0]):
                tfs[outp].append((bs[outp], a))
        return(tfs)

    @property
    def Thetaphi(self):
        raise NotImplementedError

    @property
    def zpk(self):
        """Gain, Pole, Zero representation of the system"""
        raise NotImplementedError


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


class DT_LTV_System():
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
    """Implements the discrete-time linear, time-invariant system with input
    vector u[t], internal state vector x[t], and output vector y[t]:

        x[t+1] = A * x[t] + B * u[t]
        y[t]   = C * x[t] + D * u[t]

    where
        A: state matrix
        B: input matrix
        C: output matrix
        D: feedthrough matrix

    The system is initialized with state vector x[0] = x0.
    """

    def __init__(self, A, B, C, D, x0=np.matrix([0., 0.]).T):
        self.A, self.B, self.C, self.C = A, B, C, D
        self.x = x0

    @classmethod
    def fromTransferFunction(Theta, phi):
        """Initialize DiscreteLTI instance from transfer-function coefficients
        'Theta' and 'phi'."""

        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def stable(self):
        """Returns True if the system is strictly stable"""
        raise NotImplementedError

    def observable(self):
        """Returns true if the system is observable"""
        raise NotImplementedError

    def reachable(self):
        """Returns True if the system is observable"""
        raise NotImplementedError

    def tf(self):
        """Returns the transfer function (b, a) where 'b' are the coefficients
        of the nominator polynomial and 'a' are the coefficients of the
        denominator polynomial."""
        raise NotImplementedError

    def proper(self):
        """Returns true if the system's transfer function is strictly proper,
        i.e. the degree of the numerator is less than the degree of the
        denominator."""
        raise NotImplementedError

    def __add__(self, right):
        raise NotImplementedError

    def __radd__(self, left):
        raise NotImplementedError

    def __rsub__(self, left):
        raise NotImplementedError

    def __mul__(self, right):
        raise NotImplementedError

    def __rmul__(self, left):
        raise NotImplementedError

    def __iadd__(self, right):
        raise NotImplementedError

    def __isub__(self, right):
        raise NotImplementedError

    def __imul__(self, right):
        raise NotImplementedError

    def __idiv__(self, right):
        raise NotImplementedError


if __name__ == '__main__':
#    import matplotlib.pyplot as pl
#    pl.close('all')
#
#    w0 = 2 * np.pi * 100e3
#    zeta = 0.5
#    k = 1.
#
#    A = np.matrix([[0, w0], [-w0, -2 * zeta * w0]])
#    B = np.matrix([0, k * w0]).T
#    C = np.matrix([1., 0.])
#    D = np.matrix([0.])
#
#    G = CT_LTI_System(A, B, C, D)
#
#    pl.figure()
#    pl.plot(*G.stepResponse())

    for i in range(1000):
        M = np.random.normal(loc=0, scale=3, size=(20, 20))
        det = determinant(M)
        if det / np.linalg.det(M) - 1 > 1e-12:
            print(M)
            break

#    pl.plot(*G.impulseResponse())
#
#    # "short-circuit" filter (output = input):
#    filter0 = [np.ones(1.), np.ones(1.)]
#    # XXX low-pass?
#    filter1 = [np.array([6./50.]), np.array([1, -44./50.])]
#
#    Nsamples = 50
#    A, B, C, D = signal.tf2ss(*filter1)
#    At = A.reshape((1,) + A.shape).repeat(Nsamples, axis=0)
#    Bt = B.reshape((1,) + B.shape).repeat(Nsamples, axis=0)
#    Ct = C.reshape((1,) + C.shape).repeat(Nsamples, axis=0)
#    Dt = D.reshape((1,) + D.shape).repeat(Nsamples, axis=0)
#
#    sys = DT_LTV_System(At, Bt, Ct, Dt, np.zeros((1, 1)))
#    Yt = sys.feed(np.ones((1, Nsamples)))
#
#    pl.figure()
#    pl.plot(np.ones(Nsamples), 'ko-')
#    pl.plot(Yt.T, 'bo-')
