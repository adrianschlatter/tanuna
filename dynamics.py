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
from numpy.linalg import det
from scipy.linalg import expm


class ApproximationError(Exception):
    pass


class MatrixError(Exception):
    pass


class ConnectionError(Exception):
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


def nChoosek(n, k):
    """
    Yields all possibilities of choosing k elements from an array of length n.

    >>> list(nChoosek(3, 1))
    ... [array([True, False, False]),
    ...  array([False, True, False]),
    ...  array([False, False, True])]

    Assume you have an array, you get all choices of k elements with

    >>> x = np.array([5, 2, 8])
    ... for choice in nChoosek(len(x), 1):
    ...     choice = np.array(choice)
    ...     print(x[choice])
    ...
    ... [5]
    ... [2]
    ... [8]
    """

    # These conditions end the recursion:
    if k == 0:
        yield [False] * n
        return
    if k == n:
        yield [True] * n
        return

    # Otherwise, we have to recurse:
    for subChoice in nChoosek(n - 1, k - 1):
        # Select first element and recurse:
        yield [True] + subChoice
    for subChoice in nChoosek(n - 1, k):
        # Do not select first element and recurse:
        yield [False] + subChoice


def poly(M0, M1):
    """
    Returns the coefficients of the polynomial

        p(s) = det(M0 + s*M1)

    with M0 and M1 square matrices of equal dimension.

    Example:

        b = np.array(-1, 2.5, 0)

    refers to the polynomial

        p(s) = -s**2 + 2.5*s + 0
    """

    # Let M0 = [v0, ..., v(n-1)], M1 = [w0, ..., w(n-1)]
    # det(M0 + s*M1) = sum(det(x0, ..., x(n-1)) * s**k),
    # where xi in [vi, wi]; k the number of xi in [w0, ..., w(n-1)]
    # and the sum is running over all possibilities to choose the xi's.

    M0 = np.matrix(M0)
    M1 = np.matrix(M1)
    n = M0.shape[0]
    if M0.shape != (n, n) or M1.shape != (n, n) or n < 1:
        raise MatrixError(
                'M0 and M1 both need to be square and of the same size >= 1')

    b = np.zeros(n + 1)
    for k in range(n + 1):
        for choice in nChoosek(n, k):
            # if choice[i] == True => select wi, otherwise vi:
            M = np.where(choice, M1.T, M0.T)
            b[k] += det(M)
    return(np.poly1d(b[::-1]))


def connect(G, H, Gout=None, Hin=None):
    if issubclass(type(H), type(G)):
        try:
            connection = H.__rconnect__(G, Gout, Hin)
        except AttributeError:
            connection = NotImplemented
        if connection is NotImplemented:
            connection = G.__connect__(H, Gout, Hin)
    else:
        try:
            connection = G.__connect__(H, Gout, Hin)
        except AttributeError:
            connection = NotImplemented
        if connection is NotImplemented:
            connection = H.__rconnect__(G, Gout, Hin)

    return(connection)


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
            self.x = np.matrix(x0)

    @property
    def ABCD(self):
        return([self._A, self._B, self._C, self._D])

    @property
    def order(self):
        """The order of the system"""
        return(self._A.shape[0])

    @property
    def links(self):
        """Number of inputs and outputs"""
        return(self._D.T.shape)

    @property
    def poles(self):
        """Eigenvalues of the state matrix"""
        return(self.zpk[1])

    @property
    def stable(self):
        return(np.all(self.poles.real < 0))

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

        tau = np.abs(1. / self.poles.real)
        f = self.poles.imag / (2 * np.pi)
        period = np.abs(1. / f[f != 0])
        timescales = np.concatenate([tau, period])
        dt = timescales.min() / 20.
        T = tau.max() * 10.
        return(np.arange(0., T, dt))

    def stepResponse(self, t=None):
        """
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

    def freqResponse(self, f=None):
        """
        Returns (f, r), where

            f    : Array of frequencies
            r    : (Complex) frequency response

        f is either provided as an argument to thin function or determined
        automatically.
        """
        # see [astrom_feedback]_, page 153

        # Automatically determine frequency axis:
        if f is None:
            t = self._tResponse()
            dt = t[1] - t[0]
            T = t[-1] + dt
            fmax = 1. / (2 * dt)
            fmin = 1. / T
            f = np.logspace(np.log10(fmin), np.log10(fmax), 200)

        # prepare empty lists for results:
        nin, nout = self.links
        fr = []
        for i in range(nout):
            fr.append([[]] * nin)

        # calculate transfer function
        b, a = self.tf

        # if SISO, pack b into 1x1 list:
        if self.links == (1, 1):
            b = [[b]]

        # evaluate tf(2*pi*i*f) for each input-output combination:
        for i in range(nout):
            for j in range(nin):
                s = 2 * np.pi * 1j * f
                fr[i][j] = b[i][j](s) / a(s)

        # if SISO, unpack b from 1x1 list:
        if self.links == (1, 1):
            return(f, fr[0][0])
        else:
            return(f, fr)

    @property
    def tf(self):
        """
        Transfer-function representation [b, a] of the system. Returns
        numerator (b) and denominator (a) coefficients.

                 b[0] * s**0 + ... + b[m] * s**m
        G(s) =  ---------------------------------
                 a[0] * s**0 + ... + a[n] * s**n
        """

        A, B, C, D = self.ABCD
        n = self.order
        nin, nout = self.links

        # Denominator (poles):
        a = poly(A, -np.eye(n))

        # create list to hold nominator polynomials and gains:
        b = []
        for i in range(nout):
            b.append([[]] * nin)
        G = np.zeros((nout, nin))

        # There is 1 nominator per input-output combination:
        for i in range(nout):
            for j in range(nin):
                # DC gain:
                G[i][j] = float(D[i:i+1, j:j+1] -
                                C[i:i+1, :] * A.I * B[:, j:j+1])

                # Nominator polynomial:
                M0 = np.bmat([[A, B[:, j:j+1]],
                              [C[i:i+1, :], D[i:i+1, j:j+1]]])
                M1 = np.bmat([[-np.eye(n), np.zeros((n, 1))],
                              [np.zeros((1, n)), np.zeros((1, 1))]])
                b[i][j] = poly(M0, M1)

                # Adjust gain:
                b[i][j] = b[i][j] * G[i][j] / (b[i][j][0] / a[0])

        # For a SISO-system do not return a list of nominator polynomials but
        # simply *the* nominator polynomial:
        if self.links == (1, 1):
            return([b[0][0], a])
        else:  # MIMO
            return([b, a])

    @property
    def zpk(self):
        """
        Gain, Pole, Zero representation of the system. Returns a tuple
        (z, p, k) with z the zeros, p the poles, and k the gain of the system.
        p is an array. The format of z and k depends on the number of inputs
        and outputs of the system:

        For a SISO system z is an array and k is float. For a system with more
        inputs or outputs, z and k are lists of 'shape' (nout, nin) containing
        arrays and floats, respectively.
        """

        b, a = self.tf
        nin, nout = self.links

        # if SISO, pack b into 1x1 list:
        if self.links == (1, 1):
            b = [[b]]

        gain = np.zeros((nout, nin))
        zeros = []
        for i in range(nout):
            zeros.append([[]] * nin)

        poles = np.roots(a)
        for i in range(nout):
            for j in range(nin):
                gain[i][j] = b[i][j][0] / a[0]
                zeros[i][j] = np.roots(b[i][j])

        # Simplification for SISO system:
        if self.links == (1, 1):
            gain = gain[0, 0]
            zeros = zeros[0][0]

        return(zeros, poles, gain)

    def __add__(self, right):
        G = self
        nG = G.order
        if issubclass(type(right), CT_LTI_System):
            H = right
            nH = H.order

            A = np.bmat([[G._A, np.zeros((nG, nH))],
                         [np.zeros((nH, nG)), H._A]])
            B = np.vstack([G._B, H._B])
            C = np.hstack([G._C, H._C])
            D = G._D + H._D
            x0 = np.vstack([G.x, H.x])
            return(CT_LTI_System(A, B, C, D, x0))
        elif issubclass(type(right), np.matrix):
            if right.shape != G._D.shape:
                raise MatrixError('Shapes of right and self._D have to match')
            A = G._A
            B = G._B
            C = G._C
            D = G._D + right
            x0 = G.x
            return(CT_LTI_System(A, B, C, D, x0))
        else:
            return(NotImplementedError)

    def __radd__(self, left):
        return(NotImplementedError)

    def __rsub__(self, left):
        raise NotImplementedError

    def __connect__(self, right, Gout=None, Hin=None):
        H = self
        G = right
        if issubclass(type(G), CT_LTI_System):
            # Prepare Gout, Hin:
            # ===============================
            if Gout is None:
                # connect all outputs:
                Gout = np.arange(G.links[1])
            if Hin is None:
                # connect all inputs
                Hin = np.arange(H.links[0])
            if len(Gout) != len(Hin):
                raise ConnectionError(
                        'Number of inputs does not match number of outputs')

            # Prepare connection matrices:
            # ===============================

            # u_h = Sh * y_g:
            Sh = np.matrix(np.zeros((H.links[0], G.links[1])))
            for k in range(len(Hin)):
                i = Hin[k]
                j = Gout[k]
                Sh[i, j] = 1.

            # u_h = sh * u_h,unconnected:
            sh = np.matrix(np.zeros((H.links[0], H.links[0] - len(Hin))))
            u_h_unconnected = list(set(range(H.links[0])) - set(Hin))
            sh[u_h_unconnected, :] = np.eye(H.links[0] - len(Hin))

            # y_g,unconnected = sg * y_g:
            sg = np.matrix(np.zeros((G.links[1] - len(Gout), G.links[1])))
            y_g_unconnected = list(set(range(G.links[1])) - set(Gout))
            sg[:, y_g_unconnected] = np.eye(G.links[1] - len(Gout))

            # Setup state matrices:
            # ===============================

            nH = H.order
            nG = G.order

            A = np.bmat([[G._A, np.zeros((nG, nH))],
                         [H._B * Sh * G._C, H._A]])
            B = np.bmat([[G._B, np.zeros((nG, len(u_h_unconnected)))],
                         [H._B * Sh * G._D, H._B * sh]])
            C = np.bmat([[sg * G._C, np.zeros((len(y_g_unconnected), nH))],
                         [H._D * Sh * G._C, H._C]])
            D = np.bmat([[sg * G._D, np.zeros((len(y_g_unconnected),
                                               len(u_h_unconnected)))],
                         [H._D * Sh * G._D, H._D * sh]])
            x0 = np.vstack([G.x, H.x])
        elif issubclass(type(G), CT_System):
            # delegate to super class:
            return(G.__rconnect__(H, Gout, Hin))
        elif issubclass(type(G), np.matrix):
            # Multiply u by matrix before feeding into H:
            A = np.matrix(H._A)
            B = H._B * G
            C = np.matrix(H._C)
            D = H._D * G
            x0 = np.matrix(H.x)
        elif type(G) in [float, int]:
            # Apply gain G on input side:
            A = np.matrix(H._A)
            B = np.matrix(H._B)
            C = G * H._C
            D = G * H._D
            x0 = np.matrix(H.x)
        else:
            return(NotImplementedError)

        return(CT_LTI_System(A, B, C, D, x0))

    def __rconnect__(self, left, Gout=None, Hin=None):
        G = self
        H = left
        if issubclass(type(H), CT_LTI_System):
            return(H.__connect__(G, Gout, Hin))
        elif issubclass(type(H), CT_System):
            # delegate to super class:
            return(H.__connect__(G, Gout, Hin))
        elif issubclass(type(H), np.matrix):
            # Multiply output of G by matrix:
            A = np.matrix(G._A)
            B = np.matrix(G._B)
            C = H * G._C
            D = H * G._D
            x0 = np.matrix(G.x)
        elif type(H) in [float, int]:
            # Apply gain H on output side:
            A = np.matrix(G._A)
            B = np.matrix(G._B)
            C = H * G._C
            D = H * G._D
            x0 = np.matrix(G.x)
        else:
            return(NotImplementedError)

        return(CT_LTI_System(A, B, C, D, x0))

    def __mul__(self, right):
        return(self.__connect__(right))

    def __rmul__(self, left):
        return(self.__rconnect__(left))

    def __neg__(self):
        return(self * (-1))

    def __truediv__(self, right):
        if type(right) in [float, int]:
            invright = 1. / float(right)
            return(self * invright)
        else:
            raise NotImplementedError

    def __iadd__(self, right):
        raise NotImplementedError

    def __isub__(self, right):
        raise NotImplementedError

    def __imul__(self, right):
        raise NotImplementedError

    def __idiv__(self, right):
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
    import matplotlib.pyplot as pl
    pl.close('all')

    from CT_LTI import LowPass
    J = connect(LowPass(10.), LowPass(10.), Gout=(), Hin=())
    J = np.matrix([[1, 1]]) * J * np.matrix([[1], [1]])

    w0 = 2 * np.pi * 10
    zeta = 0.5
    k = 1.

    A = np.matrix([[0, w0], [-w0, -2 * zeta * w0]])
    B = np.matrix([0, k * w0]).T
    C = np.matrix([k, 0.])
    D = np.matrix([0.])

    G = CT_LTI_System(A, B, C, D)
    G = -G + np.matrix(1.)

    pl.figure()

    # STEP RESPONSE
    pl.subplot(4, 1, 1)
    pl.title('Step-Response')
    pl.plot(*G.stepResponse())
    pl.xlabel('Time After Step (s)')
    pl.ylabel('y')

    # IMPULSE RESPONSE
    pl.subplot(4, 1, 2)
    pl.title('Impulse-Response')
    pl.plot(*G.impulseResponse())
    pl.xlabel('Time After Impulse (s)')
    pl.ylabel('y')

    # BODE PLOT
    ax1 = pl.subplot(4, 1, 3)
    ax1.set_title('Bode Plot')
    f, Chi = G.freqResponse()
    ax1.semilogx(f, 20 * np.log10(np.abs(Chi)), r'b-')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax2 = ax1.twinx()
    ax2.semilogx(f, np.angle(Chi) / np.pi, r'r-')
    ax2.set_ylabel('Phase ($\pi$)')

    # NYQUIST PLOT
    ax = pl.subplot(4, 1, 4)
    pl.title('Nyquist Plot')
    pl.plot(np.real(Chi), np.imag(Chi))
    pl.plot([-1], [0], r'ro')
    pl.xlim([-2.5, 2])
    pl.ylim([-1.5, 0.5])
    ax.set_aspect('equal')
    pl.axhline(y=0, color='k')
    pl.axvline(x=0, color='k')
    pl.xlabel('Real Part')
    pl.ylabel('Imaginary Part')
