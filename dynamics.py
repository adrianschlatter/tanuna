# -*- coding: utf-8 -*-
"""
[ Describe here what this script / module does ]

@author: Adrian Schlatter
"""


class ApproximationError(Exception):
    pass


class OrdinarySystem():
    """
    Describes a system with dynamics described by ordinary differential
    equations.

        s:          Internal state (vector) of the system
        s0:         Initial state of the system
        u:          External input (vector)

        f(t, s, u): Dynamics of the system (ds/dt = f(t, s, u))
        s0:         Initial state
        c(s):       Function that maps state s to output y = c(s) + d(u)
        d(u):       Function describing direct term y = c(s) + d(u)

    OrdinarySystem is solved by simply calling it with an argument t. t is
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


class OrdinaryTISystem(OrdinarySystem):
    """The same as OrdinarySystem but independent of t."""

    pass


class LTISystem(OrdinaryTISystem):
    """Linear, time-invariant system"""

    pass
