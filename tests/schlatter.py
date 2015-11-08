# -*- coding: utf-8 -*-
"""
Tests against analytical results of a simple 2nd order linear system.

@author: Adrian Schlatter
"""

import sys
import unittest
import numpy as np
from tools import almostEqual
sys.path.append('..')
import dynamics as dyn


class Test_2ndOrderSystem(unittest.TestCase):
    """Testing a SISO 2nd-order LTI system"""

    def setUp(self):
        self.w0 = w0 = 2 * np.pi * 10
        self.zeta = zeta = 0.5
        self.k = k = 1.

        A = np.matrix([[0, w0], [-w0, -2*zeta*w0]])
        B = np.matrix([0, k*w0]).T
        C = np.matrix([k, 0.])
        D = np.matrix([0.])

        self.G = dyn.CT_LTI_System(A, B, C, D)

    def test_order(self):
        self.assertEqual(self.G.order, 2)

    def test_links(self):
        self.assertEqual(self.G.links, (1, 1))

    def test_poles(self):
        self.assertTrue(almostEqual(self.G.poles, self.G.zpk[1]))

    def test_stable(self):
        self.assertTrue(self.G.stable)

    def test_zpk(self):
        """Zeros, Poles, and Gain of the system"""

        z = np.array([])
        p = self.w0 * np.array([-self.zeta + 1j * np.sqrt(1 - self.zeta**2),
                                -self.zeta - 1j * np.sqrt(1 - self.zeta**2)])
        k = self.k**2
        Z, P, K = self.G.zpk
        equal = almostEqual(Z, z) and almostEqual(P, p) and almostEqual(K, k)
        self.assertTrue(equal)

    def test_Wr(self):
        """Reachability matrix"""
        Wr = self.k * self.w0 * np.matrix([[0., self.w0],
                                           [1., -2 * self.zeta * self.w0]])
        self.assertTrue(np.all(self.G.Wr == Wr))

    def test_Wo(self):
        """Observability matrix"""
        Wo = self.k * np.matrix([[1., 0.],
                                 [0., self.w0]])
        self.assertTrue(np.all(self.G.Wo == Wo))

    def test_reachable(self):
        """Reachability"""
        self.assertTrue(self.G.reachable)

    def test_observable(self):
        """Observability"""
        self.assertTrue(self.G.observable)

    def test_tf(self):
        """Transfer function"""
        a = np.poly1d([1., 2 * self.zeta * self.w0, self.w0**2], variable='s')
        b = np.poly1d([(self.k * self.w0)**2], variable='s')
        btf, atf = self.G.tf
        self.assertTrue(almostEqual(btf, b) and almostEqual(atf, a))

    def test_freqResponse(self):
        """Frequency Response is tf(2*pi*1j*f)"""
        F = np.logspace(0, 2, 200)
        b, a = self.G.tf
        R = b(2 * np.pi * 1j * F) / a(2 * np.pi * 1j * F)

        f, r = self.G.freqResponse(F)
        self.assertTrue(almostEqual(f, F) and almostEqual(r, R))

if __name__ == '__main__':
    unittest.main()
