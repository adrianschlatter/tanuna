# -*- coding: utf-8 -*-
"""
Tests against analytical results of a simple 2nd order linear system.

@author: Adrian Schlatter
"""

import unittest
import numpy as np
from tools import almostEqual
import dynamics as dyn
from CT_LTI import LowPass


class Test_2ndOrderSystem(unittest.TestCase):
    """Testing a SISO 2nd-order LTI system"""

    def setUp(self):
        self.w0 = w0 = 2 * np.pi * 10
        self.zeta = zeta = 0.5
        self.k = k = 1.

        A = np.matrix([[0, w0], [-w0, -2*zeta*w0]])
        B = np.matrix([0, k*w0]).T
        C = np.matrix([1., 0.])
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
        k = self.k
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
        Wo = np.matrix([[1., 0.],
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
        b = np.poly1d([self.k * self.w0**2], variable='s')
        btf, atf = self.G.tf
        self.assertTrue(almostEqual(btf, b) and almostEqual(atf, a))

    def test_freqResponse(self):
        """Frequency Response is tf(2*pi*1j*f)"""
        F = np.logspace(0, 2, 200)
        b, a = self.G.tf
        R = b(2 * np.pi * 1j * F) / a(2 * np.pi * 1j * F)

        f, r = self.G.freqResponse(F)
        self.assertTrue(almostEqual(f, F) and almostEqual(r, R))

    def test_numberOperators(self):
        """Test multiplication and division with / by floats and ints"""
        G = self.G
        H = 2 * G        # int from left
        H = H * (-4)     # int from right
        H = 0.5 * H      # float from left
        H = H * 0.25     # float from right
        H = H / 2        # int
        H = H / 0.5      # float
        H = -H           # negation

        self.assertTrue(almostEqual(H._A, G._A) and almostEqual(H._B, G._B) and
                        almostEqual(H._C, G._C) and almostEqual(H._D, G._D))


class Test_1stOrderSystem(unittest.TestCase):
    """Testing combinations of SISO 1st-order LTI systems"""

    def setUp(self):
        """Set up a low-pass filter G and a high-pass filter H"""

        self.wG = wG = 2 * np.pi * 40
        self.kG = kG = 1.
        self.G = LowPass(wG, kG)

        self.wH = wH = 2 * np.pi * 10
        self.kH = kH = 1.

        A = np.eye(1.) * (-wH)
        B = np.eye(1.)
        C = np.eye(1.) * kH * wH
        D = np.matrix([[0.]])
        self.H = dyn.CT_LTI_System(A, B, C, D)

    def test_connectInSeries(self):
        G2 = self.G * self.G
        z, p, k = G2.zpk
        Z = self.G.zpk[0]
        P = np.concatenate((self.G.zpk[1], self.G.zpk[1]))
        K = self.G.zpk[2]**2
        self.assertTrue(np.all(z == Z) and almostEqual(p, P, tol=1e-4) and
                        almostEqual(k, K))

    def test_connectWithLeftMatrix(self):
        M = np.matrix([[2], [0.3]])
        I = M * self.G
        self.assertEqual(I.links, (1, 2))

    def test_connectWithRightMatrix(self):
        M = np.matrix([[0.3, 5]])
        I = self.G * M
        self.assertEqual(I.links, (2, 1))

    def test_add(self):
        I = self.G + self.G
        self.assertTrue(almostEqual(2., I.zpk[2]))

    def test_add2(self):
        I = self.G + self.G
        Ialt = dyn.connect(self.G, self.G, Hin=(), Gout=())
        Ialt = np.matrix([[1, 1]]) * Ialt * np.matrix([[1], [1]])
        passed = True
        for i in range(4):
            if not almostEqual(I.ABCD[i], Ialt.ABCD[i]):
                passed = False
                break
        self.assertTrue(passed)


class Test_MIMO(unittest.TestCase):
    """Test whether MIMO-systems are handled properly"""

    def setUp(self):
        fc = 10.
        self.wc = fc * 2 * np.pi

        H = dyn.connect(LowPass(fc), np.matrix([[1, 1]]))
        G = dyn.connect(np.matrix([[1], [1]]), LowPass(fc))
        self.MIMO = dyn.connect(H, G, Gout=(0,), Hin=(1,))

    def test_connectMIMO(self):
        A = np.matrix([[-self.wc, 0], [self.wc, -self.wc]])
        B = np.eye(2.)
        C = self.wc * np.eye(2.)
        D = np.zeros((2, 2))
        a, b, c, d = self.MIMO.ABCD
        self.assertTrue(almostEqual(a, A) and almostEqual(b, B) and
                        almostEqual(c, C) and almostEqual(d, D))

    def test_impulseResponse(self):
        t, impResp = self.MIMO.impulseResponse()
        shape = (len(t),) + self.MIMO.links
        self.assertEqual(impResp.shape, shape)

    def test_stepResponse(self):
        t, stepResp = self.MIMO.stepResponse()
        shape = (len(t),) + self.MIMO.links
        self.assertEqual(stepResp.shape, shape)

    def test_freqResponse(self):
        f, freqResp = self.MIMO.freqResponse()
        shape = (len(f),) + self.MIMO.links
        self.assertEqual(freqResp.shape, shape)


if __name__ == '__main__':
    unittest.main()
