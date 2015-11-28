# -*- coding: utf-8 -*-
"""
Unit-tests for dynamics modules. The philosophy is to test examples of an
independent source (such as text books). Important references are:

.. [astrom_feedback] Karl Johan Åström and Richard M. Murray,
   "`Feedback Systems`_", Princeton University Press, 2012

.. _Feedback Systems:
   http://www.cds.caltech.edu/~murray/books/AM08/pdf/am08-hyperref_28Sep12.pdf

@author: Adrian Schlatter
"""

import unittest
import numpy as np
from tools import almostEqual
import dynamics as dyn


class Test_BalanceSystem(unittest.TestCase):
    """Testing Example 6.2 in [astrom_feedback]_"""

    def setUp(self):
        m = self.m = 70.
        self.M = 25.
        l = self.l = 1.2
        g = self.g = 9.81
        Mt = self.Mt = self.M + self.m
        J = self.J = 0.
        Jt = self.Jt = J + m * l**2
        mu = self.mu = Mt * Jt - (m * l)**2
        self.A = np.matrix([[0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [0, (m * l)**2 * g / mu, 0, 0],
                            [0, Mt * m * g * l / mu, 0, 0]])
        self.B = np.matrix([[0, 0, Jt / mu, l * m / mu]]).T
        self.C = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        self.D = np.matrix([[0, 0]]).T
        self.G = dyn.CT_LTI_System(self.A, self.B, self.C, self.D)

    def test_Wr(self):
        """Reachability matrix"""
        m = self.m
        l = self.l
        g = self.g
        Mt = self.Mt
        Jt = self.Jt
        mu = self.mu

        Wr = np.matrix([[0, Jt / mu, 0, g * (l * m)**3 / mu**2],
                        [0, l * m / mu, 0, g * (l * m)**2 * Mt / mu**2],
                        [Jt / mu, 0, g * (l * m)**3 / mu**2, 0],
                        [l * m / mu, 0, g * (l * m)**2 * Mt / mu**2, 0]])
        self.assertTrue(almostEqual(self.G.Wr, Wr))

    def test_reachable(self):
        """Reachability"""
        self.assertTrue(self.G.reachable)


if __name__ == '__main__':
    unittest.main()
