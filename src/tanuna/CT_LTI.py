# -*- coding: utf-8 -*-
"""
Library of ready-to-use continuous-time LTI systems.

@author: Adrian Schlatter
"""

import numpy as np
from .root import CT_LTI_System


class LowPass(CT_LTI_System):
    """Low-Pass Filter with 3-dB frequency fC and DC-gain k"""

    def __init__(self, fC, k=1.):
        wC = 2 * np.pi * fC
        A = np.eye(1) * (-wC)
        B = np.eye(1)
        C = np.eye(1) * k * wC
        D = np.matrix([[0.]])
        super(LowPass, self).__init__(A, B, C, D)


class HighPass(CT_LTI_System):
    """High-Pass Filter with 3-dB frequency fC and pass-band gain k"""

    def __init__(self, fC, k=1.):
        hp = (-LowPass(fC, 1) + np.matrix(1)) * k
        super(HighPass, self).__init__(*hp.ABCD)


class Order2(CT_LTI_System):
    """
    A second-order system with

    - w0:       Natural frequency
    - zeta:     Damping ratio (0: undamped, 1: critically damped)
    - k:        Gain
    """

    def __init__(self, w0, zeta, k):
        A = np.matrix([[0, w0], [-w0, -2 * zeta * w0]])
        B = np.matrix([0, k * w0]).T
        C = np.matrix([1., 0.])
        D = np.matrix([0.])

        super(Order2, self).__init__(A, B, C, D)


if __name__ == '__main__':
    G = LowPass(10.)
    H = G * G
