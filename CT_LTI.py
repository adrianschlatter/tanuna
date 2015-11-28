# -*- coding: utf-8 -*-
"""
Library of ready-to-use continuous-time LTI systems.

@author: Adrian Schlatter
"""

import numpy as np
from dynamics import CT_LTI_System


class LowPass(CT_LTI_System):
    """Low-Pass Filter with 3-dB frequency fC and gain k"""

    def __init__(self, fC, k=1.):
        wC = 2 * np.pi * fC
        A = np.eye(1.) * (-wC)
        B = np.eye(1.)
        C = np.eye(1.) * k * wC
        D = np.matrix([[0.]])
        super(LowPass, self).__init__(A, B, C, D)

if __name__ == '__main__':
    G = LowPass(10.)
    H = G * G
