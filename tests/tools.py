# -*- coding: utf-8 -*-
"""
Stuff that is useful for unittesting.

@author: Adrian Schlatter
"""

import numpy as np


def almostEqual(x, y, tol=1e-10):
    """
    Compare 2 array-like objects x and y for "almost equalness". This is
    useful for testing numerical computation where you can not expect that
    expected and real results are exactly identical.
    """

    x = np.array(x)
    y = np.array(y)

    if x.shape != y.shape:
        return(False)

    return(np.sqrt((np.abs(x - y)**2).sum()) < tol)
