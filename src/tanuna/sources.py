# -*- coding: utf-8 -*-
"""
Library of ready-to-use continuous-time sources.

.. author: Adrian Schlatter
"""

import numpy as np
from .root import CT_System


class SourceConstant(CT_System):

    def __init__(self, y=np.ones((1, 1), dtype='float')):
        y = np.matrix(y)
        super().__init__(lambda t, x, u: np.matrix([[]]).reshape(0, 1),
                         lambda t, x, u: y.repeat(np.matrix(t).shape[1]),
                         order=0, shape=(y.shape[0], 0))
