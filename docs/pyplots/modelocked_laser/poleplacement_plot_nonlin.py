# -*- coding: utf-8 -*-
"""
Stabilization of a mode-locked laser using pole placement.

Appling the feedback design for the linearized laser to the original
(non-linear) laser.

@author: Adrian Schlatter
"""

from poleplacement import stabilized_lin, system_lin, NdYVO4, Ppump
from poleplacement import stabilized, NdYVO4, system
import numpy as np
import matplotlib.pyplot as pl

pl.close('all')

pl.figure(figsize=(6, 6))
pl.suptitle('Stabilization at $P_P$ = %.1f W' % Ppump, fontsize=18)

# STEP RESPONSE
pl.title('Step-Response System')
t = np.linspace(0, 150e-6, 1000)
stepStab = stabilized(t)
step = system(t)
targetPout = NdYVO4.Toc * NdYVO4.Psteady(Ppump)
pl.plot(t / 1e-6, step[0, :].T, label='free running')
pl.plot(t / 1e-6, stepStab[0, :].T, label='controlled')
pl.axhline(y=targetPout, color='k', ls='--', label='Steady-State Pout')
pl.xlim([t[0] / 1e-6, t[-1] / 1e-6])
pl.ylim([0, 0.8])
pl.xlabel('Time After Step (us)')
pl.ylabel(r'$P_{out}$ (W)')
pl.legend(loc=1)
