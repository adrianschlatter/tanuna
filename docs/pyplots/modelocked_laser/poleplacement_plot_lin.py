# -*- coding: utf-8 -*-
"""
Stabilization of a mode-locked laser using pole placement.

@author: Adrian Schlatter
"""

from poleplacement import stabilized_lin, system_lin, NdYVO4, Ppump
from poleplacement import stabilized, NdYVO4, system
import numpy as np
import matplotlib.pyplot as pl


pl.close('all')

pl.figure(figsize=(12, 6))
pl.suptitle('Stabilization at $P_P$ = %.1f W' % Ppump, fontsize=18)

# STEP RESPONSE
pl.subplot(1, 2, 1)
pl.title('Step-Response System')
t = np.linspace(0, 150e-6, 1000)
t, stepStab = stabilized_lin.stepResponse(t)
t, step = system_lin.stepResponse(t)
targetGain = NdYVO4.Toc * NdYVO4.rho(Ppump)
pl.plot(t / 1e-6, step[:, 0, 0], label='free running')
pl.plot(t / 1e-6, stepStab[:, 0, 0], label='controlled')
pl.axhline(y=targetGain, color='k', ls='--', label='target gain')
pl.xlim([t[0] / 1e-6, t[-1] / 1e-6])
pl.ylim([0, 2])
pl.xlabel('Time After Step (us)')
pl.ylabel(r'$\delta P_{out}$ (W)')
pl.legend(loc=1)

# POLE PLOT
pl.subplot(1, 2, 2)
pl.title('Pole Plot')
pl.plot(np.real(system_lin.poles) / 1e3, np.imag(system_lin.poles) / 1e3,
    r'bo', label='free running')
pl.plot(np.real(stabilized_lin.poles) / 1e3, np.imag(stabilized_lin.poles) / 1e3,
    r'go', label='controlled')
pl.axhline(y=0, color='k')
pl.axvline(x=0, color='k')
pl.ylabel(r'$\omega$ (krad / s)')
pl.xlabel('Gain (1 / ms)')
pl.xlim([-300, 200])
pl.ylim([-3e3, 3e3])
pl.grid()
pl.legend(loc=3)

pl.subplots_adjust(left=0.07, right=0.95, top=0.87, wspace=0.25)
