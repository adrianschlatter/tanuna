# -*- coding: utf-8 -*-

import tanuna as dyn
import numpy as np
import matplotlib.pyplot as pl

w0 = 2 * np.pi * 10
zeta = 0.5
k = 1.

A = np.matrix([[0, w0], [-w0, -2*zeta*w0]])
B = np.matrix([0, k*w0]).T
C = np.matrix([k, 0.])
D = np.matrix([0.])

G = dyn.CT_LTI_System(A, B, C, D)


pl.figure(figsize=(6, 12))

# STEP RESPONSE
pl.subplot(4, 1, 1)
pl.title('Step-Response')
pl.plot(*G.stepResponse())
pl.xlabel('Time After Step (s)')
pl.ylabel('y')

# IMPULSE RESPONSE
pl.subplot(4, 1, 2)
pl.title('Impulse-Response')
pl.plot(*G.impulseResponse())
pl.xlabel('Time After Impulse (s)')
pl.ylabel('y')

# BODE PLOT
ax1 = pl.subplot(4, 1, 3)
ax1.set_title('Bode Plot')
f, Chi = G.freqResponse()
ax1.semilogx(f, 20 * np.log10(np.abs(Chi[0, 0])), r'b-')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Magnitude (dB)')
ax2 = ax1.twinx()
ax2.semilogx(f, np.angle(Chi[0, 0]) / np.pi, r'r-')
ax2.set_ylabel('Phase ($\pi$)', va='bottom', rotation=270)

# NYQUIST PLOT
ax = pl.subplot(4, 1, 4)
pl.title('Nyquist Plot')
pl.plot(np.real(Chi[0, 0]), np.imag(Chi[0, 0]))
pl.plot([-1], [0], r'ro')
pl.xlim([-3., 3])
pl.ylim([-1.5, 0.5])
ax.set_aspect('equal')
pl.axhline(y=0, color='k')
pl.axvline(x=0, color='k')
pl.xlabel('Real Part')
pl.ylabel('Imaginary Part')

pl.subplots_adjust(top=0.96, bottom=0.06, right=0.87, hspace=0.5)
