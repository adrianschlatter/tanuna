# -*- coding: utf-8 -*-
"""
Demonstrating mode-locked laser.

@author: Adrian Schlatter
"""
from modelocked_laser import *
import matplotlib.pyplot as pl

# Streamplot and Q-switching
# =============================================================================


def plot_streamplot():
    x = np.linspace(0., 0.2, 50)
    y = np.linspace(0., 25 * Psteady, 50)
    X, Y = np.meshgrid(x, y)
    V, U = NdYVO4.f(0, [Y, X])

    pl.figure(figsize=(12, 6))
    pl.suptitle('State Propagation', fontsize=18)

    pl.subplot(1, 2, 1)
    pl.title('State Diagram')
    pl.xlabel('Gain (1)')
    pl.ylabel('Intra-Cavity Power (W)')
    pl.streamplot(x, y, U, V)
    pl.plot([gsteady], [Psteady], r'ko')
    pl.xlim([x[0], x[-1]])
    pl.ylim([y[0], y[-1]])

    pl.plot(g, P, r'r-')
    pl.plot([gsteady], [Psteady], r'ko')

    pl.subplot(1, 2, 2)
    pl.title('Q-Switching')
    pl.plot(t/1e-6, NdYVO4.Toc * P, r'r-')
    pl.axhline(y=NdYVO4.Toc * Psteady, color='k', ls='--')
    pl.ylabel('Output Power (W)')
    pl.xlabel('Time (us)')
    pl.xlim([0, 300])
    pl.ylim([y[0] * NdYVO4.Toc, y[-1] * NdYVO4.Toc])

    pl.subplots_adjust(left=0.07, right=0.95)

# Output power vs pump power
# =============================================================================


def plot_slope():
    # calculate P(Ppump):
    Ppump = np.linspace(0, 1., 1000)
    steady = np.array([NdYVO4.steadystate(pp) for pp in Ppump])
    Pst, gst = steady.T

    pl.figure()
    pl.title('Output Power')
    pl.xlabel('Pump Power (W)')
    pl.ylabel('Output Power (W)')
    pl.plot(Ppump, NdYVO4.Toc * Pst, r'-', label='real')
    rho0 = NdYVO4.etaP * NdYVO4.Toc / (NdYVO4.loss + NdYVO4.DR)
    pl.plot([NdYVO4.pumpThreshold, Ppump[-1]],
            [0, rho0 * (Ppump[-1] - NdYVO4.pumpThreshold)],
            r'k--', label='if SA would not saturate')
    pl.xlim([0, 0.3])
    pl.ylim([0, 0.2])
    pl.legend(loc=2)

# Linearization
# =============================================================================


def plot_linearization():
    # calculate linearization(Ppump):
    w0 = np.array([NdYVO4.w0(pp) for pp in Ppump])
    zeta = np.array([NdYVO4.zeta(pp) for pp in Ppump])

    fig, ax1 = pl.subplots()
    pl.title('Frequency and Damping of Relaxation Oscillations')
    ax1.plot(Ppump, w0 / 2 / np.pi / 1e3, r'r-')
    ax1.set_xlabel('Pump Power (W)')
    ax1.set_ylabel('Natural Frequency (kHz)', color='r')
    ax2 = ax1.twinx()
    ax2.plot(Ppump, zeta, r'b-')
    ax2.axhline(y=0, color='b', ls='--')
    ax2.set_ylabel('Damping Ratio (1)', rotation=270, color='b')
    pl.subplots_adjust(right=0.88)

# Main
# =============================================================================

plot_streamplot()
