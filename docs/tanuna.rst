tanuna
++++++

tanuna provides tools to work with dynamic systems. This includes

* continuous- and discrete-time systems
* linear and non-linear systems
* time-independent and time-varying systems
* Single-Input Single-Output (SISO) and Multiple-Input Multiple-Output (MISO)
  systems

In the following, we will explain how to:

* create systems
* analyze systems
* solve systems
* combine systems


=========
Diving In
=========

Let's start with some examples based on a continuous-time, second-order LTI
SISO system:

.. testcode:: LTI

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

This creates the system G from state-space matrices A, B, C, D. The system
provides some interesting information:

.. doctest:: LTI

	>>> G.stable
	True
	>>> G.poles
	array([-31.41592654+54.41398093j, -31.41592654-54.41398093j])
	>>> G.reachable
	True
	>>> # Reachability matrix:
	... G.Wr
        matrix([[    0.        ,  3947.84176044],
                [   62.83185307, -3947.84176044]])
	>>> G.observable
	True
	>>> # Observability matrix:
	... G.Wo
        matrix([[  1.        ,   0.        ],
                [  0.        ,  62.83185307]])

Furthermore, it calculates step- and impulse-responses, Bode- and Nyquist-plots:

.. plot:: pyplots/diving_in/diving_in_plot.py
	:include-source: True
	:width: 50%

The duration of the trace and the density of samples is automatically determined
for you based on the Eigenvalues of the system (but you can provide your own if
you prefer).

System-algebra is supported: You can connect systems in series, in parallel
(creating a MIMO system from 2 SISO systems for example), and in feedback
configuration:

.. doctest:: LTI

	>>> # Connect G in series with G:
	... H = G * G
	>>> # Connect G in parallel with G:
	... J = G + G
	>>> # This is the same as 2 * G:
	... G + G == 2 * G
	True
	>>> # Check number of inputs and outputs:
	... (2 * G).shape
	(1, 1)
	>>> G.shape
	(1, 1)
	>>> H.shape
	(1, 1)

.. [feedback_systems] Karl Johan Åström and Richard M. Murray,
		      "`Feedback Systems`_", Princeton University Press, 2012

.. _Feedback Systems: http://www.cds.caltech.edu/~murray/books/AM08/pdf/am08-hyperref_28Sep12.pdf

