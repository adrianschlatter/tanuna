.. pyDynamics documentation master file, created by
   sphinx-quickstart on Wed Oct 21 18:30:46 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2

.. todolist::

pyDynamics
++++++++++

pyDynamics provides tools to work with dynamic systems. This includes

* continuous- and discrete-time systems
* linear and non-linear systems
* time-independent and time-varying systems
* Single-Input Single-Output (SISO) and Multiple-Input Multiple-Output (MISO) systems

In the following, we will explain how to:

* create systems
* analyze systems
* solve systems
* combine systems


=========
Diving In
=========

Let's start with some examples based on a continuous-time, second-order LTI SISO system:

.. testsetup::

	import sys
	sys.path.append('..')

.. testcode::

	import dynamics as dyn
	import numpy as np

	w0 = 2*np.pi*100e3
	zeta = 0.7
	k = 1.

	A = np.matrix([[0, w0], [-w0, -2*zeta*w0]])
	B = np.matrix([0, k*w0]).T
	C = np.matrix([1., 0.])
	D = np.matrix([0.])

	G = dyn.CT_LTI_System(A, B, C, D)

This creates the system G from state-space matrices A, B, C, D. The system provides some interesting information:

.. testsetup:: LTI_G

	import dynamics as dyn
	import numpy as np
	import matplotlib.pyplot as pl

	w0 = 2*np.pi*100e3
	zeta = 0.7
	k = 1.

	A = np.matrix([[0, w0], [-w0, -2*zeta*w0]])
	B = np.matrix([0, k*w0]).T
	C = np.matrix([1., 0.])
	D = np.matrix([0.])

	G = dyn.CT_LTI_System(A, B, C, D)

.. doctest:: LTI_G

	>>> G.stable
	True
	>>> G.eigenValues
	array([-439822.97150257+448709.18174495j, -439822.97150257-448709.18174495j])
	>>> G.reachable
	True
	>>> # Reachability matrix:
	... G.Wr
	XXX
	>>> G.observable
	True
	>>> # Observability matrix:
	... G.Wo
	XXX

.. todo:: Include correct output for reachability- and observability matrices in the code example above.

Furthermore, it calculates step- and impulse-responses, Bode- and Nyquist-plots:

.. testcode:: LTI_G

	pl.figure()
	
	# STEP RESPONSE
	pl.subplot(2, 2, 1)
	pl.title('Step-Response')
	pl.plot(*G.stepResponse())
	pl.xlabel('Time After Step (s)')
	pl.ylabel('y')
	
	# IMPULSE RESPONSE
	pl.subplot(2, 2, 2)
	pl.plot(*G.impulseResponse())
	pl.xlabel('Time After Impulse (s)')
	pl.ylabel('y')
	
	# BODE PLOT
	ax1 = pl.subplot(2, 2, 3)
	ax1.set_title('Bode Plot')
	f, Chi = G.freqResponse()
	ax1.semilogx(f[1:], 20 * np.log10(np.abs(Chi[1:] / Ch[0])),
		     r'b-')
	ax1.set_xlabel('Frequency (Hz)')
	ax1.set_ylabel('Magnitude (dB)')
	ax2 = ax1.twinx()
	ax2.semilogx(f, np.angle(Chi) / np.pi, r'o-')
	ax2.set_ylabel('Phase ($\pi$)')
	
	# NYQUIST PLOT
	pl.subplot(2, 2, 4)
	pl.title('Nyquist Plot')
	pl.plot(np.real(Chi), np.imag(Chi), r'k-')
	pl.plot([-1], [0], r'ro')
	pl.axhline(y=0)
	pl.axvline(x=0)
	pl.xlabel('Real Part')
	pl.ylabel('Imaginary Part')

.. todo:: Show the plots that the above code will generate

The duration of the trace and the density of samples is automatically determined for you based on the Eigenvalues of the system (but you can provide your own if you prefer).

System-algebra is supported: You can connect systems in series, in parallel (creating a MIMO system from 2 SISO systems for example), and in feedback configuration:

.. doctest:: LTI_G

	>>> # Connect G in series with G:
	... H = G * G
	>>> # Create a 2-input, 2-output system by paralellizing G's:
	... J = G + G
	>>> # This is the same as 2 * G:
	... G + G == 2 * G
	True
	>>> # Check number of inputs and outputs:
	... (2 * G).links
	(2, 2)
	>>> G.links
	(1, 1)
	>>> H.links
	(1, 1)

.. [feedback_systems] Karl Johan Åström and Richard M. Murray, "`Feedback Systems`_", Princeton University Press, 2012

.. _Feedback Systems: http://www.cds.caltech.edu/~murray/books/AM08/pdf/am08-hyperref_28Sep12.pdf

