"""
Spike rate neuron non-autonomous differential equation
See Section 2.3
"""

from __future__ import division
from PyDSTool import *
from common_lib import *

# Set up external input (non-autonomous ODE)
ts = linspace(0, 100, 500)
ps = 80*np.sin(0.1*ts)
ps = asarray(ps>0,int)*ps  # half-rectify (P < 0 doesn't make sense)
input_pts = Pointset(coorddict={'P': ps}, indepvararray=ts)
input_signal = pointset_to_traj(input_pts)

# Define model
DSargs = args()
DSargs.name = 'spike_rate'
# This is how to plug in a time-dependent input to the RHS
DSargs.inputs = {'P': input_signal.variables['P']}
DSargs.ics = {'R': 0.1}
DSargs.pars = {'tau': 15}
# Use threshold function maker from common_lib
DSargs.fnspecs = {'S': thresh_Naka_Rushton_fndef(N=2, half_on=40, max_val=100)}
DSargs.tdata = [0, 100]
DSargs.varspecs = {'R': '(-R + S(P))/tau'}

DS = Generator.Vode_ODEsystem(DSargs)

# Test and plot results
traj = DS.compute('test')
pts = traj.sample()
plt.plot(pts['t'], pts['R'], 'k', label='R')
plt.plot(ts, ps, 'g', label='P')
plt.legend(loc='lower right')
plt.show()