"""
1D linear ODE for exploring time scales
"""
from __future__ import division

from PyDSTool import *
from euler import euler_integrate

icdict = {'v': -75}
# tau = RC for a linear membrane model
pardict = {'tau': 1, 'v0': -60}

# v0 is a set-point for v
v_rhs = '(v0-v)/tau'

vardict = {'v': v_rhs}

# create an empty object instance of the args class, call it DSargs
DSargs = args()
# name our model
DSargs.name = 'timescales'
# assign the dictionary of ICs, icdict, to the ICs attribute
DSargs.ics = icdict
# assign the dictionary of parameters, pardict, to the pars attribute
DSargs.pars = pardict
# declare how long we expect to integrate for (can be changed later)
DSargs.tdata = [0, 60]
# assign the RHS definitions for the ODE, the vardict dictionary,
# to the varspecs attribute of DSargs
DSargs.varspecs = vardict

# Create an object representing the implementation of the model.
# In PyDSTool, this is known as a Generator object, because
# it can numerically compute ("generate") trajectories using the VODE
# integration scheme.
DS = Generator.Vode_ODEsystem(DSargs)

# This shows how to change just these entries once the Generator has been created
DS.set(ics={'v': -75})

# A function that tests the outcome of a given value of parameter tau
def test_tau(tau):
    DS.set(pars={'tau': tau},
           algparams={'init_step': min(tau/2,0.5)})
    traj = DS.compute('test')
    pts = traj.sample()
    return pts

# A loop to test a distribution of values of tau
for tau in 40*exp(-linspace(1, 6, 6)):
    pts = test_tau(tau)
    plt.plot(pts['t'], pts['v'], '.-', label='tau=%.3f (VODE)' %tau)

# Find solutions using Euler integration
def test_tau_euler(tau, step, symb):
    DS.set(pars={'tau': tau})
    print "Testing Euler method with tau =", tau
    ts, vs = euler_integrate(DS, 0, 20, step, DS.initialconditions['v'])
    plt.plot(ts, vs, symb+':', label='tau=%.2f (Euler@%.2f)' % (tau, step))

test_tau_euler(3, 0.4, 'mx')

### TRY THESE AT THE COMMAND PROMPT
#test_tau_euler(0.3, 0.4, 'k.')
#test_tau_euler(0.1, 0.4, 'r.')

plt.legend(loc='lower right')
plt.xlabel('t')
plt.xlim([-1,60])
plt.show()