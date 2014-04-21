"""
Forced linear ODE with a three-stage delay model
"""
from __future__ import division

from PyDSTool import *

DSargs = args()
DSargs.name = 'delays'
DSargs.ics =  {'x': 0.2, 'x_delay': 0.1, 'x_delay2': 0, 'x_delay3': 0}
DSargs.pars = {'tau': 0.4, 'k': .91, 'a': -1,
               'tau_d': 0.355}
DSargs.tdata = [0, 300]
DSargs.varspecs = {'x': '(a*sin(k*t)-x_delay3)/tau',
                   'x_delay': '(x-x_delay)/tau_d',
                   'x_delay2': '(x_delay-x_delay2)/tau_d',
                   'x_delay3': '(x_delay2-x_delay3)/tau_d'}

DS = Generator.Vode_ODEsystem(DSargs)

# A function that tests the outcome of a given value of parameter tau
def test_tau(tau):
    DS.set(pars={'tau_d': tau/3.})
    traj = DS.compute('test')
    pts = traj.sample()
    return pts

# show the delay of 0.1
# compare the times of two peaks
pts = test_tau(0.3)
plt.figure()
plt.plot(pts['t'], pts['x'], 'k')
plt.plot(pts['t'], pts['x_delay3'], 'b')
plt.xlim([0,10])

# show route to chaos
for tau_d in [0.6,  0.96,  1.02,  1.05,  1.065,  1.08]:
    pts = test_tau(tau_d)
    plt.figure()
    plt.plot(pts['t'], pts['x'], 'k')
    plt.xlabel('t')
    plt.xlim([-1,DS.tdata[1]])
    plt.ylim([-abs(DS.pars['a'])*3,abs(DS.pars['a'])*3])
    plt.title('tau\_d = %.3f' % tau_d)

plt.show()