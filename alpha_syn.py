"""
"Alpha function" synapse of Rall. Chapter 12.2
"""

from __future__ import division

from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *
from common_lib import *

icdict = {'a1': 0, 'a2': 0}
pardict = {'tau_syn': 2, 'vthresh': -10,
           'vpre': -80}

DSargs = args()
DSargs.name = 'alpha_syn'
DSargs.ics = icdict
DSargs.pars = pardict
DSargs.tdata = [0, 3]
DSargs.auxvars = ['vpre_aux']
DSargs.algparams = {'init_step': 1e-3}
# PyDSTool has a built-in Heaviside function
DSargs.varspecs = {'a1': '(-a1+heav(vpre-vthresh))/tau_syn',
                   'a2': '(-a2+a1)/tau_syn',
                   'vpre_aux': 'vpre'}

syn = Generator.Vode_ODEsystem(DSargs)


# primitive protocol for mimicking the pre-synaptic voltage's
# action potential for 1 ms starting at t = 5ms
t1 = 5
s1 = args(pars={'vpre': -80},
       tdata=[0, t1])
t2 = 1
s2 = args(pars={'vpre': 50},
       tdata=[0, t2])
t3 = 40
s3 = args(pars={'vpre': -80},
       tdata=[0, t3])


def alpha(t):
    """Explicit solution of alpha function for Dirac delta function impulse
    for presynaptic spike.

    Uses current value of tau_syn in model.
    Accepts scalar or vector t.
    """
    tau = syn.pars['tau_syn']
    return t/(tau*tau)*exp(-t/tau)


def test(tau):
    syn.set(pars={'tau_syn': tau})
    traj, pts = pcw_protocol(syn, [s1,s2,s3])

    plt.figure(1)
    plt.clf()
    plt.plot(pts['t'], pts['vpre_aux']*0.001, 'k', linewidth=3, label='pre-syn v /1000')
    plt.plot(pts['t'], pts['a2'], 'g', linewidth=2, label='a2 (output)')
    plt.plot(pts['t'], pts['a1'], 'r:', label='a1')

    ts = linspace(0, 30, 500)
    ss = alpha(ts)
    # offset ts for alpha function by onset of pre-synaptic spike
    plt.plot(ts+t1, ss, 'g--', label='s (explicit)')

    plt.xlabel('t')
    plt.legend(loc='upper right')
    plt.title('tau syn = %.2f ms' % tau)
    plt.ylim([-0.1, 0.9])
    plt.xlim([0, max(pts['t'])])
    plt.show()

test(0.5)




