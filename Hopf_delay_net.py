"""
"""

from __future__ import division
from WC_net import *
from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *

builder = rate_network()

S = thresh_Naka_Rushton_fndef(2, 50, 100)

builder.add_neuron('E', tau=20, ic=55, thresh_fn=S)
builder.add_rate('I', tau=50, ic=300)

builder.add_rate('D1', tau=0.05, ic=0)
builder.add_rate('D2', tau=0.05, ic=0)

builder.add_syn_input_to_neuron('D2', 'E', -1)
builder.add_bias_input('E', 350, 'K')

builder.add_interaction('E', 'D1', 1)
builder.add_interaction('I', 'D2', 1)

builder.add_interaction('D1', 'I', 6)

net = builder.make_network()

net.set(tdata=[0,2000],
        algparams={'init_step': 0.75})

# net.Rhs(0, {'E':50,'D1':50,'I':300,'D2':300})

def test_tau(tau, ics=None):
    net.set(pars={'tau_D1': tau,
                  'tau_D2': tau})
    if ics is not None:
        net.set(ics=ics)
    traj = net.compute('test')
    pts = traj.sample()
    plt.figure(1)
    plt.clf()
    plt.plot(pts['t'], pts['E'], 'g')
    plt.plot(pts['t'], pts['I'], 'r')
    return pts


pts = test_tau(10.5)
#pts = test_tau(11)

plt.show()
