# Project 1
from __future__ import division
from WC_net import *
#from PyDSTool.Toolbox.phaseplane import *

builder = rate_network()

S1 = thresh_Naka_Rushton_fndef(2, '120', 100)
S2 = thresh_Naka_Rushton_fndef(2, '120', 100)

builder.add_neuron('E_RE', tau=20, ic=1, thresh_fn=S2)
builder.add_neuron('E_RI', tau=20, ic=1, thresh_fn=S2)
builder.add_neuron('I', tau=20, ic=1, thresh_fn=S2)

builder.add_syn_input_to_neuron('E_RI', 'I', 3)
builder.add_syn_input_to_neuron('I', 'E_RE', -5)

## Add adaptation to response
#builder.add_rate('A1', tau=4000, ic=0)
#builder.add_rate('A2', tau=4000, ic=0)
#builder.add_interaction('E1', 'A1', 0.7)
#builder.add_interaction('E2', 'A2', 0.7)

builder.add_bias_input('E_RE', 0, 'sE')
builder.add_bias_input('E_RI', 0, 'sI')

net = builder.make_network()

net.set(tdata=[0,8000],
        algparams={'init_step': 0.75})

# Uncomment if you want to get a single trajectory computed
##traj = net.compute('test')
##pts = traj.sample()

# Piecewise protocol
protocol = []

protocol.append({'pars': {'sE': 0},
                 'tdur': 30})

protocol.append({'pars': {'sE': 50},
                 'tdur': 100})

protocol.append({'pars': {'sE': 50,
                          'sI': 50},
                 'tdur': 200})

protocol.append({'pars': {'sE': 50,
                          'sI': 0},
                 'tdur': 200})

protocol.append({'pars': {'sE': 0},
                 'tdur': 500})

traj, pts = pcw_protocol(net, protocol)

plt.figure(0)
plt.plot(pts['t'], pts['E_RE'], 'g')
plt.plot(pts['t'], pts['E_RI'], 'r')
plt.show()
