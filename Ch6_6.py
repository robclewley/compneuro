# Example from Ch. 6.6
from __future__ import division
from WC_net import *
from PyDSTool.Toolbox.phaseplane import *

builder = rate_network()

S1 = thresh_Naka_Rushton_fndef(2, '120+A1', 100)
S2 = thresh_Naka_Rushton_fndef(2, '120+A2', 100)

builder.add_neuron('E1', tau=20, ic=0, thresh_fn=S1)
builder.add_neuron('E2', tau=20, ic=0, thresh_fn=S2)
builder.add_syn_input_to_neuron('E1', 'E2', 3)
builder.add_syn_input_to_neuron('E2', 'E1', 3)
builder.add_rate('A1', tau=4000, ic=0)
builder.add_rate('A2', tau=4000, ic=0)
builder.add_interaction('E1', 'A1', 0.7)
builder.add_interaction('E2', 'A2', 0.7)
builder.add_bias_input('E1', 0, 'K')

net = builder.make_network()

net.set(tdata=[0,8000],
        algparams={'init_step': 0.75})

# Uncomment if you want to get a single trajectory computed
##traj = net.compute('test')
##pts = traj.sample()

# Piecewise protocol
protocol = []

protocol.append({'pars': {'K': 0},
                 'tdata': [0, 30]})

protocol.append({'pars': {'K': 50},
                 'tdata': [0, 200]})

protocol.append({'pars': {'K': 0},
                 'tdata': [0, 7000]})

traj, pts = pcw_protocol(net, protocol)

plt.figure(0)
plt.plot(pts['t'], pts['E1'], 'g')
plt.plot(pts['t'], pts['A1'], 'r')
plt.show()

1/0

# Uncomment the rest for the nullclines...

# --------------------------

# Do nullclines for case where A I.C.'s = 24
net.set(ics={'A1': 24, 'A2': 24})
jac_fn = make_Jac(net, ['E1', 'E2'])

fps = find_fixedpoints(net, {'E1': [0, 100], 'E2': [0, 100],
                             'A1': 24, 'A2': 24}, jac=jac_fn)

nullc_E1, nullc_E2 = find_nullclines(net, 'E1', 'E2',
                         {'E1': [0, 100], 'E2': [0, 100],
                          'A1': 24, 'A2': 24},
                         max_step=1, crop_tol_pc=0,
                         fps=fps, n=3,
                         jac=jac_fn)

plt.figure(1)
plt.plot(nullc_E1[:,0],nullc_E1[:,1], label='N\_E1')
plt.plot(nullc_E2[:,0],nullc_E2[:,1], label='N\_E2')
plt.show()
