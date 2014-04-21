# Example from Ch. 7.3
# Doesn't work according to book :(
from __future__ import division
from WC_net import *
from PyDSTool.Toolbox.phaseplane import *

print "Doesn't work!"
1/0

builder = rate_network()

RB = (['C', 'H', 'A'], 'max((6*C-5*H)/(1+9*A),0)')
RG = (['B'], '(50*B)/(13+B)')

builder.add_rate('C', tau=10, ic=0)
builder.add_rate('H', tau=100, ic=0)
builder.add_neuron('B', tau=10, ic=0, thresh_fn=RB)
builder.add_rate('A', tau=80, ic=0)
builder.add_rate('P', tau=4000, ic=0)
builder.add_neuron('G', tau=20, ic=0, thresh_fn=RG)
builder.add_bias_input('C', 0, 'L')
builder.add_syn_input_to_neuron('C', 'B', 1)
builder.add_syn_input_to_neuron('B', 'G', 1)
builder.add_interaction('P*H', 'C', -1, 'g_H_C')
builder.add_interaction('C', 'H', 1)
builder.add_interaction('B', 'A', 1)
builder.add_interaction('A', 'P', 0.1)

net = builder.make_network(gentype='Dopri')

net.set(tdata=[0,5000],
        algparams={'init_step': 1})

Lvals = [10, 100, 500, 1000]
Gend = []

# Piecewise protocol
for Lval in Lvals:
    protocol = []

    protocol.append({'pars': {'L': 0},
                     'tdur': 30})

    protocol.append({'pars': {'L': 300},
                     'tdur': 8000})

    protocol.append({'pars': {'L': 0},
                     'tdur': 500})

    traj, pts = pcw_protocol(net, protocol)

    Gend.append( traj(7000)['G'] )

plt.plot(pts['t'], pts['P'], 'g', label='P')
plt.plot(pts['t'], pts['A'], 'r', label='A')
plt.plot(pts['t'], pts['G'], 'k', label='G')
plt.plot(pts['t'], pts['H'], 'b', label='H')
plt.plot(pts['t'], pts['C'], 'y', label='C')
plt.plot(pts['t'], pts['B'], 'c', label='B')
plt.xlabel('t')
plt.ylabel('spike rate')
plt.legend()

plt.figure()
plt.plot(Lvals, Gend, 'o')

plt.show()
#plt.title('E1(t) = green, E2(t) = red')
