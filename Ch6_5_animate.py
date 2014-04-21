# Example from Ch. 6.5
from __future__ import division
from WC_net import *
from PyDSTool.Toolbox.phaseplane import *

builder = rate_network()

# Both ICs > 20 will lead to stable equilibrium at (80, 80)
# Saddle at (20, 20)
# Stable node at (0, 0)
# Therefore, there is hysteresis

S = thresh_Naka_Rushton_fndef(2, 120, 100)

builder.add_neuron('E1', tau=20, ic=15, thresh_fn=S)
builder.add_neuron('E2', tau=20, ic=25, thresh_fn=S)
builder.add_syn_input_to_neuron('E1', 'E2', 3)
builder.add_bias_input('E1', 0, '140*pow(sin(t/350),2) - 80')
builder.add_syn_input_to_neuron('E2', 'E1', 3)

net = builder.make_network()

net.set(tdata=[0,1400],
        algparams={'init_step': 0.5})

def do_nullcs(t):
    fp_coords = find_fixedpoints(net, {'E1': [0, 200], 'E2': [0, 200]},
                                 t=t)
    nullc_E1, nullc_E2 = find_nullclines(net, 'E1', 'E2',
                         {'E1': [0, 100], 'E2': [0, 100]},
                         max_step=1, crop_tol_pc=0,
                         fps=fp_coords, n=3, t=t)
    return fp_coords, nullc_E1, nullc_E2

traj = net.compute('test')
pts = traj.sample()

# "animate" the nullclines
ts = linspace(0, 1400, 100)
for i, t in enumerate(ts):
    fp_coords, nullc_E1, nullc_E2 = do_nullcs(t)
    plt.plot(nullc_E1[:,0],nullc_E1[:,1], label='N\_E1')
    plt.plot(nullc_E2[:,0],nullc_E2[:,1], label='N\_E2')
    plt.plot(pts['E1'], pts['E2'], 'k')
    pt = traj(t)
    plt.plot(pt['E1'], pt['E2'], 'ko')
    plt.savefig('nullc_anim_%i.png' %i)
    plt.clf()

