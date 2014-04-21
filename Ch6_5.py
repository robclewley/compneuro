# Example from Ch. 6.5
from __future__ import division
from WC_net import *
from PyDSTool.Toolbox.phaseplane import *

builder = rate_network()

# Both ICs > 20 will lead to stable equilibrium at (80, 80)
# Saddle at (20, 20)
# Stable node at (0, 0)
# Therefore, there is hysteresis

S = thresh_Naka_Rushton_fndef(2, 120, 100, with_if=False)

builder.add_neuron('E1', tau=20, ic=15, thresh_fn=S)
builder.add_neuron('E2', tau=20, ic=25, thresh_fn=S)
builder.add_syn_input_to_neuron('E1', 'E2', 3)
builder.add_bias_input('E1', 0, 'K')
builder.add_syn_input_to_neuron('E2', 'E1', 3)

net = builder.make_network()

net.set(tdata=[0,5000],
        algparams={'init_step': 0.5})



jac_fn = make_Jac(net)

def do_nullcs(K):
    net.set(pars={'K': K})
    fp_coords = find_fixedpoints(net, {'E1': [0, 200], 'E2': [0, 200]},
                       jac=jac_fn)
    nullc_E1, nullc_E2 = find_nullclines(net, 'E1', 'E2',
                         {'E1': [0, 100], 'E2': [0, 100]},
                         max_step=1, crop_tol_pc=0,
                         fps=fp_coords, n=3,
                         jac=jac_fn)
    return fp_coords, nullc_E1, nullc_E2


# Piecewise protocol
protocol = []

protocol.append({'pars': {'K': -50},
                 'tdur': 300})

protocol.append({'pars': {'K': 30},
                 'tdur': 200})

protocol.append({'pars': {'K': -50},
                 'tdur': 500})

traj, pts = pcw_protocol(net, protocol)

plt.figure(0)
plt.plot(pts['t'], pts['E1'], 'g')
plt.plot(pts['t'], pts['E2'], 'r')

# plot each part of the trajectory for the piecewise segments
plt.figure(1)
ix1 = pts.find(300,1)
plt.plot(pts['E1'][:ix1], pts['E2'][:ix1], 'k')
plt.plot(pts['E1'][ix1:], pts['E2'][ix1:], 'k--')
fp_coords, nullc_E1, nullc_E2 = do_nullcs(-50)
fp1 = fixedpoint_2D(net, Point(fp_coords[0]), coords=['E1', 'E2'],
                       eps=1e-6, jac=jac_fn)
plt.plot(nullc_E1[:,0],nullc_E1[:,1], label='N\_E1')
plt.plot(nullc_E2[:,0],nullc_E2[:,1], label='N\_E2')

plt.figure(2)
ix2 = pts.find(500,1)
plt.plot(pts['E1'][:ix1], pts['E2'][:ix1], 'k--')
plt.plot(pts['E1'][ix1:ix2], pts['E2'][ix1:ix2], 'k')
fp_coords, nullc_E1, nullc_E2 = do_nullcs(30)
plt.plot(nullc_E1[:,0],nullc_E1[:,1], label='N\_E1')
plt.plot(nullc_E2[:,0],nullc_E2[:,1], label='N\_E2')

plt.figure(3)
plt.plot(pts['E1'][:ix2], pts['E2'][:ix2], 'k--')
plt.plot(pts['E1'][ix2:], pts['E2'][ix2:], 'k')
fp_coords, nullc_E1, nullc_E2 = do_nullcs(-50)
plt.plot(nullc_E1[:,0],nullc_E1[:,1], label='N\_E1')
plt.plot(nullc_E2[:,0],nullc_E2[:,1], label='N\_E2')

plt.show()
