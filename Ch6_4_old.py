# Example from Ch. 6.4
from __future__ import division
from WC_net import *
from PyDSTool.Toolbox.phaseplane import *

builder = rate_network()

S = thresh_Naka_Rushton_fndef(2, 120, 100, with_if=False)

builder.add_neuron('E1', tau=20, ic=15, thresh_fn=S)
builder.add_neuron('E2', tau=20, ic=25, thresh_fn=S)
builder.add_syn_input_to_neuron('E1', 'E2', 3)
builder.add_syn_input_to_neuron('E2', 'E1', 3)

net = builder.make_network()

net.set(tdata=[0,300],
        algparams={'init_step': 0.5})
traj = net.compute('test')
pts = traj.sample()
plt.plot(pts['t'], pts['E1'], 'g')
plt.plot(pts['t'], pts['E2'], 'r')

# Do nullclines

jac_fn = make_Jac(net)

fps = find_fixedpoints(net, {'E1': [0, 200], 'E2': [0, 200]},
                       jac=jac_fn, n=6)
nullc_E1, nullc_E2 = find_nullclines(net, 'E1', 'E2',
                         {'E1': [0, 100], 'E2': [0, 100]},
                         max_step=1, crop_tol_pc=0,
                         fps=fps, n=3,
                         jac=jac_fn)

fp1 = fixedpoint_2D(net, Point(fps[0]), coords=['E1', 'E2'],
                       eps=1e-6, jac=jac_fn)

# Both ICs > 20 will lead to stable equilibrium at (80, 80)
# Saddle at (20, 20)
# Stable node at (0, 0)
# Therefore, there is hysteresis

plt.figure(2)
plt.plot(nullc_E1[:,0],nullc_E1[:,1], label='N\_E1')
plt.plot(nullc_E2[:,0],nullc_E2[:,1], label='N\_E2')
plt.plot(pts['E1'],pts['E2'], 'k')
plt.plot(pts['E1'][0],pts['E2'][0], 'go')
plt.plot(pts['E1'][-1],pts['E2'][-1], 'ro')
plt.legend(loc='lower right')

plt.show()
