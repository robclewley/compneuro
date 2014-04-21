# Project 1
from __future__ import division
from WC_net import *
#from PyDSTool.Toolbox.phaseplane import *

builder = rate_network()


# Coupling strengths
cEI = 2
cIE = 1
cRE = 4

# Background stimulus to all E cells
spon = 12

# lateral R -> E coupling factor (0 for off)
lateral_RE_fac = 0.1

#
# SE is the response function of I cells to excitatory inputs
SE = thresh_Naka_Rushton_fndef(4, 20, 60)

# SI is the response function of E cells to inhibitory inputs
SI = thresh_Naka_Rushton_fndef(2, 20, 60)


# very fast sensory neurons
builder.add_neuron('R1', tau=10, ic=0, thresh_fn=SE)
builder.add_neuron('R2', tau=10, ic=0, thresh_fn=SE)
builder.add_neuron('R3', tau=10, ic=0, thresh_fn=SE)


builder.add_neuron('E1', tau=80, ic=0, thresh_fn=SI)
builder.add_neuron('E2', tau=80, ic=0, thresh_fn=SI)
builder.add_neuron('E3', tau=80, ic=0, thresh_fn=SI)
builder.add_neuron('E4', tau=80, ic=0, thresh_fn=SI)
builder.add_neuron('E5', tau=80, ic=0, thresh_fn=SI)

builder.add_neuron('I1', tau=20, ic=0, thresh_fn=SE)
builder.add_neuron('I2', tau=20, ic=0, thresh_fn=SE)
builder.add_neuron('I3', tau=20, ic=0, thresh_fn=SE)
builder.add_neuron('I4', tau=20, ic=0, thresh_fn=SE)

builder.add_syn_input_to_neuron('R1', 'E1', lateral_RE_fac*cRE)
builder.add_syn_input_to_neuron('R1', 'E2', cRE)
builder.add_syn_input_to_neuron('R1', 'E3', lateral_RE_fac*cRE)
builder.add_syn_input_to_neuron('R2', 'E2', lateral_RE_fac*cRE)
builder.add_syn_input_to_neuron('R2', 'E3', cRE)
builder.add_syn_input_to_neuron('R2', 'E4', lateral_RE_fac*cRE)
builder.add_syn_input_to_neuron('R3', 'E3', lateral_RE_fac*cRE)
builder.add_syn_input_to_neuron('R3', 'E4', cRE)
builder.add_syn_input_to_neuron('R3', 'E5', lateral_RE_fac*cRE)

builder.add_syn_input_to_neuron('E1', 'I1', cEI)
builder.add_syn_input_to_neuron('E2', 'I1', cEI)
builder.add_syn_input_to_neuron('E2', 'I2', cEI)
builder.add_syn_input_to_neuron('E3', 'I2', cEI)
builder.add_syn_input_to_neuron('E3', 'I3', cEI)
builder.add_syn_input_to_neuron('E4', 'I3', cEI)
builder.add_syn_input_to_neuron('E4', 'I4', cEI)
builder.add_syn_input_to_neuron('E5', 'I4', cEI)

builder.add_syn_input_to_neuron('I1', 'E1', -cIE)
builder.add_syn_input_to_neuron('I1', 'E2', -cIE)
builder.add_syn_input_to_neuron('I2', 'E2', -cIE)
builder.add_syn_input_to_neuron('I2', 'E3', -cIE)
builder.add_syn_input_to_neuron('I3', 'E3', -cIE)
builder.add_syn_input_to_neuron('I3', 'E4', -cIE)
builder.add_syn_input_to_neuron('I4', 'E4', -cIE)
builder.add_syn_input_to_neuron('I4', 'E5', -cIE)

# External stimulus
builder.add_bias_input('R1', 0, 's1')
builder.add_bias_input('R2', 0, 's2')
builder.add_bias_input('R3', 0, 's3')

#
builder.add_bias_input('E1', spon, 'spon1')
builder.add_bias_input('E2', spon, 'spon2')
builder.add_bias_input('E3', spon, 'spon3')
builder.add_bias_input('E4', spon, 'spon4')
builder.add_bias_input('E5', spon, 'spon5')


net = builder.make_network()

net.set(tdata=[0,8000],
        algparams={'init_step': 1})

# Uncomment if you want to get a single trajectory computed
##traj = net.compute('test')
##pts = traj.sample()

# Piecewise protocol
protocol = []

protocol.append({'pars': {'s1': 0, 's2': 0, 's3': 0},
                 'tdata': [0, 400]})

protocol.append({'pars': {'s1': 15, 's2': 25, 's3': 10},
                 'tdata': [0, 600]})

protocol.append({'pars': {'s1': 0, 's2': 0, 's3': 0},
                 'tdata': [0, 500]})

traj, pts = pcw_protocol(net, protocol)

plt.figure(1)
plt.plot(pts['t'], pts['E1'], 'g')
plt.plot(pts['t'], pts['E2'], 'r')
plt.plot(pts['t'], pts['E3'], 'k')
plt.plot(pts['t'], pts['E4'], 'y')
plt.plot(pts['t'], pts['E5'], 'b')
plt.plot(pts['t'], pts['I1'], 'g--')
plt.plot(pts['t'], pts['I2'], 'r--')
plt.plot(pts['t'], pts['I3'], 'k--')
plt.plot(pts['t'], pts['I4'], 'y--')

contrast = abs(max(pts['E3']) - min(min(pts['E2']), min(pts['E4'])))
print "Contrast =", contrast

plt.figure(2)
plt.plot(1, max(pts['E1']), 'ko')
plt.plot(2, max(pts['E2']), 'ko')
plt.plot(3, max(pts['E3']), 'ko')
plt.plot(4, max(pts['E4']), 'ko')
plt.plot(5, max(pts['E5']), 'ko')
plt.ylim([0,60])
plt.show()
