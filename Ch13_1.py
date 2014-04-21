"""
Example from Ch. 13.1, Lamprey serotonin modulation.

This model reproduces Eq. (13.2) and (13.3)
"""
from __future__ import division
from WC_net import *
from PyDSTool.Toolbox.phaseplane import *
import sys

# -------------------------

### Version with 5HT (Serotonin) modulation blocked

builder_no5HT = rate_network()

S = thresh_Naka_Rushton_fndef(2, '64+g*HEL', 100,
                              sys_pars=['g'])

builder_no5HT.add_neuron('EL', tau=9, ic=1, thresh_fn=S)
builder_no5HT.add_rate('HEL', tau=400, ic=1)
builder_no5HT.add_syn_input_to_neuron('EL', 'EL', 6)
builder_no5HT.add_syn_input_to_neuron('AL', 'EL', 1)
builder_no5HT.pardefs['AL'] = 10
builder_no5HT.pardefs['g'] = 6
builder_no5HT.add_interaction('EL', 'HEL', 1)

burst_on = Events.makeZeroCrossEvent('EL-60', 1,
                                       {'name': 'thresh_ev',
                                        'eventtol': 1e-2,
                                        'precise': True,
                                        'term': False},
                                       varnames=['EL'])

net_no5HT = builder_no5HT.make_network(events=burst_on)

net_no5HT.set(tdata=[0,4500],
        algparams={'init_step': 1})


### Version with modulation ("normal")

builder_norm = rate_network()

S2 = thresh_Naka_Rushton_fndef(2, '64+g(AL)*HEL', 100,
                              sys_pars=['g', 'AL'])

builder_norm.add_neuron('EL', tau=9, ic=1, thresh_fn=S2)
builder_norm.add_rate('HEL', tau='tauH(AL)', ic=1)
builder_norm.add_syn_input_to_neuron('EL', 'EL', 6)
builder_norm.add_syn_input_to_neuron('AL', 'EL', 1)
builder_norm.pardefs['AL'] = 10
builder_norm.fndefs['g'] = (['A'], '6+(0.09*A)**2')
builder_norm.fndefs['tauH'] = (['A'], '400/(1+(0.2*A)**2)')
builder_norm.add_interaction('EL', 'HEL', 1)

net_norm = builder_norm.make_network(events=burst_on)

net_norm.set(tdata=[0,4500],
        algparams={'init_step': 1})

# -------------------------

def freq(traj):
    ts = traj.getEventTimes('thresh_ev')
    if len(ts) > 1:
        return 1000./(ts[-1]-ts[-2])
    else:
        return 0

def stim(net, A):
    net.set(pars={'AL': A})
    traj = net.compute('test')
    pts = traj.sample()
    plt.clf()
    plt.plot(pts['t'], pts['EL'], 'k', label='EL')
    plt.plot(pts['t'], pts['HEL'], 'r', label='HEL')
    plt.legend()
    plt.xlabel('t (ms)')
    plt.ylabel('firing rate (Hz)')
    return pts, freq(traj)


fs_no5HT = []
fs_norm = []
As = linspace(0, 25, 15)
for A in As:
    print "Testing A = %.3f" %A
    sys.stdout.flush()
    pts, f = stim(net_no5HT, A)
    fs_no5HT.append(f)
    pts, f = stim(net_norm, A)
    fs_norm.append(f)

plt.figure(2)
plt.plot(As, fs_norm, 'ko-', label='normal')
plt.plot(As, fs_no5HT, 'ko--', label='5HT blocked')
plt.legend()
plt.show()
