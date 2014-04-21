"""
Responses to zero-mean Gaussian white noise shown in reduced,
2D version of Hodgkin-Huxley model

Section 9.4
"""
from __future__ import division
import sys
from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *
from common_lib import *
from Ch9_HH_red import *

# default
gentype='radau'  # dopri, euler, etc.

if gentype == 'vode':
    tmax = 150
    num_vals = 20
    print "This script will run incredibly slowly with Vode."
    print "Use Dopri in VCL if necessary to see reasonable results in a minute or so..."
else:
    # improves accuracy to run longer and use more points in the graph
    tmax = 250
    num_vals = 50


# Parameter An = noise amplitude
#           As = sine wave amplitude
#           f = frequency, should be >= 50 Hz
par_args = {'tau_v': 0.5, 'tau_r': 4,
            'As': 0., 'f': 700, 'An': 0.01, 'Iapp': 0.04}
ic_args = {'v':-0.8, 'r': 0.25}

def avfreq(traj):
    """
    Average frequency in presence of noise
    """
    evs = traj.getEventTimes('thresh_ev')
    l = len(evs)
    if l == 0:
        return 0
    elif l == 1:
        print "Not enough events found"
        return 0
    else:
        # take mean average of inter-spike intervals between n events found
        # evs[1:] is all but one event ranging 1 -> n
        # evs[:-1] is all but one event ranging 1 -> n-1
        # evs is just a list, so convert to array to do array arithmetic
        return 1000./mean(array(evs[1:]) - array(evs[:-1]))


# build neuron model
HH = makeHHneuron('HHred_noise', par_args, ic_args, const_I=True, gentype=gentype)

# set special conditions for using noise
HH.set(tdata=[0,50])
if gentype == 'vode':
    HH.set(algparams={'init_step': 0.02, 'stiff': False})
else:
    HH.set(algparams={'init_step': 0.08})


def test_noise(sd, tmax=50, silent=False):
    """Optional silent option suppresses text output and plots
    """
    HH.set(pars={'An': sd},
           tdata=[0,tmax])
    traj = HH.compute('test')
    pts = traj.sample()
    av = avfreq(traj)
    if not silent:
        plt.clf()
        plt.plot(pts['t'], pts['v'], 'b')
        print "Frequency response was from last two spikes was:", freq(traj)
        print "Average frequency was:", av
    return av

# Find response to increasing noise levels
fs = []
ns = linspace(0, 1.3, num_vals)
for n in ns:
    print "noise s.d. =", n
    sys.stdout.flush()
    fs.append(test_noise(n, tmax, silent=True))

print "Any 'steps' in the plot are due to the integer number of spikes that can"
print "fit into the 250ms window at one time, even as Iapp varies smoothly"

plt.figure(1)
plt.plot(ns, fs, 'ko-')
plt.title("Av. frequency response as noise standard dev. increases")
plt.xlabel('s.d. /pA')
plt.ylabel('av. freq /Hz')

plt.figure(2)
# single call version with plots
test_noise(0.1)

plt.show()

