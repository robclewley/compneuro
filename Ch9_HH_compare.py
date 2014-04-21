"""
Human cortical neuron using A-current model in reduced, 2D version of Hodgkin-Huxley model
Section 9.5
"""
from __future__ import division
from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *
from common_lib import *
import Ch9_HH_red
import Ch9_HH

gentype='vode'  # dopri, euler, etc.

# Parameter An = noise amplitude
#           As = sine wave amplitude
#           f = frequency, should be >= 50 Hz
par_args = {'tau_v': 1, 'tau_r': 5.6,
            'As': 0, 'f': 700, 'An': 0., 'Iapp': 0.8}
ic_args = {'v':-0.8, 'r': 0.25}

def test_I(gen, Iapp, tmax=300, silent=False):
    geb.set(pars={'Iapp': Iapp},
           tdata=[0,tmax])
    traj = gen.compute('test')
    pts = traj.sample()
    f = freq(traj)
    if not silent:
        plt.clf()
        plt.plot(pts['t'], pts['v'], 'b')
        plt.ylim([-0.85, 0.4])
        print "Frequency response was:", f
    return f

# original version
HH = Ch9_HH.makeHHneuron('HH', par_args, ic_args, const_I=True,
                  gentype=gentype)

# 2D reduced version
HHred = Ch9_HH_red.makeHHneuron('HHred', par_args, ic_args, const_I=True,
                  gentype=gentype)

# vary Iapp up to 2
# 0.791 is the closest to the saddle-node bif point to 3 decimal places
test_I(0.791, 500)

plt.show()

