"""
Sub-harmonic resonance shown in reduced, 2D version of Hodgkin-Huxley model
Section 9.6
"""
from __future__ import division
from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *
from common_lib import *
from Ch9_HH_red import *

gentype='vode'  # dopri, euler, etc.

# Parameter An = noise amplitude
#           As = sine wave amplitude
#           f = frequency, should be >= 50 Hz
par_args = {'tau_v': 0.8, 'tau_r': 1.9,
            'As': 0.5, 'f': 700, 'An': 0.0, 'Iapp': 0.}
ic_args = {'v':-0.8, 'r': 0.25}


HH = makeHHneuron('HHred', par_args, ic_args, const_I=True, gentype=gentype)
HH.set(tdata=[0,45])

def test_f(f):
    HH.set(pars={'f': f})
    traj = HH.compute('test')
    pts = traj.sample()
    plt.clf()
    plt.plot(pts['t'], pts['v'], 'b')
    plt.plot(pts['t'], 0.1*pts['sine']-1, 'k')
    print "Response freq was", freq(traj)

test_f(200)
plt.show()

