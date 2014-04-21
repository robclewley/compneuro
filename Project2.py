"""
Project 2 - stochastic resonance
"""
from __future__ import division
import sys
from PyDSTool import *
from common_lib import *
from Ch9_HH_red import *

# default dopri -- you will extremely long waits with Vode!
gentype='dopri'

# Parameter An = noise amplitude
#           As = sine wave amplitude
#           f = frequency, should be 55 Hz for this project
# Don't change any of the parameters for this project except As and An!
par_args = {'tau_v': 0.8, 'tau_r': 3,
            'As': 0., 'f': 55, 'An': 0., 'Iapp': 0.01}
ic_args = {'v':-0.75, 'r': 0.5}

# build neuron model
HH = makeHHneuron('HHred_noise', par_args, ic_args, const_I=True, gentype=gentype)

# set special conditions for using noise
HH.eventstruct.setActiveFlag('min_ev', False)
HH.set(tdata=[0,50])  # this is just a temporary value, you will change this later
if gentype == 'vode':
    HH.set(algparams={'init_step': 0.02, 'stiff': False})
else:
    HH.set(algparams={'init_step': 0.08})

# PUT YOUR CODE AFTER HERE!