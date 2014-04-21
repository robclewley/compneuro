"""
Eqn (10.1) for adaptation model with I_AHP
"""

from __future__ import division
from PyDSTool import *
from Ch10_model import *

gentype = 'vode' # 'dopri' # 'vode'

par_args = {'gr': 26, 'gh': 13, 'tau_r': 5.6, 'tau_h': 99, 'tau_v': 0.9,
                'Ena': 0.48, 'Er': -0.95, 'Eh': -0.95,
                'Iapp': 1.5, 'As': 0, 'f': 50, 'An': 0.01}
ic_args = {'v': -0.82, 'r': 0.4, 'h': 0.04}


HH = makeHHneuron('HH_AHP', par_args, ic_args, const_I=True,
                  gentype=gentype, with_IA=True, adapt_AHP=True)
HH.set(tdata=[0,700])


I = 1.8

# Piecewise protocol
protocol = []

protocol.append({'pars': {'Iapp': 0},
                 'tdata': [0, 40]})

protocol.append({'pars': {'Iapp': I},
                 'tdata': [0, 180]})

protocol.append({'pars': {'Iapp': 0},
                 'tdata': [0, 150]})

traj, pts = pcw_protocol(HH, protocol)

plt.plot(pts['t'], pts['v'], 'k')
plt.plot(pts['t'], pts['r'], 'r')
plt.plot(pts['t'], pts['h'], 'g')
plt.xlim(-5,max(pts['t']))
plt.show()
