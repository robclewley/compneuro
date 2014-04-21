"""
Eqn (10.5) for mammalian cell bursting
"""

from __future__ import division
from PyDSTool import *
from Ch10_4_model import *

gentype = 'vode' # 'dopri' # 'vode'

par_args = {'gr': 26, 'gc': 13, 'gx': 1.7, 'vrest': -0.754,
            'tau_r': 2.1, 'tau_c': 56, 'tau_x': 15, 'tau_v': 1,
                'Ena': 0.48, 'Er': -0.95, 'Ec': -0.95, 'Ex': 1.4,
                'Iapp': 1.5, 'As': 0, 'f': 50, 'An': 0.0}
ic_args = {'v': -0.82, 'r': 0.4, 'x': 0.04, 'c': 0}


HH = makeHHneuron('HH_mamm', par_args, ic_args, const_I=True,
                  apply_TTX=False,
                  gentype=gentype)
HH_TTX = makeHHneuron('HH_mamm', par_args, ic_args, const_I=True,
                  apply_TTX=True,
                  gentype=gentype)
HH.set(tdata=[0,500])
HH_TTX.set(tdata=[0,500])

def test_I(I):
    HH.set(pars={'Iapp': I})
    HH_TTX.set(pars={'Iapp': I})
    traj = HH.compute('test')
    pts = traj.sample()
    traj_TTX = HH_TTX.compute('test')
    pts_TTX = traj_TTX.sample()
    plt.figure(1)
    plt.clf()
    plt.plot(pts['t'], pts['v'], 'k')
    plt.plot(pts_TTX['t'], pts_TTX['v'], 'k--')

    plt.plot(pts['t'], pts['r'], 'r')
    plt.plot(pts['t'], pts['x'], 'g')
    plt.plot(pts['t'], pts['c'], 'y')
    plt.plot(pts_TTX['t'], pts_TTX['r'], 'r--')
    plt.plot(pts_TTX['t'], pts_TTX['x'], 'g--')
    plt.plot(pts_TTX['t'], pts_TTX['c'], 'y--')

    plt.xlim(-5,max(pts['t']))
    return traj, pts

traj, pts = test_I(1.6)

plt.show()
