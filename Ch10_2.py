"""
Eqn (10.2) for bursting model with hysteresis, using I_AHP
"""

from __future__ import division
from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *
from Ch10_model import *


gentype = 'radau' # 'vode'

par_args = {'gr': 26, 'gh': 0.54, 'tau_r': 1.9, 'tau_h': 250, 'tau_v': 0.8,
                'Ena': 0.55, 'Er': -0.92, 'Eh': -0.92,
                'Iapp': 0.14, 'As': 0, 'f': 50, 'An': 0.0}
ic_args = {'v': -0.85, 'r': 0.2, 'h': 0.00}


HH = makeHHneuron('HH_AHP_burst', par_args, ic_args, const_I=True,
                  gentype=gentype, with_IA=False, adapt_AHP=False)
HH.set(tdata=[0,3000])
# temporarily disable minimum V event (speeds up long computations)
HH.eventstruct.setActiveFlag('min_ev', False)

def test_I(I):
    HH.set(pars={'Iapp': I})
    traj = HH.compute('test')
    pts = traj.sample()
    plt.figure(1)
    plt.clf()
    plt.plot(pts['t'], pts['v'], 'k')
    plt.plot(pts['t'], pts['r'], 'r')
    plt.plot(pts['t'], pts['h'], 'g')
    plt.xlim(-5,max(pts['t']))
    return traj, pts

traj, pts = test_I(0.14)

plt.show()


# phase plane projection (slow h assumed constant)
ix0 = pts.find(122,1)
ix1 = pts.find(250,1)
sub_pts = pts[ix0:ix1]
plt.figure(2)



jac_fn = make_Jac(HH, ['v', 'r'])

def nullclines(h, pts, fignum=2):
    fps = find_fixedpoints(HH, {'v': [-1, 1], 'r': [0, 1], 'h': h},
                       jac=jac_fn, eps=1e-6)

    nv, nr = find_nullclines(HH, 'v', 'r',
                         {'v': [-1, 1], 'r': [0, 1], 'h': h},
                         max_step=0.02, crop_tol_pc=0,
                         fps=fps, n=5, eps=1e-5,
                         jac=jac_fn)

    fp1 = fixedpoint_2D(HH, Point(fps[0]), coords=['v', 'r'],
                       eps=1e-6, jac=jac_fn)


    plt.figure(fignum)
    plt.clf()
    plt.plot(nv[:,0],nv[:,1], 'b', label='N\_v')
    plt.plot(nr[:,0],nr[:,1], 'r', label='N\_r')
    plt.plot(pts['v'], pts['r'], 'k')
    plt.legend(loc='lower right')
    return fp1

fp = nullclines(0.25, sub_pts, 2)
