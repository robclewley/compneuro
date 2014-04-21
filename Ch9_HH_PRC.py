"""
Phase responses to perturbations shown in reduced, 2D version of Hodgkin-Huxley model
Section 9.6
"""
from __future__ import division
from PyDSTool import *
from PyDSTool.Toolbox.PRCtools import *
from PyDSTool.Toolbox.phaseplane import *

from common_lib import *
from Ch9_HH_red import *

gentype='dopri'  # dopri, euler, etc.

# Parameter An = noise amplitude
#           As = sine wave amplitude
#           f = frequency, should be >= 50 Hz
par_args = {'tau_v': 1, 'tau_r': 5.6,
            'As': 0, 'f': 50, 'An': 0.0, 'Iapp': 0.03}
ic_args = {'v':-0.8, 'r': 0.25}


def nullclines(HH, fignum=2):
    jac_fn = make_Jac(HH)
    fps = find_fixedpoints(HH, {'v': [-1, 1], 'r': [0, 1]},
                       jac=jac_fn, eps=1e-6)

    nv, nr = find_nullclines(HH, 'v', 'r',
                         {'v': [-1, 1], 'r': [0, 1]},
                         max_step=0.02, crop_tol_pc=0,
                         fps=fps, n=5, eps=1e-5,
                         jac=jac_fn)

    plt.figure(fignum)
    plt.clf()
    plt.plot(nv[:,0],nv[:,1], label='N\_v')
    plt.plot(nr[:,0],nr[:,1], label='N\_r')
    plt.legend(loc='lower right')



HH = makeHHneuron('HHred', par_args, ic_args, const_I=True, gentype=gentype)
HH.set(tdata=[0,100],
       algparams={'init_step': 0.02, 'max_step': 0.05, 'stiff': True})
##traj = HH.compute('test')
##pts = traj.sample()
##plt.plot(pts['t'], pts['v'], 'b--')
##plt.show()

pd_info = one_period_traj(HH, 'min_ev', 1e-4, 1e-5, 15,
                    verbose=False, initial_settle=10)
ref_traj = pd_info[0]
ref_pts = pd_info[1]
T = pd_info[2]
print "Period is T=", T
assert T < 25, "Tolerances are probably bad if T > 25"
##plt.plot(ref_pts['t'], ref_pts['v'], 'b')
##plt.plot(ref_pts['t'], ref_pts['r'], 'r')

# Calculate finite Phase Response Curve
dV = 0.0005
PRC = finitePRC(HH, ref_traj, 'thresh_ev', 'v', dV, verbose=False, skip=5,
                keep_trajs=False)

# adjust for the fact that thresh_ev happens later than the min_ev used to
# more accurately define a period
dPRC = 1-ref_traj.getEventTimes('thresh_ev')[0]/T
PRC['D_phase'] -= dPRC

plt.figure(1)
plt.plot(PRC['t'], PRC['D_phase']*100, 'k')
plt.plot(ref_pts['t'], ref_pts['v'], 'b')

plt.figure(2)
nullclines(HH, 2)
plt.plot(ref_pts['v'], ref_pts['r'], 'r')
plt.plot(ref_pts['v'][0], ref_pts['r'][0], 'ro')

plt.show()



