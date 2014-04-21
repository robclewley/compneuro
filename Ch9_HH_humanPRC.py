"""
Human cortical neuron using A-current model in reduced, 2D version of Hodgkin-Huxley model
Section 9.5
"""
from __future__ import division
from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *
from PyDSTool.Toolbox.PRCtools import *

from common_lib import *
from Ch9_HH_red import *

gentype='dopri'  # dopri, euler, etc.

# Parameter An = noise amplitude
#           As = sine wave amplitude
#           f = frequency, should be >= 50 Hz
par_args = {'tau_v': 1, 'tau_r': 5.6,
            'As': 0, 'f': 700, 'An': 0., 'Iapp': 0.8}
ic_args = {'v':-0.8, 'r': 0.25}

HH = makeHHneuron('HHred', par_args, ic_args, const_I=True, with_IA=True,
                  gentype=gentype)
HH.set(tdata=[0,400],
       algparams={'init_step': 0.02, 'max_step': 0.05, 'stiff': True})

pd_info = one_period_traj(HH, 'min_ev', 1e-4, 1e-5, 15,
                    verbose=False, initial_settle=10)
ref_traj = pd_info[0]
ref_pts = pd_info[1]
T = pd_info[2]
print "Period is T=", T

dV = 0.0005
PRC = finitePRC(HH, ref_traj, 'thresh_ev', 'v', dV, verbose=False, skip=5,
                keep_trajs=False)

# adjust for the fact that thresh_ev happens later than the min_ev used to
# more accurately define a period
dPRC = 1-ref_traj.getEventTimes('thresh_ev')[0]/T
PRC['D_phase'] -= dPRC

plt.figure(1)
plt.plot(PRC['t'], PRC['D_phase']*200, 'k')
plt.plot(ref_pts['t'], ref_pts['v'], 'b')

plt.show()

