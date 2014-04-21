"""
Phase responses to perturbations shown in reduced, 2D version of Hodgkin-Huxley model
Section 9.6
"""
from __future__ import division
from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *
from PyDSTool.Toolbox.PRCtools import *
from common_lib import *
from Ch9_HH_red import *

gentype= 'vode' #'dopri'  # dopri, euler, etc.

# Parameter An = noise amplitude
#           As = sine wave amplitude
#           f = frequency, should be >= 50 Hz
par_args = {'tau_v': 1, 'tau_r': 5.6,
            'As': 0, 'f': 50, 'An': 0.0,
            'Iapp': 1.2}
ic_args = {'v':-0.8, 'r': 0.25}


HH = makeHHneuron('HHred', par_args, ic_args, const_I=True, gentype=gentype)
HH.set(tdata=[0,700])
##
##pd_info = one_period_traj(HH, 'min_ev', 1e-4, 1e-5, 30,
##                    verbose=False, initial_settle=10)
##ref_traj = pd_info[0]
##ref_pts = pd_info[1]
##T = pd_info[2]
##
##PRC = finitePRC(HH, pd_info[0], 'thresh_ev', 'v', 1, verbose=False, skip=5,
##                keep_trajs=True)

#traj = HH.compute('test')
#pts = traj.sample()
#plt.plot(pts['t'], pts['v'], 'b')
#plt.show()

# Have to make a version of the system with a constant Iapp to do
# Jacobian etc.
par_args['Iapp'] = 0.
HH_noI = makeHHneuron('HHred', par_args, ic_args, const_I=True, gentype=gentype)
HH_noI.set(tdata=[0,300])
jac_fn = make_Jac(HH_noI)

def nullclines(I, fignum=2, only_v=False):
    HH_noI.set(pars={'Iapp': I})
    fps = find_fixedpoints(HH_noI, {'v': [-1, 1], 'r': [0, 1]},
                       jac=jac_fn, eps=1e-3)

    if only_v:
        only_var = 'v'
    else:
        only_var = None
    nv, nr = find_nullclines(HH_noI, 'v', 'r',
                         {'v': [-1, 1], 'r': [0, 1]},
                         max_step=0.02, crop_tol_pc=0,
                         fps=fps, n=5, eps=1e-3,
                         jac=jac_fn, only_var=only_var)

    fp1 = fixedpoint_2D(HH_noI, Point(fps[0]), coords=['v', 'r'],
                       eps=1e-6, jac=jac_fn)


    plt.figure(fignum)
    plt.clf()
    plt.plot(nv[:,0],nv[:,1], label='N\_v')
    plt.plot(nr[:,0],nr[:,1], label='N\_r')
    plt.draw()
    plt.legend(loc='lower right')

def do_plots(I, ppfig, n=None):
    HH_noI.set(pars={'Iapp': I})
    traj = HH_noI.compute('test')
    pts = traj.sample()
    plt.figure(ppfig)
    plt.plot(pts['v'], pts['r'], 'k')
    plt.plot(pts['v'][0], pts['r'][0], 'go')
    plt.draw()
    if n is not None:
        plt.savefig('Ch9_hyst_PP_%i.png' % n)

    plt.figure(0)
    plt.plot(pts['t'], pts['v'])
    return pts[-1]


def test_hyst(n=40, Ilo=-0.7, Ihi=0.3):
    # Piecewise protocol
    protocol = []
    Is = []
    for I in linspace(Ilo, Ihi, n):
        Is.append(I)
        protocol.append({'pars': {'Iapp': I},
                 'tdur': 20})

    for I in linspace(Ihi, Ilo, n):
        Is.append(I)
        protocol.append({'pars': {'Iapp': I},
                 'tdur': 20})

    traj, pts = pcw_protocol(HH_noI, protocol)
    return Is, traj, pts

def test_hyst_PPs(n=12, Ilo=-0.5, Ihi=0.3):
    done_r = False
    i = 1
    for I in linspace(Ilo, Ihi, n):
        if not done_r:
            # only calculate the static r nullcline once
            nullclines(I, 3)
        else:
            nullclines(I, 3, only_v=True)
            done_r = True
        newic = do_plots(I, 3, i)
        #HH_noI.set(ics=newic)
        i += 1

# This will dump out phase plane pics
#test_hyst_PPs()

# hysteresis region (see p. 144 of book)
#HH_noI.set(pars={'Iapp': 0.03})

# outside unstable LC
#HH_noI.set(ics={'v': -0.7, 'r': 0.1})

# inside unstable LC
#HH_noI.set(ics={'v': -0.7, 'r': 0.095})

#1/0

Is, traj, pts = test_hyst()
n = len(Is)
t_Is = linspace(0, n*20, n)

plt.plot(pts['t'],pts['v'])
plt.plot(t_Is, Is, 'ko')

plt.show()
