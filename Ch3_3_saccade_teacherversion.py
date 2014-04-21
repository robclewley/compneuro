"""
Eye saccade, Eq. 3.29 of Ch. 3.3
"""

from __future__ import division

from PyDSTool import *
from common_lib import *
from PyDSTool.Toolbox.phaseplane import *

def make_critical_model():
    icdict = {'x': 35, 'y': 0}
    # d = 0 => critically damped
    pardict = {'a': 0.1, 'x0': 32,
               'S': 0, 'd': 0}

    DSargs = args()
    DSargs.name = 'saccade_critical'
    DSargs.ics = icdict
    DSargs.pars = pardict
    DSargs.tdata = [0, 50]
    DSargs.varspecs = {'x': 'y',
                       'y': 'S -(2*a+d)*y + a*a*(x0-x)'}
    DSargs.fnspecs = {'Jacobian': (['t', 'x', 'y'],
                               """[[0, 1],
                                   [-a*a, -(2*a+d)]]
                               """)}
    return Generator.Vode_ODEsystem(DSargs)

def make_bangbang_model(d):
    """Use this for pulse-step input. This results in control
    that is only similar to "bang bang control", not a direct
    implementation of it.
    """
    icdict = {'x': 35, 'y': 0}
    # d < 0 => under-damped
    # d > 0 => over-damped
    # d = +/- 0.025 is a good choice
    pardict = {'a': 0.1, 'x0': 35,
               'S': 0, 'd': d}

    DSargs = args()
    DSargs.name = 'saccade_bangbang'
    DSargs.ics = icdict
    DSargs.pars = pardict
    DSargs.tdata = [0, 50]
    DSargs.varspecs = {'x': 'y',
                       'y': 'S -(2*a+d)*y + a*a*(x0-x)'}
    DSargs.fnspecs = {'Jacobian': (['t', 'x', 'y'],
                               """[[0, 1],
                                   [-a*a, -(2*a+d)]]
                               """)}
    return Generator.Vode_ODEsystem(DSargs)


def do_plots(pts, style='k-', label=None, dot=None):
    """Convenient plotter for this problem. Optional line style,
    labels (default black line). Optional dot (x,pt) pair
    to mark a transition in a piecewise trajectory, where Point pt
    has fields 'x' and 'y'.
    """
    # figure for x vs t
    plt.figure(1)
    if label is None:
        plt.plot(pts['t'], pts['x'], style)
    else:
        plt.plot(pts['t'], pts['x'], style, label=label)
    if dot is not None:
        plt.plot(dot[0], dot[1]['x'], style[0]+'o')
    plt.xlabel('t (ms)')
    plt.ylabel('x (mm)')
    # figure for y vs t
    plt.figure(2)
    if label is None:
        plt.plot(pts['t'], pts['y'], style)
    else:
        plt.plot(pts['t'], pts['y'], style, label=label)
    if dot is not None:
        plt.plot(dot[0], dot[1]['y'], style[0]+'o')
    plt.xlabel('t (ms)')
    plt.ylabel('dx/dt (mm/ms)')

#####

DSc = make_critical_model()
DSb1 = make_bangbang_model(0.025)
DSb2 = make_bangbang_model(-0.025)

def compare(xtarget, with_legend=False):
    """Finish this function (piecewise protocols for DSb1 and
    DSb2) before using it!
    """
    if with_legend:
        labelc = 'crit'
        labelb1 = 'over'
        labelb2 = 'under'
    else:
        labelc = labelb1 = labelb2 = None

    print "Comparing models for xtarget = ", xtarget

    # protocol for critically damped model
    t1 = 5
    s1c = args(pars={'S': 0, 'x0': 35},
           tdur=t1)
    t2 = 100
    s2c = args(pars={'S': 0, 'x0': xtarget},
           tdur=t2)

    trajc, ptsc = pcw_protocol(DSc, [s1c,s2c])
    do_plots(ptsc, 'k-', label=labelc)
    print "KE (crit) = %.3f" % kinetic_energy(trajc)

    # protocol for bang-bang control model (more balistic)
    ### over-damped version

    ## Substitute your value and formula to replace the NaNs here:
    t2 = 1 #0.1*(35-xtarget)
    amp3 = -0.9 #0.01*(xtarget-35)
    ## ----------------------------------------

    t1 = 5
    s1b1 = args(pars={'S': 0},
           tdur=t1)
    s2b1 = args(pars={'S': -1},
           tdur=t2)
    t3 = 100
    s3b1 = args(pars={'S': amp3},
           tdur=t3)

    trajb1, ptsb1 = pcw_protocol(DSb1, [s1b1,s2b1,s3b1])
    do_plots(ptsb1, 'k--', label=labelb1, dot=(t1+t2,trajb1(t1+t2)))
    print "KE (over) = %.3f" % kinetic_energy(trajb1)

    ### under-damped version
    t1 = 5
    s1b2 = args(pars={'S': 0},
           tdur=t1)
    s2b2 = args(pars={'S': -1},
           tdur=t2)
    t3 = 100
    s3b2 = args(pars={'S': amp3},
           tdur=t3)

    trajb2, ptsb2 = pcw_protocol(DSb2, [s1b2,s2b2,s3b2])
    do_plots(ptsb2, 'k:', label=labelb2, dot=(t1+t2,trajb2(t1+t2)))
    print "KE (under) = %.3f" % kinetic_energy(trajb2)

    if with_legend:
        plt.figure(1)
        plt.legend()
        plt.figure(2)
        plt.legend()


def kinetic_energy(traj, tend=80):
    """Estimate kinetic energy total over trajectory from 5ms
    until tend (default 80ms)"""
    # Resample trajectory at regular intervals
    dt = 0.2
    dxdt = traj.sample(tlo=5, thi=tend, dt=dt, precise=True)['y']
    # Use simple Riemann sum to estimate kinetic energy cost
    KE = sum(dxdt**2)*dt
    return KE


def testDSc(DS, xtarget, col='k'):
    """Only call this with the critically-damped DSc model
    as argument. Defaults to black line plot.
    """
    # protocol for critically damped model
    t1 = 5
    s1c = args(pars={'S': 0, 'x0': 35},
           tdur=t1)
    t2 = 100
    s2c = args(pars={'S': 0, 'x0': xtarget},
           tdur=t2)

    trajc, ptsc = pcw_protocol(DSc, [s1c,s2c])
    do_plots(ptsc, col)


def testDSb(DS, t2, amp3, col='k'):
    """Only call this with the balistic DSb models as
    arguments. Defaults to black line plot.
    """
    t1 = 5
    s1b1 = args(pars={'S': 0},
           tdur=t1)
    # t2 is an argument
    s2b1 = args(pars={'S': -1},
           tdur=t2)
    t3 = 100
    s3b1 = args(pars={'S': amp3},
           tdur=t3)

    trajb, ptsb = pcw_protocol(DS, [s1b1,s2b1,s3b1])
    do_plots(ptsb, col, dot=(t1+t2, trajb(t1+t2)))

# try finding xtargets = 26, 29, 32 with DSb1 and DSb2
# and use DSc response as reference
#testDSc(DSc, 26, 'k')
#testDSb(DSb1, 1, -0.5, 'r')

## Run these when compare function is finished
#compare(26)
#compare(29)
#compare(32, with_legend=True)
##
1/0

# Check eigenvalues
DSb1.set(pars={'S':-1})
fp_coords1 = find_fixedpoints(DSb1, n=4, eps=1e-6,
                             subdomain={'x': [-80,0],
                                        'y': [-20,2]})

fpb1 = fixedpoint_2D(DSb1, Point(fp_coords1[0]), coords=['x', 'y'],
                       eps=1e-6)
print fpb1.point, '\n', fpb1.stability, fpb1.classification
print fpb1.evals, fpb1.evecs

DSb2.set(pars={'S':-1})
fp_coords2 = find_fixedpoints(DSb2, n=4, eps=1e-6,
                             subdomain={'x': [-80,0],
                                        'y': [-20,2]})

fpb2 = fixedpoint_2D(DSb2, Point(fp_coords2[0]), coords=['x', 'y'],
                       eps=1e-6)
print fpb2.point, '\n', fpb2.stability, fpb2.classification
print fpb2.evals, fpb2.evecs

plt.show()