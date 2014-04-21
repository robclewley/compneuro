"""
Simple coupled phase oscillators
Eqs. 12.2 of Ch. 12.1
"""

from __future__ import division
import time
from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *
from common_lib import *


def make_model(icdict, pardict):
    DSargs = args()
    DSargs.name = 'phase'
    DSargs.ics = icdict
    DSargs.pars = pardict
    DSargs.tdata = [0, 60]
    DSargs.algparams = {'init_step': 3e-3}
    DSargs.varspecs = {'p1': 'f1(p1,p2)',
                       'p2': 'f2(p1,p2)',
                       'phi': 'p2-p1',
                       'p1out': 'mod(p1,2*pi)',
                       'p2out': 'mod(p2,2*pi)',
                       'dphi': 'abs(f1(p1,p2)-f2(p1,p2))'}
    DSargs.auxvars = ['phi', 'p1out', 'p2out', 'dphi']
    DSargs.fnspecs  = {'H': (['p'], 'sin(p+sigma)'),
                       'f1': (['p1', 'p2'], 'w1 + a1*H(p2-p1)'),
                       'f2': (['p1', 'p2'], 'w2 + a2*H(p1-p2)')
                       }
    stop_ev = Events.makeZeroCrossEvent('abs(f1(p1,p2)-f2(p1,p2))-dphi_thresh', -1,
                                           {'name': 'stop_ev',
                                            'eventtol': 1e-2,
                                            'precise': True,
                                            'term': True},
                                           varnames=['p1', 'p2'],
                                           fnspecs=DSargs.fnspecs,
                                           parnames=pardict.keys(),
                                           targetlang='python')
    DSargs.events = [stop_ev]
    return Generator.Vode_ODEsystem(DSargs)


def animate(traj, tlo=0, thi=None, dt=0):
    if dt == 0:
        pts = traj.sample(tlo=tlo, thi=thi)
        ts = pts['t']
    else:
        pts = traj.sample(tlo=tlo, thi=thi, dt=dt, precise=True)
        ts = pts['t']
    plt.figure(0)
    plt.clf()
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    ax = plt.gca()
    ax.set_autoscale_on(False)
    ax.set_aspect('equal')
    # background circle
    p = linspace(0, 2*pi)
    plt.plot(cos(p), sin(p), 'gray')
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    theta1 = plt.plot([0,0], 'ro', label='theta 1')[0]
    theta2 = plt.plot([0,0], 'ko', label='theta 2')[0]
    plt.legend()
    ts = pts['t']
    for i, pt in enumerate(pts):
        #t = ts[i]
        theta1.set_data((cos(pt['p1']),sin(pt['p1'])))
        theta2.set_data((cos(pt['p2']),sin(pt['p2'])))
        #plt.title('t = %.3f' %t)
        plt.draw()

def test(**kw):
    icdict = filteredDict(kw, DS.funcspec.vars)
    pardict = filteredDict(kw, DS.pars.keys())
    DS.set(ics=icdict, pars=pardict)
    traj = DS.compute('test')
    pts = traj.sample()
    plt.figure(1)
    plt.clf()
    plt.plot(pts['t'],pts['p1out'], 'r', label='theta 1')
    plt.plot(pts['t'],pts['p2out'], 'k', label='theta 2')
    plt.legend()
    plt.ylim([-.1, 6.4])

    plt.figure(2)
    plt.clf()
    plt.plot(pts['t'],pts['phi'], 'b', linewidth=2, label='phi')
    plt.show()
    return traj, pts

def stability(pts):
    """See condition Eq. 12.10"""
    if pts['dphi'][-1] > DS.pars['dphi_thresh']:
        print "No steady state"
        return
    phi = pts['phi'][-1]
    print "Final, steady state phi =", pts['phi'][-1]
    a1 = DS.pars['a1']
    a2 = DS.pars['a2']
    sigma = DS.pars['sigma']
    cond = -(a2-a1)*sin(sigma)*sin(phi) - (a1+a2)*cos(sigma)*cos(phi)
    if cond < 0:
        result = '(stable)'
    else:
        result = '(unstable)'
    print 'd/d phi (D2(-phi) - H1(phi)) = %.3f' % cond, result


#-------------------------------------------
icdict = {'p1': 0, 'p2': 0.8}
pardict = {'w1': 6.2, 'w2': 5.8,
           'a1': -0.2, 'a2': -0.2,
           'sigma': 0.9,
           # dphi_thresh controls when to stop if reaching fixed point
           'dphi_thresh': 0.001}

DS = make_model(icdict, pardict)

traj, pts = test(p2=0., a1=-0.3)

stability(pts)