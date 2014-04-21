"""
Traveling waves in coupled phase oscillators
Eqs. 13.4 of Ch. 13.2
"""

from __future__ import division
import time
from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *
from common_lib import *


def make_chain(N, icdict, pardict):
    """
    N is the number of segments in chain
    """
    N = int(N)
    assert N >= 2
    DSargs = args()
    DSargs.name = 'lamprey_%i'%N
    DSargs.ics = icdict
    DSargs.pars = pardict
    DSargs.tdata = [0, 100]
    DSargs.algparams = {'init_step': 3e-3}
    specs = {}
    # vars
    base = 'wx + ax*H(thy-thx)'
    specs['th1'] = 'w1 + a_asc*H(th2-th1)'
    if N > 2:
        specs['th[i]'] = 'for(i,2,%i, w[i] + a_asc*H(th[i+1]-th[i]) + a_des*H(th[i-1]-th[i]))' %(N-1)
    specs['th%i' %N] = 'w%i + a_des*H(th%i-th%i)' %(N, N-1, N)
    # auxvars
    specs['th_out[i]'] = 'for(i,1,%i, mod(th[i],2*pi))' %N
    specs['phi[i]'] = 'for(i,1,%i, th[i+1]-th[i])' %(N-1)
    DSargs.auxvars = ['th_out%i'%i for i in range(1, N+1)] + \
                     ['phi%i'%i for i in range(1, N)]
    DSargs.varspecs = specs
    DSargs.fnspecs  = {'H': (['p'], 'sin(p+sigma)')}
    return Generator.Dopri_ODEsystem(DSargs)

def make_chain_old(N, icdict, pardict):
    """
    N is the number of segments in chain
    """
    N = int(N)
    assert N >= 2
    DSargs = args()
    DSargs.name = 'lamprey_%i'%N
    DSargs.ics = icdict
    DSargs.pars = pardict
    DSargs.tdata = [0, 100]
    DSargs.algparams = {'init_step': 3e-3}
    specs = {}
    # vars
    base = 'wx + ax*H(thy-thx)'
    specs['th1'] = 'w1 + a_asc*H(th2-th1)'
    if N > 2:
        for i in range(2, N):
            specs['th%i' %i] = 'w%i + a_asc*H(th%i-th%i) + a_des*H(th%i-th%i)' %(i, i+1, i, i-1, i)
    specs['th%i' %N] = 'w%i + a_des*H(th%i-th%i)' %(N, N-1, N)
    # auxvars
    for i in range(1, N+1):
        specs['th_out%i' %i] = 'mod(th%i,2*pi)' %i
        if i < N:
            specs['phi%i' %i] = 'th%i-th%i' %(i+1, i)
    DSargs.auxvars = ['th_out%i'%i for i in range(1, N+1)] + \
                     ['phi%i'%i for i in range(1, N)]
    DSargs.varspecs = specs
    DSargs.fnspecs  = {'H': (['p'], 'sin(p+sigma)')}
    return Generator.Dopri_ODEsystem(DSargs)


def animate(traj, tlo=0, thi=None, dt=0, ymax=5, max_angle=pi/4,
            ic_only=False):
    if dt == 0:
        pts = traj.sample(tlo=tlo, thi=thi)
        ts = pts['t']
    else:
        pts = traj.sample(tlo=tlo, thi=thi, dt=dt, precise=True)
        ts = pts['t']
    plt.figure(0)
    plt.clf()
    plt.axis([-2, N+1, -ymax, ymax])
    ax = plt.gca()
    ax.set_autoscale_on(False)
    ax.set_aspect('equal')

    plt.xlim([-2, N+1])
    plt.ylim([-ymax, ymax])
    # head at (0,0), each segment length 1
    plt.plot([0],[0], 'ko', markersize=15)
    seg = {}
    objs = {}
    for i in range(1, N):
        objs[i] = plt.plot([i-1,i], [0, 0], 'k', linewidth=4)[0]

    ts = pts['t']
    var = 'th'
    # make the physical segment angle vary between +/- max_angle based on the
    # oscillator's "angle" theta linear progression at a stable constant
    # angular velocity determined self-consistently by the coupling
    for j, pt in enumerate(pts):
        t = ts[j]
        for i in range(1, N):
            if i == 1:
                seg[i] = ([i-1, i-1+cos(max_angle*sin(pt[var+'%i'%i]))],
                          [0, sin(max_angle*sin(pt[var+'%i'%i]))])
            else:
                prev_seg = seg[i-1]
                prev_x_end = prev_seg[0][1]
                prev_y_end = prev_seg[1][1]
                seg[i] = ([prev_x_end,
                           prev_x_end+cos(max_angle*sin(pt[var+'%i'%i]))],
                          [prev_y_end,
                           prev_y_end+sin(max_angle*sin(pt[var+'%i'%i]))])
            objs[i].set_data( seg[i] )
        plt.title('t = %.3f' %t)
        plt.draw()
        if ic_only:
            # Just show initial state only -- stop now!
            break


def test(**kw):
    icdict = filteredDict(kw, DS.funcspec.vars)
    pardict = filteredDict(kw, DS.pars.keys())
    DS.set(ics=icdict, pars=pardict)
    traj = DS.compute('test')
    pts = traj.sample()
    return traj, pts

def plot_phases(pts):
    plt.figure(1)
    plt.clf()
    for i in range(1, N+1):
        plt.plot(pts['t'],pts['th_out%i' %i], label='theta %i' %i)
    if N < 10:
        plt.legend()
    plt.xlabel('t')
    plt.ylabel('angle')
    plt.ylim([-6.4, 6.4])

    plt.figure(2)
    plt.clf()
    for i in range(1, N):
        plt.plot(pts['t'],pts['phi%i' %i], linewidth=2, label='phi %i' %i)
    if N < 10:
        plt.legend()
    plt.xlabel('t')
    plt.ylabel('angle')
    plt.title('Phase differences')
    plt.show()

def rand_vals(N, base, base_val, sig):
    d = {}
    for i in range(1, N+1):
        varname = base + str(i)
        d[varname] = float(base_val + sig*random.random(1))
    return d

#-------------------------------------------

# N is a global = number of segmental oscillators
N = 30
wavenum = 2
tmax = 200

ic_vals = zeros(N) #1*sin(wavenum*linspace(0, 2*pi, N)-pi/2)
icdict = dict(zip(['th%i'%i for i in range(1,N+1)], ic_vals[:N]))

# These ICs taken from a wavenum=2 solution
##icdict = {'th1': 4.98313573676079,
## 'th10': 1.2132245524531733,
## 'th11': 0.7943455319745496,
## 'th12': 0.37546651149534682,
## 'th13': 6.2397727981973894,
## 'th14': 5.8208937777172203,
## 'th15': 5.4020147572404973,
## 'th16': 4.9831357367605484,
## 'th17': 4.5642567162808234,
## 'th18': 4.145377695811284,
## 'th19': 3.7264986753095251,
## 'th2': 4.5642567162822054,
## 'th20': 3.3076196548876062,
## 'th21': 2.8887406343050372,
## 'th22': 2.4698616140389831,
## 'th23': 2.0509825932279533,
## 'th24': 1.6321035733825191,
## 'th25': 1.2132245520475671,
## 'th26': 0.79434553318312595,
## 'th27': 0.37546651088396743,
## 'th28': 6.2397728011611662,
## 'th29': 5.8208937776409329,
## 'th3': 4.1453776958034112,
## 'th30':  5.40201476254,
## 'th4': 3.7264986753250007,
## 'th5': 3.3076196548461709,
## 'th6': 2.8887406343676005,
## 'th7': 2.4698616138889271,
## 'th8': 2.0509825934104526,
## 'th9': 1.6321035729314843}


def make_trav_wave_pars(w1=1, wavenum=1, a_asc=.1, a_des=1):
    """Make parameter set for a travelling wave, following the
    analysis on p. 213 to establish a traveling wave
    """
    phi = -2*wavenum*pi/N
    sig = -phi
    stability = sig < pi/4
    print "Sigma = %.3f < pi/4 = %.4f ? %s" % (sig, pi/4, str(stability))
    if not stability:
        print " *** Traveling wave will not be stable!"
    pardict = {'w1': w1}
    for i in range(2, N):
        pardict['w%i'%i] = pardict['w1'] - a_des*sin(sig - phi)
    pardict['w%i'%N] = pardict['w1'] - a_des*sin(sig - phi)
    print "delta w1 = %.4f" % (- a_des*sin(sig - phi))
    print "w1 = %.3f, w2 = %.3f" % (w1, pardict['w2'])
    pardict.update({'a_asc': a_asc, 'a_des': a_des,
               'sigma': sig})
    return pardict

def make_linear_freq_pars(w1=0.1, dw=0.1):
    """Make parameter set for a linear gradient of forcing frequencies
    """
    w_pairs = [('w%i'%(i+1), w1+i*dw) for i in range(0,N)]
    return dict(w_pairs)

DS = make_chain(N, icdict, make_trav_wave_pars(1.2, wavenum, 1.25, .5))
DS.set(tdata=[0,tmax])
plt.figure(0)
plt.show()

#DS.set(pars=make_linear_freq_pars(0.1, 0.1))
#DS.set(pars={'a_des': 0.5, 'a_asc': 1})

traj, pts = test()
animate(traj, 0, tmax, 0.5, ymax=N/3, max_angle=pi/6) #, ic_only=True)

plot_phases(pts)
