"""
Chapter 12.3
Hodgkin-Huxley 2D reduction in Eqn. (9.7) with alpha synapse
showing post-inhibitory rebound.
"""
from __future__ import division
from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *

from common_lib import *

# --------------------------------------

def makeHHneuron(name, par_args, ic_args, with_IA=False,
                 gentype='vode'):
    """Includes alpha function synapse -- needs to be driven with a piecewise
    constant presynaptic voltage input (as a parameter)
    """

    if gentype in ['vode', 'euler']:
        targetlang = 'python'
    else:
        targetlang = 'C'
    vfn_str = '(Iapp-ionic(v, r)-gsyn*s*(v-Esyn))/tau_v'
    if with_IA:
        rfn_str = '(1.29*v+0.79+3.3*pow(v+0.38,2)-r)/tau_r'
    else:
        rfn_str = '(1.35*v+1.03-r)/tau_r'

    auxfndict = {'ionic': (['vv', 'rr'],
            '13*(1.37+3.67*vv+2.51*vv*vv)*(vv-0.55)+26*rr*(vv+0.92)')}

    DSargs = args()
    DSargs.varspecs = {'v': vfn_str, 'r': rfn_str,
                       'a': '(-a+heav(vpre-vthresh))/tau_syn',
                       's': '(-s+a)/tau_syn',
                       'vpre_aux': 'vpre'}
    DSargs.pars = par_args
    DSargs.fnspecs = auxfndict
    DSargs.auxvars = ['vpre_aux']
    DSargs.xdomain = {'v': [-1, 1], 'r': [0,1]}
    DSargs.tdomain = [0, 500]
    DSargs.algparams = {'init_step':0.1,
                        'max_pts': 100000}
    if gentype == 'vode':
        DSargs.algparams['stiff'] = True
    DSargs.checklevel = 1
    DSargs.ics = ic_args
    DSargs.name = name

    # Event definitions
    thresh_ev = Events.makeZeroCrossEvent('v', 1,
                                       {'name': 'thresh_ev',
                                        'eventtol': 1e-5,
                                        'precise': True,
                                        'term': False},
                                       varnames=['v'],
                                       targetlang=targetlang)


    DSargs.events = [thresh_ev]
    if gentype == 'vode':
        return Generator.Vode_ODEsystem(DSargs)
    elif gentype == 'euler':
        return Generator.Euler_ODEsystem(DSargs)
    elif gentype == 'radau':
        return Generator.Radau_ODEsystem(DSargs)
    elif gentype == 'dopri':
        return Generator.Dopri_ODEsystem(DSargs)
    else:
        raise NotImplementedError("Unknown ODE system type: %s"%gentype)


# efficiency hack to save time on repeated r-nullcline calculation!
# ----------------
nullc_r = None
PC = [None, None]
# ----------------

def nullclines(pts, traj, t_now, fignum=2, zoom=None):
    global nullc_r, PC

    pt_now = traj(t_now)
    s = pt_now['s']
    if zoom is None:
        zoom = ([-0.9,0.5], [0,0.5])

    fps = find_fixedpoints(HH, {'v': zoom[0], 'r': zoom[1],
                                's': s, 'a': 0}, # a doesn't matter here
                       jac=jac_fn, eps=1e-2)

    if nullc_r is None:
        only_var = None
    else:
        only_var = 'v'
    nullc_v, nr, PC = find_nullclines(HH, 'v', 'r',
                         {'v': zoom[0], 'r': zoom[1], 's': s, 'a': 0},
                         max_step=0.015, crop_tol_pc=0.01,
                         eps=1e-2, only_var=only_var,
                         pycont_cache=PC,
                         max_num_points=10,
                         jac=jac_fn)
    if nullc_r is None:
        nullc_r = nr

    fp = fixedpoint_2D(HH, Point(fps[0]), coords=['v', 'r'],
                       eps=1e-2, jac=jac_fn)


    plt.figure(fignum)
    plt.clf()
    plt.plot(nullc_v[:,0],nullc_v[:,1], 'y--', label='N\_v')
    plt.plot(nullc_r[:,0],nullc_r[:,1], 'm--', label='N\_r')
    plot_PP_fps(fp, markersize=8)
    plt.plot(pts['v'], pts['r'], 'k')
    plt.title('Phaseplane at t = %.5f ms' % t_now)
    plt.plot(pt_now['v'], pt_now['r'], 'ro', markersize=6, label='state now')
    plt.xlim(zoom[0])
    plt.ylim(zoom[1])
    plt.legend(loc='upper right')


def test(t_pre, gsyn, tau):
    """Inputs:
    t_pre: Pre-synaptic spike time
    gsyn:  synaptic strength (max conductance) for inhibition
    tau:   synaptic time constant (for alpha function)
    """
    HH.set(pars={'tau_syn': tau,
                  'gsyn': gsyn})

    t1 = t_pre
    s1 = args(pars={'vpre': -0.85},
           tdur=t1)
    t2 = 1
    s2 = args(pars={'vpre': 0.45},
           tdur=t2)
    t3 = 25
    s3 = args(pars={'vpre': -0.85},
           tdur=t3)

    traj, pts = pcw_protocol(HH, [s1,s2,s3])

    plt.figure(1)
    plt.clf()
    plt.plot(pts['t'], pts['vpre_aux'], 'k', linewidth=1, label='pre-syn v')
    plt.plot(pts['t'], pts['v'], 'b', linewidth=2, label='post-syn v')
    plt.plot(pts['t'], pts['s'], 'g--', linewidth=1, label='s')

    plt.xlabel('t')
    plt.legend(loc='upper right')
    plt.title('g = %.2f, tau syn = %.2f ms' % (gsyn, tau))
    plt.xlim([0, max(pts['t'])])
    plt.show()

    return traj, pts


def make_anim_frames(pts, traj, tlo, thi, n, zoom):
    for i, t in enumerate(linspace(tlo, thi, n)):
        nullclines(pts, traj, t, zoom=zoom)
        # ensure figure 2 is active
        plt.figure(2)
        plt.savefig('PIR_PP_%i.png' % (i+1))


## ----------------------------------------------

par_args = {'tau_v': 0.5, 'tau_r': 4, 'tau_syn': 5, 'gsyn': 0, 'Esyn': -0.9,
            'Iapp': 0.01, 'vpre': -0.8, 'vthresh': 0}

ic_args = {'v':-0.8, 'r': 0.25, 's': 0, 'a': 0}


HH = makeHHneuron('HHred_PIR', par_args, ic_args, gentype='dopri')
# Prepare the Jacobian for the nullclines functions (only in the v-r plane)
jac_fn = make_Jac(HH, ['v', 'r'])


## Non-PIR case (very weak inhibition)
traj, pts = test(1, 1, 2)
nullclines(pts, traj, 5)

## PIR case (strong inhibition)
#traj, pts = test(1, 10, 2)
#nullclines(pts, traj, 5)

### Make 50 animation frames from t = 2 ... 9 ms
#make_anim_frames(pts, traj, 2, 9, 50, ([-0.8, -0.5], [0, 0.15]))
