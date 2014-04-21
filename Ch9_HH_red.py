"""
Hodgkin-Huxley 2D reduction in Eqn. (9.7)
Includes options for Gaussian noise and sinusoidal inputs
"""
from __future__ import division
from PyDSTool import *
from common_lib import *

# --------------------------------------

def makeHHneuron(name, par_args, ic_args, const_I=False,
                 rand_seed=None, with_IA=False, gentype='vode',
                 specials=None):
    """specials is an optional argument that permits arbitrary
    addition of entries to DSargs
    """
    if gentype in ['vode', 'euler']:
        targetlang = 'python'
    else:
        targetlang = 'C'
    if const_I:
        Iapp_str = 'Iapp'
    else:
        Iapp_str = 'Iapp(t)'
    Isignal_vardict = make_noise_signal(0.05, 10000, 0, 0.4, 1, rand_seed)
    vfn_str = '(%s-ionic(v, r)+As*sin(2*pi*t*f/1000.)+An*noise_sig)/tau_v' % Iapp_str
    if with_IA:
        rfn_str = '(1.29*v+0.79+3.3*pow(v+0.38,2)-r)/tau_r'
    else:
        rfn_str = '(1.35*v+1.03-r)/tau_r'

    auxfndict = {'ionic': (['vv', 'rr'],
            '13*(1.37+3.67*vv+2.51*vv*vv)*(vv-0.55)+26*rr*(vv+0.92)')}
    if not const_I:
        auxfndict['Iapp'] = (['t'], 'if(t<thalf, kI*t-0.25, max(kI*(2*thalf-t)-0.25,-0.25))')

    DSargs = args()
    DSargs.varspecs = {'v': vfn_str, 'r': rfn_str,
                       'noise': 'An*noise_sig',
                       'sine': 'As*sin(2*pi*t*f/1000.)'}
    DSargs.inputs = {'noise_sig': Isignal_vardict['noise1']}
    DSargs.auxvars = ['noise', 'sine']
    DSargs.pars = par_args
    if not const_I:
        DSargs.varspecs['I'] = 'Iapp(t)'
        DSargs.auxvars.append('I')
    elif 'Iapp' not in par_args:
        DSargs.pars['Iapp'] = 0
    DSargs.fnspecs = auxfndict
    DSargs.xdomain = {'v': [-130, 70], 'r': [0,1]}
    DSargs.algparams = {'init_step':0.05,
                        'max_pts': 800000,
                        'maxevtpts': 5000}
    if gentype == 'vode':
        DSargs.algparams['stiff'] = True
    DSargs.checklevel = 0
    DSargs.ics = ic_args
    DSargs.name = name

    # Event definitions
    thresh_ev = Events.makeZeroCrossEvent('v', 1,
                                       {'name': 'thresh_ev',
                                        #'eventdelay': 1e-2,
                                        #'eventinterval': 1e-2,
                                        'eventtol': 1e-5,
                                        'precise': True,
                                        'term': False},
                                       varnames=['v'],
                                       targetlang=targetlang)

    min_ev = Events.makeZeroCrossEvent('(%s-ionic(v, r)+As*sin(2*pi*t*f/1000.)+An*noise_sig)/tau_v' % Iapp_str, 1,
                                       {'name': 'min_ev',
                                        #'eventdelay': 1e-2,
                                        #'eventinterval': 1e-2,
                                        'eventtol': 1e-5,
                                        'precise': True,
                                        'term': False},
                                       varnames=['v', 'r'],
                                       parnames=par_args.keys(),
                                       inputnames=['noise_sig'],
                                       fnspecs=auxfndict,
                                       targetlang=targetlang)

    DSargs.events = [thresh_ev, min_ev]
    if specials is not None:
        for k, v in specials.items():
            if k in DSargs:
                current_v = DSargs[k]
                if isinstance(current_v, (args, dict)):
                    DSargs[k].update(v)
                elif isinstance(current_v, list):
                    if isinstance(v, list):
                        DSargs[k].extend(v)
                    else:
                        DSargs[k].append(v)
                else:
                    raise ValueError("Unrecognized item type")
            else:
                DSargs[k] = v
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

