"""Ch. 10 Adaptation in spiking and bursting using I_AHP
"""
from __future__ import division
from PyDSTool import *
from common_lib import *


# --------------------------------------

def makeHHneuron(name, par_args, ic_args, const_I=False,
                 rand_seed=None, with_IA=False, adapt_AHP=False, gentype='vode'):
    if gentype in ['vode', 'euler']:
        targetlang = 'python'
    else:
        targetlang = 'C'
    if const_I:
        Iapp_str = 'Iapp'
    else:
        Iapp_str = 'Iapp(t)'
    Isignal_vardict = make_noise_signal(0.1, 6000, 0, 0.4, 1, rand_seed)
    vfn_str = '(%s-ionic(v, r, h)+As*sin(2*pi*t*f/1000.)+An*noise_sig)/tau_v' % Iapp_str
    if with_IA:
        rfn_str = '(1.29*v+0.79+3.3*pow(v+0.38,2)-r)/tau_r'
    else:
        rfn_str = '(1.35*v+1.03-r)/tau_r'
    if adapt_AHP:
        hfn_str = '(11*(v+0.754)*(v+0.69)-h)/tau_h'
    else:
        # burst type
        hfn_str = '(9.3*(v+0.7)-h)/tau_h'

    auxfndict = {'ionic': (['vv', 'rr', 'hh'],
            '13*(1.37+3.67*vv+2.51*vv*vv)*(vv-Ena)+gr*rr*(vv-Er)+gh*hh*(vv-Eh)')}
    if not const_I:
        auxfndict['Iapp'] = (['t'], 'if(t<thalf, kI*t-0.25, max(kI*(2*thalf-t)-0.25,-0.25))')

    DSargs = args()
    DSargs.varspecs = {'v': vfn_str, 'r': rfn_str, 'h': hfn_str,
                       'noise': 'An*noise_sig',
                       'sine': 'As*sin(2*pi*t*f/1000.)'}
    DSargs.inputs = {'noise_sig': Isignal_vardict['noise1']}
    DSargs.auxvars = ['noise', 'sine']
    if not const_I:
        DSargs.varspecs['I'] = 'Iapp(t)'
        DSargs.auxvars.append('I')
    DSargs.pars = par_args
    DSargs.fnspecs = auxfndict
    DSargs.xdomain = {'v': [-2, 2], 'r': [0,1], 'h': [0,1]}
    DSargs.algparams = {'init_step':0.1,
                        'max_pts': 1000000}
    if gentype == 'vode':
        DSargs.algparams['stiff'] = True
    DSargs.checklevel = 0
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

    min_ev = Events.makeZeroCrossEvent('(%s-ionic(v, r, h)+As*sin(2*pi*t*f/1000.)+An*noise_sig)/tau_v' % Iapp_str, 1,
                                       {'name': 'min_ev',
                                        'eventtol': 1e-5,
                                        'precise': True,
                                        'term': False},
                                       varnames=['v', 'r', 'h'],
                                       parnames=par_args.keys(),
                                       inputnames=['noise_sig'],
                                       fnspecs=auxfndict,
                                       targetlang=targetlang)

    DSargs.events = [thresh_ev, min_ev]
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

