"""Ch. 10 Adaptation in spiking and bursting using I_AHP and I_T (Ch. 10.4)
"""
from __future__ import division
from PyDSTool import *
from common_lib import *


# --------------------------------------

def makeHHneuron(name, par_args, ic_args, const_I=False, apply_TTX=False,
                 rand_seed=None, gentype='vode'):
    if gentype in ['vode', 'euler']:
        targetlang = 'python'
    else:
        targetlang = 'C'
    if const_I:
        Iapp_str = 'Iapp'
    else:
        Iapp_str = 'Iapp(t)'
    Isignal_vardict = make_noise_signal(0.1, 6000, 0, 0.4, 1, rand_seed)
    vfn_str = '(%s-ionic(v, r, x, c)+As*sin(2*pi*t*f/1000.)+An*noise_sig)/tau_v' % Iapp_str
    rfn_str = '(1.29*v+0.79+3.3*pow(v+0.38,2)-r)/tau_r'
    xfn_str = '(7.33*(v+0.86)*(v+0.84)-x)/tau_x'
    cfn_str = '(3*x-c)/tau_c'

    if apply_TTX:
        v_str = 'vrest'
    else:
        v_str = 'vv'
    auxfndict = {'ionic': (['vv', 'rr', 'xx', 'cc'],
            '13*(1.37+3.67*%s+2.51*%s*%s)*(vv-Ena)+gr*rr*(vv-Er)+gx*xx*(vv-Ex)+gc*cc*(vv-Ec)' % (v_str,
                                                                                                 v_str, v_str))}
    if not const_I:
        auxfndict['Iapp'] = (['t'], 'if(t<thalf, kI*t-0.25, max(kI*(2*thalf-t)-0.25,-0.25))')

    DSargs = args()
    DSargs.varspecs = {'v': vfn_str, 'r': rfn_str, 'x': xfn_str, 'c': cfn_str,
                       'noise': 'An*noise_sig',
                       'sine': 'As*sin(2*pi*t*f/1000.)'}
    DSargs.inputs = {'noise_sig': Isignal_vardict['noise1']}
    DSargs.auxvars = ['noise', 'sine']
    if not const_I:
        DSargs.varspecs['I'] = 'Iapp(t)'
        DSargs.auxvars.append('I')
    DSargs.pars = par_args
    DSargs.fnspecs = auxfndict
    DSargs.xdomain = {'v': [-2, 2], 'r': [0,1], 'x': [0,1], 'c': [0,1]}
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

    min_ev = Events.makeZeroCrossEvent('(%s-ionic(v, r, x, c)+As*sin(2*pi*t*f/1000.)+An*noise_sig)/tau_v' % Iapp_str, 1,
                                       {'name': 'min_ev',
                                        'eventtol': 1e-5,
                                        'precise': True,
                                        'term': False},
                                       varnames=['v', 'r', 'x', 'c'],
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

