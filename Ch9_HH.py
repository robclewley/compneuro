"""
"""
from __future__ import division
from PyDSTool import *

gentype = 'dopri' # 'vode'

# --------------------------------------

def makeHHneuron(name, par_args, ic_args,
                 gentype='vode'):
    # extra_terms must not introduce new variables!
    vfn_str = '(Iapp-ionic(v,m,h,n))/C'
    mfn_str = 'ma(v)*(1-m)-mb(v)*m'
    nfn_str = 'na(v)*(1-n)-nb(v)*n'
    hfn_str = 'ha(v)*(1-h)-hb(v)*h'

    auxfndict = {'ionic': (['vv', 'mm', 'hh', 'nn'],
            'gna*mm*mm*mm*hh*(vv-vna) + gk*nn*nn*nn*nn*(vv-vk) + gl*(vv-vl)'),
               'ma': (['v'], '0.32*(v+54)/(1-exp(-(v+54)/4))'),
               'mb': (['v'], '0.28*(v+27)/(exp((v+27)/5)-1)'),
               'ha': (['v'], '.128*exp(-(50+v)/18)'),
               'hb': (['v'], '4/(1+exp(-(v+27)/5))'),
               'na': (['v'], '.032*(v+52)/(1-exp(-(v+52)/5))'),
               'nb': (['v'], '.5*exp(-(57+v)/40)')}

    DSargs = args()
    DSargs.varspecs = {'v': vfn_str, 'm': mfn_str,
                       'h': hfn_str, 'n': nfn_str}
    DSargs.pars = par_args
    DSargs.fnspecs = auxfndict
    DSargs.xdomain = {'v': [-130, 70], 'm': [0,1], 'h': [0,1], 'n': [0,1]}
    DSargs.algparams = {'init_step':0.03,
                        'max_pts': 100000}
    DSargs.checklevel = 0
    DSargs.ics = ic_args
    DSargs.name = name

    if gentype in ['vode', 'euler']:
        targetlang = 'python'
    else:
        targetlang = 'C'

    # Event definitions
    thresh_ev = Events.makeZeroCrossEvent('v', 1,
                                       {'name': 'thresh_ev',
                                        'eventtol': 1e-4,
                                        'precise': True,
                                        'term': False},
                                       varnames=['v'],
                                       targetlang=targetlang)

    min_ev = Events.makeZeroCrossEvent('(Iapp-ionic(v,m,h,n))/C', 1,
                                       {'name': 'min_ev',
                                        'eventtol': 1e-4,
                                        'precise': True,
                                        'term': False},
                                       varnames=['v', 'm', 'n', 'h'],
                                       parnames=par_args.keys(),
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


# ------------------------------------------------------------

if __name__ == '__main__':
    par_args = {'gna': 100, 'gk': 80, 'gl': 0.1,
                    'vna': 50, 'vk': -100, 'vl': -67,
                    'Iapp': 0.4, 'C': 1.0}
    ic_args = {'v':-70.0, 'm': 0, 'h': 1, 'n': 0}


    HH = makeHHneuron('HH', par_args, ic_args, gentype=gentype)
    HH.set(tdata=[0,100])

    # Some definitions for convenience
    ma = HH.auxfns.ma
    mb = HH.auxfns.mb
    na = HH.auxfns.na
    nb = HH.auxfns.nb

    def tau_m(V):
        return 1/(ma(V)+mb(V))



    traj = HH.compute('test')
    pts = traj.sample()
    plt.plot(pts['t'], pts['v'])
    plt.show()
