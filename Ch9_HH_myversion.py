"""Type I excitability of Hodgkin-Huxley-style model
"""
from __future__ import division
from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *
from PyDSTool.Toolbox.PRCtools import *
import sys

gentype = 'dopri' # 'vode'

# --------------------------------------

if gentype in ['vode', 'euler']:
    targetlang = 'python'
else:
    targetlang = 'C'

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
    elif gentype == 'radau':
        return Generator.Radau_ODEsystem(DSargs)
    elif gentype == 'dopri':
        return Generator.Dopri_ODEsystem(DSargs)
    else:
        raise NotImplementedError("Unknown ODE system type: %s"%gentype)


# ------------------------------------------------------------

par_args = {'gna': 100, 'gk': 80, 'gl': 0.1,
                'vna': 50, 'vk': -100, 'vl': -67,
                'Iapp': 0.4, 'C': 1.0}
ic_args = {'v':-70.0, 'm': 0, 'h': 1, 'n': 0}


HH = makeHHneuron('HHorig', par_args, ic_args, gentype=gentype)
HH.set(tdata=[0,100])

# Some definitions for convenience
ma = HH.auxfns.ma
mb = HH.auxfns.mb
na = HH.auxfns.na
nb = HH.auxfns.nb


traj = HH.compute('test')
pts = traj.sample()
plt.plot(pts['t'], pts['v'])
plt.show()


pd_info = one_period_traj(HH, 'min_ev', 1e-4, 1e-5, 40,
                    verbose=False, initial_settle=10)
ref_traj = pd_info[0]
ref_pts = pd_info[1]
T = pd_info[2]
print "Period is T=", T

# Calculate finite Phase Response Curve
dV = 0.1
PRC = finitePRC(HH, ref_traj, 'thresh_ev', 'v', dV, verbose=False, skip=5,
                keep_trajs=False)

# adjust for the fact that thresh_ev happens later than the min_ev used to
# more accurately define a period
dPRC = 1-ref_traj.getEventTimes('thresh_ev')[0]/T
PRC['D_phase'] -= dPRC

plt.figure(2)
plt.plot(PRC['t'], PRC['D_phase']*1000, 'k')
plt.plot(ref_pts['t'], ref_pts['v'], 'b')
plt.show()

1/0

##from common_lib import *
##pcw = []
##pcw.append(args(pars={'Iapp': 0},
##                tdata=[0, 100]))
##pcw.append(args(pars={'Iapp': 0.12},
##                tdata=[0, 2000]))
##pcw.append(args(pars={'Iapp': 0},
##                tdata=[0, 200]))
##
##traj, pts = pcw_protocol(HH, pcw)
##plt.plot(pts['t'], pts['v'])
##plt.show()
##1/0

def freq(traj):
    evs = traj.getEventTimes('thresh_ev')
    if len(evs) == 0:
        return 0
    elif len(evs) == 1:
        print "Not enough events found"
        return 0
    else:
        return 1000./(evs[-1] - evs[-2])

def amp(pts):
    return max(pts['v']) - min(pts['v'])

def test_I(Iapp):
    HH.set(pars={'Iapp': Iapp})
    traj = HH.compute('test')
    pts = traj.sample()
    return traj, pts

HH.set(tdata=[0,5000])
Is = linspace(0.11, 0.14, 100)
fs = []
amps = []

for I in Is:
    print ".",
    sys.stdout.flush()
    traj, pts = test_I(I)
    fs.append(freq(traj))
    amps.append(amp(pts))

plt.figure()
plt.plot(Is, fs, 'k.-')
plt.figure()
plt.plot(Is, amps, 'r.-')
plt.ylim(0, 150)
plt.show()
