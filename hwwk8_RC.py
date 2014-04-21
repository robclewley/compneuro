"""
Ver. 2: Adds stuff
"""
from __future__ import division
from PyDSTool import *

gentype = 'vode' # 'vode'

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
    aux_str = 'm*m*m*h'

    auxfndict = {'ionic': (['vv', 'mm', 'hh', 'nn'],
            'gna*mm*mm*mm*hh*(vv-vna) + gk*nn*nn*nn*nn*(vv-vk) + gl*(vv-vl)'),
               'ma': (['v'], '0.32*(v+54)/(1-exp(-(v+54)/4))'),
               'mb': (['v'], '0.28*(v+27)/(exp((v+27)/5)-1)'),
               'ha': (['v'], '.128*exp(-(50+v)/18)'),
               'hb': (['v'], '4/(1+exp(-(v+27)/5))'),
               'na': (['v'], '.032*(v+52)/(1-exp(-(v+52)/5))'),
               'nb': (['v'], '.5*exp(-(57+v)/40)'),
               'ptest': (['p'], '1+p+ma(-50)+C'),
               'atest': (['q'], 'q+mb(-40)')}  # tests fn cross-calling

    DSargs = args()
    DSargs.varspecs = {'v': vfn_str, 'm': mfn_str,
                       'h': hfn_str, 'n': nfn_str,
                       'v_bd0': 'getbound("v",0)',  # demo of referencing bounds
                       'v_bd1': 'getbound("v",1)'}
    DSargs.pars = par_args
    DSargs.auxvars = ['v_bd0','v_bd1']
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

# Some definitions for convenience
par_args = {'gna': 100, 'gk': 80, 'gl': 0.1,
                'vna': 50, 'vk': -100, 'vl': -67,
                'Iapp':0.1, 'C': 1.0}
ic_args = {'v':-70.0, 'm': 0, 'h': 1, 'n': 0}

i=[1,2,3,4]
HH = makeHHneuron('HH', par_args, ic_args, gentype=gentype)

def amplitude(pt):
    return max(pt['v'])-min(pt['v'])
def frequency(pt):
    return (1/(max(pt['t'])-pt['t'][0]))*1000
##    else:
##    return 'No spiking'
##    return array(freq)

def run_for_n(n, evname='min_ev'):
    """n integer
    """
    # record state prior to function call
    changed_ev = False
    if evname not in HH.eventstruct.getTermEvents():
        HH.eventstruct.setTermFlag(evname, True)
        changed_ev = True
    old_ic = HH.query('ics')
    # compute n times
    t0 = 0
    pts = None
    for i in range(1,n+1):
        traj_piece = HH.compute('part_%i' % i)
        new_pts = traj_piece.sample()
        # avoid duplicate of final point
        if pts is None:
            pts = new_pts
        else:
            new_pts.indepvararray += t0
            new_pts.indepvararray = new_pts.indepvararray.flatten()
            pts.extend(new_pts[1:])
            t_event = pts.labels[len(pts)-1]['Event:%s'%evname]['t']
            pts.labels.update(len(pts)-1, 'Event:%s'%evname, {'t': t_event+t0})
        t0 = t0 + new_pts['t'][-1]
        # update IC with final point for next time around loop
        new_ic = new_pts[-1]
        HH.set(ics=new_ic)
    # restore state prior to function call
    HH.set(ics=old_ic)
    if changed_ev:
        HH.eventstruct.setTermFlag(evname, False)
    # return a trajectory made of the points, and also just the points
    return pointset_to_traj(pts), pts

### TEMP
##HH.set(pars={'Iapp': 0.3}, tdata=[0,500])
##traj, pts = run_for_n(3, 'min_ev')
##1/0

def evtimes_from_pts(pts):
    evsdict = pts.labels.by_label['Event:min_ev']
    ixs= evsdict.keys()
    return sort(pts['t'][ixs])


Iapps=linspace(0,0.3,40)
amps=[]
freq=[]
for I in Iapps:
    HH.set(tdata=[0,5000],pars={'Iapp':I})
    ma = HH.auxfns.ma
    mb = HH.auxfns.mb
    na = HH.auxfns.na
    nb = HH.auxfns.nb
    traj, pts = run_for_n(3, 'min_ev')
##    plt.plot(pts['t'], pts['v'])

    #ts = traj.getEventTimes('min_ev')
    ts = evtimes_from_pts(pts)
    ix0 = pts.find(ts[-2],0)
    ix1 = pts.find(ts[-1],1)
    per = pts[ix0:ix1]
##    per['t'] = per['t'] - per['t'][0]

    a = amplitude(per)
    amps.append(a)
    if a>35:
        freq.append(frequency(per))
    else:
        freq.append(0)
##    print 'Frequency(Hz):', freq
##
##    print 'Amplitude(mV):',amps
##ampss=array(amps)
##freqs=array(freq)
plt.figure(2)
plt.plot(Iapps,freq)
plt.xlabel('I(pA)')
plt.ylabel('frequency(Hz)')
plt.figure(3)
plt.plot(Iapps,amps)
plt.xlabel('I(pA)')
plt.ylabel('Amplitude(mV)')
##plt.figure(2)
####plt.plot(Is,freq,'g')
##plt.plot(Is,amps,'r')
plt.show()