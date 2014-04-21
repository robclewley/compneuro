"""
Entrainment of a single biophysical neuron by a periodic pulse train.
Cf. Chapter 10 of Izhikevich.
"""
from __future__ import division
from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *
from common_lib import *

gentype='radau'  # dopri, euler, etc.

def makeHHneuron(name, excit_type, par_args, ic_args,
                 Istim_dict=None, gentype='vode', specials=None):
    """specials is an optional argument that permits arbitrary
    addition of entries to DSargs.
    Istim_dict (optional) is a pre-prepared external input dict of
     Variable objects, with the key 'Istim'.
    """
    if gentype in ['vode', 'euler']:
        targetlang = 'python'
    else:
        targetlang = 'C'
    if Istim_dict is None:
        Istim_str = ''
    else:
        Istim_str = '+Istim'
    vfn_str = '(Iapp-ionic(v,m,h,n)%s)/C' % Istim_str
    mfn_str = 'ma(v)*(1-m)-mb(v)*m'
    nfn_str = 'na(v)*(1-n)-nb(v)*n'
    hfn_str = 'ha(v)*(1-h)-hb(v)*h'

    if excit_type == 'Type I':
        auxfndict = {
               'ma': (['v'], '0.32*(v+54)/(1-exp(-(v+54)/4))'),
               'mb': (['v'], '0.28*(v+27)/(exp((v+27)/5)-1)'),
               'ha': (['v'], '.128*exp(-(50+v)/18)'),
               'hb': (['v'], '4/(1+exp(-(v+27)/5))'),
               'na': (['v'], '.032*(v+52)/(1-exp(-(v+52)/5))'),
               'nb': (['v'], '.5*exp(-(57+v)/40)')
               }
    elif excit_type == 'Type II':
        auxfndict = {
               'ma': (['v'], ' 0.1*(v+40)/(1-exp(-(v+40)/10))'),
               'mb': (['v'], '4*exp(-(v+65)/18)'),
               'ha': (['v'], '.07*exp(-(v+65)/20)'),
               'hb': (['v'], '1/(1+exp(-(v+35)/10))'),
               'na': (['v'], '.01*(v+55)/(1-exp(-(v+55)/10))'),
               'nb': (['v'], '.125*exp(-(v+65)/80)')
               }
    else:
        raise ValueError("Invalid excitability type")


    auxfndict['ionic'] = (['vv', 'mm', 'hh', 'nn'],
            'gna*mm*mm*mm*hh*(vv-vna) + gk*nn*nn*nn*nn*(vv-vk) + gl*(vv-vl)')
    DSargs = args()
    DSargs.varspecs = {'v': vfn_str,'m': mfn_str,
                       'h': hfn_str, 'n': nfn_str}
    DSargs.auxvars = []
    if Istim_dict is not None:
        inputnames = ['Istim']
        DSargs.inputs = Istim_dict
        DSargs.varspecs['I'] = 'Istim'
        DSargs.auxvars.append('I')
    else:
        inputnames = []
    DSargs.pars = par_args
    DSargs.fnspecs = auxfndict
    DSargs.xdomain = {'v': [-130, 70], 'm': [0,1], 'h': [0,1], 'n': [0,1]}
    DSargs.algparams = {'init_step':0.05,
                        'max_pts': 10000,
                        'maxevtpts': 200}
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

    min_ev = Events.makeZeroCrossEvent('(Iapp-ionic(v,m,h,n)%s)/C' % Istim_str,
                                       1,
                                       {'name': 'min_ev',
                                        'eventtol': 1e-5,
                                        'precise': True,
                                        'term': False},
                                       varnames=['v', 'm', 'n', 'h'],
                                       parnames=par_args.keys(),
                                       inputnames=inputnames,
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


def test_f(HH, Istim, freq, stim_t0, duration=1, tmax=1000):
    """Istim is amplitude of square pulse input to neuron, having
    given duration in ms and frequency in Hz. Starts at stim_t0.
    """
    baseline_Iapp = HH.pars['Iapp']
    stim_period = 1000./freq
    HH.set(tdata=[0, tmax])
    n = int(floor(tmax/stim_period))
    print "Testing with stimulus frequency %.3f Hz" % freq
    print "  (stimulus period is %.4f ms)" % stim_period
    print "Stimulus amplitude is %.3f" % Istim
    stim_ts = array([stim_t0+i*stim_period for i in range(0,n+1)])
    Istim_vardict = make_spike_signal(stim_ts, 1, tmax*1.1, loval=0, hival=Istim, dt=0.1)
    HH.inputs = Istim_vardict
    HH._extInputsChanged = True
    traj = HH.compute('stim_test')
    pts = traj.sample()
    plt.figure(1)
    plt.clf()
    plt.plot(pts['t'], pts['v'], 'b', lw=2)
    plt.plot(pts['t'], 3*pts['I']-75, 'k', lw=2)
    #plt.ylim([-100, 50])
    plt.xlabel('t')
    plt.ylabel('v')
    plt.title('Voltage trace and I(t) pulse stimulus')
    try:
        show_maps(traj, stim_ts, 0.3*tmax)
    except IndexError:
        print "Not enough spikes to show a map"
    return traj, pts, stim_ts

def show_maps(traj, stim_ts_all, settle_time=2000):
    v_ts_all = array(traj.getEventTimes('thresh_ev'))
    v_ix = find(v_ts_all, settle_time)
    v_ts0 = v_ts_all[:v_ix]
    v_ts1 = v_ts_all[v_ix:]
    stim_ix = find(stim_ts_all, settle_time)
    stim_ts0 = array(stim_ts_all[:stim_ix])
    stim_ts1 = array(stim_ts_all[stim_ix:])
    min_len = min(len(v_ts_all), len(stim_ts_all))

    plt.figure(2)
    plt.clf()
    plt.plot(stim_ts_all[:min_len], v_ts_all[:min_len]-stim_ts_all[:min_len], '.', color='gray')
    plt.plot(stim_ts_all[stim_ix:min_len-5], v_ts_all[stim_ix:min_len-5]-stim_ts_all[stim_ix:min_len-5], 'ko')
    plt.plot(stim_ts_all[min_len-5:min_len], v_ts_all[min_len-5:min_len]-stim_ts_all[min_len-5:min_len], 'ro')
    # This is the period provided the system has settled to a period-1 cycle
    Tper = v_ts1[-1] - v_ts1[-2]
    plt.xlabel('pulse stim time')
    plt.ylabel('time diff')
    plt.title('Threshold time since stim (period = %.4f)' % Tper)

    plt.figure(3)
    plt.clf()
    plt.plot([0,120],[0,120], 'r')
    dts0 = npy.diff(v_ts0)
    plt.plot(dts0[:-1], dts0[1:], '.', color='gray')
    dts1 = npy.diff(v_ts1)
    plt.plot(dts1[:-5], dts1[1:-4], 'ko')
    plt.plot(dts1[-5:-1], dts1[-4:], 'ro')
    x0 = min(min(dts1), min(dts0))*0.9
    x1 = max(max(dts1), max(dts0))*1.1
    plt.xlim([x0,x1])
    plt.ylim([x0,x1])
    plt.xlabel('thresh t diff n')
    plt.ylabel('thresh t diff n+1')
    plt.title('Threshold return map')


#--------------------

default_Istim_vardict = make_spike_signal([1], 0.5, 2000, loval=0, hival=1, dt=0.1)

specials = {'algparams': {'max_pts': 100000,
                          'maxevtpts': 1000}}

# Make Type I HH neuron
par_args_I = {'gna': 100, 'gk': 80, 'gl': 0.1,
            'vna': 50, 'vk': -100, 'vl': -67,
            'Iapp': 0.2, 'C': 1.0}

ic_args_I = {'v':-70.0, 'm': 0, 'h': 1, 'n': 0}

HH_I = makeHHneuron('HH_entrain_I', 'Type I', par_args_I, ic_args_I,
                    Istim_dict=default_Istim_vardict,
                    gentype=gentype, specials=specials)

# Make Type II HH neuron (bimodal PRC)
par_args_II = {'gna': 120, 'gk': 36, 'gl': 0.3,
            'vna': 50, 'vk': -78, 'vl': -54.4,
            'Iapp': 8, 'C': 1.0}

ic_args_II = {'v':-70.0, 'm': 0.02, 'h': 0.6, 'n': 0.4}

HH_II = makeHHneuron('HH_entrain_II', 'Type II', par_args_II, ic_args_II,
                    Istim_dict=default_Istim_vardict,
                    gentype=gentype, specials=specials)


print "For both neurons, vary stim amplitude between 0.5 and 2 (first numeric parameter)"
print "For Type I neuron, vary stim frequency between 5 and 30 Hz (second numeric parameter)"
print "For Type II neuron, vary stim frequency between 30 and 130 Hz (second numeric parameter)"

## Function signature:
# test_f( model, amplitude, frequency, duration, stimulus onset time, max integration time )


####### Type I neuron
### cyclic, high period
#traj, pts, stim_ts = test_f(HH_I, 0.5, 17, 1, 300, 5000)

### chaos
# traj, pts, stim_ts = test_f(HH_I, 0.75, 45, 1, 300, 5000)

### 1:2 entrainment
#traj, pts, stim_ts = test_f(HH_I, 0.75, 30, 1., 300, 5000)

### cyclic with period 3 map
#traj, pts, stim_ts = test_f(HH_I, 1, 20, 1., 300, 5000)

### 1:1 entrainment - in-phase synch (with excitation)
#traj, pts, stim_ts = test_f(HH_I, 1, 15, 1, 300, 5000)


####### Type II neuron
### pretty, chaotic map
#traj, pts, stim_ts = test_f(HH_II, 1, 85, 1.5, 300, 1000)

### 1:2 entrainment
#traj, pts, stim_ts = test_f(HH_II, 1.2, 120, 1.5, 300, 1000)

### 1:1 entrainment -- anti-phase synch (with excitation)
# has a transient cycle-skipping around t = 200 (compare Ch. 10 Fig 10.16)
#traj, pts, stim_ts = test_f(HH_II, 1.2, 60, 1.5, 300, 1000)

plt.show()

### Arnold tongue
# Example with Type I
# use four levels of impulse amplitude: 0.5, 0.7, 1, 1.6

# start with amp = 1
traj, pts, stim_ts = test_f(HH_I, 1, 15, 1, 300, 8000)
tongue_I = ((0.25, (13.3, 13.8)),
          (0.5, (13.3, 14.5)),
          (0.7, (13.3, 15)),
          (1.0, (13.3, 16)),
          (1.5, (13.3, 17.6))
          )

plt.figure(4)
for amp, (f0, f1) in tongue_I:
    plt.plot([f0, f1], [amp, amp], 'kx-')
plt.ylim([0, 1.6])
plt.title('Arnold tongue for 1:1 in HH Type I')

tongue_II = ((0.25, (60.8, 62.5)),
            (0.5, (60.0, 63.3)),
            (0.7, (59.3, 64.0)),
            (1.2, (55.4, 65.6)),
            (1.5, (55.3, 66.1))
            )

plt.figure(5)
for amp, (f0, f1) in tongue_II:
    plt.plot([f0, f1], [amp, amp], 'kx-')
plt.ylim([0, 1.6])
plt.title('Arnold tongue for 1:1 in HH Type II')
