"""
Project 2 - stochastic resonance
"""
from __future__ import division
import sys
from PyDSTool import *
from common_lib import *
from Ch9_HH_red import *

# default
gentype='dopri'

# Parameter An = noise amplitude
#           As = sine wave amplitude
#           f = frequency, should be >= 50 Hz
par_args = {'tau_v': 0.8, 'tau_r': 3,
            'As': 0.065, 'f': 55, 'An': 0., 'Iapp': 0.01}
ic_args = {'v':-0.75, 'r': 0.5}

# build neuron model
HH = makeHHneuron('HHred_noise', par_args, ic_args, const_I=True, gentype=gentype)

# set special conditions for using noise
HH.eventstruct.setActiveFlag('min_ev', False)
HH.set(tdata=[0,50])  # this is just a temporary value
if gentype == 'vode':
    HH.set(algparams={'init_step': 0.02, 'stiff': False})
else:
    HH.set(algparams={'init_step': 0.08})


def avfreq(traj):
    """
    Average frequency in presence of noise
    """
    evs = array(traj.getEventTimes('thresh_ev'))
    l = len(evs)
    if l == 0:
        return 0
    elif l == 1:
        print "Not enough events found"
        return 0
    else:
        # take mean average of inter-spike intervals between n events found
        # evs[1:] is all but one event ranging 1 -> n
        # evs[:-1] is all but one event ranging 1 -> n-1
        # evs is just a list, so convert to array to do array arithmetic
        return 1000./mean(evs[1:] - evs[:-1])

def integrate(An, As, tmax=50, silent=False):
    """Optional silent option suppresses text output and plots
    """
    HH.set(pars={'An': An,
                 'As': As},
           tdata=[0,tmax])
    traj = HH.compute('test')
    pts = traj.sample()
    f = avfreq(traj)
    if not silent:
        plt.figure(1)
        plt.clf()
        plt.plot(pts['t'], pts['v'], 'b')
        print "Average frequency response:", f
    return traj, pts, f

def ISI(traj, throwout=3):
    # throw out first few during transient (assume run long enough)
    ts = array(traj.getEventTimes('thresh_ev')[throwout:])
    return ts[1:] - ts[:-1]

def do_hist(ISIs, upper=None, bin_width=1):
    if upper is None:
        upper = ceil(max(ISIs))
    plt.figure(2)
    plt.clf()
    #r = np.histogram(ISIs, bins=int(upper/bin_width), range=(0,upper))
    #plt.plot(r[1][1:], r[0],'ko-')
    plt.hist(ISIs, int(upper/bin_width), range=(0,upper))


def test_run(An, As, tmax=2000):
    traj, pts, f = integrate(An, As, tmax)
    ISIs = ISI(traj)
    do_hist(ISIs, bin_width=1)
    plt.text(0.75, 0.9, '%.3f'%An, transform=plt.gca().transAxes, fontsize=20)
    return f



def testrun2(sin_amp,noise_amp,final_time):
    HH.set(pars={'As':sin_amp, 'An':noise_amp},tdata=[0,final_time])
    Isignal_vardict = make_noise_signal(.05,10000,0,0.4,1)
    HH.inputs= {'noise_sig': Isignal_vardict['noise1']}
    HH._extInputsChanged = True
    traj = HH.compute('test')
    return traj

def multiple_neurons(sin_amp,noise_amp,final_time, number):
    sp_ts = []
    for instance in range(number):
	traj = testrun2(sin_amp,noise_amp,final_time)
	new_sp_ts = traj.getEventTimes('thresh_ev')[3:]
	sp_ts.extend(new_sp_ts)
    # sorting is absolutely crucial!
    isis = np.diff(sort(sp_ts))
    figure(3)
    plt.clf()
    plt.xlabel('Inter-Spike Interval (ms)')
    plt.ylabel('Count')
    largest_isi=1500
    plt.hist(isis,bins=largest_isi,range=[0,largest_isi])
    return isis

test = multiple_neurons(.06,.03,5000,10)


1/0

ns = linspace(0.015, .05, 20)
fs = []
for n in ns:
    fs.append(test_run(n, 0.07, 1000))

figure()
plot(ns, fs)
show()
1/0


#traj, pts = test_noise(0.15, 10000)
#ISIs = ISI(traj)
#savetxt('ISIs.dat', ISIs)
ISIs = genfromtxt('ISIs.dat')

do_hist(ISIs, 40, 0.5)
plt.show()

