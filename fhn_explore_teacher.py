from __future__ import division
from PyDSTool import *
from PyDSTool.Toolbox import phaseplane as pp

from matplotlib import pyplot as plt


pars = # {'eps': 0.08, 'a': 0.7, 'b':0.8, 'I': 0.5}
icdict = #{'x': 0,
         # 'y': 1}
xstr = #'(-y - (x*x*x/3 - x) + I)/eps'
ystr = #'a + x -b*y'

event_x_a = makeZeroCrossEvent('x-a+b*y', 1,
                            {'name': 'event_x_a',
                             'eventtol': 1e-6,
                             'term': False,
                             'active': True},
                    varnames=['x'], parnames=['a','b'],
                    targetlang='C')  # 'python'

DSargs = args(name='fhn')  # struct-like data
DSargs.events = [event_x_a]
DSargs.pars = pars
DSargs.tdata = [0, 10]
DSargs.algparams = {'max_pts': 3000, 'init_step': 0.02} #, 'stiff': True}
DSargs.varspecs = {'x': xstr, 'y': ystr}
DSargs.xdomain = {'x': [-2.2, 2.5], 'y': [-3, 3]}
DSargs.fnspecs = {'Jacobian': (['t','x','y'],
                                """[[(1-x*x)/eps, -1/eps ],
                                    [ 1,  -b ]]""")}
DSargs.ics = icdict
fhn = Dopri_ODEsystem(DSargs)  # Vode_ODEsystem

traj = fhn.compute('v')
pts = traj.sample()

plt.plot(pts['t'], pts['x'], 'k.-')

def freq(I):
    # Your code here
    fhn.set(pars={'I': I})
    # let solution settle
    transient = fhn.compute('trans')
    fhn.set(ics=transient(10),
            tdata=[0,20])
    traj = fhn.compute('freq')
    ts = traj.getEventTimes('event_x_a')
    try:
        return 1/(ts[-1]-ts[-2])
    except:
        return 0

# Your code here for the frequency plot
Is = linspace(0, 2, 100)
fs = []

for I in Is:
    fs.append(freq(I))

plt.figure(2)
plt.plot(Is, fs, 'k.')
plt.xlabel('I')
plt.ylabel('frequencies')

1/0  # comment this to apply phase plane picture to whatever
# are the current parameters of FHN model

## Optional code

fp_coord = pp.find_fixedpoints(fhn, n=25, eps=1e-6)[0]
fp = pp.fixedpoint_2D(fhn, Point(fp_coord), eps=1e-6)

nulls_x, nulls_y = pp.find_nullclines(fhn, 'x', 'y', n=3, eps=1e-6,
                                      max_step=0.1, fps=[fp_coord])
plt.figure(3)
pp.plot_PP_fps(fp)
plt.plot(nulls_x[:,0], nulls_x[:,1], 'b')
plt.plot(nulls_y[:,0], nulls_y[:,1], 'g')

plt.show()

