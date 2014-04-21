from __future__ import division
from PyDSTool import *
from PyDSTool.Toolbox import phaseplane as pp

from matplotlib import pyplot as plt


pars = # Your definitions here
icdict = {'x': 0,
          'y': 1}
xstr = # Your definition here
ystr = # Your definition here

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

    # let solution settle
    transient = fhn.compute('trans')
    fhn.set(ics=transient(10),
            tdata=[0,20])

    # More of your code here


# Your code here for the frequency plot



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

