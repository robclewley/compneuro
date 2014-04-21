from __future__ import division
from PyDSTool import *
from PyDSTool.Toolbox import phaseplane as pp

from matplotlib import pyplot as plt


pars = {'eps': 2e-1, 'a': 0.8, 'I': 0}
icdict = {'x': 0,
          'y': 1}
xstr = '(y - (x*x*x/3 - x))/eps + I'
ystr = 'a - x'

print """This script will work in VCL. If this script does not work
on your own computer setup, you have two options:

1) follow the instructions at
http://www.ni.gsu.edu/~rclewley/PyDSTool/GettingStarted.html
for "C-based integrators" for your platform, e.g. Windows.

2) Make three small changes to this script to use the Vode integrator.
Unfortunately, it will run much more slowly than the C-based integrators.
  * Change 'C' to 'python' on line 34
  * Add the option 'stiff': True back into the algparams dictionary on line 40
  * Change Dopri_ODEsystem to Vode_ODEsystem on line 47
"""

event_x_a = makeZeroCrossEvent('x-a', 0,
                            {'name': 'event_x_a',
                             'eventtol': 1e-6,
                             'term': False,
                             'active': True},
                    varnames=['x'], parnames=['a'],
                    targetlang='C')  # 'python'

DSargs = args(name='vanderpol')  # struct-like data
DSargs.events = [event_x_a]
DSargs.pars = pars
DSargs.tdata = [0, 10]
DSargs.algparams = {'max_pts': 3000, 'init_step': 0.02} #, 'stiff': True}
DSargs.varspecs = {'x': xstr, 'y': ystr}
DSargs.xdomain = {'x': [-2.2, 2.5], 'y': [-2, 2]}
DSargs.fnspecs = {'Jacobian': (['t','x','y'],
                                """[[(1-x*x)/eps, 1/eps ],
                                    [ -1,  0 ]]""")}
DSargs.ics = icdict
vdp = Dopri_ODEsystem(DSargs)  # Vode_ODEsystem
vdp_e = Euler_ODEsystem(DSargs)

traj = vdp.compute('v')
pts = traj.sample()

traj_e = vdp_e.compute('e')
pts_e = traj_e.sample()

plt.plot(pts['x'], pts['y'], 'k.-', linewidth=2)
plt.plot(pts_e['x'], pts_e['y'], 'r.-', linewidth=2)

fp_coord = pp.find_fixedpoints(vdp, n=4, eps=1e-8)[0]
fp = pp.fixedpoint_2D(vdp, Point(fp_coord), eps=1e-8)

nulls_x, nulls_y = pp.find_nullclines(vdp, 'x', 'y', n=3, eps=1e-8,
                                      max_step=0.1, fps=[fp_coord])
pp.plot_PP_fps(fp)
plt.plot(nulls_x[:,0], nulls_x[:,1], 'b')
plt.plot(nulls_y[:,0], nulls_y[:,1], 'g')

plt.show()

PC = ContClass(vdp)
PCargs = args(name='EQ1', type='EP-C')
PCargs.freepars = ['a']
PCargs.StepSize = 1e-2
PCargs.MaxNumPoints = 350
PCargs.MaxStepSize = .1
PCargs.LocBifPoints = 'all'
PCargs.verbosity = 2
PCargs.SaveEigen = True
PC.newCurve(PCargs)
PC['EQ1'].forward()
PC['EQ1'].backward()

sol = PC['EQ1'].sol
Hopf_dict = sol.labels.by_label['H']
assert len(Hopf_dict) == 2, "Expect exactly 2 Hopf points!"
H_on = min(Hopf_dict.keys())
H_off = max(Hopf_dict.keys())

i = 0 # global

def ampl(a):
    global i
    vdp.set(pars={'a': a},
            ics={'x': 0, 'y': 0},
            tdata=[0,20])
    # let solution settle
    transient = vdp.compute('trans')
    vdp.set(ics=transient(20),
            tdata=[0,6])
    traj = vdp.compute('ampl')
    pts = traj.sample()
    if mod(i, 10) == 0 or 1-abs(a) < 0.02:
        plt.figure(3)
        plt.plot(pts['x'], pts['y'], 'k-')
        plt.draw()
    i += 1
    return np.linalg.norm([max(pts['x']) - min(pts['x']), max(pts['y']) - min(pts['y'])])

a_vals = linspace(sol[H_on]['a'], sol[H_off]['a'], 400)[::-1]

plt.figure(3)
plt.plot(nulls_x[:,0], nulls_x[:,1], 'b')
amps = [ampl(a) for a in a_vals]
plt.xlabel('x')
plt.ylabel('y')
plt.title('cycles as \emph{a} varies')

plt.figure(4)
plt.plot(a_vals, amps, 'o-')
plt.xlabel('a')
plt.ylabel('amplitude')