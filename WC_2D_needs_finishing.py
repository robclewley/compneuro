import PyDSTool as dst
from PyDSTool import args
import numpy as np
from matplotlib import pyplot as plt


pars = {'eps': 1e-2, 'a': 0.5}
icdict = {'x': pars['a'],
          'y': pars['a'] - pars['a']**3/3}
xstr = '(y - (x*x*x/3 - x))/eps'
ystr = 'a - x'

event_x_a = dst.makeZeroCrossEvent('x-a', 0,
                            {'name': 'event_x_a',
                             'eventtol': 1e-6,
                             'term': False,
                             'active': True},
                    varnames=['x'], parnames=['a'],
                    targetlang='python')  # targetlang is redundant (defaults to python)

DSargs = args(name='vanderpol')  # struct-like data
DSargs.events = [event_x_a]
DSargs.pars = pars
DSargs.tdata = [0, 3]
DSargs.algparams = {'max_pts': 3000, 'init_step': 0.02, 'stiff': True}
DSargs.varspecs = {'x': xstr, 'y': ystr}
DSargs.xdomain = {'x': [-2.2, 2.5], 'y': [-2, 2]}
DSargs.fnspecs = {'Jacobian': (['t','x','y'],
                                """[[(1-x*x)/eps, 1/eps ],
                                    [ -1,  0 ]]""")}
DSargs.ics = icdict
vdp = dst.Vode_ODEsystem(DSargs)

traj = vdp.compute('test_traj')
pts = traj.sample()
evs = traj.getEvents('event_x_a')


# figure 1 is the time evolution of the two variables
plt.figure(1)
plt.plot(pts['t'], pts['x'], 'b', linewidth=2)
plt.plot(pts['t'], pts['y'], 'r', linewidth=2)

# figure 2 is the phase plane
plt.figure(2)
# phase plane tools are in the Toolbox module
from PyDSTool.Toolbox import phaseplane as pp

# plot vector field, using a scale exponent to ensure arrows are well spaced
# and sized
pp.plot_PP_vf(vdp, 'x', 'y', scale_exp=-1)

# only one fixed point, hence [0] at end.
# n=4 uses three starting points in the domain to find any fixed points, to an
# accuracy of eps=1e-8.
fp_coord = pp.find_fixedpoints(vdp, n=4, eps=1e-8)[0]
fp = pp.fixedpoint_2D(vdp, dst.Point(fp_coord), eps=1e-8)

# n=3 uses three starting points in the domain to find nullcline parts, to an
# accuracy of eps=1e-8, and a maximum step for the solver of 0.1 units.
# The fixed point found is also provided to help locate the nullclines.
nulls_x, nulls_y = pp.find_nullclines(vdp, 'x', 'y', n=3, eps=1e-8,
                                      max_step=0.1, fps=[fp_coord])


# plot the fixed point
pp.plot_PP_fps(fp)

# plot the nullclines
plt.plot(nulls_x[:,0], nulls_x[:,1], 'b')
plt.plot(nulls_y[:,0], nulls_y[:,1], 'g')

# plot the trajectory
plt.plot(pts['x'], pts['y'], 'k-o', linewidth=2)

# plot the event points
plt.plot(evs['x'], evs['y'], 'rs')

plt.axis('tight')
plt.title('Phase plane')
plt.xlabel('x')
plt.ylabel('y')

# phase portrait of limit cycles (very approximate)
plt.figure(3)

# set integration time to be long enough to settle to the limit cycle,
# and extend y's domain to suit larger eps values
vdp.set(tdata=[0,30],
        xdomain={'y': [-8, 8]},
        algparams={'init_step': 0.05})

# select 6 values spaced logarithmically between 0.01 and 100
# incrementing the integration time as the oscillations get slower
for eps in np.power(10, np.linspace(-2, 1.25, 6)):
    vdp.set(pars={'eps': eps},
            tdata=[0, vdp.tdata[1]+40])
    traj = vdp.compute('eps_%.4f'%eps)
    pts = traj.sample()
    event_dict = pts.labels.by_label['Event:event_x_a']
    indices = np.sort(event_dict.keys())
    # assume there are at least 3 indices!
    # pick the last two that go through the same point
    ix1, ix2, ix3 = indices[-3:]
    plt.plot(pts['x'][ix1:ix3+1], pts['y'][ix1:ix3+1], label='%.3f'%eps)

# find nullclines for new domain (extends the y-nullcline)
nulls_x, nulls_y = pp.find_nullclines(vdp, 'x', 'y', n=3, eps=1e-8,
                                      max_step=0.2, fps=[fp_coord])

# plot nullclines again for reference (which are independent of eps)
plt.plot(nulls_x[:,0], nulls_x[:,1], 'k', lw=2)
plt.plot(nulls_y[:,0], nulls_y[:,1], 'k', lw=2)

# plot the fixed point
pp.plot_PP_fps(fp)

plt.axis('tight')
plt.title('Phase portrait as $\epsilon$ varies')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=3)  # bottom left location

plt.show()