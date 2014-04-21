from PyDSTool import *

icdict = {'x': 0.1}
pardict = {'a': 5.01, 'b': -0.5, 'c': 0.}

vardict = {'x': 'a*x*x + b*x + c'}

DSargs = args()                   # create an empty object instance of the args class, call it DSargs
DSargs.name = 'nonlin'               # name our model
DSargs.ics = icdict               # assign the icdict to the ics attribute
DSargs.pars = pardict             # assign the pardict to the pars attribute
DSargs.tdata = [0, 10]            # declare how long we expect to integrate for
DSargs.varspecs = vardict
DSargs.algparams = {'init_step': 0.01}

DS = Generator.Vode_ODEsystem(DSargs)

def test_a(a):
    DS.set(pars={'a':a})
    traj = DS.compute('demo')
    pts = traj.sample()
    return pts

for a in [0, 1, 2, 3, 4, 4.5, 5, 5.01]:
    pts = test_a(a)
    plt.plot(pts['t'], pts['x'], label='x')

plt.xlabel('t')

plt.show()