from PyDSTool import *
from common_lib import *

icdict = {'V': -75}
pardict = {'tau': 12.5, 'gl': 1, 'El': -75,
           'gna': 0, 'Ena': 50, 'gk': 0, 'Ek': -90}

V_rhs = '-1/tau*(gl*(V-El)+gna*(V-Ena)+gk*(V-Ek))'

vardict = {'V': V_rhs}

DSargs = args()                   # create an empty object instance of the args class, call it DSargs
DSargs.name = 'timescales'        # name our model
DSargs.ics = icdict               # assign the icdict to the ics attribute
DSargs.pars = pardict             # assign the pardict to the pars attribute
DSargs.tdata = [0, 90]            # declare how long we expect to integrate for
DSargs.varspecs = vardict         # assign the vardict dictionary to the 'varspecs' attribute of DSargs
DSargs.algparams = {'init_step': 0.02}
DS = Generator.Vode_ODEsystem(DSargs)

# primitive protocol for switched stages
t1 = 10
s1 = args(pars={'gna': 0.1, 'gk': 0.4},
       tdur=t1)
t2 = 1
s2 = args(pars={'gna': 5, 'gk': 0.6},
       tdur=t2)
t3 = 4
s3 = args(pars={'gna': 0, 'gk': 5},
       tdur=t3)
t4 = 10
s4 = args(pars={'gna': 0.1, 'gk': 0.4},
       tdur=t4)

traj, pts = pcw_protocol(DS, [s1,s2,s3, s4])

plt.plot(pts['t'], pts['V'])

plt.ylabel('V')
plt.xlabel('t')
plt.show()