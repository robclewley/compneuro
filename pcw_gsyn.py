from PyDSTool import *
from euler import euler_integrate
from common_lib import *

icdict = {'V': -75}
pardict = {'tau': 12.5, 'gl': 1, 'El': -75,
           'ge': 0, 'Ee': 0}

V_rhs = '-1/tau*(gl*(V-El)+ge*(V-Ee))'

vardict = {'V': V_rhs}

DSargs = args()                   # create an empty object instance of the args class, call it DSargs
DSargs.name = 'timescales'        # name our model
DSargs.ics = icdict               # assign the icdict to the ics attribute
DSargs.pars = pardict             # assign the pardict to the pars attribute
DSargs.tdata = [0, 90]            # declare how long we expect to integrate for
DSargs.varspecs = vardict         # assign the vardict dictionary to the 'varspecs' attribute of DSargs

DS = Generator.Vode_ODEsystem(DSargs)
DS_e = Generator.Euler_ODEsystem(DSargs)
DS_e.set(algparams={'init_step': 0.5,
                    'max_step': 0.5})

# primitive protocol for switched stages
t1 = 10
s1 = args(pars={'ge': 0},
       tdur=t1)
t2 = 2
s2 = args(pars={'ge': 2},
       tdur=t2)
t3 = 30
s3 = args(pars={'ge': 0},
       tdur=t3)

traj, pts = pcw_protocol(DS, [s1,s2,s3])
traj_e, pts_e = pcw_protocol(DS_e, [s1,s2,s3])

plt.plot(pts['t'], pts['V'])
plt.plot(pts_e['t'], pts_e['V'])

plt.ylabel('V')
plt.xlabel('t')
plt.show()