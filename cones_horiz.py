"""
2D linear ODE for retinal feedback circuit with cones and horizontal cells
Eq. 3.20 of Ch. 3
"""

from __future__ import division

from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *
from common_lib import *

icdict = {'C': 0, 'H': 0}
pardict = {'tauC': 0.025, 'k': 4,
           'tauH': 0.08, 'L': 10}

DSargs = args()
DSargs.name = 'H-C'
DSargs.ics = icdict
DSargs.pars = pardict
DSargs.tdata = [0, 3]
DSargs.algparams = {'init_step': 1e-3}
DSargs.varspecs = {'C': '(-C-k*H+L)/tauC',
                   'H': '(-H+C)/tauH'}
# The Jacobian matrix is the matrix of first derivatives of the RHS
# with respect to x and y. Note the use of the triple-quoted string
# to conveniently format the text. 't' is always given as the first
# argument. The Jac is used by the code that finds fixed points, etc.
DSargs.fnspecs = {'Jacobian': (['t', 'C', 'H'],
                               """[[-1/tauC, -k/tauC],
                                   [1/tauH, -1/tauH]]
                               """)}

DS = Generator.Vode_ODEsystem(DSargs)


def do_plots(pts, fig_offset=0):
    plt.figure(1+fig_offset)
    plt.plot(pts['t'], pts['H'], 'g', label='H')
    plt.plot(pts['t'], pts['C'], 'r', label='C')
    plt.xlabel('t')
    plt.legend(loc='lower right')
    plt.figure(2+fig_offset)
    plt.plot(pts['H'], pts['C'], 'k')
    plt.xlabel('H')
    plt.ylabel('C')


# eps (epsilon) specifies the accuracy for the numerical
# algorithm that finds the fixed points. Read the docstring
# for more info!
fp_coords = find_fixedpoints(DS, n=4, eps=1e-6,
                             subdomain={'H': [-4,4],
                                        'C': [-4,4]})

print "%i fixed point(s) were found" % len(fp_coords)

# Find eigenvalues and vectors for the coordinates found
# Copy this format and adapt for your model's coordinate names
fp = fixedpoint_2D(DS, Point(fp_coords[0]), coords=['C', 'H'],
                       eps=1e-6)
print fp.point, '\n', fp.stability, fp.classification


# primitive protocol for switched stages
t1 = 1
s1 = args(pars={'L': 0},
       tdur=t1)
t2 = 1
s2 = args(pars={'L': 10},
       tdur=t2)
t3 = 1
s3 = args(pars={'L': 0},
       tdur=t3)

traj, pts = pcw_protocol(DS, [s1,s2,s3])

plt.figure(1)
do_plots(pts)
plt.show()


