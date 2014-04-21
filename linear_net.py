"""
Sets up network of n linear elements with coupling matrix M

d/dt ( x1, ..., xn )^T = M_{n x n} ( x1, ..., xn )^T

See Wilson, Ch. 4.3

"""
from __future__ import division

from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *


def make_system(M, ics, pars=None):
    n = len(M)
    xvarlist = ['x%i' %i for i in range(1,n+1)]
    xvars = [Var(x) for x in xvarlist]

    varspecs = {}
    jac_str_list = []
    for i, xv in enumerate(xvars):
        M_row = M[i]
        varspecs[str(xv)] = QuantSpec(str(xv)+'_DE',
                                   '+'.join([str(M_row[j]*xvars[j]) for j in range(n)]))
        jac_str_list.append('['+','.join([str(M_row[j]) for j in range(n)])+']')

    jac_str = '[' + ','.join(jac_str_list) + ']'
    print jac_str

    DSargs = args(name='linear_net')
    DSargs.varspecs = varspecs
    DSargs.ics = dict(zip(xvars, ics))
    DSargs.pars = pars
    DSargs.tdata = [0, 1000]
    DSargs.fnspecs = {'Jacobian': (['t']+xvarlist, jac_str)}
    return Generator.Vode_ODEsystem(DSargs)


def display(pts, fignum=1):
    figure(fignum)
    ts = pts['t']
    for coord in pts.coordnames:
        plt.plot(ts, pts[coord], label=coord)
    plt.legend()

# ------------------------------------------------

g = Par(0.01, 'g')

M = [[-3, 0, -g, -5],
     [-5, -3, 0, -g],
     [-g, -5, -3, 0],
     [0, -g, -5, -3]]

net = make_system(M, [1, 1, 0.1, 0.1], [g])

net.set(tdata=[0, 20],
        algparams={'init_step': 0.01})

traj = net.compute('test')
pts = traj.sample()

display(pts)
plt.show()


origin = Point({'x1': 0, 'x2': 0, 'x3': 0, 'x4': 0})

fp = fixedpoint_nD(net, origin)

print fp.evals
print fp.stability

## Continuation to explore stability
PC = ContClass(net)

PCargs = args()
PCargs.name='EQ1'
PCargs.type = 'EP-C'
PCargs.freepars = ['g']
PCargs.StepSize = 1e-3
PCargs.MaxNumPoints = 100
PCargs.MaxStepSize = 1e-1
PCargs.LocBifPoints = 'all'
PCargs.verbosity = 1
PCargs.SaveEigen = True
PCargs.Corrector = 'Natural'
PC.newCurve(PCargs)

PC['EQ1'].forward()

# Look at the scale!
PC.display(('g','x1'), stability=True, linewidth=0.5, figure=10)
plt.figure(10)
plt.ylim([-1e-6, 1e-6])
plt.draw()

sol = PC['EQ1'].sol

g_BP = sol.bylabel('BP')['g']
g_H = sol.bylabel('H')['g']

net.set(pars={'g': 0.5*(g_BP+g_H)})

traj = net.compute('test_stable')
pts_stab = traj.sample()
display(pts_stab, 2)

net.set(pars={'g': g_H})  # degenerate Hopf (has a zero e'val)

traj = net.compute('test_osc')
pts_osc = traj.sample()
display(pts_osc, 3)

jac_H = net.Jacobian(0, origin)
print np.linalg.eigvals(jac_H)