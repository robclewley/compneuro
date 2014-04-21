"""
2D linear ODE, Eq. 3.11 of Ch. 3
"""

from __future__ import division

from PyDSTool import *
from PyDSTool.Toolbox.phaseplane import *

icdict = {'x': 0.4, 'y': 1.2}
pardict = {'a1': -9, 'a2': -5,
           'a3': 1, 'a4': -3,
           'b1': 7, 'b2': 1}

DSargs = args()
DSargs.name = 'lin2D'
DSargs.ics = icdict
DSargs.pars = pardict
DSargs.tdata = [0, 10]
DSargs.varspecs = {'x': 'a1*x + a2*y + b1',
                   'y': 'a3*x + a4*y + b2'}
# The Jacobian matrix is the matrix of first derivatives of the RHS
# with respect to x and y. Note the use of the triple-quoted string
# to conveniently format the text. 't' is always given as the first
# argument. The Jac is used by the code that finds fixed points, etc.
DSargs.fnspecs = {'Jacobian': (['t', 'x', 'y'],
                               """[[a1, a2],
                                   [a3, a4]]
                               """)}

DS = Generator.Vode_ODEsystem(DSargs)

# A function that tests the outcome of a given value of parameters or
# initial conditions (either is optional)
def test_ic(icdict=None, pardict=None):
    if icdict is not None:
        DS.set(ics=icdict)
    if pardict is not None:
        DS.set(pars=pardict)
    traj = DS.compute('test')
    pts = traj.sample()
    return pts

def do_plots(pts, fig_offset=0):
    plt.figure(1+fig_offset)
    plt.plot(pts['t'], pts['x'], 'g')
    plt.plot(pts['t'], pts['y'], 'r')
    plt.figure(2+fig_offset)
    plt.plot(pts['x'], pts['y'], 'k')

pts1 = test_ic()
do_plots(pts1)

# eps (epsilon) specifies the accuracy for the numerical
# algorithm that finds the fixed points. Read the docstring
# for more info!
fp_coords = find_fixedpoints(DS, n=4, eps=1e-6,
                             subdomain={'x': [-2,2],
                                        'y': [-2,2]})

print "%i fixed point(s) were found" % len(fp_coords)

# Find eigenvalues and vectors for the coordinates found
# Copy this format and adapt for your model's coordinate names
fp = fixedpoint_2D(DS, Point(fp_coords[0]), coords=['x', 'y'],
                       eps=1e-6)
print fp.point, '\n', fp.stability, fp.classification

plt.figure(3)

# A loop to test a distribution of initial conditions
for x0 in linspace(-1,1,10):
    pts = test_ic({'x': x0})
    plt.plot(pts['x'], pts['y'], 'b')

for y0 in linspace(1,-1,10):
    pts = test_ic({'y': y0})
    plt.plot(pts['x'], pts['y'], 'y')


for x0 in linspace(1,-1,10):
    pts = test_ic({'x': x0, 'y': -1})
    plt.plot(pts['x'], pts['y'], 'b')

for y0 in linspace(-1,1,10):
    pts = test_ic({'y': y0})
    plt.plot(pts['x'], pts['y'], 'y')


plt.plot(fp.point['x'], fp.point['y'], 'ko', markersize=8)

e1 = fp.evals[0]
e2 = fp.evals[1]

ev1 = fp.evecs[0]
ev2 = fp.evecs[1]

plt.plot([fp.point['x'], fp.point['x']+ev1['x']], [fp.point['y'], fp.point['y']+ev1['y']], 'k', linewidth=3)
plt.plot([fp.point['x'], fp.point['x']+ev2['x']], [fp.point['y'], fp.point['y']+ev2['y']], 'k', linewidth=3)


def get_real_sol(IC):
    """Return a pointset solution based on an explicit solution of the
    ODE when it has purely real eigenvalues
    """
    c1, c2 = np.dot(np.linalg.inv(array([ev1, ev2]).T), IC-fp.point)

    # define a local function that uses c1, c2, and some of the globals
    def real_sol(t):
        """Demonstration of explicit solution:
        Only works for non-oscillatory solutions
        with real eigenvalues!

        Accepts single scalars only
        """
        assert isreal(e1)
        assert isreal(e2)
        return c1*ev1*np.exp(e1*t) + c2*ev2*np.exp(e2*t) + fp.point

    # make times
    ts = linspace(0, 7, 500)
    # empty state points list
    vs = []
    for t in ts:
        # appends array for each t
        vs.append(real_sol(t))

    # change vs from list of arrays to 2 x N array
    # transposed so that vs[0] is all x values,
    # vs[1] is all y values
    vs = array(vs).T
    return Pointset(coorddict={'x': vs[0], 'y': vs[1]},
                    indepvararray=ts)


plt.figure(3)
sol = get_real_sol(Point({'x': 1, 'y': 0.8}))
plt.plot(sol['x'], sol['y'], 'm', linewidth=3)

#plt.show()

# ----------------------------------------------------------
# Explore different coupling parameters

print "\n\nRun through some matrix parameters..."

from time import sleep

plt.figure(4)
plt.xlabel('x')
plt.ylabel('y')

for a1 in linspace(-10, 8, 8):
    print "Testing a1 =", a1

    pts2 = test_ic(pardict={'a2': -8, 'a1': a1})
    fp_coords2 = find_fixedpoints(DS, n=4, eps=1e-6,
                             subdomain={'x': [-10,5],
                                        'y': [-5,5]})
    if len(fp_coords2) > 0:
        fp2 = fixedpoint_2D(DS, Point(fp_coords2[0]), coords=['x', 'y'],
                       eps=1e-6)
        if fp2.stability == 's':
            line = '-'
            fpcol = 'r'
            s = 'stable'
        else:
            # assume not the non-generic case of a 'center'
            line = '--'
            fpcol = 'g'
            s = 'unstable'
        print "a1: %.3f, f.p. coords: (%.3f, %.3f), %s %s" % (a1, fp2.point['x'],
                                                              fp2.point['y'], s, fp2.classification)
        print "   with e'vals: ", fp2.evals
        plt.plot(pts2['x'], pts2['y'], line, label='%.3f' % a1)
        plt.plot(fp2.point['x'], fp2.point['y'], fpcol+'o', markersize=8)
    else:
        print "a1: %.3f, (f.p. outside of domain)" % a1
        plt.plot(pts2['x'], pts2['y'], label='%.3f' % a1)

    plt.xlim([-15,5])
    plt.ylim([-2,2])
    #plt.draw()
    #sleep(2)

print "Finished"

#plt.legend(loc='upper left')

# limits can sometimes need refreshing
plt.xlim([-15,5])
plt.ylim([-2,2])
plt.show()


