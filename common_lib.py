from PyDSTool import pointset_to_traj, pointset_to_vars, Pointset, QuantSpec, makeSeqUnique
from PyDSTool import prepJacobian, expr2fun, remain, isincreasing
import copy
import numpy as np
from math import ceil

def make_noise_signal(dt, t_end, mean, stddev, num_cells, seed=None):
    """Helper function: Gaussian white noise at sample rate = dt for 1 or more cells,
    for a duration of t_end."""
    if seed is not None:
        np.random.seed(seed)
    N = ceil(t_end*1./dt)
    t = np.linspace(0, t_end, N)
    coorddict = {}
    for cellnum in range(num_cells):
        coorddict['noise%i' % (cellnum+1)] = np.random.normal(0, stddev, N)
    vpts = Pointset(coorddict=coorddict, indepvararray=t)
    return pointset_to_vars(vpts, discrete=False)


def make_spike_signal(ts, dur, tmax, loval=0, hival=1, dt=0.1):
    """Helper function: Square-pulse spike signal between two levels, loval
    and hival. Pulses occur at times given by ts array for duration given by
    dur scalar, from time 0 to tmax.

    Returns a single-entry dictionary of Variable objects with key 'Istim'.

    Default loval = 0, hival = 1.

    To improve performance with adaptive time-step solvers, three (two before,
    one after) points are added before and after each pulse, with a minimum
    step time given by dt.
    """
    assert len(ts) > 0, "No spike time events provided!"
    assert isincreasing(ts), "This function expects strictly increasing times"
    assert ts[0] != 0, "This function does not support initial step up at t=0"
    assert dur > dt, "Duration must be larger than dt"
    times = [0]
    vals = [loval]
    # check that ts are separated by at least dur+4*dt
    assert all(np.diff(ts) > dur+4*dt), "Separate events by at least 4*dt"
    assert tmax > ts[-1]+dur, "tmax must be larger than last event end time"
    for t in ts:
        times.extend([t-2.9*dt, t-dt,  t,     t+dur-dt, t+dur, t+dur+dt])
        vals.extend([loval,     loval, hival, hival,    hival, loval])
    if tmax > ts[-1]+dur+dt:
        times.append(tmax)
        vals.append(loval)
    coorddict = {'Istim': vals}
    vpts = Pointset(coorddict=coorddict, indepvararray=times)
    return pointset_to_vars(vpts, discrete=False)


def freq(traj):
    evs = traj.getEventTimes('thresh_ev')
    if len(evs) == 0:
        return 0
    elif len(evs) == 1:
        print "Not enough events found"
        return 0
    else:
        return 1000./(evs[-1] - evs[-2])

def amp(pts):
    return max(pts['v']) - min(pts['v'])


def pcw_protocol(DS, prot_list):
    """protocol list of dictionaries with any of the keys 'pars',
    'ics', or float 'tdur' (relative time duration of piece).

    Preserves events in the resulting Pointset.
    """
    orig_ics = DS.initialconditions.copy()
    t = 0
    for i, stage_dict in enumerate(prot_list):
        if 'pars' in stage_dict:
            DS.set(pars=stage_dict['pars'])
        if 'ics' in stage_dict:
            DS.set(ics=stage_dict['ics'])
        if 'tdata' in stage_dict:
            raise NotImplementedError("Replace 'tdata' with 'tdur' float value")
        if 'tdur' in stage_dict:
            DS.set(tdata=[t, t+stage_dict['tdur']])
        traj = DS.compute('test')
        if i == 0:
            pts = traj.sample()
            t = pts['t'][-1]
            DS.set(ics=pts[-1])
        else:
            new_pts = traj.sample()
            pts.extend(new_pts, skipMatchingIndepvar=True)
            t = new_pts['t'][-1]
            DS.set(ics=pts[-1])
    DS.set(ics=orig_ics)
    return pointset_to_traj(pts, events=DS.eventstruct.events), pts


def thresh_Naka_Rushton_fndef(N=2, half_on=120, max_val=100, with_if=True,
                              sys_pars=None):
    """Can specify strings or numbers for half_on and max_val arguments, in case of
    including parameters or other variables in the definitions.
    (Don't forget to declare those parameters, in that case.)

    Use the with_if=False case to ensure Jacobians can be calculated.
    Use sys_pars list to provide names of any model parameters or functions
      that will be declared elsewhere.
    """
    assert N == int(N), "Provide integer N"
    extra_pars = []
    if sys_pars is None:
        sys_pars = []
    if not isinstance(half_on, str):
        half_on = str(half_on)
    else:
        Q = QuantSpec('h', half_on)
        # don't add model system parameters to function parameter list
        extra_pars.extend(remain(Q.freeSymbols, sys_pars))
    if not isinstance(max_val, str):
        max_val = str(max_val)
    else:
        Q = QuantSpec('m', max_val)
        extra_pars.extend(remain(Q.freeSymbols, sys_pars))
    if with_if:
        return (['x']+extra_pars,
            'if(x>0,%s*pow(x,%i)/(pow(%s,%i) + pow(x,%i)),0)' % (max_val, N, half_on, N, N))
    else:
        return (['x']+extra_pars,
            '%s*pow(x,%i)/(pow(%s,%i) + pow(x,%i))' % (max_val, N, half_on, N, N))

def thresh_exp_fndef(sigma, half_on, max_val):
    return (['x'], '1/(1+exp((%f-x)/%f))' % (half_on, sigma))


def thresh_tanh_fndef(N, half_on, max_val):
    # not complete
    raise NotImplementedError("Class not complete!")
    assert N == int(N), "Provide integer N"
    return (['x'],
            'if(x > 0, ???, 0)' % (max_val, N, half_on))


def make_Jac(DS, varnames=None):
    if varnames is None:
        varnames = DS.funcspec.vars
    subdomain = {}
    fixedvars = remain(DS.funcspec.vars, varnames)
    for k in fixedvars:
        subdomain[k] = DS.initialconditions[k]

    jac, new_fnspecs = prepJacobian(DS.funcspec._initargs['varspecs'], varnames,
                                DS.funcspec._initargs['fnspecs'])

    scope = copy.copy(DS.pars)
    scope.update(subdomain)
    scope.update(new_fnspecs)
    return expr2fun(jac, ensure_args=['t'], **scope)
