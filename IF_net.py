# Example of IF
from __future__ import division
from PyDSTool import *
from common_lib import *

print "TO DO: Put in function and aux var for I_Esyn and I_Isyn so that PSCs can"
print "be seen, and put in shunting version\n\n"

# ---------------------------------------------
# Leaky Integrate-and-Fire based neural models

### Post-synaptic responses truncated after t_decay
### (assumed to be small)
##global t_decay
##t_decay = 10 # ms
##
### Max number of simultaneous inputs
##global n_inputs
##n_inputs = 10

def make_single_IF(gl, vl, C, I, E_spike_times, I_spike_times,
                   vthreshold=-55, vreset=-80, vic=-70):
    """Vode-specific, because of the way it looks up information in
    python functions.
    """
    leak_RHS = '(gl*(vl-v) + I + gE*( syn_exp((t_Epre_0-globalindepvar(t))/tau_Esyn) + syn_exp((t_Epre_1-globalindepvar(t))/tau_Esyn) ) ' + \
                  '- gI*( syn_exp((t_Ipre_0-globalindepvar(t))/tau_Isyn) + syn_exp((t_Ipre_1-globalindepvar(t))/tau_Isyn) ))/C'
    leak_event_args = {'name': 'thresh',
                   'eventtol': 1e-6,
                   'eventdelay': 1e-8,
                   'starttime': 0,
                   'active': True,
                   'term': True,
                   'precise': True}
    leak_thresh_ev = Events.makeZeroCrossEvent('v-(%f)' % vthreshold,
                                               1, leak_event_args,
                                               varnames = ['v'])
    spike_E_ev_args0 = {'name': 'spike_E0',
                     'eventtol': 1e-5,
                   'eventdelay': 1e-7,
                   'starttime': 0,
                   'active': True,
                   'term': True,
                   'precise': True}
    spike_E_ev_args1 = {'name': 'spike_E1',
                     'eventtol': 1e-5,
                   'eventdelay': 1e-7,
                   'starttime': 0,
                   'active': True,
                   'term': True,
                   'precise': True}

    spike_E_ev0 = Events.makeZeroCrossEvent('globalindepvar(t) - t_Epre_0',
                                               1, spike_E_ev_args0,
                                               parnames=['t_Epre_0'])

    spike_E_ev1 = Events.makeZeroCrossEvent('globalindepvar(t) - t_Epre_1',
                                               1, spike_E_ev_args1,
                                               parnames=['t_Epre_1'])

    spike_I_ev_args0 = {'name': 'spike_I0',
                     'eventtol': 1e-5,
                   'eventdelay': 1e-7,
                   'starttime': 0,
                   'active': True,
                   'term': True,
                   'precise': True}
    spike_I_ev_args1 = {'name': 'spike_I1',
                     'eventtol': 1e-5,
                   'eventdelay': 1e-7,
                   'starttime': 0,
                   'active': True,
                   'term': True,
                   'precise': True}

    spike_I_ev0 = Events.makeZeroCrossEvent('globalindepvar(t) - t_Ipre_0',
                                               1, spike_I_ev_args0,
                                               parnames=['t_Ipre_0'])

    spike_I_ev1 = Events.makeZeroCrossEvent('globalindepvar(t) - t_Ipre_1',
                                               1, spike_I_ev_args1,
                                               parnames=['t_Ipre_1'])

    assert vthreshold > vreset
    assert vreset > -200

    # inside gen / epmapping
    # set next event time in a special method
    def _check_Eevs(ds, t):
        # just have two possible simultaneous responses
        if ds._current_E_ix == 0:
            ds._current_E_ix = 1
        else:
            ds._current_E_ix = 0
        try:
            next_t = ds._Espike_list.pop()
        except IndexError:
            # no spikes left
            return None, None
        else:
            #print "Next E time for ix %i is %.5f" % (ds._current_E_ix, next_t)
            return next_t, 't_Epre_%i' % ds._current_E_ix

    def _check_Ievs(ds, t):
        # just have two possible simultaneous responses
        if ds._current_I_ix == 0:
            ds._current_I_ix = 1
        else:
            ds._current_I_ix = 0
        try:
            next_t = ds._Ispike_list.pop()
        except IndexError:
            # no spikes left
            return None, None
        else:
            #print "Next I time for ix %i is %.5f" % (ds._current_I_ix, next_t)
            return next_t, 't_Ipre_%i' % ds._current_I_ix

    leak_args = {'pars': {'C': C, 'vl': vl, 'gl': gl, 'vreset': vreset,
                          'I': I, 'tau_Esyn': 3, 'tau_Isyn': 4,
                          'gE': 10, 'gI': 5,
                          't_Epre_0': -10000, 't_Epre_1': -10000,
                          't_Ipre_0': -10000, 't_Ipre_1': -10000},
              'xdomain': {'v': [-200, vthreshold]},
              'varspecs': {'v': leak_RHS},
              'algparams': {'init_step': 0.2,
                            'specialtimes': [0.01, 0.02], # these ensure good behavior after spike
                            'use_special': True},
              'events': [leak_thresh_ev, spike_E_ev0, spike_E_ev1,
                         spike_I_ev0, spike_I_ev1],
              'abseps': 1e-7,
              'name': 'IF',
#              'checklevel': 2,
              'fnspecs': {'syn_exp': (['x'], 'if(x<0, exp(x), 0)')},
#              'vfcodeinsert_start': 'rE = ds.something(smth)',
#              'ignorespecial': ['rE']
              }

    gen = Generator.Vode_ODEsystem(leak_args)
    DS = embed(gen, tdata=[0, 100])

    epmapping_inputE = EvMapping(defString="""result, parname = self._check_Eevs(self._ds, t)
if not(result is None):
  pdict[parname] = result""", model=DS)
    epmapping_inputE._ds = DS
    epmapping_inputE._check_Eevs = _check_Eevs
    # special stuff for IF
    DS._current_E_ix = 0
    DS._Espike_list = E_spike_times[::-1]  # reverse order, for popping
    # initialize event parameters
    DS.set(pars={'t_Epre_0': DS._Espike_list.pop()})

    epmapping_inputI = EvMapping(defString="""result, parname = self._check_Ievs(self._ds, t)
if not(result is None):
  pdict[parname] = result""", model=DS)
    epmapping_inputI._ds = DS
    epmapping_inputI._check_Ievs = _check_Ievs
    # special stuff for IF
    DS._current_I_ix = 0
    DS._Ispike_list = I_spike_times[::-1]  # reverse order, for popping
    # initialize event parameters
    DS.set(pars={'t_Ipre_0': DS._Ispike_list.pop()})

    epmapping_V = EvMapping({'v': 'vreset'}, model=DS)
    DS_info = makeModelInfoEntry(intModelInterface(DS), ['IF'],
                                 [('thresh', ('IF', epmapping_V)),
                                  ('spike_E0', ('IF', epmapping_inputE)),
                                  ('spike_E1', ('IF', epmapping_inputE)),
                                  ('spike_I0', ('IF', epmapping_inputI)),
                                  ('spike_I1', ('IF', epmapping_inputI))])
    modelInfoDict = makeModelInfo([DS_info])
    return Model.HybridModel({'name': 'IF_model',
                              'ics': {'v': vic},
                              'modelInfo': modelInfoDict})



##def make_IF(vname, v0, tau, vthreshold=-40, vreset=-100, vic=-70):
##    """Vode-specific, because of the way it looks up stuff in python functions
##    """
##    v0p = 'v0_'+vname
##    taup = 'tau_'+vname
##    leak_RHS = '(' + v0p + '-' + vname + '+ INPUTS)/' + taup
##    leak_event_args = {'name': 'thresh_'+vname,
##                   'eventtol': 1e-6,
##                   'eventdelay': 1e-8,
##                   'starttime': 0,
##                   'active': True,
##                   'term': True,
##                   'precise': True}
##    leak_thresh_ev = Events.makeZeroCrossEvent(vname + '+%f' % vthreshold,
##                                               1, leak_event_args,
##                                               varnames = [vname])
##
##    assert vthreshold > vreset
##    assert vreset > -200
##
##
##    leak_args = {'pars': {taup: tau, v0p: v0, 'vreset': vreset,
##                          'tau_Esyn': 2,
##                          'tpre': -10},  # INPUTS!
##              'xdomain': {vname: [-200, vthreshold]},
##              'varspecs': {vname: leak_RHS},
##              'algparams': {'init_step': 0.5},
##              'events': leak_thresh_ev,
##              'abseps': 1.e-7,
###              'fnspecs': {'I_Esyn': ([], 'exp((tpre-t)/tau_Esyn)')},
##              'name': 'IF_net'}
##
##    ics = {vname: vic}
##
##    DS = embed(Generator.Vode_ODEsystem(leak_args), icdict=ics, tdata=[0, 100])
##    DS_MI = intModelInterface(DS)
##
##    epmapping = EvMapping({vname: "vreset"}, model=DS)
##    DS_info = makeModelInfoEntry(DS_MI, ['IF_net'],
##                                 [('time', ('leak', epmapping))])


# ----------------------------

##class IF_network(object):
##    def __init__(self):
##        self.vardefs = {}
##        self.pardefs = {}
##        self.fndefs = {}
##        self.inputs = {}
##        self.icdefs = {}
##
##    def add_neuron(self, name, tau=1, ic=0):
##        assert name not in self.vardefs
##        self.fndefs[Si] = thresh_fn
##        taui = 'tau_'+name
##        self.vardefs[name] = '(-%s + %s(INPUTS%s))/%s' % (name, Si, extra_argstr, taui)
##        self.pardefs[taui] = tau
##        self.icdefs[name] = ic
##        # make inputs later, using add_input_to_neuron method
##        self.inputs[name] = []
##
##    def add_rate(self, name, tau=1, ic=0):
##        assert name not in self.vardefs
##        taui = 'tau_'+name
##        self.vardefs[name] = '(-%s + INPUTS)/%s' % (name, taui)
##        self.pardefs[taui] = tau
##        self.icdefs[name] = ic
##        # make inputs later, using add_input_to_neuron method
##        self.inputs[name] = []
##
##    def add_interaction(self, source, dest, g, g_name=None):
##        if g_name is None:
##            g_name = 'g_%s_%s' % (source, dest)
##        assert g_name not in self.pardefs
##        self.pardefs[g_name] = g
##        self.inputs[dest].append( g_name + '*'  + source )
##
##    def add_syn_input_to_neuron(self, source, dest, g, g_name=None):
##        if g_name is None:
##            g_name = 'g_%s_%s' % (source, dest)
##        assert g_name not in self.pardefs
##        self.pardefs[g_name] = g
##        self.inputs[dest].append( g_name + '*'  + source )
##
##    def add_bias_input(self, dest, p, p_name):
##        assert p_name not in self.pardefs
##        self.pardefs[p_name] = p
##        self.inputs[dest].append(p_name)
##
##    def make_network(self, network_name='net'):
##        # assume ICs are 0
##        DSargs = args(name=network_name)
##        varspecs = {}
##        for xname, xdef in self.vardefs.items():
##            input_list = self.inputs[xname]
##            if len(input_list) > 0:
##                varspecs[xname] = xdef.replace('INPUTS', '+'.join(input_list))
##            else:
##                varspecs[xname] = xdef.replace('INPUTS', '0')
##        DSargs.varspecs = varspecs
##        DSargs.pars = self.pardefs
##        DSargs.fnspecs = self.fndefs
##        DSargs.ics = self.icdefs
##        DSargs.tdomain = [0, 100000]
##        return Generator.Vode_ODEsystem(DSargs)

def check_input_times(pts):
    Es = array(list(pts.bylabel('Event:spike_E0')['t']) + \
               list(pts.bylabel('Event:spike_E1')['t']))
    Es.sort()
    Is = array(list(pts.bylabel('Event:spike_I0')['t']) + \
               list(pts.bylabel('Event:spike_I1')['t']))
    Is.sort()
    return Es, Is


def get_spike_times(pts):
    thresh_pts = pts.bylabel('Event:thresh')
    return thresh_pts['t']


def plot_IF(pts):
    thresh_pts = pts.bylabel('Event:thresh')
    plt.plot(pts['t'], pts['v'], 'k', linewidth=1)
    plt.plot([thresh_pts['t'], thresh_pts['t']], [-55, 0], 'ko-', linewidth=1.75)
    plt.ylim([-85,5])

IF = make_single_IF(0.1, -65, 1, 3,
                    [6.2, 10, 17.1, 28, 45, 46],
                    [6.1, 12, 17.1002, 25, 50, 51])

IF.set(tdata=[0, 80],
        algparams={'init_step': 0.04})
IF.compute('test')
pts = IF.sample('test')

plot_IF(pts)


plt.show()
