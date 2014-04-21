"""
Sets up network of n spike-rate elements

dR_i/dt = (-R_i + S(__sum_of_inputs__)) / tau_i

where __sum_of_inputs__ is a sum of R_j's

See Wilson, Ch. 2.3, 6.4

Also has options to add other variables, e.g. for adaptation.

"""
from __future__ import division

from PyDSTool import *
from common_lib import *

# ---------------------------------
# Firing-rate based neural models

class rate_network(object):
    def __init__(self):
        self.vardefs = {}
        self.pardefs = {}
        self.fndefs = {}
        self.inputs = {}
        self.icdefs = {}

    def add_neuron(self, name, tau=1, ic=0, thresh_fn=None):
        """
        Add an ODE for a neuron's spiking rate variable
        Results in an ODE of the form  (-x + R(INPUTS))/ tau_x
           where x is the given name, and R is a Naka-Rushton
           or similar response function.
        """
        assert name not in self.vardefs
        Si = 'S_'+name
        if thresh_fn is None:
            thresh_fn = thresh_Naka_Rushton_fndef()
        thresh_args = thresh_fn[0]
        if len(thresh_args) > 1:
            # extra variables need to be passed
            extra_argstr = ', ' + ', '.join(thresh_args[1:])
        else:
            extra_argstr = ''
        self.fndefs[Si] = thresh_fn
        taui = 'tau_'+name
        self.vardefs[name] = '(-%s + %s(INPUTS%s))/%s' % (name, Si, extra_argstr, taui)
        self.pardefs[taui] = tau
        self.icdefs[name] = ic
        # make inputs later, using add_input_to_neuron method
        self.inputs[name] = []

    def add_rate(self, name, tau=1, ic=0):
        """
        Add an ODE for a non-neuron variable, which you will later
           add to a neuron's inputs using "add_interaction"
        Results in an ODE of the form  (-x + INPUTS)/ tau_x
           where x is the given name.
        """
        assert name not in self.vardefs
        if isinstance(tau, str):
            taui = tau
        else:
            taui = 'tau_'+name
            self.pardefs[taui] = tau
        self.vardefs[name] = '(-%s + INPUTS)/%s' % (name, taui)
        self.icdefs[name] = ic
        # make inputs later, using add_input_to_neuron method
        self.inputs[name] = []

    def add_interaction(self, source, dest, g, g_name=None):
        """
        Add a generic input from one variable to another.
        Results in a term  g * source  in the destination ODE.
        Arguments:
          source = variable name of source (pre-synaptic) neuron
          dest   = variable name of destination (post-synaptic) neuron
          g      = coupling strength parameter value, i.e. any real number
        """
        if g_name is None:
            g_name = 'g_%s_%s' % (source, dest)
        assert g_name not in self.pardefs
        if g_name not in self.vardefs:
            self.pardefs[g_name] = g
        self.inputs[dest].append( g_name + '*'  + source )

    def add_syn_input_to_neuron(self, source, dest, g, g_name=None):
        """
        Add a synapse-style input to a neuron from another neuron. T
        Arguments:
          source = variable name of source (pre-synaptic) neuron
          dest   = variable name of destination (post-synaptic) neuron
          g      = coupling strength (gain factor) parameter value,
                     which can be positive or negative
        """
        if g_name is None:
            g_name = 'g_%s_%s' % (source, dest)
        assert g_name not in self.pardefs
        self.pardefs[g_name] = g
        self.inputs[dest].append( g_name + '*'  + source )

    def add_bias_input(self, dest, p, p_name):
        """
        Add a bias input to the destination:
        Arguments:
          dest   = variable name of input destination neuron
          p      = value of the input (see below)
          p_name = (string) name of the parameter
        The bias input can be a constant or it can be a function call.
        """
        if '(' not in p_name:
            # assume a parameter rather than a time-dependent function or constant scalar
            assert p_name not in self.pardefs
            self.pardefs[p_name] = p
        self.inputs[dest].append(p_name)

    def make_network(self, network_name='net', events=None, gentype='Vode'):
        """
        Make Vode (default) Generator object for network, with optional events
        structure pre-created.
        """
        DSargs = args(name=network_name)
        varspecs = {}
        for xname, xdef in self.vardefs.items():
            input_list = self.inputs[xname]
            if len(input_list) > 0:
                varspecs[xname] = xdef.replace('INPUTS', '+'.join(input_list))
            else:
                varspecs[xname] = xdef.replace('INPUTS', '0')
        DSargs.varspecs = varspecs
        DSargs.pars = self.pardefs
        DSargs.fnspecs = self.fndefs
        DSargs.ics = self.icdefs
        DSargs.tdomain = [0, 100000]
        if events is not None:
            DSargs.events = events
        if gentype == 'Vode':
            return Generator.Vode_ODEsystem(DSargs)
        elif gentype == 'Dopri':
            return Generator.Dopri_ODEsystem(DSargs)
        else:
            raise NotImplementedError("Choose gentype argument to be 'Vode' or 'Dopri'")


