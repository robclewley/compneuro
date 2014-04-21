"""
This module holds the methods used to build a Model/Generator for a Hodgkin-
Huxley model of a neuron.  The user can build custom neurons by writing a config
file and importing it through the main run file.

This is a simplified version of the fovea file of the same name.
"""

from __future__ import division
from numpy import *

from PyDSTool import *
from PyDSTool.Toolbox.neuralcomp import *

from PyDSTool import ModelManager

global man


def make_IN_model(ic_args, alg_args, targetGen, with_spike_ev=False):
    """Type I excitability.
    """
    v = Var(voltage)
    ma = 0.32*(v+54.)/(1-Exp(-(v+54.)/4))
    mb = 0.28*(v+27.)/(Exp((v+27.)/5)-1)
    ha = .128*Exp(-(50.+v)/18)
    hb = 4/(1+Exp(-(v+27.)/5))
    channel_Na1 = makeChannel_rates('Na', voltage, 'm', False, ma, mb, 3,
                                    'h', False, ha, hb, 1, vrev=50, g=100,
                                    noauxs=False, subclass=channel_on,
                                    gamma1={voltage: ('m','h'),
                                            'm': (voltage,),
                                            'h': (voltage,)})

    na = .032*(v+52)/(1-Exp(-(v+52.)/5))
    nb = .5*Exp(-(57.+v)/40)
    channel_K1 = makeChannel_rates('K', voltage, 'n', False, na, nb, 4,
                                   vrev=-100,
                                   g=80, noauxs=False, subclass=channel_on,
                                   gamma1={voltage: ('n',),
                                           'n': (voltage,)})

    channel_Ib1 = makeBiasChannel('Ib', 2, noauxs=False, subclass=channel_on,
                                  gamma2={voltage: ('Ibias',)})
    channel_Lk1 = makeChannel_rates('Lk', vrev=-67, g=0.1, noauxs=False,
                                  subclass=channel_on,
                                  gamma1={voltage: ('Lk',)})

    return instantiate(ic_args, alg_args, [channel_Lk1, channel_Ib1,
                       channel_Na1, channel_K1], 'IN_typeI', targetGen,
                       with_spike_ev)


def make_IN_model_no_h(ic_args, alg_args, targetGen, with_spike_ev=False):
    """Type I excitability with no h term.
    For spike onset only!
    """
    v = Var(voltage)
    ma = 0.32*(v+54.)/(1-Exp(-(v+54.)/4))
    mb = 0.28*(v+27.)/(Exp((v+27.)/5)-1)
    channel_Na1 = makeChannel_rates('Na', voltage, 'm', False, ma, mb, 3,
                                    vrev=50, g=100,
                                    noauxs=False, subclass=channel_on,
                                    gamma1={voltage: ('m',),
                                            'm': (voltage,)})

    na = .032*(v+52)/(1-Exp(-(v+52.)/5))
    nb = .5*Exp(-(57.+v)/40)
    channel_K1 = makeChannel_rates('K', voltage, 'n', False, na, nb, 4,
                                   vrev=-100,
                                   g=80, noauxs=False, subclass=channel_on,
                                   gamma1={voltage: ('n',),
                                           'n': (voltage,)})

    channel_Ib1 = makeBiasChannel('Ib', 2, noauxs=False, subclass=channel_on,
                                  gamma2={voltage: ('Ibias',)})
    channel_Lk1 = makeChannel_rates('Lk', vrev=-67, g=0.1, noauxs=False,
                                  subclass=channel_on,
                                  gamma1={voltage: ('Lk',)})

    return instantiate(ic_args, alg_args, [channel_Lk1, channel_Ib1,
                       channel_Na1, channel_K1], 'IN_typeI_no_h', targetGen,
                       with_spike_ev)


def make_HH_model(ic_args, alg_args, targetGen, with_spike_ev=False):
    """Classic Type II excitability"""
    v = Var(voltage)
    ma = 0.1*(v+40)/(1-Exp(-(v+40)/10))
    mb = 4*Exp(-(v+65)/18)
    ha = .07*Exp(-(v+65)/20)
    hb = 1/(1+Exp(-(v+35)/10)) #0.01*(v+55)/(1-Exp(-(v+55)/10))
    channel_Na1 = makeChannel_rates('Na', voltage, 'm', False, ma, mb, 3,
                                    'h', False, ha, hb, 1, vrev=50, g=120,
                                    noauxs=False, subclass=channel_on,
                                    gamma1={voltage: ('m','h'),
                                            'm': (voltage,),
                                            'h': (voltage,)})

    na = .01*(v+55)/(1-Exp(-(v+55)/10))
    nb = .125*Exp(-(v+65)/80)
    channel_K1 = makeChannel_rates('K', voltage, 'n', False, na, nb, 4, vrev=-77,
                                   g=36, noauxs=False, subclass=channel_on,
                                   gamma1={voltage: ('n',),
                                           'n': (voltage,)})

    channel_Ib1 = makeBiasChannel('Ib', 8, noauxs=False, subclass=channel_on,
                                  gamma2={voltage: ('Ibias',)})
    channel_Lk1 = makeChannel_rates('Lk', vrev=-54.4, g=0.3, noauxs=False,
                                    subclass=channel_on,
                                    gamma1={voltage: ('Lk',)})

    return instantiate(ic_args, alg_args, [channel_Lk1, channel_Ib1,
                       channel_Na1, channel_K1], 'classic_typeII', targetGen,
                       with_spike_ev)


def make_HH_model_no_h(ic_args, alg_args, targetGen, with_spike_ev=False):
    """Classic type II excitability with no h term in Na.
    Used for spike onsets only!"""
    v = Var(voltage)
    ma = 0.1*(v+40)/(1-Exp(-(v+40)/10))
    mb = 4*Exp(-(v+65)/18)
    channel_Na1 = makeChannel_rates('Na', voltage, 'm', False, ma, mb, 3,
                                    vrev=50, g=120,
                                    noauxs=False, subclass=channel_on,
                                    gamma1={voltage: ('m',),
                                            'm': (voltage,)
                                            })

    na = .01*(v+55)/(1-Exp(-(v+55)/10))
    nb = .125*Exp(-(v+65)/80)
    channel_K1 = makeChannel_rates('K', voltage, 'n', False, na, nb, 4, vrev=-77,
                                   g=36, noauxs=False, subclass=channel_on,
                                   gamma1={voltage: ('n',),
                                           'n': (voltage,)})

    channel_Ib1 = makeBiasChannel('Ib', 8, noauxs=False, subclass=channel_on,
                                  gamma2={voltage: ('Ibias',)})
    channel_Lk1 = makeChannel_rates('Lk', vrev=-54.4, g=0.3, noauxs=False,
                                    subclass=channel_on,
                                    gamma1={voltage: ('Lk',)})

    return instantiate(ic_args, alg_args, [channel_Lk1, channel_Ib1,
                       channel_Na1, channel_K1], 'classic_typeII_no_h', targetGen,
                       with_spike_ev)


def make_HH_model_fast_m_no_h(ic_args, alg_args, targetGen, with_spike_ev=False):
    """Classic type II excitability with no h term in Na, fast m dynamics,
    and a constant time-scale K channel.
    Used for spike onsets only!"""
    v = Var(voltage)
    ma = 0.1*(v+40)/(1-Exp(-(v+40)/10))
    mb = 4*Exp(-(v+65)/18)
    channel_Na1 = makeChannel_rates('Na', voltage, 'm', True, ma, mb, 3,
                                    vrev=50, g=120,
                                    noauxs=False, subclass=channel_on,
                                    gamma1={voltage: ('m',),
                                            'm': (voltage,)
                                            })

    na = .01*(v+55)/(1-Exp(-(v+55)/10))
    nb = .125*Exp(-(v+65)/80)
    ninf = na/(na+nb)
    channel_K1 = makeChannel_halfact('K', voltage, 'n', False, ninf, 'taun_par', 4, vrev=-77,
                                   g=36, noauxs=False, subclass=channel_on,
                                   parlist=[Par('5', 'taun_par')],
                                   gamma1={voltage: ('n',),
                                           'n': (voltage,)})

    channel_Ib1 = makeBiasChannel('Ib', 8, noauxs=False, subclass=channel_on,
                                  gamma2={voltage: ('Ibias',)})
    channel_Lk1 = makeChannel_rates('Lk', vrev=-54.4, g=0.3, noauxs=False,
                                    subclass=channel_on,
                                    gamma1={voltage: ('Lk',)})

    return instantiate(ic_args, alg_args, [channel_Lk1, channel_Ib1,
                       channel_Na1, channel_K1], 'classic_typeII_fast_m_no_h', targetGen,
                       with_spike_ev)




def make_HH_morepars_model_no_h(ic_args, alg_args, targetGen, with_spike_ev=False):
    """Classic type II excitability with no h term in Na.
    Used to spike onsets only!"""
    v = Var(voltage)
    ma = 0.1*(v+40)/(1-Exp(-(v+40)/10))
    mb = 4*Exp(-(v+65)/18)
    channel_Na1 = makeChannel_rates('Na', voltage, 'm', False, ma, mb, 3,
                                    vrev=50, g=120,
                                    noauxs=False, subclass=channel_on,
                                    gamma1={voltage: ('m',),
                                            'm': (voltage,)
                                            })

    ca = Par('0.01', 'ca')
    cb = Par('0.125', 'cb')
    aq = Par('10', 'aq')
    bq = Par('80', 'bq')
    va = Par('-55', 'va')
    vb = Par('-65', 'vb')
    na = ca*(v-va)/(1-Exp((va-v)/aq))
    nb = cb*Exp((vb-v)/bq)
    channel_K1 = makeChannel_rates('K', voltage, 'n', False, na, nb, 4, vrev=-77,
                                   g=36, noauxs=False, subclass=channel_on,
                                   parlist=[ca,cb,aq,bq,va,vb],
                                   gamma1={voltage: ('n',),
                                           'n': (voltage,)})

    channel_Ib1 = makeBiasChannel('Ib', 8, noauxs=False, subclass=channel_on,
                                  gamma2={voltage: ('Ibias',)})
    channel_Lk1 = makeChannel_rates('Lk', vrev=-54.4, g=0.3, noauxs=False,
                                    subclass=channel_on,
                                    gamma1={voltage: ('Lk',)})

    return instantiate(ic_args, alg_args, [channel_Lk1, channel_Ib1,
                       channel_Na1, channel_K1], 'classic_typeII_morepars_no_h', targetGen,
                       with_spike_ev)


def make_WB_model(ic_args, alg_args, targetGen, with_spike_ev=False):
    """Wang-Buzsaki Type I interneuron model"""
    phi = 5
    v = Var(voltage)
    ma = 0.1*(v+35)/(1-Exp(-(v+35)/10))
    mb = 4*Exp(-(v+60)/18)
    ha = phi*.07*Exp(-(v+58)/20)
    hb = phi*1/(1+Exp(-(v+28)/10))
    # isinstant=True for m
    channel_Na1 = makeChannel_rates('Na', voltage, 'm', True, ma, mb, 3,
                                    'h', False, ha, hb, 1, vrev=55, g=35,
                                    noauxs=False, subclass=channel_on,
                                    gamma1={voltage: ('m','h'),
                                            'm': (voltage,),
                                            'h': (voltage,)})

    na = phi*.01*(v+34)/(1-Exp(-(v+34)/10))
    nb = phi*.125*Exp(-(v+44)/80)
    channel_K1 = makeChannel_rates('K', voltage, 'n', False, na, nb, 4, vrev=-90,
                                   g=9, noauxs=False, subclass=channel_on,
                                   gamma1={voltage: ('n',),
                                           'n': (voltage,)})

    channel_Ib1 = makeBiasChannel('Ib', 2.5, noauxs=False, subclass=channel_on,
                                  gamma2={voltage: ('Ibias',)})
    channel_Lk1 = makeChannel_rates('Lk', vrev=-65, g=0.1, noauxs=False,
                                    subclass=channel_on,
                                    gamma1={voltage: ('Lk',)})

    return instantiate(ic_args, alg_args, [channel_Lk1, channel_Ib1,
                       channel_Na1, channel_K1], 'WB_typeI', targetGen,
                       with_spike_ev)


def make_WB_model_no_h(ic_args, alg_args, targetGen, with_spike_ev=False):
    """Wang-Buzsaki Type I interneuron model with no h term in Na.
    Used to spike onsets only!"""
    phi = 5
    v = Var(voltage)
    ma = 0.1*(v+35)/(1-Exp(-(v+35)/10))
    mb = 4*Exp(-(v+60)/18)
    # isinstant=True for m
    channel_Na1 = makeChannel_rates('Na', voltage, 'm', True, ma, mb, 3,
                                    vrev=55, g=35,
                                    noauxs=False, subclass=channel_on,
                                    gamma1={voltage: ('m',),
                                            'm': (voltage,)})

    na = phi*.01*(v+34)/(1-Exp(-(v+34)/10))
    nb = phi*.125*Exp(-(v+44)/80)
    channel_K1 = makeChannel_rates('K', voltage, 'n', False, na, nb, 4, vrev=-90,
                                   g=9, noauxs=False, subclass=channel_on,
                                   gamma1={voltage: ('n',),
                                           'n': (voltage,)})

    channel_Ib1 = makeBiasChannel('Ib', 2.5, noauxs=False, subclass=channel_on,
                                  gamma2={voltage: ('Ibias',)})
    channel_Lk1 = makeChannel_rates('Lk', vrev=-65, g=0.1, noauxs=False,
                                    subclass=channel_on,
                                    gamma1={voltage: ('Lk',)})

    return instantiate(ic_args, alg_args, [channel_Lk1, channel_Ib1,
                       channel_Na1, channel_K1], 'WB_typeI_no_h', targetGen,
                       with_spike_ev)


def make_WB_model_minflin_no_h(ic_args, alg_args, targetGen, with_spike_ev=False):
    """Wang-Buzsaki Type I interneuron model with no h term in Na,
    and a linear minf(V).
    Used for spike onsets only!"""
    phi = 5
    v = Var(voltage)
    v0 = -65.1
    m0 = 0.0285
    slope = 0.0035
    minf_quant = slope*(v-v0)+m0
    minf = 'max(0, %s)' % str(minf_quant)
    channel_Na1 = makeChannel_halfact('Na', voltage, 'm', True, minf, spow=3,
                                    vrev=55, g=35,
                                    noauxs=False, subclass=channel_on,
                                    gamma1={voltage: ('m',),
                                            'm': (voltage,)})

    na = phi*.01*(v+34)/(1-Exp(-(v+34)/10))
    nb = phi*.125*Exp(-(v+44)/80)
    channel_K1 = makeChannel_rates('K', voltage, 'n', False, na, nb, 4, vrev=-90,
                                   g=9, noauxs=False, subclass=channel_on,
                                   gamma1={voltage: ('n',),
                                           'n': (voltage,)})

    channel_Ib1 = makeBiasChannel('Ib', 2.5, noauxs=False, subclass=channel_on,
                                  gamma2={voltage: ('Ibias',)})
    channel_Lk1 = makeChannel_rates('Lk', vrev=-65, g=0.1, noauxs=False,
                                    subclass=channel_on,
                                    gamma1={voltage: ('Lk',)})

    return instantiate(ic_args, alg_args, [channel_Lk1, channel_Ib1,
                       channel_Na1, channel_K1], 'WB_typeI_minflin_no_h', targetGen,
                       with_spike_ev)


def make_IN_morepars_model(ic_args, alg_args, targetGen, with_spike_ev=False):
    """Type I HH model with kinetics pars for K channel"""
    v = Var(voltage)
    ca = Par('0.032', 'ca')
    cb = Par('0.5', 'cb')
    aq = Par('5', 'aq')
    bq = Par('40', 'bq')
    va = Par('-52', 'va')
    vb = Par('-57', 'vb')
    sw_h = Par('1', 'sw_h')
    ma = 0.32*(v+54.)/(1-Exp(-(v+54.)/4))
    mb = 0.28*(v+27.)/(Exp((v+27.)/5)-1)
    ha = sw_h*.128*Exp(-(50.+v)/18)
    hb = 4/(1+Exp(-(v+27.)/5))
    channel_Na1 = makeChannel_rates('Na', voltage, 'm', False, ma, mb, 3,
                                    'h', False, ha, hb, 1, vrev=50, g=100,
                                    noauxs=False, subclass=channel_on,
                                    parlist=[sw_h],
                                    gamma1={voltage: ('m','h'),
                                            'm': (voltage,),
                                            'h': (voltage,)})

    na = ca*(v-va)/(1-Exp((va-v)/aq))
    nb = cb*Exp((v-vb)/bq)
    channel_K1 = makeChannel_rates('K', voltage, 'n', False, na, nb, 4, vrev=-100,
                                   g=80, noauxs=False, subclass=channel_on,
                                   parlist=[ca,cb,aq,bq,va,vb],
                                   gamma1={voltage: ('n',),
                                           'n': (voltage,)})

    channel_Ib1 = makeBiasChannel('Ib', 2.1, noauxs=False, subclass=channel_on,
                                  gamma2={voltage: ('Ibias',)})
    channel_Lk1 = makeChannel_rates('Lk', vrev=-67, g=0.1, noauxs=False,
                                    subclass=channel_on, gamma1={voltage: ('Lk',)})

    return instantiate(ic_args, alg_args, [channel_Lk1, channel_Ib1,
                       channel_Na1, channel_K1], 'typeI_morepars', targetGen, with_spike_ev)


def make_IN_morepars_model_no_h(ic_args, alg_args, targetGen, with_spike_ev=False):
    """Type I HH model with kinetics pars for K channel.
    h removed to study spike onset.
    """
    v = Var(voltage)
    ca = Par('0.032', 'ca')
    cb = Par('0.5', 'cb')
    aq = Par('5', 'aq')
    bq = Par('40', 'bq')
    va = Par('-52', 'va')
    vb = Par('-57', 'vb')
    ma = 0.32*(v+54.)/(1-Exp(-(v+54.)/4))
    mb = 0.28*(v+27.)/(Exp((v+27.)/5)-1)
    channel_Na1 = makeChannel_rates('Na', voltage, 'm', False, ma, mb, 3,
                                    vrev=50, g=100,
                                    noauxs=False, subclass=channel_on,
                                    gamma1={voltage: ('m',),
                                            'm': (voltage,)})

    na = ca*(v-va)/(1-Exp((va-v)/aq))
    nb = cb*Exp((vb-v)/bq)
    channel_K1 = makeChannel_rates('K', voltage, 'n', False, na, nb, 4, vrev=-100,
                                   g=80, noauxs=False, subclass=channel_on,
                                   parlist=[ca,cb,aq,bq,va,vb],
                                   gamma1={voltage: ('n',),
                                           'n': (voltage,)})

    channel_Ib1 = makeBiasChannel('Ib', 2.1, noauxs=False, subclass=channel_on,
                                  gamma2={voltage: ('Ibias',)})
    channel_Lk1 = makeChannel_rates('Lk', vrev=-67, g=0.1, noauxs=False,
                                    subclass=channel_on, gamma1={voltage: ('Lk',)})

    return instantiate(ic_args, alg_args, [channel_Lk1, channel_Ib1,
                       channel_Na1, channel_K1], 'typeI_morepars_no_h',
                       targetGen, with_spike_ev)


make_model = {'HH_classic_typeII': make_HH_model,
 'HH_classic_typeII_no_h': make_HH_model_no_h,
 'HH_classic_typeII_fast_m_no_h': make_HH_model_fast_m_no_h,
 'HH_classic_typeII_morepars_no_h': make_HH_morepars_model_no_h,
 'HH_WB_typeI': make_WB_model,
 'HH_WB_typeI_no_h': make_WB_model_no_h,
 'HH_WB_typeI_minflin_no_h': make_WB_model_minflin_no_h,
 'HH_IN_typeI': make_IN_model,
 'HH_IN_typeI_no_h': make_IN_model_no_h,
 'HH_IN_typeI_morepars': make_IN_morepars_model,
 'HH_IN_typeI_morepars_no_h': make_IN_morepars_model_no_h}


def instantiate(ic_args, alg_args, channel_list, name, targetGen, with_spike_ev=False,
                withJac=False):
    """Presently, cannot use the Jacobian functionality with instantaneous activations.
    """
    global man
    man = ModelManager('HH_nullc_proj')

    if targetGen == 'Vode_ODEsystem':
        targetlang='python'
    else:
        targetlang='c'

    soma_name = 'cell_'+name

    HHcell = makeSoma(soma_name, channelList=channel_list, C=1.0, noauxs=False,
                            channelclass=channel_on)

    desc = GDescriptor(modelspec=HHcell, description='HH cell',
                       target=targetGen, algparams=alg_args)
    test = desc.validate()
    if not test[0]:
        print test[1]
        raise AssertionError
    assert desc.isinstantiable()

    # build an event that picks out when RHS of cell1's Voltage eqn is 0
    # i.e. when dV/dt=0, among others
    if with_spike_ev:
        HHcell.add(Par(0, 'V_th'))
        HHcell.flattenSpec()

        max_ev_args = {'name': 'cell_max',
                        'eventtol': 1e-8,
                        'eventdelay': 1e-3,
                        'starttime': 0,
                        'term': False
                        }
        # stationary event => dv/dt = 0
        max_ev = Events.makeZeroCrossEvent(HHcell.flatSpec['vars']['V'],
                                -1, max_ev_args, targetlang=targetlang,
                                flatspec=HHcell.flatSpec)
        min_ev = copy(max_ev)
        min_ev.dircode = 1
        min_ev.name = 'cell_min'

        # voltage threshold crossing event
        v_ev_args = {'name': 'V_thresh',
                        'eventtol': 1e-8,
                        'eventdelay': 1e-3,
                        'starttime': 0,
                        'term': False
                        }
        v_ev = Events.makeZeroCrossEvent('V-V_th', 0, v_ev_args,
                                         targetlang=targetlang,
                                         flatspec=HHcell.flatSpec)
        desc.userEvents = [min_ev, max_ev, v_ev]

    modelname = 'HH_'+name
    model_desc = MDescriptor(name=modelname,
                             withJac={HHcell.name: withJac})
    model_desc.add(desc)
    man.add(model_desc)

    # instantiate model for target generator
    # (could also try building syn_model_new)
    man.build(modelname, icvalues=ic_args, tdata=[0, 100])
    return man


def get_model(path, name, ic_args, alg_args, gen='Vode'):
    global man
    try:
        man = loadObjects(path+'model_'+name+'.sav')[0]
    except:
        # make_model is a dict for looking up modelspec factory functions,
        # ultimately calling instantiate and returning model manager 'man'
        man = make_model[name](ic_args, alg_args, gen+'_ODEsystem', with_spike_ev=False)
        saveObjects(man, path+'model_'+name+'.sav')
    return man


def get_ref(HHmodel, path, name, force=False):
    try:
        if force:
            raise "recalculate"
        ref_ic, ref_pts, ref_traj = loadObjects(path+'ref_'+name+'.sav')
    except:
        HHmodel.compute(trajname='ref_traj', tdata=[0,400])
        evts = HHmodel.getTrajEventTimes('ref_traj', 'cell_min')
        d_evts = [evts[i]-evts[i-1] for i in range(1,len(evts))]
        ref_ic = HHmodel('ref_traj', evts[-1])
        ref_pts = HHmodel.sample('ref_traj', tlo=evts[-2], thi=evts[-1])
        ref_pts.indepvararray -= evts[-2]
        ref_traj = pointset_to_traj(ref_pts)
        saveObjects([ref_ic, ref_pts, ref_traj], path+'ref_'+name+'.sav', force=force)
    return ref_ic, ref_pts, ref_traj


