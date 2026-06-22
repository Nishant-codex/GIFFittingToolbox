#!/bin/env python

import brian2
import os
from brian2 import *
setattr(brian2.units, 'none', 1.0)  # add brian2.units.none so metaprogramming works
# Run Brian2 simulation with external input

def run_brian_sim(stim, dt, init_values, param_dict, method = 'exact'):
    # Model specification
    pid = os.getpid()
    print('running ',pid)
    eqs = brian2.Equations("")
    eqs += brian2.Equations("dV/dt = 1 / C * (Ie(t) - G * (V - El)) : volt (unless refractory)")
    reset = ""
    reset = "\n".join([reset, "V = V_reset"])
    threshold = "V > Th_inf"
    threshold.join('V(t+1)=50')
    refractory = param_dict['t_ref']

    Ie = brian2.TimedArray(stim, dt=dt)
    nrn = brian2.NeuronGroup(1, eqs, method=method, reset=reset, threshold=threshold, refractory=refractory, namespace=param_dict)
    nrn.V = init_values['V_init'] 

    monvars = ['V',]
    mon = brian2.StateMonitor(nrn, monvars, record=True)
    spks = brian2.SpikeMonitor(nrn)
    num_step = len(stim)
    brian2.defaultclock.dt = dt
    brian2.run(num_step * dt)
    print('finished ',pid)


    return (mon.t / brian2.units.second, mon.V[0] / brian2.units.volt, spks)


def add_parameter_units(param_dict):
    param_dict_units = {
        'El': param_dict['El'] * brian2.units.volt,
        'C': param_dict['C'] * brian2.units.farad,
        'G': param_dict['G'] * brian2.units.siemens,
        'Th_inf': param_dict['Th_inf'] * brian2.units.volt,
        't_ref': param_dict['t_ref'] * brian2.units.second,
        'V_reset': param_dict['V_reset'] * brian2.units.volt,
    }
    return param_dict_units


def parameters_from_list(param_list):
    param_dict = {
        'El': param_list[0],
        'C': param_list[1],
        'G': param_list[2],
        'Th_inf': param_list[3],
        't_ref': param_list[4],
        'V_reset': param_list[5],
    }
    return param_dict
