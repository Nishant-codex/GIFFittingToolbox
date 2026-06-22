#!/bin/env python

import bluepyopt as bpop
from bluepyopt.parameters import Parameter

import numpy as np

import GLIF_1 as glif_model
import brian2 as b2
b2.prefs.codegen.target = 'cython'

class Evaluator(bpop.evaluators.Evaluator):
    def __init__(self, input_current, dt, init_values,parameters, fitness, target_voltage,target_spiketimes):
        self.input_current = input_current
        self.dt = dt
        self.init_values = init_values
        self.parameters = parameters
        self.fitness = fitness
        self.target_voltage = target_voltage
        self.target_spiketimes = target_spiketimes
        print(parameters.keys())
        super(Evaluator, self).__init__(    
                objectives=[x for x in fitness.keys()],
                params=[Parameter('El',     parameters['El']['value'],     bounds = parameters['El']['bounds'],     frozen=parameters['El']['frozen']),
                        Parameter('C',      parameters['C']['value'],      bounds = parameters['C']['bounds'],      frozen=parameters['C']['frozen']),
                        Parameter('G',      parameters['G']['value'],      bounds = parameters['G']['bounds'],      frozen=parameters['G']['frozen']),
                        Parameter('Th_inf', parameters['Th_inf']['value'], bounds = parameters['Th_inf']['bounds'], frozen=parameters['Th_inf']['frozen']),
                        Parameter('t_ref',  parameters['t_ref']['value'],  bounds = parameters['t_ref']['bounds'],  frozen=parameters['t_ref']['frozen']),
                        Parameter('V_reset',parameters['V_reset']['value'],bounds = parameters['V_reset']['bounds'],frozen=parameters['V_reset']['frozen'])])

    def evaluate_with_dicts(self, param_dict):
        # b2.set_device('cpp_standalone')

        # Simulate with parameter set
        param_dict_units = glif_model.add_parameter_units(param_dict)
        t, V,spks  = glif_model.run_brian_sim(
                self.input_current * b2.amp,
                self.dt * b2.second,
                self.init_values,
                param_dict_units)

        T = len(V)/20
        data_spike_times = self.target_spiketimes
        model_spike_times = np.array(spks.spike_trains()[0])*1000
        #Evaluate fitness
        # fitness = {x: self.fitness[x](model_spike_times, data_spike_times, 5, T, self.dt ,) for x in self.fitness.keys()}
        fitness = {x: self.fitness[x](V,self.target_voltage ) for x in self.fitness.keys()}

        print(fitness)
        # b2.device.delete(force = True)
        # b2.device.reinit()

        return fitness

    def evaluate_with_lists(self, param_list):
        param_dict = glif_model.parameters_from_list(param_list)
        fitness = self.evaluate_with_dicts(param_dict)
        return np.array([fitness[x] for x in fitness.keys()])
    
    def init_simulator_and_evaluate_with_lists(self, param_list):
        """Calls evaluate_with_lists. Is called during IBEA optimisation."""
        return self.evaluate_with_lists(param_list)
