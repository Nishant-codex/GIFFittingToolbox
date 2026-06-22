#!/bin/env python

import bluepyopt as bpop
from bluepyopt.parameters import Parameter

import numpy as np

import GLIF_5 as glif_model
import brian2 as b2
b2.prefs.codegen.target = 'cython'

class Evaluator(bpop.evaluators.Evaluator):
    def __init__(self, input_current, dt, init_values, parameters, fitness, target_voltage,target_spiketimes):
        self.input_current = input_current
        self.dt = dt
        self.init_values = init_values
        self.parameters = parameters
        self.fitness = fitness
        self.target_voltage = target_voltage
        self.target_spiketimes = target_spiketimes 
        super(Evaluator, self).__init__(
                objectives=[x for x in fitness.keys()],
                params=[
                    Parameter('El',
                        value = parameters['El']['value'],
                        bounds = parameters['El']['bounds'],
                        frozen = parameters['El']['frozen']),
                    Parameter('C',
                        value = parameters['C']['value'],
                        bounds = parameters['C']['bounds'],
                        frozen = parameters['C']['frozen']),
                    Parameter('G',
                        value = parameters['G']['value'],
                        bounds = parameters['G']['bounds'],
                        frozen = parameters['G']['frozen']),
                    Parameter('Th_inf',
                        value = parameters['Th_inf']['value'],
                        bounds = parameters['Th_inf']['bounds'],
                        frozen = parameters['Th_inf']['frozen']),
                    Parameter('t_ref',
                        value = parameters['t_ref']['value'],
                        bounds = parameters['t_ref']['bounds'],
                        frozen = parameters['t_ref']['frozen']),
                    Parameter('a_r',
                        value = parameters['a_r']['value'],
                        bounds = parameters['a_r']['bounds'],
                        frozen = parameters['a_r']['frozen']),
                    Parameter('b_r',
                        value = parameters['b_r']['value'],
                        bounds = parameters['b_r']['bounds'],
                        frozen = parameters['b_r']['frozen']),
                    Parameter('a_s',
                        value = parameters['a_s']['value'],
                        bounds = parameters['a_s']['bounds'],
                        frozen = parameters['a_s']['frozen']),
                    Parameter('b_s',
                        value = parameters['b_s']['value'],
                        bounds = parameters['b_s']['bounds'],
                        frozen = parameters['b_s']['frozen']),
                    Parameter('a_v',
                        value = parameters['a_v']['value'],
                        bounds = parameters['a_v']['bounds'],
                        frozen = parameters['a_v']['frozen']),
                    Parameter('b_v',
                        value = parameters['b_v']['value'],
                        bounds = parameters['b_v']['bounds'],
                        frozen = parameters['b_v']['frozen']),
                    Parameter('A_0',
                        value = parameters['A_0']['value'],
                        bounds = parameters['A_0']['bounds'],
                        frozen = parameters['A_0']['frozen']),
                    Parameter('k_0',
                        value = parameters['k_0']['value'],
                        bounds = parameters['k_0']['bounds'],
                        frozen = parameters['k_0']['frozen']),
                    Parameter('R_0',
                        value = parameters['R_0']['value'],
                        bounds = parameters['R_0']['bounds'],
                        frozen = parameters['R_0']['frozen']),
                    Parameter('A_1',
                        value = parameters['A_1']['value'],
                        bounds = parameters['A_1']['bounds'],
                        frozen = parameters['A_1']['frozen']),
                    Parameter('k_1',
                        value = parameters['k_1']['value'],
                        bounds = parameters['k_1']['bounds'],
                        frozen = parameters['k_1']['frozen']),
                    Parameter('R_1',
                        value = parameters['R_1']['value'],
                        bounds = parameters['R_1']['bounds'],
                        frozen = parameters['R_1']['frozen']),
                    ]
                )

    def evaluate_with_dicts(self, param_dict):

        # Simulate with parameter set
        param_dict_units = glif_model.add_parameter_units(param_dict)
        t, V, Th_s, Th_v, I_0, I_1,spks  = glif_model.run_brian_sim(self.input_current * b2.amp,
                                                                    self.dt * b2.second,
                                                                    self.init_values,
                                                                    param_dict_units)

        #Evaluate fitness
        T = len(V)/20
        data_spike_times = self.target_spiketimes
        model_spike_times = np.array(spks.spike_trains()[0])*1000        
        # fitness = {x: self.fitness[x]( self.target_voltage, V) for x in self.fitness.keys()}
        fitness = {x: self.fitness[x](model_spike_times, data_spike_times, 5, T, self.dt ,) for x in self.fitness.keys()}
        print(fitness)

        return fitness

    def evaluate_with_lists(self, param_list):
        param_dict = glif_model.parameters_from_list(param_list)
        fitness = self.evaluate_with_dicts(param_dict)
        return np.array([fitness[x] for x in fitness.keys()])

    def init_simulator_and_evaluate_with_lists(self, param_list):
        """Calls evaluate_with_lists. Is called during IBEA optimisation."""
        return self.evaluate_with_lists(param_list)
