#!/bin/env python

import GLIF_2 as glif_model

## Translation of model parameters from AllenSDK to Brian2
def load_AllenSDK_parameters(neuron_config, dt, units = True):
    # unitless
    param_dict = {
        'El': neuron_config['El'],
        'C': neuron_config['coeffs']['C'] * neuron_config['C'],
        'G': neuron_config['coeffs']['G'] * (1.0 / neuron_config['R_input']),
        'Th_inf': neuron_config['coeffs']['th_inf'] * neuron_config['th_inf'],
        't_ref': (neuron_config['spike_cut_length'] + 1) * dt,
        'a_r': neuron_config['voltage_reset_method']['params']['a'],
        'b_r': neuron_config['voltage_reset_method']['params']['b'],
        'a_s': neuron_config['threshold_reset_method']['params']['a_spike'],
        'b_s': neuron_config['threshold_reset_method']['params']['b_spike'],
    }

    if units:
        param_dict = glif_model.add_parameter_units(param_dict)

    return param_dict


## Translation of initial values from AllenSDK to Brian2
def load_AllenSDK_initial_values(neuron_config):
    # unitless
    init_value_dict = {
        'V': neuron_config['init_voltage'],
        'Th_s': 0.0,
    }

    return init_value_dict


if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    import allensdk.core.json_utilities as json_utilities
    from allensdk.core.nwb_data_set import NwbDataSet
    from allensdk.model.glif.glif_neuron import GlifNeuron
    from sklearn.metrics import mean_squared_error as MSE
    import numpy as np

    import brian2 as b2
    b2.prefs.codegen.target = 'cython'

    # Read files from Allen Brain Atlas: Cell Types
    neuron_config = json_utilities.read(sys.argv[1])
    ephys_sweeps = json_utilities.read(sys.argv[2])
    ephys_file_name = sys.argv[3]

    ds = NwbDataSet(ephys_file_name)


    # Setup and run both simulations
    def sim_Brian2_AllenSDK(neuron_config, data):
        stim = data['stimulus']
        dt = 1.0 / data['sampling_rate']

        stim_unit = data['stimulus_unit']
        if stim_unit == 'pA':
            stim = stim * 1E12  # convert pA to Amps

        # Brian2 simulation
        shift_stim = np.pad(stim[1:],(0,1),mode='edge')  # shift stimulus left once
        param_dict = load_AllenSDK_parameters(neuron_config, dt)
        init_values = load_AllenSDK_initial_values(neuron_config)
        # With method = 'euler' the error w.r.t. AllenSDK traces is smaller in some cases
        # AllenSDK solves exponentials analytically, which results in small discrepancies
        output_Brian2 = glif_model.run_brian_sim(shift_stim * b2.amp, dt * b2.second, init_values, param_dict, 'exact')
        t, V, Th_s,  = output_Brian2
        print("info: Brian2 simulation DONE")

        # AllenSDK simulation
        glif = GlifNeuron.from_dict(neuron_config)
        glif.dt = dt
        output_AllenSDK = glif.run(stim)
        V_0 = output_AllenSDK['voltage']
        Th_0 = output_AllenSDK['threshold']
        print("info: AllenSDK simulation DONE")

        return (t, V, V_0, neuron_config['coeffs']['th_inf'] * neuron_config['th_inf'] + Th_s, Th_0, )


    # Run simulations for all sweeps
    def sim_sweep(num, plot = False):
        b2.set_device('cpp_standalone', directory='sweep_{}'.format(num))

        data = ds.get_sweep(num)
        result = sim_Brian2_AllenSDK(neuron_config, data)
        t, V, V_0, Th, Th_0,  = result

        #Compute errors where AllenSDK trace is not NaN
        errors = []

        w = np.where(~np.isnan(V_0))
        err = MSE(V_0[w], V[w], squared = False)
        errors += [err]
        print("info: RMSE [V] = {}".format(err))

        w = np.where(~np.isnan(Th_0))
        err = MSE(Th_0[w], Th[w], squared = False)
        errors += [err]
        print("info: RMSE [Th] = {}".format(err))

        if plot:
            fig = plt.figure()
            fig.suptitle('Stimulus')
            plt.plot(t,data['stimulus'])
            fig = plt.figure()
            fig.suptitle('V')
            plt.plot(t,V)
            plt.plot(t,V_0)
            fig = plt.figure()
            fig.suptitle('Th')
            plt.plot(t,Th)
            plt.plot(t,Th_0)
            plt.show()

        b2.device.delete(force = True)
        b2.device.reinit()

        return tuple(errors)


    stim_nums = [x['sweep_number'] for x in ephys_sweeps if x['stimulus_units'] in ["Amps","pA"]]

    sweep_start = 0
    sweep_stop = len(stim_nums)
    if len(sys.argv) > 5:
        sweep_start = int(sys.argv[4])
        if sweep_start < 0:
            sweep_start = 0
        sweep_stop = int(sys.argv[5])
        if sweep_stop > len(stim_nums):
            sweep_stop = len(stim_nums)

    do_plot = False
    if len(sys.argv) > 6:
        do_plot = int(sys.argv[6])

    # Pymp (dynamic scheduling)
    import pymp
    errors = pymp.shared.array((sweep_stop - sweep_start, 2))
    with pymp.Parallel() as p:
        for i in p.xrange(0, sweep_stop - sweep_start):
            errors[i] = sim_sweep(stim_nums[sweep_start:sweep_stop][i], do_plot)

    # Multiprocessing (static scheduling)
    #import multiprocessing
    #pool = multiprocessing.Pool()
    #errors = pool.map(sim_sweep, stim_nums[sweep_start:sweep_stop])

    # Serial
    #errors = np.zeros((sweep_stop - sweep_start, 2))
    #for i,num in enumerate(stim_nums[sweep_start:sweep_stop]):
        #errors[i] = sim_sweep(num, do_plot)

    result = np.hstack((np.array(stim_nums[sweep_start:sweep_stop]).reshape(-1,1),errors))
    np.savetxt("results.txt",result,"%g")
