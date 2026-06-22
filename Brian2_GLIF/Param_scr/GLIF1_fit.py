import sys
import numpy as np 
import multiprocessing
import bluepyopt as bpop
from bluepyopt.parameters import Parameter
from time import time as wall_time
import os
import brian2 as b2
sys.path.append('C:/Users/Nishant Joshi/Documents/Siamese_net/Brian2_GLIF_AllenSDK')
sys.path.append('C:/Users/Nishant Joshi/Downloads/Old_code/repo/single_cell_analysis/scripts')
from GLIF_1_Evaluator import Evaluator
from GLIF_1 import * 
from utils import *
import matplotlib.pyplot as plt
parameters = {'El':     {'value':-52.249/1000,'bounds':[-0.056,-0.051],   'frozen':True},
              'C':      {'value':0.028e-9,    'bounds':[27e-12, 29e-12],  'frozen':True},
              'G':      {'value':8.374e-9,    'bounds':[8.33e-9,8.375e-9],'frozen':True},
              't_ref':  {'value':4/1000,      'bounds':[0.0035,0.005],    'frozen':True},
              'Th_inf': {'value':-50/1000,    'bounds':[-0.055, -0.045],  'frozen':True},
              'V_reset':{'value':-55.354/1000,'bounds':[-0.056,-0.055],   'frozen':False}}

params_init = {'El':-0.052249,
               'C':28e-12,
               'G':8.3745e-9,
               'Th_inf':-0.050,
               't_ref':0.004,
               'V_reset':-0.055354}

def rms(val1,val2):
    return [np.sqrt(np.mean((val1-val2)**2))]

def get_gamma_factor(model, data, delta, time, dt, rate_correction=True):
    """
    Calculate gamma factor between model and target spike trains,\l
    with precision delta.

    Parameters
    ----------
    model: `list` or `~numpy.ndarray`
        model trace
    data: `list` or `~numpy.ndarray`
        data trace
    delta: `~brian2.units.fundamentalunits.Quantity`
        time window
    dt: `~brian2.units.fundamentalunits.Quantity`
        time step
    time: `~brian2.units.fundamentalunits.Quantity`
        total time of the simulation
    rate_correction: bool
        Whether to include an error term that penalizes differences in firing
        rate, following `Clopath et al., Neurocomputing (2007)
        <https://doi.org/10.1016/j.neucom.2006.10.047>`_.

    Returns
    -------
    float
        An error based on the Gamma factor. If ``rate_correction`` is used,
        then the returned error is :math:`1 + 2\frac{\\lvert r_\\mathrm{data} - r_\mathrm{model}\rvert}{r_\mathrm{data}} - \Gamma`
        (with :math:`r_\\mathrm{data}` and :math:`r_\mathrm{model}` being the
        firing rates in the data/model, and :math:`\Gamma` the coincidence
        factor). Without ``rate_correction``, the error is
        :math:`1 - \Gamma`. Note that the coincidence factor :math:`\Gamma`
        has a maximum value of 1 (when the two spike trains are exactly
        identical) and a value of 0 if there are only as many coincidences
        as expected from two homogeneous Poisson processes of the same rate.
        It can also take negative values if there are fewer coincidences
        than expected by chance.
    """
    model = np.array(model)
    data = np.array(data)

    model = np.array(model / dt, dtype=np.int32)
    data = np.array(data / dt, dtype=np.int32)
    delta_diff = int(np.int32(delta / dt))

    model_length = len(model)
    data_length = len(data)
    # data_rate = firing_rate(data) * Hz
    data_rate = data_length / time
    model_rate = model_length / time

    if model_length > 1:
        bins = .5 * (model[1:] + model[:-1])
        indices = np.digitize(data, bins)
        diff = abs(data - model[indices])
        matched_spikes = (diff <= delta_diff)
        coincidences = sum(matched_spikes)
    elif model_length == 0:
        coincidences = 0
    else:
        indices = [np.amin(abs(model - data[i])) <= delta_diff for i in np.arange(data_length)]
        coincidences = sum(indices)

    # Normalization of the coincidences count
    NCoincAvg = 2 * data_rate * delta * model_length  #2*v2*p*N1
    norm = .5*(1 - 2 * max(data_rate,model_rate) * delta)
    gamma = (coincidences - NCoincAvg)/(norm*(model_length + data_length))

    if rate_correction:
        rate_term = 1 + 2*abs((data_rate - model_rate)/data_rate)
    else:
        rate_term = 1
    return 1- gamma
    # return np.clip(rate_term - gamma, 0, np.inf)

if __name__ == "__main__":
    start_time = wall_time()

    ######################################## Load Data ##################################
    with open("G:/My Drive/Bernstein/170725_NC_82_INH.pickle",'rb') as f:
        data = pickle.load(f)
    I = data['I'][:1*20000]
    V = data['V'][:1*20000]
    spikes = data['spikes']   
    spiketimes = spikes[spikes<=20000]/20
    print('data_loaded')
    
    
    ################################ Initialize Parameters ##############################
    params_init = add_parameter_units(params_init)

    print('running parameter finder')
    
    
    ################################ Setup Evaluator ####################################
    eva = Evaluator(input_current=I*1e-12,dt=1/20000, init_values={'V_init':params_init['El']},parameters=parameters, fitness={'rms':rms},target_voltage= V/1000,target_spiketimes = spiketimes)

    ################################ Run Evalutation ####################################
    num_proc = 4

    with multiprocessing.Pool(num_proc) as p:
        opt = bpop.deapext.optimisations.DEAPOptimisation(eva,map_function=p.map )
        final_pop, hall_of_fame, logs, hist = opt.run(max_ngen=20,)
    print(f"Done in {wall_time() - start_time:10.3f}")


    best_ind = hall_of_fame[0]
    print('Best individual: ', best_ind)
    print('Fitness values: ', best_ind.fitness.values)
    param_list = parameters_from_list(best_ind)
    param_list_units = add_parameter_units(param_list)


    ############################ Run simulation with the best found parameters ###########################
    t, fitted_voltage,spks = run_brian_sim(stim = I*b2.pA,
                                        param_dict=param_list_units,
                                    init_values={'V_init':param_list_units['El']}, 
                                    dt=1/20000*b2.second, )
    
    data_spike_times = spiketimes
    model_spike_times = np.array(spks.spike_trains()[0])*1000
    print(rms(fitted_voltage,V/1000))
    print(get_gamma_factor(model_spike_times,data_spike_times,4,1000,1/20))

    plt.plot(t, V/1000, label='target')
    plt.plot(t, fitted_voltage, label='fit')
    plt.legend(frameon=False); 
    plt.xlabel('time (ms)')
    plt.ylabel('v (mV)')
    plt.show()

