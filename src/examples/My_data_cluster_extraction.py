import sys
sys.path.append('C:/Users/Nishant Joshi/Model_fitting/GIFFittingToolbox/src')
import os
from Experiment import *
from AEC_Badel import *
from GIF import *
from Filter_Rect_LogSpaced import *
from Filter_Rect_LinSpaced import *
import pickle
from Filter_Exps import *
import matplotlib.pyplot as plt
sys.path.append('C:/Users/Nishant Joshi/Downloads/Old_code/repo/single_cell_analysis/scripts')
from utils import * 


"""
This file shows how to fit a GIF to some experimental data.
More instructions are provided on the website. 
"""


def getBinarySpikeTrain(V,spikes,dt,type='zero'):
    spikeinds  = np.int32(spikes/dt)
    if type=='zero':
        b_spikes = np.zeros(len(V))

    else:
        b_spikes = np.zeros(len(V))*np.nan
    b_spikes[spikeinds] =1
    return b_spikes

def get_gamma_factor(modelspks, dataspks, delta, time, dt, rate_correction=True):
    """
    Calculate gamma factor between model and target spike trains,
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

    """
    model = np.array(modelspks)
    data = np.array(dataspks)

    model = np.array(np.int32(model / dt), dtype=int)
    data = np.array(np.int32(data / dt), dtype=int)
    delta_diff = int(np.int32(delta / dt))

    model_length = len(model)
    data_length = len(data)
    # data_rate = firing_rate(data) * Hz
    data_rate = data_length / time
    model_rate = model_length / time

    if model_length > 1:
        bins = .5 * (model[1:] + model[:-1])
        indices = np.digitize(data, bins)
        diff = np.abs(data - model[indices])
        matched_spikes = (diff <= delta_diff)
        coincidences = np.sum(matched_spikes)
    elif model_length == 0:
        coincidences = 0
    else:
        indices = [np.amin(abs(model - data[i])) <= delta_diff for i in np.arange(data_length)]
        coincidences = sum(indices)

    # Normalization of the coincidences count
    NCoincAvg = 2 * delta * data_length * data_rate
    norm = .5*(1 - 2 * data_rate * delta)
    gamma = (coincidences - NCoincAvg)/(norm*(model_length + data_length))

    if rate_correction:
        rate_term = 1 + 2*np.abs((data_rate - model_rate)/data_rate)
    else:
        rate_term = 1

    return gamma

    # return np.clip(rate_term - gamma, 0, np.inf)


############################################################################################################
# STEP 1: LOAD EXPERIMENTAL DATA
############################################################################################################
paramlist  =  [] 
path = 'D:/Analyzed/'
for file in os.listdir(path):

    data = loadmatInPy(path+file)
    for trial,data_i in enumerate(data):
        try:
            I_data = data_i['input_current'][:120*20000]
            V_data = data_i['membrane_potential'][:120*20000]
            spikes_data = data_i['spikeindices'] 
            cond = data_i['input_generation_settings']['condition']
            trial_i = trial
            experimentname = data_i['input_generation_settings']['experimentname']
            myExp = Experiment('Experiment 1', 0.05)

            print('Running file '+file+' trial '+str(trial)+' condition '+cond)
            # Load AEC data
            # myExp.setAECTrace(V_data[:int(10*20000)],1e-3,I_data[:int(10*20000)] ,1e-12, 10000.0, FILETYPE='Array')

            # Load training set data
            myExp.addTrainingSetTrace(V_data,1e-3,I_data, 1e-12, 120000.0, FILETYPE='Array')

            ############################################################################################################
            # STEP 3: FIT GIF MODEL TO DATA
            ############################################################################################################

            # Create a new object GIF 
            myGIF = GIF(0.05)
            myGIF.print_log=False
            # Define parameters
            myGIF.Tref = 4.0  

            #Rect Filter
            myGIF.eta = Filter_Rect_LogSpaced()
            myGIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)
            myGIF.gamma = Filter_Rect_LogSpaced()
            myGIF.gamma.setMetaParameters(length=500.0, binsize_lb=1.0, binsize_ub=1000.0, slope=5.0)



            # Exp Filter
            # myGIF.eta = Filter_Exps()
            # myGIF.eta.setFilter_Timescales([1.0, 5.0, 30.0, 70.0, 100.0, 500.0])
            # myGIF.gamma = Filter_Exps()
            # myGIF.gamma.setFilter_Timescales([1.0, 5.0, 30.0, 70.0, 100.0, 500.0])


            # Define the ROI of the training set to be used for the fit (in this example we will use only the first 100 s)
            myExp.trainingset_traces[0].setROI([[0,100000.0]])

            # To visualize the training set and the ROI call again
            myExp.detectSpikes_python()
            # myExp.plotTrainingSet()

            # Perform the fit
            myGIF.fit(myExp, DT_beforeSpike=5.0)

            # Plot the model parameters
            # myGIF.printParameters()
            # myGIF.plotParameters()   

            I = myExp.trainingset_traces[0].I
            V_exp = myExp.trainingset_traces[0].V
            spks = myExp.trainingset_traces[0].spks*myExp.dt
            (time, V, I_a, V_t, S) = myGIF.simulate(I, myGIF.El)

            spks_model = getBinarySpikeTrain(V,S,myExp.dt,type='nan')
            spks_data = getBinarySpikeTrain(V_exp,spks,myExp.dt,type='nan')
            gamma = get_gamma_factor(S/1000,spks/1000,5/1000,len(V)/20000,1/20000)
            print('gamma:',gamma)


            ## Save the model
            myGIF.saveparams(paramlist,gamma,cond,trial_i,experimentname)
        except:
            print('Problem with ',file)

with open('D:/Biophysical_cluster/cluster_params.p','wb') as f:
    
    pickle.dump(paramlist,f)





