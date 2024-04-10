import sys
sys.path.append('C:/Users/Nishant Joshi/Model_fitting/GIFFittingToolbox/src')

from Experiment import *
from AEC_Badel import *
from GIF import *
from Filter_Rect_LogSpaced import *
from Filter_Rect_LinSpaced import *

import matplotlib.pyplot as plt


"""
This file shows how to fit a GIF to some experimental data.
More instructions are provided on the website. 
"""

############################################################################################################
# STEP 1: LOAD EXPERIMENTAL DATA
############################################################################################################
myExp = Experiment('Experiment 1', 0.1)

PATH = 'C:/Users/Nishant Joshi/Model_fitting/GIFFittingToolbox/data/gif_test/'

# Load AEC data
myExp.setAECTrace(PATH + 'Cell3_Ger1Elec_ch2_1007.ibw', 1.0, PATH + 'Cell3_Ger1Elec_ch3_1007.ibw', 1.0, 10000.0, FILETYPE='Igor')

# Load training set data
myExp.addTrainingSetTrace(PATH + 'Cell3_Ger1Training_ch2_1008.ibw', 1.0, PATH + 'Cell3_Ger1Training_ch3_1008.ibw', 1.0, 120000.0, FILETYPE='Igor')

# Load test set data
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1009.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1009.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1010.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1010.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1011.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1011.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1012.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1012.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1013.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1013.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1014.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1014.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1015.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1015.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1016.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1016.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1017.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1017.ibw', 1.0, 20000.0, FILETYPE='Igor')

# Plot data
#myExp.plotTrainingSet()
#myExp.plotTestSet()


############################################################################################################
# STEP 2: ACTIVE ELECTRODE COMPENSATION
############################################################################################################

# Create new object to perform AEC
myAEC = AEC_Badel(myExp.dt)

# Define metaparametres
myAEC.K_opt.setMetaParameters(length=150.0, binsize_lb=myExp.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
myAEC.p_expFitRange = [3.0,150.0]  
myAEC.p_nbRep = 15     

# Assign myAEC to myExp and compensate the voltage recordings
myExp.setAEC(myAEC)  
myExp.performAEC()  

# # Plot AEC filters (Kopt and Ke)
# myAEC.plotKopt()
# myAEC.plotKe()

# # Plot training and test set
# myExp.plotTrainingSet()
# myExp.plotTestSet()


############################################################################################################
# STEP 3: FIT GIF MODEL TO DATA
############################################################################################################

# Create a new object GIF 
myGIF = GIF(0.1)

# Define parameters
myGIF.Tref = 4.0  

myGIF.eta = Filter_Rect_LogSpaced()
myGIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)


myGIF.gamma = Filter_Rect_LogSpaced()
myGIF.gamma.setMetaParameters(length=500.0, binsize_lb=5.0, binsize_ub=1000.0, slope=5.0)



# Define the ROI of the training set to be used for the fit (in this example we will use only the first 100 s)
myExp.trainingset_traces[0].setROI([[0,100000.0]])

# To visualize the training set and the ROI call again
myExp.detectSpikes_python()
# myExp.plotTrainingSet()

# Perform the fit
myGIF.fit(myExp, DT_beforeSpike=5.0)

# Plot the model parameters
myGIF.printParameters()
# myGIF.plotParameters()   

I = myExp.trainingset_traces[0].I
V_exp = myExp.trainingset_traces[0].V
spks =myExp.trainingset_traces[0].spks/10
(time, V, I_a, V_t, S) = myGIF.simulate(I, myGIF.El)


def get_gamma_factor(model, data, delta, time, dt, rate_correction=True):
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

    Returns
    -------
    float
        An error based on the Gamma factor. If ``rate_correction`` is used,
        then the returned error is :math:`1 + 2\frac{\lvert r_\mathrm{data} - r_\mathrm{model}\rvert}{r_\mathrm{data}} - \Gamma`
        (with :math:`r_\mathrm{data}` and :math:`r_\mathrm{model}` being the
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
    return gamma
    # return np.clip(rate_term - gamma, 0, np.inf)

def getBinarySpikeTrain(V,spikes,dt,type='zero'):
    spikeinds  = np.int32(spikes/dt)
    if type=='zero':
        b_spikes = np.zeros(len(V))

    else:
        b_spikes = np.zeros(len(V))*np.nan
    b_spikes[spikeinds] =1
    return b_spikes
spks_model = getBinarySpikeTrain(V,S,0.1,type='nan')
spks_data = getBinarySpikeTrain(V_exp,spks,0.1,type='nan')
print('gamma:',get_gamma_factor(S,spks,4,len(V)*0.1,0.1))

# plot_time = 1 # s
# plt.plot(time[:plot_time*10000],V[:plot_time*10000],c='red',label='model')
# plt.plot(time[:plot_time*10000],V_exp[:plot_time*10000],c='black',label='recording')
# plt.scatter(time[:plot_time*10000], spks_model[:plot_time*10000]*85,c='red' ,marker='|')
# plt.scatter(time[:plot_time*10000], spks_data[:plot_time*10000]*75,c='black',marker='|')
# plt.legend(loc='lower left')
# plt.show()



# plot_time = 5 # s
# plt.plot(time[:plot_time*10000],V[:plot_time*10000],c='red',label='model')
# plt.plot(time[:plot_time*10000],V_exp[:plot_time*10000],c='black',label='recording')
# plt.scatter(time[:plot_time*10000], spks_model[:plot_time*10000]*85,c='red' ,marker='|')
# plt.scatter(time[:plot_time*10000], spks_data[:plot_time*10000]*75,c='black',marker='|')
# plt.legend(loc='lower left')
# plt.show()


# plot_time = 100 # s
# plt.plot(time[:plot_time*10000],V[:plot_time*10000],c='red',label='model')
# plt.plot(time[:plot_time*10000],V_exp[:plot_time*10000],c='black',label='recording')
# plt.scatter(time[:plot_time*10000], spks_model[:plot_time*10000]*85,c='red' ,marker='|')
# plt.scatter(time[:plot_time*10000], spks_data[:plot_time*10000]*75,c='black',marker='|')
# plt.legend(loc='lower left')
# plt.show()



## Save the model
#myGIF.save('./myGIF.pck')


############################################################################################################
# STEP 3A (OPTIONAL): PLAY A BIT WITH THE FITTED MODEL
############################################################################################################

## Reload the model
#myGIF = GIF.load('./myGIF.pck')
#
## Generate OU process with temporal correlation 3 ms and mean modulated by a sinusoildal function of 1 Hz
#I_OU = Tools.generateOUprocess_sinMean(f=1.0, T=5000.0, tau=3.0, mu=0.3, delta_mu=0.5, sigma=0.1, dt=0.1)
#
## Simulate the model with the I_OU current. Use the reversal potential El as initial condition (i.e., V(t=0)=El)
#(time, V, I_a, V_t, S) = myGIF.simulate(I_OU, myGIF.El)
#
## Plot the results of the simulation
#plt.figure(figsize=(14,5), facecolor='white')
#plt.subplot(2,1,1)
#plt.plot(time, I_OU, 'gray')
#plt.ylabel('I (nA)')
#plt.subplot(2,1,2)
#plt.plot(time, V,'black', label='V')
#plt.plot(time, V_t,'red', label='V threshold')
#plt.ylabel('V (mV)')
#plt.xlabel('Time (ms)')
#plt.legend()
#plt.show()



############################################################################################################
# STEP 4: EVALUATE THE GIF MODEL PERFORMANCE (USING MD*)
############################################################################################################

# Use the myGIF model to predict the spiking data of the test data set in myExp

# myPrediction = myExp.predictSpikes(myGIF, nb_rep=500)

# # Compute Md* with a temporal precision of +/- 4ms
# Md = myPrediction.computeMD_Kistler(4.0, 0.1)    

# # Plot data vs model prediction
# myPrediction.plotRaster(delta=1000.0) 



