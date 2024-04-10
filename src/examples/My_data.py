import sys
sys.path.append('C:/Users/Nishant Joshi/Model_fitting/GIFFittingToolbox/src')

from Experiment import *
from AEC_Badel import *
from GIF import *
from Filter_Rect_LogSpaced import *
from Filter_Rect_LinSpaced import *
import pickle
import matplotlib.pyplot as plt


"""
This file shows how to fit a GIF to some experimental data.
More instructions are provided on the website. 
"""

############################################################################################################
# STEP 1: LOAD EXPERIMENTAL DATA
############################################################################################################
myExp = Experiment('Experiment 1', 0.05)

with open("G:/My Drive/Bernstein/170725_NC_82_INH.pickle",'rb') as f:
    data = pickle.load(f)
I_data = data['I'][:120*20000]
V_data = data['V'][:120*20000]
spikes_data = data['spikes']    

# Load AEC data
myExp.setAECTrace(V_data[:int(10*20000)],1e-3,I_data[:int(10*20000)] ,1e-12, 10000.0, FILETYPE='Array')

# Load training set data
myExp.addTrainingSetTrace(V_data,1e-3,I_data, 1e-12, 120000.0, FILETYPE='Array')

# Load test set data

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
myGIF = GIF(0.05)

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
myExp.plotTrainingSet()

# Perform the fit
myGIF.fit(myExp, DT_beforeSpike=5.0)

# Plot the model parameters
myGIF.printParameters()
myGIF.plotParameters()   

I = myExp.trainingset_traces[0].I
V_exp = myExp.trainingset_traces[0].V
spks =myExp.trainingset_traces[0].spks/10
(time, V, I_a, V_t, S) = myGIF.simulate(I, myGIF.El)


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


plot_time = 1 # s
plt.plot(time[:plot_time*10000],V[:plot_time*10000],c='red',label='model')
plt.plot(time[:plot_time*10000],V_exp[:plot_time*10000],c='black',label='recording')
plt.scatter(time[:plot_time*10000], spks_model[:plot_time*10000]*85,c='red' ,marker='|')
plt.scatter(time[:plot_time*10000], spks_data[:plot_time*10000]*75,c='black',marker='|')
plt.legend(loc='lower left')
plt.show()



plot_time = 5 # s
plt.plot(time[:plot_time*10000],V[:plot_time*10000],c='red',label='model')
plt.plot(time[:plot_time*10000],V_exp[:plot_time*10000],c='black',label='recording')
plt.scatter(time[:plot_time*10000], spks_model[:plot_time*10000]*85,c='red' ,marker='|')
plt.scatter(time[:plot_time*10000], spks_data[:plot_time*10000]*75,c='black',marker='|')
plt.legend(loc='lower left')
plt.show()


plot_time = 100 # s
plt.plot(time[:plot_time*10000],V[:plot_time*10000],c='red',label='model')
plt.plot(time[:plot_time*10000],V_exp[:plot_time*10000],c='black',label='recording')
plt.scatter(time[:plot_time*10000], spks_model[:plot_time*10000]*85,c='red' ,marker='|')
plt.scatter(time[:plot_time*10000], spks_data[:plot_time*10000]*75,c='black',marker='|')
plt.legend(loc='lower left')
plt.show()



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



