import sys
import numpy as np 
import multiprocessing
import random
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
from scipy.signal import fftconvolve
from scipy.optimize import leastsq
sys.path.append('C:/Users/Nishant Joshi/Model_fitting/GIFFittingToolbox/src')
import numba
from Filter_Rect_LogSpaced import *
from Filter_Rect_LinSpaced import *

from Filter_Exps import *
from numba import jit

import Tools

data = loadmatInPy('G:/My Drive/Analyzed/NC_170626_aCSF_D1ago_E4_analyzed.mat')

I_data = data[0]['input_current']
V_data = data[0]['membrane_potential']
spikes_data = data[0]['spikeindices']

class EphysData:

    def __init__(self,I,V,spikes,dt,duration):
        self.I = I[:int(duration/dt)]
        self.V = V[:int(duration/dt)]
        self.spikes = spikes[spikes<=int(duration/dt)]
        self.dt = dt
        self.spiketimes = self.spikes*dt 
        self.T = duration
    def getSpikeTimes(self):
        return self.spiketimes
    def getBinarySpikeTrain(self):
        b_spikes = np.zeros(len(self.V))
        b_spikes[self.spikes] =1
        return b_spikes
class Parameter:

    def __init__(self):
        None
    
    def getSpikeTimes(self):
        return None



def get_average_spike_shape(spikes,V,t_before=10,t_after=10,dt=1/20):
    spike_shapes = []
    t_before_idx = int((1/dt)*t_before)
    t_after_idx = int((1/dt)*t_after)

    for i in spikes:
        
        if len(V[i-t_before_idx:i+t_after_idx])==(t_before_idx+t_after_idx):
            spike_shapes.append(V[i-t_before_idx:i+t_after_idx])
        else:
           print(len(V[i:]))
    return np.mean(spike_shapes,axis=0),np.linspace(-t_before,t_after,int((t_before+t_after)/dt))

def fitVoltageReset(experiment,parameters, Tref,dt=1/20, do_plot=False):
    
    """
    Implement Step 1 of the fitting procedure introduced in Pozzorini et al. PLOS Comb. Biol. 2015
    experiment: Experiment object on which the model is fitted.
    Tref: ms, absolute refractory period. 
    The voltage reset is estimated by computing the spike-triggered average of the voltage.
    """
    
    print("Estimate voltage reset (Tref = %0.1f ms)..." % (Tref))
    
    # Fix absolute refractory period
    dt = experiment.dt
    Tref = Tref
    t_before = 10 
    t_after = 10
    all_spike_nb = len(experiment.spikes)
    spike_average,support = get_average_spike_shape(experiment.spikes,experiment.V,t_before=t_before,t_after=t_after)
    # Estimate voltage reset
    Tref_ind = np.where(support >= int(Tref))[0][0]
    Vr = spike_average[Tref_ind]
    parameters.Vr = Vr
    # Save average spike shape
    avg_spike_shape = spike_average
    avg_spike_shape_support = support
    
    if do_plot :
        plt.figure()
        plt.plot(support, spike_average, 'black')
        plt.plot([support[Tref_ind]], [Vr], '.', color='red')            
        plt.show()
    
    print("Done! Vr = %0.2f mV (computed on %d spikes)" % (Vr, all_spike_nb))

def simulateDeterministic_forceSpikes(experiment, parameters, eta, I, V0, spks):
    """
    Simulate the subthreshold response of the GIF model to an input current I (nA) with time step dt.
    Adaptation currents are forced to occur at times specified in the list spks (in ms) given as an argument
    to the function. V0 indicates the initial condition V(t=0)=V0.
    
    The function returns:
    
    - time     : ms, support for V, eta_sum, V_T, spks
    - V        : mV, membrane potential
    - eta_sum  : nA, adaptation current
    """
    # Input parameters
    p_T = len(I)
    p_dt = experiment.dt
      
    # Model parameters
    p_gl = parameters.gl
    p_C = parameters.C 
    p_El = parameters.El
    p_Vr = parameters.Vr
    p_Tref = parameters.Tref
    p_Tref_i = int(parameters.Tref / parameters.dt)
    
    # Model kernel      
    (p_eta_support, p_eta) = eta.getInterpolatedFilter(experiment.dt)   
    p_eta = p_eta.astype('double')
    p_eta_l = len(p_eta)
    


    # Define arrays

    I = np.array(I, dtype=np.double)
    spks = np.array(spks, dtype=np.double)                      
    spks_i = np.array(Tools.timeToIndex(spks, experiment.dt), dtype=np.int32)
    
    # Compute adaptation current (sum of eta triggered at spike times in spks) 
    eta_sum = np.zeros(int(p_T + 1.1 * p_eta_l + p_Tref_i), dtype=np.double)   
    
    for s in spks_i:
        eta_sum[s + 1 + p_Tref_i: s + 1 + p_Tref_i + p_eta_l] += p_eta
    
    eta_sum = eta_sum[:p_T]  

    @numba.jit(nopython=True)
    def runforV(V0,p_T,p_dt,p_C,p_gl,p_El,p_Vr,I,eta_sum):

        # Set initial condition
        V = np.zeros(p_T, dtype=np.double)
        V[0] = V0

        # Simulate the model
        for t in range(p_T - 1):
            # Integrate voltage
            V[t+1] = V[t] + p_dt / p_C * (-p_gl * (V[t] - p_El) + I[t] - eta_sum[t])

            # Check for forced spikes
            if t in spks_i:
                V[t-1] = 0
                V[t] = p_Vr
        return V
    V = runforV(V0,p_T,p_dt,p_C,p_gl,p_El,p_Vr,I,eta_sum)
    # Compute time array
    time = np.arange(p_T) * experiment.dt

    # Trim eta_sum to match time array length
    eta_sum = eta_sum[:p_T]

    return time, V, eta_sum

def printParameters(parameters):

    """
    Print model parameters on terminal.
    """

    print("\n-------------------------")        
    print("GIF model parameters:")
    print("-------------------------")
    print("tau_m (ms):\t%0.3f"  % (parameters.C/parameters.gl))
    print("R (MOhm):\t%0.3f"    % (1.0/parameters.gl))
    print("C (nF):\t\t%0.3f"    % (parameters.C))
    print("gl (nS):\t%0.6f"     % (parameters.gl))
    print("El (mV):\t%0.3f"     % (parameters.El))
    print("Tref (ms):\t%0.3f"   % (parameters.Tref))
    print("Vr (mV):\t%0.3f"     % (parameters.Vr))     
    # print("Vt* (mV):\t%0.3f"    % (parameters.Vt_star))    
    # print("DV (mV):\t%0.3f"     % (parameters.DV)) 
    print("dt (ms):\t%0.3f"     % (parameters.dt)) 

    print("-------------------------\n")

def simulate(params, I, V0, eta, gamma):
 
        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt.
        V0 indicate the initial condition V(0)=V0.
        The function returns:
        - time     : ms, support for V, eta_sum, V_T, spks
        - V        : mV, membrane potential
        - eta_sum  : nA, adaptation current
        - V_T      : mV, firing threshold
        - spks     : ms, list of spike times 
        """
 
        # Input parameters
        p_T         = len(I)
        p_dt        = params.dt
        
        # Model parameters
        p_gl        = params.gl
        p_C         = params.C
        p_El        = params.El
        p_Vr        = params.Vr
        p_Tref      = params.Tref
        p_Vt_star   = params.Vt_star
        p_DV        = params.DV
        p_lambda0   = params.lambda0
        
        # Model kernels   
        (p_eta_support, p_eta) = eta.getInterpolatedFilter(p_dt)   
        p_eta       = p_eta.astype(np.float64)
        p_eta_l     = len(p_eta)

        (p_gamma_support, p_gamma) = gamma.getInterpolatedFilter(p_dt)   
        p_gamma     = p_gamma.astype(np.float64)
        p_gamma_l   = len(p_gamma)
      
        # Define arrays
        V = np.array(np.zeros(p_T), dtype=np.float64)
        I = np.array(I, dtype=np.float64)
        spks = np.array(np.zeros(p_T), dtype=np.float64)                      
        # eta_sum = np.array(np.zeros(p_T + 2*p_eta_l), dtype="double")
        # gamma_sum = np.array(np.zeros(p_T + 2*p_gamma_l), dtype="double")            
 
        # Set initial condition
        @numba.jit(nopython=True)
        def inner_simulation(I, V0, dt, gl, C, El, Vr, Tref, Vt_star, DV, lambda0, p_eta, p_gamma):
            T_ind = len(I)
            T_ref_ind =int(Tref / dt)
            eta = p_eta.astype(np.float64)
            gamma = p_gamma.astype(np.float64)
            eta_l = len(eta)
            gamma_l = len(gamma)
            V = np.zeros(T_ind, dtype=np.float64)
            spks = np.zeros(T_ind, dtype=np.float64)
            eta_sum = np.zeros(T_ind + 2*eta_l, dtype=np.float64)
            gamma_sum = np.zeros(T_ind + 2*gamma_l, dtype=np.float64)            
    
            V[0] = V0

            # rand_max = np.float64(np.iinfo(np.uint32).max)

                  
            # 
            for t in range(T_ind-1):
                V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + I[t] - eta_sum[t] )
                # COMPUTE PROBABILITY OF EMITTING ACTION POTENTIAL
                lambda_val = lambda0*np.exp( (V[t+1]-Vt_star-gamma_sum[t])/DV )
                p_dontspike = np.exp(-lambda_val*(dt/1000.0)) #since lambda0 is in Hz, dt must also be in Hz (this is why dt/1000.0)

                # PRODUCE SPIKE STOCHASTICALLY
                r = np.random.rand()
                if r > p_dontspike:
                                    
                    if (t+1 < T_ind-1) :               
                        spks[t+1] = 1.0 
                    
                    t = t + T_ref_ind 
                    
                    if (t+1 < T_ind-1):
                        V[t-T_ref_ind]=80
                        V[t+1] = Vr
                    # // UPDATE ADAPTATION PROCESSES   
                          
                    for j in range(eta_l) : 
                        eta_sum[t+1+j] += eta[j]
                    for j in range(gamma_l) : 
                        gamma_sum[t+1+j] += gamma[j]        
                                
            time = np.arange(T_ind) * dt
            eta_sum = eta_sum[:T_ind]
            V_T = gamma_sum[:T_ind] + Vt_star
            spk_times = np.where(spks == 1.)[0] * dt

            return time, V, eta_sum,gamma_sum, V_T, spk_times

        time, V, eta_sum,gamma_sum, V_T, spk_times = inner_simulation(I, V0, p_dt, p_gl, p_C, p_El, p_Vr,
                                                            p_Tref, p_Vt_star, p_DV, p_lambda0, p_eta,
                                                            p_gamma)
    
        return (time, V, eta_sum, V_T, spk_times)

def fitSubthresholdDynamics(experiment,eta, parameters, DT_beforeSpike=5.0):
        
    """
    Implement Step 2 of the fitting procedure introduced in Pozzorini et al. PLOS Comb. Biol. 2015
    The voltage reset is estimated by computing the spike-triggered average of the voltage.
    experiment: Experiment object on which the model is fitted.
    DT_beforeSpike: in ms, data right before spikes are excluded from the fit. This parameter can be used to define that time interval.
    """  
                
    print("\nGIF MODEL - Fit subthreshold dynamics..." )
        
    # Expand eta in basis functions
    dt = experiment.dt
    
    
    # Build X matrix and Y vector to perform linear regression (use all traces in training set)    
    # For each training set an X matrix and a Y vector is built.   
    ####################################################################################################
    X = []
    Y = []
    tr = experiment
            
    # Compute the the X matrix and Y=\dot_V_data vector used to perform the multilinear linear regression (see Eq. 17.18 in Pozzorini et al. PLOS Comp. Biol. 2015)
    (X_tmp, Y_tmp) = fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr, experiment.Tref,DT_beforeSpike=DT_beforeSpike)


    # Concatenate matrixes associated with different traces to perform a single multilinear regression
    ####################################################################################################

    X = X_tmp
    Y = Y_tmp

    # Perform linear Regression defined in Eq. 17 of Pozzorini et al. PLOS Comp. Biol. 2015
    ####################################################################################################
    
    print("\nPerform linear regression...")
    XTX     = np.dot(np.transpose(X), X)
    XTX_inv = np.linalg.inv(XTX)
    XTY     = np.dot(np.transpose(X), Y)
    b       = np.dot(XTX_inv, XTY)
    b       = b.flatten()


    # Extract explicit model parameters from regression result b
    ####################################################################################################

    C  = 1./b[1]
    gl = -b[0]*C
    El = b[2]*C/gl
    parameters.gl = gl
    parameters.C = C
    parameters.El = El

    eta.setFilter_Coefficients(-b[3:]*C)

    # self.printParameters()   
    
    
    # Compute percentage of variance explained on dV/dt
    ####################################################################################################

    var_explained_dV = 1.0 - np.mean((Y - np.dot(X,b))**2)/np.var(Y)
    print("Percentage of variance explained (on dV/dt): %0.2f" % (var_explained_dV*100.0))

    
    # Compute percentage of variance explained on V (see Eq. 26 in Pozzorini et al. PLOS Comp. Biol. 2105)
    ####################################################################################################

    SSE = 0     # sum of squared errors
    VAR = 0     # variance of data
    
    # for tr in experiment.trainingset_traces :
    
    #     if tr.useTrace :

    # Simulate subthreshold dynamics 
    (time, V_est, eta_sum_est) = simulateDeterministic_forceSpikes(experiment, parameters, eta, tr.I, tr.V[0], tr.getSpikeTimes())
    
    indices_tmp = getROI_FarFromSpikes(experiment, 0.0, experiment.Tref)
    
    SSE += sum((V_est[indices_tmp] - tr.V[indices_tmp])**2)
    VAR += len(indices_tmp)*np.var(tr.V[indices_tmp])
            
    var_explained_V = 1.0 - SSE / VAR
    
    print("Percentage of variance explained (on V): %0.2f" % (var_explained_V*100.0))

def getROI_FarFromSpikes(experiment, DT_before, DT_after):

        """
        Return indices of the trace which are in ROI. Exclude all datapoints which are close to a spike.
        DT_before: ms
        DT_after: ms
        These two parameters define the region to cut around each spike.
        """
        
        L = len(experiment.V)
        
        LR_flag = np.ones(L)    
        
        
        # Select region in ROI
        ROI_ind = np.arange(len(experiment.V))
        LR_flag[ROI_ind] = 0.0

        # Remove region around spikes
        DT_before_i = int(DT_before/experiment.dt)
        DT_after_i  = int(DT_after/experiment.dt)
        
        
        for s in experiment.spikes :
            
            lb = max(0, s - DT_before_i)
            ub = min(L, s + DT_after_i)
            
            LR_flag[ lb : ub] = 1
            
        
        indices = np.where(LR_flag==0)[0]  

        return indices

def fitSubthresholdDynamics_Build_Xmatrix_Yvector(experiment, Tref, DT_beforeSpike=5.0):
    """
    Compute the X matrix and the Y vector (i.e. \dot_V_data) used to perfomr the linear regression 
    defined in Eq. 17-18 of Pozzorini et al. 2015 for an individual experimental trace provided as parameter.
    The input parameter trace is an ojbect of class Trace.
    """
            
    # Length of the voltage trace       
    Tref_ind = int(Tref/experiment.dt)
    
    
    # Select region where to perform linear regression (specified in the ROI of individual taces)
    ####################################################################################################
    selection = getROI_FarFromSpikes(experiment, DT_beforeSpike, Tref)
    selection_l = len(selection)
    
    
    # Build X matrix for linear regression (see Eq. 18 in Pozzorini et al. PLOS Comp. Biol. 2015)
    ####################################################################################################
    X = np.zeros( (selection_l, 3) )
    
    # Fill first two columns of X matrix        
    X[:,0] = experiment.V[selection]
    X[:,1] = experiment.I[selection]
    X[:,2] = np.ones(selection_l) 
    
    
    # Compute and fill the remaining columns associated with the spike-triggered current eta               
    X_eta = eta.convolution_Spiketrain_basisfunctions(experiment.getSpikeTimes() + Tref, experiment.T, experiment.dt) 
    X = np.concatenate( (X, X_eta[selection,:]), axis=1 )


    # Build Y vector (voltage derivative \dot_V_data)    
    ####################################################################################################
    Y = np.array( np.concatenate( (np.diff(experiment.V)/experiment.dt, [0]) ) )[selection]      

    return (X, Y)

# Initialize the spike-triggered current eta with an exponential function        

def expfunction_eta(x):
    return 0.2*np.exp(-x/100.0)

def expfunction_gamma(x):
    return 10.0*np.exp(-x/100.0)


        
def fitStaticThreshold(experiment,parameters,eta):
    
    """
    Implement Step 3 of the fitting procedure introduced in Pozzorini et al. PLOS Comb. Biol. 2015
    Instead of directly fitting a dynamic threshold, this function just fit a constant threshold.
    The output of this fit can be used as a smart initial condition to fit the full GIF model (i.e.,
    a model featuting a spike-triggered current gamma). See Pozzorini et al. PLOS Comp. Biol. 2015
    experiment: Experiment object on which the model is fitted.
    """

    print("\nGIF MODEL - Fit static threshold...\n")

    
    

        
    # Define initial conditions (based on the average firing rate in the training set)
    ###############################################################################################
    
    nbSpikes = len(experiment.spikes) #TODO
    duration = experiment.T #TODO
    
    mean_firingrate = 1000.0*nbSpikes/duration      
    print('mean firing rate ',mean_firingrate)
    lambda0 = 1.0
    DV = 50.0
    Vt_star = -np.log(mean_firingrate)*DV


    # Perform maximum likelihood fit (Newton method)    
    ###############################################################################################

    beta0_staticThreshold = [1/DV, -Vt_star/DV] 
    beta_opt = maximizeLikelihood(experiment,parameters,eta, beta0_staticThreshold, buildXmatrix_staticThreshold) 
        
        
    # Store result of constnat threshold fitting  
    ###############################################################################################
    
    DV      = 1.0/beta_opt[0]
    Vt_star = -beta_opt[1]*DV 
    gamma.setFilter_toZero()
    
    parameters.DV = DV
    parameters.Vt_star = Vt_star

    printParameters(parameters)

def fitThresholdDynamics(experiment,parameters,gamma):
                
    """
    Implement Step 3 of the fitting procedure introduced in Pozzorini et al. PLOS Comb. Biol. 2015
    Fit firing threshold dynamics by solving Eq. 20 using Newton method.
    
    experiment: Experiment object on which the model is fitted.
    """        
    
    print("\nGIF MODEL - Fit dynamic threshold...\n")
    
    

    print(gamma.getNbOfBasisFunctions())    
    # Perform maximum likelihood fit (Newton method) 
    ###############################################################################################
    # Define initial conditions
    beta0_dynamicThreshold = np.concatenate( ( [1/parameters.DV], [-parameters.Vt_star/parameters.DV], gamma.getCoefficients()/parameters.DV))

    beta_opt = maximizeLikelihood(experiment,parameters,gamma, beta0_dynamicThreshold, buildXmatrix_dynamicThreshold)

    
    # Store result
    ###############################################################################################
    
    DV      = 1.0/beta_opt[0]
    Vt_star = -beta_opt[1]*DV 
    gamma.setFilter_Coefficients(-beta_opt[2:]*DV)

    parameters.DV = DV
    parameters.Vt_star = Vt_star

    printParameters(parameters)        
    
def maximizeLikelihood(experiment,parameters,filter , beta0, buildXmatrix, maxIter=10**3, stopCond=10**-6) :

    ###
    ### THIS IMPLEMENTATION IS NOT SO COOL :(
    ### IN NEW VERSION OF THE CODE I SHOULD IMPLEMENT A NEW CLASS THAT TAKES CARE OF MAXLIKELIHOOD ON lambda=exp(Xbeta) model
    ###
    
    """
    Maximize likelihood. This function can be used to fit any model of the form lambda=exp(Xbeta).
    This function is used to fit both:
    - static threshold
    - dynamic threshold
    The difference between the two functions is in the size of beta0 and the returned beta, as well
    as the function buildXmatrix.
    """
    
    # Precompute all the matrices used in the gradient ascent (see Eq. 20 in Pozzorini et al. 2015)
    ################################################################################################
    
    # here X refer to the matrix made of y vectors defined in Eq. 21 (Pozzorini et al. 2015)
    # since the fit can be perfomed on multiple traces, we need lists
    all_X        = []           
    
    # similar to X but only contains temporal samples where experimental spikes have been observed 
    # storing this matrix is useful to improve speed when computing the likelihood as well as its derivative
    all_X_spikes = []
    
    # sum X_spikes over spikes. Precomputing this quantity improve speed when the gradient is evaluated
    all_sum_X_spikes = []
    
    
    # variables used to compute the loglikelihood of a Poisson process spiking at the experimental firing rate
    T_tot = 0.0
    N_spikes_tot = 0.0
    
    traces_nb = 1
    
    # for tr in experiment.trainingset_traces:
        
    #     if tr.useTrace :              
            
    #         traces_nb += 1
            
    # Simulate subthreshold dynamics 
    (time, V_est, eta_sum_est) = simulateDeterministic_forceSpikes(experiment, parameters, filter ,experiment.I, experiment.V[0], experiment.getSpikeTimes())
                    
    # Precomputes matrices to compute gradient ascent on log-likelihood
    # depeinding on the model being fitted (static vs dynamic threshodl) different buildXmatrix functions can be used
    (X_tmp, X_spikes_tmp, sum_X_spikes_tmp, N_spikes, T) = buildXmatrix(experiment, V_est,filter) 
        
    T_tot        += T
    N_spikes_tot += N_spikes
        
    all_X.append(X_tmp)
    all_X_spikes.append(X_spikes_tmp)
    all_sum_X_spikes.append(sum_X_spikes_tmp)
    
    # Compute log-likelihood of a poisson process (this quantity is used to normalize the model log-likelihood)
    ################################################################################################
    
    logL_poisson = N_spikes_tot*(np.log(N_spikes_tot/T_tot)-1)


    # Perform gradient ascent
    ################################################################################################

    print("Maximize log-likelihood (bit/spks)...")
    lambda0 = 1
    beta = beta0
    # print('beta ',beta)
    old_L = 1

    for i in range(maxIter) :
        
        learning_rate = 1.0
        
        # In the first iterations using a small learning rate makes things somehow more stable
        if i<=10 :                      
            learning_rate = 0.1
        
        
        L=0; G=0; H=0;  
            
        for trace_i in np.arange(traces_nb):
            # compute log-likelihood, gradient and hessian on a specific trace (note that the fit is performed on multiple traces)
            (L_tmp,G_tmp,H_tmp) = computeLikelihoodGradientHessian(experiment,lambda0,beta, all_X[trace_i], all_X_spikes[trace_i], all_sum_X_spikes[trace_i])
            
            # note that since differentiation is linear: gradient of sum = sum of gradient ; hessian of sum = sum of hessian
            L+=L_tmp 
            G+=G_tmp 
            H+=H_tmp
        
        
        # Update optimal parametes (ie, implement Newton step) by tacking into account multiple traces
        
        beta = beta - learning_rate*np.dot(np.linalg.inv(H),G)
            
        if (i>0 and abs((L-old_L)/old_L) < stopCond) :              # If converged
            print("\nConverged after %d iterations!\n" % (i+1))
            break
        
        old_L = L
        
        # Compute normalized likelihood (for print)
        # The likelihood is normalized with respect to a poisson process and units are in bit/spks
        L_norm = (L-logL_poisson)/np.log(2)/N_spikes_tot
        print(L_norm)
        
        if np.isnan(L_norm):
            print("Problem during gradient ascent. Optimizatino stopped.")
            break

    if (i==maxIter - 1) :                                           # If too many iterations
        
        print("\nNot converged after %d iterations.\n" % (maxIter))


    return beta
    
def computeLikelihoodGradientHessian(experiment,lambda0, beta, X, X_spikes, sum_X_spikes) : 
    
    """
    Compute the log-likelihood, its gradient and hessian for a model whose 
    log-likelihood has the same form as the one defined in Eq. 20 (Pozzorini et al. PLOS Comp. Biol. 2015)
    """
    
    # IMPORTANT: in general we assume that the lambda_0 = 1 Hz
    # The parameter lambda0 is redundant with Vt_star, so only one of those has to be fitted.
    # We genearlly fix lambda_0 adn fit Vt_star
            
    dt = experiment.dt/1000.0     # put dt in units of seconds (to be consistent with lambda_0)
    # print('X_spikes shape',np.array(X_spikes).shape,' beta ',np.array(beta).shape)
    X_spikesbeta    = np.dot(X_spikes,beta)
    Xbeta           = np.dot(X,beta)
    expXbeta        = np.exp(Xbeta)

    # Compute loglikelihood defined in Eq. 20 Pozzorini et al. 2015
    L = sum(X_spikesbeta) - lambda0*dt*sum(expXbeta)
                                    
    # Compute its gradient
    G = sum_X_spikes - lambda0*dt*np.dot(np.transpose(X), expXbeta)
    
    # Compute its Hessian
    H = -lambda0*dt*np.dot(np.transpose(X)*expXbeta, X)
    
    return (L,G,H)

def buildXmatrix_staticThreshold(experiment, V_est,filter) :

    """
    Use this function to fit a model in which the firing threshold dynamics is defined as:
    V_T(t) = Vt_star (i.e., no spike-triggered movement of the firing threshold).
    This function computes the matrix X made of vectors y simlar to the ones defined in Eq. 21 (Pozzorini et al. 2015).
    In contrast ot Eq. 21, the X matrix computed here does not include the columns related to the spike-triggered threshold movement.
    """        
    
    # Get indices be removing absolute refractory periods (-self.dt is to not include the time of spike)       
    selection = getROI_FarFromSpikes(experiment,-experiment.dt,-experiment.Tref )
    T_l_selection  = len(selection)

        
    # Get spike indices in coordinates of selection   
    spk_train = experiment.getBinarySpikeTrain()
    spks_i_afterselection = np.where(spk_train[selection]==1)[0]


    # Compute average firing rate used in the fit   
    T_l = T_l_selection*experiment.dt/1000.0                # Total duration of trace used for fit (in s)
    N_spikes = len(spks_i_afterselection)           # Nb of spikes in the trace used for fit

    
    # Define X matrix
    X       = np.zeros((T_l_selection, 2))
    X[:,0]  = V_est[selection]
    X[:,1]  = np.ones(T_l_selection)
    
    # Select time steps in which the neuron has emitted a spike
    X_spikes = X[spks_i_afterselection,:]
        
    # Sum X_spike over spikes    
    sum_X_spikes = np.sum( X_spikes, axis=0)
    
    return (X, X_spikes, sum_X_spikes, N_spikes, T_l)
        
def buildXmatrix_dynamicThreshold( experiment, V_est, filter) :

    """
    Use this function to fit a model in which the firing threshold dynamics is defined as:
    V_T(t) = Vt_star + sum_i gamma(t-\hat t_i) (i.e., model with spike-triggered movement of the threshold)
    This function computes the matrix X made of vectors y defined as in Eq. 21 (Pozzorini et al. 2015).
    """
        
    # Get indices be removing absolute refractory periods (-self.dt is to not include the time of spike)       
    selection = getROI_FarFromSpikes(experiment, -experiment.dt, experiment.Tref)
    T_l_selection  = len(selection)

        
    # Get spike indices in coordinates of selection   
    spk_train = experiment.getBinarySpikeTrain()
    spks_i_afterselection = np.where(spk_train[selection]==1)[0]


    # Compute average firing rate used in the fit   
    T_l = T_l_selection*experiment.dt/1000.0                # Total duration of trace used for fit (in s)
    N_spikes = len(spks_i_afterselection)           # Nb of spikes in the trace used for fit
    
    
    # Define X matrix
    X       = np.zeros((T_l_selection, 2))
    X[:,0]  = V_est[selection]
    X[:,1]  = np.ones(T_l_selection)
    
    # Compute and fill the remaining columns associated with the spike-triggered current gamma              
    X_gamma = filter.convolution_Spiketrain_basisfunctions(experiment.getSpikeTimes() + experiment.Tref, experiment.T, experiment.dt)

    X = np.concatenate( (X, X_gamma[selection,:]), axis=1 )

    # Precompute other quantities to speedup fitting
    X_spikes = X[spks_i_afterselection,:]
    sum_X_spikes = np.sum( X_spikes, axis=0)
    return (X, X_spikes, sum_X_spikes,  N_spikes, T_l)




if __name__ == "__main__":

    experiment = EphysData(I_data,V_data,spikes_data,1/20,20*1000)

    parameter_glif = Parameter


    fitVoltageReset(experiment=experiment,parameters=parameter_glif,Tref=4)

    eta     = Filter_Rect_LogSpaced()    # nA, spike-triggered current (must be instance of class Filter)
    gamma   = Filter_Rect_LogSpaced()    # mV, spike-triggered movement of the firing threshold (must be instance of class Filter)

    eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)
    gamma.setMetaParameters(length=500.0, binsize_lb=5.0, binsize_ub=1000.0, slope=5.0)

    gamma.setFilter_Function(expfunction_gamma)        
            
    eta.setFilter_Function(expfunction_eta)

    experiment.Tref = 4
    parameter_glif.Tref = experiment.Tref
    parameter_glif.dt = experiment.dt

    fitSubthresholdDynamics(experiment,eta,parameter_glif)


    fitStaticThreshold(experiment=experiment,parameters=parameter_glif,eta=eta)
    fitThresholdDynamics(experiment,parameter_glif,gamma)


    I_step = np.zeros(int(20/parameter_glif.dt))
    I_step[int(8/parameter_glif.dt):int(10/parameter_glif.dt)] = 100
    parameter_glif.lambda0 = 1
    # parameter_glif.Tref = parameter_glif.Tref*1000
    print('******Running Simulation ****************')
    time,V, eta_sum, V_T, spks = simulate(params=parameter_glif, I =experiment.I[:2*20000],V0=-66,eta=eta,gamma=gamma)

    print('******Finished ****************')
    plt.plot(time[:20000], V[:20000])
    plt.plot(time[:20000], experiment.V[:20000],alpha=0.3)
    plt.show()
