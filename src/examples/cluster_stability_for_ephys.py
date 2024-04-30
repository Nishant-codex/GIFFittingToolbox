import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
from sknetwork.clustering import Louvain,get_modularity
import sys 
import os 
from Cluster_stability import *
import pickle

if __name__ == "__main__":
    ephys = pd.read_csv("D:/Cluster_stability/ephys_EI_class.csv")
    biophys = pd.read_csv("D:/Cluster_stability/biophys_EI_class.csv")
    cols_to_compare_ephys = ephys.columns[1:-5]
    cols_to_compare_biophys = ['tau_m (ms)', 'R (MOhm):', 'C (nF):', 'gl (nS):', 'El (mV):', 'Vr (mV):', 'Vt* (mV):', 'DV (mV):']


    ephys_E = ephys[ephys.labels_wave ==1]
    ephys_I = ephys[ephys.labels_wave ==0]

    biophys_E = biophys[biophys.labels_wave ==1]
    biophys_I = biophys[biophys.labels_wave ==0]

    ephys_e_cluster_data = find_optimum_res_with_cols(ephys_E[cols_to_compare_ephys].to_numpy(),list(cols_to_compare_ephys))
    ephys_i_cluster_data = find_optimum_res_with_cols(ephys_I[cols_to_compare_ephys].to_numpy(),list(cols_to_compare_ephys))

    biophys_e_cluster_data = find_optimum_res_with_cols(biophys_E[cols_to_compare_biophys].to_numpy(),list(cols_to_compare_biophys))
    biophys_i_cluster_data = find_optimum_res_with_cols(biophys_I[cols_to_compare_biophys].to_numpy(),list(cols_to_compare_biophys))

    path_save = os.getcwd()
    with open(path_save+"/cluster_stablity.pkl",'wb') as f:
        pickle.dump({'e_ephys':ephys_E,
                     'i_ephys':ephys_I,
                     'e_biophys':biophys_E,
                     'i_biophys':biophys_I,},ls
                     
                     file=f)