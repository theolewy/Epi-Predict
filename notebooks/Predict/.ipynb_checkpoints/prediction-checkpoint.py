import sys
sys.path.append("../") 

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpyro
import pandas as pd
import arviz as az

numpyro.set_host_device_count(4)

from epimodel import run_model, default_model, arma_model, latent_nn_model, EpidemiologicalParameters, preprocess_data
from epimodel.models.model_predict_utils import *
from epimodel.models.model_build_utils import *




# dates for data processing
start_date="2020-08-01"
end_date="2021-01-02"

# Gets data
data = preprocess_data('../data/all_merged_data_2021-01-22.csv', start_date=start_date, end_date=end_date)
all_data = preprocess_data('../data/all_merged_data_2021-01-22.csv')

data.featurize()
all_data.featurize()




# Runs model, getting all infered values (eg Rt, total_infections etc)

ep = EpidemiologicalParameters()

arma_params = [[1,0], [0,1], [1,1], [2,0], [2,1], [2,2], [1,2], [0,2]]

for arma_param in arma_params:
    
    output_fname = 'p' + str(arma_param[0]) + 'q' + str(arma_param[1]) +'_some.netcdf'

    model_settings = {'cm_method':'nn',
                      'noise_method':'arma_fix',
                      'interact':[],
                      'arma_p':arma_param[0],
                      'arma_q':arma_param[1],
                      'noise_period':7}


    samples, warmup_samples, info, mcmc = run_model(arma_model, 
                                                    data, 
                                                    ep,  
                                                    num_samples=25, 
                                                    target_accept=0.75, 
                                                    num_warmup=50, 
                                                    num_chains=4, 
                                                    save_results=True, 
                                                    output_fname=output_fname,
                                                    model_kwargs=model_settings, 
                                                    max_tree_depth=15)





