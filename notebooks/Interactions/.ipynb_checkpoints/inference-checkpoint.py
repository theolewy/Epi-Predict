import sys
sys.path.append("../") 

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpyro
import pandas as pd

numpyro.set_host_device_count(4)

from epimodel import run_model, default_model,arma_model, latent_nn_model, EpidemiologicalParameters, preprocess_data
from epimodel.models.model_predict_utils import *
from epimodel.models.model_build_utils import *


# dates for data processing
start_date="2020-08-01"
end_date="2021-01-02"
# end_date="2020-12-10"

# Gets data
data = preprocess_data('../data/all_merged_data_2021-01-22.csv', start_date=start_date, end_date=end_date)
all_data = preprocess_data('../data/all_merged_data_2021-01-22.csv')

data.featurize()
all_data.featurize()


# Runs model, getting all infered values (eg Rt, total_infections etc)
ep = EpidemiologicalParameters()

# get a list of all interact lists, so that all possible pairs occur at some point,
# and each interact list contains each number at most once

from itertools import combinations
import random

a = list(combinations(list(range(19)),2))

interact = list(map(list, a))

random.shuffle(interact)

interacting_lists = []

for i in range(33):
    interacting_lists.append(interact[5*i:5*(i+1)])

interacting_lists.append(interact[5*33:])

all_samples = []
all_model_settings = []

for ind, interact in enumerate(interacting_lists):
    
    print(ind)
    print(interact)
    
    model_settings = {'cm_method':'linear_interact',
                  'noise_method':'no_noise',
                  'interact':interact}
        
    samples, warmup_samples, info, mcmc = run_model(arma_model, 
                                                    data, 
                                                    ep,  
                                                    num_samples=30, 
                                                    target_accept=0.75, 
                                                    num_warmup=60, 
                                                    num_chains=4, 
                                                    save_results=False, 
                                                    model_kwargs=model_settings, 
                                                    max_tree_depth=15)

    all_samples.append(samples)
    all_model_settings.append(model_settings)

# Save interaction data
R_reduction_interaction_all_samples(all_samples, all_model_settings, save_file='interactions3.csv')

