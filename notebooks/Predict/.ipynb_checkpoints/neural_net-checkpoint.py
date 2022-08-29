import sys
sys.path.append("../") 
sys.path.append("../../") 


import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpyro
import pandas as pd
import arviz as az
import random
import copy

numpyro.set_host_device_count(4)

from epimodel import run_model_with_settings, default_model, arma_model, latent_nn_model, latent_nn_model_legacy, EpidemiologicalParameters, preprocess_data
from epimodel.models.model_predict_utils import *
from epimodel.models.model_build_utils import *


# dates for data processing
start_date="2020-08-01"
end_date="2020-12-10"



ep = EpidemiologicalParameters()

ind = int(sys.argv[1])
num_warmup = int(sys.argv[2])
num_samples = int(sys.argv[3])

# Runs model from scratch, getting all infered values (eg Rt, total_infections etc)

settings_list = []

settings_list.append({'preprocessed_data': 'None',
                    'num_percentiles':None,
                      
                    'input_death':None,
                    'n_days_nn_input':None,
                    'D_layers': None})

settings_list.append({'preprocessed_data': 'moving_average',
                    'num_percentiles':None,
                      
                    'input_death':None,
                    'n_days_nn_input':None,
                    'D_layers': None})

settings_list.append({'preprocessed_data': 'summary',
                    'num_percentiles':7,
                      
                    'input_death':None,
                    'n_days_nn_input':None,
                    'D_layers': None})

settings_list.append({'preprocessed_data': 'summary',
                    'num_percentiles':11,
                      
                    'input_death':None,
                    'n_days_nn_input':None,
                    'D_layers': None})

settings_list.append({'preprocessed_data': 'summary',
                    'num_percentiles':15,
                      
                    'input_death':None,
                    'n_days_nn_input':None,
                    'D_layers': None})

settings_list.append({'preprocessed_data': 'moving_average_summary',
                    'num_percentiles':7,
                      
                    'input_death':None,
                    'n_days_nn_input':None,
                    'D_layers': None})

settings_list.append({'preprocessed_data': 'moving_average_summary',
                    'num_percentiles':11,
                      
                    'input_death':None,
                    'n_days_nn_input':None,
                    'D_layers': None})

settings_list.append({'preprocessed_data': 'moving_average_summary',
                    'num_percentiles':15,
                      
                    'input_death':None,
                    'n_days_nn_input':None,
                    'D_layers': None})

made_samples = os.listdir('../samples/nn/cross_validate')

print(made_samples)

possible_settings = []

# gets all settings that we have not already saved to samples/nn/cross_validate
for input_death in [True, False]:
    for n_days_nn_input in [14,21,28]:
        for D_layers in [[10,10,5,5], [10,10,5], [20,10]]:

            model_settings_nn = copy.deepcopy(settings_list[ind])

            model_settings_nn['input_death'] = input_death
            model_settings_nn['n_days_nn_input'] = n_days_nn_input
            model_settings_nn['D_layers'] = D_layers

            f_name = end_date + '_warmup_'+ str(num_warmup)

            for key, value in model_settings_nn.items():

                 f_name += '_' + key + '_' + str(value)

            f_name += '.netcdf'

            if f_name in made_samples:
                pass
            else:
                possible_settings.append(model_settings_nn)

# randomise the order of the settings we want to test
random.shuffle(possible_settings)

for model_settings_nn in possible_settings:
    
    print(model_settings_nn)
            
    run_model_with_settings(
                end_date="2020-12-10",
                num_samples=num_samples,
                num_warmup=num_warmup,
                save_results=True,
                model_kwargs=model_settings_nn,
            )