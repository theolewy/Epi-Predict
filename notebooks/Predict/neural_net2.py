import sys
sys.path.append("../") 
sys.path.append("../../") 

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpyro
import pandas as pd
import arviz as az
import random

numpyro.set_host_device_count(4)

from epimodel import run_model_with_settings, default_model, arma_model, latent_nn_model, latent_nn_model_legacy, EpidemiologicalParameters, preprocess_data
from epimodel.models.model_predict_utils import *
from epimodel.models.model_build_utils import *


# dates for data processing
start_date="2020-08-01"
end_date="2020-12-10"

ep = EpidemiologicalParameters()

# Runs model from scratch, getting all infered values (eg Rt, total_infections etc)

model_settings_nn = {'preprocessed_data': 'None',
                    'num_percentiles':None,
                      
                    'input_death':True,
                    'n_days_nn_input':21,
                    'D_layers': [10,10,5]}


run_model_with_settings(
    end_date="2020-12-10",
    num_samples=50,
    num_warmup=100,
    save_results=True,
    model_kwargs=model_settings_nn,
)

