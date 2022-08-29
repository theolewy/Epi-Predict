
import sys, os
sys.path.append("../") 
sys.path.append("../../") 

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpyro
import pandas as pd
import arviz as az


from epimodel import run_model, default_model, arma_model, latent_nn_model, EpidemiologicalParameters, preprocess_data
from epimodel.models.model_predict_utils import *
from epimodel.models.model_build_utils import *

# dates for data processing
start_date="2020-08-01"
end_date="2020-12-10"

# Gets data
data = preprocess_data('../../data/all_merged_data_2021-01-22.csv', start_date=start_date, end_date=end_date)
all_data = preprocess_data('../../data/all_merged_data_2021-01-22.csv')

data.featurize()
all_data.featurize()

# Num days to predict after end of data
skip_crps = False

look_ahead = 23

ep = EpidemiologicalParameters()

samples = []

split_num = sys.argv[1]

for sample_name in os.listdir('../samples/nn/cross_validate/split'+split_num):

    f_name = '../samples/nn/cross_validate/split' + split_num + '/' + sample_name
    
    try:
        samples.append(netcdf_to_dict(f_name))
    except:
        pass

settings_list = []

cases_nmse_list = []
deaths_nmse_list = []

cases_ncrps_list = []
deaths_ncrps_list = []

cases_sums_nmse_list = []
deaths_sums_nmse_list = []

cases_sums_ncrps_list = []
deaths_sums_ncrps_list = []

# for sample in samples:
for ind, sample in enumerate(samples):
    
    print(ind)
    
    settings_list.append(samples_to_model_settings(sample))
    
    cases_total_nn, deaths_total_nn, Rt_total_nn, cfr_total_nn = nn_predictor(sample, data, ep, look_ahead)
    
    cases_nmse, cases_ncrps = get_prediction_metrics2(cases_total_nn, all_data.new_cases.data, data.nDs, verbose=False, skip_crps=skip_crps)
    deaths_nmse, deaths_ncrps = get_prediction_metrics2(deaths_total_nn, all_data.new_deaths.data, data.nDs, verbose=False, skip_crps=skip_crps)
    
    cases_nmse_list.append(cases_nmse)
    deaths_nmse_list.append(deaths_nmse)
    
    cases_ncrps_list.append(cases_ncrps)
    deaths_ncrps_list.append(deaths_ncrps)
    
    cases_sums_nmse_list.append(jnp.sum(cases_nmse[-look_ahead:]))
    deaths_sums_nmse_list.append(jnp.sum(deaths_nmse[-look_ahead:]))
    
    cases_sums_ncrps_list.append(jnp.sum(cases_ncrps[-look_ahead:]))
    deaths_sums_ncrps_list.append(jnp.sum(deaths_ncrps[-look_ahead:]))
    
    line = (settings_list[-1], cases_sums_nmse_list[-1], deaths_sums_nmse_list[-1], cases_sums_ncrps_list[-1], deaths_sums_ncrps_list[-1])
    
    if 'nn_validation.csv' in os.listdir():
        f = open('nn_validation.csv', 'a')
        writer = csv.writer(f,  lineterminator='\n')
    else:
        f = open('nn_validation.csv', 'w')
        writer = csv.writer(f,  lineterminator='\n')
        writer.writerow(('hyperparameters','case_nmse', 'death_nmse', 'case_ncrps', 'death_ncrps'))
        
    writer.writerow(line)
    f.close()
    
prediction_date = pd.to_datetime(end_date)

plot_graph(tuple(cases_nmse_list), 'nn_cases_nmse.png', prediction_date, title='Cases NMSE', region=None)
plot_graph(tuple(deaths_nmse_list), 'nn_deaths_nmse.png', prediction_date, title='Deaths NMSE', region=None)

plot_graph(tuple(cases_ncrps_list), 'nn_cases_ncrps.png', prediction_date, title='Cases NCRPS', region=None)
plot_graph(tuple(deaths_ncrps_list), 'nn_deaths_ncrps.png', prediction_date, title='Deaths NCRPS', region=None)

        
