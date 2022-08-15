
import sys
sys.path.append("../") 

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpyro
import pandas as pd
import arviz as az
import csv

numpyro.set_host_device_count(4)

from epimodel import run_model, default_model, arma_model, latent_nn_model, EpidemiologicalParameters, preprocess_data
from epimodel.models.model_predict_utils import *
from epimodel.models.model_build_utils import *

# dates for data processing
start_date="2020-08-01"
end_date="2020-12-10"

prediction_date = pd.to_datetime(end_date)

# Gets data
data = preprocess_data('../data/all_merged_data_2021-01-22.csv', start_date=start_date, end_date=end_date)
all_data = preprocess_data('../data/all_merged_data_2021-01-22.csv')

data.featurize()
all_data.featurize()

# Num days to predict after end of data

look_ahead = 23

ep = EpidemiologicalParameters()

samples = []

for sample_name in os.listdir('samples/arma/learn/cross_validate'):

    f_name = 'samples/arma/learn/cross_validate/' + sample_name
    try:
        samples.append(netcdf_to_dict(f_name))
    except:
        continue
        
        
all_settings = []
for sample in samples:
    
    all_settings.append(samples_to_model_settings(sample))


future_cms = all_data.active_cms[:,:, data.nDs:]

cases_list = []
deaths_list = []
Rt_list = []
Rt_cms_list = []
Rt_noise_list = []

cases_mse_sums_list = []
deaths_mse_sums_list = []

cases_crps_sums_list = []
deaths_crps_sums_list = []

hyperparameters_list = []

for ignore_last in [0,2,4,6,8,10]: #
    
    print(ignore_last)
        
    cases_mse_list = []
    deaths_mse_list = []

    cases_crps_list = []
    deaths_crps_list = []
    
    labels = []

    for ind, sample in enumerate(samples):
        
        arma_p = all_settings[ind]['arma_p']
        arma_q = all_settings[ind]['arma_q']
        
        hyperparameters = [arma_p, arma_q, ignore_last]
        
        labels.append('(p,q)=(' + str(arma_p) +',' + str(arma_q) +')')
        hyperparameters_list.append(hyperparameters)
        
        cases_total_arma, deaths_total_arma, Rt_total_arma, \
                Rt_cms_total_arma, Rt_noise_total_arma = arma_noise_predictor(sample, 
                                                                              ep,                  
                                                                              data.active_cms, 
                                                                              look_ahead, 
                                                                              future_cms=future_cms,        
                                                                              ignore_last_days=5)
        cases_list.append(cases_total_arma)
        deaths_list.append(deaths_total_arma)
        Rt_list.append(Rt_total_arma)
        Rt_cms_list.append(Rt_cms_total_arma)
        Rt_noise_list.append(Rt_noise_total_arma)

        cases_mse_arma, cases_crps_arma = get_prediction_metrics(cases_total_arma, all_data.new_cases.data, data.nDs, verbose=False, skip_crps=False)
        deaths_mse_arma, deaths_crps_arma = get_prediction_metrics(deaths_total_arma, all_data.new_deaths.data, data.nDs, verbose=False, skip_crps=False)

        cases_mse_list.append(cases_mse_arma)
        deaths_mse_list.append(deaths_mse_arma)
        cases_crps_list.append(cases_crps_arma)
        deaths_crps_list.append(deaths_crps_arma)

        cases_mse_sums_list.append(jnp.sum(cases_mse_arma[-look_ahead:]))
        deaths_mse_sums_list.append(jnp.sum(deaths_mse_arma[-look_ahead:]))
        cases_crps_sums_list.append(jnp.sum(cases_crps_arma[-look_ahead:]))
        deaths_crps_sums_list.append(jnp.sum(deaths_crps_arma[-look_ahead:]))
        
    
    f_name1 = 'arma_p_q_(ignore_last_' + str(ignore_last) + ')_cases_mse.png'
    f_name2 = 'arma_p_q_(ignore_last_' + str(ignore_last) + ')_deaths_mse.png'
    f_name3 = 'arma_p_q_(ignore_last_' + str(ignore_last) + ')_cases_crps.png'
    f_name4 = 'arma_p_q_(ignore_last_' + str(ignore_last) + ')_deaths_crps.png'

    plot_graph(tuple(cases_mse_list), f_name1 , prediction_date, label=labels, region=None, title='Cases MSE')
    plot_graph(tuple(deaths_mse_list), f_name2 , prediction_date, label=labels, region=None, title='Deaths MSE')

    plot_graph(tuple(cases_crps_list), f_name3, prediction_date, label=labels, region=None, title='Cases CRPS')
    plot_graph(tuple(deaths_crps_list), f_name4 , prediction_date, label=labels, region=None, title='Deaths CRPS')
    

    
f = open('arma_validation.csv', 'w')
        
writer = csv.writer(f,  lineterminator='\n')
writer.writerow(('hyperparameters','case_mse', 'death_mse', 'case_crps', 'death_crps'))

for i in range(len(cases_mse_sums_list)):
    line = (hyperparameters_list[i], cases_mse_sums_list[i], deaths_mse_sums_list[i], cases_crps_sums_list[i], deaths_crps_sums_list[i])
    writer.writerow(line)
    
f.close()
