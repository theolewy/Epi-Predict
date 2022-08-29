
import sys
sys.path.append("../") 
sys.path.append("../../") 

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpyro
import pandas as pd
import arviz as az
import csv


from epimodel import run_model, default_model, arma_model, latent_nn_model, EpidemiologicalParameters, preprocess_data
from epimodel.models.model_predict_utils import *
from epimodel.models.model_build_utils import *

# dates for data processing
start_date="2020-08-01"
end_date="2020-12-10"

prediction_date = pd.to_datetime(end_date)

# Gets data
data = preprocess_data('../../data/all_merged_data_2021-01-22.csv', start_date=start_date, end_date=end_date)
all_data = preprocess_data('../../data/all_merged_data_2021-01-22.csv')

data.featurize()
all_data.featurize()

# Num days to predict after end of data

look_ahead = 23

ep = EpidemiologicalParameters()

samples = []

for sample_name in os.listdir('../samples/arma/learn/cross_validate'):

    f_name = '../samples/arma/learn/cross_validate/' + sample_name
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

cases_nrmse_sums_list = []
deaths_nrmse_sums_list = []

cases_ncrps_sums_list = []
deaths_ncrps_sums_list = []

hyperparameters_list = []

settings = sys.argv[1]

if settings == '0':
    ignore_last_list = [0]
elif settings == '3':
    ignore_last_list = [3]
elif settings == '6':
    ignore_last_list = [6]
elif settings == '9':
    ignore_last_list = [9]
else:
    ignore_last_list = [0,3,6,9]

    
print(ignore_last_list)

for ignore_last in ignore_last_list: 
        
    cases_nrmse_list = []
    deaths_nrmse_list = []

    cases_ncrps_list = []
    deaths_ncrps_list = []
    
    labels = []

    for ind, sample in enumerate(samples):

        print(ignore_last, ind)
        
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
                                                                              ignore_last_days=ignore_last)
        cases_list.append(cases_total_arma)
        deaths_list.append(deaths_total_arma)
        Rt_list.append(Rt_total_arma)
        Rt_cms_list.append(Rt_cms_total_arma)
        Rt_noise_list.append(Rt_noise_total_arma)

        cases_nrmse_arma, cases_ncrps_arma = get_prediction_metrics2(cases_total_arma, all_data.new_cases.data, data.nDs, verbose=False, skip_crps=False)
        deaths_nrmse_arma, deaths_ncrps_arma = get_prediction_metrics2(deaths_total_arma, all_data.new_deaths.data, data.nDs, verbose=False, skip_crps=False)

        cases_nrmse_list.append(cases_nrmse_arma)
        deaths_nrmse_list.append(deaths_nrmse_arma)
        cases_ncrps_list.append(cases_ncrps_arma)
        deaths_ncrps_list.append(deaths_ncrps_arma)

        cases_nrmse_sums_list.append(jnp.sum(cases_nrmse_arma[-look_ahead:]))
        deaths_nrmse_sums_list.append(jnp.sum(deaths_nrmse_arma[-look_ahead:]))
        cases_ncrps_sums_list.append(jnp.sum(cases_ncrps_arma[-look_ahead:]))
        deaths_ncrps_sums_list.append(jnp.sum(deaths_ncrps_arma[-look_ahead:]))
        
    
    f_name1 = 'arma_p_q_(ignore_last_' + str(ignore_last) + ')_cases_nrmse.png'
    f_name2 = 'arma_p_q_(ignore_last_' + str(ignore_last) + ')_deaths_nrmse.png'
    f_name3 = 'arma_p_q_(ignore_last_' + str(ignore_last) + ')_cases_ncrps.png'
    f_name4 = 'arma_p_q_(ignore_last_' + str(ignore_last) + ')_deaths_ncrps.png'

    plot_graph(tuple(cases_nrmse_list), f_name1 , prediction_date, label=labels, region=None, title='Cases NRMSE')
    plot_graph(tuple(deaths_nrmse_list), f_name2 , prediction_date, label=labels, region=None, title='Deaths NRMSE')

    plot_graph(tuple(cases_ncrps_list), f_name3, prediction_date, label=labels, region=None, title='Cases NCRPS')
    plot_graph(tuple(deaths_ncrps_list), f_name4 , prediction_date, label=labels, region=None, title='Deaths NCRPS')
    
    
if 'arma_validation.csv' in os.listdir():
    f = open('arma_validation.csv', 'a')
    writer = csv.writer(f,  lineterminator='\n')
else:
    f = open('arma_validation.csv', 'w')
    writer = csv.writer(f,  lineterminator='\n')
    writer.writerow(('hyperparameters','case_nmse', 'death_nmse', 'case_ncrps', 'death_ncrps'))
        
for i in range(len(cases_nrmse_sums_list)):
    line = (hyperparameters_list[i], cases_nrmse_sums_list[i], deaths_nrmse_sums_list[i], cases_ncrps_sums_list[i], deaths_ncrps_sums_list[i])
    writer.writerow(line)
    
f.close()
