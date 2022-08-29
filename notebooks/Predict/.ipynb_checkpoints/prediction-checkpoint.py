import sys
sys.path.append("../") 
sys.path.append("../../") 

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpyro
import pandas as pd
import random

import arviz as az

numpyro.set_host_device_count(4)

from epimodel import run_model_with_settings, default_model, latent_nn_model_legacy, arma_model, latent_nn_model, EpidemiologicalParameters, preprocess_data
from epimodel.models.model_predict_utils import *
from epimodel.models.model_build_utils import *


##################################
# Things that might need editing #
##################################

end_date = "2021-01-02"
look_ahead = 20

skip_crps=False
verbose=False

###############################
#### Data and Parameters ######
###############################
all_data = preprocess_data('../../data/all_merged_data_2021-01-22.csv')
all_data.featurize()

data = preprocess_data('../../data/all_merged_data_2021-01-22.csv', end_date=end_date)
data.featurize()

ep = EpidemiologicalParameters()

df = pd.DataFrame()

###############################
######### Epi-ARMA ############
###############################

# samples_arma = netcdf_to_dict('../samples/arma/learn/test/2021-01-02_p2q1_learn.netcdf')
# future_cms = all_data.active_cms[:,:, data.nDs:]

# gets expected cases and deaths
# cases_total_arma, deaths_total_arma, Rt_total_arma, \
#             Rt_cms_total_arma, Rt_noise_total_arma = arma_noise_predictor(samples_arma, 
#                                                                           ep,                  
#                                                                           data.active_cms, 
#                                                                           look_ahead, 
#                                                                           future_cms=future_cms,        
#                                                                           ignore_last_days=0)

# cases_nmse_arma, cases_ncrps_arma = get_prediction_metrics2(cases_total_arma, all_data.new_cases.data, data.nDs, skip_crps=skip_crps, verbose=verbose)

# deaths_nmse_arma, deaths_ncrps_arma = get_prediction_metrics2(deaths_total_arma, all_data.new_deaths.data, data.nDs, skip_crps=skip_crps, verbose=verbose)

# df['arma_case_nmse'] = pd.DataFrame(cases_nmse_arma)
# df['arma_death_nmse'] = pd.DataFrame(deaths_nmse_arma)
# df['arma_case_ncrps'] = pd.DataFrame(cases_ncrps_arma)
# df['arma_death_ncrps'] = pd.DataFrame(deaths_ncrps_arma)

###############################
########### Epi-NN ############
###############################

samples_nn = netcdf_to_dict('../Predict/2.netcdf')

cases_total_nn, deaths_total_nn, Rt_total_nn, cfr_total_nn = nn_predictor(samples_nn, data, ep, look_ahead)

nS = jnp.shape(cases_total_nn)[0]
ind = list(range(0,nS))
random.shuffle(ind)
ind = ind[0:250]

cases_total_nn = cases_total_nn[ind,:]
deaths_total_nn = deaths_total_nn[ind,:]

cases_nmse_nn, cases_ncrps_nn = get_prediction_metrics2(cases_total_nn, all_data.new_cases.data, data.nDs, skip_crps=skip_crps, verbose=verbose)

deaths_nmse_nn, deaths_ncrps_nn = get_prediction_metrics2(deaths_total_nn, all_data.new_deaths.data, data.nDs, skip_crps=skip_crps, verbose=verbose)

df['nn_case_nmse_2'] = pd.DataFrame(cases_nmse_nn)
df['nn_death_nmse_2'] = pd.DataFrame(deaths_nmse_nn)
df['nn_case_ncrps_2'] = pd.DataFrame(cases_ncrps_nn)
df['nn_death_ncrps_2'] = pd.DataFrame(deaths_ncrps_nn)

###############################
########### EpiNow2 ###########
###############################

# cases_total_epinow2 = get_data_from_epinow2()

# cases_nmse_epinow2, cases_ncrps_epinow2 = get_prediction_metrics2(cases_total_epinow2, all_data.new_cases.data, data.nDs, skip_crps=skip_crps, verbose=verbose)

# df['EpiNow2_case_nmse'] = pd.DataFrame(cases_nmse_epinow2)
# df['EpiNow2_case_ncrps'] = pd.DataFrame(cases_ncrps_epinow2)

###############################
########### Prophet ###########
###############################

# cases_total_prophet, deaths_total_prophet = prophet_predictor(data, look_ahead, end_date, nS=100)

# cases_nmse_prophet, cases_ncrps_prophet = get_prediction_metrics2(cases_total_prophet, all_data.new_cases.data, data.nDs, skip_crps=skip_crps, verbose=verbose)

# deaths_nmse_prophet, deaths_ncrps_prophet = get_prediction_metrics2(deaths_total_prophet, all_data.new_deaths.data, data.nDs, skip_crps=skip_crps, verbose=verbose)

# df['prophet_case_nmse'] = pd.DataFrame(cases_nmse_prophet)
# df['prophet_death_nmse'] = pd.DataFrame(deaths_nmse_prophet)
# df['prophet_case_ncrps'] = pd.DataFrame(cases_ncrps_prophet)
# df['prophet_death_ncrps'] = pd.DataFrame(deaths_ncrps_prophet)


df.to_csv('arma_test_2.csv')










