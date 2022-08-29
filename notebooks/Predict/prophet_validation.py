
import sys
sys.path.append("../")
sys.path.append("../../")

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
end_date="2020-12-10"

# Gets data
data = preprocess_data('../../data/all_merged_data_2021-01-22.csv', start_date=start_date, end_date=end_date)
all_data = preprocess_data('../../data/all_merged_data_2021-01-22.csv')

data.featurize()
all_data.featurize()

skip_crps=False
verbose=False

look_ahead = 23

cases_list = []

nmse_cases_list = []
ncrps_cases_list = []

sum_nmse_cases_list = []
sum_ncrps_cases_list = []


floor_cap_list_cases = []

cap_case_list = [10000, 12000, 24000, 36000, 48000, 60000]


for floor_case in [0]:
    for cap_case in cap_case_list:

        floor_cap = [floor_case, cap_case, 0, 100]
        floor_cap_list_cases.append(floor_cap)

        cases_total_prophet, deaths_total_prophet = prophet_predictor(data, look_ahead, end_date, "2020-08-01", floor_cap, nS=100)

        cases_nmse_prophet, cases_ncrps_prophet = get_prediction_metrics2(cases_total_prophet, all_data.new_cases.data, data.nDs, skip_crps=skip_crps, verbose=verbose)

        nmse_cases_list.append(cases_nmse_prophet)
        ncrps_cases_list.append(cases_ncrps_prophet)

        sum_nmse_cases_list.append(jnp.sum(cases_nmse_prophet[-23:]))
        sum_ncrps_cases_list.append(jnp.sum(cases_ncrps_prophet[-23:]))

        
prediction_date = pd.to_datetime(end_date)
labels = ['cap=10000', 'cap=12000', 'cap=24000', 'cap=36000', 'cap=48000', 'cap=60000']


plot_graph(tuple(nmse_cases_list), 'cases_nmse_prophet.png', prediction_date,label=labels,region=None,title='Cases NMSE')
plot_graph(tuple(ncrps_cases_list), 'cases_ncrps_prophet.png',prediction_date,label=labels,region=None,title='Cases NCRPS')

skip_crps=False
verbose=False

look_ahead = 23

deaths_list = []

nmse_deaths_list = []
ncrps_deaths_list = []

sum_ncrps_deaths_list = []
sum_nmse_deaths_list = []

floor_cap_list_death = []

cap_death_list = [300,450,800, 1600, 3200, 4800, 6400, 8000]


for floor_death in [0]: #1
    for cap_death in cap_death_list:

        floor_cap = [0, 10000, floor_death, cap_death]
        floor_cap_list_death.append(floor_cap)

        cases_total_prophet, deaths_total_prophet = prophet_predictor(data, look_ahead, end_date, start_date, floor_cap, nS=100)

        deaths_nmse_prophet, deaths_ncrps_prophet = get_prediction_metrics2(deaths_total_prophet, all_data.new_deaths.data, data.nDs, skip_crps=skip_crps, verbose=verbose)

        nmse_deaths_list.append(deaths_nmse_prophet)
        ncrps_deaths_list.append(deaths_ncrps_prophet)

        sum_nmse_deaths_list.append(jnp.sum(deaths_nmse_prophet[-23:]))
        sum_ncrps_deaths_list.append(jnp.sum(deaths_ncrps_prophet[-23:]))
        
prediction_date = pd.to_datetime(end_date)
labels = ['cap=300', 'cap=450', 'cap=800', 'cap=1600', 'cap=3200', 'cap=4800','cap=6400','cap=8000' ]

plot_graph(tuple(nmse_deaths_list), 'deaths_nmse_prophet.png', prediction_date,label=labels,region=None,title='Deaths NMSE')
plot_graph(tuple(ncrps_deaths_list), 'deaths_ncrps_prophet.png',prediction_date,label=labels,region=None,title='Deaths NCRPS')


    
f = open('prophet_validation.csv', 'w')
        
writer = csv.writer(f,  lineterminator='\n')

writer.writerow(('hyperparameters','case_nmse', 'case_ncrps'))

for i in range(len(sum_nmse_cases_list)):
    line = (floor_cap_list_cases[i], sum_nmse_cases_list[i], sum_ncrps_cases_list[i])
    writer.writerow(line)
    
writer.writerow(('hyperparameters','death_nmse', 'death_ncrps'))

for i in range(len(sum_nmse_deaths_list)):
    line = (floor_cap_list_death[i], sum_nmse_deaths_list[i], sum_ncrps_deaths_list[i])
    writer.writerow(line)
    
f.close()


