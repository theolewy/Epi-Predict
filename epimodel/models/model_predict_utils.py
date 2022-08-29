import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

from epimodel.models.model_build_utils import *
from epimodel.preprocessing import summarise_some_data


def get_future_R(samples, past_cms, model_settings, future_cms=None, look_ahead=10, ignore_last_days=10):
    
    cm_method = model_settings['cm_method']
    noise_method = model_settings['noise_method']
    noise_period = model_settings['noise_period']
    arma_p = model_settings['arma_p']
    arma_q = model_settings['arma_q']
    interact = model_settings['interact']

    
    # if no future_cms supplied, assume cms_doesn't change in future
    if isinstance(future_cms, type(None)):
        
        future_cms = jnp.repeat(past_cms[:,:,[-1]], look_ahead+ignore_last_days, axis=-1)
    
    # if the inputted future is not long enough, assume no change after it finishes
    elif jnp.shape(future_cms)[-1] < look_ahead+ignore_last_days:
        
        extra_padding = jnp.repeat(future_cms[:,:,[-1]], look_ahead + ignore_last_days - jnp.shape(future_cms)[-1], axis=-1)
        future_cms = jnp.concatenate((future_cms, extra_padding), axis=-1)
        
    # if the inputted future is too long, truncate it
    elif jnp.shape(future_cms)[-1] > look_ahead + ignore_last_days:
        
        future_cms = future_cms[:, :, :look_ahead+ignore_last_days]
    
    # get parameters
    num_samples = len(samples['basic_R'])
    nRs = len(samples['basic_R'][0])
    
    # get basic_R
    basic_R = samples['basic_R']

    nDs = len(samples['Rt'][0][0])
    
    # get cm_reduction
    
    if cm_method =='linear':
        
        alpha_i = jnp.expand_dims(samples['alpha_i'], (1,3))
        alpha_i = jnp.repeat(alpha_i, nRs, axis=1)
        
        future_cms = jnp.expand_dims(future_cms, 0)
        future_cms = jnp.repeat(future_cms, num_samples, axis=0)
        
        cm_reduction = jnp.sum(future_cms * alpha_i, axis=2)
        
    elif cm_method == 'nn':
        
        # if no interacting pairs...
        if len(interact) == 0:
            param_list = num_samples
            
        # if there are interacting pairs...
        else:
            # get the nn parameters
            w1 = jnp.repeat(samples['w1'][:,:,:,[0],:,:], look_ahead+ignore_last_days, axis=3)
            w2 = jnp.repeat(samples['w2'][:,:,:,[0],:,:], look_ahead+ignore_last_days, axis=3)
            w3 = jnp.repeat(samples['w3'][:,:,:,[0],:,:], look_ahead+ignore_last_days, axis=3)

            # put parameters in a list for convenience
            param_list = [w1, w2, w3]

            # use the pairwise NN's to get the linear layer input
        lin_input = get_lin_layer(param_list, future_cms, interact)
        
        # get the linear parameters
        alpha_i = jnp.expand_dims(samples['alpha_i'], (1,3))
        alpha_i = jnp.repeat(alpha_i, nRs, axis=1)
        alpha_i = jnp.repeat(alpha_i, look_ahead+ignore_last_days, axis=3)
        
        # get the reduction
        cm_reduction = jnp.sum(lin_input * alpha_i, axis=2)
        
    # get Rt_noise
    Rt_noise = samples['Rt_noise']

    Rt_noise_unexpanded = Rt_noise[:,:,::noise_period]
    
    if noise_method=='arma_fix' or noise_method=='arma_learn':

        # Note that when we unexpand Rt_noise our lookahead decreases by a factor of noise_period
        effective_lookahead = (look_ahead+ignore_last_days+6)//noise_period
        effective_ignore = (ignore_last_days+6)//noise_period
        
        if arma_p != 0:
            arma_p_coeff = samples['arma_p_coeff']
        else:
            arma_p_coeff = None
            
        if arma_q != 0:
            arma_q_coeff = samples['arma_q_coeff']
        else:
            arma_q_coeff = None
            
        arma_const_coeff = samples['arma_const_coeff']
        arma_noise_scale = samples['arma_error_scale']
        
        arma_noise_scale_repeat = jnp.expand_dims(arma_noise_scale, (1,2))
        
        arma_noise = np.random.normal(
                loc=jnp.zeros((num_samples, nRs, effective_lookahead)),
                scale=arma_noise_scale_repeat
        )
        
        if effective_ignore == 0:
            
            # last p values of logR
            p_padding_logR = jnp.log(samples['Rt_noise'])[:,:,(-arma_p):]
            # last q values of errors
        
            if arma_q != 0:
                q_padding_err = samples['log_R_res'][:,:,(-arma_q):] * arma_noise_scale_repeat[:,:,[0]*arma_q] * 10.0
            else:
                q_padding_err = None

            
        else:
            
            # last p values of logR
            p_padding_logR = jnp.log(samples['Rt_noise'])[:,:,(-arma_p-effective_ignore):-effective_ignore]
            # last q values of errors

            q_padding_err = samples['log_R_res'][:,:,(-arma_q-effective_ignore):-effective_ignore] * arma_noise_scale_repeat[:,:,[0]*arma_q] * 10.0
            
        arima_transitions = get_arima_transition_function(arma_p_coeff, arma_q_coeff, arma_const_coeff)

        _, log_R_arma = jax.lax.scan(arima_transitions, (q_padding_err, p_padding_logR),
                                      jnp.transpose(arma_noise, (2,0,1)))
        
        log_R_arma = jnp.transpose(log_R_arma, (1,2,0))

        Rt_noise_future_unexpanded = jnp.exp(log_R_arma)
        
        Rt_noise_future = jnp.repeat(Rt_noise_future_unexpanded, noise_period, axis=-1)[:,:,:look_ahead+ignore_last_days]
    
    elif noise_method == 'no_noise':
        
        Rt_noise_future = jnp.zeros_like(cm_reduction)
        
    
    # Put basic_R, R_cms and Rt_noise together to get Rt_future
    Rt_future = jnp.exp(jnp.log(basic_R.reshape((num_samples, nRs, 1))) - cm_reduction + jnp.log(Rt_noise_future))
    
    # get total_Rt, total_Rt_cms and total_Rt_noise (include past and future for each)
    if ignore_last_days == 0:
        Rt = samples['Rt']
        R_cms = samples['R_cms']
        Rt_noise = samples['Rt_noise']
    else:
        Rt = samples['Rt'][:,:,:-ignore_last_days]
        R_cms = samples['R_cms'][:,:,:-ignore_last_days]
        Rt_noise = samples['Rt_noise'][:,:,:-ignore_last_days]
    
    R_cms_future = jnp.exp(jnp.log(basic_R.reshape((num_samples, nRs, 1))) - cm_reduction)
    
    total_Rt_noise = concatenate_past_and_future(Rt_noise, Rt_noise_future)
    total_Rt = concatenate_past_and_future(Rt, Rt_future)
    total_Rt_cms = concatenate_past_and_future(R_cms, R_cms_future)
    
    return Rt_future, total_Rt, total_Rt_cms, total_Rt_noise

def get_future_infections(infections, Rt_future, ep):
    """
    Finds future infections by taking most recent infections, and applying consecutive Rt_future transformation
    
    :param infections: sampled infections for days with data
    :param Rt_future: the predicted values of Rt in the future
    :param ep: EpidemiologicalParameters
    
    """

    # get the most recent infection numbers that are needed to calculate the future due to the delay function
    total_padding = ep.GIv.size - 1
    recent_infections = infections[:,:,-total_padding:]

    # get the renewal transition function
    discrete_renewal_transition = get_discrete_renewal_transition(ep)

  # use renewal transition function to find future infections
    num_samples = len(Rt_future)
    num_regions = len(Rt_future[0])
    num_days    = len(Rt_future[0][0])
    future_infections = jnp.zeros((num_samples, num_regions, num_days))

    for i in range(num_samples):

        _, future_infections_sample = jax.lax.scan(
              discrete_renewal_transition,
              recent_infections[i,:,:],
              Rt_future[i,:,:].T)

        future_infections_sample = future_infections_sample.T     # want dim1=region, dim2=time

        future_infections = jax.ops.index_update(
            future_infections,
            jax.ops.index[i, :, :],
            future_infections_sample,
            )

    return future_infections


def concatenate_past_and_future(past, future):

    d1 = len(future)
    d2 = len(future[0])
    d3_past = len(past[0][0])
    d3_future = len(future[0][0])

    total = jnp.zeros((d1, d2, d3_past + d3_future))

    total = jax.ops.index_update(
          total,
          jax.ops.index[:, :, :d3_past],
          past)

    total = jax.ops.index_update(
              total, jax.ops.index[:, :, d3_past:], future)

    return total

# convert future_infections into future_cases, future_deaths

def convert_infections_to_cases_deaths(infections, cfr, ep):

    num_samples = len(infections)
    num_regions = len(infections[0])
    num_days    = len(infections[0][0])-7     # ignore the 7 days of seeding

    expected_cases = jnp.zeros((num_samples, num_regions, num_days))
    expected_deaths = jnp.zeros((num_samples, num_regions, num_days))

    for i in range(num_samples):

        expected_cases_sample = jax.scipy.signal.convolve2d(
                                infections[i,:,:], ep.DPC, mode="full")[:,7:num_days+7]

        deaths = jnp.multiply(infections[i,:,:], cfr[i,:])

        expected_deaths_sample = jax.scipy.signal.convolve2d(
                                  deaths, ep.DPD, mode="full")[:,7:num_days+7]

        expected_cases = jax.ops.index_update(
            expected_cases,
            jax.ops.index[i, :, :],
            expected_cases_sample,
            )

        expected_deaths = jax.ops.index_update(
            expected_deaths,
            jax.ops.index[i, :, :],
            expected_deaths_sample,
            )
        
    return expected_cases, expected_deaths

def get_moving_average(sample):
    
    # sample must be 3 dim, and we get moving average of last dim
    moving_averages = np.zeros_like(sample)
    
    moving_averages[:,:,:6] = sample[:,:,:6]
    
    for i in range(6, len(moving_averages[0][0])):
        moving_averages[:,:,i] = jnp.sum(sample[:,:,i-7:i], axis=2)/7
        
    return moving_averages

def arma_noise_predictor(samples, ep, past_cms, look_ahead, future_cms=None, ignore_last_days=10):
    
    model_settings = samples_to_model_settings(samples)

    Rt = samples['Rt']
    cfr = samples['cfr']
    infections = samples['total_infections']

    # get the last time in the dataset
    end_of_data = len(Rt[0][0])

    # compute the future values of Rt, and produce the various Rt over all time
    Rt_future, total_Rt, total_Rt_cms, total_Rt_noise = get_future_R(samples, past_cms, model_settings, future_cms, look_ahead, ignore_last_days)
    
    if ignore_last_days == 0:
        infections_ignore_last = infections
    else:
        infections_ignore_last = infections[:,:,:-ignore_last_days]
    
    # convert Rt values into future infections
    future_infections = get_future_infections(infections_ignore_last, Rt_future, ep)

    # concatenate to get infections over all time
    total_infections = concatenate_past_and_future(infections_ignore_last, future_infections)

    # convert infections into cases and deaths
    expected_cases, expected_deaths = convert_infections_to_cases_deaths(total_infections, cfr, ep)

    return expected_cases, expected_deaths, total_Rt, total_Rt_cms, total_Rt_noise

def nn_predictor_legacy(samples, data, ep, look_ahead):

    model_settings = samples_to_model_settings(samples)

    Rt = samples['Rt']
    cfr = samples['cfr']
    infections = samples['total_infections']
    expected_cases = samples['expected_cases']
    expected_deaths = samples['expected_deaths']
    
    scaler_mean_cases = samples["scaler_mean_cases"]
    scaler_std_cases = samples["scaler_std_cases"]
    scaler_mean_deaths = samples["scaler_mean_deaths"]
    scaler_std_deaths = samples["scaler_std_deaths"]
    
    n_days_nn_input = model_settings['n_days_nn_input']
    infer_cfr = model_settings['infer_cfr']
    input_death = model_settings['input_death']
    
    # get bnn parameters   
    parameter_list = get_param_list_from_samples(samples)

    # get the number of samples and regions
    nS = len(Rt)
    nRs = len(Rt[0])
    nDs = len(Rt[0][0])
    
    # intialise the outputs
    Rt_total = Rt
    infections_total = infections
    cases_total = expected_cases
    deaths_total = expected_deaths
    cfr_total = cfr
    
    # get case scaler mean and std
    mean_cases = jnp.expand_dims(scaler_mean_cases, 2)
    std_cases = jnp.expand_dims(scaler_std_cases, 2)
    mean_deaths = jnp.expand_dims(scaler_mean_deaths, 2)
    std_deaths = jnp.expand_dims(scaler_std_deaths, 2)
    
    
    for day in range(look_ahead):
        
        # log and scale most recent case data, then format correctly
        log_case_last = (jnp.log(1+cases_total[:,:,-n_days_nn_input:])  - mean_cases)/std_cases
        log_case_last = jnp.expand_dims(log_case_last, (2,3))
        log_death_last = (jnp.log(1+deaths_total[:,:,-n_days_nn_input:])  - mean_deaths)/std_deaths
        log_death_last = jnp.expand_dims(log_death_last, (2,3))
        
        if input_death:
            nn_input = jnp.concatenate((log_case_last, log_death_last), axis=4)
        else:
            nn_input = log_case_last
        
        
        nn_output = bnn_feedforward_nn_method(parameter_list, nn_input)
        nn_output = jnp.squeeze(nn_output, axis=2)  
        
        R_new = nn_output[:,:,[0]]
        Rt_total = jnp.concatenate((Rt_total, R_new), 2)
        
        if infer_cfr:
            cfr_new = nn_output[:,:,[1]]
            cfr_total = jnp.concatenate((cfr_total, cfr_new), 2)
            
        else:
            cfr_total = cfr
            
        dist1 = jnp.expand_dims(ep.GI_flat_rev, (0,2))
        dist2 = jnp.transpose(jnp.expand_dims(ep.DPC, 0), (0,2,1))
        dist3 = jnp.transpose(jnp.expand_dims(ep.DPD, 0), (0,2,1))
        
        new_infections = jnp.multiply(R_new, infections_total[:,:,-(ep.GIv.size - 1):] @ jnp.flip(dist1, 1))

        # append to infections_total
        infections_total = jnp.concatenate((infections_total, new_infections), 2)
        
        new_expected_case = infections_total[:,:,-(ep.DPC.size):] @ jnp.flip(dist2, 1)

        deaths = jnp.multiply(infections_total, cfr_total)

        new_expected_death = deaths[:,:,-(ep.DPD.size):] @ jnp.flip(dist3, 1)
        
        # append deaths and cases to deaths_total and cases_total
        cases_total = jnp.concatenate((cases_total, new_expected_case), 2)
        deaths_total = jnp.concatenate((deaths_total, new_expected_death), 2)
            
                
    return cases_total, deaths_total, Rt_total, cfr_total


def nn_predictor(samples, data, ep, look_ahead, preprocessed_data=False):

    model_settings = samples_to_model_settings(samples)

    Rt = samples['Rt']
    cfr = samples['cfr']
    infections = samples['total_infections']
    expected_cases = samples['expected_cases']
    expected_deaths = samples['expected_deaths']
    max_change_R_nn = samples['max_change_R_nn']
    
    case_weekly_multiplier_base = samples['case_weekly_multiplier']
    death_weekly_multiplier_base = samples['death_weekly_multiplier']

    final_case_multiplier = 1/jnp.expand_dims(jnp.prod(case_weekly_multiplier_base, axis=2), 2)
    final_death_multiplier = 1/jnp.expand_dims(jnp.prod(death_weekly_multiplier_base, axis=2), 2)

    case_weekly_multiplier = jnp.concatenate((case_weekly_multiplier_base, final_case_multiplier), axis=2)
    death_weekly_multiplier = jnp.concatenate((death_weekly_multiplier_base, final_death_multiplier), axis=2)
        
    scaler_mean_cases = data.cases_mean
    scaler_std_cases = data.cases_std
    scaler_mean_deaths = data.deaths_mean
    scaler_std_deaths = data.deaths_std
    
    n_days_nn_input = model_settings['n_days_nn_input']
    num_percentiles = model_settings['num_percentiles']
    infer_cfr = model_settings['infer_cfr']
    input_death = model_settings['input_death']
    preprocessed_data = model_settings['preprocessed_data']


    # get bnn parameters   
    parameter_list_R = get_param_list_from_samples(samples, 'R')
    
    if infer_cfr:
        max_change_cfr_nn = samples['max_change_cfr_nn']
        parameter_list_cfr = get_param_list_from_samples(samples, 'cfr')

    # get the number of samples and regions
    nS = len(Rt)
    nRs = len(Rt[0])
    nDs = len(Rt[0][0])
    
    # intialise the outputs
    Rt_total = Rt
    infections_total = infections
    cases_total = expected_cases
    deaths_total = expected_deaths
    cases_moving_ave_total = get_moving_average(expected_cases)
    deaths_moving_ave_total = get_moving_average(expected_deaths)
    cfr_total = cfr
    
    # get case scaler mean and std
    mean_cases = jnp.expand_dims(scaler_mean_cases, 0)
    std_cases = jnp.expand_dims(scaler_std_cases, 0)
    mean_deaths = jnp.expand_dims(scaler_mean_deaths, 0)
    std_deaths = jnp.expand_dims(scaler_std_deaths, 0)
    
    
    for day in range(look_ahead):
        
        if preprocessed_data == 'None':
            case_input = (jnp.log(1+cases_total[:,:,-n_days_nn_input:])  - mean_cases)/std_cases
            death_input = (jnp.log(1+deaths_total[:,:,-n_days_nn_input:])  - mean_deaths)/std_deaths
            
        elif preprocessed_data == 'moving_average':
            case_input = (jnp.log(1+cases_moving_ave_total[:,:,-n_days_nn_input:])  - mean_cases)/std_cases
            death_input = (jnp.log(1+cases_moving_ave_total[:,:,-n_days_nn_input:])  - mean_deaths)/std_deaths
            
        elif preprocessed_data == 'summary':
            a = (jnp.log(1+cases_total[:,:,-n_days_nn_input:])  - mean_cases)/std_cases
            b = (jnp.log(1+deaths_total[:,:,-n_days_nn_input:])  - mean_deaths)/std_deaths
            case_input = summarise_some_data(a, num_percentiles)
            death_input = summarise_some_data(b, num_percentiles)
            
        elif preprocessed_data == 'moving_average_summary':
            a = (jnp.log(1+cases_moving_ave_total[:,:,-n_days_nn_input:])  - mean_cases)/std_cases
            b = (jnp.log(1+cases_moving_ave_total[:,:,-n_days_nn_input:])  - mean_deaths)/std_deaths
            case_input = summarise_some_data(a, num_percentiles)
            death_input = summarise_some_data(b, num_percentiles)
        
        case_input = jnp.expand_dims(case_input, (2,3))
        death_input = jnp.expand_dims(death_input, (2,3))

        if input_death:
            nn_input = jnp.concatenate((case_input, death_input), axis=4)
        else:
            nn_input = case_input
                
        nn_output = bnn_feedforward_nn_method2(parameter_list_R, nn_input, max_change_R_nn)
        nn_output = jnp.squeeze(nn_output, axis=2)  
        
        R_new = nn_output[:,:,[0]] + Rt_total[:,:,[-1]]
        R_new = jax.nn.relu(R_new)
        Rt_total = jnp.concatenate((Rt_total, R_new), 2)
        
        if infer_cfr:
            nn_output = bnn_feedforward_nn_method2(parameter_list_cfr, nn_input, max_change_cfr_nn)
            nn_output = jnp.squeeze(nn_output, axis=2)  

            cfr_new = nn_output[:,:,[0]] + cfr_total[:,:,[-1]]
            cfr_new = jax.nn.relu(cfr_new)
            cfr_total = jnp.concatenate((cfr_total, cfr_new), 2)
            
        else:
            cfr_total = cfr
            
        dist1 = jnp.expand_dims(ep.GI_flat_rev, (0,2))
        dist2 = jnp.transpose(jnp.expand_dims(ep.DPC, 0), (0,2,1))
        dist3 = jnp.transpose(jnp.expand_dims(ep.DPD, 0), (0,2,1))
        
        new_infections = jnp.multiply(R_new, infections_total[:,:,-(ep.GIv.size - 1):] @ jnp.flip(dist1, 1))

        # append to infections_total
        infections_total = jnp.concatenate((infections_total, new_infections), 2)
        
        # add in reporting delay
        day_in_weekly_cycle = (day+data.nDs)%7
        reporting_case_multiplier = case_weekly_multiplier[:,:,day_in_weekly_cycle]
        reporting_death_multiplier = death_weekly_multiplier[:,:,day_in_weekly_cycle]
        
            
        new_expected_case = (infections_total[:,:,-(ep.DPC.size):] @ jnp.flip(dist2, 1)) 
        new_expected_case = new_expected_case * jnp.expand_dims(reporting_case_multiplier, 2)
        
        deaths = jnp.multiply(infections_total, cfr_total)

        new_expected_death = deaths[:,:,-(ep.DPD.size):] @ jnp.flip(dist3, 1)
        new_expected_death = new_expected_death * jnp.expand_dims(reporting_death_multiplier, 2)
        
        # append deaths and cases to deaths_total and cases_total
        cases_total = jnp.concatenate((cases_total, new_expected_case), 2)
        deaths_total = jnp.concatenate((deaths_total, new_expected_death), 2)
        
        new_expected_case_ave = jnp.expand_dims(jnp.sum(cases_total[:,:,-7:], axis=2)/7, axis=2)
        new_expected_death_ave = jnp.expand_dims(jnp.sum(deaths_total[:,:,-7:], axis=2)/7, axis=2)
         
        cases_moving_ave_total = jnp.concatenate((cases_moving_ave_total, new_expected_case_ave), 2)
        deaths_moving_ave_total = jnp.concatenate((deaths_moving_ave_total, new_expected_death_ave), 2)

    return cases_total, deaths_total, Rt_total, cfr_total


def prophet_predictor(data, look_ahead, end_date, start_date="2020-08-01", floor_cap=[0,36000,0,4800], nS=40):

    case_data = data.new_cases.data
    death_data = data.new_deaths.data

    case_samples = jnp.zeros((nS, data.nRs, data.nDs+look_ahead))
    death_samples = jnp.zeros((nS, data.nRs, data.nDs+look_ahead))
    
    # initialise predictor

    for region in range(data.nRs):
            
        predictor_case = Prophet(growth='logistic', interval_width=0.9, changepoint_prior_scale=1,  uncertainty_samples=nS)
        predictor_death = Prophet(growth='logistic', interval_width=0.9, changepoint_prior_scale=1,  uncertainty_samples=nS)

        # make dataframe
        case_data_region = case_data[region,:]
        df_case = pd.DataFrame()
        df_case.insert(0,'ds',pd.date_range(start_date, end_date))
        df_case.insert(1,'y', case_data_region)
        df_case['floor'] = floor_cap[0]
        df_case['cap'] = floor_cap[1]

        
        death_data_region = death_data[region,:]
        df_death = pd.DataFrame()
        df_death.insert(0,'ds',pd.date_range(start_date, end_date))
        df_death.insert(1,'y', death_data_region)
        df_death['floor'] = floor_cap[2]
        df_death['cap'] = floor_cap[3]

        # fit to data
        predictor_case.fit(df_case)
        predictor_death.fit(df_death)

        future_case = predictor_case.make_future_dataframe(periods=look_ahead)
        future_case['floor'] = floor_cap[0]
        future_case['cap'] = floor_cap[1]
        samples_case = predictor_case.predictive_samples(future_case)['yhat'].transpose()
        
        future_death = predictor_death.make_future_dataframe(periods=look_ahead)
        future_death['floor'] = floor_cap[2]
        future_death['cap'] = floor_cap[3]
        samples_death = predictor_death.predictive_samples(future_death)['yhat'].transpose()
        
        case_samples = jax.ops.index_update(
          case_samples,
          jax.ops.index[:, region, :],
          samples_case)
        
        death_samples = jax.ops.index_update(
          death_samples,
          jax.ops.index[:, region, :],
          samples_death)
        
    return case_samples, death_samples
        
def check_confidence_interval(distribution, true, lower_percentile, upper_percentile, verbose):

    # get number of samples
    nS = len(distribution)
    
    # lower percentiles (of predicted data in future)
    pred_low = np.percentile(distribution, lower_percentile, axis=0) 

    # upper percentiles (of predicted data in future)
    pred_high = np.percentile(distribution, upper_percentile, axis=0) 

    # find if each case/death is in the confidence interval
    in_interval = np.bitwise_and(true<pred_high, true>pred_low)

    # get proportion of cases/deaths that lie in interval
    nD_predict = len(in_interval[0])
    
    prop_in_interval = jnp.mean(jnp.sum(in_interval, axis=-1)==nD_predict)

    if verbose:
        print('We expect to get ', (upper_percentile-lower_percentile)/100, 'in range')
        print('We got: ', prop_in_interval)

def get_prediction_metrics(distribution, true, end_of_data, verbose=True, skip_crps=False):
    
    nDs = len(distribution[0][0])
    true = true[:,:nDs]
    
    mse = get_mse(distribution, true)
    
    if not skip_crps:
        crps = get_crps(distribution, true)
    else:
        crps = [0]
        
    true_future = true[:,end_of_data:]

    expected_future = distribution[:,:,end_of_data:]
    
    check_confidence_interval(expected_future, true_future, 5, 95, verbose)
    check_confidence_interval(expected_future, true_future, 10, 90, verbose)
    check_confidence_interval(expected_future, true_future, 25, 75, verbose)

    return mse, crps


def get_prediction_metrics2(distribution, true, end_of_data, verbose=True, skip_crps=False):
    
    nDs = len(distribution[0][0])
    true = true[:,:nDs]
    
    working = jnp.nanmean(np.nanpercentile(distribution, 50, axis=0)[:end_of_data], axis=1) > 2

    distribution = distribution[:,working,:]
    true = true[working,:]
    
    nmse = get_nmse(distribution, true)
    
    if not skip_crps:
        ncrps = get_ncrps(distribution, true)
        ncrps = jnp.squeeze(ncrps)
    else:
        ncrps = jnp.array([0]*nDs)
        
    true_future = true[:,end_of_data:]

    expected_future = distribution[:,:,end_of_data:]
    
    check_confidence_interval(expected_future, true_future, 5, 95, verbose)
    check_confidence_interval(expected_future, true_future, 10, 90, verbose)
    check_confidence_interval(expected_future, true_future, 25, 75, verbose)

    return nmse, ncrps
