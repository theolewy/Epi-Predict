"""
Contains a bunch of model utility functions, used to construct models while trying to minimise copy and pasteing code.
"""

from itertools import combinations
import copy
import os 

import matplotlib.pyplot as plt

import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import arviz as az
import csv
import pyreadr


from epimodel.distributions import AsymmetricLaplace


def sample_intervention_effects(nCMs, intervention_prior=None):
    """
    Sample interventions from some options

    :param nCMs: number of interventions
    :param intervention_prior: dictionary with relevant keys. usually type and scale
    :return: sample parameters
    """
    if intervention_prior is None:
        intervention_prior = {
            "type": "asymmetric_laplace",
            "scale": 30,
            "asymmetry": 0.5,
        }

    if intervention_prior["type"] == "trunc_normal":
        alpha_i = numpyro.sample(
            "alpha_i",
            dist.TruncatedNormal(
                low=-0.1, loc=jnp.zeros(nCMs), scale=intervention_prior["scale"]
            ),
        )
    elif intervention_prior["type"] == "half_normal":
        alpha_i = numpyro.sample(
            "alpha_i",
            dist.HalfNormal(scale=jnp.ones(nCMs) * intervention_prior["scale"]),
        )
    elif intervention_prior["type"] == "normal":
        alpha_i = numpyro.sample(
            "alpha_i",
            dist.Normal(loc=jnp.zeros(nCMs), scale=intervention_prior["scale"]),
        )
    elif intervention_prior["type"] == "asymmetric_laplace":
        alpha_i = numpyro.sample(
            "alpha_i",
            AsymmetricLaplace(
                asymmetry=intervention_prior["asymmetry"],
                scale=jnp.ones(nCMs) * intervention_prior["scale"],
            ),
        )
    else:
        raise ValueError(
            "Intervention effect prior must take a value in [trunc_normal, normal, asymmetric_laplace, half_normal]"
        )

    return alpha_i


def sample_basic_R(nRs, basic_r_prior=None):
    """
    Sample basic r

    :param nRs: number of regions
    :param basic_r_prior: dict contains basic r prior info.
    :return: basic R
    """
    if basic_r_prior is None:
        basic_r_prior = {"mean": 1.35, "type": "trunc_normal", "variability": 0.3}

    if basic_r_prior["type"] == "trunc_normal":
        basic_R = numpyro.sample(
            "basic_R",
            dist.TruncatedNormal(
                low=0.1,
                loc=basic_r_prior["mean"],
                scale=basic_r_prior["variability"] * jnp.ones(nRs),
            ),
        )
    else:
        raise ValueError("Basic R prior type must be in [trunc_normal]")

    return basic_R

def sample_bnn_weights(D_h1, D_h2, D_in=2, D_out=2, num_pairs_interact=0):

    # samples each weight/bias variable,
    # dimensions are nn_pair x region x time x input x output
    
    # if no interacting pairs then return None
    if num_pairs_interact == 0:
        return 0

    w1 = numpyro.sample("w1_base", dist.Normal(
                                        jnp.zeros((num_pairs_interact, 1, 1, D_in, D_h1)), 
                                        jnp.ones((num_pairs_interact, 1, 1, D_in, D_h1))))

    w2 = numpyro.sample("w2_base", dist.Normal(
                                        jnp.zeros((num_pairs_interact, 1, 1, D_h1, D_h2)), 
                                        jnp.ones((num_pairs_interact, 1, 1, D_h1, D_h2))))

    w3 = numpyro.sample("w3_base", dist.Normal(
                                        jnp.zeros((num_pairs_interact, 1, 1, D_h2, D_out)), 
                                        jnp.ones((num_pairs_interact, 1, 1, D_h2, D_out))))

    # repeating to ensure the params are const over region and time
            
    # save to trace
    numpyro.deterministic("w1", w1)
    numpyro.deterministic("w2", w2)
    numpyro.deterministic("w3", w3)
    
    return [w1, w2, w3]

def bnn_feedforward(parameter_list, X):

    [w1, w2, w3] = parameter_list

    z1 = jnp.tanh(jnp.matmul(X, w1))
    z2 = jnp.tanh(jnp.matmul(z1, w2))
    z3 = jnp.matmul(z2, w3)
    
    output = jnp.squeeze(z3, axis=-2) 

    return output

def sample_bnn_weights_nn_method(D_layers, reg=1, site=''):

    # samples each weight/bias variable,
    # dimensions of each weigh/bias are  region x time x input x output
    # return param_list of form [w1, b1, w2, b2, ...]
    
    if reg is None:
        reg = -1
    
    numpyro.deterministic("bnn_regulariser"+site, reg)
    
    D_in = D_layers[0]
    D_out = D_layers[-1]
    
    param_list=[]
    
    # if we inputted reg=None originally then:
    if isinstance(reg, int) and reg < 0:
        reg = numpyro.sample('reg'+site, dist.HalfNormal(scale=jnp.ones(len(D_layers)-1)))
        
    # else if we inputted a single regulariser then we make it the regulariser for all layers
    elif not isinstance(reg, list):
        reg = [reg]*(len(D_layers)-1)

    
    for i in range(len(D_layers)-1):
        
        w_base_site_name = 'w'+str(i+1)+'_base'+site
        b_base_site_name = 'b'+str(i+1)+'_base'+site
        
        w_site_name = 'w'+str(i+1)+site
        b_site_name = 'b'+str(i+1)+site

        next_w =   numpyro.sample(w_base_site_name, dist.Normal(
                                        jnp.zeros((1, 1, D_layers[i], D_layers[i+1])), 
                                        jnp.ones((1, 1, D_layers[i], D_layers[i+1]))))
        next_w = next_w  * reg[i]
        
        param_list.append(next_w)

        numpyro.deterministic(w_site_name, next_w)

        next_b =  numpyro.sample(b_base_site_name, dist.Normal(
                                        jnp.zeros((1, 1, 1, D_layers[i+1])), 
                                        jnp.ones((1, 1, 1, D_layers[i+1]))))
        
        next_b = next_b * reg[i]
        
        param_list.append(next_b)
        
        numpyro.deterministic(b_site_name, next_b)
        

    return param_list


def sample_bnn_weights_nn_method2(D_layers, reg=1, site=''):

    # samples each weight/bias variable,
    # dimensions of each weigh/bias are  region x time x input x output
    # return param_list of form [w1, b1, w2, b2, ...]
    
    if reg is None:
        reg = -1
    
    numpyro.deterministic("bnn_regulariser"+site, reg)
    
    D_in = D_layers[0]
    D_out = D_layers[-1]
    
    param_list=[]
    
    # if we inputted reg=None originally then:
    if isinstance(reg, int) and reg < 0:
        reg = numpyro.sample('reg'+site, dist.HalfNormal(scale=jnp.ones(len(D_layers)-1)))
        reg_gap = 0.5 * numpyro.sample('reg_gap'+site, dist.HalfNormal(scale=jnp.ones(len(D_layers)-1)))

    # else if we inputted a single regulariser then we make it the regulariser for all layers
    elif not isinstance(reg, list):
        reg = [reg]*(len(D_layers)-1)
        
    offset_b_prior = 0.3 * numpyro.sample('offset_b_prior'+site, dist.HalfNormal(scale=1))

    
    for i in range(len(D_layers)-1):
        
        w_base_site_name = 'w'+str(i+1)+'_base'+site
        b_base_site_name = 'b'+str(i+1)+'_base'+site
        
        w_site_name = 'w'+str(i+1)+site
        b_site_name = 'b'+str(i+1)+site

        next_w = numpyro.sample(w_base_site_name, dist.Normal(
                                        jnp.zeros((1, 1, D_layers[i], D_layers[i+1])), 
                                        jnp.ones((1, 1, D_layers[i], D_layers[i+1]))))
        next_w = next_w  * reg[i]
        
        param_list.append(next_w)

        numpyro.deterministic(w_site_name, next_w)
        
        if D_layers[i+1] == 1:
            
            next_b = reg_gap[i] * numpyro.sample(b_base_site_name, dist.Normal(
                                        jnp.zeros((1, 1, 1, D_layers[i+1])), 
                                        jnp.ones((1, 1, 1, D_layers[i+1]))))
            
        else:
            
            offset_b = offset_b_prior * numpyro.sample(b_base_site_name+'offset', dist.Normal(
                                        jnp.zeros((1, 1, 1, 1)), 
                                         jnp.ones((1, 1, 1, 1))))
                
            next_b_gap = reg_gap[i] * numpyro.sample(b_base_site_name+'gap', dist.HalfNormal(
                                        scale=jnp.ones((1, 1, 1, D_layers[i+1]-1))))
            
            next_b_gap = jnp.concatenate((jnp.zeros((1,1,1,1)), next_b_gap), axis=3)
        
            next_b = jnp.cumsum(next_b_gap, axis=3)
            next_b = next_b - jnp.mean(next_b, axis=3) + offset_b
        
        param_list.append(next_b)
        
        numpyro.deterministic(b_site_name, next_b)
        

    return param_list


    
def bnn_feedforward_nn_method(parameter_list, X):

    z = X
    
    for i in range(0,len(parameter_list)-2, 2):
    
        z = jnp.tanh(jnp.matmul(z, parameter_list[i]) + parameter_list[i+1])
    
    z = jnp.matmul(z, parameter_list[-2]) + parameter_list[-1]
        
    output = jnp.log(1+ jnp.exp(jnp.squeeze(z, axis=-2)))

    return output

def bnn_feedforward_nn_method2(parameter_list, X, max_change_nn):
    
    z = X
          
    for i in range(0,len(parameter_list)-2, 2):
    
        z = jnp.tanh(jnp.matmul(z, parameter_list[i]) + parameter_list[i+1])
    
    z = jnp.matmul(z, parameter_list[-2]) + parameter_list[-1]
    
    z = jnp.squeeze(jnp.tanh(z), axis=-2)
    
    output = max_change_nn * z
    
    return output

def get_lin_layer(parameter_list, active_cms, interact):
    '''
    We require X to be either 1) nRs x nDs x 1 x nCMs
                        or    2) num_samples x nRs x nDs x 1 x nCMs
    
    '''
    # if no interacting pairs then...
    if parameter_list == 0:
        return active_cms
    elif isinstance(parameter_list, int) :
        active_cms = jnp.repeat(jnp.expand_dims(active_cms, 0),parameter_list, 0)
        return active_cms
    
    # get all bnn parameters
    w1, w2, w3 = parameter_list
    
    # Now ensure input is correct shape
    
    # if we have not generated samples yet then
    if w1.ndim == 5:
        
        X = jnp.transpose(jnp.expand_dims(active_cms, 2), (0, 3, 2, 1))
        
    # if we have generated samples then
    elif w1.ndim == 6:
        
        num_samples = len(w1)

        active_cms = jnp.repeat(jnp.expand_dims(active_cms, 0), num_samples, 0)
        
        X = jnp.expand_dims(active_cms, 3)
        X = jnp.transpose(X, (0, 1, 4, 3, 2))

        
    # initialise the input to the linear regression layer - we will fill in
    # entries with the neural net outputs as we get them shortly...
    lin_input = active_cms
    
    for ind, pair in enumerate(interact):
        
        # if we have not generated samples yet then
        if w1.ndim == 5:
            nn_input = X[:,:,:,pair]
            parameter_list_pair = [w1[ind,:,:,:,:], w2[ind,:,:,:,:], w3[ind,:,:,:,:]]
            nn_output = bnn_feedforward(parameter_list_pair, nn_input)
            nn_output = jnp.transpose(nn_output, (0,2,1))
            lin_input = jax.ops.index_update(
                        lin_input, jax.ops.index[:, pair, :], nn_output
                            )
        # if we have generated samples then
        elif w1.ndim == 6:
            nn_input = X[:,:,:,:,pair]
            parameter_list_pair = [w1[:,ind,:,:,:,:], w2[:,ind,:,:,:,:], w3[:,ind,:,:,:,:]]
            nn_output = bnn_feedforward(parameter_list_pair, nn_input)
            nn_output = jnp.transpose(nn_output, (0,1,3,2))
            lin_input = jax.ops.index_update(
                        lin_input, jax.ops.index[:,:, pair, :], nn_output
                            )   
    return lin_input

def get_cm_reductions(samples, model_settings, cms_row):

    interact = model_settings['interact']
    cm_method = model_settings['cm_method']
    
    if cm_method == 'nn':

        cms = jnp.expand_dims(cms_row, (0,2))

        w1 = samples['w1'][:,:,0:1,0:1,:,:]
        w2 = samples['w2'][:,:,0:1,0:1,:,:]
        w3 = samples['w3'][:,:,0:1,0:1,:,:]

        param_list = [w1, w2, w3]

        lin_input = get_lin_layer(param_list, cms, interact)

        alpha_i = jnp.expand_dims(samples['alpha_i'], (1,3))

        cm_reduction = np.sum(alpha_i * lin_input, axis=2)

        return cm_reduction
    
    else: 
        pass
                          
    
def R_reduction_interaction(samples, model_settings, save_file):
    
    cm_method = model_settings['cm_method']
    interact = model_settings['interact']

    cms_row = jnp.zeros(19)
    
    if cm_method == 'nn':

        # function to convert cm_reduction into reduction in R
        f = lambda x: 1 - jnp.exp(-x)
        
        all_info = []

        for ind, pair in enumerate(interact):
            
            cms_pair = jnp.array(pair)

            # get the cms array with 00, 01, 10 and 11 instead of cms_pair
            cms01 = jax.ops.index_update(
                                cms_row, jax.ops.index[cms_pair], jnp.array([0,1])
                                    )   
            cms10 = jax.ops.index_update(
                                cms_row, jax.ops.index[cms_pair], jnp.array([1,0])
                                    )   
            cms11 = jax.ops.index_update(
                                cms_row, jax.ops.index[cms_pair], jnp.array([1,1])
                                    ) 

            # gets the cm_reduction for each of the above cms arrays
            cm_reduction01 = get_cm_reductions(samples, model_settings, cms01)
            cm_reduction10 = get_cm_reductions(samples, model_settings, cms10)
            cm_reduction11 = get_cm_reductions(samples, model_settings, cms11)

            # computes how much extra reduction was due to the interaction
            bonus_reduction = cm_reduction11 - cm_reduction10 - cm_reduction01
            
            if not save_file:

                print('Reduction due to just ', cms_pair[1],' is ', jnp.mean(f(cm_reduction01)), jnp.std(f(cm_reduction01)))
                print('Reduction due to just ', cms_pair[0],' is ', jnp.mean(f(cm_reduction10)), jnp.std(f(cm_reduction10)))
                print('Reduction due to both of these is ', jnp.mean(f(cm_reduction11)), jnp.std(f(cm_reduction11)))
                print('Interaction due to ', pair ,'is ', jnp.mean(f(bonus_reduction)), jnp.std(f(bonus_reduction)))
    
            mean = round(jnp.mean(f(bonus_reduction)).item(),5)
            std = round(jnp.std(f(bonus_reduction)).item(),5)
            info = (pair, mean, std, jnp.abs(mean)>std)
            all_info.append(info)
        
        return all_info
        
    elif cm_method == 'linear_interact':
        
        alpha_i = samples['alpha_i']
        
        f = lambda x:  1-jnp.exp(-x)
        
        all_info = []

        for ind, pair in enumerate(interact):
            first = f(alpha_i[:,pair[0]]) 
            second = f(alpha_i[:,pair[1]])
            both = f(alpha_i[:,19+ind])
    
            mean = round(jnp.mean(both).item(),5)
            std = round(jnp.std(both).item(),5)
            info = (pair, mean, std, jnp.abs(mean)>std)
            all_info.append(info)
        
        return all_info
    
def R_reduction_interaction_all_samples(all_samples, all_model_settings, save_file=False):
    
    if save_file:
        f = open(save_file, 'w')
        
        writer = csv.writer(f,  lineterminator='\n')
        
        writer.writerow(('Interacting Pair', 'Mean Reduction', 'Standard Deviation', 'Interaction Seen' ))
    
        for i in range(len(all_samples)):
            
            all_info = R_reduction_interaction(all_samples[i], all_model_settings[i], save_file=save_file)
            
            for info in all_info:
                writer.writerow(info)
            
        f.close()
                
                
    else:
        
        for i in range(len(all_samples)):
            print('Interactions between: ', all_model_settings[i]['interact'])
            R_reduction_interaction(all_samples[i], all_model_settings[i], save_file=save_file) 
            
            
def get_interact_lists(nCMs, num_pairs_interact):
    
    interact_lists = []
    all_pairs = list(combinations(np.arange(nCMs), 2))

    while len(all_pairs) != 0:
        
        interact = []
        interact_flat = []
        
        while len(interact) != num_pairs_interact: 
            
            all_pairs_copy = copy.deepcopy(all_pairs)
            
            for pair in all_pairs_copy:
                
                pair = list(pair)
                
                # if no elements shared between pair and interact_flat
                if []==[i for i in pair if i in interact_flat]:
                    
                    # add pair to interact, and also to interact_flat
                    interact.append(pair)
                    interact_flat.append(pair[0])
                    interact_flat.append(pair[1])
                    
                    # and remove pair from all_pairs 
                    all_pairs.remove(tuple(pair))
                                    
                # if interact is desired length then...
                if len(interact) == num_pairs_interact:
                    interact_lists.append(interact)
                    break
                    
            # if interact is desired length then...
            if len(interact) == num_pairs_interact:
                break
            
            # get remaining numbers that aren't used
            remaining_numbers = list(np.arange(nCMs))
            
            for i in interact_flat:
                remaining_numbers.remove(i)
                
            # get number of extra pairs we need
            remaining_pairs_to_form = num_pairs_interact - len(interact)
            
            # add random remaining pairs on
            for _ in range(remaining_pairs_to_form):
                pair = list(np.random.choice(remaining_numbers, size=2, replace=False))
                interact.append(pair)
                interact_flat.append(pair[0])
                interact_flat.append(pair[1])
                remaining_numbers.remove(pair[0])
                remaining_numbers.remove(pair[1])
                
            # if interact is desired length then...
            if len(interact) == num_pairs_interact:
                interact_lists.append(interact)
                break
            
    return interact_lists
                    
            
    
    
def get_arima_transition_function(arma_p_coeff, arma_q_coeff, arma_const_coeff):
    
    def arima_transitions(carry, noise):
        last_q_err, last_p_logR = carry
        
        if arma_q_coeff is not None:
            q_term = jnp.sum(arma_q_coeff * last_q_err, -1)
        else:
            q_term = 0
            
        if arma_p_coeff is not None:
            p_term = jnp.sum(arma_p_coeff*last_p_logR, -1)
        else:
            p_term = 0
        
        logR = noise + jnp.squeeze(arma_const_coeff, axis=-1) + p_term + q_term 
    
        # update last_q_err and last_p_logR, unless q or p = 0 then we don't need to update recent err or logR
        if arma_q_coeff is not None:
            last_q_err = jnp.concatenate((last_q_err[:,:,1:], jnp.expand_dims(noise, -1)), axis=-1) 
        else:
            pass
        
        
        if arma_p_coeff is not None:
            last_p_logR = jnp.concatenate((last_p_logR[:,:,1:], jnp.expand_dims(logR, -1)), axis=-1) 
        else:
            pass
        
        return (last_q_err, last_p_logR), logR
    
    return arima_transitions






def seed_infections(seeding_scale, nRs, nDs, seeding_padding, total_padding):
    """
    Seed infections

    :param seeding_scale: seeding scale prior (used for lognormal prior)
    :param nRs: number of regions
    :param nDs: number of days
    :param seeding_padding: number of days of seeding
    :param total_padding: number of days of padding (same as gi trunctation)
    :return: (initial infections (nRs x total_padding), total_infections placeholder (nRs x (seeding_padding + nDs))) tuple
    """
    total_infections_placeholder = jnp.zeros((nRs, seeding_padding + nDs))
    seeding = numpyro.sample("seeding", dist.LogNormal(jnp.zeros((nRs, 1)), 1.0))
    init_infections = jnp.zeros((nRs, total_padding))
    init_infections = jax.ops.index_add(
        init_infections,
        jax.ops.index[:, -seeding_padding:],
        jnp.repeat(seeding ** seeding_scale, seeding_padding, axis=-1),
    )
    return init_infections, total_infections_placeholder

def stagger_matrix_func(new_case_block, next_case_data):

    new_case_block = jnp.concatenate((new_case_block[:,1:], jnp.expand_dims(next_case_data, 1)), axis=-1)

    return new_case_block, new_case_block


def get_discrete_renewal_transition(ep, type="noiseless"):
    """
    Create discrete renewal transition function, used by `jax.lax.scan`

    :param ep: EpidemiologicalParameters() objective
    :param type: either noiseless, optim, or matmul
    :return: Discrete Renewal Transition function, with relevant GI parameters
    """

    if type == "optim":

        def discrete_renewal_transition(infections, R_with_noise_tuple):
            # infections is an nR x total_padding size array of infections in the previous
            # total_padding days.
            R, inf_noise = R_with_noise_tuple
            new_infections_t = jax.nn.softplus(
                jnp.multiply(R, infections @ ep.GI_flat_rev) + inf_noise
            )
            new_infections = infections
            new_infections = jax.ops.index_update(
                new_infections, jax.ops.index[:, :-1], infections[:, 1:]
            )
            new_infections = jax.ops.index_update(
                new_infections, jax.ops.index[:, -1], new_infections_t
            )
            return new_infections, new_infections_t

    elif type == "matmul":

        def discrete_renewal_transition(infections, R_with_noise_tuple):
            # infections is an nR x total_padding size array of infections in the previous
            # total_padding days.
            R, inf_noise = R_with_noise_tuple
            new_infections = infections @ ep.GI_projmat
            new_infections = jax.ops.index_update(
                new_infections,
                jax.ops.index[:, -1],
                jax.nn.softplus(jnp.multiply(new_infections[:, -1], R) + inf_noise),
            )
            return new_infections, new_infections[:, -1]

    elif type == "noiseless":

        def discrete_renewal_transition(infections, R):
            new_infections_t = jnp.multiply(R, infections @ ep.GI_flat_rev)
            new_infections = infections
            new_infections = jax.ops.index_update(
                new_infections, jax.ops.index[:, :-1], infections[:, 1:]
            )
            new_infections = jax.ops.index_update(
                new_infections, jax.ops.index[:, -1], new_infections_t
            )
            return new_infections, new_infections_t

    else:
        raise ValueError(
            "Discrete renewal transition type must be in [matmul, optim, noiseless]"
        )

    return discrete_renewal_transition


def get_output_delay_transition(seeding_padding, data):
    """
    output delay scan, if using country specific delays

    :param seeding_padding: gi trunction // number of days seeding
    :param data: Preprocessed data
    :return: transition function
    """
    def output_delay_transition(loop_carry, scan_slice):
        # this scan function scans over local areas, using their country specific delay, rather than over days
        # therefore the input functions are ** not ** transposed.
        # Also, we don't need a loop carry, so we just return 0 and ignore the loop carry!
        (
            future_cases,
            future_deaths,
            country_cases_delay,
            country_deaths_delay,
        ) = scan_slice
        expected_cases = jax.scipy.signal.convolve(
            future_cases, country_cases_delay, mode="full"
        )[seeding_padding : data.nDs + seeding_padding]
        expected_deaths = jax.scipy.signal.convolve(
            future_deaths, country_deaths_delay, mode="full"
        )[seeding_padding : data.nDs + seeding_padding]

        return 0.0, (expected_cases, expected_deaths)

    return output_delay_transition

def get_mse(distribution, true):
    
    # get median of distribution
    median = np.nanpercentile(distribution, 50, axis=0) 
    
    # get mse
    mse_cases = jnp.nanmean(jnp.power(median - true, 2), axis=0)
    
    return mse_cases

def get_crps(distribution, true):
    
    nS = len(distribution)
    nRs = len(distribution[0])
    nDs = len(distribution[0][0])

    cprs_list = []

    for day in range(nDs):

        cprs = 0

        for region in range(nRs):

            true_value = true[region,day]
            samples_value = distribution[:,region,day]
            samples_low = sorted([i for i in samples_value if i<true_value])
            samples_low.append(true_value)
            samples_high = sorted([i for i in samples_value if i>=true_value], reverse=True)
            samples_high.append(true_value)
            for ind, samp in enumerate(samples_low[:-1]):
                cprs += (samples_low[ind+1] - samp)*((ind+1)/nS)**2
            for ind, samp in enumerate(samples_high[:-1]):
                cprs += (samp - samples_high[ind+1])*((ind+1)/nS)**2 

        cprs_list.append(cprs / nRs)
        
    cprs_list = jnp.array(cprs_list)

    return cprs_list

def get_nmse(distribution, true):
    
    # get median of distribution
    median = np.nanpercentile(distribution, 50, axis=0) 
    
    # get mse
    pse_percentage_error = jnp.power((median - true),2) / jnp.expand_dims(jnp.power(jnp.nanmean(median, axis=1),2), 1)
    
    nmse = jnp.nanmean(pse_percentage_error, axis=0)
    
    return nmse

def get_ncrps(distribution, true):
    
    nS = len(distribution)
    nRs = len(distribution[0])
    nDs = len(distribution[0][0])

    median = np.percentile(distribution, 50, axis=0) 
    normaliser = jnp.expand_dims(jnp.nanmean(median, axis=1), 1)
    
    ncprs_list = []

    for day in range(nDs):

        ncprs = 0

        for region in range(nRs):
            
            true_value = true[region,day]
            samples_value = distribution[:,region,day]
            samples_low = sorted([i for i in samples_value if i<true_value])
            samples_low.append(true_value)
            samples_high = sorted([i for i in samples_value if i>=true_value], reverse=True)
            samples_high.append(true_value)
            for ind, samp in enumerate(samples_low[:-1]):
                ncprs += ((samples_low[ind+1] - samp)*((ind+1)/nS)**2)/normaliser[region]
            for ind, samp in enumerate(samples_high[:-1]):
                ncprs += ((samp - samples_high[ind+1])*((ind+1)/nS)**2)/normaliser[region]

        ncprs_list.append(ncprs / nRs)
        
    ncprs_list = jnp.array(ncprs_list)

    return ncprs_list

def plot_graph(datasets, output_fname, prediction_date=None, start_date="2020-08-01", percentiles=(5,95), region=0, label=None, title=False):
    
    start_date = pd.to_datetime(start_date)
    prediction_date = pd.to_datetime(prediction_date)
    
    if label is True:
        label = list(range(len(datasets)))
    
    if not isinstance(datasets, tuple):
        datasets = (datasets,)
        
    for n, data in enumerate(datasets):
        
        if data is None:
            pass
        
        elif isinstance(data, list) or data.ndim == 1:
            
            data_end = start_date + pd.Timedelta(days=len(data)-1)
            data_range = pd.date_range(start_date, data_end)

            if label:
                plt.plot(data_range, data, label=label[n])
            else:
                plt.plot(data_range, data)

        
        elif data.ndim == 2:
            
            data_end = start_date + pd.Timedelta(days=len(data[0])-1)
            data_range = pd.date_range(start_date, data_end)

            if label:
                plt.plot(data_range, data[region,:], label=label[n])
            else:
                plt.plot(data_range, data[region,:])
            
        
        elif data.ndim == 3:
            
            data_end = start_date + pd.Timedelta(days=len(data[0][0])-1)
            data_range = pd.date_range(start_date, data_end)   
            
            if percentiles is None:
                
                nS = len(data)

                for i in range(nS):
                    if label:
                        plt.plot(data_range, data[i,region,:], label=label[n])
                    else:
                        plt.plot(data_range, data[i,region,:])
                    
            
            else:
     
                plt.fill_between(data_range, 
                                 np.percentile(data[:,region,:], percentiles[0], axis=0), 
                                 np.percentile(data[:,region,:], percentiles[1], axis=0), 
                                 alpha=0.2, 
                                 color="orange", 
                                 linewidth=0)
        
                if label:
                    plt.plot(data_range, np.percentile(data[:,region,:], 50, axis=0), color="orange", label=label[n])
                else:
                    plt.plot(data_range, np.percentile(data[:,region,:], 50, axis=0), color="orange")
        
    if prediction_date is not None:
        plt.axvline(x=prediction_date)
    
    plt.legend(loc = 'upper left')
    plt.xticks(data_range[::30])
    
    if title:
        plt.title(title)
    
    if region is not None:
        # if no Region folder exists, build it
        if not os.path.isdir('../Graphs/Region' + str(region)):
            os.mkdir('../Graphs/Region' + str(region))
        
        plt.savefig('../Graphs/Region' + str(region) + '/' + output_fname)
    else:
        plt.savefig('../Graphs/' + output_fname)
    plt.clf()

def get_data_for_R(path, end_date="2021-01-02" , f_name='data_for_r.csv'):
    
    df = pd.read_csv(path)
    df = df[['Date', 'New Cases', 'Area']]
    df = df.rename(columns={'Date':'date', 'New Cases':'confirm', 'Area':'region'})

    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")
    
    rows_to_keep = df['date']<=pd.to_datetime(end_date, format="%Y/%m/%d")
    
    df = df.loc[rows_to_keep]

    df.to_csv(f_name, index=False)
    
def get_data_from_epinow2():
    
    regions = sorted(os.listdir('../EpiNow2/results'))
    regions.remove('.ipynb_checkpoints')
    regions.remove('runtimes.csv')
    
    # get useful values by opening one region's case data up
    result = pyreadr.read_r('../EpiNow2/results/' + regions[1] + '/latest/estimate_samples.rds')
    df = result[None] # extract the pandas data frame 
    df = df.loc[df['variable'] == 'reported_cases']
    
    nS = int(max(df['sample']))
    nRs = len(regions)
    nDs = max(df['time'])
    
    
    cases_total = jnp.zeros((nS, nRs, nDs))
    
    for ind, region in enumerate(regions):
    

        try:
            result = pyreadr.read_r('../EpiNow2/results/'+region+'/latest/estimate_samples.rds')
            df = result[None] # extract the pandas data frame 
        except:
            continue

        # only look at case data
        df = df.loc[df['variable'] == 'reported_cases']
        
        # initialise regional_samples to NaN
        regional_samples = np.zeros((nS, nDs))
        
        # fill in regional_samples
        for i in range(nS):

            # gets cases of sample i
            cases_sample_i = df.loc[df['sample'] == i+1]['value'].to_numpy()

            num_days_i = len(cases_sample_i)

            # append that case data to regional_samples
            regional_samples = jax.ops.index_update(
                                regional_samples, jax.ops.index[i, -num_days_i:], cases_sample_i
                                    )    
        
        cases_total = jax.ops.index_update(
                                cases_total, jax.ops.index[:,ind, :], regional_samples
                                    )
    return cases_total
    
def netcdf_to_dict(path):
    
    results = az.from_netcdf(path).posterior
    
    site_names = results.data_vars
    
    samples = {}
    
    for site in site_names:
        
        site_array = results[site].to_numpy()
        
        # flatten chain+draws into samples
        shape = list(jnp.shape(site_array))
        shape[1] = -1
        shape = shape[1:]
        site_array = np.reshape(site_array, shape, order='C')
        
        samples.update({str(site):site_array})
        
    return samples

def get_param_list_from_samples(samples, site=''):
    
    param_list = []
    
    for i in range(1,100):

        w_site_name = 'w'+str(i)
        b_site_name = 'b'+str(i)
        
        if w_site_name in samples.keys():
            
            w_new = samples[w_site_name]
            b_new = samples[b_site_name]
            
        elif w_site_name+site in samples.keys():
            
            w_new = samples[w_site_name+site]
            b_new = samples[b_site_name+site]
            
        else:
            break
            
        param_list.append(w_new)
        param_list.append(b_new)
        
    return param_list

def get_bnn_dimensions_from_param_list(param_list):

    bnn_dim = []
    
    for i in range(0, len(param_list), 2):
        
        bnn_dim.append(jnp.shape(param_list[i])[3])
    
    bnn_dim.append(jnp.shape(param_list[-1])[4])
    
    return bnn_dim


def samples_to_model_settings(samples):
    
    model_settings = {}
    
    sites = samples.keys()
    
    if 'arma_const_coeff' in sites:
        
        model_settings.update({'interact':[]})
        
        if jnp.array_equiv(samples['arma_const_coeff'], jnp.zeros_like(samples['arma_const_coeff'])):
            model_settings.update({'noise_method':'arma_fix'})
        else:
            model_settings.update({'noise_method':'arma_learn'})
            
        if 'w1' in sites:
            model_settings.update({'cm_method':'nn'})
        else:
            model_settings.update({'cm_method':'linear'})
            
        if 'arma_p_coeff' in sites:
            arma_p = jnp.shape(samples['arma_p_coeff'])[-1]
            model_settings.update({'arma_p':arma_p})
        else:
            model_settings.update({'arma_p':0})
            
        if 'arma_q_coeff' in sites:
            arma_q = jnp.shape(samples['arma_q_coeff'])[-1]
            model_settings.update({'arma_q':arma_q})
        else:
            model_settings.update({'arma_q':0})
            
        noise_period = (jnp.sum(samples['Rt_noise'][0,0,30] == samples['Rt_noise'][0,0,:])).item()
        model_settings.update({'noise_period':noise_period})
        
    else:
        
        param_list = get_param_list_from_samples(samples, 'R')
        bnn_dim = get_bnn_dimensions_from_param_list(param_list)
        
        n_days_nn_input = (jnp.sum(samples['Rt'][0,0,0] == samples['Rt'][0,0,:])).item()
        model_settings.update({'n_days_nn_input':n_days_nn_input})

        infer_cfr = True if bnn_dim[-1]==2 else False
        model_settings.update({'infer_cfr':infer_cfr})
        
        input_death = True if samples['input_death'][0]==1 else False
        model_settings.update({'input_death':input_death})
        

        D_layers = bnn_dim[1:-1]
        model_settings.update({'D_layers':D_layers})

        R_period = (jnp.sum(samples['Rt'][0,0,n_days_nn_input+4] == samples['Rt'][0,0,:])).item()
        model_settings.update({'R_period':R_period})
        
        try:
            bnn_regulariser = samples['bnn_regulariser'][0]
            if bnn_regulariser < 0:
                bnn_regulariser = None
            model_settings.update({'bnn_regulariser':bnn_regulariser})
        except:
            pass
        
        try:
            if samples['preprocessed_data'][0]==0:
                preprocessed_data = 'moving_average'
            elif samples['preprocessed_data'][0]==1:
                preprocessed_data = 'summary'
            elif samples['preprocessed_data'][0]==2:
                preprocessed_data = 'None'
            elif samples['preprocessed_data'][0]==3:
                preprocessed_data = 'moving_average_summary'
            model_settings.update({'preprocessed_data':preprocessed_data})
            
        except:
            pass
        
        if preprocessed_data == 'summary' or preprocessed_data == 'moving_average_summary':
            num_percentiles = (bnn_dim[0] if input_death is False else bnn_dim[0]//2)-2
            model_settings.update({'num_percentiles':num_percentiles})
        else:
            model_settings.update({'num_percentiles':None})

        
    return model_settings
        