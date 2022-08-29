"""
Models
"""
import jax.scipy.signal
import jax.numpy as jnp
import jax
import numpyro

from epimodel.models.model_build_utils import *

def latent_nn_model(
        data,
        ep,
        basic_R_prior=None,
        input_death=False,
        D_layers = [10,10],
        bnn_regulariser = None,
        preprocessed_data = 'summary',
        infer_cfr=False,
        report_period=True,
        n_days_seeding=7,
        R_period=1,
        n_days_nn_input = 21,
        seeding_scale=3.0,
        infection_noise_scale=5.0,
        output_noise_scale_prior=5.0,
        **kwargs,
):
    """
    Main model.

    :param data: PreprocessedData object
    :param ep: EpidemiologicalParameters object
    :param predict: int representing number of days in the future to predict
    :param basic_R_prior: basic r prior dict
    :param preprocessed_data: in ['summary', 'moving_average', 'moving_average_summary', 'None']
    :param r_walk_period: period of random walk
    :param n_days_seeding: number of days of seeding
    :param seeding_scale: scale of seeded infection prior
    :param infection_noise_scale: scale of infection noise
    :param output_noise_scale_prior: output noise scale prior
    :param kwargs: additional kwargs (not used, but maintain function signature)
    """
    for k in kwargs.keys():
        print(f"{k} is not being used")
    
    
    # get the nn_inputs for cases and deaths based on the method.
    # Each of these regions x time x 1 x nn_input_dim
    # Note the time dimension starts from time n_days_nn_input, 
    if preprocessed_data == 'moving_average':
        cases_input = data.moving_ave_cases_nn_input
        deaths_input = data.moving_ave_deaths_nn_input
        numpyro.deterministic("preprocessed_data", 0)
    elif preprocessed_data == 'summary':
        cases_input = data.new_cases_summary
        deaths_input = data.new_deaths_summary
        numpyro.deterministic("preprocessed_data", 1)
    elif preprocessed_data == 'None':
        cases_input = data.cases_nn_input
        deaths_input = data.deaths_nn_input
        numpyro.deterministic("preprocessed_data", 2)
    elif preprocessed_data == 'moving_average_summary':
        cases_input = data.new_cases_moving_ave_summary
        deaths_input = data.new_deaths_moving_ave_summary
        numpyro.deterministic("preprocessed_data", 3)
        
    # Here we only take the cases_input every R_period time
    cases_input = cases_input[:,:-1:R_period,:,:]
    deaths_input = deaths_input[:,:-1:R_period,:,:]

    if input_death:
        nn_input = jnp.concatenate((cases_input, deaths_input), axis=3)
        numpyro.deterministic("input_death", 1)
    else:
        nn_input = cases_input
        numpyro.deterministic("input_death", 0)
        
    D_in = jnp.shape(nn_input)[-1]
    
    D_out = 1
    
    bnn_dims = [D_in] + D_layers + [D_out]
    
    # Sort out R neural net
    
    parameter_list_R = sample_bnn_weights_nn_method2(D_layers=bnn_dims,
                                                    reg=bnn_regulariser,
                                                    site='R')
    
    # Find the maximum value that we allow R to change by in a day
    max_change_R_nn_prior = jnp.array([0.2])

    max_change_R_nn = max_change_R_nn_prior * numpyro.sample("max_change_R_nn_unexpanded", dist.HalfNormal(1))

    max_change_R_nn = jnp.expand_dims(max_change_R_nn,(0,1))
                                   
    numpyro.deterministic("max_change_R_nn", max_change_R_nn)

    # get how R changes 
    nn_output_R = bnn_feedforward_nn_method2(parameter_list_R, nn_input, max_change_R_nn)

    # get sum of the changes in R (we have not including the initial_R yet)
    nn_output_R = jnp.cumsum(nn_output_R, axis=1)
    
    # repeat the same output R_period times so that we model R as a weekly value
    nn_output_R = jnp.repeat(nn_output_R, R_period, axis=1)[:,:data.nDs-n_days_nn_input,:]
    
    # Compute R for the first set of days (NN cannot work these out as no prior info) 
    initial_R = sample_basic_R(data.nRs, basic_R_prior)
    initial_R = jnp.repeat(jnp.expand_dims(initial_R, 1), n_days_nn_input, axis=1)
    
    # Noting that we will softplus this, we begin by inverse softplusing (so outputted initial R is in the right range)
    initial_R = jnp.log(jnp.exp(initial_R) - 1)
    
    Rt = jnp.concatenate((initial_R, initial_R[:,[0]] + nn_output_R[:,:,0]), axis=1)

    Rt = jax.nn.softplus(Rt)
    
    # Now collect Rt in the trace
    Rt = numpyro.deterministic("Rt", Rt)
    
    
    if infer_cfr:
     
        # Initial case fatality rate for first set of days (ascertainment rate assumed to be 1
        # throughout the whole period).

        parameter_list_cfr = sample_bnn_weights_nn_method2(D_layers=bnn_dims,
                                                    reg=bnn_regulariser,
                                                    site='cfr')
            
        max_change_cfr_nn_prior = jnp.array([0.05])

        max_change_cfr_nn = max_change_cfr_nn_prior * numpyro.sample("max_change_cfr_nn_unexpanded", dist.HalfNormal(1))

        max_change_cfr_nn = jnp.expand_dims(max_change_cfr_nn,(0,1))
        
        numpyro.deterministic("max_change_cfr_nn", max_change_cfr_nn)
        
        # get how cfr changes 
        nn_output_cfr = bnn_feedforward_nn_method2(parameter_list_cfr, nn_input, max_change_cfr_nn)

        # get sum of the changes in cfr (not including the initial_cfr yet)
        nn_output_cfr = jnp.cumsum(nn_output_cfr, axis=1)

        # repeat the same output cfr_period times so that we model R as a weekly value
        nn_output_cfr = jnp.repeat(nn_output_cfr, R_period, axis=1)[:,:data.nDs-n_days_nn_input,:]

        initial_cfr = numpyro.sample("initial_cfr_base", dist.Uniform(low=1e-3, high=jnp.ones((data.nRs, 1))))
        initial_cfr_repeated = jnp.repeat(initial_cfr, n_days_seeding+n_days_nn_input, axis=1)
        
        # Noting that we will softplus this, we begin by inverse softplusing (so outputted initial cfr is in the right range)
        initial_cfr_repeated = jnp.log(jnp.exp(initial_cfr_repeated) - 1)
        
        cfr = jnp.concatenate((initial_cfr_repeated, initial_cfr_repeated[:,[0]] + nn_output_cfr[:,:,0]), axis=1)
        
        cfr = jax.nn.softplus(cfr)
        
    else:
    
        # Time constant case fatality rate (ascertainment rate assumed to be 1
        # throughout the whole period).
        
        cfr = numpyro.sample("cfr_base", dist.Uniform(low=(1e-3), high=jnp.ones((data.nRs, 1))))

    numpyro.deterministic("cfr", cfr)

    # Infection Model
    seeding_padding = n_days_seeding
    total_padding = ep.GIv.size - 1

    # note; seeding is also rescaled
    init_infections, total_infections_placeholder = seed_infections(
        seeding_scale, data.nRs, data.nDs, seeding_padding, total_padding
    )
    discrete_renewal_transition = get_discrete_renewal_transition(ep)

    # we need to transpose R because jax.lax.scan scans over the first dimension.
    # We want to scan over time instead of regions!
    _, infections = jax.lax.scan(
        discrete_renewal_transition,
        init_infections,
        Rt.T,
    )

    # corrupt infections with additive noise, adding robustness at small case and death
    # counts
    infection_noise =   numpyro.sample(
        "infection_noise",
        dist.Normal(loc=0, scale=jnp.ones((data.nRs, data.nDs))),
    )
    # enforce positivity!
    infections = jax.nn.softplus(
        infections + (infection_noise_scale * infection_noise.T)
    )

    total_infections = jax.ops.index_update(
        total_infections_placeholder,
        jax.ops.index[:, :seeding_padding],
        init_infections[:, -seeding_padding:],
    )
    total_infections = numpyro.deterministic(
        "total_infections",
        jax.ops.index_update(
            total_infections, jax.ops.index[:, seeding_padding:], infections.T
        ),
    )    
    
    future_cases_t = numpyro.deterministic("future_cases_t", total_infections)
  
    future_deaths_t = numpyro.deterministic(
        "future_deaths_t", jnp.multiply(future_cases_t, cfr)
    )

    # collect expected cases and deaths
    expected_cases_pre = numpyro.deterministic(
        "expected_cases_pre_report",
        jax.scipy.signal.convolve2d(future_cases_t, ep.DPC, mode="full")[
        :, seeding_padding: seeding_padding + data.nDs
        ],
    )

    expected_deaths_pre = numpyro.deterministic(
        "expected_deaths_pre_report",
        jax.scipy.signal.convolve2d(future_deaths_t, ep.DPD, mode="full")[
        :, seeding_padding: seeding_padding + data.nDs
        ],
    )
    
    if report_period:
        
        case_weekly_multiplier_base = numpyro.sample(
        "case_weekly_multiplier",
        dist.LogNormal(loc=0, scale=1 * jnp.ones((data.nRs, 6))))
        
        death_weekly_multiplier_base = numpyro.sample(
        "death_weekly_multiplier",
        dist.LogNormal(loc=0, scale=1 * jnp.ones((data.nRs, 6))))
        
        final_case_multiplier = 1/jnp.expand_dims(jnp.prod(case_weekly_multiplier_base, axis=1), 1)
        final_death_multiplier = 1/jnp.expand_dims(jnp.prod(death_weekly_multiplier_base, axis=1), 1)
        
        case_weekly_multiplier = jnp.concatenate((case_weekly_multiplier_base, final_case_multiplier), axis=1)
        death_weekly_multiplier = jnp.concatenate((death_weekly_multiplier_base, final_death_multiplier), axis=1)

        case_weekly_multiplier = jnp.tile(case_weekly_multiplier, (1, 7 + data.nDs//7))[:,:data.nDs]
        death_weekly_multiplier = jnp.tile(death_weekly_multiplier, (1, 7 + data.nDs//7))[:,:data.nDs]
        
        expected_cases = numpyro.deterministic("expected_cases", (expected_cases_pre * case_weekly_multiplier))
        expected_deaths = numpyro.deterministic("expected_deaths", (expected_deaths_pre * death_weekly_multiplier))
    
    else:
    
        expected_cases = numpyro.deterministic("expected_cases", expected_cases_pre)
        expected_deaths = numpyro.deterministic("expected_deaths", expected_deaths_pre)

    # country specific psi cases and deaths.
    # We will use the 'RC' matrix to pull the correct local area value.
    psi_cases = numpyro.sample(
        "psi_cases",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )
    psi_deaths = numpyro.sample(
        "psi_deaths",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )

    # use the per country psi_cases and psi_deaths and form a nRs x nDs array
    # to use for the output distribution.
    cases_conc = (
        (data.RC_mat @ psi_cases).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )
    deaths_conc = (
        (data.RC_mat @ psi_deaths).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_cases.mask)):
        numpyro.sample(
            "observed_cases",
            dist.GammaPoisson(
                concentration=cases_conc,
                rate=cases_conc / expected_cases,
            ),
            obs=data.new_cases.data,
        )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_deaths.mask)):
        numpyro.sample(
            "observed_deaths",
            dist.GammaPoisson(
                concentration=deaths_conc,
                rate=deaths_conc / expected_deaths,
            ),
            obs=data.new_deaths.data,
        )
        

def latent_nn_model_legacy(
        data,
        ep,
        basic_R_prior=None,
        input_death=False,
        D_layers=[10,5],
        infer_cfr=True,
        n_days_seeding=7,
        R_period=1,
        bnn_regulariser = 1,
        n_days_nn_input = 21,
        seeding_scale=3.0,
        infection_noise_scale=5.0,
        output_noise_scale_prior=5.0,
        **kwargs,
):
    """
    Main model.

    :param data: PreprocessedData object
    :param ep: EpidemiologicalParameters object
    :param predict: int representing number of days in the future to predict
    :param basic_R_prior: basic r prior dict
    :param cm_method: in ['linear', 'linear_interact', 'nn']
    :param interact: list of cms indices to consider interactions of
    :param noise_method: in ['no_noise', 'random_walk', 'arma_learn', 'arma_fix']
    :param r_walk_noise_scale_prior: scale of random walk noise scale prior
    :param r_walk_period: period of random walk
    :param n_days_seeding: number of days of seeding
    :param seeding_scale: scale of seeded infection prior
    :param infection_noise_scale: scale of infection noise
    :param output_noise_scale_prior: output noise scale prior
    :param kwargs: additional kwargs (not used, but maintain function signature)
    """
    for k in kwargs.keys():
        print(f"{k} is not being used")
        

    # First, compute R for the first set of days (NN cannot work these out as no prior info) 
    initial_R = sample_basic_R(data.nRs, basic_R_prior)
    initial_R = jnp.repeat(jnp.expand_dims(initial_R, 1), n_days_nn_input-1, axis=1)
    
    new_cases = data.new_cases.data 
    new_deaths = data.new_deaths.data
    
    # we use log data so that feature scaling works better
    new_cases_log = jnp.log(1+new_cases) 
    new_deaths_log = jnp.log(1+new_cases)
    
    #feature scale
    mean_cases = jnp.mean(new_cases_log, axis=1)
    std_cases = jnp.std(new_cases_log, axis=1)
    mean_deaths = jnp.mean(new_deaths_log, axis=1)
    std_deaths = jnp.std(new_deaths_log, axis=1)

    mean_cases_expand = jnp.expand_dims(mean_cases,1)
    std_cases_expand = jnp.expand_dims(std_cases,1)
    mean_deaths_expand = jnp.expand_dims(mean_deaths,1)
    std_deaths_expand = jnp.expand_dims(std_deaths,1)

    new_cases_log_scaled =  (new_cases_log - mean_cases_expand)/std_cases_expand
    new_deaths_log_scaled =  (new_deaths_log - mean_deaths_expand)/std_deaths_expand

    # save scaler mean and std to trace
    numpyro.deterministic("scaler_mean_cases", mean_cases)
    numpyro.deterministic("scaler_std_cases", std_cases)
    numpyro.deterministic("scaler_mean_deaths", mean_deaths)
    numpyro.deterministic("scaler_std_deaths", std_deaths)
    
    # get the nn input of the past n_days_nn_input case data
    _, cases_input = jax.lax.scan(
        stagger_matrix_func,
        new_cases_log_scaled[:, :n_days_nn_input],
        new_cases_log_scaled[:, n_days_nn_input:].transpose()
    )

    cases_input = cases_input.transpose((1,2,0))
    cases_input = jnp.concatenate((jnp.expand_dims(new_cases_log_scaled[:, :n_days_nn_input],2), cases_input), axis=-1)
    cases_input = jnp.transpose(jnp.expand_dims(cases_input, 2), (0, 3, 2, 1))
    
    # Here we make cases_input constant in time for R_period time
    cases_input = cases_input[:,::R_period,:,:]

    if input_death:
        # get the nn input of the past n_days_nn_input death data
        _, deaths_input = jax.lax.scan(
            stagger_matrix_func,
            new_deaths_log_scaled[:, :n_days_nn_input],
            new_deaths_log_scaled[:, n_days_nn_input:].transpose()
        )

        deaths_input = deaths_input.transpose((1,2,0))
        deaths_input = jnp.concatenate((jnp.expand_dims(new_deaths_log_scaled[:, :n_days_nn_input],2), deaths_input), axis=-1)
        deaths_input = jnp.transpose(jnp.expand_dims(deaths_input, 2), (0, 3, 2, 1))
        
        # only get every R_period values of the input
        deaths_input = deaths_input[:,::R_period,:,:]
        
        nn_input = jnp.concatenate((cases_input, deaths_input), axis=3)
        
    else:
        
        nn_input = cases_input
        
    D_in = jnp.shape(nn_input)[-1]    
    D_out = 2 if infer_cfr else 1

    bnn_dims = [D_in] + D_layers + [D_out]
    parameter_list = sample_bnn_weights_nn_method(D_layers=bnn_dims, reg=bnn_regulariser)
    
    nn_output = bnn_feedforward_nn_method(parameter_list, nn_input)

    # repeat the same input R_period times
    nn_output = jnp.repeat(nn_output, R_period, axis=1)[:,:data.nDs-n_days_nn_input+1,:]

    # get nn_output[:,:,0] as R and nn_output[:,:,1] as cfr (if infering this)
    if infer_cfr:
     
        # Initial case fatality rate for first set of days (ascertainment rate assumed to be 1
        # throughout the whole period).
        initial_cfr = numpyro.sample("initial_cfr", dist.Uniform(low=1e-3, high=jnp.ones((data.nRs, 1))))
        initial_cfr = jnp.repeat(initial_cfr, n_days_seeding+n_days_nn_input-1, axis=1)
        
        cfr = jnp.concatenate((initial_cfr, nn_output[:,:,1]), axis=1)
        
        Rt = jnp.concatenate((initial_R, nn_output[:,:,0]), axis=1)
        
        numpyro.deterministic("cfr", cfr)
            
    else:
    
        # Time constant case fatality rate (ascertainment rate assumed to be 1
        # throughout the whole period).
        cfr = 10 * numpyro.sample("cfr", dist.Uniform(low=1e-4, high=jnp.ones((data.nRs, 1))/10))
        
        Rt = jnp.concatenate((initial_R, nn_output[:,:,0]), axis=1)
    
    # Now collect Rt in the trace
    Rt = numpyro.deterministic("Rt", Rt)

    # Infection Model
    seeding_padding = n_days_seeding
    total_padding = ep.GIv.size - 1

    # note; seeding is also rescaled
    init_infections, total_infections_placeholder = seed_infections(
        seeding_scale, data.nRs, data.nDs, seeding_padding, total_padding
    )
    discrete_renewal_transition = get_discrete_renewal_transition(ep)

    # we need to transpose R because jax.lax.scan scans over the first dimension.
    # We want to scan over time instead of regions!
    _, infections = jax.lax.scan(
        discrete_renewal_transition,
        init_infections,
        Rt.T,
    )

    # corrupt infections with additive noise, adding robustness at small case and death
    # counts
    infection_noise = numpyro.sample(
        "infection_noise",
        dist.Normal(loc=0, scale=0.1 * jnp.ones((data.nRs, data.nDs))),
    )
    # enforce positivity!
    infections = jax.nn.softplus(
        infections + (infection_noise_scale * (10.0 * infection_noise.T))
    )

    total_infections = jax.ops.index_update(
        total_infections_placeholder,
        jax.ops.index[:, :seeding_padding],
        init_infections[:, -seeding_padding:],
    )
    total_infections = numpyro.deterministic(
        "total_infections",
        jax.ops.index_update(
            total_infections, jax.ops.index[:, seeding_padding:], infections.T
        ),
    )

    future_cases_t = numpyro.deterministic("future_cases_t", total_infections)
    future_deaths_t = numpyro.deterministic(
        "future_deaths_t", jnp.multiply(total_infections, cfr)
    )

    # collect expected cases and deaths
    expected_cases = numpyro.deterministic(
        "expected_cases",
        jax.scipy.signal.convolve2d(future_cases_t, ep.DPC, mode="full")[
        :, seeding_padding: seeding_padding + data.nDs
        ],
    )

    expected_deaths = numpyro.deterministic(
        "expected_deaths",
        jax.scipy.signal.convolve2d(future_deaths_t, ep.DPD, mode="full")[
        :, seeding_padding: seeding_padding + data.nDs
        ],
    )

    # country specific psi cases and deaths.
    # We will use the 'RC' matrix to pull the correct local area value.
    psi_cases = numpyro.sample(
        "psi_cases",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )
    psi_deaths = numpyro.sample(
        "psi_deaths",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )

    # use the per country psi_cases and psi_deaths and form a nRs x nDs array
    # to use for the output distribution.
    cases_conc = (
        (data.RC_mat @ psi_cases).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )
    deaths_conc = (
        (data.RC_mat @ psi_deaths).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_cases.mask)):
        numpyro.sample(
            "observed_cases",
            dist.GammaPoisson(
                concentration=cases_conc,
                rate=cases_conc / expected_cases,
            ),
            obs=data.new_cases.data,
        )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_deaths.mask)):
        numpyro.sample(
            "observed_deaths",
            dist.GammaPoisson(
                concentration=deaths_conc,
                rate=deaths_conc / expected_deaths,
            ),
            obs=data.new_deaths.data,
        )
        
        
def arma_model(
        data,
        ep,
        basic_R_prior=None,
        cm_method='nn',
        interact = [],
        noise_method='arma_fix',
        noise_scale_prior=0.15,        # this was 0.15 originally, then 0.02 (v small so as to make sure R_cms is not dominated by R_noise)
        noise_period=7,
        arma_p=2,
        arma_q=1,
        n_days_seeding=7,
        seeding_scale=3.0,
        infection_noise_scale=5.0,
        output_noise_scale_prior=5.0,
        **kwargs,
):
    """
    Main model.

    :param data: PreprocessedData object
    :param ep: EpidemiologicalParameters object
    :param predict: int representing number of days in the future to predict
    :param basic_R_prior: basic r prior dict
    :param cm_method: in ['linear', 'nn']
    :param interact: list of cms indices to consider interactions of
    :param noise_method: in ['no_noise', 'random_walk', 'arma_learn', 'arma_fix']
    :param r_walk_noise_scale_prior: scale of random walk noise scale prior
    :param r_walk_period: period of random walk
    :param n_days_seeding: number of days of seeding
    :param seeding_scale: scale of seeded infection prior
    :param infection_noise_scale: scale of infection noise
    :param output_noise_scale_prior: output noise scale prior
    :param kwargs: additional kwargs (not used, but maintain function signature)
    """
    for k in kwargs.keys():
        print(f"{k} is not being used")
        

    # First, compute R using a bayesian neural network

    # sample basic R
    basic_R = sample_basic_R(data.nRs, basic_R_prior)
    
    if cm_method == 'linear':

        alpha_i = sample_intervention_effects(data.nCMs, None)
        # transmission reduction
        cm_reduction = jnp.sum(data.active_cms * alpha_i.reshape((1, data.nCMs, 1)), axis=1)
        
    elif cm_method == 'nn':
        
        # Idea is to use many neural nets on pairs of cms as defined by "interact". Each pair produces 2 outputs. 
        # Linear layer combines all nn outputs with the rest of the cms
    
        # Sample linear layer coefficients
        alpha_i = sample_intervention_effects(data.nCMs, None) 
        
        # Number of neural nets we need to produce
        num_pairs_interact = len(jnp.ravel(jnp.array(interact)))//2
        
        # neural net hidden layers
        D_h1, D_h2 = 2, 2
        
        # get neural net parameters
        parameter_list = sample_bnn_weights(D_h1, D_h2, D_in=2, D_out=2, num_pairs_interact=num_pairs_interact)
        
        # use neural nets to get the linear layer
        lin_input = get_lin_layer(parameter_list, data.active_cms, interact)
        
        # compute cm_reduction using the linear layer
        cm_reduction = jnp.sum(lin_input * alpha_i.reshape((1, data.nCMs, 1)), axis=1)
        
    elif cm_method == 'linear_interact':
        
        num_pairs_interact = len(jnp.ravel(jnp.array(interact)))//2
        
        cms = data.active_cms.astype(int)
    
        CMs_interactions = cms
        
        for interacting_pair in interact:
            
            new_CM_interaction = np.bitwise_and(cms[:,[interacting_pair[0]],:], cms[:,[interacting_pair[1]],:])
            CMs_interactions = jnp.concatenate((CMs_interactions,new_CM_interaction), axis=1)
        
        alpha_i = sample_intervention_effects(data.nCMs+num_pairs_interact, None)
        
        # transmission reduction
        cm_reduction = jnp.sum(CMs_interactions * alpha_i.reshape((1, data.nCMs+num_pairs_interact, 1)), axis=1)

    # Collect in numpyro trace
    R_cms = numpyro.deterministic("R_cms", jnp.exp(jnp.log(basic_R.reshape((data.nRs, 1))) - cm_reduction))

    # Introduce noise into R_cms using... (Rt_noise is what we multiply R_cms with to get R_total)

    if noise_method == 'no_noise':

        # Collect in numpyro trace
        Rt_noise = numpyro.deterministic("Rt_noise", jnp.ones((data.nRs, data.nDs)))

    elif noise_method == 'arma_learn' or 'arma_fix':

        # number of 'noise points'
        # -1 since no change for the first 2 weeks.
        nNP = int(data.nDs / noise_period) - 1
        
        if noise_method == 'arma_learn':
            
        # We will learn the parameters of arma
            if arma_p != 0:
                arma_p_coeff = numpyro.sample(
                "arma_p_coeff",
                dist.Normal(loc=jnp.zeros((1, arma_p)), scale=0.3),)
                arma_p_coeff = jnp.expand_dims(arma_p_coeff, 0)
                p_padding_logR = jnp.zeros((1, data.nRs, arma_p))
            else:
                arma_p_coeff=None
                p_padding_logR=None

            if arma_q != 0:
                arma_q_coeff = numpyro.sample(
                "arma_q_coeff",
                dist.Normal(loc=jnp.zeros((1, arma_q)), scale=0.3),)
                arma_q_coeff = jnp.expand_dims(arma_q_coeff, 0)
                q_padding_err = jnp.zeros((1, data.nRs, arma_q))
            else:
                arma_q_coeff=None
                q_padding_err=None

            arma_const_coeff = numpyro.sample(
                "arma_const_coeff",
                dist.Normal(loc=jnp.zeros((1, 1)), scale=0.5),)
            arma_const_coeff = jnp.expand_dims(arma_const_coeff, 0)

        
        elif noise_method == 'arma_fix':
            
        # We fix the parameters of arma
            if arma_p != 0:
                arma_p_coeff = jnp.ones((data.nRs, arma_p))/arma_p
                numpyro.deterministic("arma_p_coeff", arma_p_coeff)
                arma_p_coeff = jnp.expand_dims(arma_p_coeff, 0)
                p_padding_logR = jnp.zeros((1, data.nRs, arma_p))
            else:
                arma_p_coeff=None
                p_padding_logR=None
            
            if arma_q != 0:
                arma_q_coeff = jnp.ones((data.nRs, arma_q))/arma_q
                numpyro.deterministic("arma_q_coeff", arma_q_coeff)
                arma_q_coeff = jnp.expand_dims(arma_q_coeff, 0)
                q_padding_err = jnp.zeros((1, data.nRs, arma_q))
            else:
                arma_q_coeff=None
                q_padding_err=None
            
            arma_const_coeff = jnp.zeros(((data.nRs, 1)))
            numpyro.deterministic("arma_const_coeff", arma_const_coeff)
            arma_const_coeff = jnp.expand_dims(arma_const_coeff, 0)

    
        arma_noise_scale = numpyro.sample(
            "arma_error_scale", dist.HalfNormal(scale=noise_scale_prior)
        )
        
        # rescaling variables by 10 for better NUTS adaptation
        log_R_noise_unscaled = numpyro.sample(
            "log_R_res",
            dist.Normal(loc=jnp.zeros((data.nRs, nNP)), scale=1.0 / 10),
        )
        log_R_noise = jnp.expand_dims(log_R_noise_unscaled, 0) * arma_noise_scale * 10.0
        
        arima_transitions = get_arima_transition_function(arma_p_coeff, arma_q_coeff, arma_const_coeff)

        _, log_R_arma = jax.lax.scan(arima_transitions, (q_padding_err, p_padding_logR),
                                      jnp.transpose(log_R_noise, (2,0,1)))
        
        log_R_arma = jnp.transpose(log_R_arma, (1,2,0))
        
        expanded_log_R_arma = jnp.repeat(
            log_R_arma,
            noise_period,
            axis=-1,
        )[:, : data.nRs, : (data.nDs - 2 * noise_period)]
            
        # except that we assume no noise for the first 3 weeks
        full_log_Rt_noise = jnp.zeros((1, data.nRs, data.nDs))
        full_log_Rt_noise = jax.ops.index_update(
            full_log_Rt_noise, jax.ops.index[:, :, 2 * noise_period:], expanded_log_R_arma
        )
        full_log_Rt_noise = full_log_Rt_noise[0,:,:]
        
        Rt_noise = numpyro.deterministic("Rt_noise", jnp.exp(full_log_Rt_noise))

    # Now collect Rt in the trace, no matter what Rt_noise was generated with
    Rt = numpyro.deterministic("Rt", 
                jnp.exp(jnp.log(basic_R.reshape((data.nRs, 1))) - cm_reduction + jnp.log(Rt_noise)))

    # Infection Model
    seeding_padding = n_days_seeding
    total_padding = ep.GIv.size - 1

    # note; seeding is also rescaled
    init_infections, total_infections_placeholder = seed_infections(
        seeding_scale, data.nRs, data.nDs, seeding_padding, total_padding
    )
    discrete_renewal_transition = get_discrete_renewal_transition(ep)

    # we need to transpose R because jax.lax.scan scans over the first dimension.
    # We want to scan over time instead of regions!
    _, infections = jax.lax.scan(
        discrete_renewal_transition,
        init_infections,
        Rt.T,
    )

    # corrupt infections with additive noise, adding robustness at small case and death
    # counts
    infection_noise = numpyro.sample(
        "infection_noise",
        dist.Normal(loc=0, scale=0.1 * jnp.ones((data.nRs, data.nDs))),
    )
    # enforce positivity!
    infections = jax.nn.softplus(
        infections + (infection_noise_scale * (10.0 * infection_noise.T))
    )

    total_infections = jax.ops.index_update(
        total_infections_placeholder,
        jax.ops.index[:, :seeding_padding],
        init_infections[:, -seeding_padding:],
    )
    total_infections = numpyro.deterministic(
        "total_infections",
        jax.ops.index_update(
            total_infections, jax.ops.index[:, seeding_padding:], infections.T
        ),
    )

    # Time constant case fatality rate (ascertainment rate assumed to be 1
    # throughout the whole period).
    cfr = numpyro.sample("cfr", dist.Uniform(low=1e-3, high=jnp.ones((data.nRs, 1))))

    future_cases_t = numpyro.deterministic("future_cases_t", total_infections)
    future_deaths_t = numpyro.deterministic(
        "future_deaths_t", jnp.multiply(total_infections, cfr)
    )

    # collect expected cases and deaths
    expected_cases = numpyro.deterministic(
        "expected_cases",
        jax.scipy.signal.convolve2d(future_cases_t, ep.DPC, mode="full")[
        :, seeding_padding: seeding_padding + data.nDs
        ],
    )

    expected_deaths = numpyro.deterministic(
        "expected_deaths",
        jax.scipy.signal.convolve2d(future_deaths_t, ep.DPD, mode="full")[
        :, seeding_padding: seeding_padding + data.nDs
        ],
    )

    # country specific psi cases and deaths.
    # We will use the 'RC' matrix to pull the correct local area value.
    psi_cases = numpyro.sample(
        "psi_cases",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )
    psi_deaths = numpyro.sample(
        "psi_deaths",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )

    # use the per country psi_cases and psi_deaths and form a nRs x nDs array
    # to use for the output distribution.
    cases_conc = (
        (data.RC_mat @ psi_cases).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )
    deaths_conc = (
        (data.RC_mat @ psi_deaths).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_cases.mask)):
        numpyro.sample(
            "observed_cases",
            dist.GammaPoisson(
                concentration=cases_conc,
                rate=cases_conc / expected_cases,
            ),
            obs=data.new_cases.data,
        )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_deaths.mask)):
        numpyro.sample(
            "observed_deaths",
            dist.GammaPoisson(
                concentration=deaths_conc,
                rate=deaths_conc / expected_deaths,
            ),
            obs=data.new_deaths.data,
        )


def default_model(
        data,
        ep,
        intervention_prior=None,
        basic_R_prior=None,
        r_walk_noise_scale_prior=0.15,
        r_walk_period=7,
        n_days_seeding=7,
        seeding_scale=3.0,
        infection_noise_scale=5.0,
        output_noise_scale_prior=5.0,
        **kwargs,
):
    """
    Main model.

    :param data: PreprocessedData object
    :param ep: EpidemiologicalParameters object
    :param intervention_prior: intervention prior dict
    :param basic_R_prior: basic r prior dict
    :param r_walk_noise_scale_prior: scale of random walk noise scale prior
    :param r_walk_period: period of random walk
    :param n_days_seeding: number of days of seeding
    :param seeding_scale: scale of seeded infection prior
    :param infection_noise_scale: scale of infection noise
    :param output_noise_scale_prior: output noise scale prior
    :param kwargs: additional kwargs (not used, but maintain function signature)
    """
    for k in kwargs.keys():
        print(f"{k} is not being used")

    # First, compute R.
    # sample intervention effects from their priors.
    # mean intervention effects
    alpha_i = sample_intervention_effects(data.nCMs, intervention_prior)
    # transmission reduction
    cm_reduction = jnp.sum(data.active_cms * alpha_i.reshape((1, data.nCMs, 1)), axis=1)
    # sample basic R
    basic_R = sample_basic_R(data.nRs, basic_R_prior)

    # number of 'noise points'
    # -1 since no change for the first 2 weeks.
    nNP = int(data.nDs / r_walk_period) - 1

    r_walk_noise_scale = numpyro.sample(
        "r_walk_noise_scale", dist.HalfNormal(scale=r_walk_noise_scale_prior)
    )

    # rescaling variables by 10 for better NUTS adaptation
    r_walk_noise = numpyro.sample(
        "r_walk_noise",
        dist.Normal(loc=jnp.zeros((data.nRs, nNP)), scale=1.0 / 10),
    )

    # only apply the noise every "r_walk_period" - to get full noise, repeat
    expanded_r_walk_noise = jnp.repeat(
        r_walk_noise_scale * 10.0 * jnp.cumsum(r_walk_noise, axis=-1),
        r_walk_period,
        axis=-1,
    )[: data.nRs, : (data.nDs - 2 * r_walk_period)]
    # except that we assume no noise for the first 3 weeks
    full_log_Rt_noise = jnp.zeros((data.nRs, data.nDs))
    full_log_Rt_noise = jax.ops.index_update(
        full_log_Rt_noise, jax.ops.index[:, 2 * r_walk_period:], expanded_r_walk_noise
    )

    Rt = numpyro.deterministic(
        "Rt",
        jnp.exp(
            jnp.log(basic_R.reshape((data.nRs, 1))) + full_log_Rt_noise - cm_reduction
        ),
    )

    # collect variables in the numpyro trace
    numpyro.deterministic("Rt_walk", jnp.exp(full_log_Rt_noise))
    numpyro.deterministic(
        "Rt_cm", jnp.exp(jnp.log(basic_R.reshape((data.nRs, 1))) - cm_reduction)
    )

    # Infection Model
    seeding_padding = n_days_seeding
    total_padding = ep.GIv.size - 1

    # note; seeding is also rescaled
    init_infections, total_infections_placeholder = seed_infections(
        seeding_scale, data.nRs, data.nDs, seeding_padding, total_padding
    )
    discrete_renewal_transition = get_discrete_renewal_transition(ep)

    # we need to transpose R because jax.lax.scan scans over the first dimension.
    # We want to scan over time instead of regions!
    _, infections = jax.lax.scan(
        discrete_renewal_transition,
        init_infections,
        Rt.T,
    )

    # corrupt infections with additive noise, adding robustness at small case and death
    # counts
    infection_noise = numpyro.sample(
        "infection_noise",
        dist.Normal(loc=0, scale=0.1 * jnp.ones((data.nRs, data.nDs))),
    )
    # enforce positivity!
    infections = jax.nn.softplus(
        infections + (infection_noise_scale * (10.0 * infection_noise.T))
    )

    total_infections = jax.ops.index_update(
        total_infections_placeholder,
        jax.ops.index[:, :seeding_padding],
        init_infections[:, -seeding_padding:],
    )
    total_infections = numpyro.deterministic(
        "total_infections",
        jax.ops.index_update(
            total_infections, jax.ops.index[:, seeding_padding:], infections.T
        ),
    )

    # Time constant case fatality rate (ascertainment rate assumed to be 1
    # throughout the whole period).
    cfr = numpyro.sample("cfr", dist.Uniform(low=1e-3, high=jnp.ones((data.nRs, 1))))

    future_cases_t = numpyro.deterministic("future_cases_t", total_infections)
    future_deaths_t = numpyro.deterministic(
        "future_deaths_t", jnp.multiply(total_infections, cfr)
    )

    # collect expected cases and deaths
    expected_cases = numpyro.deterministic(
        "expected_cases",
        jax.scipy.signal.convolve2d(future_cases_t, ep.DPC, mode="full")[
        :, seeding_padding: seeding_padding + data.nDs
        ],
    )

    expected_deaths = numpyro.deterministic(
        "expected_deaths",
        jax.scipy.signal.convolve2d(future_deaths_t, ep.DPD, mode="full")[
        :, seeding_padding: seeding_padding + data.nDs
        ],
    )

    # country specific psi cases and deaths.
    # We will use the 'RC' matrix to pull the correct local area value.
    psi_cases = numpyro.sample(
        "psi_cases",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )
    psi_deaths = numpyro.sample(
        "psi_deaths",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )

    # use the per country psi_cases and psi_deaths and form a nRs x nDs array
    # to use for the output distribution.
    cases_conc = (
        (data.RC_mat @ psi_cases).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )
    deaths_conc = (
        (data.RC_mat @ psi_deaths).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_cases.mask)):
        numpyro.sample(
            "observed_cases",
            dist.GammaPoisson(
                concentration=cases_conc,
                rate=cases_conc / expected_cases,
            ),
            obs=data.new_cases.data,
        )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_deaths.mask)):
        numpyro.sample(
            "observed_deaths",
            dist.GammaPoisson(
                concentration=deaths_conc,
                rate=deaths_conc / expected_deaths,
            ),
            obs=data.new_deaths.data,
        )


def default_model_uk_ifriar(
        data,
        ep,
        intervention_prior=None,
        basic_R_prior=None,
        r_walk_noise_scale_prior=0.15,
        r_walk_period=7,
        n_days_seeding=7,
        seeding_scale=3.0,
        infection_noise_scale=5.0,
        output_noise_scale_prior=5.0,
        **kwargs,
):
    """
    Identical to base model, except use hard coded IFR/IAR estimates.

    :param data: PreprocessedData object
    :param ep: EpidemiologicalParameters object
    :param intervention_prior: intervention prior dict
    :param basic_R_prior: basic r prior dict
    :param r_walk_noise_scale_prior: scale of random walk noise scale prior
    :param r_walk_period: period of random walk
    :param n_days_seeding: number of days of seeding
    :param seeding_scale: scale of seeded infection prior
    :param infection_noise_scale: scale of infection noise
    :param output_noise_scale_prior: output noise scale prior
    :param kwargs: additional kwargs (not used, but maintain function signature)
    """
    for k in kwargs.keys():
        print(f"{k} is not being used")

    # First, compute R.
    # sample intervention effects from their priors.
    # mean intervention effects
    alpha_i = sample_intervention_effects(data.nCMs, intervention_prior)
    # transmission reduction
    cm_reduction = jnp.sum(data.active_cms * alpha_i.reshape((1, data.nCMs, 1)), axis=1)

    basic_R = sample_basic_R(data.nRs, basic_R_prior)

    # number of 'noise points'
    # -2 since no change for the first 3 weeks.
    nNP = int(data.nDs / r_walk_period) - 1

    r_walk_noise_scale = numpyro.sample(
        "r_walk_noise_scale", dist.HalfNormal(scale=r_walk_noise_scale_prior)
    )

    # rescaling variables by 10 for better NUTS adaptation
    r_walk_noise = numpyro.sample(
        "r_walk_noise",
        dist.Normal(loc=jnp.zeros((data.nRs, nNP)), scale=1.0 / 10),
    )

    # only apply the noise every "r_walk_period" - to get full noise, repeat
    expanded_r_walk_noise = jnp.repeat(
        r_walk_noise_scale * 10.0 * jnp.cumsum(r_walk_noise, axis=-1),
        r_walk_period,
        axis=-1,
    )[: data.nRs, : (data.nDs - 2 * r_walk_period)]
    # except that we assume no noise for the first 3 weeks
    full_log_Rt_noise = jnp.zeros((data.nRs, data.nDs))
    full_log_Rt_noise = jax.ops.index_update(
        full_log_Rt_noise, jax.ops.index[:, 2 * r_walk_period:], expanded_r_walk_noise
    )

    Rt = numpyro.deterministic(
        "Rt",
        jnp.exp(
            jnp.log(basic_R.reshape((data.nRs, 1))) + full_log_Rt_noise - cm_reduction
        ),
    )

    # collect variables in the numpyro trace
    numpyro.deterministic("Rt_walk", jnp.exp(full_log_Rt_noise))
    numpyro.deterministic(
        "Rt_cm", jnp.exp(jnp.log(basic_R.reshape((data.nRs, 1))) - cm_reduction)
    )

    # Infection Model
    seeding_padding = n_days_seeding
    total_padding = ep.GIv.size - 1

    # note; seeding is also rescaled
    init_infections, total_infections_placeholder = seed_infections(
        seeding_scale, data.nRs, data.nDs, seeding_padding, total_padding
    )
    discrete_renewal_transition = get_discrete_renewal_transition(ep)

    # we need to transpose R because jax.lax.scan scans over the first dimension.
    # We want to scan over time instead of regions!
    _, infections = jax.lax.scan(
        discrete_renewal_transition,
        init_infections,
        Rt.T,
    )

    # corrupt infections with additive noise, adding robustness at small case and death
    # counts
    infection_noise = numpyro.sample(
        "infection_noise",
        dist.Normal(loc=0, scale=0.1 * jnp.ones((data.nRs, data.nDs))),
    )
    # enforce positivity!
    infections = jax.nn.softplus(
        infections + (infection_noise_scale * (10.0 * infection_noise.T))
    )

    total_infections = jax.ops.index_update(
        total_infections_placeholder,
        jax.ops.index[:, :seeding_padding],
        init_infections[:, -seeding_padding:],
    )
    total_infections = numpyro.deterministic(
        "total_infections",
        jax.ops.index_update(
            total_infections, jax.ops.index[:, seeding_padding:], infections.T
        ),
    )

    # Scale by fixed UK numbers
    iar_t = jnp.array(
        [
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
        ]
    )

    ifr_t = jnp.array(
        [
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
        ]
    )

    # use the `RC_mat` to pull the country level change in the rates for the relevant local area
    future_cases_t = numpyro.deterministic(
        "future_cases_t", jnp.multiply(total_infections, iar_t)
    )
    future_deaths_t = numpyro.deterministic(
        "future_cases_t", jnp.multiply(total_infections, ifr_t)
    )

    # collect expected cases and deaths
    expected_cases = numpyro.deterministic(
        "expected_cases",
        jax.scipy.signal.convolve2d(future_cases_t, ep.DPC, mode="full")[
        :, seeding_padding: seeding_padding + data.nDs
        ],
    )

    expected_deaths = numpyro.deterministic(
        "expected_deaths",
        jax.scipy.signal.convolve2d(future_deaths_t, ep.DPD, mode="full")[
        :, seeding_padding: seeding_padding + data.nDs
        ],
    )

    # country specific psi cases and deaths.
    # We will use the 'RC' matrix to pull the correct local area value.
    psi_cases = numpyro.sample(
        "psi_cases",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )
    psi_deaths = numpyro.sample(
        "psi_deaths",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )

    # use the per country psi_cases and psi_deaths and form a nRs x nDs array
    # to use for the output distribution.
    cases_conc = (
        (data.RC_mat @ psi_cases).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )
    deaths_conc = (
        (data.RC_mat @ psi_deaths).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_cases.mask)):
        numpyro.sample(
            "observed_cases",
            dist.GammaPoisson(
                concentration=cases_conc,
                rate=cases_conc / expected_cases,
            ),
            obs=data.new_cases.data,
        )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_deaths.mask)):
        numpyro.sample(
            "observed_deaths",
            dist.GammaPoisson(
                concentration=deaths_conc,
                rate=deaths_conc / expected_deaths,
            ),
            obs=data.new_deaths.data,
        )


def random_walk_model(
        data,
        ep,
        basic_R_prior=None,
        r_walk_noise_scale_prior=0.15,
        r_walk_period=7,
        n_days_seeding=7,
        seeding_scale=3.0,
        infection_noise_scale=5.0,
        output_noise_scale_prior=5.0,
        **kwargs,
):
    """
    Random walk only model

    :param data: PreprocessedData object
    :param ep: EpidemiologicalParameters object
    :param basic_R_prior: basic r prior dict
    :param r_walk_noise_scale_prior: scale of random walk noise scale prior
    :param r_walk_period: period of random walk
    :param n_days_seeding: number of days of seeding
    :param seeding_scale: scale of seeded infection prior
    :param infection_noise_scale: scale of infection noise
    :param output_noise_scale_prior: output noise scale prior
    :param kwargs: additional kwargs (not used, but maintain function signature)
    """
    for k in kwargs.keys():
        print(f"{k} is not being used")

    basic_R = sample_basic_R(data.nRs, basic_R_prior)

    # number of 'noise points'
    nNP = (
            int(data.nDs / r_walk_period) - 1
    )  # -1 since no change for the first 2 weeks. +1 (round up) - 2 since fixed for the first 2 weeks

    r_walk_noise_scale = numpyro.sample(
        "r_walk_noise_scale", dist.HalfNormal(scale=r_walk_noise_scale_prior)
    )

    # rescaling variables by 10 for better NUTS adaptation
    r_walk_noise = numpyro.sample(
        "r_walk_noise",
        dist.Normal(loc=jnp.zeros((data.nRs, nNP)), scale=1.0 / 10),
    )

    # only apply the noise every "r_walk_period" - to get full noise, repeat
    expanded_r_walk_noise = jnp.repeat(
        r_walk_noise_scale * 10.0 * jnp.cumsum(r_walk_noise, axis=-1),
        r_walk_period,
        axis=-1,
    )[: data.nRs, : (data.nDs - 2 * r_walk_period)]
    # except that we assume no noise for the first 3 weeks
    full_log_Rt_noise = jnp.zeros((data.nRs, data.nDs))
    full_log_Rt_noise = jax.ops.index_update(
        full_log_Rt_noise, jax.ops.index[:, 2 * r_walk_period:], expanded_r_walk_noise
    )

    Rt = numpyro.deterministic(
        "Rt",
        jnp.exp(jnp.log(basic_R.reshape((data.nRs, 1))) + full_log_Rt_noise),
    )

    # collect variables in the numpyro trace
    numpyro.deterministic("Rt_walk", jnp.exp(full_log_Rt_noise))
    numpyro.deterministic("Rt_cm", jnp.exp(jnp.log(basic_R.reshape((data.nRs, 1)))))

    # Infection Model
    seeding_padding = n_days_seeding
    total_padding = ep.GIv.size - 1

    # note; seeding is also rescaled
    init_infections, total_infections_placeholder = seed_infections(
        seeding_scale, data.nRs, data.nDs, seeding_padding, total_padding
    )
    discrete_renewal_transition = get_discrete_renewal_transition(ep)

    # we need to transpose R because jax.lax.scan scans over the first dimension.
    # We want to scan over time instead of regions!
    _, infections = jax.lax.scan(
        discrete_renewal_transition,
        init_infections,
        Rt.T,
    )

    # corrupt infections with additive noise, adding robustness at small case and death
    # counts
    infection_noise = numpyro.sample(
        "infection_noise",
        dist.Normal(loc=0, scale=0.1 * jnp.ones((data.nRs, data.nDs))),
    )
    # enforce positivity!
    infections = jax.nn.softplus(
        infections + (infection_noise_scale * (10.0 * infection_noise.T))
    )

    total_infections = jax.ops.index_update(
        total_infections_placeholder,
        jax.ops.index[:, :seeding_padding],
        init_infections[:, -seeding_padding:],
    )
    total_infections = numpyro.deterministic(
        "total_infections",
        jax.ops.index_update(
            total_infections, jax.ops.index[:, seeding_padding:], infections.T
        ),
    )

    # Time constant case fatality rate (ascertainment rate assumed to be 1
    # throughout the whole period).
    cfr = numpyro.sample("cfr", dist.Uniform(low=1e-3, high=jnp.ones((data.nRs, 1))))

    future_cases_t = numpyro.deterministic("future_cases_t", total_infections)
    future_deaths_t = numpyro.deterministic(
        "future_deaths_t", jnp.multiply(total_infections, cfr)
    )

    # collect expected cases and deaths
    expected_cases = numpyro.deterministic(
        "expected_cases",
        jax.scipy.signal.convolve2d(future_cases_t, ep.DPC, mode="full")[
        :, seeding_padding: seeding_padding + data.nDs
        ],
    )

    expected_deaths = numpyro.deterministic(
        "expected_deaths",
        jax.scipy.signal.convolve2d(future_deaths_t, ep.DPD, mode="full")[
        :, seeding_padding: seeding_padding + data.nDs
        ],
    )

    # country specific psi cases and deaths.
    # We will use the 'RC' matrix to pull the correct local area value.
    psi_cases = numpyro.sample(
        "psi_cases",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )
    psi_deaths = numpyro.sample(
        "psi_deaths",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )

    # use the per country psi_cases and psi_deaths and form a nRs x nDs array
    # to use for the output distribution.
    cases_conc = (
        (data.RC_mat @ psi_cases).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )
    deaths_conc = (
        (data.RC_mat @ psi_deaths).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_cases.mask)):
        numpyro.sample(
            "observed_cases",
            dist.GammaPoisson(
                concentration=cases_conc,
                rate=cases_conc / expected_cases,
            ),
            obs=data.new_cases.data,
        )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_deaths.mask)):
        numpyro.sample(
            "observed_deaths",
            dist.GammaPoisson(
                concentration=deaths_conc,
                rate=deaths_conc / expected_deaths,
            ),
            obs=data.new_deaths.data,
        )
