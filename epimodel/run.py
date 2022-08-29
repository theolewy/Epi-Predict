from epimodel import EpidemiologicalParameters, latent_nn_model, preprocess_data, run_model

def run_model_with_settings(
    data_path='../data/all_merged_data_2021-01-22.csv',
    end_date="2020-12-10",
    num_samples=50,
    num_warmup=100,
    num_chains=4,
    save_results=True,
    model_kwargs=None,
):
    
    f_name = end_date + '_warmup_'+ str(num_warmup)
    
    for key, value in model_kwargs.items():
        
         f_name += '_' + key + '_' + str(value)

    f_name += '.netcdf'
    
    try:
        num_percentiles = model_kwargs['num_percentiles']
    except:
        num_percentiles = 9
        
    try:
        n_days_nn_input = model_kwargs['n_days_nn_input']
    except:
        n_days_nn_input = 21
    
    data = preprocess_data('../../data/all_merged_data_2021-01-22.csv', start_date="2020-08-01", end_date=end_date, num_percentiles=num_percentiles, n_days_nn_input=n_days_nn_input)

    model_settings = model_kwargs
    
    try:
        del model_settings['num_percentiles']
    except:
        pass
    
    run_model(
    latent_nn_model,
    data,
    EpidemiologicalParameters(),
    num_samples=num_samples,
    num_warmup=num_warmup,
    save_results=save_results,
    num_chains=num_chains,
    output_fname=f_name,
    model_kwargs=model_settings
    )

