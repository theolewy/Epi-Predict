# packages
# install.packages(c("data.table", "remotes", "EpiNow2"))
library(data.table)
library(EpiNow2)
library(covidregionaldata)

# options(mc.cores = ifelse(interactive(), 4, 1))
options(mc.cores = 4)

# construct example distributions
generation_time <- get_generation_time(disease = "SARS-CoV-2", source = "ganyani")
incubation_period <- get_incubation_period(disease = "SARS-CoV-2", source = "lauer")
reporting_delay <- list(mean = convert_to_logmean(3,1),
                        mean_sd = 0.1,
                        sd = convert_to_logsd(3,1),
                        sd_sd = 0.1, max = 15)
                        


cases <- fread('data_for_r_remainder.csv')



## Run basic nowcasting pipeline
def <- regional_epinow(reported_cases = cases, 
                       generation_time = generation_time,
                       delays = delay_opts(incubation_period, reporting_delay),
                       rt = rt_opts(prior = list(mean = 2, sd = 0.2)),
                       stan = stan_opts(samples = 100, warmup = 100, 
                                        control = list(adapt_delta = 0.95)),
                      horizon = 20, 
                      verbose=TRUE,
                      target_folder = "new_results",
                      logs = file.path("logs", Sys.Date()),
                      output = c("regions", "summary", "samples", "plots", "latest"),
                      return_output = TRUE, )