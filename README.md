# latentNoise
Latent Noise imputation bivariate causal discovery method

## Main Function files
1. src/latentNoise_funcs_gen.py: contains functions to estimate latent noise, fit KRR and calculate complexity measures
2. src/processResults.py: contains functions to pair up and weight models and to calculate performance measures

## Additional function files
1. src/ANLSMN_genfuncs.py: generate artificial data from AN, AN-s, LS, LS-s and MN-U classes
2. src/slope.py: SLOPE bivariate causal algorithm (from Marx & Vreeken 2019)
3. src/func_entropy_v1: different entropy estimation methods.
4. src/kdpee: kdp entropy estimation c files called from func_entropy_v1

## Main script files
1. src/slurm_script.py: carry out all LNc experiments. 
2. src/experiment.py: carry out one LNc experiment (called by slurm_script.py)
3. src/slurm_script_van.py: carry out all ANMh experiments.
4. src/experiment_van.py: carry one ANMh experiment (called by slurm_script_van.py)
5. 

## Other script files
1. 

## Slurm Files


## Data Files
