from typing import Dict
import numpy as onp
import jax.numpy as np
from latentNoise_funcs_gen   import *
import json
import sys
# allows for arguments
#import argparse




def main_experiment_van(dataset: np.ndarray, lam: np.ndarray, sig: np.ndarray, name: str) -> Dict:
    #lam = np.array([0.001]) # function f(x,z) complexity
    num_epochs = 501
    report_freq = 500
    num_reps = 5
    batch_size = 100


    res_van = getModels_van(dataset, lam, sig)

    res = {"van": res_van}
    print("res_van")
    print(res_van)

    return res



if __name__ == "__main__":
    # load dataset
    # dataset = ...

    lam = np.array([float(sys.argv[1])]) # function f(x,z) complexity
    sig = np.array([float(sys.argv[2])])

    server = str(sys.argv[3])
    if server == "erc":
        file= "/media/disk/databases/latentNoise/TCEPs/dag2-ME2-SIM-1000_withZ_sims.json"
        #file= "/media/disk/databases/latentNoise/TCEPs/dag2-ME2-TCEP-all_sims.json"

    if server == "myLap":
        #file = "../data/TCEPs/dag2-ME2-SIM-1000_withZ_sims.json"
        file = "../data/ANLSMN/dag2-ME2-ANLSMN_withZ_sims.json"
        #file = "../data/TCEPs/dag2-ME2-TCEP-all_sims.json"

    with open(file) as json_file:
        dataset = json.load(json_file)


    nm = "AN.1" #"1"#"AN.67"#"SIM.1", "107"
    dataset = onp.array(dataset['xs'][nm])
    print("dataset shape", dataset.shape)
    #dataset = jitter(dataset)
    dataset = np.array(norml_mat(dataset))

    print("dataset shape",dataset.shape)
    # run experiment
    results = main_experiment_van(dataset, lam, nm)
    print("hsic_zzhat")
    print("finished")
    # save to somewhere
    # save_shit
