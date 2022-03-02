from typing import Dict
import numpy as onp
import jax.numpy as np
from latentNoise_funcs_gen   import *
import json
import sys
# allows for arguments
#import argparse




def main_experiment(dataset: np.ndarray, beta: np.ndarray, neta: np.ndarray, eta: np.ndarray, lam: np.ndarray,sig: np.ndarray, nu: np.ndarray, lu: np.ndarray,lr: float, name: str, optType: list, epchs:int, bs:int, reps:int, job:int) -> Dict:
    #beta = np.array([0.0]) # mse and hsic(z,x) penalty
    #eta = np.array([0.0]) # dependence on z_mani helper
    #neta = np.array([0.001])
    #lam = np.array([0.001]) # function f(x,z) complexity

    #num_epochs = 501
    #report_freq = 500
    #num_reps = 5
    #batch_size = 100
    #learning_rate = lr #0.1

    print("optTypes: ", optType)


    start = time.process_time()
    res_latZ = getLatentZs(dataset, name, optType, lam, sig, beta, neta, eta, nu, lu, epchs+1, epchs,
                      reps, bs, lr, job)
    print(time.process_time() - start)  #

    #res_van = getModels_van(dataset, lam)


    res = {"Z": res_latZ}

    return res



if __name__ == "__main__":
    # load dataset
    # dataset = ...

    beta = np.array([float(sys.argv[1])]) # mse and hsic(z,x) penalty
    eta = np.array([float(sys.argv[2])]) # dependence on z_mani helper
    neta = np.array([float(sys.argv[3])])
    lam = np.array([float(sys.argv[4])]) # function f(x,z) complexity
    nu = np.array([float(sys.argv[5])])
    lu = np.array([float(sys.argv[6])])
    lr = np.array([float(sys.argv[7])])

    epchs = 500
    bs=100
    reps = 5

    server = str(sys.argv[8])
    if server == "erc":
        file= "/media/disk/databases/latentNoise/ANLSMN/dag2-ME2-ANLSMN_withZ_sims.json"        
	#file= "/media/disk/databases/latentNoise/TCEPs/dag2-ME2-SIM-1000_withZ_sims.json"
        #file= "/media/disk/databases/latentNoise/TCEPs/dag2-ME2-TCEP-all_sims.json"

    if server == "myLap":
        #file = "../data/TCEPs/dag2-ME2-SIM-1000_withZ_sims.json"
        file = "../data/ANLSMN/dag2-ME2-ANLSMN_withZ_sims.json"
        #file = "../data/TCEPs/dag2-ME2-TCEP-all_sims.json"

    with open(file) as json_file:
        dataset = json.load(json_file)


    print("beta: ", beta)

    nm = "AN.1" #"1"#"AN.67"#"SIM.1", "107"
    dataset = onp.array(dataset['xs'][nm])
    print("dataset shape", dataset.shape)
    #dataset = jitter(dataset)
    dataset = np.array(norml_mat(dataset))
    # ots = ["freeZ-iniMani_mani-postZ", "freeZ-iniMani", "mani","freeZ"]
    optType = "freeZ-iniR"#"freeZ-iniMani_mani-postZ_freeZ"#[1]
    #optType = ["mani"]

    print("dataset shape",dataset.shape)
    # run experiment
    results = main_experiment(dataset, beta, neta, eta, lam, nu, lu,lr, nm, optType,epchs, bs, reps)
    print("hsic_zzhat")
    print(results["Z"]["path_xy"]["hsic_zzhat"])
    print(results["Z"]["path_yx"]["hsic_zzhat"])
    print("finished")
    # save to somewhere
    # save_shit
