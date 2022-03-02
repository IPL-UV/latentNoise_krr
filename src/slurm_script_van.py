from typing import Dict
import numpy as onp
import pandas as pd
import jax.numpy as np
import json
from latentNoise_funcs_gen import *
from processResults import *
import bisect
import pickle
import os.path

from experiment_van import main_experiment_van
# allows for arguments
import argparse

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)



def load_dataset(dataset_num: int = 0, server: str="myLap") -> (np.ndarray, str):
    # ======================
    # GET THE DATASET
    # ======================

    """
    1) take the job
        (dataset_num)

    2) match the number of the id to the dataset

    3) load the dataset to memory

    4) check it's numpy array

    4) return
    """
    print("enter load_dataset")
    # Read in and prepare files
    if server == "erc":
        repos = "/media/disk/databases/latentNoise/"

    if server == "myLap":
        repos = paste("/home/emiliano/causaLearner/data/", sep="")



    # declare parmeters


    pars = {"lambda": [0.0001, 0.001, 0.01, 0.1],
            "sig": [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]}

    fileDict = {"TCEP-all": ['tcep'],
                "SIM-1000_withZ": ['SIM', 'SIMc', 'SIMG', 'SIMln'],
                "ANLSMN_withZ": ['AN', 'AN-s', 'LS', 'LS-s','MN-U']}

    #fileDict = {"ANLSMN_withZ": ['LS-s']}

    datasetTab, data = getDataSetTab(repos, pars, fileDict, func_dict)

    job = dataset_num
    print(f"Starting job: {job}")

    indx_set = bisect.bisect_left(datasetTab["cumJobs_fin"], job)
    set = datasetTab["fileNms"][indx_set]
    indxSet = list(onp.where([setDt == set for setDt in list(fileDict.keys())]))[0][0]
    indx_dataset = job - (datasetTab["cumJobs_ini"][indx_set])
    file = datasetTab["fileNames"][indx_set]
    fileNm = datasetTab["fileNms"][indx_set]
    lam = datasetTab["lambda"][indx_set]
    sig = datasetTab["sig"][indx_set]
    pars = {"lambda": lam, "sig": sig}

    # with open(file) as json_file:
    #    data = json.load(json_file)
    data = data[indxSet]
    nm = list(data.keys())[indx_dataset]
    X = data[nm]
    X = onp.array(X)

    print("set: ", datasetTab["fileNms"][indx_set])
    print("dataset: ", nm)

    # cap data
    maxData = 1000
    if X.shape[0] > maxData:
        smpl = onp.random.randint(low=1, high=X.shape[0], size=maxData)
        X = X[smpl, :]

    if (str(nm) == "8") | (str(nm) == "107") | (str(nm) == "70") & (fileNm == "TCEP-all"):
        print("jittering")
        X = jitter(X)

    X = np.array(norml_mat(X))

    return nm, X, pars  # load shit


def main(args):

    # load dataset from job array id
    job = int(args.job) + int(args.offset)
    print("load")
    nm, data, pars = load_dataset(dataset_num=job, server=args.server)
    print("nm: ", nm)
    print("shape data", data.shape)
    print("pars: ", pars)

    # do stuffs (Latent Noise-KRR over the data)
    print("getLatenZs etc")
    lam = np.array([pars["lambda"]])
    sig = np.array([pars["sig"]])

    # save shit (to json)
    print("save")

    if args.server == "erc":
        reposResults = "/home/emiliano/latentnoise_krr/results_van/"

    if args.server == "myLap":
        reposResults = "/home/emiliano/ISP/proyectos/latentNoise_krr/results_van/"


    #save_shit(results, name=f"{args.save}_results_{args.job}.json")
    fileRes = reposResults+"latent_noise_van"+str(job)+".pkl"
    #with open(fileRes, 'w') as outfile:
    #    json.dump(results, outfile)


    if os.path.isfile(fileRes):
        print("File exist")
    else:
        print("File not exist")
        results = main_experiment_van(data, lam, sig, nm)
    	# sample usage
        save_object(results, fileRes)
    
    return "bla"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments LNKRR.")

    # FOR THE JOB ARRRAY
    parser.add_argument("-j", "--job", default=0, type=int, help="job array for dataset")
    parser.add_argument("-o", "--offset", default=0, type=int, help="which job to begin after")
    parser.add_argument("-s", "--save", default="0", type=str, help="version string")
    parser.add_argument("-v", "--server", default="myLap", type=str, help="server to run in")
    # run experiment
    
    args = parser.parse_args()
    print(args)
    print("run experiment")
    results = main(args)
    print("finished")

