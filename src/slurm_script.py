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

from experiment import main_experiment
# allows for arguments
import argparse

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)



def load_dataset_old(dataset_num: int = 0, server: str="myLap") -> (np.ndarray, str):
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

    # declare files and repositores
    reposRel = (["TCEPs/" for i in range(2)] + ["ANLSMN/"])
    #fileNms = ["TCEP-all", "SIM-1000", "ANLSMN"]
    fileNms = ["TCEP-all", "SIM-1000_withZ", "ANLSMN_withZ"]
    files = ["dag2-ME2-" + nm for nm in fileNms]
    fileNames = [repos + repRel + file + "_sims.json" for (repRel, file) in zip(reposRel, files)]
    numBlocks = [102, 400, 500]

    # declare parmeters

    betas = [0.1, 1.0, 10.0]
    netas = [0.001, 0.01, 0.1]
    etas = [0.01]
    lams = [0.01, 0.1, 1]
    nus = [10.0]
    lus = [0.0, 1.0]
    lrs = [0.1]
    ots = ["freeZ-iniR"]  # ["freeZ-iniMani_mani-postZ", "freeZ-iniMani", "mani","freeZ"]


    combos = {"lambda": [l for l in lams for e in etas for b in betas for n in netas for nu in nus for lu in lus for lr in lrs for ot in ots],
              "eta": [e for l in lams for e in etas for b in betas for n in netas for nu in nus for lu in lus for lr in lrs for ot in ots],
              "beta": [b for l in lams for e in etas for b in betas for n in netas for nu in nus for lu in lus for lr in lrs for ot in ots],
              "neta": [n for l in lams for e in etas for b in betas for n in netas for nu in nus for lu in lus for lr in lrs for ot in ots],
              "nu": [nu for l in lams for e in etas for b in betas for n in netas for nu in nus for lu in lus for lr in lrs for ot in ots],
              "lu": [lu for l in lams for e in etas for b in betas for n in netas for nu in nus for lu in lus for lr in lrs for ot in ots],
              "lr": [lr for l in lams for e in etas for b in betas for n in netas for nu in nus for lu in lus for lr in lrs for ot in ots],
              "ot": [ot for l in lams for e in etas for b in betas for n in netas for nu in nus for lu in lus for lr in lrs for ot in ots]}



    print("create datasetTab")
    datasetTab = {"fileNms": fileNms, "fileNames": fileNames, "numJobs": numBlocks,
                  "cumJobs_ini": [1] + list(onp.cumsum(numBlocks) + 1), "cumJobs_fin": list(onp.cumsum(numBlocks))}

    aux = {"fileNms": fileNms, "fileNames": fileNames, "numJobs": numBlocks}
    aux = pd.DataFrame.from_dict(aux)
    datasetTab2 = {"fileNms": [f for f in fileNms for i in range(len(combos["lambda"]))],
                   "lambda": [l for f in fileNms for l in combos["lambda"]],
                   "eta": [e for f in fileNms for e in combos["eta"]],
                   "beta": [b for f in fileNms for b in combos["beta"]],
                   "neta": [n for f in fileNms for n in combos["neta"]],
                   "nu": [nu for f in fileNms for nu in combos["nu"]],
                   "lu": [lu for f in fileNms for lu in combos["lu"]],
                   "lr": [lr for f in fileNms for lr in combos["lr"]],
                   "ot": [ot for f in fileNms for ot in combos["ot"]]}
    datasetTab2 = pd.DataFrame.from_dict(datasetTab2)
    datasetTab2 = datasetTab2.merge(aux, on="fileNms")
    cumJobs_ini = onp.cumsum(datasetTab2["numJobs"]) + 1
    datasetTab2["cumJobs_ini"] = [1] + list(cumJobs_ini[0:(len(cumJobs_ini) - 1)])
    cumJobs_fin = onp.cumsum(datasetTab2["numJobs"])
    datasetTab2["cumJobs_fin"] = list(cumJobs_fin)

    job = dataset_num
    print(f"Starting job: {job}")


    indx_set = bisect.bisect_left(datasetTab2["cumJobs_fin"], job)
    indx_dataset = job - (datasetTab2["cumJobs_ini"][indx_set])
    file = datasetTab2["fileNames"][indx_set]
    fileNm = datasetTab2["fileNms"][indx_set]
    lam = datasetTab2["lambda"][indx_set]
    beta = datasetTab2["beta"][indx_set]
    eta = datasetTab2["eta"][indx_set]
    neta = datasetTab2["neta"][indx_set]
    nu = datasetTab2["nu"][indx_set]
    lu = datasetTab2["lu"][indx_set]
    lr = datasetTab2["lr"][indx_set]
    ot = datasetTab2["ot"][indx_set]
    pars = {"lambda": lam, "beta": beta, "eta": eta, "neta": neta, "nu":nu,"lu":lu, "lr": lr, "optType":ot}

    with open(file) as json_file:
        data = json.load(json_file)
    nm = list(data["xs"].keys())[indx_dataset]
    X = data["xs"][nm]
    X = onp.array(X)

    print("set: ", datasetTab2["fileNms"][indx_set])
    print("dataset: ", nm)

    # cap data
    maxData = 1000
    if X.shape[0]>maxData:
        smpl = onp.random.randint(low=1, high=X.shape[0], size=maxData)
        X = X[smpl,:]

    if (str(nm)=="8") | (str(nm)=="107") | (str(nm)=="70") & (fileNm == "TCEP-all"):
        print("jittering")
        X = jitter(X)

    X = np.array(norml_mat(X))

    return nm, X, pars  # load shit

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
    print("enter load_dataset NEW")
    # Read in and prepare files
    if server == "erc":
        repos = "/media/disk/databases/latentNoise/"

    if server == "myLap":
        repos = "/home/emiliano/causaLearner/data/"


    # declare parmeters

  
    pars = {"lambda": [0.001, 0.01, 0.1, 1.0],
            "sig":[0.5],
            "eta": [0.001],
            "beta": [0.1, 1.0, 10.0],
            "neta": [0.001,0.01, 0.1],
            "nu": [10.0],
            "lu": [0.0],
            "lr": [0.1],
            "ot": ["freeZ"],
            "epchs": [500],
            "bs":[100],
            "reps":[7]}

    pars = {"lambda": [0.001, 0.01, 0.1, 1.0],
            "sig":[0.5],
            "eta": [0.1],
            "beta": [0.1, 1.0, 10.0],
            "neta": [0.1, 1.0, 10.0],
            "nu": [1000.0],
            "lu": [0.0],
            "lr": [0.1],
            "ot": ["freeZ"],
            "epchs": [500],
            "bs":[100],
            "reps":[7]}
    


    #fileDict = {"TCEP-all": ['tcep'],
    #            "SIM-1000_withZ": ['SIM', 'SIMc', 'SIMG', 'SIMln'],
    #            "ANLSMN_withZ": ['AN', 'AN-s', 'LS', 'LS-s','MN-U']}

    #fileDict = {"ANLSMN_withZ": ['LS', 'LS-s','MN-U']}

    #fileDict = {"SIM-1000_withZ": ['SIM'],"ANLSMN_withZ": ['LS-s','MN-U']}

    #fileDict = {"TCEP-all": ['tcep']}


    fileDict = {"TCEP-all": ['tcep'],
                "ANLSMN_withZ": ['LS-s','MN-U']}

    #fileDict = {"ANLSMN_withZ": ['LS-s']}

    #fileDict = {"SIM-1000_withZ": ['SIM', 'SIMc', 'SIMG', 'SIMln'],
    #            "ANLSMN_withZ": ['AN', 'AN-s', 'LS']}


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
    beta = datasetTab["beta"][indx_set]
    eta = datasetTab["eta"][indx_set]
    neta = datasetTab["neta"][indx_set]
    nu = datasetTab["nu"][indx_set]
    lu = datasetTab["lu"][indx_set]
    lr = datasetTab["lr"][indx_set]
    ot = datasetTab["ot"][indx_set]
    epchs = datasetTab["epchs"][indx_set]
    bs = datasetTab["bs"][indx_set]
    reps = datasetTab["reps"][indx_set]
    pars = {"lambda": lam,"sig":sig, "beta": beta, "eta": eta, "neta": neta, "nu":nu,"lu":lu, "lr": lr, "optType":ot, "epchs":epchs, "bs":bs, "reps":reps}

    #with open(file) as json_file:
    #    data = json.load(json_file)
    data = data[indxSet]
    nm = list(data.keys())[indx_dataset]
    X = data[nm]
    X = onp.array(X)

    dataInfo = {"type":fileNm, "dataset":nm}

    print("set: ", datasetTab["fileNms"][indx_set])
    print("dataset: ", nm)

    # cap data
    maxData = 1000
    if X.shape[0]>maxData:
        smpl = onp.random.randint(low=1, high=X.shape[0], size=maxData)
        X = X[smpl,:]
	
    X = norml_mat(X)

    #if ((str(nm)=="8") | (str(nm)=="107") | (str(nm)=="70")) & (fileNm == "TCEP-all"):
    #if ((str(nm)=="8") | (str(nm)=="107") | (str(nm)=="70") | (str(nm)=="87")) & (fileNm == "TCEP-all"):
    if ((str(nm)=="8") | (str(nm)=="107") | (str(nm)=="70") | (str(nm)=="87") | (str(nm)=="27") | (str(nm)=="47")) & (fileNm == "TCEP-all"):
    #if (fileNm == "TCEP-all"):
        print("jittering")
        X = jitter(X)
        #X = onp.apply_along_axis(jitterByDist, 0, X)

    X = np.array(norml_mat(X))

    return nm, X, pars, dataInfo  # load shit

def main(args):

    job = int(args.job) + int(args.offset)
    # load dataset from job array id
    print("load")
    nm, data, pars, dataInfo = load_dataset(dataset_num=job, server=args.server)
    print("nm: ", nm)
    print("shape data", data.shape)
    print("pars: ", pars)


    N = data.shape[0]
    maxMonitor = 1000
    parts = int(onp.ceil(N/maxMonitor))
    print("parts: ", parts)
    


    # do stuffs (Latent Noise-KRR over the data)
    print("getLatenZs etc")
    beta = np.array([pars["beta"]])
    neta = np.array([pars["neta"]])
    eta = np.array([pars["eta"]])
    lam = np.array([pars["lambda"]])
    sig = np.array([pars["sig"]])
    nu = np.array([pars["nu"]])
    lu = np.array([pars["lu"]])
    lr = float(pars["lr"])
    optType = str(pars["optType"])
    epchs = int(pars["epchs"])
    bs = int(pars["bs"])
    reps = int(pars["reps"])

    
    #batch_size2 = int(onp.floor(onp.min([onp.max([30, bs/1000*N]), 300])))
    batch_size2 = int(onp.floor(onp.min([onp.max([90, bs/1000*N]), 300])))
    epochs2 = int(onp.max([onp.ceil((50*N)/batch_size2), 500]))
    print("epochs2: ", epochs2)
    print("batch_size2: ", batch_size2)


    # save shit (to json)
    print("save")

    if args.server == "erc":
        reposResults = "/home/emiliano/latentnoise_krr/results/"

    if args.server == "myLap":
        reposResults = "/home/emiliano/ISP/proyectos/latentNoise_krr/results/"


    #save_shit(results, name=f"{args.save}_results_{args.job}.json")
    fileRes = reposResults+"latent_noise"+str(job)+".pkl"
    #with open(fileRes, 'w') as outfile:
    #    json.dump(results, outfile)
    
    pars = {"lambda": lam,
            "sig":sig,
            "eta": eta,
            "beta": beta,
            "neta": neta,
            "nu": nu,
            "lu": lu,
            "lr": lr,
            "ot": optType,
            "epchs": epochs2,
            "bs":batch_size2,
            "reps":reps}


    if os.path.isfile(fileRes):
        print("File exist")
        results = pickle.load( open( fileRes, "rb" ) )
        results["pars"] = pars
        results["dataInfo"] = dataInfo
        print(results)
    	# sample usage
        save_object(results, fileRes)
    else:
        print("File not exist")
        results = main_experiment(data, beta, neta, eta, lam,sig, nu, lu, lr, nm, optType, epochs2, batch_size2, reps, job)
        results["pars"] = pars
        results["dataInfo"] = dataInfo
        #print(results)
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

