from typing import Dict
import numpy as onp
import pandas as pd
import jax.numpy as np
import json
from ANLSMN_genfuncs import *
from latentNoise_funcs_gen import *
import bisect
import pickle
import os.path
import warnings

from experiment import main_experiment
# allows for arguments
import argparse

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def getRandResids(n):
    x, y, z = sample_AN(n)
    output = onp.array(y)
    input = onp.array(x[:,None])
    gpKern = RBF()+WhiteKernel()
    gpModel = GaussianProcessRegressor(kernel=gpKern)
    gpModel.fit(input, output)
    yhat = gpModel.predict(input)
    resids = y - yhat
    errs = mse(y, yhat)
    hsic = hsicRBF(resids, input)
    return errs, hsic


def main(args):

    job = int(args.job)
    # load dataset from job array id

    if args.server == "erc":
        reposResults = "/home/emiliano/latentnoise_krr/null_dists/hsicRX/parts/"

    if args.server == "myLap":
        reposResults = "/home/emiliano/ISP/proyectos/latentNoise_krr/null_dists/hsicRX/parts/"


    #save_shit(results, name=f"{args.save}_results_{args.job}.json")
    fileRes = reposResults+"null_hsicRX"+str(job)+".pkl"
    #with open(fileRes, 'w') as outfile:
    #    json.dump(results, outfile)


    if os.path.isfile(fileRes):
        print("File exist")
    else:
        print("File not exist")


        warnings.filterwarnings('ignore')
        N = 100
        n = 1000
        print("starting hsicRX calc")
        start = time.process_time()
        hsicMse = onp.array([getRandResids(n) for i in range(N)])
        print(time.process_time() - start)  #
        warnings.filterwarnings('default')
        print("finished")
    	# sample usage
        save_object(hsicMse, fileRes)
    
    return "bla"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments LNKRR.")

    # FOR THE JOB ARRRAY
    parser.add_argument("-j", "--job", default=0, type=int, help="job array for dataset")
    parser.add_argument("-v", "--server", default="myLap", type=str, help="server to run in")
    # run experiment
    
    args = parser.parse_args()
    print(args)
    print("run experiment")
    results = main(args)
    print("finished")

