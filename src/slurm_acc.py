import sys
print("python version:", sys.version_info)
from latentNoise_funcs_gen import *
from ANLSMN_genfuncs import *
from processResults import *


import jax.numpy as np
import numpy as onp
import pandas as pd
#import pickle5 as pickle5
import pickle
import time
import json
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import bisect
import itertools
from scipy.spatial import distance
from itertools import chain, combinations
import os

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

#from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.tree import DecisionTreeClassifier
#from fairlearn.metrics import MetricFrame, selection_rate, count
import sklearn.metrics as skm

from sklearn.utils import check_random_state

# allows for arguments
import argparse


def main(args):

    job = int(args.job) + int(args.offset)

    # save shit (to json)
    print("save")

    if args.server == "erc":
        reposPairUp = "/home/emiliano/latentnoise_krr/resultsPost/dfsPairUp/"
        reposAcc = "/home/emiliano/latentnoise_krr/resultsPost/accs/"
        reposDecision = "/home/emiliano/latentnoise_krr/resultsPost/decisions/"
        repos = "/home/emiliano/latentnoise_krr/resultsPost/"

    if args.server == "myLap":
        reposPairUp = "/home/emiliano/ISP/proyectos/latentNoise_krr/results/post/"
        reposAcc = "/home/emiliano/ISP/proyectos/latentNoise_krr/results/post/"
        repos = "/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/results/"


    # load df
    # read in experiments
    print("current working directory: ", os.getcwd())
    print("files in repos:", os.listdir(repos))
    filename = "df_bnch_v1_optType.pkl"
    df_bnch = pickle.load(open(repos + filename, "rb"))
    #filename = "df_v3-5_optType.pkl"
    #df = pickle.load(open(repos + filename, "rb"))
    # get benchmark additive results
    pars = {"lambda": [0.01, 0.1, 1]}
    df_long_bnch = getLongFormat(df_bnch, pars)
    _ , res_bnch, res_bnch2 = getResBnch(df_long_bnch)

    # original pipeline
    pairUpNm = ["byParm", "rand", "intsc", "NN"]
    weightNm = ["uniform", "lowestHsic", "effFront", "modSimp_logis", "modSimp_RF"]

    initStrats = [["freeZ-iniMani", "freeZ", "freeZ-iniR"], ["freeZ-iniMani", "freeZ"], ["freeZ"], ["freeZ-iniMani"]]

    parsPairUp = [{"m": [1]}, {"m": [1000, 10000]},
                  {"varsss": [["hsic", "hsicc", "errs"]], "sig": [1, 100.0, 1000.0], "m": [100, 1000]},
                  {"varsss": [["hsic", "hsicc", "errs"]], "numPts": [1, 10, 20]}]

    var_smpl_nm = "smpld"
    parsWeight = [{}, {"var": ["hsic"], "sig": [0.1, 1, 10]},
                  {"varsss": [["hsic", "hsicc"]], "sig": [0.1, 1, 10]},
                  {"var_smpl_nm": [var_smpl_nm]},
                  {"var_smpl_nm": [var_smpl_nm]}]

    # hail mary version

    pairUpNm = ["NN"]
    weightNm = ["lowestHsic"]

    initStrats = [["freeZ"]]

    parsPairUp = [{"varsss": [["hsicx", "hsicc", "errs"]], "numPts": [1,2,5,10,15,20]}]

    var_smpl_nm = "smpld"
    parsWeight = [{"var": ["hsicx"], "sig": [1, 5, 10]}]

    # hail mary version 2

    pairUpNm = ["byParm", "rand", "intsc", "NN"]
    weightNm = ["uniform", "lowestHsic", "effFront", "modSimp_logis", "modSimp_RF"]

    initStrats = [["freeZ"]]

    parsPairUp = [{"m": [1]}, {"m": [1000, 10000]},
                  {"varsss": [["hsicx", "hsicc", "errs"]], "sig": [1, 100.0, 1000.0], "m": [100, 1000]},
                  {"varsss": [["hsicx", "hsicc", "errs"]], "numPts": [1, 2, 5, 10, 15, 20]}]

    var_smpl_nm = "smpld"
    parsWeight = [{}, {"var": ["hsicx"], "sig": [1, 5, 10]},
                  {"varsss": [["hsicx", "hsicc"]], "sig": [1, 5, 10]},
                  {"var_smpl_nm": [var_smpl_nm]},
                  {"var_smpl_nm": [var_smpl_nm]}]

    # consistency experiment

    pairUpNm = ["NN"]
    weightNm = ["lowestHsic"]

    initStrats = [["freeZ"]]

    parsPairUp = [{"varsss": [["hsicx", "hsicc", "errs"]], "numPts": [20]}]

    var_smpl_nm = "smpld"
    parsWeight = [{"var": ["hsicx"], "sig": [5]}]


    initStrat_pairUp_df, initStrat_pairUp_weight_df = getPipelineDF(initStrats, parsPairUp, parsWeight, pairUpNm=pairUpNm, weightNm=weightNm)

    print("initStrat_pairUp_df.shape", initStrat_pairUp_df.shape)
    print("initStrat_pairUp_weight_df.shape", initStrat_pairUp_weight_df.shape)
    i = job-1

    weightModMatFunct = initStrat_pairUp_weight_df.iloc[i]["weightModMatFunct"]
    weightModMatPars = initStrat_pairUp_weight_df.iloc[i]["weightModMatPars"]
    weightModFunct = initStrat_pairUp_weight_df.iloc[i]["weightModFunct"]
    weightModPars = initStrat_pairUp_weight_df.iloc[i]["weightModPars"]
    weightVar = initStrat_pairUp_weight_df.iloc[i]["weightVar"]
    weightFunct = initStrat_pairUp_weight_df.iloc[i]["weightFunct"]
    weightParsFunct = initStrat_pairUp_weight_df.iloc[i]["weightParsFunct"]
    weightParsTup = initStrat_pairUp_weight_df.iloc[i]["weightParsTup"]
    parsWeight = initStrat_pairUp_weight_df.iloc[i]["parsWeight"]

    initStrat = initStrat_pairUp_weight_df.iloc[i]["initStrat"]
    pairUpNm = initStrat_pairUp_weight_df.iloc[i]["pairUpNm"]
    parsPairUp = initStrat_pairUp_weight_df.iloc[i]["parsPairUp"]
    weightNm = initStrat_pairUp_weight_df.iloc[i]["weightNm"]


    df_id = initStrat_pairUp_weight_df.iloc[i]["df_id"]
    tabAcc_id = initStrat_pairUp_weight_df.iloc[i]["acc_id"]
    jobPrev,  = onp.where(initStrat_pairUp_df["df_id"]==df_id)
    jobPrev = int(jobPrev) + 1

    print("initStrat: ", initStrat)
    print("pairUpNm: ", pairUpNm)
    print("parsPairUp: ", parsPairUp)
    print("weightVar: ", weightVar)
    print("parsWeight: ", parsWeight)
    print("weightParsTup:", weightParsTup)

    filename = str(jobPrev)+"_"+"df_pairUp_" + df_id 
    cons_v = "_cons_800"
    filename = filename + cons_v
    filename = filename + ".pkl"
    df2 = pickle.load(open(reposPairUp + filename, "rb"))

    print("dfPairUp.shape:", df2.shape)

    filenameAcc = str(job)+"_"+"tabAcc_" + tabAcc_id + ".pkl"
    filenameDecision = str(job) + "_" + "decision_" + tabAcc_id + ".pkl"
    fileResAcc = reposAcc + filenameAcc
    fileResDecision = reposDecision + filenameDecision
    if os.path.isfile(fileResAcc):
        print("File exist")
    else:
        print("get tabAcc")
        print(df2.columns)
        res_long, tabAcc = getAcc(df2, res_bnch, res_bnch2, initStrat, pairUpNm, parsPairUp, weightNm, weightModMatFunct,
                        weightModMatPars, weightModFunct, weightModPars, weightFunct, weightParsFunct, weightParsTup,
                        parsWeight)
        print("dfPairUp.shape:", df2.shape)
        print("tabAcc.shape: ", tabAcc.shape)
        resAcc = tabAcc.to_dict()
        resDecision = res_long.to_dict()
        print("length dict values: ", [len(resAcc[k]) for k in resAcc.keys()])
        # save
        with open(fileResAcc, 'wb') as output:
            pickle.dump(resAcc, output, pickle.HIGHEST_PROTOCOL)
        with open(fileResDecision, 'wb') as output:
            pickle.dump(resDecision, output, pickle.HIGHEST_PROTOCOL)


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

