import sys
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
        reposResults = "/home/emiliano/latentnoise_krr/resultsPost/dfsPairUp/"
        repos = "/home/emiliano/latentnoise_krr/resultsPost/"

    if args.server == "myLap":
        reposResults = "/home/emiliano/ISP/proyectos/latentNoise_krr/results/post/"
        repos = "/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/results/"


    # load df
    # read in experiments
    print("current working directory: ", os.getcwd())
    print("files in repos:", os.listdir(repos))
    filename = "df_bnch_v1_optType.pkl"
    df_bnch = pickle.load(open(repos + filename, "rb"))
    #filename = "df_v3-5_optType.pkl"
    #filename = "df_v3-5_optType_vhsicx.pkl"
    #filename = "df_v10_UAI.pkl"
    filename = "df_v0_sens_800_UAI.pkl"
    df = pickle.load(open(repos + filename, "rb"))
    print("cols1: ", df.columns)
    # get benchmark additive results
    pars = {"lambda": [0.01, 0.1, 1]}
    df_long_bnch = getLongFormat(df_bnch, pars)
    #_, res_bnch, res_bnch2 = getResBnch(df_long_bnch)


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
                  {"var_smpl_nm": var_smpl_nm},
                  {"var_smpl_nm": var_smpl_nm}]

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

    print("pairup df shape",initStrat_pairUp_df.shape)
    print("pairup-weight df shape",initStrat_pairUp_weight_df.shape)

    i = job-1
    initStrat = initStrat_pairUp_df.iloc[i]["initStrat"]
    pairUpFunct = initStrat_pairUp_df.iloc[i]["pairUpFunct"]
    pairUpParsFunct = initStrat_pairUp_df.iloc[i]["pairUpParsFunct"]
    parsPairUp = initStrat_pairUp_df.iloc[i]["parsPairUp"]
    df_id = initStrat_pairUp_df.iloc[i]["df_id"]

    print("initStrat: ", initStrat)
    print("pairUpFunct: ", pairUpFunct)
    print("pairUpParsFunct: ", pairUpParsFunct)
    print("parsPairUp: ", parsPairUp)

    filename = str(job)+"_"+"df_pairUp_" + df_id + ".pkl"
    fileRes = reposResults + filename
    if os.path.isfile(fileRes):
        print("File exist")
    else:
        df2 = getPairUpDF(df, initStrat, pairUpFunct, pairUpParsFunct, parsPairUp)
        print("df2.shape: ", df2.shape)
        # save
        with open(fileRes, 'wb') as output:
            pickle.dump(df2, output, pickle.HIGHEST_PROTOCOL)


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

