import sys
print(sys.path)
print("python version:" sys.version_info[0])
sys.path.insert(1,'/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/src/ITE-1.1_code')
sys.path.insert(1,'/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/src/')
from latentNoise_funcs_gen import *
from ANLSMN_genfuncs import *

from processResults import *


import jax.numpy as np
import numpy as onp
import pandas as pd
import pickle5 as pickle5
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

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

#from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.tree import DecisionTreeClassifier
#from fairlearn.metrics import MetricFrame, selection_rate, count
import sklearn.metrics as skm

from sklearn.utils import check_random_state



if __name__ == "__main__":
    # load dataset
    # dataset = ...

    print("enters acc")

    repos = "/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/results/"

    # load df
    # read in experiments

    filename = "df_bnch_v1_optType.pkl"
    df_bnch = pickle5.load(open(repos + filename, "rb"))
    pars = {"lambda": [0.01, 0.1, 1]}
    df_long_bnch = getLongFormat(df_bnch, pars)
    _, res_bnch, res_bnch2 = getResBnch(df_long_bnch)



    initStrats = [["freeZ-iniMani", "freeZ", "freeZ-iniR"], ["freeZ-iniMani", "freeZ"], ["freeZ"], ["freeZ-iniMani"]]

    parsPairUp = [{"m": [1]}, {"m": [1000, 10000]},
                  {"varsss": [["hsic", "hsicc", "errs"]], "sig": [1, 100.0, 1000.0], "m": [100, 1000]},
                  {"varsss": [["hsic", "hsicc", "errs"]], "numPts": [1, 10, 100]}]

    var_smpl_nm = "smpld"
    parsWeight = [{}, {"var": ["hsic"], "sig": [0.1, 1, 10]},
                  {"varsss": [["hsic", "hsicc"]], "sig": [0.1, 1, 10]},
                  {"var_smpl_nm": var_smpl_nm},
                  {"var_smpl_nm": var_smpl_nm}]

    initStrat_pairUp_df, initStrat_pairUp_weight_df = getPipelineDF(initStrats, parsPairUp, parsWeight)

    i = 51
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

    repos = "/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/results/post/"
    filename = "df_pairUp_" + df_id + ".pkl"
    df2 = pickle5.load(open(repos + filename, "rb"))

    res_long, tabAcc = getAcc(df2, res_bnch, res_bnch2, initStrat, pairUpNm, parsPairUp, weightNm, weightModMatFunct,
                    weightModMatPars, weightModFunct, weightModPars, weightFunct, weightParsFunct, weightParsTup,
                    parsWeight)

    # save
    repos = "/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/results/post/"
    filename1 = "tabAcc_" + tabAcc_id + ".pkl"
    filename2 = "res_long_" + tabAcc_id + ".pkl"
    with open(repos + filename1, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(tabAcc, output, pickle.HIGHEST_PROTOCOL)
    with open(repos + filename2, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(res_long, output, pickle.HIGHEST_PROTOCOL)

    print("finished acc")
    # save to somewhere
    # save_shit
