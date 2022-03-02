import sys
print(sys.path)
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

    print("enters pairUp")

    repos = "/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/results/"

    # load df
    # read in experiments

    filename = "df_bnch_v1_optType.pkl"
    df_bnch = pickle5.load(open(repos + filename, "rb"))
    filename = "df_v3-5_optType.pkl"
    df = pickle5.load(open(repos + filename, "rb"))
    # get benchmark additive results
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

    i = 3
    initStrat = initStrat_pairUp_df.iloc[i]["initStrat"]
    pairUpFunct = initStrat_pairUp_df.iloc[i]["pairUpFunct"]
    pairUpParsFunct = initStrat_pairUp_df.iloc[i]["pairUpParsFunct"]
    parsPairUp = initStrat_pairUp_df.iloc[i]["parsPairUp"]
    df_id = initStrat_pairUp_df.iloc[i]["df_id"]

    df2 = getPairUpDF(df, initStrat, pairUpFunct, pairUpParsFunct, parsPairUp)

    # save
    repos = "/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/results/post/"
    filename = "df_pairUp_" + df_id + ".pkl"
    with open(repos + filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(df2, output, pickle.HIGHEST_PROTOCOL)

    print("finished pairUp")
    # save to somewhere
    # save_shit
