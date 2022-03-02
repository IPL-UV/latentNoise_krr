import pickle5 as pickle
import pandas as pd
import numpy as onp
import bisect
import json
import itertools
from itertools import chain, combinations
import scipy.stats as stats
from latentNoise_funcs_gen   import *
from scipy.spatial import distance

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc




#from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.tree import DecisionTreeClassifier
#from fairlearn.metrics import MetricFrame, selection_rate, count
import sklearn.metrics as skm

from sklearn.utils import check_random_state


print("enters processResults")
#########################################################
# REad in benchmark results

def getRawMsrsDir_bnch(res_path, dir):
    errs = res_path["mse"]
    hsics = res_path["hsic_rx"]
    hsicsq1 = res_path["hsic_rx_q1"]
    hsicsq2 = res_path["hsic_rx_q2"]
    hsicsq3 = res_path["hsic_rx_q3"]
    hsicsq4 = res_path["hsic_rx_q4"]
    hsicsq5 = res_path["hsic_rx_q5"]
    hsicsq6 = res_path["hsic_rx_q6"]
    hsicsq7 = res_path["hsic_rx_q7"]
    hsicsq8 = res_path["hsic_rx_q8"]
    hsicsq9 = res_path["hsic_rx_q9"]
    pvals = res_path["pval_rx"]
    hsics_null = res_path["hsic_rx_null"]
    hsicRob = res_path["hsicRob_rx"]
    hsicOptOrd = res_path["hsic_ord"]
    optOrd = res_path["hsic_ordOpt"]


    ents = res_path["h_x"] + res_path["h_r"]
    slopes = res_path["cost_slope"]
    slopes_krr = res_path["cost_slope_krr"]
    res = {"errs_" + dir: errs.tolist(), "hsic_" + dir: hsics.tolist(),
           "hsicq1_" + dir: hsicsq1.tolist(), "hsicq2_" + dir: hsicsq2.tolist(),
           "hsicq3_" + dir: hsicsq3.tolist(), "hsicq4_" + dir: hsicsq4.tolist(),
           "hsicq5_" + dir: hsicsq5.tolist(), "hsicq6_" + dir: hsicsq6.tolist(),
           "hsicq7_" + dir: hsicsq7.tolist(), "hsicq8_" + dir: hsicsq8.tolist(),
           "hsicq9_" + dir: hsicsq9.tolist(),
           "hsicNULL_" + dir: hsics_null.tolist(),
           "pvalAdpt_" + dir: pvals.tolist(),
           "hsicRob_"+ dir: hsicRob.tolist(), "hsicOptOrd_"+ dir: hsicOptOrd.tolist(), "optOrd_"+dir:optOrd.tolist(),
           "ent_" + dir: ents.tolist(),
           "slopes_" + dir: slopes.tolist(), "slopeskrr_" + dir: slopes_krr.tolist()}



    df = pd.DataFrame(res)

    return df

def getRawMsrs_bnch(res, job):
    df_xy = getRawMsrsDir_bnch(res["xy"],"xy")
    df_yx = getRawMsrsDir_bnch(res["yx"],"yx")
    df = pd.concat([df_xy, df_yx], axis=1)
    if ("cy" in res.keys()) & False:
        df_cy = getRawMsrsDir_bnch(res["cy"],"cy")
        df_cx = getRawMsrsDir_bnch(res["cx"],"cx")
        df = pd.concat([df, df_cy, df_cx], axis=1)
    df["job"] = job
    df["rep"] = [i for i in range(df_xy.shape[0])]
    return df

def readGetMsrs_bnch(folder, file, job):
    #print("file", file)
    res = pickle.load( open( folder+file+job+".pkl", "rb" ) )
    res = res["van"]
    msrs = getRawMsrs_bnch(res, job)
    return msrs

#########################################################
# REad in main results

def getRawMsrsDir_legacy(res_path, dir):
    n = res_path["ent_c"].shape[0] - 1
    # print("n: ",n)
    errs = res_path["errs"][n, :]
    hsics = res_path["hsic_r"][n, :]
    hsics_c = res_path["hsic"][n, :]
    ents = res_path["ent_c"][n, :] + res_path["ent_r"][n, :]
    slopes = res_path["cost_slope"][n, :]
    slopes_krr = res_path["cost_slope_krr"][n, :]
    res = {"errs_" + dir: errs.tolist(), "hsic_" + dir: hsics.tolist(), "hsicc_" + dir: hsics_c.tolist(),
           "ent_" + dir: ents.tolist(), "slopes_" + dir: slopes.tolist(), "slopeskrr_" + dir: slopes_krr.tolist()}
    if "hsic_zzhat" in res_path.keys():
        hsic_zzhat = res_path["hsic_zzhat"][n, :]
        res["hsiczz_" + dir] = hsic_zzhat.tolist()

    if "hsic_zhat" in res_path.keys():
        hsic_zhat = res_path["hsic_zhat"][n, 0]
        res["hsicz_" + dir] = hsic_zhat.tolist()

    if "MMDzn" in res_path.keys():
        mmd = res_path["MMDzn"][n, 0]
        res["mmd_" + dir] = hsic_zhat.tolist()

    df = pd.DataFrame(res)

    return df

def getRawMsrsDir(res_path, dir):
    n = res_path["ent_c"].shape[0] - 1
    # print("n: ",n)
    #print("shape: ",res_path["errs"][n, :].shape)
    #print("shape mean: ", onp.apply_along_axis(onp.mean, 1, res_path["errs"][n, :]).shape)

    useAllParts = False
    if useAllParts:
        errs = onp.ndarray.flatten(res_path["errs"][n, :][:,None])
        hsics = onp.ndarray.flatten(res_path["hsic_r"][n, :][:,None])
        hsicsx = onp.ndarray.flatten(res_path["hsic_rx"][n, :][:,None])
        hsicsz = onp.ndarray.flatten(res_path["hsic_rz"][n, :][:,None])
        hsics_c = onp.ndarray.flatten(res_path["hsic"][n, :][:,None])
        ents = onp.ndarray.flatten(res_path["ent_c"][n, :][:,None] + res_path["ent_r"][n, :][:,None])
        entsx = onp.ndarray.flatten(res_path["ent_x"][n, :][:,None] + res_path["ent_r"][n, :][:,None])
        #slopes = onp.ndarray.flatten(res_path["cost_slope"][n, :][:,None])
        #slopes_krr = onp.ndarray.flatten(res_path["cost_slope_krr"][n, :][:,None])

    else:
        errs = onp.ndarray.flatten(onp.apply_along_axis(onp.mean, 1, res_path["errs"][n, :])[:, None])
        hsics = onp.ndarray.flatten(onp.apply_along_axis(onp.mean, 1, res_path["hsic_r"][n, :])[:, None])
        hsicsx = onp.ndarray.flatten(onp.apply_along_axis(onp.mean, 1, res_path["hsic_rx"][n, :])[:, None])
        hsicsz = onp.ndarray.flatten(onp.apply_along_axis(onp.mean, 1, res_path["hsic_rz"][n, :])[:, None])
        hsics_c = onp.ndarray.flatten(onp.apply_along_axis(onp.mean, 1, res_path["hsic"][n, :])[:, None])
        ents = onp.ndarray.flatten(onp.apply_along_axis(onp.mean, 1, res_path["ent_c"][n, :])[:, None] + onp.apply_along_axis(onp.mean, 1, res_path["ent_r"][n, :])[:, None])
        entsx = onp.ndarray.flatten(onp.apply_along_axis(onp.mean, 1, res_path["ent_x"][n, :])[:, None] + onp.apply_along_axis(onp.mean, 1, res_path["ent_r"][n, :])[:, None])
        #slopes = onp.ndarray.flatten(onp.apply_along_axis(onp.mean, 1, res_path["cost_slope"][n, :])[:, None])
        #slopes_krr = onp.ndarray.flatten(onp.apply_along_axis(onp.mean, 1, res_path["cost_slope_krr"][n, :])[:, None])

    res = {"errs_" + dir: errs.tolist(), "hsic_" + dir: hsics.tolist(), "hsicc_" + dir: hsics_c.tolist(),
           "ent_" + dir: ents.tolist(),  "entx_"+ dir:entsx.tolist(),#"slopes_" + dir: slopes.tolist(), "slopeskrr_" + dir: slopes_krr.tolist(),
           "hsicx_" + dir: hsicsx.tolist(), "hsicz_" + dir: hsicsz.tolist()}

    if "hsic_zzhat" in res_path.keys():
        if useAllParts:
            hsic_zzhat = onp.ndarray.flatten(res_path["hsic_zzhat"][n, :][:,None])
        else:
            hsic_zzhat = onp.ndarray.flatten(onp.apply_along_axis(onp.mean, 1, res_path["hsic_zzhat"][n, :])[:, None])
        res["hsiczz_" + dir] = hsic_zzhat.tolist()

    if "ent_c2" in res_path.keys():
        if useAllParts:
            ents2 = onp.ndarray.flatten(res_path["ent_c2"][n, :][:, None] + res_path["ent_r2"][n, :][:, None])
        else:
            ents2 = onp.ndarray.flatten(
                onp.apply_along_axis(onp.mean, 1, res_path["ent_c2"][n, :])[:, None] + onp.apply_along_axis(onp.mean, 1,
                                                                                                           res_path[
                                                                                                               "ent_r2"][
                                                                                                           n, :])[:,
                                                                                      None])
        res["ent2_" + dir] = ents2.tolist()


    if "hsic_zhat" in res_path.keys():
        #print("shape zhat: ", res_path["hsic_zhat"][n, 0].shape)
        #print("shape zhat mean: ", onp.mean(res_path["hsic_zhat"][n, 0]).shape)
        if useAllParts:
            #print("hsic_zhat 1", res_path["hsic_zhat"])
            hsic_zhat = onp.ndarray.flatten(res_path["hsic_zhat"][n, 0][None,None])
            #print("hsic_zhat 2", hsic_zhat)
        else:
            hsic_zhat = onp.ndarray.flatten(onp.mean(res_path["hsic_zhat"][n, 0])[None, None])
        res["hsicz_" + dir] = hsic_zhat.tolist()

    if "MMDzn" in res_path.keys():
        if useAllParts:
            mmd = onp.ndarray.flatten(res_path["MMDzn"][n, :][:,None])
        else:
            mmd = onp.ndarray.flatten(onp.apply_along_axis(onp.mean, 1, res_path["MMDzn"][n, :])[:, None])
        res["mmd_" + dir] = hsic_zhat.tolist()

    def dealNan(x):
        if onp.any(onp.isnan(x)):
            res = onp.sqrt(-1)

        elif len(x)==1:
            res = x[0]
        else:
            res = x
        return res

    res = {k: dealNan(res[k]) for k in res.keys()}
    df = pd.DataFrame(res)

    return df

def getRawMsrs_legacy(res, job):
    df_xy = getRawMsrsDir_legacy(res["path_xy"],"xy")
    df_yx = getRawMsrsDir_legacy(res["path_yx"],"yx")
    df = pd.concat([df_xy, df_yx], axis=1)
    df["job"] = job
    df["rep"] = [i for i in range(df_xy.shape[0])]
    return df

def getRawMsrs(res, job):
    df_xy = getRawMsrsDir(res["path_xy"],"xy")
    df_yx = getRawMsrsDir(res["path_yx"],"yx")
    df = pd.concat([df_xy, df_yx], axis=1)
    df["job"] = job
    df["rep"] = [i for i in range(df_xy.shape[0])]
    return df

def readGetMsrs(folder, file, job):
    res = pickle.load( open( folder+file+job+".pkl", "rb" ) )

    #res = res["Z"]
    msrs = getRawMsrs(res["Z"], job)
    def convPars(x):
        if type(x) == onp.ndarray:
            res = onp.round(x, 6)[0]
        else:
            res = x
        return res

    if "pars" in list(res.keys()):
        parsJob = res["pars"]
        parsJob = {k: convPars(parsJob[k]) for k in parsJob.keys()}
        for k in parsJob.keys():
            msrs[k] = parsJob[k]
    if "dataInfo" in list(res.keys()):
        dataInfo = res["dataInfo"]
        for k in dataInfo.keys():
            msrs[k] = dataInfo[k]

    return msrs

#######################################################


# oextract "set" names from full dataset anmes
# TCEP
def funcType_tcep(nm):
    return "tcep"

# SIM
def funcType_SIM(nm):
    res  = nm.split(".")[0]
    return res

# ANLSMN, AddMultCmplx
def funcType3(nm):
    res  = nm.split(".")[0]
    return res

fileNms = ["TCEP-all","SIM-1000_withZ","ANLSMN_withZ"]
funcsType = [  [funcType_tcep], [funcType_SIM], [funcType3]]
funcsType = [el for sublist in funcsType for el in sublist]
func_dict = {n:f for (n,f) in zip(fileNms,funcsType)}

# read in benchmark datasets
def getData(file):
    with open(file) as json_file:
        data = json.load(json_file)
    return data["xs"]

def getDats(fileNames, fileNms, fileDict, func_dict):
    dats = [getData(f) for f in fileNames]
    datasets = [list(d.keys()) for d in dats]
    sets = [ [list(fileNms)[i]  for j in range(len(datasets[i]))] for i in range(len(datasets)) ]
    datasets = [ [datasets[i][j]  for j in range(len(datasets[i]))] for i in range(len(datasets)) ]
    types = [ [func_dict[sets[i][j]](datasets[i][j]) for j in range(len(datasets[i]))] for i in range(len(datasets))]
    dsUsed = [[types[i][j] in fileDict[list(fileDict.keys())[i]] for j in range(len(types[i]))] for i in range(len(types))]
    indxDsUsed = [list(onp.where(dsUsed[i]))[0] for i in range(len(types))]
    keysDsUsed = [[list(dats[i].keys())[j] for j in indxDsUsed[i].tolist()] for i in range(len(types))]
    datsNew = [{k:dats[i][k] for k in keysDsUsed[i]} for i in range(len(types))]
    return datsNew


def getDataSetTab(repos, par_dict, fileDict, func_dict):
    fileNmsAll = ["TCEP-all", "SIM-1000_withZ", "ANLSMN_withZ"]
    fileNms = list(fileDict.keys())
    #repos = "/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/data/"
    reposRelAll = (["TCEPs/" for i in range(2)] + ["ANLSMN/"])
    reposRelAll = {f: r for (f, r) in zip(fileNmsAll, reposRelAll)}
    reposRel = [reposRelAll[k] for k in fileNms]
    files = ["dag2-ME2-" + nm for nm in fileNms]
    fileNames = [repos + repRel + file + "_sims.json" for (repRel, file) in zip(reposRel, files)]
    datsNew = getDats(fileNames, fileNms, fileDict, func_dict)
    numBlocks = [len(datsNew[i].keys()) for i in range(len(datsNew))]
    datasetTab = {"fileNms": fileNms, "numJobs": numBlocks, "cumJobs_ini": [1] + list(onp.cumsum(numBlocks) + 1),
                  "cumJobs_fin": list(onp.cumsum(numBlocks))}

    # declare parmeters
    # v0_benchmark

    # combos = {"lambda": [l for l in lams ]}
    parsList = [par_dict[k] for k in par_dict.keys()]
    combos = [list(it) for it in itertools.product(*parsList)]
    combos = [[combos[j][i] for j in range(len(combos))] for i in range(len(combos[0]))]
    combos = {k: v for k, v in zip(par_dict.keys(), combos)}

    print("create datasetTab")
    datasetTab = {"fileNms": fileNms, "fileNames": fileNames, "numJobs": numBlocks,
                  "cumJobs_ini": [1] + list(onp.cumsum(numBlocks) + 1), "cumJobs_fin": list(onp.cumsum(numBlocks))}

    aux = {"fileNms": fileNms, "fileNames": fileNames, "numJobs": numBlocks}

    aux = pd.DataFrame.from_dict(aux)

    # datasetTab2 = {"fileNms": [f for f in fileNms for i in range(len(combos["lambda"]))],
    #               "lambda": [l for f in fileNms for l in combos["lambda"]]}

    datasetTab2 = {"fileNms": [f for f in fileNms for i in range(len(combos["lambda"]))]}

    for par in par_dict.keys():
        datasetTab2[par] = [p for f in fileNms for p in combos[par]]

    datasetTab2 = pd.DataFrame.from_dict(datasetTab2)

    datasetTab2 = datasetTab2.merge(aux, on="fileNms")
    cumJobs_ini = onp.cumsum(datasetTab2["numJobs"]) + 1
    datasetTab2["cumJobs_ini"] = [1] + list(cumJobs_ini[0:(len(cumJobs_ini) - 1)])
    cumJobs_fin = onp.cumsum(datasetTab2["numJobs"])
    datasetTab2["cumJobs_fin"] = list(cumJobs_fin)

    return datasetTab2, datsNew


def get_df_legacy(msrs, datasetTab, dats, pars, func_dict):
    df = pd.concat(msrs, axis=0)
    jobs = df["job"].tolist()
    jobs = [int(jobs[i]) for i in range(df.shape[0])]
    indxDT = [bisect.bisect_left(datasetTab["cumJobs_fin"], j) for j in jobs]
    df["set"] = datasetTab["fileNms"][indxDT][:, None][:, 0]
    for par in pars.keys():
        df[par] = datasetTab[par][indxDT][:, None][:, 0]

    matchMat = datasetTab["fileNames"][:, None] == datasetTab["fileNames"].unique()[:, None].T
    indxDat = onp.apply_along_axis(onp.where, 1, matchMat)[:, 0, 0]
    # dats = [getData(f) for f in datasetTab["fileNames"].unique()]

    # filter by fileDict

    nms = [list(dats[i].keys()) for i in indxDat]
    nums = [len(dats[indxDat[i]][nms[i][j]]) for i in range(len(nms)) for j in range(len(nms[i]))]
    nms = [nms[i][j] for i in range(len(nms)) for j in range(len(nms[i]))]
    reps = len(df.rep.unique())
    nums = [nums[i] for i in range(len(nums)) for j in range(reps)]
    nms = [nms[i] for i in range(len(nms)) for j in range(reps)]
    df["dataset"] = nms
    df["num"] = nums
    # add type column based on name of set and name of dataset
    df["type"] = [func_dict[st](ds) for (st, ds) in zip(df.set, df.dataset)]

    return df

def get_df(msrs, datasetTab, dats, pars, func_dict):
    print("get df2")
    df = pd.concat(msrs, axis=0)
    jobs = df["job"].tolist()
    jobs = [int(jobs[i]) for i in range(df.shape[0])]
    indxDT = [bisect.bisect_left(datasetTab["cumJobs_fin"], j) for j in jobs]
    df["set"] = datasetTab["fileNms"][indxDT][:, None][:, 0]
    if not onp.any([p in list(df.columns) for p in list(pars.keys())]):
        for par in pars.keys():
            df[par] = datasetTab[par][indxDT][:, None][:, 0]

    matchMat = datasetTab["fileNames"][:, None] == datasetTab["fileNames"].unique()[:, None].T
    indxDat = onp.apply_along_axis(onp.where, 1, matchMat)[:, 0, 0]
    # dats = [getData(f) for f in datasetTab["fileNames"].unique()]

    # filter by fileDict

    nms = [list(dats[i].keys()) for i in indxDat]
    nums = [len(dats[indxDat[i]][nms[i][j]]) for i in range(len(nms)) for j in range(len(nms[i]))]
    nms = [nms[i][j] for i in range(len(nms)) for j in range(len(nms[i]))]


    def maxRep(x):
        return onp.max(x.rep)

    reps = onp.array(df[list(pars.keys()) + ["rep","job"]].groupby(list(pars.keys())+["job"], sort=False).apply(maxRep)) + 1
    reps = reps.tolist()

    #numDatasets = int(len(nms) / len(reps))
    #reps = [reps[i] for i in range(len(reps)) for j in range(numDatasets)]
    #nms = [nms[i] for i in range(len(reps)) for j in range(reps[i])]
    #nums = [nums[i] for i in range(len(reps)) for j in range(reps[i])]

    nms = [nms[j] for j in range(len(reps)) for i in range(reps[j])]
    nums = [nums[j] for j in range(len(reps)) for i in range(reps[j])]
    reps = [reps[j] for j in range(len(reps)) for i in range(reps[j])]

    if not "dataset" in list(df.columns):
        df["dataset"] = nms
    df["num"] = nums
    # add type column based on name of set and name of dataset
    df["type"] = [func_dict[st](ds) for (st, ds) in zip(df.set, df.dataset)]


    return df

def get_df_fromFile(msrsFunc, reposData, version, file, num_data, fileDict,  pars, func_dict):
    num_pars = onp.prod([len(pars[k]) for k in pars.keys()])
    num_files = num_data*num_pars
    print("num_files", num_files)
    folder = "/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/results/"+version+"/"
    print("folder: ", folder)
    start = time.process_time()
    msrs = [msrsFunc(folder, file,str(i+1)) for i in range(num_files)]
    print(time.process_time() - start) #15 secs
    datasetTab, datsNew = getDataSetTab(reposData, pars, fileDict, func_dict)
    print(msrs[0].columns)
    if "lambda" in list(msrs[0].columns):
        df = get_df(msrs, datasetTab, datsNew, pars, func_dict)
    else :
        df = get_df_legacy(msrs, datasetTab, datsNew, pars, func_dict)

    return df

def getPvals_bnch(df_bnch):
    repos = "/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/null_dists/hsicRX/"
    #filename = repos + "hsicRX_nulldist.pkl"
    #hsic(r, x) 0.5-0.5 dist
    filename = repos + "hsicRX_nulldist" + "_" + str(0.5 * 100) + "_" + ".pkl"
    distHsicRX = pickle.load( open(filename, "rb"))
    pval_ln_xy= 1-stats.lognorm.cdf(df_bnch["hsic_xy"][:,None], s=distHsicRX["ln_pars"]["shape"], loc=distHsicRX["ln_pars"]["loc"], scale=distHsicRX["ln_pars"]["scale"])
    pval_ln_yx= 1-stats.lognorm.cdf(df_bnch["hsic_yx"][:,None], s=distHsicRX["ln_pars"]["shape"], loc=distHsicRX["ln_pars"]["loc"], scale=distHsicRX["ln_pars"]["scale"])
    df_bnch["hsicPval_xy"] = pval_ln_xy
    df_bnch["hsicPval_yx"] = pval_ln_yx

    # hsic(r, x) q-0.5 dist
    q_rs = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    q_rs = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    q_rs = [0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.45]
    filename = repos + "hsicRX_nulldist.pkl"
    cnt = 0
    for q_r in q_rs:
        cnt = cnt + 1
        filename = repos + "hsicRX_nulldist" + "_" + str(onp.round(q_r* 100)) + "_" + ".pkl"
        distHsicRX = pickle.load(open(filename, "rb"))
        pval_ln_xy = 1 - stats.lognorm.cdf(df_bnch["hsicq"+str(cnt)+"_xy"][:, None], s=distHsicRX["ln_pars"]["shape"],
                                           loc=distHsicRX["ln_pars"]["loc"], scale=distHsicRX["ln_pars"]["scale"])
        pval_ln_yx = 1 - stats.lognorm.cdf(df_bnch["hsicq"+str(cnt)+"_yx"][:, None], s=distHsicRX["ln_pars"]["shape"],
                                           loc=distHsicRX["ln_pars"]["loc"], scale=distHsicRX["ln_pars"]["scale"])
        df_bnch["hsicPvalq"+str(cnt)+"_xy"] = pval_ln_xy
        df_bnch["hsicPvalq"+str(cnt)+"_yx"] = pval_ln_yx




    df_bnch["hsicNrm_xy"] = df_bnch["hsic_xy"] / df_bnch["hsicNULL_xy"]
    df_bnch["hsicNrm_yx"] = df_bnch["hsic_yx"] / df_bnch["hsicNULL_yx"]

    df_bnch["hsicMax_xy"] = onp.amax(df_bnch[["hsicq1_xy","hsicq2_xy", "hsic_xy"]],1)
    df_bnch["hsicMax_yx"] = onp.amax(df_bnch[["hsicq1_yx","hsicq2_yx","hsic_yx"]],1)

    return df_bnch

def getPvals(df, df_long_full, res_bnch):
    repos = "/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/null_dists/mmd/"
    filename = repos + "mmd_nulldist.pkl"
    # MMD(z)
    distMMD = pickle.load( open(filename, "rb"))
    pval_ln_xy= 1-stats.lognorm.cdf(df["mmd_xy"][:,None], s=distMMD["ln_pars"]["shape"], loc=distMMD["ln_pars"]["loc"], scale=distMMD["ln_pars"]["scale"])
    pval_ln_yx= 1-stats.lognorm.cdf(df["mmd_yx"][:,None], s=distMMD["ln_pars"]["shape"], loc=distMMD["ln_pars"]["loc"], scale=distMMD["ln_pars"]["scale"])
    df["mmd_pval_xy"] = pval_ln_xy
    df["mmd_pval_yx"] = pval_ln_yx
    #pval_emp_xy = onp.apply_along_axis(onp.sum, 1, df2["mmd_xy"][:,None]<distMMD["dist"][:,None].T)/10000
    #pval_emp_yx = onp.apply_along_axis(onp.sum, 1, df2["mmd_yx"][:,None]<distMMD["dist"][:,None].T)/10000
    # HSIC(X, Z)
    repos = "/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/null_dists/hsicc/"
    filename = repos + "hsicc_nulldist.pkl"
    distHsicc = pickle.load( open(filename, "rb"))
    pval_ln_xy= 1-stats.lognorm.cdf(df["hsicc_xy"][:,None], s=distHsicc["ln_pars"]["shape"], loc=distHsicc["ln_pars"]["loc"], scale=distHsicc["ln_pars"]["scale"])
    pval_ln_yx= 1-stats.lognorm.cdf(df["hsicc_yx"][:,None], s=distHsicc["ln_pars"]["shape"], loc=distHsicc["ln_pars"]["loc"], scale=distHsicc["ln_pars"]["scale"])
    df["hsicc_pval_xy"] = pval_ln_xy
    df["hsicc_pval_yx"] = pval_ln_yx
    # HSIC(R, C)
    repos = "/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/null_dists/hsicRC/"
    filename = repos + "hsicRC_nulldist.pkl"
    distHsicRC = pickle.load( open(filename, "rb"))
    pval_ln_xy= 1-stats.lognorm.cdf(df["hsic_xy"][:,None], s=distHsicRC["ln_pars"]["shape"], loc=distHsicRC["ln_pars"]["loc"], scale=distHsicRC["ln_pars"]["scale"])
    pval_ln_yx= 1-stats.lognorm.cdf(df["hsic_yx"][:,None], s=distHsicRC["ln_pars"]["shape"], loc=distHsicRC["ln_pars"]["loc"], scale=distHsicRC["ln_pars"]["scale"])
    df["hsic_pval_xy"] = pval_ln_xy
    df["hsic_pval_yx"] = pval_ln_yx
    # MSE - AS A FUNCTION OF TYPE X SET DISTRIBUTION
    # get quantile of mse per type x dataset
    def aggReps(x):
        errs =  onp.array(x["value"][(x["var"]=="errs")])
        d = {}
        d['errRef1'] =  onp.quantile(errs, 0.1)
        res = pd.Series(d, index=['errRef1'])
        return res

    res = df_long_full.groupby(['type',"dataset"]).apply(aggReps)
    df = df.join(res, on=["type","dataset"],how="left")
    # MSE - from benchmarks
    d = {"errs":"errRef2"}
    df = df.join(res_bnch[["errs"]].rename(columns = d, inplace = False), on=["type","dataset"],how="left")
    return df


def getLongFormat(df, pars):
    id_vars = ["rep","set","type","dataset","num","job"]
    id_vars = id_vars + list(pars.keys())
    df_long_full = pd.melt(df, id_vars=id_vars)
    df_long_full["var"] = [var.split("_")[0] for var in df_long_full["variable"].tolist()]
    df_long_full["dir"] = [var.split("_")[len(var.split("_"))-1] for var in df_long_full["variable"].tolist()]

    df_long_full["smpl"] = [turnToIndex([mywhere(df_long_full[k][i] == pars[k]) for k in pars.keys()], pars) for i in range(df_long_full.shape[0])]
    return df_long_full

def getLongFormat2(df, id_vars):
    df_long_full = pd.melt(df, id_vars=id_vars)
    df_long_full["var"] = [var.split("_")[0] for var in df_long_full["variable"].tolist()]
    df_long_full["dir"] = [var.split("_")[len(var.split("_"))-1] for var in df_long_full["variable"].tolist()]
    return df_long_full


###################################################################3
# PAIR-UP FUNCS
#############################################################3
def pairup(x, varss, funct, getParsFunct, pars):
    # n = x.shape[0]
    # m = 100
    # func = smplRand
    # pars = (n,m)
    parsExtX = (x, *pars)
    parsExt = getParsFunct(*parsExtX)
    indxSmpl_xy, indxSmpl_yx = funct(*parsExt)

    vars_xy = [v + "_xy" for v in varss]
    vars_yx = [v + "_yx" for v in varss]
    x_xy = x[vars_xy]
    x_yx = x[vars_yx]


    res_xy = x_xy.iloc[indxSmpl_xy]
    res_yx = x_yx.iloc[indxSmpl_yx]
    res_xy = res_xy.reset_index()
    res_yx = res_yx.reset_index()

    res = pd.concat([res_xy, res_yx], axis=1)
    res.drop('index', inplace=True, axis=1)


    return res

# by parm

# RAND
def smplParm(n, m):
    indxSmpl_xy = onp.linspace(0,n-1,n, dtype=int)
    indxSmpl_yx = onp.linspace(0,n-1,n, dtype=int)
    return indxSmpl_xy, indxSmpl_yx

def getParsParm(x, m):
    n = x.shape[0]
    return n, m

# RAND
def smplRand(n, m):
    #indxSmpl_xy = onp.hstack([onp.linspace(0, 1, 1, dtype=int), onp.random.randint(low=0, high=n, size=m)])
    #indxSmpl_yx = onp.hstack([onp.linspace(0, 1, 1, dtype=int), onp.random.randint(low=0, high=n, size=m)])
    indxSmpl_xy = onp.hstack([onp.linspace(0,n-1,n, dtype=int),onp.random.randint(low=0, high=n, size=m)])
    indxSmpl_yx = onp.hstack([onp.linspace(0,n-1,n, dtype=int),onp.random.randint(low=0, high=n, size=m)])
    return indxSmpl_xy, indxSmpl_yx

def getParsRand(x, m):
    n = x.shape[0]
    return n,m


# INTERSECTION

def smplPt(d,m):
    #pr = 1/d#Ds[i,:]
    pr = d
    if onp.sum(pr)==0:
        pr= onp.ones(pr.shape[0])
    pr = pr/onp.sum(pr)
    smpl = onp.random.choice(a=len(d), size = m, p = pr)
    return smpl

def smplFromIntersection(X_xy, X_yx, sig, m):


    Ds = rbf_kernel_matrix({"gamma": sig}, X_xy, X_yx)

    indxSmpl = smplPt(onp.reshape(Ds, onp.prod(Ds.shape)), m)
    indxSmpl_xy = onp.int64(onp.floor(indxSmpl / Ds.shape[0]))
    indxSmpl_yx = onp.int64(onp.floor(indxSmpl % Ds.shape[0]))
    return indxSmpl_xy, indxSmpl_yx


def getParsIntersection(x, varsss, sig, m):
    vars_xy = [v + "_xy" for v in varsss]
    vars_yx = [v + "_yx" for v in varsss]
    X_xy = onp.log10(onp.array(x[vars_xy]))
    X_yx = onp.log10(onp.array(x[vars_yx]))

    return X_xy, X_yx, sig, m




# Nearest Neighbors

def nearestNeighbors(X_xy, X_yx, numPts):
    #X_xy = onp.vstack([onp.log10(v1_xy), onp.log10(v2_xy)]).T
    #X_yx = onp.vstack([onp.log10(v1_yx), onp.log10(v2_yx)]).T
    Ds = distance.cdist(X_xy, X_yx)
    indx_xy = onp.apply_along_axis(onp.argsort, 1, Ds)
    indx_yx = onp.apply_along_axis(onp.argsort, 1, Ds.T)
    indx_xy = indx_xy[:,0:(numPts)]
    indx_yx = indx_yx[:,0:(numPts)]
    indx_aux_xy = onp.arange(X_xy.shape[0])[:,None]*onp.ones(numPts, dtype=int)
    indx_aux_yx = onp.arange(X_yx.shape[0])[:,None]*onp.ones(numPts, dtype=int)
    indx_xy = onp.reshape(indx_xy, onp.prod(indx_xy.shape))
    indx_yx = onp.reshape(indx_yx, onp.prod(indx_yx.shape))
    indx_aux_xy = onp.reshape(indx_aux_xy, onp.prod(indx_aux_xy.shape))
    indx_aux_yx = onp.reshape(indx_aux_yx, onp.prod(indx_aux_yx.shape))
    indx_xy2 = onp.hstack([indx_aux_xy, indx_yx])
    indx_yx2 = onp.hstack([indx_xy, indx_aux_yx])
    return indx_xy2, indx_yx2

def getParsNN(x, varsss, numPts):
    vars_xy = [v + "_xy" for v in varsss]
    vars_yx = [v + "_yx" for v in varsss]
    X_xy = onp.log10(onp.array(x[vars_xy]))
    X_yx = onp.log10(onp.array(x[vars_yx]))

    return X_xy, X_yx, numPts


###################################################################3
# WEIGHTING FUNCS
#############################################################3

def getWeights(x, nm, funct, getParsFunct, pars):
    # n = x.shape[0]
    # m = 100
    # func = smplRand
    # pars = (n,m)

    parsExtX = (x, *pars)
    parsExt = getParsFunct(*parsExtX)
    ws = funct(*parsExt)
    res = x
    res[nm] = ws
    #if (res.shape[0] != 270) | (res.shape[1] != 37):
    #    print("type", onp.unique(x.type))
    #    print("dataset", onp.unique(x.dataset))
    #    print("res shape",res.shape)

    return res

def getModMatVan(df, m):
    return df

def getModVan(df, varsExpl):
    df["smpld"] = 1
    return None


def getWeightsWrapper(df, getModMatFunct, m, getModFunct, varsExpl, nm, funct, getParsFunct, parsTup, **pars):
    print("get model matrix")
    start = time.process_time()
    modMat = getModMatFunct(df, m)
    print("model matrix time ", (time.process_time() - start) / 60, " mins")

    print("get model")
    start = time.process_time()
    mod = getModFunct(modMat, varsExpl)
    print("model fitting time ", (time.process_time() - start) / 60, " mins")

    scope = locals()

    ks = list(pars.keys())

    parsTup2 = []
    for i in range(len(parsTup)):
        if parsTup[i] in ks:
            parsTup2.append(pars[parsTup[i]])
        else:
            parsTup2.append(eval(parsTup[i], scope))

    # df["smpld"] = modMat["smpld"]

    print("measure time for one dataset")
    ds = "3"
    st = "LS-s"
    x = modMat.loc[(modMat["dataset"] == st + "." + ds)]
    start = time.process_time()
    aux = getWeights(x, nm=nm, funct=funct, getParsFunct=getParsFunct, pars=tuple(parsTup2))
    timePerUnit = time.process_time() - start
    print(timePerUnit)

    print("estimate for all datasets")
    numUnits = df[["type", "dataset", "smpl"]].groupby(["type", "dataset"]).count().shape[0]
    estimatedTime = timePerUnit * numUnits
    print("estimated time for model application:", estimatedTime / 60, " mins")

    print("estimated time ", estimatedTime / 60, " mins")
    print("estimated time ", estimatedTime / 60 / 60, " hours")
    
    print("estimated time time ", estimatedTime / 60 / 60 /24 , " days")
    if (estimatedTime / 60 / 60 /24) > 3:
    	raise ValueError('Would take too long! More than 3 days.')

    print("apply model to each datset")
    start = time.process_time()
    df = modMat.groupby(['type', "dataset"]).apply(getWeights, nm=nm, funct=funct, getParsFunct=getParsFunct,
                                                   pars=tuple(parsTup2))
    finish = (time.process_time() - start)
    print("actual time ", finish / 60, " mins")
    

    if varsExpl is None:
        varsDel = []
    else:
        varsDel = varsExpl

    varsDel = varsDel + ["ws", "count"]
    indxS, = onp.where([(v.split("_")[0] == "type") & (len(v.split("_")) == 2) for v in list(df.columns)])
    colsS = [list(df.columns)[indxS[i]] for i in range(len(indxS))]
    varsDel = varsDel + colsS
    varsDel = list(set(varsDel).intersection(list(df.columns)))
    if len(varsDel) > 0:
        df.drop(varsDel, axis=1, inplace=True)

    return df

# UNIFORM weighting

def getParsUnifW(x):
    n = x.shape[0]
    return n,

def unifo(n):
    ws = onp.ones(n)
    ws = ws/onp.sum(ws)
    return ws

# UNIVARIATE weighting

def getParsHsicW(x, var, sig):
    var_xy = var + "_xy"
    var_yx = var + "_yx"
    X_xy = onp.log10(onp.array(x[var_xy]))
    X_yx = onp.log10(onp.array(x[var_yx]))

    return X_xy, X_yx, sig

def lowestHsic(X_xy, X_yx, sig):
    #minHsic = onp.array(onp.log10(0.00001))[None,None]
    minHsic = onp.min(onp.hstack([X_xy, X_yx]))[None,None]
    Ds_xy = rbf_kernel_matrix({"gamma":sig},onp.array(minHsic), X_xy)
    Ds_yx = rbf_kernel_matrix({"gamma":sig},onp.array(minHsic), X_yx)
    Ds = onp.vstack([Ds_xy, Ds_yx])
    ws = onp.apply_along_axis(onp.mean, 0, Ds)
    ws = ws/onp.sum(ws)
    return ws

# BIVARIATE weighting (efficient frontier)

def angle(dlt_err, dlt_hsic):
    res = np.nan
    if dlt_hsic == 0:
        if dlt_err > 0:
            res = 90
        if dlt_err < 0:
            res = 270
    if dlt_hsic != 0:
        res = onp.arctan2(dlt_err, dlt_hsic) / (2 * onp.pi) * 360
    if (dlt_err == 0) & (dlt_hsic == 0):
        res = 0

    return res

def angle2(err_a, err_b, hsic_a, hsic_b):
    dlt_err = err_a - err_b
    dlt_hsic = hsic_a - hsic_b
    res = angle(dlt_err, dlt_hsic)
    return res

# b) column by column chek which angles are in [0,90] or [-180,-90] ie are dominated
# c) chk which columns are all false (pts which are dominated)
def is_non_dom_all(col):
    res = onp.all(onp.logical_not(((col > -180) & (col < -90))))
    return res

def is_dom_all(col):
    res = onp.all(((col > -180) & (col < -90)))
    return res

def is_non_dom(col):
    res = onp.logical_not(((col > -180) & (col < -90)))
    return res

def getParsEffFrontW(x, varsss, sig):
    vars_xy = [v + "_xy" for v in varsss]
    vars_yx = [v + "_yx" for v in varsss]
    X_xy = onp.log10(onp.array(x[vars_xy]))
    X_yx = onp.log10(onp.array(x[vars_yx]))

    return X_xy, X_yx, sig

def effFront(X_xy, X_yx, sig):
    X = onp.vstack([X_xy, X_yx])
    angles = [[angle2(X[i, 1], X[j, 1], X[i, 0], X[j, 0]) for i in range(X.shape[0])] for j in range(X.shape[0])]
    angles = np.array(angles).T
    non_dom_pts = onp.apply_along_axis(is_non_dom_all, 1, angles.T)
    # plt.scatter(X[:,0], X[:,1])
    # plt.scatter(X[onp.where(non_dom_pts),0], X[onp.where(non_dom_pts),1])
    Ds_xy = rbf_kernel_matrix({"gamma": sig}, X[non_dom_pts, :], X_xy)
    Ds_xy = onp.apply_along_axis(onp.min, 0, Ds_xy)
    Ds_yx = rbf_kernel_matrix({"gamma": sig}, X[non_dom_pts, :], X_yx)
    Ds_yx = onp.apply_along_axis(onp.min, 0, Ds_yx)
    Ds = onp.vstack([Ds_xy, Ds_yx])
    ws = onp.apply_along_axis(onp.mean, 0, Ds)
    ws = ws / onp.sum(ws)
    return ws

# SIMPLE MODEL -logistic weighting
def getModMat(df, m):
    df2 = df.copy()

    df2["resp_ent"] = (df2["ent_yx"] > df2["ent_xy"]) * 1

    df2["dif_err"] = onp.log(onp.abs(df2["errs_yx"] - df2["errs_xy"]))
    df2["dif_hsic"] = onp.log(onp.abs(df2["hsic_yx"] - df2["hsic_xy"]))
    df2["dif_hsicc"] = onp.log(onp.abs(df2["hsicc_yx"] - df2["hsicc_xy"]))
    df2["dif_hsicz"] = onp.log(onp.abs(df2["hsicz_yx"] - df2["hsicz_xy"]))
    df2["dif_mmd"] = onp.log(onp.abs(df2["mmd_yx"] - df2["mmd_xy"]))

    df2["min_err"] = onp.log(onp.min(df2[["errs_yx", "errs_xy"]], axis=1))
    df2["min_hsic"] = onp.log(onp.min(df2[["hsic_yx", "hsic_xy"]], axis=1))
    df2["min_hsicc"] = onp.log(onp.min(df2[["hsicc_yx", "hsicc_xy"]], axis=1))
    df2["min_hsicz"] = onp.log(onp.min(df2[["hsicz_yx", "hsicz_xy"]], axis=1))
    df2["min_mmd"] = onp.log(onp.min(df2[["mmd_yx", "mmd_xy"]], axis=1))

    df2["max_err"] = onp.log(onp.max(df2[["errs_yx", "errs_xy"]], axis=1))
    df2["max_hsic"] = onp.log(onp.max(df2[["hsic_yx", "hsic_xy"]], axis=1))
    df2["max_hsicc"] = onp.log(onp.max(df2[["hsicc_yx", "hsicc_xy"]], axis=1))
    df2["max_hsicz"] = onp.log(onp.max(df2[["hsicz_yx", "hsicz_xy"]], axis=1))
    df2["max_mmd"] = onp.log(onp.max(df2[["mmd_yx", "mmd_xy"]], axis=1))

    n = df2.shape[0]

    ws_tab = df2[["type", "resp_ent", "dataset"]].groupby(["type", "resp_ent"]).count()
    ws_tab = ws_tab.rename(columns={"dataset": "count"})
    ws_tab = ws_tab.reset_index()
     	
    df2 = df2.merge(ws_tab, how="left", on=["type", "resp_ent"])
    df2["ws"] = 1 / df2["count"]

    # dont include tcep or LS-s in model
    df2.loc[df2["type"]=="tcep", "ws"] = 0
    df2.loc[df2["type"]=="LS-s", "ws"] = 0
    df2.loc[df2["type"] == "AN", "ws"] = 0
    df2.loc[df2["type"] == "AN-s", "ws"] = 0

    # make sure we dont sample rows with infinites
    varsExpl = ["dif_err", "dif_hsic", "dif_hsicc", "dif_mmd", "dif_hsicz", "min_err", "min_hsic", "min_hsicc",
                "min_mmd", "min_hsicz", "max_err", "max_hsic", "max_hsicc", "max_mmd", "max_hsicz"]
    infRows = onp.apply_along_axis(onp.sum, 1, onp.isinf(df2[varsExpl]))
    df2.loc[infRows != 0, "ws"] = 0
    print("number of rows with infinite values delteted: ", onp.sum(infRows))

    df2["ws"] = df2["ws"] / onp.sum(df2["ws"])

    onp.random.seed(10)
    m2 = int(onp.ceil(0.1*n))#int(m)
    print("n:",n, " m:",m2)
    smpl = onp.random.choice(a=n, size=m2, replace=False, p=df2["ws"])

    # smpl = onp.random.randint(low=0, high=n, size=m)
    universe = onp.linspace(0, n - 1, n - 1, dtype=int)
    # print(universe)
    smpl_pred = onp.setdiff1d(universe, smpl)
    df2["smpld"] = 0
    df2.loc[smpl, "smpld"] = 1
    oneHotType = pd.get_dummies(df2.type, prefix='type')
    df2 = pd.concat([df2, oneHotType], axis=1)

    return df2

def getModSimp_logis(df, varsExpl, calcType=False):
    print("get Mod Mat")
    X = df[varsExpl]
    # set -inf to min and inf to max columns wise
    minsX = (onp.ones([X.shape[0], 1]) * onp.min(X * onp.logical_not(onp.isinf(X)))[:, None].T)
    X = onp.amax(onp.stack([X, minsX], axis=-1), axis=2)
    X = pd.DataFrame(X, columns=varsExpl)
    maxsX = (onp.ones([X.shape[0], 1]) * onp.max(X * onp.logical_not(onp.isinf(X)))[:, None].T)
    X = onp.amin(onp.stack([X, maxsX], axis=-1), axis=2)
    X = pd.DataFrame(X, columns=varsExpl)
    poly_reg = PolynomialFeatures(degree=4)
    X = poly_reg.fit_transform(X)

    y = df["resp_ent"].to_numpy()

    smpl = onp.where(df["smpld"])
    smpl_pred = onp.where(df["smpld"] == 0)

    print("fit Model")
    mod = LogisticRegression(random_state=0, penalty="l2", solver="lbfgs", class_weight="balanced")

    mod.fit(X[smpl], y[smpl])
    print("on all", balanced_accuracy_score(y, mod.predict(X)))
    print("on pred", balanced_accuracy_score(y[smpl_pred], mod.predict(X[smpl_pred])))

    df2 = df.copy()
    df2["probRight"] = mod.predict_proba(X)[:, 1]
    df2["right"] = mod.predict(X)
    df2_pred = df2.iloc[smpl_pred]

    def accuracy(x):
        return balanced_accuracy_score(x.resp_ent, x.right)

    res_acc = df2_pred.groupby(["type"]).apply(accuracy)

    print("pred by type")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(res_acc)

    # can we use these features to identify the LS-s class? (or other types)... this wd be dangerous because then
    # our model is kinda just first identifying the class then building separate assumption conditions for each one
    # and we wont be able to extrapolate out of these classes. Quite likely.
    print("accuracy of simple logistic model using score of log model above in predicting type")

    if calcType:
    	x_score = mod.predict_log_proba(X)[:, 1][:, None]
    	# set infs to max or min non-inf
    	x_score = pd.DataFrame(x_score, columns=["x_score"])
    	minsX = (onp.ones([x_score.shape[0], 1]) * onp.min(x_score * onp.logical_not(onp.isinf(x_score)))[:, None].T)
    	x_score = onp.amax(onp.stack([x_score, minsX], axis=-1), axis=2)
    	x_score = pd.DataFrame(x_score, columns=["x_score"])
    	maxsX = (onp.ones([x_score.shape[0], 1]) * onp.max(x_score * onp.logical_not(onp.isinf(x_score)))[:, None].T)
    	x_score = onp.amin(onp.stack([x_score, maxsX], axis=-1), axis=2)

    	print("sum nans: ", onp.sum(onp.isnan(x_score)))
    	print("sum infinites: ", onp.sum(onp.isinf(x_score)))
    	mod2 = LogisticRegression(random_state=0, penalty="l2", solver="lbfgs", class_weight="balanced")

    	for ty in ["AN", "AN-s", "LS", "LS-s", "MN-U", "SIM", "SIMG", "SIMln", "SIMc", "tcep"]:
            y_type = (df["type"] == ty) * 1
            mod2.fit(x_score[smpl], y_type.iloc[smpl])
            pred = mod2.predict(x_score)
            print(ty, balanced_accuracy_score(y_type, pred))

    return mod

def getParsModSimp_logisW(x, varsss, var_smpl_nm, mod):
    X = x[varsss]
    minsX = (onp.ones([X.shape[0],1])*onp.min(X*onp.logical_not(onp.isinf(X)))[:,None].T)
    X = onp.amax(onp.stack( [X, minsX], axis=-1), axis=2)
    X = pd.DataFrame(X, columns=varsss)
    maxsX = (onp.ones([X.shape[0],1])*onp.max(X*onp.logical_not(onp.isinf(X)))[:,None].T)
    X = onp.amin(onp.stack( [X, maxsX], axis=-1), axis=2)
    X = pd.DataFrame(X, columns=varsss)
    poly_reg = PolynomialFeatures(degree=4)
    X2 = poly_reg.fit_transform(X)
    if not var_smpl_nm in list(x.columns):
        print("var_smpl_nm:", var_smpl_nm)
        print("x.columns:", list(x.columns))
    var_smpl = onp.array(x[var_smpl_nm])
    return X2, mod, var_smpl

def modSimp_logis(X, mod, var_smpl):
    ws = mod.predict_proba(X)[:, 1]
    # only rows not used for model are given a vote
    ws = ws * (var_smpl == 0)
    ws = ws * ((ws > 0.5) * 1)
    ws = ws / onp.sum(ws)
    return ws

# SIMPLE MODEL -Random Forest weighting

def getTreeProbs(leaf_indx_tr, y_tr, ws, leaf_indx_pr):
    res = pd.crosstab(leaf_indx_tr, y_tr, ws[y_tr], aggfunc=sum, normalize="index")
    res = res.reindex(index=onp.unique(leaf_indx_pr))  # leaves that do not have any element form training have a NAN
    res = res.reindex(columns=onp.arange(2))
    # probability
    # res = res.reset_index()
    return res


def predProbRF(forestProbs_tr, Leaves_pred):
    # here we use the Leaves index Leaves_pred[:,i] to find the probability of the leaf

    def debugThis(forestProbs_tr, Leaves_pred, i):
        res = forestProbs_tr[i]
        res = res.loc[Leaves_pred[:, i], 1]
        res = onp.array(res)

        return res

    probPred_forest = [debugThis(forestProbs_tr, Leaves_pred, i) for i in range(Leaves_pred.shape[1])]
    probPred_forest = onp.array(probPred_forest).T

    probPred = onp.apply_along_axis(onp.nanmean, 1, probPred_forest)
    return probPred


def applyRF(clf, Xtr, Xpr, ytr, boots):
    Leaves_tr = clf.apply(Xtr)
    Leaves_pr = clf.apply(Xpr)
    n_classes = 2
    n_samples = Xtr.shape[0]
    ws = n_samples / (n_classes * onp.bincount(ytr))

    def DebugGetTreeProbs(Leaves_tr, boots, ytr, ws, Leaves_pr, i):
        return getTreeProbs(Leaves_tr[boots[:, i], i], ytr.to_numpy()[boots[:, i]], ws, Leaves_pr[:, i])

    forestProbs_tr = [DebugGetTreeProbs(Leaves_tr, boots, ytr, ws, Leaves_pr, i) for i in range(Leaves_tr.shape[1])]

    probPred = predProbRF(forestProbs_tr, Leaves_pr)
    return probPred

def getModSimp_RF(df, varsExpl, calcTypeRF=False):
    X = df[varsExpl]
    y = df["resp_ent"]
    # set -inf to min and inf to max columns wise
    minsX = (onp.ones([X.shape[0], 1]) * onp.min(X * onp.logical_not(onp.isinf(X)))[:, None].T)
    X = onp.amax(onp.stack([X, minsX], axis=-1), axis=2)
    X = pd.DataFrame(X, columns=varsExpl)
    maxsX = (onp.ones([X.shape[0], 1]) * onp.max(X * onp.logical_not(onp.isinf(X)))[:, None].T)
    X = onp.amin(onp.stack([X, maxsX], axis=-1), axis=2)
    X = pd.DataFrame(X, columns=varsExpl)

    num_trees = 100
    max_depth = 50
    mod = RandomForestClassifier(max_depth=max_depth, random_state=0, n_estimators=num_trees, bootstrap=True,
                                 class_weight="balanced")
    smpl = onp.where(df["smpld"])
    smpl_pred = onp.where(df["smpld"] == 0)
    mod.fit(X.iloc[smpl], y.iloc[smpl])


    print("on all", balanced_accuracy_score(y, mod.predict(X)))
    print("on pred", balanced_accuracy_score(y.iloc[smpl_pred], mod.predict(X.iloc[smpl_pred])))

    df2 = df.copy()
    df2["probRight"] = mod.predict_proba(X)[:, 1]
    df2["right"] = mod.predict(X)
    df2_pred = df2.iloc[smpl_pred]

    def accuracy(x):
        return balanced_accuracy_score(x.resp_ent, x.right)

    res_acc = df2_pred.groupby(["type"]).apply(accuracy)

    print("pred by type")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(res_acc)

    if calcTypeRF:
        # can we use these features to identify the LS-s class? (or other types)... this wd be dangerous because then
        # our model is kinda just first identifying the class then building separate assumption conditions for each one
        # and we wont be able to extrapolate out of these classes. Quite likely.
        print("accuracy of simple logistic model using score of log model above in predicting type")

        n_samples_bootstrap = onp.sum(df["smpld"])
        random_instance = check_random_state(0)
        smpls = np.array([random_instance.randint(0, n_samples_bootstrap, n_samples_bootstrap) for i in range(num_trees)]).T

        for ty in ["AN", "AN-s", "LS", "LS-s", "MN-U", "SIM", "SIMG", "SIMln", "SIMc", "tcep"]:
            y_resp = (df["type"] == ty) * 1
            probRF = applyRF(mod, X.iloc[smpl], X, y_resp.iloc[smpl], smpls)
            print(ty, balanced_accuracy_score(y_resp, (probRF > 0.5) * 1))

    return mod

def getParsModSimp_RFW(x, varsss, var_smpl_nm, mod):
    X = x[varsss]
    minsX = (onp.ones([X.shape[0], 1]) * onp.min(X * onp.logical_not(onp.isinf(X)))[:, None].T)
    X = onp.amax(onp.stack([X, minsX], axis=-1), axis=2)
    X = pd.DataFrame(X, columns=varsss)
    maxsX = (onp.ones([X.shape[0], 1]) * onp.max(X * onp.logical_not(onp.isinf(X)))[:, None].T)
    X = onp.amin(onp.stack([X, maxsX], axis=-1), axis=2)
    X = pd.DataFrame(X, columns=varsss)
    var_smpl = onp.array(x[var_smpl_nm])
    return X, mod, var_smpl

def modSimp_RF(X, mod, var_smpl):
    ws = mod.predict_proba(X)[:, 1]
    # only rows not used for model are given a vote
    ws = ws * (var_smpl == 0)
    ws = ws * ((ws > 0.5) * 1)
    ws = ws / onp.sum(ws)
    return ws


# COMPLEX LOGISTIC FAIR MODEL weighting

def getLogProba_mitigator(mitigator, X):
    x_score = [mitigator.predictors_[i].predict_log_proba(X)[:, 1][:, None] for i in
               range(mitigator.predictors_.shape[0])]
    x_score = onp.array(x_score)[:, :, 0].T
    indx_row = onp.arange(x_score.shape[0])
    indx_col = onp.random.choice(a=x_score.shape[1], size=x_score.shape[0], replace=True, p=mitigator.weights_)
    x_score = x_score[indx_row, indx_col][:, None]
    return x_score

def getProba_mitigator(mitigator, X):
    x_score = [mitigator.predictors_[i].predict_proba(X)[:, 1][:, None] for i in range(mitigator.predictors_.shape[0])]
    x_score = onp.array(x_score)[:, :, 0].T
    indx_row = onp.arange(x_score.shape[0])
    indx_col = onp.random.choice(a=x_score.shape[1], size=x_score.shape[0], replace=True, p=mitigator.weights_)
    x_score = x_score[indx_row, indx_col][:, None]
    return x_score

def getModCmplx_logis(df, varsExpl):
        X = df[varsExpl]
        minsX = (onp.ones([X.shape[0], 1]) * onp.min(X * onp.logical_not(onp.isinf(X)))[:, None].T)
        X = onp.amax(onp.stack([X, minsX], axis=-1), axis=2)
        X = pd.DataFrame(X, columns=varsExpl)
        maxsX = (onp.ones([X.shape[0], 1]) * onp.max(X * onp.logical_not(onp.isinf(X)))[:, None].T)
        X = onp.amin(onp.stack([X, maxsX], axis=-1), axis=2)
        X = pd.DataFrame(X, columns=varsExpl)
        y = df["resp_ent"]
        poly_reg = PolynomialFeatures(degree=4)
        X = poly_reg.fit_transform(X)
        y = y.to_numpy()
        mod = LogisticRegression(random_state=0, penalty="l2", solver="lbfgs", class_weight="balanced")
        smpl = onp.where(df["smpld"])
        smpl_pred = onp.where(df["smpld"] == 0)
        mod.fit(X[smpl], y[smpl])
        y_pred = mod.predict(X[smpl_pred])
        # indxS,  = onp.where([v.split("_")[0]=="type" for v in list(df.columns)])
        # colsS = [list(df.columns)[indxS[i]] for i in range(len(indxS))]
        # S = df[colsS]
        S = df["type"]

        onp.random.seed(0)  # set seed for consistent results with ExponentiatedGradient
        constraint = DemographicParity(difference_bound=0.0)
        classifier = LogisticRegression(random_state=0, penalty="l2", solver="lbfgs", class_weight="balanced")
        mitigator = ExponentiatedGradient(classifier, constraint)
        mitigator.fit(X[smpl], y[smpl], sensitive_features=S.iloc[smpl])
        y_pred_mitigated = mitigator.predict(X[smpl_pred])

        print("UN-mitigated")
        print("************************************************************************")
        print("************************************************************************")

        sr = MetricFrame(metrics={'selection_rate': selection_rate,
                                  'accuracy': skm.accuracy_score,
                                  'balanced accuracy': skm.balanced_accuracy_score,
                                  'count': count}, y_true=y[smpl_pred], y_pred=y_pred, sensitive_features=S.iloc[smpl_pred])
        print("overall")
        print("**************")
        print(sr.overall)

        print("by group")
        print("**************")
        print(sr.by_group)

        print("differences")
        print("**************")
        print(sr.difference(method='between_groups'))

        df2 = df.copy()
        # df2["probRight"] = mod.predict_proba(X)[:,1]
        df2["right"] = mod.predict(X)
        df2_pred = df2.iloc[smpl_pred]

        def accuracy(x):
            return balanced_accuracy_score(x.resp_ent, x.right)

        res_acc = df2_pred.groupby(["type"]).apply(accuracy)

        print("pred by type")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(res_acc)

        # can we use these features to identify the LS-s class? (or other types)... this wd be dangerous because then
        # our model is kinda just first identifying the class then building separate assumption conditions for each one
        # and we wont be able to extrapolate out of these classes. Quite likely.
        print("accuracy of simple logistic model using score of log model above in predicting type")

        x_score = mod.predict_log_proba(X)[:, 1][:, None]
        mod2 = LogisticRegression(random_state=0, penalty="l2", solver="lbfgs", class_weight="balanced")

        for ty in ["AN", "AN-s", "LS", "LS-s", "MN-U", "SIM", "SIMG", "SIMln", "SIMc", "tcep"]:
            y_type = (df["type"] == ty) * 1
            mod2.fit(x_score[smpl], y_type.iloc[smpl])
            print(ty, balanced_accuracy_score(y_type, mod2.predict(x_score)))

        print("mitigated")
        print("************************************************************************")
        print("************************************************************************")

        sr_mitigated = MetricFrame(metrics={'selection_rate': selection_rate,
                                            'accuracy': skm.accuracy_score,
                                            'balanced accuracy': skm.balanced_accuracy_score,
                                            'count': count}, y_true=y[smpl_pred], y_pred=y_pred_mitigated,
                                   sensitive_features=S.iloc[smpl_pred])
        print("overall")
        print("**************")
        print(sr_mitigated.overall)

        print("by group")
        print("**************")
        print(sr_mitigated.by_group)

        print("differences")
        print("**************")
        print(sr_mitigated.difference(method='between_groups'))

        print("on all", balanced_accuracy_score(y, mod.predict(X)))
        print("on pred", balanced_accuracy_score(y[smpl_pred], mod.predict(X[smpl_pred])))

        df2 = df.copy()
        # df2["probRight"] = mitigator.predict_proba(X)[:,1]
        df2["right"] = mitigator.predict(X)
        df2_pred = df2.iloc[smpl_pred]

        res_acc = df2_pred.groupby(["type"]).apply(accuracy)

        print("pred by type")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(res_acc)

        # can we use these features to identify the LS-s class? (or other types)... this wd be dangerous because then
        # our model is kinda just first identifying the class then building separate assumption conditions for each one
        # and we wont be able to extrapolate out of these classes. Quite likely.
        print("accuracy of simple logistic model using score of log model above in predicting type")

        x_score = getLogProba_mitigator(mitigator, X)
        mod2 = LogisticRegression(random_state=0, penalty="l2", solver="lbfgs", class_weight="balanced")

        for ty in ["AN", "AN-s", "LS", "LS-s", "MN-U", "SIM", "SIMG", "SIMln", "SIMc", "tcep"]:
            y_type = (df["type"] == ty) * 1
            mod2.fit(x_score[smpl], y_type.iloc[smpl])
            print(ty, balanced_accuracy_score(y_type, mod2.predict(x_score)))

        return mitigator

def getParsModCmplx_logisW(x, varsss, var_smpl_nm, mod):
    X = onp.array(x[varsss])
    minsX = (onp.ones([X.shape[0], 1]) * onp.min(X * onp.logical_not(onp.isinf(X)))[:, None].T)
    X = onp.amax(onp.stack([X, minsX], axis=-1), axis=2)
    X = pd.DataFrame(X, columns=varsss)
    maxsX = (onp.ones([X.shape[0], 1]) * onp.max(X * onp.logical_not(onp.isinf(X)))[:, None].T)
    X = onp.amin(onp.stack([X, maxsX], axis=-1), axis=2)
    X = pd.DataFrame(X, columns=varsss)
    poly_reg = PolynomialFeatures(degree=4)
    X2 = poly_reg.fit_transform(X)
    var_smpl = onp.array(x[var_smpl_nm])
    return X2, mod, var_smpl

def modCmplx_logis(X, mod, var_smpl):
    ws = getProba_mitigator(mod, X)[:, 0]
    ws = ws * (var_smpl == 0)
    ws = ws * ((ws > 0.5) * 1)
    ws = ws / onp.sum(ws)
    return ws

# COMPLEX RANDOM FOREST FAIR MODEL weighting

# 1. write function that gets class probas for each leave in a tree using the training sample
# do the crosstab by index to get leave-level proportion


def getProba_mitigator_RF(mitigator, Xtr, Xpr, ytr, boots):
    x_score = [applyRF(mitigator.predictors_[i], Xtr, Xpr, ytr, boots) for i in range(mitigator.predictors_.shape[0])]
    x_score = onp.array(x_score)
    x_score = x_score.T
    indx_row = onp.arange(x_score.shape[0])
    indx_col = onp.random.choice(a=x_score.shape[1], size=x_score.shape[0], replace=True, p=mitigator.weights_)
    x_score = x_score[indx_row, indx_col][:, None]
    return x_score

def getModCmplx_RF(df, varsExpl, calcTypeRF=False):
    X = df[varsExpl]
    y = df["resp_ent"]
    num_trees = 100
    max_depth = 50
    mod = RandomForestClassifier(max_depth=max_depth, random_state=0, n_estimators=num_trees, bootstrap=True,
                                 class_weight="balanced")
    smpl = onp.where(df["smpld"])
    smpl_pred = onp.where(df["smpld"] == 0)
    mod.fit(X.iloc[smpl], y.iloc[smpl])
    y_pred = mod.predict(X.iloc[smpl_pred])
    # indxS,  = onp.where([v.split("_")[0]=="type" for v in list(df.columns)])
    # colsS = [list(df.columns)[indxS[i]] for i in range(len(indxS))]
    # S = df[colsS]
    S = df["type"]

    onp.random.seed(0)  # set seed for consistent results with ExponentiatedGradient
    constraint = DemographicParity(difference_bound=0.0)
    classifier = RandomForestClassifier(max_depth=max_depth, random_state=0, n_estimators=num_trees, bootstrap=True,
                                        class_weight="balanced")
    mitigator = ExponentiatedGradient(classifier, constraint)
    mitigator.fit(X.iloc[smpl], y.iloc[smpl], sensitive_features=S.iloc[smpl])
    y_pred_mitigated = mitigator.predict(X.iloc[smpl_pred])

    print("UN-mitigated")
    print("************************************************************************")
    print("************************************************************************")

    sr = MetricFrame(metrics={'selection_rate': selection_rate,
                              'accuracy': skm.accuracy_score,
                              'balanced accuracy': skm.balanced_accuracy_score,
                              'count': count}, y_true=y.iloc[smpl_pred], y_pred=y_pred,
                     sensitive_features=S.iloc[smpl_pred])
    print("overall")
    print("**************")
    print(sr.overall)

    print("by group")
    print("**************")
    print(sr.by_group)

    print("differences")
    print("**************")
    print(sr.difference(method='between_groups'))

    df2 = df.copy()
    # df2["probRight"] = mod.predict_proba(X)[:,1]
    df2["right"] = mod.predict(X)
    df2_pred = df2.iloc[smpl_pred]

    def accuracy(x):
        return balanced_accuracy_score(x.resp_ent, x.right)

    res_acc = df2_pred.groupby(["type"]).apply(accuracy)

    print("pred by type")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(res_acc)

    # can we use these features to identify the LS-s class? (or other types)... this wd be dangerous because then
    # our model is kinda just first identifying the class then building separate assumption conditions for each one
    # and we wont be able to extrapolate out of these classes. Quite likely.
    print("accuracy of simple logistic model using score of log model above in predicting type")

    # x_score = mod.predict_log_proba(X)[:,1][:,None]
    # mod2 = LogisticRegression(random_state=0, penalty="l2", solver="lbfgs", class_weight="balanced")

    # for ty in ["AN","AN-s","LS", "LS-s","MN-U","SIM","SIMG","SIMln","SIMc","tcep"]:
    #    y_type = (df["type"]==ty)*1
    #    mod2.fit(x_score[smpl], y_type.iloc[smpl])
    #    print(ty,balanced_accuracy_score(y_type, mod2.predict(x_score)))

    n_samples_bootstrap = onp.sum(df["smpld"])
    random_instance = check_random_state(0)
    smpls = np.array([random_instance.randint(0, n_samples_bootstrap, n_samples_bootstrap) for i in range(num_trees)]).T

    if calcTypeRF:
        for ty in ["AN", "AN-s", "LS", "LS-s", "MN-U", "SIM", "SIMG", "SIMln", "SIMc", "tcep"]:
            y_resp = (df["type"] == ty) * 1
            probRF = applyRF(mod, X.iloc[smpl], X, y_resp.iloc[smpl], smpls)
            print(ty, balanced_accuracy_score(y_resp, (probRF > 0.5) * 1))

    print("mitigated")
    print("************************************************************************")
    print("************************************************************************")

    sr_mitigated = MetricFrame(metrics={'selection_rate': selection_rate,
                                        'accuracy': skm.accuracy_score,
                                        'balanced accuracy': skm.balanced_accuracy_score,
                                        'count': count}, y_true=y.iloc[smpl_pred], y_pred=y_pred_mitigated,
                               sensitive_features=S.iloc[smpl_pred])
    print("overall")
    print("**************")
    print(sr_mitigated.overall)

    print("by group")
    print("**************")
    print(sr_mitigated.by_group)

    print("differences")
    print("**************")
    print(sr_mitigated.difference(method='between_groups'))

    print("on all", balanced_accuracy_score(y, mod.predict(X)))
    print("on pred", balanced_accuracy_score(y.iloc[smpl_pred], mod.predict(X.iloc[smpl_pred])))

    df2 = df.copy()
    # df2["probRight"] = mitigator.predict_proba(X)[:,1]
    df2["right"] = mitigator.predict(X)
    df2_pred = df2.iloc[smpl_pred]

    res_acc = df2_pred.groupby(["type"]).apply(accuracy)

    print("pred by type")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(res_acc)

    # can we use these features to identify the LS-s class? (or other types)... this wd be dangerous because then
    # our model is kinda just first identifying the class then building separate assumption conditions for each one
    # and we wont be able to extrapolate out of these classes. Quite likely.
    print("accuracy of simple logistic model using score of log model above in predicting type")

    if calcTypeRF:
        for ty in ["AN", "AN-s", "LS", "LS-s", "MN-U", "SIM", "SIMG", "SIMln", "SIMc", "tcep"]:
            y_resp = (df["type"] == ty) * 1
            x_score = getProba_mitigator_RF(mitigator, X.iloc[smpl], X, y_resp.iloc[smpl], smpls)
            print(ty, balanced_accuracy_score(y_resp, (x_score > 0.5) * 1))

    return mitigator

def getParsModCmplx_RFW(x, varsss, var_smpl_nm, mod):
    X = x[varsss]
    var_smpl = onp.array(x[var_smpl_nm])

    return X, mod, var_smpl  # , boots, Xtr, ytr

def modCmplx_RF(X, mod, var_smpl):
    ws = getProba_mitigator(mod, X)[:, 0]
    # ws = getProba_mitigator_RF(mod,X)[:,0]
    ws = ws * (var_smpl == 0)
    ws = ws * ((ws > 0.5) * 1)
    ws = ws / onp.sum(ws)
    return ws

###################################################################3
# VOTING USING WEIGHTS
#############################################################3

def vote(x, ws_nm):
    errs_xy = onp.array(x["value"][(x["var"] == "errs") & (x["dir"] == "xy")])
    errs_yx = onp.array(x["value"][(x["var"] == "errs") & (x["dir"] == "yx")])
    hsic_xy = onp.array(x["value"][(x["var"] == "hsic") & (x["dir"] == "xy")])
    hsic_yx = onp.array(x["value"][(x["var"] == "hsic") & (x["dir"] == "yx")])
    hsicx_xy = onp.array(x["value"][(x["var"] == "hsicx") & (x["dir"] == "xy")])
    hsicx_yx = onp.array(x["value"][(x["var"] == "hsicx") & (x["dir"] == "yx")])
    hsicc_xy = onp.array(x["value"][(x["var"] == "hsicc") & (x["dir"] == "xy")])
    hsicc_yx = onp.array(x["value"][(x["var"] == "hsicc") & (x["dir"] == "yx")])
    ent_xy = onp.array(x["value"][(x["var"] == "ent") & (x["dir"] == "xy")])
    ent_yx = onp.array(x["value"][(x["var"] == "ent") & (x["dir"] == "yx")])
    #ent2_xy = onp.array(x["value"][(x["var"] == "ent2") & (x["dir"] == "xy")])
    #ent2_yx = onp.array(x["value"][(x["var"] == "ent2") & (x["dir"] == "yx")])
    entx_xy = onp.array(x["value"][(x["var"] == "entx") & (x["dir"] == "xy")])
    entx_yx = onp.array(x["value"][(x["var"] == "entx") & (x["dir"] == "yx")])
    #slope_xy = onp.array(x["value"][(x["var"] == "slopes") & (x["dir"] == "xy")])
    #slope_yx = onp.array(x["value"][(x["var"] == "slopes") & (x["dir"] == "yx")])
    #slope_krr_xy = onp.array(x["value"][(x["var"] == "slopeskrr") & (x["dir"] == "xy")])
    #slope_krr_yx = onp.array(x["value"][(x["var"] == "slopeskrr") & (x["dir"] == "yx")])
    hsicz_xy = onp.array(x["value"][(x["var"] == "hsicz") & (x["dir"] == "xy")])
    hsicz_yx = onp.array(x["value"][(x["var"] == "hsicz") & (x["dir"] == "yx")])
    mmd_xy = onp.array(x["value"][(x["var"] == "mmd") & (x["dir"] == "xy")])
    mmd_yx = onp.array(x["value"][(x["var"] == "mmd") & (x["dir"] == "yx")])

    # ws = onp.array(x["value"][(x["dir"]==ws_nm)])
    ws = onp.array(x[ws_nm][(x["var"] == "errs") & (x["dir"] == "xy")])


    scr_err = (errs_yx < errs_xy) * -ws + (errs_yx > errs_xy) * ws
    scr_hsic = (hsic_yx < hsic_xy) * -ws + (hsic_yx > hsic_xy) * ws
    #scr_hsicx = (hsicx_yx < hsicx_xy) * -ws + (hsicx_yx > hsicx_xy) * ws
    scr_hsicc = (hsicc_yx < hsicc_xy) * -ws + (hsicc_yx > hsicc_xy) * ws
    scr_ent = (ent_yx < ent_xy) * -ws + (ent_yx > ent_xy) * ws
    #scr_ent2 = (ent2_yx < ent2_xy) * -ws + (ent2_yx > ent2_xy) * ws
    #scr_entx = (entx_yx < entx_xy) * -ws + (entx_yx > entx_xy) * ws
    #scr_slope = (slope_yx < slope_xy) * -ws + (slope_yx > slope_xy) * ws
    #scr_slope_krr = (slope_krr_yx < slope_krr_xy) * -ws + (slope_krr_yx > slope_krr_xy) * ws
    hiscz = (hsicz_yx < hsicz_xy) * ws + (hsicz_yx > hsicz_xy) * -ws
    scr_mmd = (mmd_yx < mmd_xy) * -ws + (mmd_yx > mmd_xy) * ws

    d = {}
    d['errs'] = onp.sum(scr_err)
    d['hsic'] = onp.sum(scr_hsic)
    #d['hsicx'] = onp.sum(scr_hsicx)
    d['hsicc'] = onp.sum(scr_hsicc)
    d['ent'] = onp.sum(scr_ent)
    #d['ent2'] = onp.sum(scr_ent2)
    #d['entx'] = onp.sum(scr_entx)
    #d['slope'] = onp.sum(scr_slope)
    #d['slope_krr'] = onp.sum(scr_slope_krr)
    d['hsicz'] = onp.sum(hiscz)
    d['mmd'] = onp.sum(scr_mmd)


    varsDict = ['errs', 'ent', 'entx', 'hsic', 'hsicx', "hsicc", "hsicz"]
    varsDict = ['errs', 'ent', 'entx', 'hsic', 'hsicx', "hsicc", "hsicz"]
    varsDict = ['errs', 'ent', 'hsic',  "hsicc", "hsicz","mmd"]
    res = pd.Series(d, index=varsDict) #'slope', "slope_krr",

    return res

def aggBnch(x):

    pvals_hsic = onp.array(x["value"][(x["var"] == "hsicPval")])

    pvals_hsicq1 = onp.array(x["value"][(x["var"] == "hsicPvalq1")])
    pvals_hsicq2 = onp.array(x["value"][(x["var"] == "hsicPvalq2")])
    pvals_hsicq3 = onp.array(x["value"][(x["var"] == "hsicPvalq3")])
    pvals_hsicq4 = onp.array(x["value"][(x["var"] == "hsicPvalq4")])
    pvals_hsicq5 = onp.array(x["value"][(x["var"] == "hsicPvalq5")])
    pvals_hsicq6 = onp.array(x["value"][(x["var"] == "hsicPvalq6")])
    pvals_hsicq7 = onp.array(x["value"][(x["var"] == "hsicPvalq7")])
    pvals_hsicq8 = onp.array(x["value"][(x["var"] == "hsicPvalq8")])
    pvals_hsicq9 = onp.array(x["value"][(x["var"] == "hsicPvalq9")])
    pvals_hsicqs = [pvals_hsicq1, pvals_hsicq2,pvals_hsicq3, pvals_hsicq4, pvals_hsicq5, pvals_hsicq6, pvals_hsicq7, pvals_hsicq8,pvals_hsicq9]

    pvals_adpt = onp.array(x["value"][(x["var"] == "pvalAdpt")])
    hsics = onp.array(x["value"][(x["var"] == "hsic")])

    hsicsq1 = onp.array(x["value"][(x["var"] == "hsicq1")])
    hsicsq2 = onp.array(x["value"][(x["var"] == "hsicq2")])
    hsicsq3 = onp.array(x["value"][(x["var"] == "hsicq3")])
    hsicsq4 = onp.array(x["value"][(x["var"] == "hsicq4")])
    hsicsq5 = onp.array(x["value"][(x["var"] == "hsicq5")])
    hsicsq6 = onp.array(x["value"][(x["var"] == "hsicq6")])
    hsicsq7 = onp.array(x["value"][(x["var"] == "hsicq7")])
    hsicsq8 = onp.array(x["value"][(x["var"] == "hsicq8")])
    hsicsq9 = onp.array(x["value"][(x["var"] == "hsicq9")])
    hsicsqs = [hsicsq1, hsicsq2, hsicsq3, hsicsq4,hsicsq5, hsicsq6, hsicsq7, hsicsq8, hsicsq9]

    hsicsMax = onp.array(x["value"][(x["var"] == "hsicMax")])
    hsicNrm = onp.array(x["value"][(x["var"] == "hsicNrm")])

    hsicRob = onp.array(x["value"][(x["var"] == "hsicRob")])
    hsicOptOrd = onp.array(x["value"][(x["var"] == "hsicOptOrd")])
    optOrd = onp.array(x["value"][(x["var"] == "optOrd")])
    lambdas = onp.array(x["lambda"][(x["var"] == "optOrd")])


    errs = onp.array(x["value"][(x["var"] == "errs")])


    d = {}
    d['minErrsBnch'] = onp.nanmin(errs)
    d['difErrsBnch'] = onp.nanmax(errs) - onp.nanmin(errs)
    d['maxPvalHsicBnch'] = onp.nanmax(pvals_hsic)
    d['difPvalHsicBnch'] = onp.nanmax(pvals_hsic) - onp.nanmin(pvals_hsic)

    q_rs = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    q_rs = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    q_rs = [0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.45]
    cnt = 0
    for q_r in q_rs:
        d['maxPvalHsicq'+str(cnt+1)+'Bnch'] = onp.nanmax(pvals_hsicqs[cnt])
        d['difPvalHsicq'+str(cnt+1)+'Bnch'] = onp.nanmax(pvals_hsicqs[cnt]) - onp.nanmin(pvals_hsicqs[cnt])
        cnt = cnt + 1


    d['maxPvalHsicAdptBnch'] = onp.nanmax(pvals_adpt)
    d['difPvalHsicAdptBnch'] = onp.nanmax(pvals_adpt) - onp.nanmin(pvals_adpt)
    d['minHsicBnch'] = onp.nanmin(hsics)
    d['difHsicBnch'] = onp.nanmax(hsics) - onp.nanmin(hsics)

    cnt = 0
    for q_r in q_rs:
        d['minHsicq'+str(cnt+1)+'Bnch'] = onp.nanmin(hsicsqs[cnt])
        d['difHsicq'+str(cnt+1)+'Bnch'] = onp.nanmax(hsicsqs[cnt]) - onp.nanmin(hsicsqs[cnt])
        cnt = cnt + 1

    d['minHsicMaxBnch'] = onp.nanmin(hsicsMax)
    d['difHsicMaxBnch'] = onp.nanmax(hsicsMax) - onp.nanmin(hsicsMax)
    d['minHsicNormBnch'] = onp.nanmin(hsicNrm)
    d['difHsicNormBnch'] = onp.nanmax(hsicNrm) - onp.nanmin(hsicNrm)

    d['minHsicRobBnch'] = onp.nanmin(hsicRob)
    d['difHsicRobBnch'] = onp.nanmax(hsicRob) - onp.nanmin(hsicRob)
    d['minHsicOptOrdBnch'] = onp.nanmin(hsicOptOrd)
    d['difHsicOptOrdBnch'] = onp.nanmax(hsicOptOrd) - onp.nanmin(hsicOptOrd)

    indx = onp.nanargmax(pvals_hsic)
    indxGP, = onp.where(lambdas==-1.0)
    d["minOrdBnch"] = onp.nanmean(optOrd)#onp.nanmean(optOrd[indxGP]) #optOrd[indx]



    res = pd.Series(d, index=['minErrsBnch', 'difErrsBnch', 'maxPvalHsicBnch', 'difPvalHsicBnch',
                              'maxPvalHsicq1Bnch', 'difPvalHsicq1Bnch','maxPvalHsicq2Bnch', 'difPvalHsicq2Bnch',
                              'maxPvalHsicq3Bnch', 'difPvalHsicq3Bnch', 'maxPvalHsicq4Bnch', 'difPvalHsicq4Bnch',
                              'maxPvalHsicq5Bnch', 'difPvalHsicq5Bnch', 'maxPvalHsicq6Bnch', 'difPvalHsicq6Bnch',
                              'maxPvalHsicq7Bnch', 'difPvalHsicq7Bnch', 'maxPvalHsicq8Bnch', 'difPvalHsicq8Bnch',
                              'maxPvalHsicq9Bnch', 'difPvalHsicq9Bnch',
                              'maxPvalHsicAdptBnch', 'difPvalHsicAdptBnch',
                              'minHsicBnch', 'difHsicBnch',
                              'minHsicq1Bnch', 'difHsicq1Bnch','minHsicq2Bnch', 'difHsicq2Bnch',
                              'minHsicq3Bnch', 'difHsicq3Bnch', 'minHsicq4Bnch', 'difHsicq4Bnch',
                              'minHsicq5Bnch', 'difHsicq5Bnch', 'minHsicq6Bnch', 'difHsicq6Bnch',
                              'minHsicq7Bnch', 'difHsicq7Bnch', 'minHsicq8Bnch', 'difHsicq8Bnch',
                              'minHsicq9Bnch', 'difHsicq9Bnch',
                              'minHsicMaxBnch', 'difHsicMaxBnch',
                              'minHsicNormBnch','difHsicNormBnch',
                              'minHsicRobBnch','difHsicRobBnch',
                              'minHsicOptOrdBnch','difHsicOptOrdBnch','minOrdBnch'])

    return res

def voteBnch(x):
    errs_xy = onp.array(x["value"][(x["var"] == "errs") & (x["dir"] == "xy")])
    errs_yx = onp.array(x["value"][(x["var"] == "errs") & (x["dir"] == "yx")])
    hsic_xy = onp.array(x["value"][(x["var"] == "hsic") & (x["dir"] == "xy")])
    hsic_yx = onp.array(x["value"][(x["var"] == "hsic") & (x["dir"] == "yx")])
    argMinHsic_xy = onp.nanargmin(hsic_xy)
    argMinHsic_yx = onp.nanargmin(hsic_yx)

    ent_xy = onp.array(x["value"][(x["var"] == "ent") & (x["dir"] == "xy")])
    ent_yx = onp.array(x["value"][(x["var"] == "ent") & (x["dir"] == "yx")])
    slope_xy = onp.array(x["value"][(x["var"] == "slopes") & (x["dir"] == "xy")])
    slope_yx = onp.array(x["value"][(x["var"] == "slopes") & (x["dir"] == "yx")])
    slope_krr_xy = onp.array(x["value"][(x["var"] == "slopeskrr") & (x["dir"] == "xy")])
    slope_krr_yx = onp.array(x["value"][(x["var"] == "slopeskrr") & (x["dir"] == "yx")])

    scr_err = (errs_yx < errs_xy) * -1 + (errs_yx > errs_xy) * 1
    scr_hsic = (hsic_yx < hsic_xy) * -1 + (hsic_yx > hsic_xy) * 1
    scr_hsic2 = onp.nanmin(hsic_yx)-onp.nanmin(hsic_xy)
    scr_ent = (ent_yx < ent_xy) * -1 + (ent_yx > ent_xy) * 1
    scr_ent2 = ent_yx[argMinHsic_yx]-ent_xy[argMinHsic_xy]
    scr_slope = (slope_yx < slope_xy) * -1 + (slope_yx > slope_xy) * 1
    scr_slope_krr = (slope_krr_yx < slope_krr_xy) * -1 + (slope_krr_yx > slope_krr_xy) * 1

    d = {}
    d['errs'] = onp.mean(scr_err)
    d['hsic'] = onp.mean(scr_hsic)
    d['hsic2'] = scr_hsic2
    d['ent'] = onp.mean(scr_ent)
    d['ent2'] = scr_ent2
    d['slope'] = onp.mean(scr_slope)
    d['slope_krr'] = onp.mean(scr_slope_krr)

    res = pd.Series(d, index=['errs', 'ent','ent2', 'hsic','hsic2', 'slope', "slope_krr"])

    return res

def accuracy(x):
    return onp.sum(x.value > 0) / len(x.value)

def discCol(x):
    xun = onp.unique(x)
    return xun.shape[0]/x.shape[0]

def discData(X):
    return onp.min(onp.apply_along_axis(discCol, 0, X))


def addWeights(repos, dec_long):


    # obtain "discrete" tcep data
    fileDict = {"TCEP-all": ['tcep']}
    fileNmsAll = ["TCEP-all"]
    fileNms = list(fileDict.keys())
    repos2 = "/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/data/"
    reposRelAll = (["TCEPs/" for i in range(2)])
    reposRelAll = {f: r for (f, r) in zip(fileNmsAll, reposRelAll)}
    reposRel = [reposRelAll[k] for k in fileNms]
    files = ["dag2-ME2-" + nm for nm in fileNms]
    fileNames = [repos2 + repRel + file + "_sims.json" for (repRel, file) in zip(reposRel, files)]
    datsNew = getDats(fileNames, fileNms, fileDict, func_dict)
    discLevel = onp.array([discData(datsNew[0][k]) for k in datsNew[0].keys()])
    lvl = 0.2
    indx, = onp.where(discLevel < lvl)
    dts = list(datsNew[0].keys())
    dtsDisc = [dts[i] for i in indx]
    indx, = onp.where(discLevel >= lvl)
    dtsNonDisc = [dts[i] for i in indx]

    dtsGeo = ["1", "2", "3", "4",
              "20", "21",
              "42", "43", "44", "45", "46",
              "49", "50", "51",
              "72", "73", "77", "78", "79", "80",
              "81", "82", "83", "87",
              "89", "90", "91", "92", "93", "94"]
    dtsNonGeo = list(set(dts).difference(dtsGeo))

    #print("num discrete:", len(dtsDisc))
    #print("num non-discrete:", len(dtsNonDisc))
    #print("num GEO:", len(dtsGeo))
    #print("num non-GEO:", len(dtsNonGeo))

    grpDef = [d in dtsGeo for d in dec_long["dataset"]]
    dec_long = addGrp2Dec(dec_long, grpDef, "tcep_GEO")
    grpDef = [d in dtsNonGeo for d in dec_long["dataset"]]
    dec_long = addGrp2Dec(dec_long, grpDef, "tcep_NONGEO")

    grpDef = [(d in dtsDisc) & (t == "tcep") for (d,t) in zip(dec_long["dataset"],dec_long["type"])]
    dec_long = addGrp2Dec(dec_long, grpDef, "tcep_Disc")
    grpDef = [(d in dtsNonDisc) & (t == "tcep") for (d,t) in zip(dec_long["dataset"],dec_long["type"])]
    dec_long = addGrp2Dec(dec_long, grpDef, "tcep_NonDisc")


    # read in tcep weights
    file = "TCEPs/pairs/pairmeta.txt"
    tcepWts= pd.read_csv(repos+file, sep=" ",names=["dataset","caus_first","caus_last","effect_first","effect_last","smpl_wts"])
    tcepWts["dataset"]=[str(ds) for ds in tcepWts["dataset"]]
    if "smpl_wts" in list(dec_long.columns):
        dec_long.drop(["smpl_wts"], axis=1, inplace=True)
    dec_long_tcep = dec_long.loc[dec_long["type"]=="tcep"]
    dec_long_tcep["type"] = "tcep_w"
    dec_long_tcep = dec_long_tcep.merge(tcepWts[["dataset","smpl_wts"]], how="left", on="dataset")
    dec_long["smpl_wts"] = 1.0
    dec_long = pd.concat([dec_long, dec_long_tcep])

    if ("value" in dec_long.columns) & ("conf" in dec_long.columns):
        dec_long["confValue"] = onp.sign(dec_long["value"])*dec_long["conf"]

    return dec_long

def roc(x, var):
    onp.random.seed(0)
    trueVal = onp.random.choice(onp.array([-1, 1]), size=x.shape[0], replace=True)
    pred = x[var] * trueVal
    res = roc_auc_score(trueVal, pred, sample_weight=x["smpl_wts"])
    return res

def proc(x, var):
    onp.random.seed(0)
    trueVal = onp.random.choice(onp.array([-1, 1]), size=x.shape[0], replace=True)
    pred = x[var] * trueVal
    precision, recall, thresholds = precision_recall_curve(trueVal, pred, sample_weight=x["smpl_wts"])
    o = onp.argsort(recall)
    res = auc(recall[o], precision[o])

    return res


def f1(x, var):
    onp.random.seed(0)
    trueVal = onp.random.choice(onp.array([-1, 1]), size=x.shape[0], replace=True)
    pred = (x[var] > 0) * 1 + (x[var] <= 0) * -1
    pred = pred * trueVal
    res = f1_score(trueVal, pred, sample_weight=x["smpl_wts"])
    return res

def bal_acc(x, var):
    onp.random.seed(0)
    trueVal = onp.random.choice(onp.array([-1, 1]), size=x.shape[0], replace=True)
    pred = (x[var] > 0) * 1 + (x[var] <= 0) * -1
    pred = pred * trueVal
    res = balanced_accuracy_score(trueVal, pred, sample_weight=x["smpl_wts"])
    return res

def acc(x, var):
    onp.random.seed(0)
    trueVal = onp.random.choice(onp.array([-1, 1]), size=x.shape[0], replace=True)
    pred = (x[var] > 0) * 1 + (x[var] <= 0) * -1
    pred = pred * trueVal
    res = accuracy_score(trueVal, pred, sample_weight=x["smpl_wts"])
    return res

def getPerfTab(dec, msr, var):
    varsAcc = ["type", 'variable']
    res = dec.groupby(varsAcc).apply(msr, var=var)
    res = res.reset_index()
    res = res.rename(columns={0: "value"})
    res = pd.pivot_table(res, index=["type"], columns=["variable"], values="value")
    res = res.sort_index(key=lambda x: sortCols(x))
    return res

def addGrp2Dec(df, grpDef, grpNm):
    aux = df.loc[grpDef]
    aux["type"] = grpNm
    df = pd.concat([df, aux])
    return df


def nonRejAdd(x, sig):
    return onp.sum(x.maxPvalHsicBnch>sig)/len(x.maxPvalHsicBnch)

def nonRejAdd2(x):
    return onp.sum(x.nonRejAdd)/x.shape[0]


# for obtaining confidence of a measure: voteSmpl and getConf and bnch counterparts
def voteSmpl(smpl, x, ws_nm):
    x2 = x.set_index('smpl')
    res =  vote(x2.loc[smpl], ws_nm)
    return res


def getConf(x, ws_nm, n_smpls):
    n = onp.max(x.smpl)
    n_samples_bootstrap = n_smpls
    random_instance = check_random_state(0)
    smpls = random_instance.randint(0, n, (n, n_samples_bootstrap))
    #res = onp.apply_along_axis(voteSmpl, 0, smpls, ws_nm=ws_nm, x=x).T
    #res = onp.apply_along_axis(accuracy2, 0, res)
    res = [voteSmpl(smpls[:,i], x, ws_nm) for i in range(smpls.shape[1])]
    res = pd.concat(res, axis=1).T
    res = pd.melt(res).groupby(["variable"]).apply(accuracy)
    return res

def voteSmpl_bnch(smpl, x):
    x2 = x.set_index('smpl')
    res = voteBnch(x2.loc[smpl])
    return res

def getConf_bnch(x,  n_smpls):
    n = onp.max(x.smpl)+1
    n_samples_bootstrap = n_smpls
    random_instance = check_random_state(0)
    smpls = random_instance.randint(0, n, (n, n_samples_bootstrap))
    #res = onp.apply_along_axis(voteSmpl, 0, smpls, ws_nm=ws_nm, x=x).T
    #res = onp.apply_along_axis(accuracy2, 0, res)
    res = [voteSmpl_bnch(smpls[:,i], x) for i in range(smpls.shape[1])]
    res = pd.concat(res, axis=1).T
    res = pd.melt(res).groupby(["variable"]).apply(accuracy)
    return res

def getAccuracy(df_long, ws_nm, parVars=[], mask=None,  sig=0.001, res_bnch=None, res_bnch2=None, conf=False, n_smpls=100):
    varsAcc = ["type", 'variable']
    varsAcc = varsAcc + parVars
    varsID = ["type", "dataset"]
    varsID = varsID + parVars
    varsID2 = list(set(varsID).intersection(list(df_long.columns)))
    print("varsID2:",varsID2)
    res = df_long.groupby(varsID2).apply(vote, ws_nm=ws_nm)
    res = res.reset_index()
    repos = "/home/emiliano/Documents/ISP/proyectos/causality/latentNoise_krr/data/"
    res = addWeights(repos, res)

    print("n_smpls:", n_smpls)

    if res_bnch is not None:
        res_bnch = addWeights(repos, res_bnch)
        res = res.merge(res_bnch, how="inner", on=["type", "dataset","smpl_wts"])
        res["additive"] = res["maxPvalHsicBnch"] > sig
        dtsAdd = list(onp.unique(res.loc[(res["additive"]) & (res["type"] == "tcep")]["dataset"]))
        dtsNonAdd = list(onp.unique(
            res.loc[(onp.logical_not(res["additive"])) & (res["type"] == "tcep")]["dataset"]))
        grpDef = [(d in dtsAdd) & (t == "tcep") for (d, t) in zip(res["dataset"], res["type"])]
        res = addGrp2Dec(res, grpDef, "tcep_Add")
        grpDef = [(d in dtsNonAdd) & (t == "tcep") for (d, t) in zip(res["dataset"], res["type"])]
        res = addGrp2Dec(res, grpDef, "tcep_NonAdd")
        #print("cols 1:", res.columns)
        #print("types 1:", onp.unique(res["type"]))

    if mask == "additive":
        res_bnch2 = addWeights(repos, res_bnch2)
        res_bnch2 = res_bnch2.merge(res_bnch, how="inner", on=["type", "dataset"])
        res_bnch2["additive"] = res_bnch2["maxPvalHsicBnch"] > sig
        dtsAdd = list(onp.unique(res_bnch2.loc[(res_bnch2["additive"]) & (res_bnch2["type"] == "tcep")]["dataset"]))
        dtsNonAdd = list(onp.unique(
            res_bnch2.loc[(onp.logical_not(res_bnch2["additive"])) & (res_bnch2["type"] == "tcep")]["dataset"]))
        grpDef = [(d in dtsAdd) & (t == "tcep") for (d, t) in zip(res_bnch2["dataset"], res_bnch2["type"])]
        res_bnch2 = addGrp2Dec(res_bnch2, grpDef, "tcep_Add")
        grpDef = [(d in dtsNonAdd) & (t == "tcep") for (d, t) in zip(res_bnch2["dataset"], res_bnch2["type"])]
        res_bnch2 = addGrp2Dec(res_bnch2, grpDef, "tcep_NonAdd")
        #nmsBnch = [v.split("Bnch")[0] for v in list(set(list(res_bnch2.columns)).difference(["type", "dataset"]))]
        nmsBnch = list(set(list(res_bnch2.columns)).difference(["type", "dataset"]))
        varss = list(set(list(res.columns)).difference(["type", "dataset"]).intersection(nmsBnch))
        res = res.merge(res_bnch2, how="inner", on=["type", "dataset"], suffixes=('', '_bnch'))

        for v in varss:
            res.loc[res["additive"], v] = res.loc[res["additive"], v + "_bnch"]
        res = res[["type", "dataset","additive"] + varss]
        print("types 2:", onp.unique(res["type"]))
    if conf:
        ds = onp.unique(df_long["dataset"])[0]
        st = onp.unique(df_long["type"])[0]
        x = df_long.loc[(df_long["dataset"] == st + "." + ds)]

        print("measure time for one dataset")
        start = time.process_time()
        _ = getConf(x, ws_nm=ws_nm, n_smpls=n_smpls)
        timePerUnit = time.process_time() - start
        print(timePerUnit, " secs")

        print("estimate for all datasets")
        numUnits = df_long[varsID+[ "smpl"]].groupby(varsID).count().shape[0]
        print("numUnits: ", numUnits)
        estimatedTime = timePerUnit * numUnits
        print("estimated time:", estimatedTime / 60, " mins")

        start = time.process_time()
        res1 = df_long.groupby(varsID).apply(getConf, ws_nm=ws_nm, n_smpls=n_smpls)
        actualTime = time.process_time() - start
        print("actual time:", actualTime / 60, " mins")

        res1 = res1.reset_index()
        res2 = res.merge(res1, how="left", on=["type", "dataset"], suffixes=('_decis', '_conf'))
        varsID2 = list(set(varsID).intersection(list(res2.columns)))
        res_long = pd.melt(res2, id_vars=varsID2)
        res_long["typeVar"] = [res_long["variable"].to_list()[i].split("_")[len(res_long["variable"].to_list()[i].split("_")) - 1] for i in range(res_long.shape[0])]
        res_long["variable"] = [res_long["variable"].to_list()[i].split("_")[0] for i in range(res_long.shape[0])]
        res_long = pd.pivot_table(res_long, index=["type", "dataset", "variable"], columns=["typeVar"], values="value")
        res_long = res_long.reset_index()
        res_long.rename(columns={"decis": "value"}, inplace=True)
        res_long.columns.name = None
        res_long["conf"].loc[res_long["value"] < 0] = 1 - res_long["conf"].loc[res_long["value"] < 0]

    else:
        print("cols 2:", res.columns)
        #print("types 3:", onp.unique(res["type"]))
        varsID2 = list(set(varsID).intersection(list(res.columns)))+["smpl_wts"]
        res_long = pd.melt(res, id_vars=varsID2)


    varsIndx =[v in ["ent","ent2","entx","errs", "hsic","hsicc","hsicx","hsicz","mmd"] for v in res_long["variable"]]
    res_long = res_long.loc[varsIndx]

    res_acc = res_long.groupby(varsAcc).apply(acc, "value")
    tabAcc = pd.DataFrame(res_acc)
    tabAcc = tabAcc.reset_index()
    tabAcc = tabAcc.rename(columns={0: "value"}, errors="raise")
    return res_long, tabAcc

# helper for turnToIndex
def mywhere(x):
    res, = onp.where(x)
    return res[0]

# to add "smpl" var to non sampled dfs based on unique parameter combo
def turnToIndex(indexList, pars):
    numPars = len(pars.keys())
    numPerPar = [len(pars[k]) for k in pars.keys()]
    # print("totalPars:", onp.prod(numPerPar))
    indexList2 = onp.array([i for i in indexList])
    aux = onp.array(onp.cumprod(numPerPar[(len(numPerPar) - 1):0:-1])[(len(numPerPar) - 2):None:-1].tolist() + [1])
    # print("aux vec", aux)
    idx = onp.sum(aux * indexList2)
    return idx

def getResBnch(df_long_bnch, conf=False, n_smpls=100):
    res = df_long_bnch.groupby(["type","dataset"]).apply(aggBnch)
    res = res.reset_index()
    res_long = pd.melt(res, id_vars=["type","dataset"])
    res_bnch = res_long.pivot_table(index=["type","dataset"], columns=["variable"], values="value")
    res_bnch = res_bnch.reset_index()
    res_bnch2 = df_long_bnch.groupby(['type',"dataset"]).apply(voteBnch)
    res_bnch2 = res_bnch2.reset_index()

    if conf:
        ds = onp.unique(df_long_bnch["dataset"])[0]
        st = onp.unique(df_long_bnch["type"])[0]
        x = df_long_bnch.loc[(df_long_bnch["dataset"] == st + "." + ds)]
        start = time.process_time()
        aux = getConf_bnch(x, n_smpls=n_smpls)
        timePerUnit = time.process_time() - start
        print(timePerUnit, " secs")

        print("estimate for all datasets")
        varsID = ["type", "dataset"]
        numUnits = df_long_bnch[varsID + ["smpl"]].groupby(varsID).count().shape[0]
        print("numUnits: ", numUnits)
        estimatedTime = timePerUnit * numUnits
        print("estimated time:", estimatedTime / 60, " mins")

        start = time.process_time()
        res1 = df_long_bnch.groupby(varsID).apply(getConf_bnch, n_smpls=n_smpls)
        actualTime = time.process_time() - start
        print("actual time:", actualTime / 60, " mins")

        res1 = res1.reset_index()
        res2 = res_bnch2.merge(res1, how="left", on=["type", "dataset"], suffixes=('_decis', '_conf'))
        res_long = pd.melt(res2, id_vars=varsID)
        res_long["typeVar"] = [
            res_long["variable"].to_list()[i].split("_")[len(res_long["variable"].to_list()[i].split("_")) - 1] for i in
            range(res_long.shape[0])]
        res_long["variable"] = [res_long["variable"].to_list()[i].split("_")[0] for i in range(res_long.shape[0])]
        res_long = pd.pivot_table(res_long, index=["type", "dataset", "variable"], columns=["typeVar"], values="value")
        res_long = res_long.reset_index()
        res_long.rename(columns={"decis": "value"}, inplace=True)
        res_long.columns.name = None
        res_long["conf"].loc[res_long["value"] < 0] = 1 - res_long["conf"].loc[res_long["value"] < 0]


    return res_long, res_bnch, res_bnch2

def getAccuracyBnch(res_bnch2):
    res_bnch2_long = pd.melt(res_bnch2, id_vars=["type","dataset"])
    res_acc = res_bnch2_long.groupby(["type",'variable']).apply(accuracy)
    tabAcc = pd.DataFrame(res_acc)
    tabAcc = tabAcc.reset_index()
    tabAcc = tabAcc.rename(columns={0:"value"}, errors="raise")
    return tabAcc

# for combining results from two tables based on confidence or additivity
def confLeft(x):
    return onp.sum(x.conf_x > x.conf_y) / x.shape[0]

def combine_conf(x):
    return onp.sum((x.value_x > 0) * (x.conf_x > x.conf_y) + (x.value_y > 0) * (x.conf_y > x.conf_x)) / x.shape[0]

def combine_add(x, sig):
    return onp.sum((x.value_x > 0) * (x.maxPvalHsicBnch > sig) + (x.value > 0) * (x.maxPvalHsicBnch <= sig)) / x.shape[0]

def combineDecision(resAdd_long, resNonAdd_long, res_bnch, res_bnch2, sig):
    tab = resAdd_long.merge(resNonAdd_long, how="left", on=["type", "dataset", "variable"])
    res_bnch2_long = pd.melt(res_bnch2, id_vars=["type", "dataset"])
    res_bnch2_long["variable"] = [v.split("Bnch")[0] for v in res_bnch2_long["variable"]]
    tab = tab.merge(res_bnch, how="left", on=["type", "dataset"]).merge(res_bnch2_long, how="left",
                                                                        on=["type", "dataset", "variable"])

    tab_leftMoreconf = tab.groupby(["type", "variable"]).apply(confLeft)
    tab_leftMoreconf = tab_leftMoreconf.reset_index()
    tab_leftMoreconf = tab_leftMoreconf.rename(columns={0: "value"}, errors="raise")
    tab_leftMoreconf = pd.pivot_table(tab_leftMoreconf, index="type", columns="variable", values="value")

    tab_acc_combConf = tab.groupby(["type", "variable"]).apply(combine_conf)
    tab_acc_combConf = tab_acc_combConf.reset_index()
    tab_acc_combConf = tab_acc_combConf.rename(columns={0: "value"}, errors="raise")
    tab_acc_combConf = pd.pivot_table(tab_acc_combConf, index="type", columns="variable", values="value")

    tab_addit = tab.groupby(["type", "variable"]).apply(nonRejAdd, sig=sig)
    tab_addit = tab_addit.reset_index()
    tab_addit = tab_addit.rename(columns={0: "value"}, errors="raise")
    tab_addit = pd.pivot_table(tab_addit, index="type", columns="variable", values="value")

    tab_acc_combAdd = tab.groupby(["type", "variable"]).apply(combine_add, sig=sig)
    tab_acc_combAdd = tab_acc_combAdd.reset_index()
    tab_acc_combAdd = tab_acc_combAdd.rename(columns={0: "value"}, errors="raise")
    tab_acc_combAdd = pd.pivot_table(tab_acc_combAdd, index="type", columns="variable", values="value")

    return tab, tab_leftMoreconf, tab_acc_combConf, tab_addit, tab_acc_combAdd

###################################################################3
# PIPELINE UTILS
#############################################################3

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    res = chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))
    return [list(inSt) for inSt in res]

# format and return result table
def formatTabAcc(tabAcc, initStrat, pairUpNm, parsPairUp, weightNm, parsWeight, mask, sig):
    tabAcc["initStrat"] = "_".join(initStrat)
    tabAcc["pairUp"] = pairUpNm
    parsPairUpNms = [k+"_"+v for k, v in zip(parsPairUp.keys(), [str(el) for el in parsPairUp.values()])]
    parsPairUpNms = '-'.join(parsPairUpNms)
    tabAcc["parPairUpNm"] = parsPairUpNms
    tabAcc["weight"] = weightNm
    parsWeightNms = [k+"_"+v for k, v in zip(parsWeight.keys(), [str(el) for el in parsWeight.values()])]
    parsWeightNms = '-'.join(parsWeightNms)
    tabAcc["parWeightNm"] = parsWeightNms
    tabAcc["mask"] = mask
    tabAcc["sig"] = sig
    return tabAcc

# expand list of dictionaries: for each dictionary do all possible prermutations of the list
# of values given for each key... return a dataframe
def expandPars(dictFixed, pars, nm):
    parsCombos = [[ { k:it for k,it in zip(pars[i].keys(), list(ite))}  for ite in itertools.product(*[pars[i][ke] for ke in  pars[i].keys()]) ] for i in range(len(pars))]
    print(nm)
    print(parsCombos)
    indxRest = [i for i in range(len(parsCombos)) for j in range(len(parsCombos[i])) ]
    parsCombos = [parsCombos[i][j]  for i in range(len(parsCombos)) for j in range(len(parsCombos[i]))]
    dictFixed2 = {k:[dictFixed[k][i] for i in indxRest] for k in dictFixed.keys()}
    dictFixed2["pars"+nm] = parsCombos
    res = pd.DataFrame(dictFixed2)
    res["pars"+nm+"_id"] = ["-".join([k+"_"+str(v) for k,v in zip(d.keys(), d.values())]) for d in res["pars"+nm]]
    return res

# "expand dataframes" by doing cross-product concatenation of their rows
def makeCopy(df, i):
    df2 = df.copy()
    df2["on"] = i
    return df2

def permDFs(df1, df2):
    df1c = df1.copy()
    df1c["on"] = onp.arange(df1.shape[0])
    df2c = pd.concat([ makeCopy(df2,i) for i in range(df1.shape[0])])
    df = df1c.merge(df2c, on="on")
    df.drop(["on"], inplace=True, axis=1)
    return df

def getPairUpDF(df, initStrat, pairUpFunct, parsPairUpFunct, parsPairUp):
    # filter by init strategy
    df2 = df.copy()
    df2 = df2.loc[[ot in initStrat for ot in df2["ot"]]]
    # pair-up
    parsPairUp2 = tuple(i for i in parsPairUp.values())
    varss = ['errs','hsic','hsicc','ent','slopes','slopeskrr','hsicz','hsiczz','mmd']
    df2 = df2.groupby(['type',"dataset"]).apply(pairup,varss=varss, funct=pairUpFunct, getParsFunct=parsPairUpFunct, pars=parsPairUp2)
    df2 = df2.reset_index()
    df2 = df2.rename(columns={"level_2": "smpl"})
    return df2

def getAcc(df, res_bnch, res_bnch2, initStrat, pairUpNm, parsPairUp, weightNm, weightModMatFunct, weightModMatPars,
            weightModFunct, weightModPars, weightFunct, getWeightPars, parsTup, parsWeight, conf=False, n_smpls=100):
    nm = "w_" + weightNm
    df = getWeightsWrapper(df, weightModMatFunct, weightModMatPars, weightModFunct, weightModPars, nm, weightFunct,
                           getWeightPars, parsTup, **parsWeight)
    # long format
    vars_weights = list(df.columns[list(onp.where([var.split("_")[0] == "w" for var in list(df.columns)]))[0]])
    print(vars_weights)
    df_long = getLongFormat2(df, ["type", "dataset", "smpl"] + vars_weights)
    # get non-masked accuracy
    sig = 0.001
    res_long , tabAcc = getAccuracy(df_long,nm, parVars=["additive"], conf=conf, n_smpls=n_smpls, sig=sig, res_bnch=res_bnch)
    print(pd.pivot_table(tabAcc, index=["type"], columns=["variable"], values="value", aggfunc=onp.sum))
    # get masked accuracy
    _ , tabAcc_mask = getAccuracy(df_long, nm, parVars=["additive"], mask="additive", sig=sig, res_bnch=res_bnch, res_bnch2=res_bnch2)
    # df, df_rand, df_intsc, df_nn
    # "w_unif", "w_lowestHsic", "w_effFront", 'w_modSimpLogis', "w_modSimpRF", 'w_modCmplxLogis',"w_modCmplxRF"
    print(pd.pivot_table(tabAcc_mask, index=["type"], columns=["variable"], values="value", aggfunc=onp.sum))
    tabAcc = formatTabAcc(tabAcc, initStrat, pairUpNm, parsPairUp, weightNm, parsWeight, "None", -1.0)
    tabAcc_mask = formatTabAcc(tabAcc_mask, initStrat, pairUpNm, parsPairUp, weightNm, parsWeight, "additive", sig)
    tabAcc = pd.concat([tabAcc, tabAcc_mask])
    tabAcc = tabAcc.reset_index()
    return res_long, tabAcc


def getPipelineDF(initStrats, parsPairUp, parsWeight, pairUpNm=None, weightNm=None):
    pairUpNm2 = ["byParm", "rand", "intsc", "NN"]
    weightNm2 = ["uniform", "lowestHsic", "effFront", "modSimp_logis", "modSimp_RF"]

    if pairUpNm is None:
        pairUpNm = pairUpNm2
    if weightNm is None:
        weightNm = weightNm2

    initStratsDF = pd.DataFrame.from_dict({"initStrat": initStrats}, orient="columns")
    initStratsDF["initStrat_id"] = ["_".join(el) for el in initStratsDF["initStrat"]]

    # pair up options
    idx = onp.array([mywhere(pu in pairUpNm2) for pu in pairUpNm2])
    funct = [[smplParm, smplRand, smplFromIntersection, nearestNeighbors][i] for i in idx]
    parsFunct = [[getParsParm, getParsRand, getParsIntersection, getParsNN][i] for i in idx]
    pairUpDict = {"pairUpNm": pairUpNm, "pairUpFunct": funct, "pairUpParsFunct": parsFunct}
    pairUpDF = expandPars(pairUpDict, parsPairUp, "PairUp")

    # weight options
    varsExpl = ["dif_err", "dif_hsic", "dif_hsicc", "dif_mmd", "dif_hsicz", "min_err", "min_hsic", "min_hsicc",
                "min_mmd", "min_hsicz", "max_err", "max_hsic", "max_hsicc", "max_mmd", "max_hsicz"]

    m = int(5000)

    idx = onp.array([mywhere(ws in weightNm2) for ws in weightNm2])
    getModMatFunct = [[getModMatVan, getModMatVan, getModMatVan, getModMat, getModMat][i] for i in idx]
    getModMatPars = [[None, None, None, m, m][i] for i in idx]
    getModFunct = [[getModVan, getModVan, getModVan, getModSimp_logis, getModSimp_RF][i] for i in idx]
    getModPars = [[None, None, None, varsExpl, varsExpl][i] for i in idx]
    nm = [["w_unif", "w_lowestHsic", "w_effFront", "w_modSimpLogis", "w_modSimpRF"][i] for i in idx]
    weightFunct = [[unifo, lowestHsic, effFront, modSimp_logis, modSimp_RF][i] for i in idx]
    getWeightPars = [[getParsUnifW, getParsHsicW, getParsEffFrontW, getParsModSimp_logisW, getParsModSimp_RFW][i] for i in idx]
    parsTup = [[[], ["var", "sig"], ["varsss", "sig"], ["varsExpl", "var_smpl_nm", "mod"],
               ["varsExpl", "var_smpl_nm", "mod"]][i] for i in idx]
    weightDict = {"weightNm": weightNm, "weightModMatFunct": getModMatFunct, "weightModMatPars": getModMatPars,
                  "weightModFunct": getModFunct, "weightModPars": getModPars, "weightVar": nm,
                  "weightFunct": weightFunct, "weightParsFunct": getWeightPars, "weightParsTup": parsTup}

    weightDF = expandPars(weightDict, parsWeight, "Weight")
    print("weightDF.shape: ", weightDF.shape)
    print(weightDF)

    # initStratsDF, pairUpDF, weightDF
    initStrat_pairUp_df = permDFs(initStratsDF, pairUpDF)
    initStrat_pairUp_df["df_id"] = "initStrat:" + initStrat_pairUp_df["initStrat_id"] + "*" + initStrat_pairUp_df[
        "pairUpNm"] + ":" + initStrat_pairUp_df["parsPairUp_id"]
    initStrat_pairUp_weight_df = permDFs(initStrat_pairUp_df, weightDF)
    initStrat_pairUp_weight_df["acc_id"] = initStrat_pairUp_weight_df["df_id"] + "*" + initStrat_pairUp_weight_df[
        "weightNm"] + ":" + initStrat_pairUp_weight_df["parsWeight_id"]
    return initStrat_pairUp_df, initStrat_pairUp_weight_df

def gridSize(pars):
    return onp.prod([len(pars[k]) for k in pars.keys()])


def sortCols(x):
    rightOrder = ["SIM","SIMc","SIMln","SIMG","AN","AN-s","LS","LS-s","MN-U","all","add","nonAdd","tcep", "tcep_w", "tcep_Add", "tcep_NonAdd", "tcep_GEO", "tcep_NONGEO", "tcep_Disc",
                  "tcep_NonDisc"]
    if x.name == "type":
        indx = [mywhere([xx == r for r in rightOrder]) for xx in list(x)]
        res = indx
    elif x.name == "nonRejAdd":
        indx = [mywhere([xx == r for r in [True, False,"pct"]]) for xx in list(x)]
        res = indx
    else:
        res = x

    return res


def addAdd(dec_combo):
    dec_combo["type2"] = [t.split("_")[0] for t in dec_combo["type"]]

    # all non tcep
    dtsAll = list(onp.unique(dec_combo.loc[ (dec_combo["type"] != "tcep")]["dataset"]))
    grpDef = [(t != "tcep") for t in dec_combo["type2"]]
    dec_combo = addGrp2Dec(dec_combo, grpDef, "all")

    # additive/non-additive tcep
    dtsAdd = list(onp.unique(dec_combo.loc[(dec_combo["nonRejAdd"]) & (dec_combo["type"]=="tcep")]["dataset"]))
    dtsNonAdd = list(onp.unique(dec_combo.loc[(onp.logical_not(dec_combo["nonRejAdd"])) & (dec_combo["type"]=="tcep")]["dataset"]))
    print("dtsAdd: ", len(dtsAdd))
    print("dtsNonAdd: ", len(dtsNonAdd))
    grpDef = [(d in dtsAdd) & (t == "tcep") for (d,t) in zip(dec_combo["dataset"],dec_combo["type"])]
    dec_combo = addGrp2Dec(dec_combo, grpDef, "tcep_Add")
    grpDef = [(d in dtsNonAdd) & (t == "tcep") for (d,t) in zip(dec_combo["dataset"],dec_combo["type"])]
    dec_combo = addGrp2Dec(dec_combo, grpDef, "tcep_NonAdd")

    #additive/non additive non-tcep
    dtsAdd = list(onp.unique(dec_combo.loc[(dec_combo["nonRejAdd"]) & (dec_combo["type2"] !="tcep")]["dataset"]))
    dtsNonAdd = list(onp.unique(dec_combo.loc[(onp.logical_not(dec_combo["nonRejAdd"])) & (dec_combo["type2"] !="tcep")]["dataset"]))
    #print("dtsAdd: ", dtsAdd)
    #print("dtsNonAdd: ", dtsNonAdd)
    print("dtsAdd: ", len(dtsAdd))
    print("dtsNonAdd: ", len(dtsNonAdd))
    grpDef = [(d in dtsAdd) & (t != "tcep") for (d,t) in zip(dec_combo["dataset"],dec_combo["type2"])]
    dec_combo = addGrp2Dec(dec_combo, grpDef, "add")
    grpDef = [(d in dtsNonAdd) & (t != "tcep") for (d,t) in zip(dec_combo["dataset"],dec_combo["type2"])]
    dec_combo = addGrp2Dec(dec_combo, grpDef, "nonAdd")
    return dec_combo

def sortRows(x):
    rightOrder = ["none","ANM","LN"]
    rightOrderComb = ["none","add-test","hsicx-hsic","hsicx-hsicx"]
    rightOrderVar = ["hsic","hsicx","ent","hsicz","hsicEnt","hsicxEnt"]
    if (x.name == "model-add") | (x.name == "model-nonAdd"):
        indx = [mywhere([xx == r for r in rightOrder]) for xx in list(x)]
        res = indx
    elif (x.name=="comb-crit"):
        indx = [mywhere([xx == r for r in rightOrderComb]) for xx in list(x)]
        res = indx
    elif (x.name=="variable"):
        indx = [mywhere([xx == r for r in rightOrderVar]) for xx in list(x)]
        res = indx
    else:
        res = x

    return res

def getPerfTabAll(dec, msr, var):
    varsAcc = ["type", 'variable',"model-add","model-nonAdd","comb-crit"]
    res = dec.groupby(varsAcc).apply(msr, var=var)
    res = res.reset_index()
    res = res.rename(columns={0: "value"})
    res = pd.pivot_table(res, index=["model-nonAdd","model-add","comb-crit","variable"], columns=["type"], values="value")
    res = res.sort_index(axis=1, key=lambda x: sortCols(x))
    res = res.sort_index(axis=0, key=lambda x: sortRows(x))
    return res

print("here")
def getPerfTabAllByAdd(dec, msr, var):
    varsAcc = ["type", 'variable',"model-add","model-nonAdd","comb-crit","nonRejAdd"]
    res = dec.groupby(varsAcc).apply(msr, var=var)
    res = res.reset_index()
    res = res.rename(columns={0: "value"})
    res = pd.pivot_table(res, index=["model-nonAdd","model-add","comb-crit","variable"], columns=["type","nonRejAdd"], values="value")
    res = res.sort_index(axis=1, key=lambda x: sortCols(x))
    res = res.sort_index(axis=0, key=lambda x: sortRows(x))
    return res