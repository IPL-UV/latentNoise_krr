from sklearn.manifold import Isomap
import sys
sys.path.insert(1,'./ITE-1.1_code')
import ite
from typing import Tuple, Optional, Dict, Callable, Union

# JAX SETTINGS
import jax
import jax.numpy as np
import jax.random as random
from jax.ops import index, index_add, index_update
from jax.scipy.special import ndtri
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI
from jax.scipy import linalg
from jax.scipy.stats import norm


# NUMPY SETTINGS
import numpy as onp
import time
onp.set_printoptions(precision=3, suppress=True)

# LOGGING SETTINGS
import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger()
#logger.setLevel(logging.INFO)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from slope import *

# SCIPY
import scipy.stats as stats


print("enters latentNoise_funcs_gen.py")

# @title Kernel Functions

# Squared Euclidean Distance Formula
@jax.jit
def sqeuclidean_distance(x, y):
    return np.sum((x - y) ** 2)


# RBF Kernel
@jax.jit
def rbf_kernel(params, x, y):
    return np.exp(- params['gamma'] * sqeuclidean_distance(x, y))


# Covariance Matrix
def covariance_matrix(kernel_func, x, y):
    mapx1 = jax.vmap(
        lambda x, y: kernel_func(x, y), in_axes=(0, None), out_axes=0)
    mapx2 = jax.vmap(
        lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
    return mapx2(x, y)


# Covariance Matrix
def rbf_kernel_matrix(params, x, y):
    mapx1 = jax.vmap(lambda x, y: rbf_kernel(params, x, y), in_axes=(0, None), out_axes=0)
    mapx2 = jax.vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
    return mapx2(x, y)


@jax.jit
def kernel_mat(D_x, x, z, sig_x_h, sig_z_h, sigs_f):
    sig_x_f = sigs_f[0]
    sig_z_f = sigs_f[1]
    sig_xz_f = sigs_f[2]
    K_x_f = np.exp(-sig_x_f * D_x)
    K_z_f = rbf_kernel_matrix({'gamma': sig_z_f}, z, z)
    K_xz_f = rbf_kernel_matrix({'gamma': sig_xz_f}, x, z)
    K_zx_f = np.transpose(K_xz_f)

    K_x_h = K_x_f #np.exp(-sig_x_h * D_x)
    K_z_h = K_z_f #rbf_kernel_matrix({'gamma': sig_z_h}, z, z)
    #K_a_h = K_x_h * K_z_h
    K_a_f = (2 * K_x_f + 2 * K_z_f + K_xz_f + K_zx_f) / 6
    K_a_h = K_a_f

    return K_a_f, K_a_h, K_x_h, K_z_h

# this version has diff kernel calculations for objective than for function
@jax.jit
def kernel_mat(D_x, x, z, sig_x_h, sig_z_h, sigs_f):
    sig_x_f = sigs_f[0]
    sig_z_f = sigs_f[1]
    sig_xz_f = sigs_f[2]
    K_x_f = np.exp(-sig_x_f * D_x)
    K_z_f = rbf_kernel_matrix({'gamma': sig_z_f}, z, z)
    #K_xz_f = rbf_kernel_matrix({'gamma': sig_xz_f}, x, z)
    #K_zx_f = np.transpose(K_xz_f)

    K_x_h = np.exp(-sig_x_h * D_x)
    K_z_h = rbf_kernel_matrix({'gamma': sig_z_h}, z, z)
    K_a_h = K_x_h * K_z_h
    #K_a_f = 2 * K_x_f + 2 * K_z_f + K_xz_f + K_zx_f
    K_a_f = K_x_f * K_z_f

    return K_a_f, K_a_h, K_x_h, K_z_h

# this version is to save on matrix computation when sigs_f are fixed
@jax.jit
def kernel_mat(D_x, x, z, sig_x_h, sig_z_h, sigs_f):
    sig_x_f = sigs_f[0]
    sig_z_f = sigs_f[1]
    sig_xz_f = sigs_f[2]
    K_x_f = np.exp(-sig_x_f * D_x)
    K_z_f = rbf_kernel_matrix({'gamma': sig_z_f}, z, z)
    #K_xz_f = rbf_kernel_matrix({'gamma': sig_xz_f}, x, z)
    #K_zx_f = np.transpose(K_xz_f)

    K_x_h = K_x_f #np.exp(-sig_x_h * D_x)
    K_z_h = K_z_f #rbf_kernel_matrix({'gamma': sig_z_h}, z, z)
    K_a_h = K_x_h * K_z_h
    #K_a_f = 2 * K_x_f + 2 * K_z_f + K_xz_f + K_zx_f
    K_a_f = K_a_h

    return K_a_f, K_a_h, K_x_h, K_z_h

# p-value helper functions
@jax.jit
def kernel_mat_gen(D_x, z_hat, z, sig_x_f, sig_z_f):
    K_x_f = np.exp(-sig_x_f * D_x)
    K_z_f = rbf_kernel_matrix({'gamma': sig_z_f}, z, z_hat)

    K_a_f = K_x_f * K_z_f

    return K_a_f

@jax.jit
def getYp(D_x, z, zp_i, sig_x_f, sig_z_f, eps_i, alfa):
    K_a_f = kernel_mat_gen(D_x, z,  zp_i, sig_x_f, sig_z_f)
    Ey = np.dot(K_a_f, alfa)
    yp = Ey + eps_i
    return yp[:,0]

@jax.jit
def getWeightsResids(D_x, zp_i, sig_x_f, sig_z_f, lam, yp_i):
    K_a_f = kernel_mat_gen(D_x, zp_i, zp_i, sig_x_f, sig_z_f)
    n = K_a_f.shape[0]
    ws = np.ones(n)
    weights_p, resids_p, _ = krrModel(lam, K_a_f, yp_i, ws)
    return np.hstack([weights_p, resids_p])

@jax.jit
def hsicRBFs_cause(z, K_x, sig_z):
    K_z = rbf_kernel_matrix({'gamma': sig_z}, z, z)
    K_x = centering(K_x)
    K_z = centering(K_z)
    return np.sum(K_x * K_z) / np.linalg.norm(K_x) / np.linalg.norm(K_z)

@jax.jit
def hsicRBFs_resids(K_x, z, r, sig_r):
    K_z = rbf_kernel_matrix({'gamma': 1}, z, z)
    K_c = K_z*K_x
    K_r = rbf_kernel_matrix({'gamma': sig_r}, r, r)
    K_c = centering(K_c)
    K_r = centering(K_r)
    return np.sum(K_c * K_r) / np.linalg.norm(K_c) / np.linalg.norm(K_r)
# end of p-value helper functions


@jax.jit
def krrModel(lam, K, y, ws):
    # cho factor the cholesky
    # L = linalg.cho_factor(K + lam * np.eye(K.shape[0]))
    L = linalg.cho_factor(K + lam * np.diag(1 / ws))

    # weights
    weights = linalg.cho_solve(L, y)

    # save the params

    y_hat = np.dot(K, weights)

    resids = y - y_hat

    # return the predictions
    return weights, resids, y_hat


def myComputeScoreKrr(K,sigma, target,rows,mindiff, mindiff_x, lam):
    #base_cost=model_score(k) + k*onp.log2(V); # same for both directions
    ws = np.ones(K.shape[0])
    weights, resids, _ = krrModel(lam, K,target, ws);
    sse = onp.sum(resids**2)
    cost_alpha  = onp.sum([onp.sum(onp.array([model_score(onp.array([float(weights[i,j])])) for i in range(weights.shape[0])])) for j in range(weights.shape[1])])
    cost_sigma = model_score(onp.array([float(sigma)]))
    cost_resids = gaussian_score_emp_sse(sse,rows,mindiff)
    cost = cost_alpha + cost_sigma + cost_resids
    # add cost of marginal which is not included here
    cost = cost - rows * logg(mindiff_x)
    return cost


def yhatW(ws, lam, K, y):
    a,b, c, y_hat = krrModel(lam, K, y, ws)
    res = y_hat[:, 0]
    return res

@jax.jit
def centering(K):
    n_samples = K.shape[0]
    logging.debug(f"N: {n_samples}")
    logging.debug(f"I: {np.ones((n_samples, n_samples)).shape}")
    H = np.eye(K.shape[0], ) - (1 / n_samples) * np.ones((n_samples, n_samples))
    return np.dot(np.dot(H, K), H)


# Normalized Hsic - from kernels
@jax.jit
def hsic(K_x, K_z):
    K_x = centering(K_x)
    K_z = centering(K_z)
    return np.sum(K_x * K_z) / np.linalg.norm(K_x) / np.linalg.norm(K_z)


# Normalized Hsic - from features using rbf kernels
@jax.jit
def hsicRBF(x, z):
    distsX = covariance_matrix(sqeuclidean_distance, x, x)
    sigma = 1 / np.median(distsX)
    K_x = rbf_kernel_matrix({'gamma': sigma}, x, x)
    distsZ = covariance_matrix(sqeuclidean_distance, z, z)
    sigma = 1 / np.median(distsZ)
    K_z = rbf_kernel_matrix({'gamma': sigma}, z, z)
    K_x = centering(K_x)
    K_z = centering(K_z)
    return np.sum(K_x * K_z) / np.linalg.norm(K_x) / np.linalg.norm(K_z)


@jax.jit
def MMD_norm(z):
    n = z.shape[0]
    distsZ = covariance_matrix(sqeuclidean_distance, z, z)
    sigma_z = 1  # /np.median(distsZ)
    gamma = 1 / np.sqrt(2 * sigma_z)
    K_z = rbf_kernel_matrix({'gamma': sigma_z}, z, z)
    diagSum = np.sum(np.diag(K_z))
    offDiagSum = np.sum(K_z) - diagSum
    sigma_z = 1 / (2 * (1 + gamma ** 2))
    K_z = rbf_kernel_matrix({'gamma': sigma_z}, z, np.array([0]))
    diagSum2 = np.sum(K_z)
    res = np.sqrt((gamma ** 2) / (2 + gamma ** 2)) - (2 / n) * np.sqrt((gamma ** 2) / (1 + gamma ** 2)) * diagSum2 + (
                1 / (n * (n - 1))) * (offDiagSum)

    return res


@jax.jit
def MMDb_norm(z):
    n = z.shape[0]
    distsZ = covariance_matrix(sqeuclidean_distance, z, z)
    sigma_z = 1  # /np.median(distsZ)
    gamma = 1 / np.sqrt(2 * sigma_z)
    K_z = rbf_kernel_matrix({'gamma': sigma_z}, z, z)
    sumTot = np.sum(K_z)
    sigma_z = 1 / (2 * (1 + gamma ** 2))
    K_z = rbf_kernel_matrix({'gamma': sigma_z}, z, np.array([0]))
    diagSum2 = np.sum(K_z)
    res = np.sqrt((gamma ** 2) / (2 + gamma ** 2)) - (2 / n) * np.sqrt((gamma ** 2) / (1 + gamma ** 2)) * diagSum2 + (
                1 / (n ** 2)) * (sumTot)

    return res


@jax.jit
def var_MMD(gamma, n):
    res = ((gamma ** 2) / (2 + gamma ** 2))
    res = res + np.sqrt((gamma ** 2) / (4 + gamma ** 2))
    res = res - 2 * np.sqrt((gamma ** 4) / ((1 + gamma ** 2) * (3 + gamma ** 2)))
    res = (2 / (n * (n - 1))) * res
    return res


@jax.jit
def SMMD_norm(z):
    n = z.shape[0]
    distsZ = covariance_matrix(sqeuclidean_distance, z, z)
    sigma_z = 1  # /np.median(distsZ)
    gamma = 1 / np.sqrt(2 * sigma_z)
    res = MMD_norm(z)
    res = res / np.sqrt(var_MMD(gamma, n))
    return res


@jax.jit
def SMMDb_norm(z):
    n = z.shape[0]
    distsZ = covariance_matrix(sqeuclidean_distance, z, z)
    sigma_z = 1  # /np.median(distsZ)
    gamma = 1 / np.sqrt(2 * sigma_z)
    res = MMDb_norm(z)
    res = res / np.sqrt(var_MMD(gamma, n))
    return res


@jax.jit
def mse(y, y_hat):
    return np.sqrt(np.mean((y - y_hat) ** 2))

# for comparing against no z vanilla model

def model_van(lam, K_x, x, y):
    # find kernel stuffs
    n = K_x.shape[0]
    ws = np.ones(n)
    weights, resids, y_hat = krrModel(lam, K_x, y, ws)

    # right now we simply overwrite residual lengthscale sig_r_h to be median heuristic
    distsR = covariance_matrix(sqeuclidean_distance, resids, resids)
    sig_r_h = 1 / np.quantile(distsR, 0.5)

    K_r = rbf_kernel_matrix({'gamma': sig_r_h}, resids, resids)

    # hsic
    hsic_resids_x = hsic(K_x, K_r)

    # entropy
    co1 = ite.cost.BHShannon_KnnK()  # initialize the entropy (2nd character = ’H’) estimator
    h_x = co1.estimation(x)  # entropy estimation
    h_r = co1.estimation(resids)

    # slope score
    source = onp.array(x)
    target = onp.array(y)
    rows = y.shape[0]
    mindiff = CalculateMinDiff(y)
    mindiff_x = CalculateMinDiff(x[:,0])
    k = onp.array([2])
    V = 3
    M = 2
    F = 9
    cost_slope, _ = myComputeScore(source, target, rows, mindiff, mindiff_x, k, V, M, F)
    distsX = covariance_matrix(sqeuclidean_distance, x, x)
    sigma = 1 / np.median(distsX)
    cost_slope_krr = myComputeScoreKrr(K_x, sigma, target, rows, mindiff, mindiff_x, lam)

    # calculate norm
    penalize = np.linalg.norm(weights.T @ K_x @ weights)
    monitor = {}
    monitor = {
        'mse': mse(y, y_hat),
        'penalty': penalize,
        'hsic_rx': hsic_resids_x,
        'h_x': h_x,
        'h_r': h_r,
        'cost_slope': cost_slope,
        'cost_slope_krr': cost_slope_krr,
    }

    return monitor


def getModels_van(X, lam, sig):
    print("enters getModels_van")
    print("X.shape", X.shape)
    x = X[:, 0][:, None]
    y = X[:, 1][:, None]
    N = x.shape[0]
    p = X.shape[1]


    distsX = covariance_matrix(sqeuclidean_distance, x, x)
    #sigmaX = 1 / np.median(distsX)
    sigmaX = 1 / np.quantile(distsX, sig[0])
    K_x = rbf_kernel_matrix({'gamma': sigmaX}, x, x)
    distsY = covariance_matrix(sqeuclidean_distance, y, y)
    #sigmaY = 1 / np.median(distsY)
    sigmaY = 1 / np.quantile(distsY, sig[0])
    K_y = rbf_kernel_matrix({'gamma': sigmaY}, y, y)

    monitor_xy = model_van(lam, K_x, x, y)
    monitor_yx = model_van(lam, K_y, y, x)

    res = {"xy": monitor_xy, "yx": monitor_yx}

    if p > 2:
        print("enters z section")
        z = X[:, 2:]
        distsZ = covariance_matrix(sqeuclidean_distance, z, z)
        sigmaZ = 1 / np.median(distsZ)
        K_z = rbf_kernel_matrix({'gamma': sigmaZ}, z, z)
        K_cx = K_x*K_z
        K_cy = K_y * K_z
        caus_x = np.hstack([x, z])
        caus_y = np.hstack([y, z])
        monitor_cy = model_van(lam, K_cx, caus_x, y)
        monitor_cx = model_van(lam, K_cy, caus_y, x)
        res["cy"] = monitor_cy
        res["cx"] = monitor_cx
    print("exits getModels_van")
    return res


# for reporting purposes give back 3 terms separatley
def model(optType, params, lam, D_x, x, y, z, M, K_zmani, K_xy):
    # find kernel stuffs
    z_hat = getZ(params, optType, M, K_xy)
    z_hat = stdrze_mat(z_hat)
    sig_x_h = np.exp(params["ln_sig_x_h"])
    sig_z_h = np.exp(params["ln_sig_z_h"])
    # sig_r_h = np.exp(params["ln_sig_r_h"])

    # for now use median heuristic for reporting hsic values of loss. Later we will want to
    # use the sigs being used in the loss function (as they move up) and perhaps also report
    # various "hsic views" with pre-determined quantiles to show how annealing works
    # distsZ = covariance_matrix(sqeuclidean_distance, z, z)
    # sig_z_h = 1/np.quantile(distsZ, 0.5)

    sig_z_f = np.exp(params["ln_sig_z_f"])
    sig_x_f = np.exp(params["ln_sig_x_f"])
    sig_xz_f = np.exp(params["ln_sig_xz_f"])
    sigs_f = np.hstack([sig_x_f, sig_z_f, sig_xz_f])
    K_a_f, K_a_h, K_x_h, K_z_h = kernel_mat(D_x, x, z_hat, sig_x_h, sig_z_h, sigs_f)

    n = K_a_f.shape[0]
    ws = np.ones(n)
    weights, resids, y_hat = krrModel(lam, K_a_f, y, ws)

    # right now we simply overwrite residual lengthscale sig_r_h to be median heuristic
    distsR = covariance_matrix(sqeuclidean_distance, resids, resids)
    sig_r_h = 1 / np.quantile(distsR, 0.5)

    K_r_h = rbf_kernel_matrix({'gamma': sig_r_h}, resids, resids)

    # hsic
    hsic_val = hsic(K_x_h, K_z_h)
    # hsic resids
    hsic_resids_x = hsic(K_x_h, K_r_h)
    hsic_resids_z = hsic(K_z_h, K_r_h)
    hsic_resids_c = hsic(K_a_h, K_r_h)
    # hsic zmani
    hsic_zmani = hsic(K_z_h, K_zmani)
    # dependence to true z
    if z is not None:
        hsic_zzhat = hsicRBF(z, z_hat)
    else:
        hsic_zzhat = None

    # algorithmic dependency
    #hsic_alg = hsicRBF(W, Yhat)

    #MMD
    MMDzn = SMMDb_norm(z_hat)

    z2 = norml(z_hat)
    # entropy
    co1 = ite.cost.BHShannon_KnnK()  # initialize the entropy (2nd character = ’H’) estimator
    caus = np.hstack([x, z2])
    h_z = co1.estimation(z2)
    h_x = co1.estimation(x)
    h_c = co1.estimation(caus)  # entropy estimation
    h_r = co1.estimation(resids)
    h_a = co1.estimation(weights)

    # slope score
    source = onp.array(caus)
    target = onp.array(y)
    rows = y.shape[0]
    mindiff = CalculateMinDiff(y)
    mindiff_x = CalculateMinDiff(x)
    k = onp.array([2])
    V = 3
    M = 2
    F = 9
    cost_slope, _ = myComputeScore(source, target, rows, mindiff, mindiff_x, k, V, M, F)
    distsX = covariance_matrix(sqeuclidean_distance, x, x)
    sigma = 1 / np.median(distsX)
    cost_slope_krr = myComputeScoreKrr(K_a_f, sigma, target, rows, mindiff, mindiff_x, lam)
    #cost_slope_z = onp.sum([onp.sum(onp.array([model_score(onp.array([float(params['Z'][i, j])])) for i in range(params['Z'].shape[0])])) for j in range(params['Z'].shape[1])])
    mindiff_z = CalculateMinDiff(onp.array(z_hat).flatten())
    cost_slope_z = - rows * logg(mindiff_z)

    # calculate norm
    penalize = np.linalg.norm(weights.T @ K_a_f @ weights)
    monitor = {}
    monitor = {
        'mse': mse(y, y_hat),
        'penalty': penalize,
        'hsic': hsic_val,
        'hsic_rx': hsic_resids_x,
        'hsic_rz': hsic_resids_z,
        'hsic_r': hsic_resids_c,
        'hsic_zmani': hsic_zmani,
        'MMDzn': MMDzn,
     #   'hsic_alg': hsic_alg,
        'h_xx': h_x,
        'h_z': h_z,
        'h_x': h_c,
        'h_r': h_r,
        'h_a': h_a,
        'cost_slope': cost_slope,
        'cost_slope_krr': cost_slope_krr,
        'cost_slope_z': cost_slope_z,
        'hsic_zzhat': hsic_zzhat

    }

    return monitor


# for reporting purposes give back 3 terms separatley
def modelGP(params, x, y, M, lam):

    z = params["Z"]

    input_z = onp.array(z)  # note we use all the zs
    input_x = onp.array(x)
    output = onp.array(y)
    input = onp.hstack([input_x, input_z])

    gpKern = RBF() + WhiteKernel()
    gpModel_with = GaussianProcessRegressor(kernel=gpKern)
    gpModel_with.fit(input, output)
    gpModel_without = GaussianProcessRegressor(kernel=gpKern)
    gpModel_without.fit(input_x, output)

    yhat_with = gpModel_with.predict(input)
    yhat_without = gpModel_without.predict(input_x)
    resids = y - yhat_with
    resids_without = y - yhat_without

    distsX = covariance_matrix(sqeuclidean_distance, x, x)
    sigma_x = 1 / np.median(distsX)
    distsZ = covariance_matrix(sqeuclidean_distance, input_z, input_z)
    sigma_z = 1 / np.median(distsZ)
    distsR = covariance_matrix(sqeuclidean_distance, resids, resids)
    sigma_r = 1 / np.median(distsR)
    distsRW = covariance_matrix(sqeuclidean_distance, resids_without, resids_without)
    sigma_rw = 1 / np.median(distsRW)

    K_x = rbf_kernel_matrix({'gamma': sigma_x}, x, x)
    K_z = rbf_kernel_matrix({'gamma': sigma_z}, input_z, input_z)
    K = K_x * K_z
    K_r = rbf_kernel_matrix({'gamma': sigma_r}, resids, resids)
    K_rw = rbf_kernel_matrix({'gamma': sigma_rw}, resids_without, resids_without)
    # hsic
    hsic_val = hsicRBF(x, input_z)
    # hsic resids
    # hsic_resids = hsic(K_x, K_r)
    hsic_resids_x = hsic(K_x, K_r)
    hsic_resids_z = hsic(K_z, K_r)
    hsic_resids_c = hsic(K, K_r)

    hsic_resids_without_x = hsic(K_x, K_rw)

    # hsic_alg = hsicRBF(W, Yhat)

    z2 = norml(input_z)
    # entropy
    co1 = ite.cost.BHShannon_KnnK()  # initialize the entropy (2nd character = ’H’) estimator
    caus = np.hstack([x, z2])  # [:,None]
    h_z = co1.estimation(z2)  # [:,None]
    h_x = co1.estimation(x)  # entropy estimation
    h_c = co1.estimation(caus)  # entropy estimation
    h_r = co1.estimation(resids)
    h_r_without = co1.estimation(resids_without)

    # slope score
    source = onp.array(caus)
    target = onp.array(y)
    rows = y.shape[0]
    mindiff = CalculateMinDiff(y)
    mindiff_x = CalculateMinDiff(x)
    k = onp.array([2])
    V = 3
    M = 2
    F = 9
    cost_slope, _ = myComputeScore(source, target, rows, mindiff, mindiff_x, k, V, M, F)
    distsX = covariance_matrix(sqeuclidean_distance, x, x)
    sigma = 1 / np.median(distsX)
    cost_slope_krr = myComputeScoreKrr(K, sigma, target, rows, mindiff, mindiff_x, lam)
    #cost_slope_z = onp.sum([onp.sum(onp.array([model_score(onp.array([float(params['Z'][i, j])])) for i in range(params['Z'].shape[0])])) for j in range(params['Z'].shape[1])])
    mindiff_z = CalculateMinDiff(onp.array(params['Z']).flatten())
    cost_slope_z = - rows * logg(mindiff_z)

    # calculate norm
    monitor = {}
    monitor = {
        'hsic': hsic_val,
        'hsic_rx': hsic_resids_x,
        'hsic_rz': hsic_resids_z,
        'hsic_rc': hsic_resids_c,
        'hsic_rx_without': hsic_resids_without_x,
        'h_x': h_x,
        'h_z': h_z,
        'h_c': h_c,
        'h_r': h_r,
        'h_r_without': h_r_without,
        "cost_slope": cost_slope,
        "cost_slope_krr": cost_slope_krr,
        "cost_slope_z": cost_slope_z
    }

    return monitor


# loss for free z approach
@jax.jit
def loss_freeZ(params, beta, neta, eta, lam, nu, lu, D_x, x, y, M, K_zmani, ws):
    sig_x_h = np.exp(params["ln_sig_x_h"])
    sig_z_h = np.exp(params["ln_sig_z_h"])


    z = params["Z"]
    z = (z - np.mean(z)) / (np.std(z))

    sig_x_f = np.exp(params["ln_sig_x_f"])
    sig_z_f = np.exp(params["ln_sig_z_f"])
    sig_xz_f = np.exp(params["ln_sig_xz_f"])
    sigs_f = np.hstack([sig_x_f, sig_z_f, sig_xz_f])

    K_a_f, K_a_h, K_x_h, K_z_h = kernel_mat(D_x, x, z, sig_x_h, sig_z_h, sigs_f)
    n = K_a_f.shape[0]

    weights, resids, y_hat = krrModel(lam, K_a_f, y, ws)

    # right now we simply overwrite residual lengthscale sig_r_h to be median heuristic
    distsR = covariance_matrix(sqeuclidean_distance, resids, resids)
    sig_r_h = 1 / np.quantile(distsR, 0.5)

    K_r_h = rbf_kernel_matrix({'gamma': sig_r_h}, resids, resids)

    # hsic
    hsic_val = hsic(K_x_h, K_z_h)

    # hsic resids
    hsic_resids = hsic(K_a_h, K_r_h)

    # hsic zmani
    hsic_zmani = hsic(K_z_h, K_zmani)

    # MMD
    MMDzn = SMMDb_norm(z)
    correctFactor = 1  # jax_stats.norm.cdf(MMDzn, loc=-0.1447, scale=1.321)
    MMDzn = MMDzn * correctFactor

    # calcualte compute loss
    loss_value = np.log(hsic_resids) + beta * np.log(mse(y, y_hat))  + n * neta * np.log(hsic_val)  - n * eta * np.log(hsic_zmani) + nu*np.log(np.dot(z.T, z)[1,1]) + lu*np.log(MMDzn)
    #loss_value = np.log(1+hsic_resids) + beta * np.log(1+mse(y, y_hat))  + n * neta * np.log(1+hsic_val)  - n * eta * np.log(1+hsic_zmani) + nu*np.log(1+np.dot(z.T, z)[1,1]) + lu*np.log(1+MMDzn)
    #loss_value = hsic_resids + beta * mse(y, y_hat)  + n * neta * hsic_val  - n * eta * hsic_zmani + nu*np.dot(z.T, z)[1,1] + lu*MMDzn

    return loss_value[0]

# loss for z = M * alfa
@jax.jit
def loss_mani(params, beta, neta, eta, lam, nu, lu, D_x, x, y, M, K_zmani, ws):
    sig_x_h = np.exp(params["ln_sig_x_h"])
    sig_z_h = np.exp(params["ln_sig_z_h"])

    bbeta = params["bbeta"]
    z = np.dot(M, bbeta)
    # so z_ini will be normal and we only have to adjust for variance
    z = (z - np.mean(z)) / (np.std(z))

    sig_x_f = np.exp(params["ln_sig_x_f"])
    sig_z_f = np.exp(params["ln_sig_z_f"])
    sig_xz_f = np.exp(params["ln_sig_xz_f"])
    sigs_f = np.hstack([sig_x_f, sig_z_f, sig_xz_f])

    K_a_f, K_a_h, K_x_h, K_z_h = kernel_mat(D_x, x, z, sig_x_h, sig_z_h, sigs_f)
    n = K_a_f.shape[0]

    weights, resids, y_hat = krrModel(lam, K_a_f, y, ws)

    # right now we simply overwrite residual lengthscale sig_r_h to be median heuristic
    distsR = covariance_matrix(sqeuclidean_distance, resids, resids)
    sig_r_h = 1 / np.quantile(distsR, 0.5)

    K_r_h = rbf_kernel_matrix({'gamma': sig_r_h}, resids, resids)

    # hsic
    hsic_val = hsic(K_x_h, K_z_h)

    # hsic resids
    hsic_resids = hsic(K_a_h, K_r_h)

    # hsic zmani
    hsic_zmani = hsic(K_z_h, K_zmani)

    # MMD
    MMDzn = SMMDb_norm(z)
    correctFactor = 1  # jax_stats.norm.cdf(MMDzn, loc=-0.1447, scale=1.321)
    MMDzn = MMDzn * correctFactor

    # calcualte compute loss
    loss_value = np.log(hsic_resids) + beta * np.log(mse(y, y_hat)) + n * neta * np.log(hsic_val) - n * eta * np.log(
        hsic_zmani) + nu * np.log(np.dot(z.T, z)[1, 1]) + lu * np.log(MMDzn)


    return loss_value[0]

# loss for z = K_xy * alfa
@jax.jit
def loss_hilb(params, beta, neta, eta, lam, nu, lu, D_x, x, y, M, K_zmani, ws):
    sig_x_h = np.exp(params["ln_sig_x_h"])
    sig_z_h = np.exp(params["ln_sig_z_h"])

    # sig_z_h = np.exp(params["ln_sig_z_h"])
    # sig_r_h = np.exp(params["ln_sig_r_h"])
    alfa = params["alfa"]
    # distsX = covariance_matrix(sqeuclidean_distance, x, x)
    sigmaX = 1 / np.median(D_x)
    K_x = rbf_kernel_matrix({'gamma': sigmaX}, x, x)
    distsY = covariance_matrix(sqeuclidean_distance, y, y)
    sigmaY = 1 / np.median(distsY)
    K_y = rbf_kernel_matrix({'gamma': sigmaY}, y, y)
    K_xy = K_x * K_y
    z = np.dot(K_xy, alfa)
    z = (z - np.mean(z)) / (np.std(z))
    # z = np.exp(params['ln_Z'])
    # z = z/np.sum(z)
    sig_x_f = np.exp(params["ln_sig_x_f"])
    sig_z_f = np.exp(params["ln_sig_z_f"])
    sig_xz_f = np.exp(params["ln_sig_xz_f"])
    sigs_f = np.hstack([sig_x_f, sig_z_f, sig_xz_f])

    # right now we simply overwrite z lengthscale sig_z_h to be median heuristic
    # distsZ = covariance_matrix(sqeuclidean_distance, z, z)
    # sig_z_h = 1/np.quantile(distsZ, 0.5)
    # sig_z_h = 30 #sig_z_f
    # sig_x_h = sig_x_f

    K_a_f, K_a_h, K_x_h, K_z_h = kernel_mat(D_x, x, z, sig_x_h, sig_z_h, sigs_f)
    n = K_a_f.shape[0]

    weights, resids, y_hat = krrModel(lam, K_a_f, y, ws)
    pen = alfa.T @ K_xy @ alfa
    pen = np.sum(pen)

    # right now we simply overwrite residual lengthscale sig_r_h to be median heuristic
    distsR = covariance_matrix(sqeuclidean_distance, resids, resids)
    sig_r_h = 1 / np.quantile(distsR, 0.5)

    K_r_h = rbf_kernel_matrix({'gamma': sig_r_h}, resids, resids)

    # hsic
    hsic_val = hsic(K_x_h, K_z_h)

    # hsic resids
    hsic_resids = hsic(K_a_h, K_r_h)
    # hsic_resids = hsic(K_x_h, K_r_h) + 10*hsic(K_z_h, K_r_h)

    # hsic zmani
    hsic_zmani = hsic(K_z_h, K_zmani)

    # MMD
    MMDzn = SMMDb_norm(z)
    correctFactor = 1  # jax_stats.norm.cdf(MMDzn, loc=-0.1447, scale=1.321)
    MMDzn = MMDzn * correctFactor

    # calcualte compute loss
    loss_value = np.log(hsic_resids) + beta * np.log(mse(y, y_hat)) + n * neta * np.log(hsic_val) - n * eta * np.log(
        hsic_zmani) + nu * np.log(np.dot(z.T, z)[1, 1]) + lu * np.log(MMDzn) + 0.1*lam * np.log(pen)



    return loss_value[0]


dloss_freeZ = jax.grad(loss_freeZ, )
dloss_freeZ_jitted = jax.jit(dloss_freeZ)

dloss_mani = jax.grad(loss_mani, )
dloss_mani_jitted = jax.jit(dloss_mani)

dloss_hilb = jax.grad(loss_hilb, )
dloss_hilb_jitted = jax.jit(dloss_hilb)



# initialize parameters - getIniPar(optType, N, reps, y)
# called only from getLatentZs to initialize parameters
def getIniPar(optType, N, m, reps, y, M):
    onp.random.seed(seed=4)
    if optType=="freeZ":
        z_ini = y * np.ones(reps)
        z_ini = np.array(onp.apply_along_axis(normalize, 0, z_ini))
        params = {
            'Z': z_ini,  # [:, None],
        }
    elif optType=="mani":
        bbeta_ini = np.array(onp.random.randn(m, reps))
        params = {
            'bbeta': bbeta_ini,  # [:, None],
        }
    elif optType == "hilb":
        alfa_ini = np.array(onp.random.randn(N, reps))
        params = {
            'alfa': alfa_ini,  # [:, None],
        }
    elif optType == "freeZ-iniMani":
        Mp = M[:,0:(reps)]
        Mp = onp.apply_along_axis(normalize, 0, Mp)
        z_ini = np.array(Mp)
        params = {
            'Z': z_ini
        }
    elif optType == "freeZ-iniR":
        z_ini = np.array(onp.random.randn(N, reps))
        params = {
            'Z': z_ini,  # [:, None],
        }


    return params

# calculate ini z getIniZ(params, optType, M)
# from getLatentZ to initialize z and M depending on
# what has been run before (ie what is available) and
# what optType has been specified
def getIniZ(params, optType, M, K_xy):

    print("enters getIniZ with optType: ", optType)
    n = K_xy.shape[0]
    m = M.shape[1]

    if optType == "freeZ":
        if ("Z" not in params.keys()) & ("bbeta" in params.keys()):
            z_ini = np.dot(M, params["bbeta"])
            # M should be normal(0,1) under any circumstance
            # so z_ini will be normal and we only have to adjust for variance
            z_ini =  stdrze_mat(z_ini)
            params["Z"] = z_ini
        elif ("Z" not in params.keys()) & ("alfa" in params.keys()):
            z_ini = np.dot(K_xy, params["alfa"])
            z_ini = np.array(onp.apply_along_axis(normalize, 0, z_ini))
            params["Z"] = z_ini
        else:
            z_ini = params["Z"]

    elif (optType == "freeZ-iniMani")|(optType == "freeZ-iniR"):
        z_ini = params["Z"]

    elif optType == "mani":
        if ("bbeta" not in params.keys()) & ("Z" in params.keys()) :
            L = linalg.cho_factor(np.dot(M.T, M) + 0.0001 * np.eye(m))
            bbeta = linalg.cho_solve(L, np.dot(M.T, params["Z"]))
            params["bbeta"] = bbeta
        elif ("bbeta" not in params.keys()) & ("alfa" in params.keys()):
            z_hat = np.dot(K_xy, params["alfa"])
            z_hat = np.array(onp.apply_along_axis(normalize, 0, z_hat))
            L = linalg.cho_factor(np.dot(M.T, M) + 0.0001 * np.eye(n))
            bbeta = linalg.cho_solve(L, np.dot(M.T, z_hat))
            params["bbeta"] = bbeta
        else:
            bbeta = params["bbeta"]
        print("bbeta shape: ", bbeta.shape)
        z_ini = np.dot(M, bbeta)
        # so z_ini will be normal and we only have to adjust for variance
        z_ini =  stdrze_mat(z_ini)

    elif optType == "mani-postZ":
        M = params["Z"]
        reps = M.shape[1]
        m = M.shape[1]
        bbeta_ini = np.array(onp.random.randn(m, reps))
        params['bbeta']= bbeta_ini  # [:, None],

        M = onp.apply_along_axis(normalize, 0, M)
        z_ini  = np.dot(M, bbeta_ini)
        # so z_ini will be normal and we only have to adjust for variance
        z_ini = stdrze_mat(z_ini)

    elif optType == "hilb":
        if ("alfa" not in params.keys()) & ("Z" in params.keys()):
            L = linalg.cho_factor(K_xy + 0.0001 * np.eye(n))
            alfa = linalg.cho_solve(L, params["Z"])
            params["alfa"] = alfa
        elif ("alfa" not in params.keys()) & ("bbeta" in params.keys()):
            z_hat = np.dot(M, params["bbeta"])
            z_hat = stdrze_mat(z_hat)
            L = linalg.cho_factor(K_xy + 0.0001 * np.eye(n))
            alfa = linalg.cho_solve(L, z_hat)
            params["alfa"] = alfa
        else:
            alfa = params["alfa"]
        print("alfa shape: ", alfa.shape)
        z_ini = np.dot(K_xy, alfa)
        z_ini = np.array(onp.apply_along_axis(normalize, 0, z_ini))

    print("leaves getIni Z with z_ini.shape: ", z_ini.shape)

    return z_ini, M

# called from model and from getLatentZ for last rep to calculate hsic(zhat)
def getZ(params, optType, M, K_xy):
    print("enters getZ with optType: ", optType)


    if (optType == "freeZ")|(optType == "freeZ-iniMani")|(optType == "freeZ-iniR"):
        z_ini = params["Z"]

    elif (optType == "mani") | (optType == "mani-postZ"):
        bbeta = params["bbeta"]
        z_ini = np.dot(M, bbeta)
        # so z_ini will be normal and we only have to adjust for variance
        z_ini =  stdrze_mat(z_ini)


    elif optType == "hilb":
        alfa = params["alfa"]
        z_ini = np.dot(K_xy, alfa)
        z_ini = np.array(onp.apply_along_axis(normalize, 0, z_ini))

    print("leaves get Z with z_ini.shape: ", z_ini.shape)

    return z_ini


 # get parameters for grad - getParamsForGrad(params, rep, optType, smpl)

def getParamsForGrad(params, rep, optType, smpl):

    ln_sig_x_f_aux = params["ln_sig_x_f"][rep]
    ln_sig_z_f_aux = params["ln_sig_z_f"][rep]
    ln_sig_xz_f_aux = params["ln_sig_xz_f"][rep]

    ln_sig_x_h_aux = params["ln_sig_x_h"][rep]
    ln_sig_z_h_aux = params["ln_sig_z_h"][rep]
    ln_sig_r_h_aux = params["ln_sig_r_h"][rep]
    params_aux = params.copy()

    if (optType == "freeZ")|(optType == "freeZ-iniMani")|(optType == "freeZ-iniR"):
        z_aux = params['Z'][:, rep]
        z_aux = z_aux[smpl,][:,None]
        params_aux['Z'] = z_aux
    elif (optType == "mani")|(optType == "mani-postZ"):
        bbeta_aux = params['bbeta'][:, rep]
        bbeta_aux = bbeta_aux[:, None]
        params_aux['bbeta'] = bbeta_aux
    elif optType == "hilb":
        alfa_aux = params['alfa'][:, rep]
        alfa_aux = alfa_aux[smpl,][:, None]
        params_aux['alfa'] = alfa_aux

    params_aux['ln_sig_x_f'] = ln_sig_x_f_aux
    params_aux['ln_sig_z_f'] = ln_sig_z_f_aux
    params_aux['ln_sig_xz_f'] = ln_sig_xz_f_aux

    params_aux['ln_sig_x_h'] = ln_sig_x_h_aux
    params_aux['ln_sig_z_h'] = ln_sig_z_h_aux
    params_aux['ln_sig_r_h'] = ln_sig_r_h_aux

    return params_aux

# update params - updateParams(params, grad_params, optType, smpl, rep)
def updateParams(params, grad_params, optType, smpl, iteration, rep, learning_rate):

    if (optType == "freeZ")|(optType == "freeZ-iniMani")|(optType == "freeZ-iniR"):
        idx_rows = smpl[:, None]
        idx_cols = np.array(rep)[None, None]
        idx = jax.ops.index[tuple([idx_rows, idx_cols])]
        A = params['Z'][tuple([idx_rows, idx_cols])]
        B = learning_rate * grad_params['Z']
        params['Z'] = index_update(params['Z'], idx, A - B)
        n = params['Z'].shape[0]
        idx_rows2 = np.linspace(0, n - 1, n, dtype=int)
        idx2 = jax.ops.index[tuple([idx_rows2, idx_cols])]
        if ((iteration + 1) % 100) == 0:
            params["Z"] = index_update(params['Z'], idx2, normalize(params['Z'][:, rep]))

    elif (optType == "mani")|(optType == "mani-postZ"):
        m = params['bbeta'].shape[0]
        idx_rows = np.linspace(0, m, m, dtype=int)
        idx_rows = idx_rows[:, None]
        idx_cols = np.array(rep)[None, None]
        idx = jax.ops.index[tuple([idx_rows, idx_cols])]
        A = params['bbeta'][tuple([idx_rows, idx_cols])]
        B = learning_rate * grad_params['bbeta']
        params['bbeta'] = index_update(params['bbeta'], idx, A - B)  # [:,None]


    elif optType == "hilb":
        idx_rows = smpl[:, None]
        idx_cols = np.array(rep)[None, None]
        idx = jax.ops.index[tuple([idx_rows, idx_cols])]
        A = params['alfa'][tuple([idx_rows, idx_cols])]
        B = learning_rate * grad_params['alfa']
        if onp.sum(onp.isnan(B)) > 0:
            print("nans in iteration: ", iteration, " rep: ", rep) 	
        params['alfa'] = index_update(params['alfa'], idx, A- B)  # [:,None]

    #params['ln_sig_x_f'] = index_update(params['ln_sig_x_f'], rep,
    #	                           params['ln_sig_x_f'][rep] - learning_rate * grad_params['ln_sig_x_f'])
            
    #params['ln_sig_z_f'] = index_update(params['ln_sig_z_f'], rep,
    #                           params['ln_sig_z_f'][rep] - learning_rate * grad_params['ln_sig_z_f'])

    #params['ln_sig_xz_f'] = index_update(params['ln_sig_xz_f'], rep,
    #                           params['ln_sig_xz_f'][rep] - learning_rate * grad_params['ln_sig_xz_f'])


    return None

# prepare parameters for report - getParamsForReport(params, rep, optType)

def getParamsForReport(params, rep, optType):
    params_aux = params.copy()
    if (optType == "freeZ")|(optType == "freeZ-iniMani")|(optType == "freeZ-iniR"):
        params_aux['Z'] = params_aux['Z'][:, rep][:, None]
    elif (optType == "mani")|(optType == "mani-postZ"):
        params_aux['bbeta'] = params_aux['bbeta'][:, rep][:, None]
    elif optType == "hilb":
        params_aux['alfa'] = params_aux['alfa'][:, rep][:, None]

    params_aux['ln_sig_x_f'] = params_aux['ln_sig_x_f'][rep]
    params_aux['ln_sig_z_f'] = params_aux['ln_sig_z_f'][rep]
    params_aux['ln_sig_xz_f'] = params_aux['ln_sig_xz_f'][rep]
    params_aux['ln_sig_x_h'] = params_aux['ln_sig_x_h'][rep]
    params_aux['ln_sig_z_h'] = params_aux['ln_sig_z_h'][rep]
    params_aux['ln_sig_r_h'] = params_aux['ln_sig_r_h'][rep]

    return params_aux

def getLatentZ(params, loss_as_par, dloss_as_par_jitted, optType, x, y, z, M, lam, sig, beta, neta, eta, nu, lu, epochs, report_freq, reps, batch_size, learning_rate):
    N = x.shape[0]

    D_x = covariance_matrix(sqeuclidean_distance, x, x)
    sigma_x_med = 1 / np.median(D_x)
    sigma_x_q = 1 / np.quantile(D_x, sig[0])
    K_x = rbf_kernel_matrix({'gamma': sigma_x_med}, x, x)

    D_y = covariance_matrix(sqeuclidean_distance, y, y)
    sigma_y = 1 / np.median(D_y)
    K_y = rbf_kernel_matrix({'gamma': sigma_y}, y, y)

    K_xy = K_x * K_y

    distsZm = covariance_matrix(sqeuclidean_distance, M, M)
    sigma_zm = 1 / np.median(distsZm)
    K_zmani = rbf_kernel_matrix({'gamma': sigma_zm}, M, M)

    num_reports = int(np.ceil(epochs / report_freq))-1

    # initialize parameters
    m = M.shape[1]

    z_ini, M = getIniZ(params, optType, M, K_xy)
    print("z_ini.shape: ", z_ini.shape)

    D_z = covariance_matrix(sqeuclidean_distance, z_ini, z_ini)
    sigma_z_med = 1 / np.median(D_z)
    sigma_z_q = 1 / np.quantile(D_z, sig[0])
    D_xz = covariance_matrix(sqeuclidean_distance, x, z_ini)
    D_xz = D_xz + np.transpose(D_xz)
    sigma_xz_med = 1 / np.median(D_xz)
    sigma_xz_q = 1 / np.quantile(D_xz, sig[0])

    n = D_x.shape[0]

    K_z = rbf_kernel_matrix({'gamma': sigma_z_med}, z_ini, z_ini)
    K_xz = K_x*K_z

    ws = np.ones(N)
    print("lam: ", lam)
    print("sig: ", sig[0])
    print("sigma_z: ", sigma_z_q)
    print("sigma_x: ", sigma_x_q)
    print("type(sig): ", type(sig))
    weights, resids, y_hat = krrModel(lam, K_xz, y, ws)
    print("resids.shape: ", resids.shape)
    print("resids.mean", onp.mean(resids))
    # right now we simply overwrite residual lengthscale sig_r_h to be median heuristic
    distsR = covariance_matrix(sqeuclidean_distance, resids, resids)
    sigma_r_med = 1 / np.quantile(distsR, 0.5)

    print("sigma_x_med", sigma_x_med)
    print("sigma_z_med", sigma_z_med)
    print("sigma_xz_med", sigma_xz_med)
    print("sigma_r_med", sigma_r_med)

    # params['ln_sig_z_f']=np.log(sigma_z)*np.ones(reps)
    # params['ln_sig_z_f']=np.log(0.1)*np.ones(reps)
    params['ln_sig_z_f']= np.log(sigma_z_q) * np.ones(reps)
    params['ln_sig_x_f']=np.log(sigma_x_q) * np.ones(reps)
    # params['ln_sig_xz_f']=np.log(sigma_xz)*np.ones(reps)
    params['ln_sig_xz_f']=np.log(sigma_xz_q) * np.ones(reps)
    # params['ln_sig_z_h']=np.log(sigma_z)*np.ones(reps)
    params['ln_sig_z_h']=np.log(sigma_z_med) * np.ones(reps)
    # params['ln_sig_x_h']=np.log(sigma_x*10)*np.ones(reps)
    params['ln_sig_x_h']=np.log(sigma_x_med) * np.ones(reps)
    params['ln_sig_r_h']=np.log(sigma_r_med) * np.ones(reps)

    monitors = {
        'loss': onp.zeros([num_reports, reps]),
        'hsic': onp.zeros([num_reports, reps]),
        'hsic_r': onp.zeros([num_reports, reps]),
        'hsic_rx': onp.zeros([num_reports, reps]),
        'hsic_rz': onp.zeros([num_reports, reps]),
        'hsic_zmani': onp.zeros([num_reports, reps]),
        'MMDzn': onp.zeros([num_reports, reps]),
        #'hsic_alg': onp.zeros([num_reports, reps]),
        'errs': onp.zeros([num_reports, reps]),
        'ent_c': onp.zeros([num_reports, reps]),
        'ent_x': onp.zeros([num_reports, reps]),
        'ent_z': onp.zeros([num_reports, reps]),
        'ent_r': onp.zeros([num_reports, reps]),
        'ent_alpha': onp.zeros([num_reports, reps]),
        'pen': onp.zeros([num_reports, reps]),
        'cost_slope': onp.zeros([num_reports, reps]),
        'cost_slope_krr': onp.zeros([num_reports, reps]),
        'cost_slope_z': onp.zeros([num_reports, reps]),
        "hsic_zzhat": onp.zeros([num_reports, reps]),
        'hsic_zhat': onp.zeros([num_reports, 1])
        #'hsic_pval': onp.zeros([num_reports, reps]),
        #'mmd_pval': onp.zeros([num_reports, reps]),
        #"hsic_r_pval": onp.zeros([num_reports, reps])
        #'errs_pval': onp.zeros([num_reports, reps])
    }


    for iteration in range(epochs):

        # print("*********************")
        #print("iteration: ", iteration)
        #print("nans: ", onp.sum(onp.isnan(onp.array(params["Z"]))))


        # get the gradient of the loss
        for rep in range(reps):
            #print("rep: ", rep)

            smpl = onp.random.randint(low=0, high=n, size=batch_size)

            # caluclate K_x kernel

            D_x_aux = D_x[smpl, :]
            D_x_aux = D_x_aux[:, smpl]
            K_zmani_aux = K_zmani[smpl, :]
            K_zmani_aux = K_zmani_aux[:, smpl]
            M_aux = M[smpl,:]

            # equal weights
            #ws = np.ones(batch_size)


            # algorithmic independence forcing

            # random weights - so that E[y|x] alg indep of p(x)
            indx = onp.random.randint(low=1, high=batch_size, size=1)
            x_aux = x[smpl,]
            #distsX = covariance_matrix(sqeuclidean_distance, x_aux, x_aux)
            sigmaX = 10  # /np.median(distsX)
            K_x_aux = rbf_kernel_matrix({'gamma': sigmaX}, x_aux, x_aux)
            ws = K_x_aux[:, indx].squeeze()
            ws = ws + 1e-9
            ws = ws / sum(ws)*batch_size

            x_aux = x[smpl,]
            y_aux = y[smpl,]

            # prepare parameters for grad calculation (subsample)
            params_aux = getParamsForGrad(params, rep, optType, smpl)


            grad_params = dloss_as_par_jitted(params_aux, beta, neta, eta, lam, nu, lu, D_x_aux, x_aux, y_aux, M_aux, K_zmani_aux, ws)

            
            # update params
            updateParams(params, grad_params, optType, smpl, iteration, rep, learning_rate)



            # prepare parameters for reporting (full smpl)
            params_aux = getParamsForReport(params, rep, optType)

            if (iteration % report_freq == 0) & (iteration != 0):
                #print("report")
                indxRep = int(iteration / report_freq)-1
                monitor = model(optType, params_aux, lam, D_x, x, y, z, M, K_zmani, K_xy)
                monitors['hsic'][indxRep, rep] = monitor['hsic']
                monitors['hsic_r'][indxRep, rep] = monitor['hsic_r']
                monitors['hsic_rx'][indxRep, rep] = monitor['hsic_rx']
                monitors['hsic_rz'][indxRep, rep] = monitor['hsic_rz']
                monitors['hsic_zmani'][indxRep, rep] = monitor['hsic_zmani']
                monitors['MMDzn'][indxRep, rep] = monitor['MMDzn']
                #monitors['hsic_alg'][indxRep, rep] = monitor['hsic_alg']
                monitors['errs'][indxRep, rep] = monitor['mse']
                monitors['ent_c'][indxRep, rep] = monitor['h_x']
                monitors['ent_x'][indxRep, rep] = monitor['h_xx']
                monitors['ent_z'][indxRep, rep] = monitor['h_z']
                monitors['ent_r'][indxRep, rep] = monitor['h_r']
                monitors['ent_alpha'][indxRep, rep] = monitor['h_a']
                monitors['pen'][indxRep, rep] = monitor['penalty']
                monitors['cost_slope'][indxRep, rep] = monitor['cost_slope']
                monitors['cost_slope_krr'][indxRep, rep] = monitor['cost_slope_krr']
                monitors['cost_slope_z'][indxRep, rep] = monitor['cost_slope_z']
                monitors["hsic_zzhat"][indxRep, rep] = monitor["hsic_zzhat"]

                ws = np.ones(K_x.shape[0])
                loss_val = loss_as_par(params_aux, beta, neta, eta, lam, nu, lu, D_x, x, y, M, K_zmani, ws)
                monitors['loss'][indxRep, rep] = loss_val

                if rep == (reps - 1):
                    # print("enters hsic zhat calc")
                    z_hat = getZ(params, optType, M, K_xy)
                    hsics_zs = [onp.array(hsicRBF(z_hat[:, i], z_hat[:, j])) for i in range(z_hat.shape[1]) for j in range(i)]
                    monitors["hsic_zhat"][indxRep, 0] = onp.mean(hsics_zs)
            # print("loss: ", loss_val)

    if False:

        start = time.process_time()
        print("p-value calcu")
        num_smpl1 = x.shape[0]
        num_smpl2 = 1000
        zp = onp.random.randn(num_smpl1, num_smpl2)
        distsZ = covariance_matrix(sqeuclidean_distance, zp[:, 0][:, None], zp[:, 0][:, None])
        sig_z = 1 / np.median(distsZ)
        distHSIC_cause = onp.apply_along_axis(hsicRBFs_cause, 0, zp, K_x, sig_z)
        print("done distHSIC_cause", time.process_time() - start)  #
        fit_alpha_hsic_cause, fit_loc_hsic_cause, fit_beta_hsic_cause = stats.gamma.fit(distHSIC_cause)
        distSMMDb = onp.apply_along_axis(SMMDb_norm, 0, zp)
        fit_alpha_mmd, fit_loc_mmd, fit_beta_mmd = stats.gamma.fit(distSMMDb)
        print("done distMMD", time.process_time() - start)  #

        z_hat = getZ(params, optType, M, K_xy)
        for rep in range(reps):
            print("rep: ", rep)
            z_hat_rep = z_hat[:,rep][:,None]
            sig_x_f = np.exp(params['ln_sig_x_f'][rep])
            sig_z_f = np.exp(params['ln_sig_z_f'][rep])
            weightsResids = getWeightsResids(D_x, z_hat_rep, sig_x_f, sig_z_f, lam, y)
            resids = weightsResids[:,1][:,None]
            weights = weightsResids[:,0][:,None]
            distsR = covariance_matrix(sqeuclidean_distance, resids, resids)
            sig_r = 1 / np.median(distsR)
            std_err = np.std(resids)
            eps = std_err * onp.random.randn(num_smpl1, num_smpl2)
            yp = np.array([getYp(D_x, z_hat_rep, zp[:, i][:, None], sig_x_f, sig_z_f, eps[:, i][:, None], weights) for i in
                           range(zp.shape[1])]).T
            #weightsResids_p = np.array(
            #    [getWeightsResids(D_x, zp[:, i][:, None], sig_x_f, sig_z_f, lam, yp[:, i][:, None]) for i in
            #     range(yp.shape[1])])
            #weights_p = weightsResids_p[:, :, 0].T
            #resids_p = weightsResids_p[:, :, 1].T

            pvals = 1 - stats.gamma.cdf(monitors["hsic"][:,rep], a=fit_alpha_hsic_cause, loc=fit_loc_hsic_cause, scale=fit_beta_hsic_cause)
            monitors["hsic_pval"][:, rep] = pvals
            pvals = 1 - stats.gamma.cdf(monitors["MMDzn"][:,rep], a=fit_alpha_mmd, loc=fit_loc_mmd,
                                        scale=fit_beta_mmd)
            monitors["mmd_pval"][:, rep] = pvals
            #distHSIC_resids = np.array(
            #    [hsicRBFs_resids(K_x, zp[:, i][:, None], resids_p[:, i][:, None], sig_r) for i in range(zp.shape[1])])
            distHSIC_resids = np.array(
                [hsicRBFs_resids(K_x, zp[:, i][:, None], eps[:, i][:, None], sig_r) for i in range(zp.shape[1])])
            print("done distHSIC_resids", time.process_time() - start)  #

            fit_alpha_hsic_r, fit_loc_hsic_r, fit_beta_hsic_r = stats.gamma.fit(distHSIC_resids)
            pvals = 1 - stats.gamma.cdf(monitors["hsic_r"][:,rep], a=fit_alpha_hsic_r, loc=fit_loc_hsic_r,
                                        scale=fit_beta_hsic_r)
            monitors["hsic_r_pval"][:, rep] = pvals
            #distMSE = onp.apply_along_axis(np.sum, 0, resids_p * resids_p)
            #fit_alpha_mse, fit_loc_mse, fit_beta_mse = stats.gamma.fit(distMSE)
            #pvals = 1 - stats.gamma.cdf(monitors["errs"][:,rep], a=fit_alpha_mse, loc=fit_loc_mse,
            #                            scale=fit_beta_mse)
            #monitors["errs_pval"][:, rep] = pvals
        print("done p-val calc",time.process_time() - start)  #


    return params, monitors


def getLatentZs(X, nm, optType, lam, sig, beta, neta, eta, nu, lu, num_epochs, report_freq, num_reps, batch_size, learning_rate):
    print("nm:", nm)
    p = X.shape[1]
    N = X.shape[0]
    x = X[:,0][:,None]
    y = X[:,1][:,None]
    if p > 2:
        z = X[:,2:]
    else:
        z = None
    m = 5
    embedding = Isomap(n_components=m)
    z_mani = embedding.fit_transform(onp.hstack([x, y]))
    z_mani = np.array(z_mani)  # [:,None]
    z_mani = onp.apply_along_axis(normalize, 0, z_mani)
    #z_mani = stdrze_mat(z_mani)

    optType2 = optType.split("_")

    params_xy = getIniPar(optType2[0], N, m, num_reps, y, z_mani)
    params_yx = getIniPar(optType2[0], N, m, num_reps, x, z_mani)

    res_xy = []
    res_yx = []



    for ot in optType2:
        print("optType: ", ot)
        if (ot == "freeZ")|(ot == "freeZ-iniMani")|(optType == "freeZ-iniR"):
            loss_as_par = loss_freeZ
            dloss_as_par_jitted = dloss_freeZ_jitted
        elif (ot == "mani")|(ot == "mani-postZ"):
            loss_as_par = loss_mani
            dloss_as_par_jitted = dloss_mani_jitted
        elif ot == "hilb":
            loss_as_par = loss_hilb
            dloss_as_par_jitted = dloss_hilb_jitted

        print("x to y")
        start = time.process_time()
        params_xy, path_xy = getLatentZ(params_xy, loss_as_par, dloss_as_par_jitted, ot, x, y, z, z_mani, lam=lam, sig=sig, beta=beta, neta=neta, eta=eta, nu=nu, lu=lu,epochs=num_epochs, report_freq=report_freq,
                                                         reps=num_reps, batch_size=batch_size,learning_rate=learning_rate)
        print("done x->y", time.process_time() - start)  #
        print("y to x")
        start = time.process_time()
        params_yx, path_yx = getLatentZ(params_yx, loss_as_par, dloss_as_par_jitted, ot, y, x, z, z_mani, lam=lam, sig=sig, beta=beta, neta=neta, eta=eta, nu=nu, lu=lu, epochs=num_epochs, report_freq=report_freq,
                                                         reps=num_reps, batch_size=batch_size,learning_rate=learning_rate)
        print("done y->x", time.process_time() - start)  #
        res_xy.append(path_xy)
        res_yx.append(path_yx)

    res_xy = {k: onp.concatenate([r[k] for r in res_xy], axis=0) for k in res_xy[0].keys()}
    res_yx = {k: onp.concatenate([r[k] for r in res_yx], axis=0) for k in res_yx[0].keys()}

    print("results")
    res = {"params_xy": params_xy, "params_yx": params_yx, "path_xy": res_xy, "path_yx": res_yx}
    print(res["path_xy"])
    print(res["path_yx"])
    #"last_xy": last_xy,"last_yx": last_yx, "last_xy_gp": last_xy_gp, "last_yx_gp": last_yx_gp

    #print("hsic_alg")
    #print(res["path_xy"]["hsic_alg"][1, :])
    #print(res["path_yx"]["hsic_alg"][1, :])
    print("ent resids")
    n = res["path_xy"]["ent_r"].shape[0]-1
    print(res["path_xy"]["ent_r"][n, :] + res["path_xy"]["ent_c"][n, :])
    print(res["path_yx"]["ent_r"][n, :] + res["path_yx"]["ent_c"][n, :])
    print("ent alpha")
    print(res["path_xy"]["ent_alpha"][n, :])
    print(res["path_yx"]["ent_alpha"][n, :])

    return res


def getMsrsDirAdditive(x, y):
    x = x[:, None]
    y = y[:, None]
    gpKern = RBF() + WhiteKernel()
    gpModel = GaussianProcessRegressor(kernel=gpKern)
    gpModel.fit(x, y)
    yhat = gpModel.predict(x)
    resids = y - yhat

    distsX = covariance_matrix(sqeuclidean_distance, x, x)
    sigmaX = 1 / np.median(distsX)
    K_x = rbf_kernel_matrix({'gamma': sigmaX}, x, x)

    distsR = covariance_matrix(sqeuclidean_distance, resids, resids)
    sigma = 1 / np.median(distsR)
    K_r = rbf_kernel_matrix({'gamma': sigma}, resids, resids)

    # hsic resids
    hsic_resids = hsic(K_x, K_r)

    # entropy
    co1 = ite.cost.BHShannon_KnnK()  # initialize the entropy (2nd character = ’H’) estimator
    h_x = co1.estimation(x)  # entropy estimation
    h_r = co1.estimation(resids)

    # calculate norm
    monitor = {
        'mse': mse(y, y_hat),
        'hsic_r': hsic_resids,
        'h_x': h_x,
        'h_r': h_r,
    }

    return monitor

def getMsrsAdditive(x, y):
    res = {'x->y':getMsrsDirAdditive(x, y),
          'y->x':getMsrsDirAdditive(y, x)}
    return(res)

def norml(x):
    return (x-onp.min(x))/(onp.max(x)-onp.min(x))-0.5
def norml_mat(x):
    return onp.apply_along_axis(norml,0,x)

def stdrze(x):
    return (x-onp.mean(x))/onp.std(x)
def stdrze_mat(x):
    return onp.apply_along_axis(stdrze,0,x)


def jitter(X):
    N = X.shape[0]
    p = X.shape[1]
    nois_x = onp.random.randn(N,1)*0.001
    nois_y = onp.random.randn(N,1)*0.001
    x = X[:,0][:,None]+nois_x
    y = X[:,1][:,None]+nois_y
    res = onp.hstack([x,y])
    if p > 2:
        z = X[:,2:]
        res = onp.hstack([res, z])
    return res

@jax.jit
def normalize(m):
    o = np.argsort(m)
    mp = np.argsort(o)
    #sns.distplot(mp)
    min_mp = np.min(mp)-0.001
    max_mp = np.max(mp)+0.001
    mpp = (mp-min_mp)/(max_mp-min_mp)
    #sns.distplot(mpp)
    #print(np.min(mpp), np.max(mpp))
    mppp = ndtri(mpp)
    return(mppp)
