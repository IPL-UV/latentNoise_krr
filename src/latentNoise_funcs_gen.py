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
from sklearn.preprocessing import PowerTransformer

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

# cross-kernel version - obj and est the same
@jax.jit
def kernel_mat(D_x, x, z, sig_x_h, sig_z_h, sigs_f):
    sig_x_f = sigs_f[0]
    sig_z_f = sigs_f[1]
    sig_xz_f = sigs_f[2]
    sig_x_f2 = sigs_f[3]
    sig_z_f2 = sigs_f[4]
    K_x_f = np.exp(-sig_x_f * D_x)
    K_x_f2 = np.exp(-sig_x_f2 * D_x)
    K_z_f = rbf_kernel_matrix({'gamma': sig_z_f}, z, z)
    K_z_f2 = rbf_kernel_matrix({'gamma': sig_z_f2}, z, z)
    #K_xz_f = rbf_kernel_matrix({'gamma': sig_xz_f}, x, z)
    #K_zx_f = np.transpose(K_xz_f)

    K_x_h = K_x_f #np.exp(-sig_x_h * D_x)
    K_z_h = K_z_f #rbf_kernel_matrix({'gamma': sig_z_h}, z, z)
    #K_a_h = K_x_h * K_z_h
    #K_a_f = (2 * K_x_f + 2 * K_z_f + K_xz_f + K_zx_f) / 6
    #K_a_f = (2 * K_x_f + K_xz_f + K_zx_f) / 4
    K_a_f = (K_x_f2 + K_z_f2 + K_x_f*K_z_f) / 3
    #K_a_f = (K_xz_f+K_zx_f)/2
    #D = np.diag(np.diag(K_a_f))
    #n = D.shape[0]
    #I = np.diag(np.diag(np.ones((n,n))))
    #K_a_f = I + K_a_f - D
    K_a_h = K_a_f

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
    #L = linalg.cho_factor(K + lam * np.diag(1 / ws))
    #n = K.shape[0]
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

#@jax.jit
def hsic_sat(K_x, K_z, thrs):
    K_xz =  K_x * K_z
    rowSums = onp.apply_along_axis(onp.sum, 1,K_xz)
    orde = onp.argsort(rowSums)
    K_x = K_x[:,orde]
    K_x = K_x[orde,:]
    K_z = K_z[:,orde]
    K_z = K_z[orde,:]
    indx ,  = onp.where(onp.cumsum(rowSums[orde]/onp.sum(rowSums))>thrs)#[::-1])>thrs)
    satLevel = indx[0]#/rowSums.shape[0]
    K_x = K_x[0:satLevel, 0:satLevel]
    K_z = K_z[0:satLevel, 0:satLevel]
    hsicSat = hsic(K_x, K_z)
    return hsicSat



# lin Kernel
@jax.jit
def lin_kernel(x, y):
    return -sqeuclidean_distance(x, y)

def lin_kernel_matrix(x, y):
    mapx1 = jax.vmap(lambda x, y: lin_kernel(x, y), in_axes=(0, None), out_axes=0)
    mapx2 = jax.vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
    return mapx2(x, y)

@jax.jit
def hsicLIN_RBF(x, K_z):
    K_x = lin_kernel_matrix(x, x)
    K_x = centering(K_x)
    K_z = centering(K_z)
    return np.sum(K_x * K_z) / np.linalg.norm(K_x) / np.linalg.norm(K_z)



# obtain a null hypothesis (independence) hsic
@jax.jit
def permHsic(smp, K_x, K_z):
    K_xp = K_x[smp,:]
    K_xp = K_xp[:,smp]
    
    return hsic(K_xp, K_z)

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

#@jax.jit
def hsicRBFq(x, z, q_x, q_z):
    distsX = covariance_matrix(sqeuclidean_distance, np.unique(x), np.unique(x))
    distsX = onp.array(distsX)
    onp.fill_diagonal(distsX, onp.nan)
    sigma = 1/np.nanquantile(distsX, q_x)
    K_x = rbf_kernel_matrix({'gamma': sigma}, x, x)
    distsZ = covariance_matrix(sqeuclidean_distance, z, z)
    sigma = 1/np.quantile(distsZ, q_z)
    #print("sigma z: ", sigma)
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

#@jax.jit
def KCDC(x,y, lam):
    distsX = covariance_matrix(sqeuclidean_distance, x, x)
    sigma = 1.0 / np.median(distsX)
    K_x = rbf_kernel_matrix({'gamma': sigma}, x, x)
    distsY = covariance_matrix(sqeuclidean_distance, y, y)
    sigma = 1.0 / np.median(distsY)
    K_y = rbf_kernel_matrix({'gamma': sigma}, y, y)
    K_y = centering(K_y)
    n = K_x.shape[0]
    I = np.eye(n)
    Blambda = np.linalg.inv(K_x+n*lam*I)
    Alambda = np.dot(np.dot(Blambda, K_y),Blambda.T)
    LAL = K_x@Alambda@K_x.T
    dLAL = np.diag(LAL)
    indx, = np.where(dLAL<0)
    dLAL = index_update(dLAL, indx, 0)
    b = np.sum(dLAL**(0.5))
    c = np.sum(dLAL)
    res = (c/n) - (b/n)**2
    return res

# for comparing against no z vanilla model

def transformResids(resids, lam):
        pt = PowerTransformer()
        pt.fit(resids)
        pt.lambdas_= [lam]
        res = pt.transform(resids)
        #res = norml(res)
        return res

def model_van(lam, K_x, x, y):
    # find kernel stuffs
    n = K_x.shape[0]
    ws = np.ones(n)
    weights, resids, y_hat = krrModel(lam, K_x, y, ws)

    # right now we simply overwrite residual lengthscale sig_r_h to be median heuristic
    distsR = covariance_matrix(sqeuclidean_distance, resids, resids)
    distsR = onp.array(distsR)
    onp.fill_diagonal(distsR,onp.nan)
    sig_r_h = 1 / np.nanquantile(distsR, 0.5)
    q1 = 0.26
    q2 = 0.28
    q3 = 0.30
    q4 = 0.32
    q5 = 0.34
    q6 = 0.36
    q7 = 0.38
    q8 = 0.4
    q9 = 0.45
    sig_r_h_q1 = 1 / np.nanquantile(distsR, q1)
    sig_r_h_q2 = 1 / np.nanquantile(distsR, q2)
    sig_r_h_q3 = 1 / np.nanquantile(distsR, q3)
    sig_r_h_q4 = 1 / np.nanquantile(distsR, q4)
    sig_r_h_q5 = 1 / np.nanquantile(distsR, q5)
    sig_r_h_q6 = 1 / np.nanquantile(distsR, q6)
    sig_r_h_q7 = 1 / np.nanquantile(distsR, q7)
    sig_r_h_q8 = 1 / np.nanquantile(distsR, q8)
    sig_r_h_q9 = 1 / np.nanquantile(distsR, q9)

    K_r = rbf_kernel_matrix({'gamma': sig_r_h}, resids, resids)
    K_r_q1 = rbf_kernel_matrix({'gamma': sig_r_h_q1}, resids, resids)
    K_r_q2 = rbf_kernel_matrix({'gamma': sig_r_h_q2}, resids, resids)
    K_r_q3 = rbf_kernel_matrix({'gamma': sig_r_h_q3}, resids, resids)
    K_r_q4 = rbf_kernel_matrix({'gamma': sig_r_h_q4}, resids, resids)
    K_r_q5 = rbf_kernel_matrix({'gamma': sig_r_h_q5}, resids, resids)
    K_r_q6 = rbf_kernel_matrix({'gamma': sig_r_h_q6}, resids, resids)
    K_r_q7 = rbf_kernel_matrix({'gamma': sig_r_h_q7}, resids, resids)
    K_r_q8 = rbf_kernel_matrix({'gamma': sig_r_h_q8}, resids, resids)
    K_r_q9 = rbf_kernel_matrix({'gamma': sig_r_h_q9}, resids, resids)


    # hsic
    hsic_resids_x = hsic(K_x, K_r)
    hsic_resids_x_q1 = hsic(K_x, K_r_q1)
    hsic_resids_x_q2 = hsic(K_x, K_r_q2)
    hsic_resids_x_q3 = hsic(K_x, K_r_q3)
    hsic_resids_x_q4 = hsic(K_x, K_r_q4)
    hsic_resids_x_q5 = hsic(K_x, K_r_q5)
    hsic_resids_x_q6 = hsic(K_x, K_r_q6)
    hsic_resids_x_q7 = hsic(K_x, K_r_q7)
    hsic_resids_x_q8 = hsic(K_x, K_r_q8)
    hsic_resids_x_q9 = hsic(K_x, K_r_q9)
   

    p = 1000
    smpl = onp.random.randint(low=0, high=n, size=(n, p))
    hsic_resids_x_null = onp.apply_along_axis(permHsic, 0, smpl, K_x=K_x, K_z=K_r)
    print("null dist shape : ",hsic_resids_x_null.shape)
    

    gamma_alpha, gamma_loc, gamma_beta=stats.gamma.fit(hsic_resids_x_null)  
    ln_shape, ln_loc, ln_scale = stats.lognorm.fit(hsic_resids_x_null)
    gamma_pars = {"alpha": gamma_alpha, "loc": gamma_loc, "beta":gamma_beta}
    ln_pars = {"shape": ln_shape, "loc": ln_loc, "scale":ln_scale}
    print("ln_pars:", ln_pars)


    pval_ln= 1-stats.lognorm.cdf(hsic_resids_x, s=ln_pars["shape"], loc=ln_pars["loc"], scale=ln_pars["scale"])
    
    hsic_resids_x_null = onp.mean(hsic_resids_x_null)

    #robust hsic
    hsicRob_resids_x = hsic_sat(K_x, K_r, 0.95)

    # order of dependence
    lams = onp.linspace(0,10,20)
    hsics_rx_byOrd = onp.array([hsicLIN_RBF(transformResids(resids, lam), K_x) for lam in lams])
    hsic_ordOpt = onp.argmax(hsics_rx_byOrd)
    hsic_ord = onp.max(hsics_rx_byOrd)


    # entropy
    co1 = ite.cost.BHShannon_KnnK()  # initialize the entropy (2nd character = ’H’) estimator
    #co1 = ite.cost.BHShannon_KnnK(knn_method="cKDTree", k=2, eps=0.1)
    h_x = co1.estimation(x)  # entropy estimation
    #h_r = co1.estimation(resids)
    
    h_x = onp.array(funcs_r["Shannon_KDP"](onp.array(x)))[0]
    h_r = onp.array(funcs_r["Shannon_KDP"](onp.array(resids)))[0]

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
        'hsic_rx_q1': hsic_resids_x_q1,
        'hsic_rx_q2': hsic_resids_x_q2,
        'hsic_rx_q3': hsic_resids_x_q3,
        'hsic_rx_q4': hsic_resids_x_q4,
        'hsic_rx_q5': hsic_resids_x_q5,
        'hsic_rx_q6': hsic_resids_x_q6,
        'hsic_rx_q7': hsic_resids_x_q7,
        'hsic_rx_q8': hsic_resids_x_q8,
        'hsic_rx_q9': hsic_resids_x_q9,
        'hsicRob_rx': hsicRob_resids_x, 
        'hsic_rx_null': hsic_resids_x_null,
        'pval_rx': pval_ln,
        'hsic_ord':hsic_ord,
        'hsic_ordOpt':hsic_ordOpt,
        'h_x': h_x,
        'h_r': h_r,
        'cost_slope': cost_slope,
        'cost_slope_krr': cost_slope_krr,
    }

    return monitor

def model_van_GP(K_x, x, y):
    # find kernel stuffs
    n = K_x.shape[0]
    ws = np.ones(n)
    
    input_x = onp.array(x)
    output = onp.array(y)
    gpKern = RBF()+WhiteKernel()
    gpModel = GaussianProcessRegressor(kernel=gpKern)
    gpModel.fit(input_x, output)
    y_hat = gpModel.predict(input_x)
    resids = y-y_hat

    # right now we simply overwrite residual lengthscale sig_r_h to be median heuristic
    distsR = covariance_matrix(sqeuclidean_distance, resids, resids)
    distsR = onp.array(distsR)
    onp.fill_diagonal(distsR,onp.nan)
    sig_r_h = 1 / np.nanquantile(distsR, 0.5)
    q1 = 0.26
    q2 = 0.28
    q3 = 0.30
    q4 = 0.32
    q5 = 0.34
    q6 = 0.36
    q7 = 0.38
    q8 = 0.4
    q9 = 0.45
    sig_r_h_q1 = 1 / np.nanquantile(distsR, q1)
    sig_r_h_q2 = 1 / np.nanquantile(distsR, q2)
    sig_r_h_q3 = 1 / np.nanquantile(distsR, q3)
    sig_r_h_q4 = 1 / np.nanquantile(distsR, q4)
    sig_r_h_q5 = 1 / np.nanquantile(distsR, q5)
    sig_r_h_q6 = 1 / np.nanquantile(distsR, q6)
    sig_r_h_q7 = 1 / np.nanquantile(distsR, q7)
    sig_r_h_q8 = 1 / np.nanquantile(distsR, q8)
    sig_r_h_q9 = 1 / np.nanquantile(distsR, q9)

    K_r = rbf_kernel_matrix({'gamma': sig_r_h}, resids, resids)
    K_r_q1 = rbf_kernel_matrix({'gamma': sig_r_h_q1}, resids, resids)
    K_r_q2 = rbf_kernel_matrix({'gamma': sig_r_h_q2}, resids, resids)
    K_r_q3 = rbf_kernel_matrix({'gamma': sig_r_h_q3}, resids, resids)
    K_r_q4 = rbf_kernel_matrix({'gamma': sig_r_h_q4}, resids, resids)
    K_r_q5 = rbf_kernel_matrix({'gamma': sig_r_h_q5}, resids, resids)
    K_r_q6 = rbf_kernel_matrix({'gamma': sig_r_h_q6}, resids, resids)
    K_r_q7 = rbf_kernel_matrix({'gamma': sig_r_h_q7}, resids, resids)
    K_r_q8 = rbf_kernel_matrix({'gamma': sig_r_h_q8}, resids, resids)
    K_r_q9 = rbf_kernel_matrix({'gamma': sig_r_h_q9}, resids, resids)


    # hsic
    hsic_resids_x = hsic(K_x, K_r)
    hsic_resids_x_q1 = hsic(K_x, K_r_q1)
    hsic_resids_x_q2 = hsic(K_x, K_r_q2)
    hsic_resids_x_q3 = hsic(K_x, K_r_q3)
    hsic_resids_x_q4 = hsic(K_x, K_r_q4)
    hsic_resids_x_q5 = hsic(K_x, K_r_q5)
    hsic_resids_x_q6 = hsic(K_x, K_r_q6)
    hsic_resids_x_q7 = hsic(K_x, K_r_q7)
    hsic_resids_x_q8 = hsic(K_x, K_r_q8)
    hsic_resids_x_q9 = hsic(K_x, K_r_q9)
   

    p = 1000
    smpl = onp.random.randint(low=0, high=n, size=(n, p))
    hsic_resids_x_null = onp.apply_along_axis(permHsic, 0, smpl, K_x=K_x, K_z=K_r)
    print("null dist shape : ",hsic_resids_x_null.shape)
    

    gamma_alpha, gamma_loc, gamma_beta=stats.gamma.fit(hsic_resids_x_null)  
    ln_shape, ln_loc, ln_scale = stats.lognorm.fit(hsic_resids_x_null)
    gamma_pars = {"alpha": gamma_alpha, "loc": gamma_loc, "beta":gamma_beta}
    ln_pars = {"shape": ln_shape, "loc": ln_loc, "scale":ln_scale}
    print("ln_pars:", ln_pars)


    pval_ln= 1-stats.lognorm.cdf(hsic_resids_x, s=ln_pars["shape"], loc=ln_pars["loc"], scale=ln_pars["scale"])
    
    hsic_resids_x_null = onp.mean(hsic_resids_x_null)

    #robust hsic
    hsicRob_resids_x = hsic_sat(K_x, K_r, 0.95)

    # order of dependence
    lams = onp.linspace(0,10,20)
    hsics_rx_byOrd = onp.array([hsicLIN_RBF(transformResids(resids, lam), K_x) for lam in lams])
    hsic_ordOpt = onp.argmax(hsics_rx_byOrd)
    hsic_ord = onp.max(hsics_rx_byOrd)


    # entropy
    co1 = ite.cost.BHShannon_KnnK()  # initialize the entropy (2nd character = ’H’) estimator
    #co1 = ite.cost.BHShannon_KnnK(knn_method="cKDTree", k=2, eps=0.1)
    h_x = co1.estimation(x)  # entropy estimation
    #h_r = co1.estimation(resids)
    
    h_x = onp.array(funcs_r["Shannon_KDP"](onp.array(x)))[0]
    h_r = onp.array(funcs_r["Shannon_KDP"](onp.array(resids)))[0]

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
    cost_slope_krr = myComputeScoreKrr(K_x, sigma, target, rows, mindiff, mindiff_x, 0.1)

    # calculate norm
    penalize = 0 #np.linalg.norm(weights.T @ K_x @ weights)
    monitor = {}
    monitor = {
        'mse': mse(y, y_hat),
        'penalty': penalize,
        'hsic_rx': hsic_resids_x,
        'hsic_rx_q1': hsic_resids_x_q1,
        'hsic_rx_q2': hsic_resids_x_q2,
        'hsic_rx_q3': hsic_resids_x_q3,
        'hsic_rx_q4': hsic_resids_x_q4,
        'hsic_rx_q5': hsic_resids_x_q5,
        'hsic_rx_q6': hsic_resids_x_q6,
        'hsic_rx_q7': hsic_resids_x_q7,
        'hsic_rx_q8': hsic_resids_x_q8,
        'hsic_rx_q9': hsic_resids_x_q9,
        'hsicRob_rx': hsicRob_resids_x, 
        'hsic_rx_null': hsic_resids_x_null,
        'pval_rx': pval_ln,
        'hsic_ord':hsic_ord,
        'hsic_ordOpt':hsic_ordOpt,
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

    if (p > 2) & False:
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


def getModels_van_GP(X):
    print("enters getModels_van")
    print("X.shape", X.shape)
    x = X[:, 0][:, None]
    y = X[:, 1][:, None]
    N = x.shape[0]
    p = X.shape[1]

    sig = [0.5]
    distsX = covariance_matrix(sqeuclidean_distance, x, x)
    #sigmaX = 1 / np.median(distsX)
    sigmaX = 1 / np.quantile(distsX, sig[0])
    K_x = rbf_kernel_matrix({'gamma': sigmaX}, x, x)
    distsY = covariance_matrix(sqeuclidean_distance, y, y)
    #sigmaY = 1 / np.median(distsY)
    sigmaY = 1 / np.quantile(distsY, sig[0])
    K_y = rbf_kernel_matrix({'gamma': sigmaY}, y, y)

    monitor_xy = model_van_GP(K_x, x, y)
    monitor_yx = model_van_GP(K_y, y, x)

    res = {"xy": monitor_xy, "yx": monitor_yx}

    if (p > 2) & False:
        print("enters z section")
        z = X[:, 2:]
        distsZ = covariance_matrix(sqeuclidean_distance, z, z)
        sigmaZ = 1 / np.median(distsZ)
        K_z = rbf_kernel_matrix({'gamma': sigmaZ}, z, z)
        K_cx = K_x*K_z
        K_cy = K_y * K_z
        caus_x = np.hstack([x, z])
        caus_y = np.hstack([y, z])
        monitor_cy = model_van_GP(K_cx, caus_x, y)
        monitor_cx = model_van_GP(K_cy, caus_y, x)
        res["cy"] = monitor_cy
        res["cx"] = monitor_cx
    print("exits getModels_van")
    return res

# for reporting purposes give back 3 terms separatley
def model(optType, params, lam, D_x, x, y, z, M, K_zmani, K_y):
    # find kernel stuffs
    #z_hat = getZ(params, optType, M, K_xy)
    #z_hat = getZ(params, M, K_xy)
    z_hat = getZ(params)

    #print("num nans: ", onp.sum(onp.isnan(onp.array(z_hat))))
    #print("num infs: ", onp.sum(onp.isinf(onp.array(z_hat))))


    z_hat = stdrze_mat(z_hat)
    sig_x_h = np.exp(params["ln_sig_x_h"])
    sig_z_h = np.exp(params["ln_sig_z_h"])
    
    

    # for now use median heuristic for reporting hsic values of loss. Later we will want to
    # use the sigs being used in the loss function (as they move up) and perhaps also report
    # various "hsic views" with pre-determined quantiles to show how annealing works
    # distsZ = covariance_matrix(sqeuclidean_distance, z, z)
    # sig_z_h = 1/np.quantile(distsZ, 0.5)

    sig_z_f = np.exp(params["ln_sig_z_f"])
    sig_x_f = np.exp(params["ln_sig_x_f"])
    sig_z_f2 = np.exp(params["ln_sig_z_f2"])
    sig_x_f2 = np.exp(params["ln_sig_x_f2"])
    sig_xz_f = np.exp(params["ln_sig_xz_f"])
    sig_z_f2 = np.exp(params["ln_sig_z_f2"])


    #sigs_f = np.hstack([sig_x_f, sig_z_f, sig_xz_f])
    sigs_f = np.hstack([sig_x_f, sig_z_f, sig_xz_f, sig_x_f2, sig_z_f2])
    K_a_f, K_a_h, K_x_h, K_z_h = kernel_mat(D_x, x, z_hat, sig_x_h, sig_z_h, sigs_f)

    n = K_a_f.shape[0]
    ws = np.ones(n)
    weights, resids, y_hat = krrModel(lam, K_a_f, y, ws)

    # right now we simply overwrite residual lengthscale sig_r_h to be median heuristic
    distsR = covariance_matrix(sqeuclidean_distance, resids, resids)
    #sig_r_h = np.exp(params["ln_sig_r_h"])
    #sig_r_h_q = params["sig_r_h_q"]    
    sig_r_h = 1 / np.quantile(distsR, 0.5)
    #sig_r_h = 1 / np.quantile(distsR, sig_r_h_q)
    
    
    K_r_h = rbf_kernel_matrix({'gamma': sig_r_h}, resids, resids)

    # hsic
    hsic_val = hsic(K_x_h, K_z_h)
    # hsic resids
    hsic_resids_x = hsic(K_x_h, K_r_h)
    hsic_resids_z = hsic(K_z_h, K_r_h)
    hsic_resids_c = hsic(K_a_h, K_r_h)
    # hsic zmani
    hsic_zmani = hsic(K_z_h, K_zmani)
    #hsic_zmani = hsic(K_z_h, K_y)
    # dependence to true z
    if z is not None:
        hsic_zzhat = hsicRBF(z, z_hat)
    else:
        hsic_zzhat = None

    # algorithmic dependency
    #hsic_alg = hsicRBF(W, Yhat)

    #MMD
    MMDzn = SMMDb_norm(z_hat)

    #xjit = jitterByDist2(x[:,0])[:,None]
    #xun, indx, numReps = onp.unique(xjit, return_counts=True, return_inverse=True)
    #residsjit = jitterByDist2(resids[:,0])[:,None]
    #z_hatjit = jitterByDist2(z_hat[:,0])[:,None]	    

    z2 = norml(z_hat)
    #z2 = norml(z_hatjit)
    
    # entropy
    co1 = ite.cost.BHShannon_KnnK()  # initialize the entropy (2nd character = ’H’) estimator
    #caus = np.hstack([x, z2])
    caus = np.hstack([x, z2])
    h_z = co1.estimation(z2)
    #h_x = co1.estimation(x)
    h_x = co1.estimation(x)
    h_c = co1.estimation(caus)  # entropy estimation
    #h_r = co1.estimation(resids)
    h_r = co1.estimation(resids)
    #h_a = co1.estimation(weights)

    #h_x2 = onp.array(funcs_r["Shannon_KDP"](onp.array(x)))[0]
    #h_c2 = onp.array(funcs_r["Shannon_KDP"](onp.array(caus)))[0]
    #h_z2 = onp.array(funcs_r["Shannon_KDP"](onp.array(z2)))[0]
    #h_r2 = onp.array(funcs_r["Shannon_KDP"](onp.array(resids)))[0]
    
    #k = 5
    #h_x2 = onp.array(funcs_r["Shannon_Edgeworth"](onp.array(x)))[0]
    #h_c2 = onp.array(funcs_r["Shannon_Edgeworth"](onp.array(caus)))[0]
    #h_z2 = onp.array(funcs_r["Shannon_Edgeworth"](onp.array(z2)))[0]
    #h_r2 = onp.array(funcs_r["Shannon_Edgeworth"](onp.array(resids)))[0]

    #run, indx, numReps = onp.unique(resids, return_counts=True, return_inverse=True)
    #print("num reps resids: " ,onp.sum(numReps>1))

    #print("h_x: ", h_x)
    #print("h_z: ", h_z)
    #print("h_r: ", h_r)

    # slope score
    #source = onp.array(caus)
    #target = onp.array(y)
    #rows = y.shape[0]
    #mindiff = CalculateMinDiff(y)
    #mindiff_x = CalculateMinDiff(x)
    #k = onp.array([2])
    #V = 3
    #M = 2
    #F = 9
    #cost_slope, _ = myComputeScore(source, target, rows, mindiff, mindiff_x, k, V, M, F)
    #distsX = covariance_matrix(sqeuclidean_distance, x, x)
    #sigma = 1 / np.median(distsX)
    #cost_slope_krr = myComputeScoreKrr(K_a_f, sigma, target, rows, mindiff, mindiff_x, lam)
    #cost_slope_z = onp.sum([onp.sum(onp.array([model_score(onp.array([float(params['Z'][i, j])])) for i in range(params['Z'].shape[0])])) for j in range(params['Z'].shape[1])])
    #mindiff_z = CalculateMinDiff(onp.array(z_hat).flatten())
    #cost_slope_z = - rows * logg(mindiff_z)

    # calculate norm
    #penalize = np.linalg.norm(weights.T @ K_a_f @ weights)
    monitor = {}
    monitor = {
        'resids': resids,
        'mse': mse(y, y_hat),
        'penalty': None, #penalize,
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
        #'h_xx2': None,
        #'h_z2': h_z2,
        #'h_x2': h_c2,
        #'h_r2': h_r2,
        'h_a': None,#h_a,
        'cost_slope': None,#cost_slope,
        'cost_slope_krr': None,#cost_slope_krr,
        'cost_slope_z': None,#cost_slope_z,
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
def loss_freeZ(params, beta, neta, eta, lam, nu, lu, D_x, x, y, K_y, M, K_zmani, ws):
    sig_x_h = np.exp(params["ln_sig_x_h"])
    sig_z_h = np.exp(params["ln_sig_z_h"])
    sig_r_h = np.exp(params["ln_sig_r_h"])
    sig_r_h_q = params["sig_r_h_q"]


    z = params["Z"]
    z = (z - np.mean(z)) / (np.std(z))

    sig_x_f = np.exp(params["ln_sig_x_f"])
    sig_z_f = np.exp(params["ln_sig_z_f"])
    sig_xz_f = np.exp(params["ln_sig_xz_f"])
    sig_x_f2 = np.exp(params["ln_sig_x_f2"])    
    sig_z_f2 = np.exp(params["ln_sig_z_f2"])
    #sigs_f = np.hstack([sig_x_f, sig_z_f, sig_xz_f])
    sigs_f = np.hstack([sig_x_f, sig_z_f, sig_xz_f, sig_x_f2, sig_z_f2])

    K_a_f, K_a_h, K_x_h, K_z_h = kernel_mat(D_x, x, z, sig_x_h, sig_z_h, sigs_f)
    n = K_a_f.shape[0]

    weights, resids, y_hat = krrModel(lam, K_a_f, y, ws)

    # right now we simply overwrite residual lengthscale sig_r_h to be median heuristic
    distsR = covariance_matrix(sqeuclidean_distance, resids, resids)
    sig_r_h = 1 / np.quantile(distsR, 0.5)
    #sig_r_h = 1 / np.quantile(distsR, sig_r_h_q)


    K_r_h = rbf_kernel_matrix({'gamma': sig_r_h}, resids, resids)

    # hsic
    hsic_val = hsic(K_x_h, K_z_h)

    # hsic resids
    hsic_resids = hsic(K_a_h, K_r_h)

    # hsic zmani
    hsic_zmani = hsic(K_z_h, K_zmani)
    #hsic_zmani = hsic(K_z_h, K_y)

    # MMD
    MMDzn = SMMDb_norm(z)
    correctFactor = 1  # jax_stats.norm.cdf(MMDzn, loc=-0.1447, scale=1.321)
    MMDzn = MMDzn * correctFactor

    # calcualte compute loss
    #loss_value = np.log(hsic_resids+0.1) + beta * np.log(mse(y, y_hat)+0.1)  + n * neta * np.log(hsic_val+0.1)  - n * eta * np.log(hsic_zmani+0.01) + nu*np.log(np.dot(z.T, z)[1,1]+0.1) + lu*np.log(MMDzn+0.1)    
    #loss_value = np.log(hsic_resids) + beta * np.log(mse(y, y_hat))  + n * neta * np.log(hsic_val)  - n * eta * np.log(hsic_zmani) + nu*np.log(np.dot(z.T, z)[1,1]) + lu*np.log(MMDzn)
    loss_value = np.log(hsic_resids) + beta * np.log(mse(y, y_hat))  + neta * np.log(hsic_val)  - eta * np.log(hsic_zmani) + nu*np.log(np.dot(z.T, z)[1,1])/n + lu*np.log(MMDzn)
    
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

#@jax.jit
def getIniPar(reps, y): #N, m, reps, y, M
    #onp.random.seed(seed=4)
    z_ini = y * np.ones(reps)
    z_ini = np.array(onp.apply_along_axis(normalize, 0, z_ini))
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

@jax.jit
def getIniZ(params, M, K_xy):

    #n = K_xy.shape[0]
    #m = M.shape[1]

    z_ini = params["Z"]

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


@jax.jit
def getZ(params): #, M, K_xy
    z_ini = params["Z"]

    return z_ini


 # get parameters for grad - getParamsForGrad(params, rep, optType, smpl)

def getParamsForGrad(params, rep, optType, smpl):

    ln_sig_x_f_aux = params["ln_sig_x_f"][rep]
    ln_sig_z_f_aux = params["ln_sig_z_f"][rep]
    ln_sig_xz_f_aux = params["ln_sig_xz_f"][rep]
    ln_sig_z_f2_aux = params["ln_sig_z_f2"][rep]
    ln_sig_x_f2_aux = params["ln_sig_x_f2"][rep]

    ln_sig_x_h_aux = params["ln_sig_x_h"][rep]
    ln_sig_z_h_aux = params["ln_sig_z_h"][rep]
    ln_sig_r_h_aux = params["ln_sig_r_h"][rep]
    sig_r_h_q_aux = params["sig_r_h_q"][rep]
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
    params_aux['ln_sig_z_f2'] = ln_sig_z_f2_aux
    params_aux['ln_sig_x_f2'] = ln_sig_x_f2_aux

    params_aux['ln_sig_x_h'] = ln_sig_x_h_aux
    params_aux['ln_sig_z_h'] = ln_sig_z_h_aux
    params_aux['ln_sig_r_h'] = ln_sig_r_h_aux
    params_aux['sig_r_h_q'] = sig_r_h_q_aux

    return params_aux


@jax.jit
def getParamsForGrad(params, rep, smpl):

    ln_sig_x_f_aux = params["ln_sig_x_f"][rep]
    ln_sig_z_f_aux = params["ln_sig_z_f"][rep]
    ln_sig_xz_f_aux = params["ln_sig_xz_f"][rep]
    ln_sig_z_f2_aux = params["ln_sig_z_f2"][rep]
    ln_sig_x_f2_aux = params["ln_sig_x_f2"][rep]

    ln_sig_x_h_aux = params["ln_sig_x_h"][rep]
    ln_sig_z_h_aux = params["ln_sig_z_h"][rep]
    ln_sig_r_h_aux = params["ln_sig_r_h"][rep]
    sig_r_h_q_aux = params["sig_r_h_q"][rep]
    params_aux = params.copy()

    z_aux = params['Z'][:, rep]
    z_aux = z_aux[smpl,][:,None]
    params_aux['Z'] = z_aux
    
    params_aux['ln_sig_x_f'] = ln_sig_x_f_aux
    params_aux['ln_sig_z_f'] = ln_sig_z_f_aux
    params_aux['ln_sig_xz_f'] = ln_sig_xz_f_aux
    params_aux['ln_sig_z_f2'] = ln_sig_z_f2_aux
    params_aux['ln_sig_x_f2'] = ln_sig_x_f2_aux

    params_aux['ln_sig_x_h'] = ln_sig_x_h_aux
    params_aux['ln_sig_z_h'] = ln_sig_z_h_aux
    params_aux['ln_sig_r_h'] = ln_sig_r_h_aux
    params_aux['sig_r_h_q'] = sig_r_h_q_aux

    return params_aux


# update params - updateParams(params, grad_params, optType, smpl, rep)
def updateParams(params, grad_params, optType, smpl, iteration, rep, learning_rate):

    if (optType == "freeZ")|(optType == "freeZ-iniMani")|(optType == "freeZ-iniR"):
        idx_rows = smpl[:, None]
        idx_cols = np.array(rep)[None, None]
        idx = jax.ops.index[tuple([idx_rows, idx_cols])]
        A = params['Z'][tuple([idx_rows, idx_cols])]
        B = learning_rate * grad_params['Z']
        #if onp.sum(onp.isnan(B))==0:
        #	    params['Z'] = index_update(params['Z'], idx, A - B)
        params['Z'] = index_update(params['Z'], idx, A - B)
        n = params['Z'].shape[0]
        idx_rows2 = np.linspace(0, n - 1, n, dtype=int)
        idx2 = jax.ops.index[tuple([idx_rows2, idx_cols])]
        if ((iteration + 1) % 100) == 0:
            params["Z"] = index_update(params['Z'], idx2, normalize(params['Z'][:, rep]))

        if onp.sum(onp.isnan(B))>0:
            idx_nan, _ = onp.where(onp.isnan(B))
            print("nans in grad Z, iteration: ", iteration, " rep: ", rep)
            raise ValueError('Nans in gradient.')
            

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
    # 	                           params['ln_sig_x_f'][rep] - learning_rate * grad_params['ln_sig_x_f'])
            
    #params['ln_sig_z_f'] = index_update(params['ln_sig_z_f'], rep,
    #                           params['ln_sig_z_f'][rep] - learning_rate * grad_params['ln_sig_z_f'])

    #params['ln_sig_xz_f'] = index_update(params['ln_sig_xz_f'], rep,
    #                           params['ln_sig_xz_f'][rep] - learning_rate * grad_params['ln_sig_xz_f'])

    #params['ln_sig_x_f2'] = index_update(params['ln_sig_x_f2'], rep,
    #                           params['ln_sig_x_f2'][rep] - learning_rate * grad_params['ln_sig_x_f2'])

    #params['ln_sig_z_f2'] = index_update(params['ln_sig_z_f2'], rep,
    #                           params['ln_sig_z_f2'][rep] - learning_rate * grad_params['ln_sig_z_f2'])

    #params['ln_sig_x_h'] = index_update(params['ln_sig_x_h'], rep,
    # 	                           params['ln_sig_x_h'][rep] - learning_rate * grad_params['ln_sig_x_h'])
            
    #params['ln_sig_z_h'] = index_update(params['ln_sig_z_h'], rep,
    #                           params['ln_sig_z_h'][rep] - learning_rate * grad_params['ln_sig_z_h'])

    #params['ln_sig_r_h'] = index_update(params['ln_sig_r_h'], rep,
    #                           params['ln_sig_r_h'][rep] - learning_rate * grad_params['ln_sig_r_h'])

    if onp.sum(onp.isnan(grad_params['ln_sig_x_f'])) > 0:
        print("nans in grad ln_sig_x_f, iteration: ", iteration, " rep: ", rep)
    if onp.sum(onp.isnan(grad_params['ln_sig_z_f'])) > 0:
        print("nans in grad ln_sig_z_f, iteration: ", iteration, " rep: ", rep)
    if onp.sum(onp.isnan(grad_params['ln_sig_xz_f'])) > 0:
        print("nans in grad ln_sig_xz_f, iteration: ", iteration, " rep: ", rep)

    #print("iteration:", iteration, "onp.max(Z): ", onp.max(params["Z"]) )
    #print("iteration:", iteration, "onp.min(Z): ", onp.min(params["Z"]) )
    #print("iteration:", iteration, "onp.max(gradZ): ", onp.max(grad_params["Z"]) )
    #print("iteration:", iteration, "onp.min(gradZ): ", onp.min(grad_params["Z"]) )
    #print("iteration:", iteration, "sig_x_f: ", onp.exp(params["ln_sig_x_f"]) )
    #print("iteration:", iteration, "sig_z_f: ", onp.exp(params["ln_sig_z_f"]) )
    #print("iteration:", iteration, "sig_xz_f: ", onp.exp(params["ln_sig_xz_f"]) )
    #print("iteration:", iteration, "grad_ln_sig_x_f: ", grad_params["ln_sig_x_f"] )
    #print("iteration:", iteration, "grad_ln_sig_z_f: ", grad_params["ln_sig_z_f"] )
    #print("iteration:", iteration, "grad_ln_sig_xz_f: ", grad_params["ln_sig_xz_f"] )

    return None

def updateParams(params, grad_params, optType, smpl, iteration, rep, learning_rate):

    idx_rows = smpl[:, None]
    idx_cols = np.array(rep)[None, None]
    idx = jax.ops.index[tuple([idx_rows, idx_cols])]
    A = params['Z'][tuple([idx_rows, idx_cols])]
    B = learning_rate * grad_params['Z']
    #if (onp.sum(onp.isnan(B))==0) & (onp.sum(onp.isinf(B))==0) & (onp.sum(onp.isnan([grad_params['ln_sig_x_f'], grad_params['ln_sig_x_f2'], grad_params['ln_sig_z_f'], grad_params['ln_sig_z_f2'],grad_params['ln_sig_xz_f'], grad_params[' ln_sig_x_h'], grad_params['ln_sig_z_h'], grad_params['ln_sig_r_h']]))==0) & (onp.sum(onp.isinf([grad_params['ln_sig_x_f'], grad_params['ln_sig_x_f2'], grad_params['ln_sig_z_f'], grad_params['ln_sig_z_f2'],grad_params['ln_sig_xz_f'], grad_params['ln_sig_x_h'], grad_params['ln_sig_z_h'], grad_params['ln_sig_r_h']]))==0):
    if True:
    	    params['Z'] = index_update(params['Z'], idx, A - B)
    
    n = params['Z'].shape[0]
    idx_rows2 = np.linspace(0, n - 1, n, dtype=int)
    idx2 = jax.ops.index[tuple([idx_rows2, idx_cols])]
    if ((iteration + 1) % 100) == 0:
        params["Z"] = index_update(params['Z'], idx2, normalize(params['Z'][:, rep]))

    if (onp.sum(onp.isnan(B))!=0) | (onp.sum(onp.isinf(B))!=0) | (onp.sum(onp.isnan([grad_params['ln_sig_x_f'], grad_params['ln_sig_x_f2'], grad_params['ln_sig_z_f'], grad_params['ln_sig_z_f2'],grad_params['ln_sig_xz_f'], grad_params['ln_sig_x_h'], grad_params['ln_sig_z_h'], grad_params['ln_sig_r_h']]))!=0) | (onp.sum(onp.isinf([grad_params['ln_sig_x_f'], grad_params['ln_sig_x_f2'], grad_params['ln_sig_z_f'], grad_params['ln_sig_z_f2'],grad_params['ln_sig_xz_f'], grad_params['ln_sig_x_h'], grad_params['ln_sig_z_h'], grad_params['ln_sig_r_h']]))!=0):
        idx_nan, _ = onp.where(onp.isnan(B))
        print("nans in grad Z, iteration: ", iteration, " rep: ", rep)
        raise ValueError('Nans in gradient.')
            
    
    #if (onp.sum(onp.isnan(B))==0) & (onp.sum(onp.isinf(B))==0) & (onp.sum(onp.isnan([grad_params['ln_sig_x_f'], grad_params['ln_sig_x_f2'], grad_params['ln_sig_z_f'], grad_params['ln_sig_z_f2'],grad_params['ln_sig_xz_f'], grad_params[' ln_sig_x_h'], grad_params['ln_sig_z_h'], grad_params['ln_sig_r_h']]))==0) & (onp.sum(onp.isinf([grad_params['ln_sig_x_f'], grad_params['ln_sig_x_f2'], grad_params['ln_sig_z_f'], grad_params['ln_sig_z_f2'],grad_params['ln_sig_xz_f'], grad_params['ln_sig_x_h'], grad_params['ln_sig_z_h'], grad_params['ln_sig_r_h']]))==0):
    if False:
    
        params['ln_sig_x_f'] = index_update(params['ln_sig_x_f'], rep,
     	                           params['ln_sig_x_f'][rep] - learning_rate * grad_params['ln_sig_x_f'])
            
        params['ln_sig_z_f'] = index_update(params['ln_sig_z_f'], rep,
                               params['ln_sig_z_f'][rep] - learning_rate * grad_params['ln_sig_z_f'])

        #params['ln_sig_xz_f'] = index_update(params['ln_sig_xz_f'], rep,
        #                           params['ln_sig_xz_f'][rep] - learning_rate * grad_params['ln_sig_xz_f'])

        #params['ln_sig_x_f2'] = index_update(params['ln_sig_x_f2'], rep,
        #                           params['ln_sig_x_f2'][rep] - learning_rate * grad_params['ln_sig_x_f2'])

        #params['ln_sig_z_f2'] = index_update(params['ln_sig_z_f2'], rep,
        #                           params['ln_sig_z_f2'][rep] - learning_rate * grad_params['ln_sig_z_f2'])

        #params['ln_sig_x_h'] = index_update(params['ln_sig_x_h'], rep,
        #	 	                           params['ln_sig_x_h'][rep] - learning_rate * grad_params['ln_sig_x_h'])
            
        #params['ln_sig_z_h'] = index_update(params['ln_sig_z_h'], rep,
        #                           params['ln_sig_z_h'][rep] - learning_rate * grad_params['ln_sig_z_h'])

        #params['ln_sig_r_h'] = index_update(params['ln_sig_r_h'], rep,
        #                           params['ln_sig_r_h'][rep] - learning_rate * grad_params['ln_sig_r_h'])

    
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
    params_aux['ln_sig_x_f2'] = params_aux['ln_sig_x_f2'][rep]
    params_aux['ln_sig_z_f'] = params_aux['ln_sig_z_f'][rep]
    params_aux['ln_sig_xz_f'] = params_aux['ln_sig_xz_f'][rep]
    params_aux['ln_sig_z_f2'] = params_aux['ln_sig_z_f2'][rep]
    params_aux['ln_sig_x_h'] = params_aux['ln_sig_x_h'][rep]
    params_aux['ln_sig_z_h'] = params_aux['ln_sig_z_h'][rep]
    params_aux['ln_sig_r_h'] = params_aux['ln_sig_r_h'][rep]
    params_aux['sig_r_h_q'] = params_aux['sig_r_h_q'][rep]

    return params_aux

@jax.jit
def getParamsForReport(params, rep):
    params_aux = params.copy()
    params_aux['Z'] = params_aux['Z'][:, rep][:, None]
    
    params_aux['ln_sig_x_f'] = params_aux['ln_sig_x_f'][rep]
    params_aux['ln_sig_x_f2'] = params_aux['ln_sig_x_f2'][rep]
    params_aux['ln_sig_z_f'] = params_aux['ln_sig_z_f'][rep]
    params_aux['ln_sig_xz_f'] = params_aux['ln_sig_xz_f'][rep]
    params_aux['ln_sig_z_f2'] = params_aux['ln_sig_z_f2'][rep]
    params_aux['ln_sig_x_h'] = params_aux['ln_sig_x_h'][rep]
    params_aux['ln_sig_z_h'] = params_aux['ln_sig_z_h'][rep]
    params_aux['ln_sig_r_h'] = params_aux['ln_sig_r_h'][rep]
    params_aux['sig_r_h_q'] = params_aux['sig_r_h_q'][rep]

    return params_aux


def getLatentZ(params, loss_as_par, dloss_as_par_jitted, optType, x, y, z, M, lam, sig, beta, neta, eta, nu, lu, epochs, report_freq, reps, batch_size, learning_rate, seed):
    N = x.shape[0]
    
    onp.random.seed(seed=seed)
    
    maxMonitor = 1000
    parts = int(onp.ceil(N/maxMonitor))
    smplsParts = onp.random.choice(parts, size=N)
    def myWhere(x):
       res,  = onp.where(x)
       return res

    smplsParts = [myWhere(smplsParts==i) for i in range(parts)]
    

    D_x = covariance_matrix(sqeuclidean_distance, x, x)
    sigma_x_med = 1 / np.median(D_x)
    sigma_x_q = 1 / np.quantile(D_x, sig[0])
    K_x = rbf_kernel_matrix({'gamma': sigma_x_med}, x, x)

    D_y = covariance_matrix(sqeuclidean_distance, y, y)
    sigma_y = 1 / np.median(D_y)
    K_y = rbf_kernel_matrix({'gamma': sigma_y}, y, y)

    #K_xy = K_x * K_y

    distsZm = covariance_matrix(sqeuclidean_distance, M, M)
    sigma_zm = 1 / np.median(distsZm)
    K_zmani = rbf_kernel_matrix({'gamma': sigma_zm}, M, M)

    num_reports = int(np.ceil(epochs / report_freq))-1
    print("num_reports: ", num_reports)

    # initialize parameters
    #m = M.shape[1]

    #z_ini, M = getIniZ(params, optType, M, K_xy)
    z_ini, M = getIniZ(params, M, None)
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

    
    print("lam: ", lam)
    print("sig: ", sig[0])
    print("sigma_z: ", sigma_z_q)
    print("sigma_x: ", sigma_x_q)
    print("type(sig): ", type(sig))
    
    smplPart1 = smplsParts[0]
    N_part1 = smplPart1.shape[0]
    y_part1 = y[smplPart1,]
    K_xz_part1 = K_xz[smplPart1, :]
    K_xz_part1 = K_xz_part1[:, smplPart1]
    ws = np.ones(N_part1)
    print("K_xz_part1.shape: ", K_xz_part1.shape)
    print("y_part1.shape: ", y_part1.shape)
    
    if True:
        weights, resids, y_hat = krrModel(lam, K_xz_part1, y_part1, ws)
        print("resids.shape: ", resids.shape)
        print("resids.mean", onp.mean(onp.abs(resids)))
        # right now we simply overwrite residual lengthscale sig_r_h to be median heuristic
        distsR = covariance_matrix(sqeuclidean_distance, resids, resids)
        sigma_r_med = 1 / np.quantile(distsR, 0.5)   
        sigma_r_q = 1 / np.quantile(distsR, sig[0])
    else:
        sigma_r_med = 1000
        sigma_r_q = 1000

    print("sigma_x_med", sigma_x_med)
    print("sigma_z_med", sigma_z_med)
    print("sigma_xz_med", sigma_xz_med)
    print("sigma_r_med", sigma_r_med)
    #qs = [0.3, 0.4,0.5, 0.6,0.7,0.8,0.9, 0.9,0.99, 0.99, 0.999,0.999]
    #qs = [0.1, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7,0.9,0.99, 1]
    qs = [0.1, 0.3, 0.5, 0.7,0.9,0.99, 1]
    #qs = [0.5, 0.5, 0.5, 0.5,0.5]
    #qs = [0.01, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1]
    qs2 = [0.3, 0.4, 0.5, 0.6, 0.7,0.9,0.99]
    
    if False : 
        sigs_z_f= sigma_z_med * np.ones(reps)
        sigs_z_f2= sigma_z_med * np.ones(reps)
        sigs_x_f= sigma_x_med * np.ones(reps)
        sigs_x_f2= sigma_x_med * np.ones(reps)
        sigs_xz_f= sigma_xz_med * np.ones(reps)
        sigs_z_h= sigma_z_med * np.ones(reps)
        sigs_x_h= sigma_x_med * np.ones(reps)
        sigs_r_h= sigma_r_med * np.ones(reps)

    elif True :
        qs_smpl_z_f = onp.random.choice(qs, size=reps)
        sigs_z_f= 1/np.quantile(D_z, qs_smpl_z_f) 
        sigs_z_f2= 1/np.quantile(D_z, onp.random.choice(qs, size=reps))
        qs_smpl_x_f = onp.random.choice(qs, size=reps)
        sigs_x_f= 1/np.quantile(D_x, qs_smpl_x_f)
        print("qs_x_f:", qs_smpl_x_f)
        print("qs_z_f:", qs_smpl_z_f)
        print("sigs_x_f:", sigs_x_f)
        print("sigs_z_f:", sigs_z_f)
        sigs_x_f2= 1/np.quantile(D_x, onp.random.choice(qs, size=reps))
        sigs_xz_f= 1/np.quantile(D_xz, onp.random.choice(qs, size=reps))
        sigs_z_h= 1/np.quantile(D_z, onp.random.choice(qs, size=reps))
        sigs_x_h= 1/np.quantile(D_x, onp.random.choice(qs, size=reps))
        sigs_r_h= 1/np.quantile(distsR, onp.random.choice(qs, size=reps))

    else :
        sigs_z_f= 1/np.quantile(D_z, np.array(qs)) 
        #sigs_z_f = index_update(sigs_z_f, len(qs)-1, 0.000000001)
        K_z_aux = rbf_kernel_matrix	({'gamma': sigs_z_f[0]}, z_ini, z_ini)
        print("min K_z_aux: ", onp.min(K_z_aux))        

        sigs_z_f2= sigma_z_med * np.ones(reps)
        sigs_x_f= sigma_x_med * np.ones(reps)
        sigs_x_f2= sigma_x_med * np.ones(reps)
        sigs_xz_f= sigma_xz_med * np.ones(reps)
        sigs_z_h= sigma_z_med * np.ones(reps)
        sigs_x_h= sigma_x_med * np.ones(reps)
        sigs_r_h= sigma_r_med * np.ones(reps)
        #sigs_z_f2= 1/np.quantile(D_z, onp.random.choice(qs, size=reps, replace=False))
        #sigs_x_f= 1/np.quantile(D_x, onp.random.choice(qs, size=reps, replace=False))
        #sigs_x_f2= 1/np.quantile(D_x, onp.random.choice(qs, size=reps, replace=False))
        #sigs_xz_f= 1/np.quantile(D_xz, onp.random.choice(qs, size=reps, replace=False))
        #sigs_z_h= 1/np.quantile(D_z, onp.random.choice(qs, size=reps, replace=False))
        #sigs_x_h= 1/np.quantile(D_x, onp.random.choice(qs, size=reps, replace=False))
        #sigs_r_h= 1/np.quantile(distsR, onp.random.choice(qs2, size=reps, replace=False))
    
    #print("z_ini[0:10]: ", z_ini[0:10])
    #print("x[0:10]: ", normalize(x[:,0])[0:10])
    #print("y[0:10]: ", normalize(y[:,0])[0:10])
    #print("D_z[0:4,0:4]:", D_z[0:4,0:4])
    print("sigs_z_f: ", sigs_z_f)
    #print("1/np.quantile(D_z, qs): ", 1/np.quantile(D_z, np.array(qs)))
    params['ln_sig_z_f']= np.log(sigs_z_f)
    params['ln_sig_z_f2']= np.log(sigs_z_f)
    params['ln_sig_x_f']= np.log(sigs_x_f)
    params['ln_sig_x_f2']= np.log(sigs_x_f)
    params['ln_sig_xz_f']= np.log(sigs_xz_f)
    params['ln_sig_z_h']= np.log(sigs_z_h)
    params['ln_sig_x_h']= np.log(sigs_x_h)
    params['ln_sig_r_h']= np.log(sigs_r_h)
    params['sig_r_h_q']= 0.5* np.ones(reps)


    monitors = {
        'loss': onp.zeros([num_reports, reps, parts]),
        'hsic': onp.zeros([num_reports, reps, parts]),
        'hsic_r': onp.zeros([num_reports, reps, parts]),
        'hsic_rx': onp.zeros([num_reports, reps, parts]),
        'hsic_rz': onp.zeros([num_reports, reps, parts]),
        'hsic_zmani': onp.zeros([num_reports, reps, parts]),
        'MMDzn': onp.zeros([num_reports, reps, parts]),
        #'hsic_alg': onp.zeros([num_reports, reps, parts]),
        'errs': onp.zeros([num_reports, reps, parts]),
        'ent_c': onp.zeros([num_reports, reps, parts]),
        'ent_x': onp.zeros([num_reports, reps, parts]),
        'ent_z': onp.zeros([num_reports, reps, parts]),
        'ent_r': onp.zeros([num_reports, reps, parts]),
        #'ent_c2': onp.zeros([num_reports, reps, parts]),
        #'ent_x2': onp.zeros([num_reports, reps, parts]),
        #'ent_z2': onp.zeros([num_reports, reps, parts]),
        #'ent_r2': onp.zeros([num_reports, reps, parts]),
        #'ent_alpha': onp.zeros([num_reports, reps, parts]),
        #'pen': onp.zeros([num_reports, reps, parts]),
        #'cost_slope': onp.zeros([num_reports, reps, parts]),
        #'cost_slope_krr': onp.zeros([num_reports, reps, parts]),
        #'cost_slope_z': onp.zeros([num_reports, reps, parts]),
        "hsic_zzhat": onp.zeros([num_reports, reps, parts]),
        'hsic_zhat': onp.zeros([num_reports, 1, parts])
        #'hsic_pval': onp.zeros([num_reports, reps, parts]),
        #'mmd_pval': onp.zeros([num_reports, reps, parts]),
        #"hsic_r_pval": onp.zeros([num_reports, reps, parts])
        #'errs_pval': onp.zeros([num_reports, reps, parts])
    }

    loss_vals = onp.ones([reps])*onp.Inf

    resids = onp.ones([n, reps])*onp.Inf  
    bestZ = onp.ones([n, reps])*onp.Inf

    for iteration in range(epochs):

        # print("*********************")
        if (iteration % 50 == 0):
            print("iteration: ", iteration)
        #print("nans: ", onp.sum(onp.isnan(onp.array(params["Z"]))))


        # get the gradient of the loss
        for rep in range(reps):
            #print("rep: ", rep)

	    # sampling with replacemnt
            smpl = onp.random.randint(low=0, high=n, size=batch_size)
            if iteration == 5:
            	print("smpl: ", smpl[0:4])
            #smpl2 = onp.random.randint(low=0, high=n, size=batch_size)
            #smpl3 = onp.linspace(0,n-1,n, dtype=int)
	    # sampling without replacemnt
            #smpl = onp.random.choice(a=n, size=batch_size, replace=False)

	    
            # caluclate K_x kernel

            D_x_aux = D_x[smpl, :]
            D_x_aux = D_x_aux[:, smpl]
            K_zmani_aux = K_zmani[smpl, :]
            K_zmani_aux = K_zmani_aux[:, smpl]
            K_y_aux = K_y[smpl, :]
            K_y_aux = K_y_aux[:, smpl]
            M_aux = M[smpl,:]

            #D_x_aux2 = D_x[smpl2, :]
            #D_x_aux2 = D_x_aux2[:, smpl2]
            #K_zmani_aux2 = K_zmani[smpl2, :]
            #K_zmani_aux2 = K_zmani_aux2[:, smpl2]
            #K_y_aux2 = K_y[smpl2, :]
            #K_y_aux2 = K_y_aux2[:, smpl2]
            #M_aux2 = M[smpl2,:]

            # equal weights
            ws = np.ones(batch_size)
            


            # algorithmic independence forcing

            # random weights - so that E[y|x] alg indep of p(x)
            #indx = onp.random.randint(low=1, high=batch_size, size=1)
            #x_aux = x[smpl,]
            #distsX = covariance_matrix(sqeuclidean_distance, x_aux, x_aux)
            #sigmaX = 10  # /np.median(distsX)
            #K_x_aux = rbf_kernel_matrix({'gamma': sigmaX}, x_aux, x_aux)
            #ws = K_x_aux[:, indx].squeeze()
            #ws = ws + 1e-9
            #ws = ws / sum(ws)*batch_size

            x_aux = x[smpl,]
            y_aux = y[smpl,]
            #x_aux2 = x[smpl2,]
            #y_aux2 = y[smpl2,]

            # prepare parameters for grad calculation (subsample)
            #params_aux = getParamsForGrad(params, rep, optType, smpl)
            params_aux = getParamsForGrad(params, rep, smpl)
            #params_aux2 = getParamsForGrad(params, rep, smpl2)
            #params_aux3 = getParamsForGrad(params, rep, smpl3)

                                   
            grad_params = dloss_as_par_jitted(params_aux, beta, neta, eta, lam, nu, lu, D_x_aux, x_aux, y_aux, K_y_aux, M_aux, K_zmani_aux, ws)
            #loss_val = loss_as_par(params_aux2, beta, neta, eta, lam, nu, lu, D_x_aux2, x_aux2, y_aux2, K_y_aux2, M_aux2, K_zmani_aux2, ws)

            #grad_params = dloss_as_par_jitted(params_aux, beta, neta, eta, lam, nu, lu, D_x_aux, x_aux, y_aux, K_y_aux, None, None, ws)
            #loss_val = loss_as_par(params_aux2, beta, neta, eta, lam, nu, lu, D_x_aux2, x_aux2, y_aux2, K_y_aux2, None, None, ws)

            #if (loss_val < loss_vals[rep]):
            #    loss_vals[rep] = loss_val
            #    bestZ[:,rep] = params_aux3["Z"][:,0]
                
            


            # update params
            updateParams(params, grad_params, optType, smpl, iteration, rep, learning_rate)



            # prepare parameters for reporting (full smpl)
            #params_aux = getParamsForReport(params, rep, optType)
            params_aux = getParamsForReport(params, rep)

            if (iteration % report_freq == 0) & (iteration != 0):
                #print("report")
                print("iteration report: ", iteration)
                indxRep = int(iteration / report_freq)-1
                for part in range(parts):
                    smplPart = smplsParts[part]
                    params_part = getParamsForGrad(params, rep, smplPart)
                    x_part = x[smplPart,]#[:,None]
                    y_part = y[smplPart,]#[:,None]
                    if z is None:
                        z_part = None
                    else:
                        z_part = z[smplPart,]#[:,None]
                    
                    M_part = M[smplPart,:]
                    K_zmani_part = K_zmani[smplPart, :]
                    K_zmani_part = K_zmani_part[:, smplPart]
                    K_y_part = K_y[smplPart, :]
                    K_y_part = K_y_part[:, smplPart]
                    D_x_part = D_x[smplPart, :]
                    D_x_part = D_x_part[:, smplPart]

                    nPerPart = smplPart.shape[0]
                    ws = np.ones(nPerPart)
                    loss_val = loss_as_par(params_part, beta, neta, eta, lam, nu, lu, D_x_part, x_part, y_part, K_y_part, M_part, K_zmani_part, ws)
                    #loss_val = loss_as_par(params_part, beta, neta, eta, lam, nu, lu, D_x_part, x_part, y_part, K_y_part, None, None, ws)
                          
                                

                              
                    monitor = model(optType, params_part, lam, D_x_part, x_part, y_part, z_part, M_part, K_zmani_part, K_y_part)
                    #monitor = model(optType, params_part, lam, D_x_part, x_part, y_part, z_part, None, None, K_y_part)  
                    monitors['hsic'][indxRep, rep, part] = monitor['hsic']
                    monitors['hsic_r'][indxRep, rep, part] = monitor['hsic_r']
                    monitors['hsic_rx'][indxRep, rep, part] = monitor['hsic_rx']
                    monitors['hsic_rz'][indxRep, rep, part] = monitor['hsic_rz']
                    monitors['hsic_zmani'][indxRep, rep, part] = monitor['hsic_zmani']
                    monitors['MMDzn'][indxRep, rep] = monitor['MMDzn']
                    #monitors['hsic_alg'][indxRep, rep, part] = monitor['hsic_alg']
                    monitors['errs'][indxRep, rep, part] = monitor['mse']
                    monitors['ent_c'][indxRep, rep, part] = monitor['h_x']
                    monitors['ent_x'][indxRep, rep, part] = monitor['h_xx']
                    monitors['ent_z'][indxRep, rep, part] = monitor['h_z']
                    monitors['ent_r'][indxRep, rep, part] = monitor['h_r']
                    #monitors['ent_c2'][indxRep, rep, part] = monitor['h_x2']
                    #monitors['ent_x2'][indxRep, rep, part] = monitor['h_xx2']
                    #monitors['ent_z2'][indxRep, rep, part] = monitor['h_z2']
                    #monitors['ent_r2'][indxRep, rep, part] = monitor['h_r2']
                    #monitors['ent_alpha'][indxRep, rep, part] = monitor['h_a']
                    #monitors['pen'][indxRep, rep, part] = monitor['penalty']
                    #monitors['cost_slope'][indxRep, rep, part] = monitor['cost_slope']
                    #monitors['cost_slope_krr'][indxRep, rep, part] = monitor['cost_slope_krr']
                    #monitors['cost_slope_z'][indxRep, rep, part] = monitor['cost_slope_z']
                    monitors["hsic_zzhat"][indxRep, rep, part] = monitor["hsic_zzhat"]
                    monitors['loss'][indxRep, rep, part] = loss_val
                    
                    #resids[smplPart, rep] = monitor["resids"][:,0]
		    

                    if rep == (reps - 1):
                        #z_hat = getZ(params, optType, M, K_xy)
                        #z_hat = getZ(params, M, K_xy)
                        z_hat = getZ(params)
                        #z_hat = getZ(params_part, M_part, K_xy_part)
                        z_hat = z_hat[smplPart,]
                        hsics_zs = [onp.array(hsicRBF(z_hat[:, i], z_hat[:, j])) for i in range(z_hat.shape[1]) for j in range(i)]
                        monitors["hsic_zhat"][indxRep, 0, part] = onp.mean(hsics_zs)
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

        #z_hat = getZ(params, optType, M, K_xy)
        z_hat = getZ(params, M, K_xy)
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

    #bestResids = resids

    return params, monitors #, resids, bestResids, bestZ


def getLatentZs(X, nm, optType, lam, sig, beta, neta, eta, nu, lu, num_epochs, report_freq, num_reps, batch_size, learning_rate, job):
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

    #params_xy = getIniPar(optType2[0], N, m, num_reps, y, z_mani)
    #params_yx = getIniPar(optType2[0], N, m, num_reps, x, z_mani)
    #params_xy = getIniPar(N, m, num_reps, y, z_mani)
    #params_yx = getIniPar(N, m, num_reps, x, z_mani)
    params_xy = getIniPar(num_reps, y)
    params_yx = getIniPar(num_reps, x)

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
                                                         reps=num_reps, batch_size=batch_size,learning_rate=learning_rate, seed=job)
        print("done x->y", time.process_time() - start)  #, resids_xy, bestResids_xy, bestZ_xy
        print("y to x")
        start = time.process_time()
        params_yx, path_yx = getLatentZ(params_yx, loss_as_par, dloss_as_par_jitted, ot, y, x, z, z_mani, lam=lam, sig=sig, beta=beta, neta=neta, eta=eta, nu=nu, lu=lu, epochs=num_epochs, report_freq=report_freq,
                                                         reps=num_reps, batch_size=batch_size,learning_rate=learning_rate, seed=job)
        print("done y->x", time.process_time() - start)  # , resids_yx, bestResids_yx, bestZ_yx
        res_xy.append(path_xy)
        res_yx.append(path_yx)

    res_xy = {k: onp.concatenate([r[k] for r in res_xy], axis=0) for k in res_xy[0].keys()}
    res_yx = {k: onp.concatenate([r[k] for r in res_yx], axis=0) for k in res_yx[0].keys()}

    print("results")
    res = {"params_xy": params_xy, "params_yx": params_yx, "path_xy": res_xy, "path_yx": res_yx}
    print(res["path_xy"])
    print(res["path_yx"])
    #, "resids_xy": resids_xy, "resids_yx":resids_yx,"bestResids_xy": bestResids_xy, "bestResids_yx": bestResids_yx, "bestZ_xy":bestZ_xy , "bestZ_yx": bestZ_yx
    #print(res)
    #"last_xy": last_xy,"last_yx": last_yx, "last_xy_gp": last_xy_gp, "last_yx_gp": last_yx_gp


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

def jitterByDist(x):
    x = x[:,None]
    Ds = x - x.T 
    Ds = Ds +( Ds==0)*100
    DsPos = Ds + (Ds<0)*100
    DsNeg = Ds + (Ds>0)*100
    leftJitter = onp.apply_along_axis(onp.min, 1, onp.abs(DsNeg))
    rightJitter = onp.apply_along_axis(onp.min, 1, DsPos)
    leftJitter[leftJitter>1] = 0
    rightJitter[rightJitter>1] = 0
    leftJitter = leftJitter/2
    rightJitter = rightJitter/2
    jitx = onp.random.uniform(low=-rightJitter, high=leftJitter)
    xjit = x+jitx[:,None]
    return xjit[:,0]

def jitterByDist2(x):
    x = x[:,None]
    Ds = x - x.T 
    Ds = Ds +( Ds==0)*100
    DsPos = Ds + (Ds<0)*100
    DsNeg = Ds + (Ds>0)*100
    leftJitter = onp.apply_along_axis(onp.min, 1, onp.abs(DsNeg))
    rightJitter = onp.apply_along_axis(onp.min, 1, DsPos)
    leftJitter[leftJitter>1] = 0
    rightJitter[rightJitter>1] = 0
    xun, indx, numReps = onp.unique(x, return_counts=True, return_inverse=True)
    numReps = numReps[indx]
    leftJitter[numReps==1] = 0
    rightJitter[numReps==1] = 0
    leftJitter = leftJitter/2
    rightJitter = rightJitter/2
    jitx = onp.random.uniform(low=-rightJitter, high=leftJitter)
    xjit = x+jitx[:,None]
    return xjit[:,0]

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
