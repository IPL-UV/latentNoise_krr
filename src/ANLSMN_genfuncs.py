# generate LS-s style data
import numpy as onp
from latentNoise_funcs_gen import *

def sample_LSs(n):
    ran = onp.random.normal(size=n)
    noise_exp = 1
    noise_var = onp.random.uniform(size=n, low=1, high=2)
    noisetmp = (onp.sqrt(noise_var) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_pa = noisetmp

    a_sig = onp.random.uniform(size=1, low=-2, high=2)
    bern = onp.random.binomial(size=1, n=1, p=0.5)
    b_sig = bern * onp.random.uniform(size=1, low=0.5, high=2) + (1 - bern) * onp.random.uniform(size=1, low=-2,
                                                                                                 high=-0.5)
    c_sig = onp.random.exponential(size=1, scale=1 / 4) + 1
    x_child = c_sig * (b_sig * (x_pa + a_sig)) / (1 + abs(b_sig * (x_pa + a_sig)))
    ran = onp.random.normal(size=n)
    noise_var_ch = onp.random.uniform(size=n, low=1, high=2)
    z = (0.2 * onp.sqrt(noise_var_ch) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_child = x_child + (x_child - min(x_child)) * z
    return x_pa, x_child, z


def sample_LS(n):
    ran = onp.random.normal(size=n)
    noise_exp = 1
    noise_var = onp.random.uniform(size=n, low=1, high=2)
    noisetmp = (onp.sqrt(noise_var) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_pa = noisetmp

    kern_pa = rbf_kernel_matrix({'gamma': 0.5}, x_pa, x_pa)
    mu = onp.zeros(n)
    x_child = onp.random.multivariate_normal(mean=mu, cov=kern_pa, size=1)
    x_child = x_child.flatten()

    ran = onp.random.normal(size=n)
    noise_var_ch = onp.random.uniform(size=n, low=1, high=2)
    z = (0.2 * onp.sqrt(noise_var_ch) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_child = x_child + (x_child - min(x_child)) * z
    return x_pa, x_child, z


def sample_AN(n):
    ran = onp.random.normal(size=n)
    noise_exp = 1
    noise_var = onp.random.uniform(size=n, low=1, high=2)
    noisetmp = (onp.sqrt(noise_var) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_pa = noisetmp

    kern_pa = rbf_kernel_matrix({'gamma': 0.5}, x_pa, x_pa)
    mu = onp.zeros(n)
    x_child = onp.random.multivariate_normal(mean=mu, cov=kern_pa, size=1)
    x_child = x_child.flatten()

    ran = onp.random.normal(size=n)
    noise_var_ch = onp.random.uniform(size=n, low=1, high=2)
    z = (0.2 * onp.sqrt(noise_var_ch) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_child = x_child + z
    return x_pa, x_child, z


def sample_ANs(n):
    ran = onp.random.normal(size=n)
    noise_exp = 1
    noise_var = onp.random.uniform(size=n, low=1, high=2)
    noisetmp = (onp.sqrt(noise_var) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_pa = noisetmp

    a_sig = onp.random.uniform(size=1, low=-2, high=2)
    bern = onp.random.binomial(size=1, n=1, p=0.5)
    b_sig = bern * onp.random.uniform(size=1, low=0.5, high=2) + (1 - bern) * onp.random.uniform(size=1, low=-2,
                                                                                                 high=-0.5)
    c_sig = onp.random.exponential(size=1, scale=1 / 4) + 1
    x_child = c_sig * (b_sig * (x_pa + a_sig)) / (1 + abs(b_sig * (x_pa + a_sig)))
    ran = onp.random.normal(size=n)
    noise_var_ch = onp.random.uniform(size=n, low=1, high=2)
    z = (0.2 * onp.sqrt(noise_var_ch) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_child = x_child + z
    return x_pa, x_child, z


def sample_MNU(n):
    ran = onp.random.normal(size=n)
    noise_exp = 1.0
    noise_var = onp.random.uniform(size=n, low=1.0, high=2.0)
    noisetmp = (onp.sqrt(noise_var) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_pa = noisetmp

    a_sig = onp.random.uniform(size=1, low=-2.0, high=2.0)
    bern = onp.random.binomial(size=1, n=1, p=0.5)
    b_sig = bern * onp.random.uniform(size=1, low=0.5, high=2.0) + (1 - bern) * onp.random.uniform(size=1, low=-2.0,high=-0.5)
    c_sig = onp.random.exponential(size=1, scale=1 / 4) + 1
    x_child = c_sig * (b_sig * (x_pa + a_sig)) / (1.0 + abs(b_sig * (x_pa + a_sig)))
    ran = onp.random.normal(size=n)
    noise_var_ch = onp.random.uniform(size=n, low=1.0, high=2.0)
    z = onp.random.uniform(size=n, low=0.0, high=1.0)
    x_child = x_child * z
    return x_pa, x_child, z
