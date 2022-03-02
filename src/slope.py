# slope score stuffs
from rpy2.robjects import r
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
#%load_ext rpy2.ipython
from rpy2 import robjects;
from rpy2.robjects.packages import importr
MARS = importr('earth');
import re;
import numpy as onp
import jax.numpy as np


r['source']('func_entropy_v1.R')
funcs_r = robjects.globalenv

def REarth(X,Y,M=1):
    #print("X.shape:", X.shape)
    row,col=X.shape;
    rX=r.matrix(X,ncol=col,byrow=False);
    rY=r.matrix(Y,ncol=1,byrow=True);

    coeffs=[];


    try:
        rearth=MARS.earth(x=rX,y=rY,degree=M);
    except:
        print("Singular fit encountered, retrying with Max Interactions=1");
        rearth=MARS.earth(x=rX,y=rY,degree=1);

    COEFF_INDEX=11;
    CUTS_INDEX=6;
    SELECTED_INDEX=7;
    RSS_INDEX=0;

    arg_count=onp.size(rearth[COEFF_INDEX]);
    cut_count=onp.size(rearth[SELECTED_INDEX]);

    z=str(rearth[COEFF_INDEX]);
    lines=z.split('\n');
    interactions=[];
    for i in range(2,len(lines)):
        val=1+ lines[i].count('*');
        interactions.append(val);


    #print(rearth[COEFF_INDEX]);
    #print("-----------");
    #print(rearth[CUTS_INDEX]);
    #print("-----------");
    #print(rearth[SELECTED_INDEX]);
    #print("-----------");
    #'''
    tst=str(rearth[COEFF_INDEX]);
    listed=re.split('\n|h\(x[0-9]{1,1000}\-|h\(|\-x[0-9]{1,1000}\)|\)',tst)
    #print(listed);
    for vs in listed:
        try:
                    if vs is not None:
                        coeffs.append(float(vs.strip()));
        except ValueError:
                pass;
    '''
    for i in range(0,arg_count):
            coeffs.append(rearth[COEFF_INDEX][i]);
    '''

#   print(coeffs);
#   print(len(coeffs));
#   import ipdb; ipdb.set_trace();
    sse=rearth[RSS_INDEX][0]
    #print("sse: ",sse);
    #print(arg_count);
    return sse,[coeffs],onp.array([cut_count]),interactions;


def logg(x):
    if x == 0:
        return 0
    else:
        return onp.log2(x)


def logN(z):
    z = onp.ceil(z);

    if z < 1:
        return 0;
    else:
        log_star = logg(z);
        sum = log_star;

        while log_star > 0:
            log_star = logg(log_star);
            sum = sum + log_star;

    return sum + logg(2.865064)


def model_score(coeff):
    Nans = onp.isnan(coeff);
    if any(Nans):
        print('Warning: Found Nans in regression coefficients. Setting them to zero...')
    coeff[Nans] = 0;
    sum = 0;
    for c in coeff:
        if np.abs(c) > 1e-12:
            c_abs = onp.abs(c);
            c_dummy = c_abs;
            precision = 1;

            while c_dummy < 1000:
                c_dummy *= 10;
                precision += 1;
            sum = sum + logN(c_dummy) + logN(precision) + 1
    return sum;


def FitSpline(source, target, M=2, temp=False):
    sse, coeff, hinge_count, interactions = REarth(source, target, M)
    score = model_score(onp.copy(coeff[0]))
    return sse, score, coeff, hinge_count, interactions;


from scipy.special import comb


def Combinator(M, k):
    sum = comb(M + k - 1, M);
    if sum == 0:
        return 0;
    return onp.log2(sum);


# F - dont know but it seems its fixed to 9
def AggregateHinges(hinges, k, M, F):
    cost = 0;
    flag = 1;

    for M in hinges:
        cost = logN(M) + Combinator(M, k) + M * onp.log2(F);

    return cost;


def gaussian_score_sse(sigma, sse, n, resolution):
    sigmasq = sigma ** 2;
    if sse == 0.0 or sigmasq == 0.0:
        return onp.array([0.0]);
    else:
        err = (sse / (2 * sigmasq * onp.log(2))) + ((n / 2) * logg(2 * onp.pi * sigmasq)) - n * logg(resolution)
        return max(err, onp.array([0]));


def gaussian_score_emp_sse(sse, n, min_diff):
    var = sse / n
    sigma = onp.sqrt(var)
    return gaussian_score_sse(sigma, sse, n, min_diff)


# k - number of parents (2)
# rows - n
# V - total nodes (3)
# mindiff - resolution for gaussian score - min difference in the target/child variable
# M - interactions for Earth (mars) model
# source - x
# target - y
def myComputeScore(source, target, rows, mindiff, mindiff_x, k, V, M, F):
    base_cost = model_score(k) + k * onp.log2(V);
    sse, model, coeffs, hinges, interactions = FitSpline(source, target, M);
    base_cost = base_cost + model_score(hinges) + AggregateHinges(interactions, k, M, F);
    cost = gaussian_score_emp_sse(sse, rows, mindiff) + model + base_cost;
    # add cost of marginal which is not included here
    cost = cost - rows * logg(mindiff_x)
    return cost, coeffs;



def CalculateMinDiff(variable):
    sorted_v = onp.copy(variable);
    sorted_v.sort(axis=0);
    diff = onp.abs(sorted_v[1] - sorted_v[0]);

    if diff == 0: diff = onp.array([10.01]);

    for i in range(1, len(sorted_v) - 1):
        curr_diff = onp.abs(sorted_v[i + 1] - sorted_v[i]);
        if curr_diff != 0 and curr_diff < diff:
            diff = curr_diff;
    return diff;
