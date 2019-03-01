import matplotlib.pyplot as plt
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable, grad

'''
Implements Nonparametric Variational Inference (NPV)
http://gershmanlab.webfactional.com/pubs/GershmanHoffmanBlei12.pdf

together with our constrained version prior formulation
'''


def nonparametric_variational_inference(test_params, logp, violation, num_samples=1, constrained=False, num_batches=1):
    
    '''Fully-factored Gaussian'''
    def unpack_params(params):
        means, log_stds = params[:, 1:], params[:, 0]
        return means, log_stds


    '''
    Entropy of variational distribution

    Compute value of log q_n using log-sum-exp 

        log[ (1/N) sum_j exp(logp_j) ]
     =  log[sum_j exp(logp_j)] - log[N]
     =  logsumexp(logp_j) - log[N]

    '''


    def log_q_n(n, params):
        means, log_stds = unpack_params(params)
        vars = log_stds.exp().pow(2)
        N = means.shape[0]
        W = means.shape[1]
        logprobs = torch.zeros(N)
        for j in range(N):
            mvn = ds.MultivariateNormal(means[j], (vars[j] + vars[n]) * torch.eye(W))
            logprobs[j] = mvn.log_prob(means[n])
        return torch.logsumexp(logprobs, 0) - torch.tensor(N).float().log()
    
    '''
    First- and second-order approximation of ELBO
    '''

    # First-order approximation of ELBO
    def elbo_approx_1(params, iter):
        means, _ = unpack_params(params)
        N = means.shape[0]
        compts = torch.zeros(N)
        for j in range(N):
            compts[j] = logp(means[j].unsqueeze(0), iter) - log_q_n(j, params)
        return compts.mean(0)

    # Trace(Hessian)
    def trhess(params):
        # pretty sure this has to be done via sampling...
        f = 0
        return 0
  
    r = elbo_approx_1(test_params, 0)

    print(r)
    exit(0)

    elbo_approx_2 = 0

    if constrained:
        return elbo_approx_1, elbo_approx_2
    else:
        return elbo_approx_1, elbo_approx_2
