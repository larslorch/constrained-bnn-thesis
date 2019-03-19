import matplotlib.pyplot as plt
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable

'''
Implements Bayes by Backprop
https://arxiv.org/pdf/1505.05424.pdf 

together with our constrained version prior formulation
'''


def bayes_by_backprop_variational_inference(logp, violation, num_samples=1, constrained=False, num_batches=1):
    
    '''Fully-factored Gaussian'''
    def unpack_params(params):
        mean, log_std = params[:, 0], params[:, 1]
        return mean, log_std

    def sample_q(samples, params):
        mean, log_std = unpack_params(params)
        weights = mean + torch.randn(samples,
                                     mean.shape[0]) * log_std.exp()
        return weights


    '''
    log q(x) of variational distribution (negative entropy)
    
    (closed-form Gaussian entropy 
    E[- log q(w)] = 1/2 + 1/2 log(2 * pi * std^2))
    '''

    def logq(log_std):
        return - torch.sum(0.5 + 0.5 * torch.log(2 * torch.tensor(math.pi)) + log_std) 

    '''
    Stochastic estimate of variational objective (neg. ELBO)
    using reparameterization trick (not REINFORCE/score function)

    corrects for batch_size/num_batches by scaling KL[q(w)|p(w)]
    as explained in https://arxiv.org/pdf/1505.05424.pdf  

    iter is needed for batch training
    '''

    def evidence_lower_bound(params, iter):
        mean, log_std = unpack_params(params)
        weights = sample_q(num_samples, params)
        elbo = (logp(weights, iter) - logq(log_std) / num_batches).mean()
        return elbo

    def variational_objective_regular(params, iter):
        return - evidence_lower_bound(params, iter)
    
    def variational_objective_constrained(params, iter):
        return - evidence_lower_bound(params, iter) + violation(params, sample_q)

    
    if constrained:
        return variational_objective_constrained, evidence_lower_bound, unpack_params, sample_q
    else:
        return variational_objective_regular, evidence_lower_bound, unpack_params, sample_q
