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


def bayes_by_backprop_variational_inference(logp, violation, num_samples=1, constrained=False):
    
    '''Fully-factored Gaussian'''
    def unpack_params(params):
        mean, log_std = params[:, 0], params[:, 1]
        return mean, log_std

    '''
    log q(x) of variational distribution (negative entropy)
    
    (closed-form Gaussian entropy 
    E[- log q(w)] = 1/2 + 1/2 log(2 * pi * std^2))
    '''

    def logq(log_std):
        return - torch.sum(0.5 + 0.5 * torch.log(2 * torch.tensor(math.pi)) + (1.0 + log_std.exp()).log().log()) # log_std)

    '''
    Stochastic estimate of variational objective (neg. ELBO)
    using reparameterization trick (not REINFORCE/score function)
    '''

    def evidence_lower_bound(params):
        mean, log_std = unpack_params(params)
        weights = mean + torch.randn(num_samples,
                                     params.shape[0]) * torch.log(1.0 + log_std.exp())# log_std.exp()
        elbo = (logp(weights) - logq(log_std)).mean()
        return elbo

    def variational_objective_regular(params):
        return - evidence_lower_bound(params)
    
    def variational_objective_constrained(params):
        return - evidence_lower_bound(params) + violation(params)

    
    if constrained:
        return variational_objective_constrained, evidence_lower_bound
    else:
        return variational_objective_regular, evidence_lower_bound
