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


def nonparametric_variational_inference(logp, violation, num_samples=1, constrained=False, num_batches=1):
    
    
    def unpack_params(params):
        '''Uniform mixture of Gaussian'''
        means, log_stds = params[:, 1:], params[:, 0]
        return means, log_stds

    def sample_q(samples, params):
        '''Samples from mixture of Gaussian q'''
        N = params.shape[0]
        k = ds.Categorical(probs=torch.ones(N)).sample(torch.Size([samples]))
        means, log_stds = unpack_params(params)
        means, log_stds = means[k], log_stds[k]
        num_weights = means.shape[1]
        covs = torch.zeros(samples, num_weights, num_weights)
        I = torch.eye(num_weights)
        for j in range(samples):
            covs[j] = log_stds[j].exp().pow(2) * I
        return ds.MultivariateNormal(means, covs).sample()

    def log_q_n(n, params):
        '''
        Entropy of variational distribution
        Compute value of log q_n using log-sum-exp 
        
        log[ (1/N) sum_j exp(logp_j) ]
        =  log[sum_j exp(logp_j)] - log[N]
        =  logsumexp(logp_j) - log[N]

        '''
        means, log_stds = unpack_params(params)
        vars = log_stds.exp().pow(2)
        N = means.shape[0]
        W = means.shape[1]
        logprobs = torch.zeros(N)
        for j in range(N):
            mvn = ds.MultivariateNormal(means[j], (vars[j] + vars[n]) * torch.eye(W))
            logprobs[j] = mvn.log_prob(means[n])
        return torch.logsumexp(logprobs, 0) - torch.tensor(N).float().log()
    

    '''First- and second-order approximation of ELBO'''

    def trace_hessian(f, x):
        #  Computes trace of hessian of f w.r.t. x (x is 1D)
        dim = x.shape[0]
        df = grad(f, x, create_graph=True)[0]
        tr_hess = 0
        # iterate over every entry in df/dx and compute derivate
        for j in range(dim):
            d2fj = grad(df[j], x, create_graph=True)[0]
            # add d2f/dx2 to trace
            tr_hess += d2fj[j]
        return tr_hess

    def elbo_approx_1(params, iter):
        means, _ = unpack_params(params)
        N = means.shape[0]
        components = torch.zeros(N)
        for j in range(N):
            components[j] = logp(means[j].unsqueeze(0),iter) \
                - log_q_n(j, params)
        return components.mean(0)

    def elbo_approx_2(params, iter):

        # first-order approx
        elbo_1 = elbo_approx_1(params, iter)
        
        # second-order term
        means, log_stds = unpack_params(params)
        vars = log_stds.exp().pow(2)
        N = means.shape[0]
        traces = torch.zeros(N)

        for j in range(N):
            x = means[j]
            y = logp(x.unsqueeze(0), iter)
            traces[j] = 0.5 * vars[j] * trace_hessian(y, x)
        return elbo_1 + traces.sum()

    def variational_objective_regular(params, iter):
        return - elbo_approx_2(params, iter)
    
    def variational_objective_constrained(params, iter):
        return - elbo_approx_2(params, iter) + violation(params, sample_q)

    if constrained:
        return variational_objective_constrained, elbo_approx_2, unpack_params, sample_q
    else:
        return variational_objective_regular, elbo_approx_2, unpack_params, sample_q


    
