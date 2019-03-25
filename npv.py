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


def nonparametric_variational_inference(logp, violation, num_weights, num_samples=1, constrained=False, num_batches=1, general_mixture=True):
    
    if not general_mixture:
        
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
            entropy = 0
            for j in range(N):
                entropy_ = log_q_n(j, params)
                entropy += entropy_
                components[j] = logp(means[j].unsqueeze(0), iter) - entropy_
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
            print('ELBO_1', elbo_1.item())
            print('Traces', traces.sum().item())
            return elbo_1 + traces.sum()

        def variational_objective_regular(params, iter):
            return - elbo_approx_2(params, iter)
        
        def variational_objective_constrained(params, iter):
            return - elbo_approx_2(params, iter) + violation(params, sample_q)

        if constrained:
            return variational_objective_constrained, elbo_approx_2, unpack_params, sample_q
        else:
            return variational_objective_regular, elbo_approx_2, unpack_params, sample_q

    else:
        
        'Extension of NPV paper'
        print('Using generalization of NPV')

        def unpack_params(params):
            '''Uniform mixture of Gaussian'''
            pi, means, log_stds = params[:, 0], params[:, 1:num_weights + 1], params[:, num_weights + 1:]
            pi_norm = pi.exp() / pi.exp().sum()
            return pi_norm, means, log_stds

        def sample_q(samples, params):
            '''Samples from mixture of Gaussian q'''
            pi, means, log_stds = unpack_params(params)
            k = ds.Categorical(probs=pi).sample(torch.Size([samples]))
            means, log_stds = means[k], log_stds[k]
            covs = torch.zeros(samples, num_weights, num_weights)
            for j in range(samples):
                covs[j] = torch.diag(log_stds[j].exp().pow(2))
            s = ds.MultivariateNormal(means, covs).sample()
            return s

        def log_q_n(n, params):
            '''
            Lower bound of entropy of variational distribution 
            
            log[  sum_j  pi_j exp(logp_j) ]
            =  log[sum_j exp(log(pi_j)) exp(logp_j)] 
            =  log[sum_j exp(logp_j + log(pi_j))] 
            =  logsumexp(logp_j + log(pi_j))]

            '''
            pi, means, log_stds = unpack_params(params)
            vars = log_stds.exp().pow(2)
            N = means.shape[0]
            logprobs = torch.zeros(N)
            for j in range(N):
                mvn = ds.MultivariateNormal(
                    means[j], torch.diag(vars[j] + vars[n]))
                logprobs[j] = mvn.log_prob(means[n]) + pi[j].log()
            return torch.logsumexp(logprobs, 0)
            

        '''First- and second-order approximation of ELBO'''

        def weighted_trace_hessian(f, x, w):
            #  Computes trace of hessian of f w.r.t. x (x is 1D)
            dim = x.shape[0]
            df = grad(f, x, create_graph=True)[0]
            tr_hess = 0
            # iterate over every entry in df/dx and compute derivate
            for j in range(dim):
                d2fj = grad(df[j], x, create_graph=True)[0]
                # add d2f/dx2 to trace
                tr_hess += d2fj[j] * w[j]  # weighted hessian
            return tr_hess

        def elbo_approx_1(params, iter):
            pi, means, _ = unpack_params(params)
            N = means.shape[0]
            components = torch.zeros(N)
            entropy = 0
            for j in range(N):
                entropy_ = log_q_n(j, params)
                entropy += entropy_
                components[j] = logp(means[j].unsqueeze(0), iter) - entropy_
            return components.dot(pi)

        def elbo_approx_2(params, iter):

            # first-order approx
            elbo_1 = elbo_approx_1(params, iter)

            # second-order term
            pi, means, log_stds = unpack_params(params)
            vars = log_stds.exp().pow(2)
            N = means.shape[0]
            traces = torch.zeros(N)

            for j in range(N):
                x = means[j]
                y = logp(x.unsqueeze(0), iter)
                traces[j] = 0.5 * weighted_trace_hessian(y, x, vars[j])
            
            print('ELBO_1', elbo_1.item())
            print('Traces', traces.dot(pi).item())
            return elbo_1 + traces.dot(pi)

        def variational_objective_regular(params, iter):
            return - elbo_approx_2(params, iter)

        def variational_objective_constrained(params, iter):
            return - elbo_approx_2(params, iter) + violation(params, sample_q)

        if constrained:
            return variational_objective_constrained, elbo_approx_2, unpack_params, sample_q
        else:
            return variational_objective_regular, elbo_approx_2, unpack_params, sample_q



'''

Own formulation of approximating ELBO of mixture of gaussian via gumbel-softmax trick

'''
def gumbel_softmax_mix_of_gauss(logp, violation, num_weights, tau=1.0, num_samples=1, constrained=False, num_batches=1):
    

    def unpack_params(params):
        '''Uniform mixture of Gaussian'''
        pi, means, log_stds = params[:, 0], params[:, 1:num_weights + 1], params[:, num_weights + 1:]
        pi_softm = nn.functional.softmax(pi, dim=0)
        return pi_softm, means, log_stds

    def sample_q(samples, params):
        '''Samples from mixture of Gaussian q using the Gumbel-softmax relaxation'''
        mixtures = params.shape[0]
        G = ds.Gumbel(torch.tensor(0.0), torch.tensor(1.0)).sample(
            torch.Size([samples, mixtures]))
        E = ds.Normal(torch.tensor(0.0), torch.tensor(1.0)).sample(
            torch.Size([samples, mixtures, num_weights]))
        pi, means, log_stds = unpack_params(params)
        
        w = nn.functional.gumbel_softmax(pi + G, tau=tau, hard=False)
        all_gaussians = means + log_stds.exp() * E

        sampled_mixture_rv = torch.einsum('bk,bkw->bw', [w, all_gaussians])
        return sampled_mixture_rv

    def log_q_n(n, params):
        '''
        Entropy of variational distribution approximated 
        using actual mixture of gaussian lower bound
        
        log[  sum_j  pi_j exp(logp_j) ]
        =  log[sum_j exp(log(pi_j)) exp(logp_j)] 
        =  log[sum_j exp(logp_j + log(pi_j))] 
        =  logsumexp(logp_j + log(pi_j))]

        '''
        pi, means, log_stds = unpack_params(params)
        vars = log_stds.exp().pow(2)
        mixtures = means.shape[0]
        logprobs = torch.zeros(mixtures)
        for j in range(mixtures):
            mvn = ds.MultivariateNormal(
                means[j], torch.diag(vars[j] + vars[n]))
            logprobs[j] = mvn.log_prob(means[n]) + pi[j].log()
        return torch.logsumexp(logprobs, 0) 

    def log_q_lb(params):
        pi, _, _ = unpack_params(params)
        mixtures = pi.shape[0]
        ind_entropies = torch.zeros(mixtures)
        for j in range(mixtures):
            ind_entropies[j] = log_q_n(j, params)
        return ind_entropies.mean(0)

    entropy_fun = lambda params: - log_q_lb(params)

    def elbo(params, iter):
        
        # sample using Gumbel-Softmax trick
        W = sample_q(num_samples, params)
        
        # reparam. trick using relaxed samples
        logprob = logp(W, iter)

        # entropy using lower bound of mix. of Gaussians
        entropy = entropy_fun(params)

        elbo = logprob + entropy
        return elbo

        
    def variational_objective_regular(params, iter):
        return - elbo(params, iter)

    def variational_objective_constrained(params, iter):
        return - elbo(params, iter) + violation(params, sample_q)

    if constrained:
        return variational_objective_constrained, elbo, unpack_params, sample_q, entropy_fun
    else:
        return variational_objective_regular, elbo, unpack_params, sample_q, entropy_fun


