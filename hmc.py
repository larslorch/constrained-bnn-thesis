import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable

from bnn import make_BNN
from darting_hmc import make_darting_HMC_sampler
from utils import *

'''
Implements Hamiltonian Monte Carlo for a given distribution p(x)
p(x) is the target distribution up to a normalization constant 1/Z where Z = int p(x) dx 
For BNNs, p(x) is the posterior over weights p(w|X,Y)

init arguments:
    L                   -- number of steps
    epsilon             -- stepsize
    logp                -- log(target distribution), avoiding numerical inaccuracies
    loglik              -- log(likelihood of the model)
    kinetic_energy_dist -- kinetic energy distribution p(p|x)

'''


def make_HMC_sampler(logp, L, epsilon, loglik=None):

    U = lambda x: -logp(x)    # potential energy U(x)
    K = ds.Normal(0.0, 1.0)   # kinetic energy K(p)

    def U_grad(x):
        x_ = Variable(x.data, requires_grad=True)
        U(x_).backward()
        return x_.grad

    '''Computes Hamiltonian, energy function for the joint state 
       of position x and momentum p'''

    def H(x, p):
        # p^T.p
        return p.pow(2).sum() / 2.0 - logp(x)

    '''Draws N samples x(i) from the target distribution  p(x) / Z using
       the Metropolis-Hastings algorithm with HMC proposals'''

    def sample_from_target(N, x_init, live=False):
        
        samples = torch.zeros(N, *x_init.shape)
        samples[0] = x_init
        acceptance_probs = torch.zeros(N - 1)

        for i in tqdm(range(N - 1)):

            x = samples[i]
            x0 = x

            '''1 - New momentum variable drawn from its distribution p(p|x) = K(p)'''
            p = K.sample(x.shape)
            p0 = p

            # half step
            p = p - epsilon / 2.0 * U_grad(x)

            '''2- Perform Metropolis update, using Hamiltonian dynamics to propose a new state'''
            # walk L steps
            for t in range(L):
                x = x + epsilon * p
                if t != L - 1:
                    p = p - epsilon * U_grad(x)
                else:
                    p = p - epsilon / 2.0 * U_grad(x)
            
            # inversibility
            p = -p

            
            # acceptance decision
            u = torch.rand(1)
            a = min(1.0, torch.exp(H(x0, p0) - H(x, p)))

            if bool(u < a):
                samples[i + 1] = x
            else:
                samples[i + 1] = x0

            # diagnostic information
            acceptance_probs[i] = a

            # check for nan's
            if torch.isnan(x).any():
                print('NaN occurred in x. Exiting.')
                break


        return samples, acceptance_probs

    '''Diagnostics'''

    # '''Plots autocorrelation of a row of HMC samples; ought to be approx. 0
    #    One sign of poor convergence is if the autocorrelation remains large.'''

    # def plot_autocorrelation(self, samples):
    #     plt.figure()
    #     autocorrelation_plot(samples)
    #     plt.show()

    # '''Plots acceptance probabilities; heuristic is that ought to be approx. 0.65
    #    One sign of poor mixing is if the acceptance rate is very small. If the 
    #    acceptance rate is large, you may not be moving very far'''

    # def plot_acceptance_probs(self, acceptance_probs):
    #     plt.plot(np.arange(1, len(acceptance_probs) + 1, 1),
    #              acceptance_probs, c='black')
    #     plt.ylim((-0.05, 1.05))
    #     plt.xlabel('iteration')
    #     plt.title(
    #         'HMC acceptance probs.: L={} / stepsize={}'.format(self.L, self.epsilon))
    #     plt.show()
    #     plt.close('all')

    #     plt.hist(acceptance_probs)
    #     plt.xlim((-0.05, 1.05))
    #     plt.xlabel('acceptance probability')
    #     plt.title(
    #         'Distribution of HMC acceptance probs.: L={} / stepsize={}'.format(self.L, self.epsilon))
    #     plt.show()

    # '''Plots log likelihood over iterations of HMC
    #    A log-likelihood that is clearly headed up is a sign of non-convergence; 
    #    a log-likelihood plot that is clearly headed down is a sign of a coding bug.'''

    # def plot_log_likelihoods(self, ws):
    #     if self.loglik is not None:
    #         lls = [self.loglik(w) for w in ws]
    #         plt.plot(np.arange(1, len(lls) + 1, 1), lls, c='black')
    #         plt.title(
    #             'Log likelihood: L={} / stepsize={}'.format(self.L, self.epsilon))
    #         plt.xlabel('iteration')
    #         plt.show()
    #     else:
    #         print('Need to instantiate HMC with a log likelihood function.')

    # '''Plots traces of m x n selected parameters over iterations
    #    Large temporal correlations is another sign of non-convergence.'''

    # def plot_selected_sample_traces(self, ws, selected, m, n):
    #     fig, axarr = plt.subplots(m, n)
    #     w_selected = ws[:, np.array(selected)]
    #     for i in range(m):
    #         for j in range(n):
    #             id = n * i + j
    #             axarr[i, j].set_title('w[{}]'.format(selected[id]), loc='left')
    #             axarr[i, j].plot(
    #                 np.arange(1, w_selected.shape[0] + 1, 1), w_selected[:, id])

    #     plt.suptitle(
    #         'Trace plot of {} neural network parameters over iterations'.format(m * n))
    #     plt.show()

    # ''' Plots marginal distribution of m x n selected parameters
    #     Note: on a large problem, we will almost never see a sampler move across modes 
    #     because it has to make a large number of unlikely moves.'''

    # def plot_selected_marginals(self, ws, selected, m, n):
    #     fig, axarr = plt.subplots(m, n)
    #     w_selected = ws[:, np.array(selected)]
    #     for i in range(m):
    #         for j in range(n):
    #             id = n * i + j
    #             axarr[i, j].set_title('w[{}]'.format(selected[id]), loc='left')
    #             axarr[i, j].hist(w_selected[:, id])

    #     plt.suptitle(
    #         'Sampled marginal distributions of {} neural network parameters over iterations'.format(m * n))
    #     plt.show()

    return sample_from_target


def main_HMC(all_experiments):

    for id, experiment in enumerate(all_experiments):

        print('Experiment {} / {}.'.format(id + 1, len(all_experiments)))

        '''BNN '''
        architecture = experiment['nn']['architecture']
        nonlinearity = experiment['nn']['nonlinearity']
        prior_ds = experiment['nn']['prior_ds']

        '''Data '''
        noise_ds = experiment['data']['noise_ds']
        X = experiment['data']['X']
        Y = experiment['data']['Y']
        X_plot = experiment['data']['X_plot']

        '''BbB settings'''
        batch_size = experiment['bbb']['batch_size']
        num_batches = int(torch.ceil(torch.tensor(
            X.shape[0] / batch_size))) if batch_size else 1

        '''Constraints'''
        gamma = experiment['bbb']['constrained']['gamma']
        tau = experiment['bbb']['constrained']['tau_tuple']
        violation_samples = experiment['bbb']['constrained']['violation_samples']
        constrained_region_sampler = experiment['bbb']['constrained']['constrained_region_sampler']
        constr = experiment['constraints']['constr']
        constr_plot = experiment['constraints']['plot']

        '''Define BNN (constrained and unconstrained)'''
        num_weights, forward, log_prob = \
            make_BNN(layer_sizes=architecture,
                    prior_ds=prior_ds,
                    noise_ds=noise_ds,
                    nonlinearity=nonlinearity,
                    num_batches=num_batches)
    
        '''Computes expected violation via constraint function, of distribution implied by param'''
        def violation(weights):
            x = constrained_region_sampler(violation_samples)
            y = forward(weights, x)

            tau_c, tau_g = tau
            c = torch.zeros(y.shape)
            for region in constr:
                x_consts, y_constr = region
                d = torch.ones(y.shape)
                for constraint in x_consts:
                    d *= sig(constraint(x, y), tau_c)
                for constraint in y_constr:
                    d *= psi(constraint(x, y), tau_c, tau_g)
                c += d
            l = gamma * c.sum() / y.numel()
            return l

        def target(weights):
            return log_prob(weights, X, Y)  - violation(weights) 


        '''HMC'''
        stepsize = experiment['hmc']['stepsize']
        steps = experiment['hmc']['steps']
        hmc_samples = experiment['hmc']['hmc_samples']
        burnin = experiment['hmc']['burnin']
        thinning = experiment['hmc']['thinning']

        darting = experiment['hmc']['darting']['bool']
        if darting:
            print('Darting HMC.')

            dct = experiment['hmc']['darting']
            dct['num_weights'] = num_weights
            
            sampler = make_darting_HMC_sampler(target, steps, stepsize, dct, loglik=None)
        else:
            print('HMC.')
            sampler = make_HMC_sampler(target, steps, stepsize, loglik=None)

        init = torch.max(torch.cat([torch.randn(1).abs(), torch.tensor([1]).float()])) \
               * torch.randn(1, num_weights) # batch of 1 for BNN weights
        samples, acceptance_probs = sampler(hmc_samples, init)
        samples = samples.squeeze()  # align batch dim
        print('Average acceptance probability after burnin: {}'.format(
            acceptance_probs[burnin:].mean()))

        # burnin and thinning 
        samples = samples[burnin:, :]  
        samples = samples[torch.arange(0, hmc_samples - burnin - 1, thinning), :]

        y_pred = forward(samples, X_plot)
        mean = y_pred.mean(0, keepdim=True)
        std = y_pred.std(0, keepdim=True)

        '''Approximate posterior predictive for test points'''

        fig, ax = plt.subplots()
        for p in constr_plot:
            ax.add_patch(p.get())
        ax.plot(X_plot.squeeze().repeat(y_pred.shape[0], 1).transpose(0, 1).numpy(),
                    y_pred.squeeze().transpose(0, 1).numpy(), 
                    c='blue',
                    alpha=0.02)
        ax.scatter(X.numpy(), Y.numpy(), c='black', marker='x')
        ax.set_title(
                'Function samples for {} BNN using HMC'.format(architecture))
        plt.show()

        # plt.plot(X_plot.squeeze().numpy(), mean.squeeze().numpy(), c='black')
        # plt.fill_between(X_plot.squeeze().numpy(),
        #                 (mean - 2 * std).squeeze().numpy(),
        #                 (mean + 2 * std).squeeze().numpy(),
        #                 color='black',
        #                 alpha=0.3)
        # plt.scatter(X.numpy(), Y.numpy(), c='black', marker='x')
        # plt.title(
        #     'Posterior predictive for {} BNN using HMC'.format(architecture))
        # plt.show()
