import matplotlib
matplotlib.rcParams['text.usetex'] = True
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
from pp_violation import compute_posterior_predictive_violation_hmc

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
            if i % 200 == 0 and i > 0:
                print('200 ave acceptance prob: {}'.format(torch.mean(acceptance_probs[i - 200: i])))

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


def main_hmc(all_experiments):

    for id, experiment in enumerate(all_experiments):

        print('Experiment {} / {}.'.format(id + 1, len(all_experiments)))
        print(experiment['title'])
        '''BNN '''
        architecture = experiment['nn']['architecture']
        nonlinearity = experiment['nn']['nonlinearity']
        prior_ds = experiment['nn']['prior_ds']

        '''Data '''
        noise_ds = experiment['data']['noise_ds']
        X = experiment['data']['X']
        Y = experiment['data']['Y']
        X_plot = experiment['data']['X_plot']
        Y_plot = experiment['data']['Y_plot']
        X_v_id = experiment['data']['X_v_id']
        Y_v_id = experiment['data']['Y_v_id']
        X_v_ood = experiment['data']['X_v_ood']
        Y_v_ood = experiment['data']['Y_v_ood']
        plt_size = experiment['data']['plt_size']

        '''BbB settings'''
        batch_size = experiment['vi']['batch_size']
        num_batches = int(torch.ceil(torch.tensor(
            X.shape[0] / batch_size))) if batch_size else 1

        '''Constraints'''
        gamma = experiment['hmc']['gamma']
        tau = experiment['vi']['constrained']['tau_tuple']
        violation_samples = experiment['vi']['constrained']['violation_samples']
        constrained_region_sampler = experiment['vi']['constrained']['constrained_region_sampler']
        constr = experiment['constraints']['constr']
        plot_patch = experiment['constraints']['plot_patch']
        plot_between = experiment['constraints']['plot_between']

        '''Make directory for results'''
        current_directory = make_unique_dir(experiment, method='hmc')
        load_saved = experiment['hmc']['load_saved']
        load_from = experiment['hmc']['load_from']

        
        '''Define BNN (constrained and unconstrained)'''
        num_weights, forward, log_prob = \
            make_BNN(layer_sizes=architecture,
                    prior_ds=prior_ds,
                    noise_ds=noise_ds,
                    nonlinearity=nonlinearity,
                    num_batches=num_batches)

        '''Compute RMSE of validation dataset given optimizated params'''
        def compute_rmse(x, y, samples):
            samples = forward(samples, x)
            pred = samples.mean(0)  # prediction is mean
            rmse = (pred - y).pow(2).mean(0).pow(0.5)
            return rmse.item()

        '''Computes held-out log likelihood of x,y given distribution implied by samples'''
        def held_out_loglikelihood(x, y, samples):
            samples = forward(samples, x)
            mean = samples.mean(0).squeeze()
            std = samples.std(0).squeeze()
            return ds.Normal(mean, std).log_prob(y).sum().item()


        '''Computes expected violation via constraint function, of distribution implied by param'''
        def violation(weights):
            x = constrained_region_sampler(violation_samples)
            y = forward(weights, x)
            tau_c, tau_g = tau
            c = torch.zeros(y.shape)
            for region in constr:
                d = torch.ones(y.shape)
                for constraint in region:
                    d *= psi(constraint(x, y), tau_c, tau_g)
                c += d
            
            if experiment['hmc']['max_violation_heuristic']:
                l = gamma * c.max() 
                # returns max violation along y.shape 
                # empirically better when using preprocessing procedure for darting hmc
            else:
                l = gamma * c.sum() / y.numel()
            return l

        if experiment['hmc']['constrained']:
            def target(weights):
                return log_prob(weights, X, Y)  - violation(weights) 
        else:
            def target(weights):
                return log_prob(weights, X, Y)
      


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
            
            sampler = make_darting_HMC_sampler(target, steps, stepsize, dct, loglik=None,
                                               forward=forward, experiment=experiment, current_directory=current_directory)
        else:
            print('HMC.')
            sampler = make_HMC_sampler(target, steps, stepsize, loglik=None)

        if load_saved:
            
            samples = torch.load('experiment_results/' + load_from  + '/hmc/' + experiment['title'] + '_samples.pt')

        else:
            init = torch.max(torch.cat([torch.randn(1).abs(), torch.tensor([1]).float()])) \
                * torch.randn(1, num_weights) # batch of 1 for BNN weights
            samples, acceptance_probs = sampler(hmc_samples, init)
            samples = samples.squeeze()  # align batch dim
            print('Average acceptance probability after burnin: {}'.format(
                acceptance_probs[burnin:].mean()))

            # burnin and thinning 
            samples = samples[burnin:, :]  
            samples = samples[torch.arange(0, hmc_samples - burnin - 1, thinning), :]


            # save
            torch.save(samples, current_directory + '/hmc/' +
                       experiment['title'] + '_samples.pt')

        y_pred = forward(samples, X_plot)
        mean = y_pred.mean(0, keepdim=True)
        std = y_pred.std(0, keepdim=True)

        '''Print table info to out'''
        pcv = compute_posterior_predictive_violation_hmc(samples, forward, experiment)
        rmse_id = compute_rmse(X_v_id, Y_v_id, samples)
        rmse_ood = compute_rmse(X_v_ood, Y_v_ood, samples)
        held_out_ll_id = held_out_loglikelihood(X_v_id, Y_v_id, samples)
        held_out_ll_ood = held_out_loglikelihood(X_v_ood, Y_v_ood, samples)

        print('PCV: {}\nHeld-out LogLik ID: {}\nRMSE ID: {}\nHeld-out LogLik OOD: {}\nRMSE OOD: {}'.format(
            pcv, held_out_ll_id, rmse_id, held_out_ll_ood, rmse_ood))

        '''Approximate posterior predictive for test points'''

        fig, ax = plt.subplots(figsize=plt_size)
        for p in plot_patch:
            ax.add_patch(p.get())
        for x, y1, y2 in plot_between:
            ax.fill_between(x.squeeze().numpy(),
                            y1.squeeze().numpy(), y2.squeeze().numpy(), alpha=0.3, color='red')
        ax.plot(X_plot.squeeze().numpy(), Y_plot.squeeze().numpy(), c='black', linestyle=':')
        ax.plot(X_plot.squeeze().repeat(y_pred.shape[0], 1).transpose(0, 1).numpy(),
                    y_pred.squeeze().transpose(0, 1).numpy(), 
                    c='blue',
                    alpha=0.02,
                    linewidth=2)
        ax.scatter(X.numpy(), Y.numpy(), c='black', marker='x')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # ax.set_title(
        #         'Function samples for {} BNN using HMC'.format(architecture))
        # plt.savefig(current_directory + '/hmc/' + experiment['title'] + '.png',
        #             format='png', frameon=False, dpi=400)
        
        ax.set_xlim(experiment['data']['plt_x_domain'])
        ax.set_ylim(experiment['data']['plt_y_domain'])
        # ax.spines['bottom'].set_bounds(-4.5, 4.5)
        bot, top = ax.get_ylim()
        # ax.spines['left'].set_bounds(bot + 0.5, top - 0.5)

        ax.set_xlabel(r"$x$", fontsize=12)
        tmp = ax.set_ylabel(r"$\phi(x; \mathcal{W})$", fontsize=12)
        tmp.set_rotation(0)
        ax.yaxis.set_label_coords(-0.15, 0.50)
        plt.gcf().subplots_adjust(left=0.2)
        
        plt.savefig(current_directory + '/hmc/' + experiment['title'] + '_particles.png',
                    format='png', frameon=False, dpi=400)
        plt.show()
        plt.close('all')


        #
        fig, ax = plt.subplots(figsize=plt_size)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        for p in plot_patch:
            ax.add_patch(p.get())
        for x, y1, y2 in plot_between:
            ax.fill_between(x.squeeze().numpy(),
                            y1.squeeze().numpy(), y2.squeeze().numpy(), alpha=0.3, color='red')
        ax.plot(X_plot.squeeze().numpy(), Y_plot.squeeze().numpy(), c='black', linestyle=':')
        ax.plot(X_plot.squeeze().numpy(), mean.squeeze().numpy(), c='blue')
        ax.fill_between(X_plot.squeeze().numpy(),
                        (mean - 1 * std).squeeze().numpy(),
                        (mean + 1 * std).squeeze().numpy(),
                        color='blue',
                        alpha=0.3)
        ax.fill_between(X_plot.squeeze().numpy(),
                         (mean - 2 * std).squeeze().numpy(),
                         (mean + 2 * std).squeeze().numpy(),
                         color='blue',
                         alpha=0.2)
        ax.scatter(X.numpy(), Y.numpy(), c='black', marker='x')
        ax.set_xlim(experiment['data']['plt_x_domain'])
        ax.set_ylim(experiment['data']['plt_y_domain'])
        # ax.spines['bottom'].set_bounds(-4.5, 4.5)
        bot, top = ax.get_ylim()
        # ax.spines['left'].set_bounds(bot + 0.5, top - 0.5)
        ax.set_xlabel(r"$x$", fontsize=12)
        
        tmp = ax.set_ylabel(r"$\phi(x; \mathcal{W})$", fontsize=12)
        tmp.set_rotation(0)

        ax.yaxis.set_label_coords(-0.15, 0.50)
        plt.gcf().subplots_adjust(left=0.2)
        # ax.set_title(
        #     'Posterior predictive for {} BNN using HMC'.format(architecture))
        plt.savefig(current_directory + '/hmc/'+ experiment['title'] + '_filled.png',
                    format='png', frameon=False, dpi=400)
        plt.show()
