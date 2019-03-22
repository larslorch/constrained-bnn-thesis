import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable

from bnn import make_BNN
from utils import *

from clustering import hac_reps, dist_min, dist_max, dist_centroid, dist_ave
import numpy as np

'''
Generalized Darting Monte Carlo with HMC
http://proceedings.mlr.press/v2/sminchisescu07a/sminchisescu07a.pdf


'''


def make_darting_HMC_sampler(logp, L, epsilon, dct, loglik=None, forward=None, experiment=None, current_directory=None):
    
    '''Preprocessing'''
    print('Searching p(x) for modes...')

    num_weights = dct['num_weights']
    load_saved = dct['preprocessing']['load_saved']
    load_from = dct['preprocessing']['load_from']
    searched_modes = dct['preprocessing']['searched_modes']
    mode_searching_convergence = dct['preprocessing']['mode_searching_convergence']
    norm_f = dct['preprocessing']['norm_f']
    scale = dct['preprocessing']['random_restart_scale']
    n_darting_regions = dct['preprocessing']['n_darting_regions']

    if not load_saved:
        # loss
        def loss_func(w):
            return - logp(w)


        # find m modes of log p(x)
        modes = torch.zeros(searched_modes, num_weights)
        for m in tqdm(range(searched_modes)):
            w = scale * torch.max(torch.cat([torch.randn(1).abs(), torch.tensor([1]).float()])) \
                    * torch.randn(1, num_weights)  # batch of 1 for BNN weights
            w.requires_grad_(True)

            optimizer = optim.Adam([w], lr=0.01)
            ps = [(- logp(w)).item()]

            for t in range(10000):
                # optim
                optimizer.zero_grad()
                loss = loss_func(w)
                loss.backward()
                optimizer.step()
                
                # check for convergence
                ps.append(loss.item())
                if t > 50:
                    diff = torch.mean(torch.tensor(ps[-49:-2]) - torch.tensor(ps[-48:-1]), dim=0)
                    if diff < mode_searching_convergence:
                        break

            print('log p(w) = {}'.format(logp(w.detach())))

            if torch.any(torch.isnan(torch.tensor(ps))):
                print('NaNs encountered in optimization. Consider decreasing learning rate.')
                
            modes[m] = w.squeeze()

        cutoff = 0.5
        modes = modes.detach()
        logp_modes = torch.zeros(modes.shape[0])
        for i in range(modes.shape[0]):
            logp_modes[i] = logp(modes[i].unsqueeze(0))
        _, indices = torch.topk(logp_modes, math.floor(modes.shape[0] * cutoff))
        modes = modes[indices]
        # remove lowest quarter

        darting_points = hac_reps(
            modes, n_darting_regions, dist_centroid, f=norm_f, verbose=True)
        
        print('log p of darting points:  ', end='')
        for j in range(n_darting_regions):
            print('{}'.format(
                logp(darting_points[j].unsqueeze(0)).item()))
        print()


        torch.save(darting_points, current_directory + '/hmc/' + experiment['title'] + '_preprocessing.pt' )
        
    # Use pre-loaded darting points
    else: 
        darting_points = torch.load(
            'experiment_results/' + load_from + '/hmc/' + experiment['title'] + '_preprocessing.pt')
       
    # darting settings
    darting_region_radius = dct['algorithm']['darting_region_radius']
    p_check = torch.tensor(dct['algorithm']['p_check'])


    # Samples uniformly at random from ball around center
    def sample_Unif_ball(center, radius=darting_region_radius):
        u = torch.rand(1) * radius
        dir = torch.randn(center.shape) 
        dir_ = dir / torch.norm(dir, p=2)
        return center + u * dir_

    # Samples from darting region k
    def sample_region(k):
        return sample_Unif_ball(darting_points[k])

    # Number of regions that contain particle
    def get_n(x, radius=darting_region_radius):
        return (torch.norm(x - darting_points, p=2, dim=1) <= radius).sum().float()


    '''Generalized Darting Monte Carlo with HMC'''

    print('log p of darting peaks:  ', end='')
    for j in range(n_darting_regions):
        print('{}, '.format(round(logp(darting_points[j].unsqueeze(0)).item())), end='')
    print()

    fig, ax = plt.subplots()
    X_plot = experiment['data']['X_plot']
    y_pred = forward(darting_points, X_plot)
    for p in experiment['constraints']['plot_patch']:
        ax.add_patch(p.get())
    for x, y1, y2 in experiment['constraints']['plot_between']:
        ax.fill_between(x.squeeze().numpy(),
                        y1.squeeze().numpy(), y2.squeeze().numpy(), alpha=0.3, color='red')

    ax.plot(X_plot.squeeze().repeat(y_pred.shape[0], 1).transpose(0, 1).numpy(),
            y_pred.squeeze().transpose(0, 1).numpy())
    ax.set_title(
        'Functions corresponding to darting modes')
    plt.show()
    plt.close('all')


    '''HMC'''

    def U(x): return -logp(x)    # potential energy U(x)
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
        successful_darts = 0
        unsuccessful_darts = 0

        # burned in init at mode
        samples[0] = darting_points[0]

        for i in tqdm(range(N - 1)):
            
            '''Check whether we dart or not'''
            u_1 = torch.rand(1)
            n_x = get_n(samples[i])

            if u_1 < p_check and n_x == 0:
                print('Not in region. ')

            if u_1 < p_check and n_x > 0:

                print('Darting... ')

                x = samples[i]

                '''2 - Sample a new region according to relative volumes'''

                # here: Uniform because volumes are the same size
                probs = torch.ones(darting_points.shape[0])
                idx = ds.Categorical(probs=probs).sample(torch.Size([1]))
                

                '''3 - Propose new location (here: uniformly at random, deterministically also possible)'''
                prop = sample_region(idx)
                prop = darting_points[idx] # test to see if samples get accepted

                '''4 - Identify the number of regions n(t) that contain the proposed sample.'''
                n_t = get_n(prop)

                '''5 - Reject/Accept'''
                log_p_accept = torch.min(torch.tensor([0,
                    torch.log(n_x) + logp(prop) - torch.log(n_t) - logp(x)
                ]))
                u_2 = torch.rand(1)
                
                s = '|p accept = {} |'.format(round(log_p_accept.exp().item(), 3))
                s += 'log p(x) = {} |'.format(round(logp(x).item()))
                s += 'log p(prop) = {} |'.format(round(logp(prop).item()))
                print(s, end='')

                if u_2 < log_p_accept.exp():
                    print('.. ACCEPTED. ')
                    # accept
                    samples[i + 1] = prop
                    successful_darts += 1

                else:
                    print('.. rejected. ')
                    # reject
                    samples[i + 1] = x
                    unsuccessful_darts += 1

            else:

                '''Perform one step of local HMC sampler'''
                
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

        print('Successful darts: {}'.format(successful_darts))
        print('Rejected darting attempts: {}'.format(unsuccessful_darts))

        return samples, acceptance_probs


    return sample_from_target



'''Run PCA on matrix mat'''

def runPCA(mat, k=2):

    # standardize
    mu = mat.mean(0, keepdim=True)
    centered = mat - mu

    # singular value decomposition
    U, S, V = torch.svd(centered.transpose(0, 1))
    return torch.mm(centered, U[:, :k])
