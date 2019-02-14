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

from kmeans import kmediods
import numpy as np

'''
Generalized Darting Monte Carlo with HMC
http://proceedings.mlr.press/v2/sminchisescu07a/sminchisescu07a.pdf


'''


def make_darting_HMC_sampler(logp, L, epsilon, dct, loglik=None):
    
    '''Preprocessing'''
    print('Searching p(x) for modes...')

    num_weights = dct['num_weights']
    do_preprocessing = dct['preprocessing']['bool']
    searched_modes = dct['preprocessing']['searched_modes']
    search_SGD_lr = dct['preprocessing']['search_SGD_lr']
    mode_searching_convergence = dct['preprocessing']['mode_searching_convergence']
    show_mode_cluster_analysis = dct['preprocessing']['show_mode_cluster_analysis']
    pca_dim = dct['preprocessing']['pca_dim']
    k_means_tried = dct['preprocessing']['k_means_tried']
    k_means_selected = dct['preprocessing']['k_means_selected']

    if do_preprocessing:
        # loss
        def loss_func(w):
            return - logp(w)


        # find m modes of log p(x)
        modes = torch.zeros(searched_modes, num_weights)
        for m in tqdm(range(searched_modes)):
            scale = 10
            w = scale * torch.max(torch.cat([torch.randn(1).abs(), torch.tensor([1]).float()])) \
                    * torch.randn(1, num_weights)  # batch of 1 for BNN weights
            w.requires_grad_(True)

            optimizer = optim.Adam([w], lr=0.001)
            ps = [(- logp(w)).item()]

            for t in range(10000):
                # optim
                optimizer.zero_grad()
                loss = loss_func(w)
                loss.backward()
                optimizer.step()
                
                # check for convergence
                ps.append(loss.item())
                if t > 6:
                    diff = torch.mean(torch.tensor(ps[-6:-2]) - torch.tensor(ps[-5:-1]), dim=0)
                    if diff < mode_searching_convergence:
                        break
                        

            if torch.any(torch.isnan(torch.tensor(ps))):
                print('NaNs encountered in optimization. Consider decreasing learning rate.')
                
            
        
            modes[m] = w.squeeze()

        modes = modes.detach()

        X = runPCA(modes, k=pca_dim)

        # pre-analysis of preprocessing parameters
        if show_mode_cluster_analysis:

            # plot modes using PCA
            X_plot = runPCA(modes, k=2)
            plt.scatter(X_plot[:, 0].numpy(), X_plot[:, 1].numpy())
            plt.title('PCA of log p(x) modes')
            plt.show()
            plt.close('all')

            # search over possible cluster sizes
            sses = []
            print('Clustering modes...')
            for k in tqdm(range(2, k_means_tried)):
                best_sse = torch.tensor(float('inf'))
                for _ in range(10):
                    clusters_index, centers = kmediods(X.clone(), k)
                    sse = 0
                    for i in range(k):
                        if i not in clusters_index:
                            break
                        selected = X[clusters_index == i]
                        sse += (selected - centers[i].unsqueeze(0)).pow(2).sum()
                    if sse < best_sse:
                        best_sse = sse
                sses.append(best_sse.item())

            plt.plot(list(range(2, k_means_tried)), sses)
            plt.title('Ave. Sum of Squared Error K-Mediods over 10 runs in {} PCA dims'.format(pca_dim))
            plt.xlabel('Number of clusters')
            plt.show()


        # actually find good clustering of modes for darting regions
        best_sse = torch.tensor(float('inf'))
        darting_points = None
        for _ in range(20):
            clusters_index, centers = kmediods(X.clone(), k_means_selected)
            sse = 0
            for i in range(k_means_selected):
                if i not in clusters_index:
                    break
                selected = X[clusters_index == i]
                sse += (selected - centers[i].unsqueeze(0)).pow(2).sum()
            if sse < best_sse:
                best_sse = sse
                darting_points = centers
    
    # Use pre-loaded darting points
    else: 
        darting_points = searched_modes = dct['preprocessing']['preprocessed_regions']


    # import darting settings
    darting_region_radius = dct['algorithm']['darting_region_radius']

    # Samples uniformly at random from ball around center
    def sample_Unif_ball(center, radius=darting_region_radius):
        u = torch.rand(1) * radius
        dir = torch.randn(center.shape) 
        dir_ = dir / torch.norm(dir, p=2)
        return center + u * dir_

    # Samples from darting region k
    def sample_region(k):
        return sample_Unif_ball(darting_points[k])

    print(sample_region(2))

    # Ready to iplement generalized darting mcmc

    exit(0)

    '''Generalized Darting Monte Carlo with HMC'''


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


    return sample_from_target



'''Run PCA on matrix mat'''

def runPCA(mat, k=2):

    # standardize
    mu = mat.mean(0, keepdim=True)
    std = mat.std(0, keepdim=True)
    normalized = (mat - mu) / std

    # singular value decomposition
    U, S, V = torch.svd(normalized.transpose(0, 1))
    return torch.mm(normalized, U[:, :k])
