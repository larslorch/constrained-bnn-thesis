import matplotlib.pyplot as plt
import math 
import time

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds 
from torch.autograd import Variable

from bbb import bayes_by_backprop_variational_inference 


'''
Implements BNN functions
based on https://github.com/HIPS/autograd/blob/master/examples/

Arguments:
    layer_sizes      architecture of BNN, e.g. [1, 10, 10, 1]
    prior_ds         prior distribution p(W)
    noise_ds         noise distribution for likelihood p(Y|X,W)
    nonlinearity     nonlinearity function
    num_batches      number of minibatches in optimization, necessary for
                     rescaling KL[q(w) | p(w)] as explained in 
                     https://arxiv.org/pdf/1505.05424.pdf  
'''

def make_BNN(layer_sizes, 
             prior_ds,
             noise_ds,
             nonlinearity,
             num_batches=1):

    layer_shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    num_weights = sum((m + 1) * n for m, n in layer_shapes)

    '''
    Returns generator that returns for every layer (in: m, out: n)
        W  of shape (num_weight_samples, m, n)
        b  of shape (num_weight_samples, 1, n)
    weights is shape (num_weight_samples, num_weights)
    '''

    def unpack_layers(weights):

        num_weight_samples = weights.shape[0]
        for m, n in layer_shapes:
            # yield next layer in correct shape
            yield weights[:,      : m * n]    .reshape((num_weight_samples, m, n)),\
                  weights[:, m * n: m * n + n].reshape((num_weight_samples, 1, n))

            # remove layer
            weights = weights[:, (m + 1) * n :]


    '''
    Computes forward pass of neural network given weights and inputs
    weights is shape (num_weight_samples, num_weights)
    inputs  is shape (num_datapoints, D)
    '''

    def forward(weights, inputs):
        
        # expand for batch processing
        num_weight_samples = weights.shape[0]
        inputs = inputs.expand(num_weight_samples, *inputs.shape)

        for W, b in unpack_layers(weights):
            # inputs is (num_weight_samples, N, m)
            # W      is (num_weight_samples, m, n)
            # b      is (num_weight_samples, 1, n)
            outputs = torch.einsum('mnd,mdo->mno', [inputs, W]) + b        

            # outputs is (num_weight_samples (broadcast), N, n)
            inputs = nonlinearity(outputs)

        return outputs

    '''
    Computes log model probability of weights given inputs and targets
    inputs, targets are shape (num_datapoints, D)
    '''
    def log_prob(weights, inputs, targets):
        log_prior = prior_ds.log_prob(weights).sum()
        preds = forward(weights, inputs)
        log_lik = noise_ds.log_prob(preds - targets).sum()
        return log_prior / num_batches + log_lik

    return num_weights, forward, log_prob


if __name__ == '__main__':

    '''
    Specify inference problem by 
    unnormalized log-posterior of BNN weights
    '''

    def rbf(x): return torch.exp(- x.pow(2))
    def relu(x): return x.clamp(min=0.0)
    def tanh(x): return x.tanh(x)

    num_weights, forward, log_prob = \
        make_BNN(layer_sizes=[1, 20, 20, 1], 
                 prior_ds=ds.Normal(0.0, 10.0),
                 noise_ds=ds.Normal(0.0, 0.01),
                 nonlinearity=relu)

    inputs, targets = None, None # data here

    def log_posterior(weights): return log_prob(weights, inputs, targets)

    
    '''
    Variational objective via BBVI
    '''

    num_samples = 20

    variational_objective, unpack_params = \
        bayes_by_backprop_variational_inference(log_posterior, num_samples=num_samples)
    
    '''
    Plotting
    '''

    fig = plt.figure(figsize=(10, 6), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    def callback(params, t):
        print("Iteration {} lower bound {}".format(
            t, - variational_objective(params)))

        mean, log_std = unpack_params(params)
        torch.random.manual_seed(0)
        sample_weights = torch.randn(10, num_weights) * torch.exp(log_std) + mean
        plot_inputs = torch.linspace(-8, 8, steps=400)
        outputs = forward(sample_weights, plot_inputs.unsqueeze(1))

        # Plot data and functions.
        plt.cla()
        ax.plot(inputs.reshape(-1).numpy(), targets.reshape(-1).numpy(), 'bx')
        ax.plot(plot_inputs.numpy(), outputs[:, :, 0].transpose(0, 1).detach().numpy())
        ax.set_ylim([-2, 3])
        plt.draw()
        plt.pause(1.0/60.0)
    
    '''
    Run BBVI
    '''
    epochs = 100

    init_mean = torch.randn(num_weights, 1)
    init_log_std = -5 * torch.ones(num_weights, 1)
    init_var_params = torch.cat([init_mean, init_log_std], dim=1)

    print("Optimizing variational parameters...")

    params = Variable(init_var_params, requires_grad=True)
    optimizer = optim.Adam([params], lr=0.1)
    for t in range(epochs):
        optimizer.zero_grad()
        loss = variational_objective(params)
        loss.backward()
        optimizer.step()
        callback(params, t)


