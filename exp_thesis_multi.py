import matplotlib.pyplot as plt
import math
import time
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable

import copy

from utils import *
from plot import *
from main_vi import main_vi
from main_hmc import main_hmc


'''
Preliminary definitions
'''


class ReLUActivation(torch.autograd.Function):

    def __str__(self):
        return 'relu'

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


def rbf(x): 
    return torch.exp(- x.pow(2))

relu = ReLUActivation.apply

def tanh(x): 
    return x.tanh(x)

def softrelu(x): 
    return torch.log(1.0 + x.exp())


# Data

N, n_dim = 10, 1


def ground_truth(x):
    return - 0.6666 * x.pow(4) + x.pow(2) + 1


X = torch.tensor([-2, -1.8, -1, 1, 1.8, 2]).unsqueeze(1)
Y = ground_truth(X)

X_plot = torch.linspace(-5, 5, steps=100).unsqueeze(1)
Y_plot = ground_truth(X_plot)

X_id = torch.tensor([-1.9, -1.5, 0.5, 0.0, 0.5, 1.5, 1.9]).unsqueeze(1)
Y_id = ground_truth(X_id)

X_ood = torch.tensor([-4, -3, -2.5, 2.5, 3, 4]).unsqueeze(1)
Y_ood = ground_truth(X_ood)


'''
Constraints: f(x, y) <= 0

Have to be broadcastable!!
i.e. x is (datapoints, in_dim)
     y is (weights samples, datapoints, out_dim)


'''

# DON'T FORGET:
#   1)  Double check constraint def.
#   2)  constaint sampler
#   3)  plot rectangle



# center box
def x_c00(x, y): return x - 0.5
def x_c01(x, y): return - x - 0.5
def y_c02(x, y): return y - 4.5
def y_c03(x, y): return - y + 2.5



constr = [
    [x_c00, x_c01, y_c02, y_c03],
]


def constrained_region_sampler(s):
    # out = torch.cat([
    #     ds.Uniform(3, 4).sample(
    #         sample_shape=torch.Size([int(round(s / 3)), 1])),
    #     ds.Uniform(-4, -
    #                3).sample(sample_shape=torch.Size([int(round(s / 3)), 1])),
    #     ds.Uniform(-0.5, 0.5).sample(sample_shape=torch.Size([int(round(s / 3)), 1]))], dim=0)

    out = ds.Uniform(-0.5, 0.5).sample(sample_shape=torch.Size([s, 1]))
    return out


'''
Experiment dictionary 

Constraints are of the form
    [region_0, region_1, ...]
    with region = ([x_constraint_0, ..], [y_constraint_0, ..])

'''
all_experiments = []


prototype = {
    'title': 'thesis_multi_dart',
    'nn': {
        'architecture': [n_dim, 20, 1],
        'nonlinearity': rbf,
        'prior_ds': ds.Normal(0.0, 3.5),
    },
    'data': {
        'noise_ds': ds.Normal(0.0, 0.1),
        'plt_x_domain': (-5, 5),
        'plt_y_domain': (-12, 12),
        'integral_constrained_region': 0,
        'X':  X,
        'Y':  Y,
        'X_plot': X_plot,
        'Y_plot': Y_plot,
        'X_v_id': X_id,
        'Y_v_id': Y_id,
        'X_v_ood': X_ood,
        'Y_v_ood': Y_ood,
    },
    'constraints': {
        'constr': constr,
        'plot': [
            # DrawRectangle(bottom_left=(3, -0.5), top_right=(4, 1.5)),
            DrawRectangle(bottom_left=(-0.5, 2.5), top_right=(0.5, 4.5)),
           #  DrawRectangle(bottom_left=(-4, -3.5), top_right=(-3, -1.5)),
        ],
    },
    'vi': { # alg options: bbb, npv
        'alg': 'npv',
        'bbb_param' : {
            'initialize_q': {
                'mean': 1.0,  # * torch.randn
                'std': -2.5,  # * torch.ones
            },
        },
        'npv_param': {
            'mixtures' : 10, 
            'initialize_q': {
                'mean': 1.0,  # * torch.randn
                'std': 0.0,  # * torch.ones
            },
        },
        'rv_samples': 100,
        'batch_size': 0,  # batch_size = 0 implies full dataset training
        'lr' : 0.01,
        'regular': {
            'iterations': 200,
            'restarts': 1,
            'reporting_every_': 10,
            'cores_used': 1,
        },
        'constrained': {
            'iterations': 200,
            'restarts': 1,
            'reporting_every_': 10,
            'cores_used': 1,
            'violation_samples': 5000,
            'tau_tuple': (15.0, 2.0),
            'gamma': 1000,
            'constrained_region_sampler': constrained_region_sampler,
        },
        'posterior_predictive_analysis': {
            'posterior_samples': 50,
            'constrained_region_samples_for_pp_violation': 50,
        }

    },
    'hmc': {
        'load_saved' : False,
        'load_from' : 'thesis_multi_dart_v0',
        'constrained' : True,
        'gamma' : 5000,
        'darting': {
            'bool': True,
            'preprocessing': {
                'load_saved': False,
                'load_from': 'thesis_multi_dart_v0',
                'norm_f': 0.3,
                'random_restart_scale' : 3,
                'searched_modes': 20,
                'mode_searching_convergence': 0.01,
                'n_darting_regions': 3,
            },
            'algorithm': {
                'darting_region_radius': 2.0e1,
                'p_check': 0.03,
            },
        },
        'stepsize': 0.005,
        'steps': 20,
        'hmc_samples': 2500,
        'burnin': 1000,
        'thinning': 3,
    },
    'experiment': {
        'run_regular_vi': True,
        'run_constrained_vi': False,
        'multithread_computation': False,
        'compute_held_out_loglik_id': True,
        'compute_held_out_loglik_ood': False,
        'compute_RMSE_id': True,
        'compute_RMSE_ood': False,
        'show_function_samples': True,
        'show_posterior_predictive': True,
        'show_posterior_predictive_ind': (True, 500),
        'show_plot_training_evaluations': True,
        'show_constraint_function_heatmap': False,
        'plot_size': (6, 4),  # width, height (inches)
    },
}



all_experiments.append(prototype)

# main_vi(all_experiments)
main_hmc(all_experiments)
