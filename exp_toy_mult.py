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
from main import main
from hmc import main_HMC


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


def rbf(x): return torch.exp(- x.pow(2))
relu = ReLUActivation.apply
def tanh(x): return x.tanh(x)
def softrelu(x): return torch.log(1.0 + x.exp())


# Data

N, n_dim = 10, 1

def ground_truth(x):
    return - x.pow(4) + 3 * x.pow(2) + 1
    
    
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

# right
def x_c00(x, y): return x - 4
def x_c01(x, y): return - x + 3
def y_c02(x, y): return y - 1.5
def y_c03(x, y): return - y - 0.5

# center
def x_c10(x, y): return x - 0.5
def x_c11(x, y): return - x - 0.5
def y_c12(x, y): return y - 4.5
def y_c13(x, y): return - y + 2.5

# left
def x_c20(x, y): return x + 3
def x_c21(x, y): return - x - 4
def y_c22(x, y): return y + 1.5
def y_c23(x, y): return - y - 3.5
    

constr = [
    [x_c00, x_c01, y_c02, y_c03],
    [x_c10, x_c11, y_c12, y_c13],
    [x_c20, x_c21, y_c22, y_c23]
]


def constrained_region_sampler(s):
    out = torch.cat([
        ds.Uniform(3, 4).sample(sample_shape=torch.Size([int(round(s / 3)), 1])),
        ds.Uniform(-4, -3).sample(sample_shape=torch.Size([int(round(s / 3)), 1])),
        ds.Uniform(-0.5, 0.5).sample(sample_shape=torch.Size([int(round(s / 3)), 1]))], dim=0)
    
    # out = ds.Uniform(-0.5, 0.5).sample(sample_shape=torch.Size([s, 1]))
    return out


'''
Experiment dictionary 

Constraints are of the form
    [region_0, region_1, ...]
    with region = ([x_constraint_0, ..], [y_constraint_0, ..])

'''
all_experiments = []


prototype = {
    'title': '6pt_toy_example_mult',
    'nn': {
        'architecture': [n_dim, 15, 1],
        'nonlinearity': rbf,
        'prior_ds': ds.Normal(0.0, 3.0),
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
            DrawRectangle(bottom_left=(3, -0.5), top_right=(4, 1.5)),
            DrawRectangle(bottom_left=(-0.5, 2.5), top_right=(0.5, 4.5)),
            DrawRectangle(bottom_left=(-4, -3.5), top_right=(-3, -1.5)),
        ],
    },
    'bbb': {
        'BbB_rv_samples': 100,
        'batch_size': 0,  # batch_size = 0 implies full dataset training
        'regular': {
            'iterations': 5000,
            'restarts': 4,
            'reporting_every_': 50,
            'cores_used': 1,
        },
        'constrained': {
            'iterations': 1000,
            'restarts': 1,
            'reporting_every_': 50,
            'cores_used': 1,
            'violation_samples': 5000,
            'tau_tuple': (15.0, 2.0),
            'gamma': 1000,
            'constrained_region_sampler': constrained_region_sampler,
        },
        'initialize_q': {
            'mean': 1.0,  # * torch.randn
            'std': -3.0,  # * torch.ones
        },
        'posterior_predictive_analysis': {
            'posterior_samples': 50,
            'constrained_region_samples_for_pp_violation': 50,
        }

    },
    'hmc' : {
        'darting' : {
            'bool' : True,
            'preprocessing' : {
                'bool' : False,
                'show_mode_cluster_analysis': False,
                'searched_modes': 20,
                'mode_searching_convergence': 0.005,
                'n_darting_regions': 10,
                'file_name' : 'dart_toy_mult_great'
            },
            'algorithm' : {
                'darting_region_radius' : 2.0e1,  
                'p_check' : 0.03,
            },
        },
        'stepsize' : 0.01,
        'steps' : 30,
        'hmc_samples' : 2000,
        'burnin' : 0,
        'thinning' : 3,
    },
    'experiment': {
        'run_regular_BbB': True,
        'run_constrained_BbB': False,
        'multithread_computation': False,
        'compute_held_out_loglik_id': True,
        'compute_held_out_loglik_ood': False,
        'compute_RMSE_id': True,
        'compute_RMSE_ood': False,
        'show_function_samples': True,
        'show_posterior_predictive': True,
        'show_plot_training_evaluations': True,
        'show_constraint_function_heatmap': False,
        'plot_size': (6, 4),  # width, height (inches)
    },
}


all_experiments.append(prototype)

# main(all_experiments)
main_HMC(all_experiments)




# returns inputs X s.t. x in X is from constrained X region
# def constrained_X_sampler(s):
#     # via rejection sampling
#     samples = []
#     while len(samples) < s:
#         z = 3 * np.random.normal(size=n_dim)
#         valid = True
#         for region in constr:
#             x_consts, _ = region
#             for constraint in x_consts:
#                 # in this ex., constr. is independent y
#                 if constraint.f(z, None) > 0:
#                     valid = False
#         if valid:
#             samples.append(z)

#     return np.array(samples)


