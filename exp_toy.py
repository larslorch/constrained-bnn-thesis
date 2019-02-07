import matplotlib.pyplot as plt
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable

import copy

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
Constraints

Have to be broadcastable!!
i.e. x is (datapoints, in_dim)
     y is (weights samples, datapoints, out_dim)


'''


constr = [
]


def constrained_region_sampler(s):
    return torch.cat((ds.Uniform(-14, -6).sample(sample_shape=torch.Size([round(s / 2)])),
                      ds.Uniform(4, 12).sample(sample_shape=torch.Size([round(s / 2)])))).unsqueeze(1)


'''
Experiment dictionary 

Constraints are of the form
    [region_0, region_1, ...]
    with region = ([x_constraint_0, ..], [y_constraint_0, ..])

'''
all_experiments = []


prototype = {
    'title': '6pt_toy_example',
    'nn': {
        'architecture': [n_dim, 20, 1],
        'nonlinearity': rbf,
        'prior_ds': ds.Normal(0.0, 3.0),
    },
    'data': {
        'noise_ds': ds.Normal(0.0, 0.1),
        'plt_x_domain': (-5, 5),
        'plt_y_domain': (-15, 12),
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
            # DrawRectangle(bottom_left=(-20, -0.7), top_right=(-6, 20)),
            # DrawRectangle(bottom_left=(-20, -5), top_right=(-6, -1.3)),
            # DrawRectangle(bottom_left=(4, -20), top_right=(20, 0.7)),
            # DrawRectangle(bottom_left=(4, 1.3), top_right=(20, 5)),
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
            'violation_samples': 500,
            'tau_tuple': (15.0, 2.0),
            'gamma': 20000,
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


