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

import pandas as pd

from plot import *


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


'''
Import Boston Housing data
'''
# Use 1-hidden layer, 50 nodes, BNN as in http://bayesiandeeplearning.org/2016/papers/BDL_34.pdf

df = pd.read_csv("datasets/boston_housing.csv")
print('Boston Housing: {}'.format(df.shape))
print(list(df.columns.values))

# shuffle dataset randomly
data = torch.tensor(df.values, dtype=torch.float32)
ind = torch.randperm(data.shape[0])
data = data[ind]

# test/train split
split = 0.75
N0 = int(split * data.shape[0])
train_data = data[:N0]
test_data = data[N0:]

X_train = train_data[:, :13]
Y_train = train_data[:, 13].unsqueeze(1)
X_test = test_data[:, :13]
Y_test = test_data[:, 13].unsqueeze(1)

X = X_train
Y = Y_train

X_plot = None
Y_plot = None

X_id = X_test
Y_id = Y_test


X_ood = None
Y_ood = None


'''
Constraints

Have to be broadcastable!!
i.e. inputs are
     x is (datapoints, in_dim)
     y is (weights samples, datapoints, out_dim)


'''


constr = [

]

'''
Has to sample X's from constrained input region
'''

def constrained_region_sampler(s):
    return torch.randn(s, 13) 

# TODO once constraints have been found



'''
Experiment dictionary 

Constraints are of the form
    [region_0, region_1, ...]
    with region = ([x_constraint_0, ..], [y_constraint_0, ..])

'''

all_experiments = []

prototype = {
    'title': 'boston_housing_50',
    'nn': {
        'architecture': [13, 50, 1],
        'nonlinearity': relu,
        'prior_ds': ds.Normal(0.0, 3.0),
    },
    'data': {
        'noise_ds': ds.Normal(0.0, 0.1),
        'plt_x_domain': (-13, 11),
        'plt_y_domain': (-2.5, 2.5),
        'integral_constrained_region': 14,
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
        'regular': {
            'iterations': 5000,
            'restarts': 1,
            'reporting_every_': 50,
            'cores_used': 1,
        },
        'constrained': {
            'iterations': 300,
            'restarts': 1,
            'reporting_every_': 100,
            'cores_used': 1,
            'violation_samples': 500,
            'tau_tuple': (15.0, 2.0),
            'gamma': 20000,
            'constrained_region_sampler': constrained_region_sampler,
        },
        'initialize_q': {
            'mean': 1.0,  # * torch.randn
            'std': -5.0,  # * torch.ones
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
        'show_function_samples': False,
        'show_posterior_predictive': False,
        'show_plot_training_evaluations': False,
        'show_constraint_function_heatmap': False,
        'plot_size': (6, 4),  # width, height (inches)
    },
}

all_experiments.append(prototype)

# repeat
# all_experiments = 1 * all_experiments


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
