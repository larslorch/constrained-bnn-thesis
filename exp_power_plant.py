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
Import 
Combined Cycle Power Plant Data Set 
'''
# Use 2-hidden layer, 100 nodes, BNN as in http://bayesiandeeplearning.org/2016/papers/BDL_34.pdf

df = pd.read_csv("datasets/power_plant.csv")
print('Combined Cycle Power Plant Data Set: {}'.format(df.shape))
print(list(df.columns.values))
# print(df.head())

# shuffle dataset randomly
data = torch.tensor(df.values, dtype=torch.float32)
ind = torch.randperm(data.shape[0])
data = data[ind]

# test/train split
split = 0.75
N0 = int(split * data.shape[0])
train_data = data[:N0]
test_data = data[N0:]

X_train = train_data[:, :4]
Y_train = train_data[:, 4].unsqueeze(1)
X_test = test_data[:, :4]
Y_test = test_data[:, 4].unsqueeze(1)

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

Has to return y.shape

Should mark forbidden region, not feasible region
'''

'''
Power plant data:
- Temperature (T) in the range 1.81°C and 37.11°C,
- Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
- Relative Humidity (RH) in the range 25.56% to 100.16%
- Exhaust Vacuum (V) in teh range 25.36-81.56 cm Hg
- Net hourly electrical energy output (EP) 420.26-495.76 MW

'''


# Temp (0) between 1.81°C and 37.11°C :
def at_c0(x, y): return (x[:, 0] - 37.11).unsqueeze(1)
def at_c1(x, y): return (1.81 - x[:, 0]).unsqueeze(1)

# Vac (1) range 25.36-81.56 cm Hg
def v_c0(x, y): return (x[:, 1] - 81.56).unsqueeze(1)
def v_c1(x, y): return (25.36 - x[:, 1]).unsqueeze(1)

# Ambient Pressure (2) 992.89-1033.30 milibar
def ap_c0(x, y): return (x[:, 2] - 1033.30).unsqueeze(1)
def ap_c1(x, y): return (992.89 - x[:, 2]).unsqueeze(1)

# Relative Humidity (3) 25.56% to 100.16%
def rh_c0(x, y): return (x[:, 3] - 100.16).unsqueeze(1)
def rh_c1(x, y): return (25.56 - x[:, 3]).unsqueeze(1)

# Y: Net hourly electrical energy output (EP) 420.26-495.76 MW
def ep_c0(x, y): return (y[:, 0] - 495.76).unsqueeze(1)
def ep_c1(x, y): return (420.26 - y[:, 0]).unsqueeze(1)


constr = [
    ([at_c0, at_c1, v_c0, v_c1, ap_c0, ap_c1, rh_c0, rh_c1], [ep_c0, ep_c1]),
]

'''
Has to sample X's from constrained input region
'''

def constrained_region_sampler(s):
    return torch.cat([
        ds.Uniform(1.81, 37.11).sample(sample_shape=torch.Size([s, 1])),
        ds.Uniform(25.36, 81.56).sample(sample_shape=torch.Size([s, 1])),
        ds.Uniform(992.89, 1033.30).sample(sample_shape=torch.Size([s, 1])),
        ds.Uniform(25.56, 100.16).sample(sample_shape=torch.Size([s, 1])),
    ], dim=1)



'''
Experiment dictionary 

Constraints are of the form
    [region_0, region_1, ...]
    with region = ([x_constraint_0, ..], [y_constraint_0, ..])

'''

all_experiments = []

prototype = {
    'title': 'boston_housing_50_relu',
    'nn': {
        'architecture': [4, 100, 100, 1],
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
        'BbB_rv_samples': 200,
        'batch_size': 1024,  # batch_size = 0 implies full dataset training
        'regular': {
            'iterations': 10000,
            'restarts': 1,
            'reporting_every_': 50,
            'cores_used': 1,
        },
        'constrained': {
            'iterations': 10090,
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
        'run_regular_BbB': False,
        'run_constrained_BbB': True,
        'multithread_computation': False,
        'compute_held_out_loglik_id': True,
        'compute_held_out_loglik_ood': False,
        'compute_RMSE_id': True,
        'compute_RMSE_ood': False,
        'show_function_samples': False,
        'show_posterior_predictive': False,
        'show_plot_training_evaluations': True,
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
